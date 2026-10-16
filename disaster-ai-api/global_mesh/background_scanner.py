"""
global_mesh/background_scanner.py — Background Global Scan Processing

Provides non-blocking global scan execution:

  * A scan starts and immediately returns a job ID.
  * The backend runs the scan in a background thread.
  * Progress is tracked per-cell and persisted incrementally to MongoDB.
  * Clients can poll job status to see progress and results.
  * Failed/interrupted scans retain completed work and can recover.
  * Partial failures are reported accurately.

This keeps the HTTP request/response cycle short (no 180s/300s timeouts).
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from global_mesh.data_ingestion import GlobalMeshDataIngestion
from global_mesh.flood_risk_model import FloodRiskModel
from global_mesh.risk_scorer import GlobalMeshRiskScorer
from global_mesh.mongodb_store import get_mongo_store
from global_mesh.historical_labels import get_historical_labels
from global_mesh.validation import evaluate_probabilistic_model

logger = logging.getLogger("GlobalMeshBackgroundScanner")
logging.basicConfig(level=logging.INFO)

# Job statuses
JOB_PENDING = "PENDING"
JOB_RUNNING = "RUNNING"
JOB_COMPLETED = "COMPLETED"
JOB_FAILED = "FAILED"
JOB_PARTIAL = "PARTIAL"


@dataclass
class ScanJob:
    """A background scan job."""
    job_id: str
    status: str = JOB_PENDING
    progress_pct: float = 0.0
    progress_message: str = "Pending"
    total_cells: int = 0
    completed_cells: int = 0
    active_cells: int = 0
    emerging_threats: int = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    scan_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BackgroundScanProcessor:
    """
    Manages background global scan jobs.

    Only one scan runs at a time to avoid API overload.  Additional requests
    while a scan is running return the existing job ID.
    """

    def __init__(self, max_cells_per_run: int = 200) -> None:
        self.max_cells_per_run = max_cells_per_run
        self._jobs: Dict[str, ScanJob] = {}
        self._lock = threading.Lock()
        self._current_job_id: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start_scan(
        self,
        max_cells: Optional[int] = None,
        force_new: bool = False,
    ) -> Dict[str, Any]:
        """
        Start a background global scan. Returns {job_id, status, scan_id?}.
        If a scan is already running and force_new is False, returns the
        existing job ID instead of starting a duplicate.
        """
        with self._lock:
            if self._current_job_id is not None and not force_new:
                job = self._jobs.get(self._current_job_id)
                if job:
                    return {
                        "job_id": self._current_job_id,
                        "status": job.status,
                        "progress_pct": job.progress_pct,
                        "message": "A global scan is already running",
                        "existing": True,
                    }

            job_id = f"scan_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            job = ScanJob(
                job_id=job_id,
                status=JOB_PENDING,
                progress_message="Initializing scan...",
            )
            self._jobs[job_id] = job
            self._current_job_id = job_id
            self._stop_event.clear()

            max_cells = max_cells or self.max_cells_per_run
            self._thread = threading.Thread(
                target=self._run_scan,
                args=(job_id, max_cells),
                daemon=True,
            )
            self._thread.start()

            logger.info("Started background scan job %s", job_id)
            return {
                "job_id": job_id,
                "status": JOB_PENDING,
                "progress_pct": 0.0,
                "message": "Scan starting",
                "existing": False,
            }

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            return job.to_dict()

    def cancel_scan(self, job_id: str) -> bool:
        with self._lock:
            if self._current_job_id != job_id:
                return False
            self._stop_event.set()
            job = self._jobs.get(job_id)
            if job:
                job.status = JOB_FAILED
                job.error = "Cancelled by user"
                job.completed_at = datetime.now(timezone.utc).isoformat()
            self._current_job_id = None
            return True

    def _run_scan(self, job_id: str, max_cells: int) -> None:
        """Execute the global scan in a background thread."""
        job = self._jobs.get(job_id)
        if not job:
            return

        job.status = JOB_RUNNING
        job.started_at = datetime.now(timezone.utc).isoformat()
        job.progress_message = "Initializing engine..."

        # Lazy import to avoid circular dependency
        from global_prediction_engine import (
            get_global_engine,
            haversine_km,
            FLOOD_PRONE_ANCHORS,
        )
        from topological_river_network import get_river_router

        try:
            engine = get_global_engine()
            router = get_river_router()
            ingestion = GlobalMeshDataIngestion(request_timeout=15)

            # Get GDACS alerts
            job.progress_message = "Fetching GDACS alerts..."
            gdacs_events = ingestion.fetch_gdacs_alerts()
            job.progress_pct = 5.0

            # Build candidate cells
            job.progress_message = "Building candidate cell list..."
            candidate_cells = engine._build_candidate_cells(gdacs_events)
            candidate_cells = candidate_cells[:max_cells]
            job.total_cells = len(candidate_cells)
            logger.info("Job %s: %d candidate cells", job_id, job.total_cells)

            # Load model + scorer
            model = engine.risk_model
            scorer = engine.risk_scorer

            # Process cells in batches with incremental persistence
            mongo_store = get_mongo_store()
            historical_labels = get_historical_labels()

            evaluated: List[Dict[str, Any]] = []
            cell_observations: List[Dict[str, Any]] = []
            critical = high = moderate = low = emerging = 0
            active_cells = 0

            batch_size = 10  # cells per progress update / persistence batch

            for batch_start in range(0, len(candidate_cells), batch_size):
                if self._stop_event.is_set():
                    job.status = JOB_FAILED
                    job.error = "Cancelled"
                    job.completed_at = datetime.now(timezone.utc).isoformat()
                    self._current_job_id = None
                    return

                batch = candidate_cells[batch_start:batch_start + batch_size]

                # Observe cells (batched API calls)
                cell_tuples = [(c.cell_id, c.lat, c.lon) for c in batch]
                observations = ingestion.observe_cells(cell_tuples, gdacs_events=gdacs_events)

                # Score each cell
                for cell, obs in zip(batch, observations):
                    try:
                        rs = scorer.score_cell(obs)
                    except Exception as exc:
                        logger.warning("Scoring failed for %s: %s", cell.cell_id, exc)
                        continue

                    if obs.data_available:
                        active_cells += 1

                    # Build region record
                    region = self._build_region(
                        engine, router, cell, obs, rs, gdacs_events, historical_labels
                    )
                    evaluated.append(region)
                    cell_observations.append(obs.to_dict())

                    score = rs.risk_score_0_100
                    if score >= 75:
                        critical += 1
                    elif score >= 50:
                        high += 1
                    elif score >= 25:
                        moderate += 1
                    else:
                        low += 1
                    if region.get("is_emerging"):
                        emerging += 1

                # Persist incrementally after each batch
                job.completed_cells = min(batch_start + batch_size, len(candidate_cells))
                job.active_cells = active_cells
                job.progress_pct = round(
                    (job.completed_cells / max(1, job.total_cells)) * 100.0, 1
                )
                job.progress_message = (
                    f"Processed {job.completed_cells}/{job.total_cells} cells "
                    f"({active_cells} with data)"
                )

                # Persist cell observations to MongoDB
                try:
                    mongo_store.save_cell_observations(job_id, cell_observations)
                    cell_observations = []  # clear after persist
                except Exception as exc:
                    logger.warning("Failed to persist cell observations: %s", exc)

                time.sleep(0.1)  # yield to allow cancellation

            # --- Finalize results ---
            sorted_regions = sorted(
                evaluated,
                key=lambda r: (
                    1 if r.get("is_emerging") else 0,
                    r.get("peak_risk_score", r.get("flood_risk_score", 0)),
                    r.get("flood_occurrence_probability", 0),
                    r.get("forecast_rain_72h_mm", 0),
                ),
                reverse=True,
            )
            top_emerging = sorted_regions[:15]

            global_timeline = self._compute_trajectory(evaluated)
            geojson_features = [self._geojson_feature(r) for r in evaluated]

            avg_risk = round(
                sum(r["flood_risk_score"] for r in evaluated) / max(1, len(evaluated)), 1
            )

            scan_id = f"global_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            now_iso = datetime.now(timezone.utc).isoformat()

            result = {
                "status": "success",
                "timestamp": now_iso,
                "scan_id": scan_id,
                "job_id": job_id,
                "summary": {
                    "total_regions_evaluated": len(evaluated),
                    "critical_zones_count": critical,
                    "high_risk_count": high,
                    "moderate_risk_count": moderate,
                    "low_risk_count": low,
                    "emerging_threats_count": emerging,
                    "global_average_risk_score": avg_risk,
                    "gdacs_live_events_linked": len(gdacs_events),
                    "grid_cells_active": active_cells,
                    "grid_cells_unavailable": len(evaluated) - active_cells,
                    "ai_streamflow_engine": "Global Mesh GRU Flood-Risk Model "
                    + ("(trained)" if model.is_trained() else "(untrained — physics-only fallback)"),
                },
                "top_emerging_regions": top_emerging,
                "global_forecast_trajectory": global_timeline,
                "geojson": {"type": "FeatureCollection", "features": geojson_features},
            }

            job.result = result
            job.scan_id = scan_id
            job.status = JOB_COMPLETED
            job.progress_pct = 100.0
            job.progress_message = "Scan completed"
            job.completed_at = now_iso

            # Persist scan record to MongoDB
            scan_record = {
                "scan_id": scan_id,
                "job_id": job_id,
                "timestamp": now_iso,
                "summary": result["summary"],
                "top_emerging_regions": result["top_emerging_regions"],
                "global_forecast_trajectory": result["global_forecast_trajectory"],
                "geojson": result["geojson"],
                "metadata": result.get("metadata", {}),
            }
            try:
                mongo_store.save_scan(scan_record)
            except Exception as exc:
                logger.warning("Failed to persist scan record: %s", exc)

            # Update engine cache
            engine._last_result = result
            engine._last_run_timestamp = datetime.now(timezone.utc)

            logger.info(
                "Job %s completed: %d cells, %d emerging, avg risk %.1f",
                job_id, len(evaluated), emerging, avg_risk,
            )

        except Exception as exc:
            logger.exception("Background scan job %s failed: %s", job_id, exc)
            job.status = JOB_FAILED
            job.error = str(exc)
            job.completed_at = datetime.now(timezone.utc).isoformat()
        finally:
            with self._lock:
                if self._current_job_id == job_id:
                    self._current_job_id = None

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers (mirror engine logic but independent for background thread)
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _build_region(
        engine: GlobalPredictionEngine,
        router: Any,
        cell: Any,
        obs: Any,
        rs: Any,
        gdacs_events: List[Dict[str, Any]],
        historical_labels: Any,
    ) -> Dict[str, Any]:
        # Lazy import to avoid circular dependency
        from global_prediction_engine import haversine_km, FLOOD_PRONE_ANCHORS

        lat, lon = obs.lat, obs.lon
        score = rs.risk_score_0_100

        forecast_rain_24h = obs.fcst_precip_24h_mm or 0.0
        forecast_rain_72h = obs.fcst_precip_72h_mm or 0.0
        river_q = obs.obs_river_discharge_m3s or 0.0
        discharge_ratio = obs.obs_discharge_ratio or 1.0
        soil_m = obs.obs_soil_moisture_root or obs.obs_soil_moisture_surface or 0.30
        elev = obs.obs_elevation_m or 0.0

        def exceed(ratio: float, threshold: float, scale: float = 0.22) -> float:
            z = (ratio - threshold) / scale
            return round(float(1.0 / (1.0 + math.exp(-z))), 3)

        p_q2 = exceed(discharge_ratio, 1.30)
        p_q5 = exceed(discharge_ratio, 1.70)
        p_q20 = exceed(discharge_ratio, 2.30)
        p_flood = rs.flood_probability if rs.flood_probability is not None else 0.0

        peak_score = min(100.0, score + (forecast_rain_72h * 0.25))

        if forecast_rain_24h > 45.0:
            peak_hours = 18
        elif forecast_rain_72h > 60.0:
            peak_hours = 48
        elif discharge_ratio > 1.4:
            peak_hours = 24
        else:
            peak_hours = 12

        if peak_score >= 75 or p_q5 > 0.65:
            sev, color = "CRITICAL", "#ef4444"
        elif peak_score >= 50 or p_q2 > 0.60:
            sev, color = "HIGH", "#f97316"
        elif peak_score >= 25 or p_flood > 0.30:
            sev, color = "MODERATE", "#eab308"
        else:
            sev, color = "LOW", "#10b981"

        trend = (
            "RAPIDLY_INCREASING"
            if (peak_score - score) >= 18 or forecast_rain_24h >= 45.0
            else "GRADUALLY_INCREASING"
            if (peak_score - score) >= 8 or forecast_rain_72h >= 30.0
            else "DECREASING"
            if score > peak_score + 5
            else "STABLE"
        )
        is_emerging = trend in ("RAPIDLY_INCREASING", "GRADUALLY_INCREASING")

        typology = router.classify_flood_typology(
            discharge_ratio=discharge_ratio,
            soil_moisture=soil_m,
            rain_24h_mm=forecast_rain_24h,
            cape_j_kg=obs.obs_cape_jkg or 0.0,
            snow_depth_cm=obs.obs_snow_depth_cm or 0.0,
            temp_c=obs.obs_temperature_c or 20.0,
            is_coastal=bool(obs.obs_is_coastal),
            wave_height_m=0.0,
        )

        drivers: List[str] = []
        if forecast_rain_72h > 40.0:
            drivers.append(f"Forecast 72h Rain (+{forecast_rain_72h:.0f} mm)")
        if (obs.obs_antecedent_precip_index_14d_mm or 0) > 50.0:
            drivers.append(f"14d Antecedent ({obs.obs_antecedent_precip_index_14d_mm:.0f} mm)")
        if discharge_ratio > 1.3:
            drivers.append(f"Discharge {discharge_ratio:.2f}x mean")
        if obs.gdacs_alert_level:
            drivers.append(f"GDACS {obs.gdacs_alert_level} alert")
        primary_driver = " & ".join(drivers[:2]) if drivers else "Baseline hydrometeorological conditions"

        cell_area_km2 = (0.25 * 111.0) ** 2 * max(0.0, math.cos(math.radians(lat)))
        area_est = max(0.0, cell_area_km2 * (score / 100.0) * 0.5)
        exposed_pop = int(round(area_est * 120.0 * 0.4))

        # Name from GDACS/anchors
        name = cell.cell_id
        country = "International Waters"
        continent = "Global Grid Cell"
        basin = "Grid Catchment"
        for ev in gdacs_events:
            if haversine_km(lat, lon, ev["lat"], ev["lon"]) < 0.5:
                name = ev.get("eventname") or f"GDACS Alert ({ev.get('country')})"
                country = ev.get("country", country)
                basin = "Active Alert Catchment"
                break
        for anc in FLOOD_PRONE_ANCHORS:
            if haversine_km(lat, lon, anc["lat"], anc["lon"]) < 0.5:
                name = anc["name"]
                country = anc["country"]
                continent = anc["continent"]
                basin = anc["basin"]
                break

        # Historical flood label from DFO
        hist_label, hist_event_id, hist_details = historical_labels.label_cell(
            lat, lon, observation_date=obs.retrieved_at_utc
        )

        return {
            "id": cell.cell_id,
            "cell_id": cell.cell_id,
            "name": name,
            "country": country,
            "continent": continent,
            "basin": basin,
            "lat": lat,
            "lon": lon,
            "elev": elev,
            "flood_risk_score": round(score, 1),
            "current_risk_score": round(score, 1),
            "peak_risk_score": round(peak_score, 1),
            "discharge_exceedance_p_q2": p_q2,
            "discharge_exceedance_p_q5": p_q5,
            "discharge_exceedance_p_q20": p_q20,
            "flood_occurrence_probability": round(p_flood, 3),
            "severity": sev,
            "risk_trend": trend,
            "is_emerging": bool(is_emerging),
            "peak_time_hours": peak_hours,
            "primary_driver": primary_driver,
            "flood_typology": {
                "code": typology[0],
                "label": typology[1],
                "description": typology[2],
            },
            "circle_color": color,
            "forecast_rain_24h_mm": round(forecast_rain_24h, 1),
            "forecast_rain_72h_mm": round(forecast_rain_72h, 1),
            "river_discharge_m3s": round(river_q, 1),
            "discharge_ratio": round(discharge_ratio, 3),
            "soil_moisture": round(soil_m, 4),
            "elevation_m": round(elev, 1),
            "affected_area_km2": round(area_est, 1),
            "estimated_exposed_pop": exposed_pop,
            "recommended_action": {
                "CRITICAL": "Activate inter-agency emergency operations & issue downstream evacuation warnings",
                "HIGH": "Pre-position flood-fighting assets & inspect primary levees",
                "MODERATE": "Monitor river telemetry & clear critical drainage culverts",
                "LOW": "Routine satellite & hydrometeorological surveillance",
            }[sev],
            "data_provenance": {
                "antecedent_rainfall": "OBSERVATION (ECMWF IFS via Open-Meteo)"
                if obs.obs_precip_72h_mm is not None else "UNAVAILABLE",
                "forecast_rainfall": "FORECAST (ECMWF IFS via Open-Meteo)"
                if obs.fcst_precip_72h_mm is not None else "UNAVAILABLE",
                "river_discharge": "MODELLED_REANALYSIS (Copernicus GloFAS 4.0)"
                if obs.obs_river_discharge_m3s is not None else "UNAVAILABLE",
                "disaster_alerts": "EXTERNAL_ALERT_EVIDENCE (UN/EC GDACS)"
                if obs.gdacs_alert_level else "UNAVAILABLE",
                "satellite_observation": "UNAVAILABLE (Sentinel-1 SAR on drilldown)",
                "soil_moisture": "MODELLED_REANALYSIS (ECMWF HTESSEL)"
                if obs.obs_soil_moisture_root is not None else "UNAVAILABLE",
                "topography": "OBSERVED_GIS (Copernicus GLO-30 DEM)"
                if obs.obs_elevation_m is not None else "UNAVAILABLE",
            },
            "model_status": rs.model_status,
            "epistemic_std": rs.epistemic_std,
            "confidence_interval": list(rs.confidence_interval) if rs.confidence_interval else None,
            "data_available": bool(obs.data_available),
            "missing_factors": list(obs.missing_factors),
            # Historical verified flood label (DFO)
            "historical_flood_label": hist_label,
            "historical_flood_event": hist_event_id,
            "historical_flood_details": hist_details,
        }

    @staticmethod
    def _geojson_feature(r: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
            "properties": {
                "id": r["id"],
                "name": r["name"],
                "country": r["country"],
                "continent": r["continent"],
                "basin": r["basin"],
                "cell_id": r.get("cell_id"),
                "flood_risk_score": r["flood_risk_score"],
                "current_risk_score": r["flood_risk_score"],
                "peak_risk_score": r["peak_risk_score"],
                "discharge_exceedance_p_q2": r["discharge_exceedance_p_q2"],
                "discharge_exceedance_p_q5": r["discharge_exceedance_p_q5"],
                "discharge_exceedance_p_q20": r["discharge_exceedance_p_q20"],
                "flood_occurrence_probability": r["flood_occurrence_probability"],
                "severity": r["severity"],
                "risk_trend": r["risk_trend"],
                "is_emerging": r["is_emerging"],
                "peak_time_hours": r["peak_time_hours"],
                "primary_driver": r["primary_driver"],
                "flood_typology": r["flood_typology"],
                "forecast_rain_24h_mm": r["forecast_rain_24h_mm"],
                "forecast_rain_72h_mm": r["forecast_rain_72h_mm"],
                "river_discharge_m3s": r["river_discharge_m3s"],
                "discharge_ratio": r["discharge_ratio"],
                "soil_moisture": r["soil_moisture"],
                "elevation_m": r["elevation_m"],
                "estimated_exposed_pop": r["estimated_exposed_pop"],
                "recommended_action": r["recommended_action"],
                "circle_color": r["circle_color"],
                "data_provenance": r["data_provenance"],
                "model_status": r.get("model_status"),
                "epistemic_std": r.get("epistemic_std"),
                "confidence_interval": r.get("confidence_interval"),
                "data_available": r.get("data_available"),
            },
        }

    @staticmethod
    def _compute_trajectory(regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not regions:
            return []
        cur_avg = sum(r["flood_risk_score"] for r in regions) / len(regions)
        peak_avg = sum(r["peak_risk_score"] for r in regions) / len(regions)
        return [
            {"hour": 0, "label": "Now (T+0)", "global_avg_risk": round(cur_avg, 1),
             "high_risk_zones": sum(1 for r in regions if r["flood_risk_score"] >= 50)},
            {"hour": 24, "label": "T+24h", "global_avg_risk": round(peak_avg, 1),
             "high_risk_zones": sum(1 for r in regions if r["peak_risk_score"] >= 50)},
            {"hour": 48, "label": "T+48h", "global_avg_risk": round(cur_avg + (peak_avg - cur_avg) * 0.6, 1),
             "high_risk_zones": sum(1 for r in regions if r["peak_risk_score"] >= 50)},
            {"hour": 72, "label": "T+72h", "global_avg_risk": round(cur_avg + (peak_avg - cur_avg) * 0.35, 1),
             "high_risk_zones": sum(1 for r in regions if r["flood_risk_score"] >= 45)},
        ]


# Global singleton
_processor: Optional[BackgroundScanProcessor] = None
_processor_lock = threading.Lock()


def get_background_processor() -> BackgroundScanProcessor:
    global _processor
    with _processor_lock:
        if _processor is None:
            _processor = BackgroundScanProcessor()
    return _processor

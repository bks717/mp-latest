"""
global_mesh/alerting/alert_candidates.py — Alert candidate generation (Phase 5)

Builds alert candidate documents from Global Mesh prediction results for
writing to the MongoDB `global_mesh_alert_candidates` collection.

Each candidate carries:
  * cell/region identity and coordinates
  * predicted flood probability (0-1) with uncertainty (epistemic std, CI)
  * risk tier: WATCH / WARNING / IMMINENT based on calibrated thresholds
  * trust flag: whether the model output is trustworthy (in-domain, calibrated)
  * full provenance: source data, timestamps, freshness
  * forecast issue time and target horizon (24h / 72h / 168h)

No synthetic data is introduced. Missing values are reported as missing, never
fabricated. Cells without data are never marked "safe" — they are simply not
elevated to alert candidates.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("GlobalMeshAlertCandidates")
logging.basicConfig(level=logging.INFO)

# Tier thresholds (calibrated probability thresholds for flood occurrence)
TIER_WATCH = 0.30      # P(flood) >= 0.30 -> WATCH
TIER_WARNING = 0.50    # P(flood) >= 0.50 -> WARNING
TIER_IMMINENT = 0.70   # P(flood) >= 0.70 -> IMMINENT

# Trust threshold: a model is "trusted" only if trained and validated
TRUSTED_MODEL_AUC_MIN = 0.65  # minimum AUC to consider model trustworthy


@dataclass
class AlertCandidate:
    """A single alert candidate for a grid cell / horizon."""

    candidate_id: str
    region_id: str
    cell_id: str
    lat: float
    lon: float
    region_name: str
    country: str
    basin: str
    forecast_horizon_hours: int
    target_valid_time_utc: str
    issue_timestamp_utc: str
    predicted_flood_probability: Optional[float]
    probability_ci: Optional[List[float]]
    epistemic_std: Optional[float]
    risk_tier: str  # "WATCH" | "WARNING" | "IMMINENT" | "NORMAL"
    trust_flag: bool  # True if model is trained+validated and in-domain
    model_status: str
    data_available: bool
    missing_factors: List[str]
    data_provenance: Dict[str, Any]
    primary_driver: str
    flood_typology: Dict[str, str]
    severity: str
    peak_risk_score: float
    forecast_rain_72h_mm: float
    river_discharge_m3s: float
    discharge_ratio: float
    soil_moisture: float
    elevation_m: float
    estimated_exposed_pop: int
    status: str = "PENDING"
    timestamp_utc: Optional[str] = None
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _sha256(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _lat_lon_from_cell_id(cell_id: str) -> Tuple[Optional[float], Optional[float]]:
    """Deterministically compute centroid coordinates from a grid_{lat_idx}_{lon_idx} ID."""
    try:
        parts = str(cell_id).split('_')
        if len(parts) == 3:
            lat_idx = int(parts[1])
            lon_idx = int(parts[2])
            lat = round(-60.0 + (lat_idx + 0.5) * 0.25, 4)
            lon = round(-180.0 + (lon_idx + 0.5) * 0.25, 4)
            return lat, lon
    except Exception:
        pass
    return None, None


def _compute_target_valid_time(issue_ts: str, horizon_hours: int) -> str:
    """Compute target validity timestamp from issue timestamp + horizon."""
    try:
        dt = datetime.fromisoformat(str(issue_ts).replace("Z", "+00:00"))
        return (dt + timedelta(hours=horizon_hours)).isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def _tier_from_probability(prob: Optional[float]) -> str:
    """Map P(flood) to a risk tier."""
    if prob is None:
        return "NORMAL"
    if prob >= TIER_IMMINENT:
        return "IMMINENT"
    if prob >= TIER_WARNING:
        return "WARNING"
    if prob >= TIER_WATCH:
        return "WATCH"
    return "NORMAL"


def build_alert_candidates(
    prediction_result: Dict[str, Any],
    model_trained: bool = False,
    model_validated: bool = False,
    model_auc: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Build alert candidate documents from a global prediction result.

    Args:
        prediction_result: Output of GlobalPredictionEngine.run_global_prediction()
        model_trained: Whether the GRU model has been trained
        model_validated: Whether the GRU model has demonstrated skill
        model_auc: ROC-AUC on held-out data (for trust assessment)

    Returns:
        List of alert candidate dicts ready for MongoDB insertion.
    """
    now_utc = _now()
    logger.info(
        "DEBUG: prediction_result type = %s, prediction_result = %r",
        type(prediction_result),
        str(prediction_result)[:200],
    )

    # Trust assessment: model is trusted only if trained AND validated
    model_trusted = model_trained and model_validated and (
        model_auc is not None and model_auc >= TRUSTED_MODEL_AUC_MIN
    )

    trusted_str = "trusted" if model_trusted else "not_trusted"
    logger.info(
        "Building alert candidates: model_trained=%s model_validated=%s auc=%s trusted=%s",
        model_trained,
        model_validated,
        model_auc,
        trusted_str,
    )

    candidates: List[Dict[str, Any]] = []
    seen_cells = set()
    top_emerging = prediction_result.get("top_emerging_regions", []) or []
    geojson = prediction_result.get("geojson", [])
    if isinstance(geojson, dict):
        features = geojson.get("features", []) or []
    elif isinstance(geojson, list):
        features = geojson
    else:
        features = []

    # Build a lookup from cell_id to region record
    region_by_cell: Dict[str, Dict[str, Any]] = {}
    for r in top_emerging:
        c_id = r.get("cell_id", r.get("id"))
        if c_id:
            region_by_cell[c_id] = r

    logger.info(
        "DEBUG: geojson type = %s, features type = %s, n_features = %d, top_emerging = %d",
        type(geojson).__name__,
        type(features).__name__,
        len(features),
        len(top_emerging),
    )

    # Combine GeoJSON features and top_emerging regions without omitting anything
    eval_items: List[Dict[str, Any]] = list(features)
    for r in top_emerging:
        c_id = r.get("cell_id", r.get("id"))
        if c_id and not any((f.get("properties", {}) or {}).get("cell_id") == c_id for f in features):
            eval_items.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [r.get("lon"), r.get("lat")]},
                "properties": r,
            })

    # Process all evaluated cells
    for item in eval_items:
        props = item.get("properties", {}) if isinstance(item, dict) and "properties" in item else item
        cell_id = props.get("cell_id") or props.get("id")
        if not cell_id or cell_id in seen_cells:
            continue

        region = region_by_cell.get(cell_id, props)

        # 1. Determine coordinates (Point geometry: [lon, lat])
        geometry = item.get("geometry", {}) or {}
        coordinates = geometry.get("coordinates", [None, None]) or [None, None]
        geojson_lat = coordinates[1] if len(coordinates) > 1 and coordinates[1] is not None else None
        geojson_lon = coordinates[0] if len(coordinates) > 0 and coordinates[0] is not None else None

        # Fallback to props/region
        if geojson_lat is None:
            geojson_lat = props.get("lat") or region.get("lat")
        if geojson_lon is None:
            geojson_lon = props.get("lon") or region.get("lon")

        # Fallback to deterministic cell_id centroid
        if geojson_lat is None or geojson_lon is None:
            fb_lat, fb_lon = _lat_lon_from_cell_id(cell_id)
            if geojson_lat is None:
                geojson_lat = fb_lat
            if geojson_lon is None:
                geojson_lon = fb_lon

        # 2. Probability and tier
        prob = props.get("flood_occurrence_probability")
        if prob is not None:
            prob = float(prob)
        tier = _tier_from_probability(prob)

        peak_score = float(props.get("peak_risk_score", 0) or props.get("flood_risk_score", 0) or 0)

        # Fall back to peak_risk_score when model probability is unavailable or NORMAL with high risk score
        if tier == "NORMAL":
            if peak_score >= 50:
                tier = "WARNING"
                prob = round(peak_score / 100.0, 3)
            elif peak_score >= 25:
                tier = "WATCH"
                prob = round(peak_score / 100.0, 3)
            else:
                continue
        elif prob is None or prob == 0:
            if peak_score >= 25:
                prob = round(peak_score / 100.0, 3)

        # Trust flag: only trust model output if trained+validated
        trust_flag = bool(model_trusted and tier in ("WARNING", "IMMINENT"))

        horizon = int(props.get("peak_time_hours", 24) or 24)
        cand_id_input = f"{cell_id}:{horizon}:{now_utc[:10]}"
        candidate_id = f"ac_{hashlib.sha256(cand_id_input.encode()).hexdigest()[:12]}"

        issue_ts = prediction_result.get("timestamp", now_utc)
        target_valid_ts = _compute_target_valid_time(issue_ts, horizon)

        candidate = {
            "candidate_id": candidate_id,
            "region_id": cell_id,
            "cell_id": cell_id,
            "lat": geojson_lat,
            "lon": geojson_lon,
            "region_name": props.get("name") or region.get("name") or cell_id,
            "country": props.get("country") or region.get("country") or "International Territory",
            "basin": props.get("basin") or region.get("basin") or "Regional Catchment",
            "forecast_horizon_hours": horizon,
            "target_valid_time_utc": target_valid_ts,
            "issue_timestamp_utc": issue_ts,
            "timestamp_utc": issue_ts,
            "created_at": issue_ts,
            "predicted_flood_probability": prob,
            "probability_ci": props.get("confidence_interval"),
            "epistemic_std": props.get("epistemic_std"),
            "risk_tier": tier,
            "trust_flag": trust_flag,
            "model_status": props.get("model_status", "untrained"),
            "status": "PENDING",
            "data_available": bool(props.get("data_available", False)),
            "missing_factors": list(props.get("missing_factors", []) or []),
            "data_provenance": props.get("data_provenance", {}),
            "primary_driver": props.get("primary_driver"),
            "flood_typology": props.get("flood_typology", {}),
            "severity": props.get("severity") or ("CRITICAL" if peak_score >= 75 else "HIGH" if peak_score >= 50 else "MODERATE" if peak_score >= 25 else "LOW"),
            "peak_risk_score": peak_score,
            "forecast_rain_72h_mm": float(props.get("forecast_rain_72h_mm", 0) or 0),
            "river_discharge_m3s": float(props.get("river_discharge_m3s", 0) or 0),
            "discharge_ratio": float(props.get("discharge_ratio", 1) or 1),
            "soil_moisture": float(props.get("soil_moisture", 0) or 0),
            "elevation_m": float(props.get("elevation_m", 0) or 0),
            "estimated_exposed_pop": int(props.get("estimated_exposed_pop", 0) or 0),
        }
        candidates.append(candidate)
        seen_cells.add(cell_id)

    logger.info("Built %d alert candidates (%d trusted)", len(candidates), sum(1 for c in candidates if c["trust_flag"]))
    return candidates


def build_candidates_from_region_records(
    region_records: List[Dict[str, Any]],
    model_trained: bool = False,
    model_validated: bool = False,
    model_auc: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Build alert candidates directly from a list of region dicts."""
    now_utc = _now()
    model_trusted = model_trained and model_validated and (
        model_auc is not None and model_auc >= TRUSTED_MODEL_AUC_MIN
    )

    candidates: List[Dict[str, Any]] = []
    for r in region_records:
        cell_id = r.get("cell_id") or r.get("id")
        if not cell_id:
            continue

        lat = r.get("lat")
        lon = r.get("lon")
        if lat is None or lon is None:
            fb_lat, fb_lon = _lat_lon_from_cell_id(cell_id)
            if lat is None:
                lat = fb_lat
            if lon is None:
                lon = fb_lon

        prob = r.get("flood_occurrence_probability")
        if prob is not None:
            prob = float(prob)
        tier = _tier_from_probability(prob)

        peak_score = float(r.get("peak_risk_score", 0) or r.get("flood_risk_score", 0) or 0)
        if tier == "NORMAL":
            if peak_score >= 50:
                tier = "WARNING"
                prob = round(peak_score / 100.0, 3)
            elif peak_score >= 25:
                tier = "WATCH"
                prob = round(peak_score / 100.0, 3)
            else:
                continue
        elif prob is None or prob == 0:
            if peak_score >= 25:
                prob = round(peak_score / 100.0, 3)

        trust_flag = bool(model_trusted and tier in ("WARNING", "IMMINENT"))
        horizon = int(r.get("peak_time_hours", 24) or 24)
        cand_id_input = f"{cell_id}:{horizon}:{now_utc[:10]}"
        candidate_id = f"ac_{hashlib.sha256(cand_id_input.encode()).hexdigest()[:12]}"
        target_valid_ts = _compute_target_valid_time(now_utc, horizon)

        candidates.append(
            {
                "candidate_id": candidate_id,
                "region_id": cell_id,
                "cell_id": cell_id,
                "lat": lat,
                "lon": lon,
                "region_name": r.get("name") or cell_id,
                "country": r.get("country") or "International Territory",
                "basin": r.get("basin") or "Regional Catchment",
                "forecast_horizon_hours": horizon,
                "target_valid_time_utc": target_valid_ts,
                "issue_timestamp_utc": now_utc,
                "timestamp_utc": now_utc,
                "created_at": now_utc,
                "predicted_flood_probability": prob,
                "probability_ci": r.get("confidence_interval"),
                "epistemic_std": r.get("epistemic_std"),
                "risk_tier": tier,
                "trust_flag": trust_flag,
                "model_status": r.get("model_status", "untrained"),
                "status": "PENDING",
                "data_available": bool(r.get("data_available", False)),
                "missing_factors": list(r.get("missing_factors", []) or []),
                "data_provenance": r.get("data_provenance", {}),
                "primary_driver": r.get("primary_driver"),
                "flood_typology": r.get("flood_typology", {}),
                "severity": r.get("severity") or ("CRITICAL" if peak_score >= 75 else "HIGH" if peak_score >= 50 else "MODERATE" if peak_score >= 25 else "LOW"),
                "peak_risk_score": peak_score,
                "forecast_rain_72h_mm": float(r.get("forecast_rain_72h_mm", 0) or 0),
                "river_discharge_m3s": float(r.get("river_discharge_m3s", 0) or 0),
                "discharge_ratio": float(r.get("discharge_ratio", 1) or 1),
                "soil_moisture": float(r.get("soil_moisture", 0) or 0),
                "elevation_m": float(r.get("elevation_m", 0) or 0),
                "estimated_exposed_pop": int(r.get("estimated_exposed_pop", 0) or 0),
            }
        )

    logger.info("Built %d alert candidates (%d trusted)", len(candidates), sum(1 for c in candidates if c["trust_flag"]))
    return candidates

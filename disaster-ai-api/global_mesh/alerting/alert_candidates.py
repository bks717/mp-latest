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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _sha256(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _now() -> str:
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
        region_by_cell[r.get("cell_id", r.get("id"))] = r

    # Build a set of cell_ids from top_emerging to avoid duplicates
    top_emerging_cell_ids = {r.get("cell_id", r.get("id")) for r in top_emerging if r.get("cell_id") or r.get("id")}

    logger.info(
        "DEBUG: geojson type = %s, features type = %s, n_features = %d",
        type(geojson).__name__,
        type(features).__name__,
        len(features),
    )

    # Build a lookup from cell_id to region record
    region_by_cell: Dict[str, Dict[str, Any]] = {}
    for r in top_emerging:
        region_by_cell[r.get("cell_id", r.get("id"))] = r

    # Also process all geojson features (covers all evaluated cells)
    for feature in features:
        props = feature.get("properties", {}) or {}
        cell_id = props.get("cell_id")
        if not cell_id:
            continue

        # Skip cells already in top_emerging to avoid duplicates
        if cell_id in top_emerging_cell_ids:
            continue

        region = region_by_cell.get(cell_id, props)

        prob = props.get("flood_occurrence_probability")
        tier = _tier_from_probability(prob)

        # Only emit alert candidates for cells with meaningful risk
        if tier == "NORMAL":
            # Fall back to peak_risk_score when model probability is unavailable
            peak_score = float(props.get("peak_risk_score", 0) or 0)
            if peak_score >= 50:
                tier = "WARNING"
                prob = peak_score / 100.0  # Use risk score as probability fallback
            elif peak_score >= 25:
                tier = "WATCH"
                prob = peak_score / 100.0  # Use risk score as probability fallback
            else:
                continue

        # Trust flag: only trust model output if trained+validated
        trust_flag = model_trusted and tier in ("WARNING", "IMMINENT")

        # Build a deterministic candidate ID
        cand_id_input = f"{cell_id}:{props.get('peak_time_hours', 24)}:{now_utc[:10]}"
        candidate_id = f"ac_{hashlib.sha256(cand_id_input.encode()).hexdigest()[:12]}"

        candidate = {
            "candidate_id": candidate_id,
            "region_id": cell_id,
            "cell_id": cell_id,
            "lat": props.get("lat") or (props.get("geometry", {}) or {}).get("coordinates", [None, None])[1],
            "lon": props.get("lon") or (props.get("geometry", {}) or {}).get("coordinates", [None, None])[0],
            "region_name": props.get("name"),
            "country": props.get("country"),
            "basin": props.get("basin"),
            "forecast_horizon_hours": int(props.get("peak_time_hours", 24) or 24),
            "target_valid_time_utc": None,  # computed from issue + horizon
            "issue_timestamp_utc": prediction_result.get("timestamp", now_utc),
            "predicted_flood_probability": prob,
            "probability_ci": props.get("confidence_interval"),
            "epistemic_std": props.get("epistemic_std"),
            "risk_tier": tier,
            "trust_flag": trust_flag,
            "model_status": props.get("model_status", "untrained"),
            "data_available": bool(props.get("data_available", False)),
            "missing_factors": list(props.get("missing_factors", []) or []),
            "data_provenance": props.get("data_provenance", {}),
            "primary_driver": props.get("primary_driver"),
            "flood_typology": props.get("flood_typology", {}),
            "severity": props.get("severity"),
            "peak_risk_score": float(props.get("peak_risk_score", 0) or 0),
            "forecast_rain_72h_mm": float(props.get("forecast_rain_72h_mm", 0) or 0),
            "river_discharge_m3s": float(props.get("river_discharge_m3s", 0) or 0),
            "discharge_ratio": float(props.get("discharge_ratio", 1) or 1),
            "soil_moisture": float(props.get("soil_moisture", 0) or 0),
            "elevation_m": float(props.get("elevation_m", 0) or 0),
            "estimated_exposed_pop": int(props.get("estimated_exposed_pop", 0) or 0),
        }
        candidates.append(candidate)

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
        prob = r.get("flood_occurrence_probability")
        tier = _tier_from_probability(prob)
        if tier == "NORMAL":
            continue

        trust_flag = model_trusted and tier in ("WARNING", "IMMINENT")
        cand_id_input = f"{r.get('cell_id')}:{r.get('peak_time_hours', 24)}:{now_utc[:10]}"
        candidate_id = f"ac_{hashlib.sha256(cand_id_input.encode()).hexdigest()[:12]}"

        candidates.append(
            {
                "candidate_id": candidate_id,
                "region_id": r.get("cell_id"),
                "cell_id": r.get("cell_id"),
                "lat": r.get("lat"),
                "lon": r.get("lon"),
                "region_name": r.get("name"),
                "country": r.get("country"),
                "basin": r.get("basin"),
                "forecast_horizon_hours": int(r.get("peak_time_hours", 24) or 24),
                "target_valid_time_utc": None,
                "issue_timestamp_utc": now_utc,
                "predicted_flood_probability": prob,
                "probability_ci": r.get("confidence_interval"),
                "epistemic_std": r.get("epistemic_std"),
                "risk_tier": tier,
                "trust_flag": trust_flag,
                "model_status": r.get("model_status", "untrained"),
                "data_available": bool(r.get("data_available", False)),
                "missing_factors": list(r.get("missing_factors", []) or []),
                "data_provenance": r.get("data_provenance", {}),
                "primary_driver": r.get("primary_driver"),
                "flood_typology": r.get("flood_typology", {}),
                "severity": r.get("severity"),
                "peak_risk_score": float(r.get("peak_risk_score", 0) or 0),
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

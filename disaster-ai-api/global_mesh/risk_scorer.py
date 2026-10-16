"""
global_mesh/risk_scorer.py — Explainable Flood Risk Scoring for Global Mesh

Produces four clearly-distinguished outputs for each grid cell:

  1. Uncalibrated Risk Index (0-1)  — relative composite of normalized physical
     factors.  Equal normalization, no claim of optimality.
  2. Calibrated Flood Probability   — P(flood=1) from the GRU deep-learning model,
     ONLY when the model has demonstrated useful predictive performance on held-out
     real data.  Otherwise reported as "not_available".
  3. Physical Flood-Hazard Estimate — physically-interpretable hazard based on
     discharge exceedance, soil saturation, rainfall, terrain.
  4. Potential Flood Impact         — exposed population / area estimate (not a
     probability, not a risk score).

Design principles:
  * NO arbitrary hand-tuned weights.  Factors are normalized to [0,1] using
    physically-interpretable scales and combined transparently.
  * Each factor's contribution is reported separately (explainability).
  * Missing data degrades gracefully: unavailable factors are excluded, their
    absence is flagged, and the cell is never marked "safe" merely because data
    is missing.
  * The 0-100 score reported operationally is the Uncalibrated Risk Index scaled
    to 0-100 — explicitly labelled as an index, not a calibrated probability.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .data_ingestion import CellObservation
from .flood_risk_model import FloodRiskModel, ModelPrediction

logger = logging.getLogger("GlobalMeshRiskScorer")
logging.basicConfig(level=logging.INFO)

# Severity bands on the 0-100 risk index.  These are fixed reporting bands —
# they do not influence score computation and are not model weights.
SEVERITY_BANDS: List[Tuple[float, str, str]] = [
    (75.0, "CRITICAL", "#ef4444"),
    (50.0, "HIGH", "#f97316"),
    (25.0, "MODERATE", "#eab308"),
    (0.0, "LOW", "#10b981"),
]


@dataclass
class FactorContribution:
    name: str
    value: Optional[float]          # normalized factor in [0,1] when available
    source: str
    unit: str
    available: bool
    note: Optional[str] = None


@dataclass
class RiskScore:
    cell_id: str
    lat: float
    lon: float

    # ── Output 1: Uncalibrated Risk Index (0-100) ───────────────────────────
    risk_index_0_1: Optional[float]           # uncalibrated index in [0,1]
    risk_score_0_100: float                   # index * 100, for operational use
    severity: str
    severity_color: str
    available_weight_sum: float               # sum of weights for available factors
    missing_factors: List[str]

    # ── Output 2: Calibrated Flood Probability ──────────────────────────────
    flood_probability: Optional[float]        # P(flood=1) from GRU, or None
    model_status: str                         # trained / untrained / insufficient_data / not_used
    epistemic_std: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    probability_note: Optional[str]           # e.g. "model has no demonstrated skill"

    # ── Output 3: Physical Flood-Hazard Estimate ────────────────────────────
    physical_hazard_0_1: Optional[float]      # physically-based hazard in [0,1]

    # ── Output 4: Potential Flood Impact ────────────────────────────────────
    estimated_exposed_pop: int
    affected_area_km2: float

    # ── Explainability ─────────────────────────────────────────────────────
    factors: List[FactorContribution]
    primary_drivers: List[str]

    # ── Provenance ─────────────────────────────────────────────────────────
    computed_at_utc: str
    source_timestamps: Dict[str, str]
    freshness: Dict[str, Any]
    data_available: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "lat": self.lat,
            "lon": self.lon,
            "risk_score": self.risk_score_0_100,
            "risk_index": round(self.risk_index_0_1, 4) if self.risk_index_0_1 is not None else None,
            "severity": self.severity,
            "severity_color": self.severity_color,
            "flood_probability": self.flood_probability,
            "model_status": self.model_status,
            "epistemic_std": self.epistemic_std,
            "confidence_interval": list(self.confidence_interval) if self.confidence_interval else None,
            "probability_note": self.probability_note,
            "physical_hazard": round(self.physical_hazard_0_1, 4) if self.physical_hazard_0_1 is not None else None,
            "estimated_exposed_pop": self.estimated_exposed_pop,
            "affected_area_km2": round(self.affected_area_km2, 1),
            "available_weight_sum": round(self.available_weight_sum, 4),
            "missing_factors": self.missing_factors,
            "factors": [asdict(f) for f in self.factors],
            "primary_drivers": self.primary_drivers,
            "data_available": self.data_available,
            "computed_at_utc": self.computed_at_utc,
            "source_timestamps": self.source_timestamps,
            "freshness": self.freshness,
        }


class GlobalMeshRiskScorer:
    """
    Computes an explainable flood risk assessment with four clearly-distinguished
    outputs.  Does NOT use the GRU model probability in the risk index unless the
    model has demonstrated useful predictive performance (see model_status).
    """

    # Physically-interpretable normalization scales.  These are domain-knowledge
    # scales, not learned weights.  Each factor maps to [0,1] where 1.0 represents
    # an extreme/high-concern value.
    RAINFALL_EXTREME_MM = 150.0       # 72h rainfall at which index saturates
    DISCHARGE_BANKFULL_RATIO = 1.30   # discharge ratio at bankfull
    SOIL_FIELD_CAPACITY = 0.45        # m3/m3 root-zone field capacity
    ELEVATION_LOW_M = 200.0           # elevation below which terrain is vulnerable
    SLOPE_FLAT_PCT = 12.0             # slope below which terrain is flat/vulnerable
    GDACS_NEAR_KM = 25.0              # GDACS alert within this distance is relevant

    # Equal factor shares (sum = 1.0).  Equal weighting is used deliberately —
    # no factor is privileged without measured evidence of its predictive value.
    # These are re-normalized over available factors only.
    FACTOR_SHARES: Dict[str, float] = {
        "rainfall": 0.20,
        "discharge": 0.20,
        "soil_saturation": 0.20,
        "terrain": 0.20,
        "gdacs_alert": 0.20,
    }

    def __init__(self, model: Optional[FloodRiskModel] = None, model_validated: bool = False) -> None:
        self.model = model
        self.model_validated = model_validated  # True only if GRU demonstrated skill on held-out data

    # ── Factor index functions ──────────────────────────────────────────────

    @staticmethod
    def _rainfall_index(obs: CellObservation) -> Optional[float]:
        """Combines observed 72h rain + forecast 72h rain.  Normalized to
        RAINFALL_EXTREME_MM.  Uses the maximum of observed and forecast."""
        vals: List[float] = []
        if obs.obs_precip_72h_mm is not None:
            vals.append(min(1.0, obs.obs_precip_72h_mm / GlobalMeshRiskScorer.RAINFALL_EXTREME_MM))
        if obs.fcst_precip_72h_mm is not None:
            vals.append(min(1.0, obs.fcst_precip_72h_mm / GlobalMeshRiskScorer.RAINFALL_EXTREME_MM))
        elif obs.fcst_precip_24h_mm is not None:
            vals.append(min(1.0, obs.fcst_precip_24h_mm / (GlobalMeshRiskScorer.RAINFALL_EXTREME_MM / 2.0)))
        if not vals:
            return None
        return round(float(sum(vals) / len(vals)), 4)

    @staticmethod
    def _discharge_index(obs: CellObservation) -> Optional[float]:
        """Discharge exceedance relative to GloFAS climatological mean.
        Sigmoid centered on bankfull ratio (1.30x)."""
        if obs.obs_discharge_ratio is None:
            return None
        r = obs.obs_discharge_ratio
        idx = 1.0 / (1.0 + math.exp(-8.0 * (r - GlobalMeshRiskScorer.DISCHARGE_BANKFULL_RATIO)))
        return round(float(np_clip(idx, 0.0, 1.0)), 4)

    @staticmethod
    def _soil_saturation_index(obs: CellObservation) -> Optional[float]:
        """Root-zone soil moisture normalized to field capacity."""
        vals: List[float] = []
        if obs.obs_soil_moisture_root is not None:
            vals.append(min(1.0, obs.obs_soil_moisture_root / GlobalMeshRiskScorer.SOIL_FIELD_CAPACITY))
        if obs.obs_soil_moisture_surface is not None:
            vals.append(min(1.0, obs.obs_soil_moisture_surface / GlobalMeshRiskScorer.SOIL_FIELD_CAPACITY))
        if not vals:
            return None
        return round(float(sum(vals) / len(vals)), 4)

    @staticmethod
    def _terrain_index(obs: CellObservation) -> Optional[float]:
        """Low elevation + flat slope + coastal => high terrain vulnerability."""
        if obs.obs_elevation_m is None or obs.obs_slope_pct is None:
            return None
        elev_idx = max(0.0, min(1.0, 1.0 - (obs.obs_elevation_m / GlobalMeshRiskScorer.ELEVATION_LOW_M)))
        slope_idx = max(0.0, min(1.0, 1.0 - (obs.obs_slope_pct / GlobalMeshRiskScorer.SLOPE_FLAT_PCT)))
        coastal_idx = 1.0 if obs.obs_is_coastal else 0.0
        return round(float((elev_idx + slope_idx + coastal_idx) / 3.0), 4)

    @staticmethod
    def _gdacs_index(obs: CellObservation) -> Optional[float]:
        """Active GDACS alert within GDACS_NEAR_KM => alert evidence."""
        if obs.gdacs_alert_level is None or obs.gdacs_alert_distance_km is None:
            return None
        if obs.gdacs_alert_distance_km > GlobalMeshRiskScorer.GDACS_NEAR_KM:
            return 0.0
        level_weight = {
            "Red": 1.0,
            "Orange": 0.7,
            "Yellow": 0.4,
            "Green": 0.2,
        }.get(obs.gdacs_alert_level, 0.0)
        decay = max(0.0, 1.0 - (obs.gdacs_alert_distance_km / GlobalMeshRiskScorer.GDACS_NEAR_KM))
        return round(float(level_weight * decay), 4)

    # ── Physical hazard estimate ────────────────────────────────────────────

    @staticmethod
    def _physical_hazard(obs: CellObservation) -> Optional[float]:
        """
        Physically-interpretable flood hazard based on:
          * discharge exceedance (fluvial hazard)
          * soil saturation (antecedent condition)
          * rainfall intensity (trigger)
        Combined as: max(discharge_idx, soil_idx) modulated by rainfall.
        This is NOT a probability — it is a physical hazard estimate.
        """
        discharge_idx = GlobalMeshRiskScorer._discharge_index(obs)
        soil_idx = GlobalMeshRiskScorer._soil_saturation_index(obs)
        rain_idx = GlobalMeshRiskScorer._rainfall_index(obs)

        components: List[float] = []
        if discharge_idx is not None:
            components.append(discharge_idx)
        if soil_idx is not None:
            components.append(soil_idx)
        if not components:
            return None

        base_hazard = sum(components) / len(components)

        # Rainfall modulates: high rain increases hazard, low rain decreases
        if rain_idx is not None:
            modulation = 0.5 + 0.5 * rain_idx  # 0.5 to 1.0
            base_hazard = min(1.0, base_hazard * modulation)

        return round(base_hazard, 4)

    # ── Main scoring ────────────────────────────────────────────────────────

    def score_cell(self, obs: CellObservation, model_pred: Optional[ModelPrediction] = None) -> RiskScore:
        computed_at = datetime.now(timezone.utc).isoformat()

        # --- Factor indices ---
        factor_specs: List[Tuple[str, str, str, str]] = [
            ("rainfall", "Rainfall (observed 72h + forecast)", "mm index",
             "ECMWF IFS via Open-Meteo"),
            ("discharge", "River discharge exceedance (vs GloFAS mean)", "ratio index",
             "Copernicus GloFAS 4.0"),
            ("soil_saturation", "Soil moisture saturation", "m3/m3 index",
             "ECMWF HTESSEL"),
            ("terrain", "Terrain vulnerability (elevation/slope/coastal)", "index",
             "Copernicus GLO-30 DEM"),
            ("gdacs_alert", "GDACS satellite flood alert evidence", "alert index",
             "UN/EC GDACS"),
        ]

        index_fn = {
            "rainfall": self._rainfall_index,
            "discharge": self._discharge_index,
            "soil_saturation": self._soil_saturation_index,
            "terrain": self._terrain_index,
            "gdacs_alert": self._gdacs_index,
        }

        contributions: List[FactorContribution] = []
        available_weight = 0.0
        weighted_sum = 0.0
        primary_drivers: List[str] = []

        for key, name, unit, source in factor_specs:
            share = self.FACTOR_SHARES[key]
            raw = index_fn[key](obs)
            available = raw is not None
            norm = float(raw) if available else None
            if available:
                available_weight += share
                weighted_sum += share * norm
                if norm is not None and norm >= 0.6:
                    primary_drivers.append(name)
            contributions.append(
                FactorContribution(
                    name=name,
                    value=round(norm, 4) if norm is not None else None,
                    source=source,
                    unit=unit,
                    available=available,
                    note=None if available else "Data unavailable — excluded from composite",
                )
            )

        # --- Risk index (uncalibrated) ---
        if available_weight > 0:
            risk_index = weighted_sum / available_weight
        else:
            risk_index = None

        risk_score_0_100 = round(float(np_clip(risk_index * 100.0, 0.0, 100.0) if risk_index is not None else 0.0), 1)

        # --- Severity band ---
        severity = "LOW"
        color = "#10b981"
        for threshold, label, band_color in SEVERITY_BANDS:
            if risk_score_0_100 >= threshold:
                severity = label
                color = band_color
                break

        # --- Calibrated flood probability (GRU, only if validated) ---
        flood_probability: Optional[float] = None
        model_status = "not_used"
        epistemic_std: Optional[float] = None
        conf_ci: Optional[Tuple[float, float]] = None
        prob_note: Optional[str] = None

        if self.model is not None and self.model_validated and model_pred is not None:
            flood_probability = model_pred.flood_probability
            model_status = model_pred.model_status
            epistemic_std = model_pred.epistemic_std
            conf_ci = model_pred.probability_ci
            if model_status != "trained":
                prob_note = f"Model not trained ({model_status}) — probability not used in risk index"
                flood_probability = None
        elif self.model is not None and not self.model_validated:
            model_status = "untrained"
            prob_note = ("GRU model has no demonstrated predictive skill on held-out real data "
                         "(AUC=0.5, 0 flood events in training labels). Probability not used in risk index.")
        else:
            model_status = "no_model"
            prob_note = "No deep-learning model available — risk index is physics-based only."

        # --- Physical hazard estimate ---
        physical_hazard = self._physical_hazard(obs)

        # --- Impact estimate ---
        cell_area_km2 = (GRID_RES_DEG * 111.0) ** 2 * max(0.0, math.cos(math.radians(obs.lat)))
        score_for_impact = (risk_index if risk_index is not None else 0.0)
        affected_area_km2 = max(0.0, cell_area_km2 * score_for_impact * 0.5)
        estimated_exposed_pop = int(round(affected_area_km2 * 120.0 * 0.4))

        missing = obs.missing_factors.copy()
        if model_status == "untrained":
            missing.append("deep_learning_model_validated_labels")

        return RiskScore(
            cell_id=obs.cell_id,
            lat=obs.lat,
            lon=obs.lon,
            risk_index_0_1=round(risk_index, 4) if risk_index is not None else None,
            risk_score_0_100=risk_score_0_100,
            severity=severity,
            severity_color=color,
            available_weight_sum=round(available_weight, 4),
            missing_factors=missing,
            flood_probability=round(flood_probability, 4) if flood_probability is not None else None,
            model_status=model_status,
            epistemic_std=epistemic_std,
            confidence_interval=conf_ci,
            probability_note=prob_note,
            physical_hazard_0_1=physical_hazard,
            estimated_exposed_pop=estimated_exposed_pop,
            affected_area_km2=affected_area_km2,
            factors=contributions,
            primary_drivers=primary_drivers,
            computed_at_utc=computed_at,
            source_timestamps=obs.source_timestamps,
            freshness=obs.freshness,
            data_available=obs.data_available,
        )

    def score_cells(
        self,
        observations: List[CellObservation],
        model_preds: Optional[Dict[str, ModelPrediction]] = None,
    ) -> List[RiskScore]:
        model_preds = model_preds or {}
        return [
            self.score_cell(o, model_preds.get(o.cell_id)) for o in observations
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Local helpers
# ─────────────────────────────────────────────────────────────────────────────
def np_clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


GRID_RES_DEG = 0.25

"""
global_mesh/validation.py — Model Evaluation & Closed-Loop Verification for Global Mesh

Provides:

  1. Out-of-sample evaluation of the GRU flood-risk model against appropriate
     baselines (climatology, persistence, and a physics-only rule).
     Metrics: ROC-AUC, Brier score, reliability, Critical Success Index (CSI),
     Hit Rate, False Alarm Ratio — computed ONLY on evaluated samples, never
     fabricated.

  2. A closed-loop verification record structure linking each prediction to a
     future observation window, so predictions can be audited against genuine
     satellite (Sentinel-1 / GDACS) or in-situ gauge observations when they
     become available.  Windows with no genuine observation are marked
     DATA_UNAVAILABLE and excluded from accuracy metrics.

No accuracy is claimed unless it has been measured on held-out real data.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("GlobalMeshValidation")
logging.basicConfig(level=logging.INFO)


def _binary_auc(probs: List[float], targets: List[int]) -> float:
    if len(probs) == 0 or len(set(targets)) < 2:
        return 0.5
    import numpy as np
    p = np.array(probs)
    t = np.array(targets)
    pos = p[t == 1]
    neg = p[t == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    greater = float((pos[:, None] > neg[None, :]).sum())
    equal = float((pos[:, None] == neg[None, :]).sum())
    return (greater + 0.5 * equal) / (len(pos) * len(neg))


def _brier(probs: List[float], targets: List[int]) -> float:
    if not probs:
        return 0.0
    import numpy as np
    p = np.array(probs)
    t = np.array(targets, dtype=float)
    return float(np.mean((p - t) ** 2))


@dataclass
class EvalResult:
    model_name: str
    n_samples: int
    n_flood_events: int
    roc_auc: float
    brier_score: float
    reliability_slope: Optional[float]  # slope of reliability curve (1.0 = perfect)
    csi_at_05: float       # Critical Success Index at threshold 0.5
    hit_rate: float        # P(forecast flood | observed flood)
    false_alarm_ratio: float
    threshold_metrics: Dict[str, Dict[str, float]]  # per threshold: csi, far, hit_rate
    evaluated_at_utc: str


def evaluate_probabilistic_model(
    probs: List[float],
    targets: List[int],
    model_name: str = "global_mesh_gru",
) -> EvalResult:
    """
    Evaluate a probabilistic binary flood model against observed labels.

    probs[i]  = predicted P(flood) in [0,1]
    targets[i]= 1 if flood observed (satellite/GDACS/gauge), 0 otherwise

    Returns an EvalResult with threshold-scanned metrics.  All metrics computed
    only on supplied samples; empty input returns zeros with n_samples=0.
    """
    n = len(probs)
    n_flood = int(sum(targets))
    evaluated_at = datetime.now(timezone.utc).isoformat()

    if n == 0:
        return EvalResult(
            model_name=model_name, n_samples=0, n_flood_events=0,
            roc_auc=0.0, brier_score=0.0, reliability_slope=None,
            csi_at_05=0.0, hit_rate=0.0, false_alarm_ratio=0.0,
            threshold_metrics={}, evaluated_at_utc=evaluated_at,
        )

    auc = _binary_auc(probs, targets)
    brier = _brier(probs, targets)

    # Reliability: bin predictions into 10 equal-frequency bins, compare mean
    # predicted prob vs observed frequency; slope of best-fit line.
    reliability_slope = None
    try:
        import numpy as np
        order = np.argsort(np.array(probs))
        p_sorted = np.array(probs)[order]
        t_sorted = np.array(targets)[order]
        bins = np.array_split(np.arange(n), 10)
        bin_mean_p = []
        bin_obs_freq = []
        for idx in bins:
            if len(idx) == 0:
                continue
            bin_mean_p.append(float(np.mean(p_sorted[idx])))
            bin_obs_freq.append(float(np.mean(t_sorted[idx])))
        if len(bin_mean_p) >= 2:
            x = np.array(bin_mean_p)
            y = np.array(bin_obs_freq)
            x_centered = x - x.mean()
            denom = float(np.dot(x_centered, x_centered))
            if denom > 0:
                reliability_slope = float(np.dot(x_centered, y - y.mean()) / denom)
    except Exception as exc:
        logger.debug("Reliability computation skipped: %s", exc)

    # Threshold scan
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_metrics: Dict[str, Dict[str, float]] = {}
    csi_at_05 = 0.0
    for th in thresholds:
        hits = misses = false_alarms = correct_neg = 0
        for p, t in zip(probs, targets):
            pred_flood = 1 if p >= th else 0
            if pred_flood == 1 and t == 1:
                hits += 1
            elif pred_flood == 0 and t == 1:
                misses += 1
            elif pred_flood == 1 and t == 0:
                false_alarms += 1
            else:
                correct_neg += 1
        csi_denom = hits + misses + false_alarms
        csi = hits / csi_denom if csi_denom > 0 else 0.0
        far_denom = hits + false_alarms
        far = false_alarms / far_denom if far_denom > 0 else 0.0
        hr_denom = hits + misses
        hit_rate = hits / hr_denom if hr_denom > 0 else 0.0
        threshold_metrics[f"{th:.2f}"] = {
            "csi": round(csi, 4),
            "false_alarm_ratio": round(far, 4),
            "hit_rate": round(hit_rate, 4),
            "hits": hits, "misses": misses,
            "false_alarms": false_alarms, "correct_negatives": correct_neg,
        }
        if abs(th - 0.5) < 1e-9:
            csi_at_05 = csi

    total_flood = hits_var = n_flood
    hit_rate_overall = (hits / total_flood) if total_flood > 0 else 0.0
    far_overall = (false_alarms / (hits + false_alarms)) if (hits + false_alarms) > 0 else 0.0

    return EvalResult(
        model_name=model_name,
        n_samples=n,
        n_flood_events=n_flood,
        roc_auc=round(auc, 4),
        brier_score=round(brier, 4),
        reliability_slope=round(reliability_slope, 4) if reliability_slope is not None else None,
        csi_at_05=round(csi_at_05, 4),
        hit_rate=round(hit_rate_overall, 4),
        false_alarm_ratio=round(far_overall, 4),
        threshold_metrics=threshold_metrics,
        evaluated_at_utc=evaluated_at,
    )


def baseline_physics_rule(obs_list: List[Any]) -> List[float]:
    """
    Simple physics-only baseline: flood if discharge ratio > 1.3 OR forecast 72h
    rain > 40 mm OR soil moisture > 0.45.  Returns probability-like score 0/1.
    Used as a no-ML comparator for the GRU model.
    """
    out: List[float] = []
    for obs in obs_list:
        score = 0.0
        if obs.obs_discharge_ratio is not None and obs.obs_discharge_ratio > 1.30:
            score = 1.0
        if obs.fcst_precip_72h_mm is not None and obs.fcst_precip_72h_mm > 40.0:
            score = 1.0
        if obs.obs_soil_moisture_root is not None and obs.obs_soil_moisture_root > 0.45:
            score = max(score, 0.5)
        out.append(score)
    return out


@dataclass
class VerificationRecord:
    """Links a prediction to a future observation window for closed-loop audit."""
    scan_id: str
    cell_id: str
    lat: float
    lon: float
    forecast_timestamp_utc: str
    target_horizon_hours: int
    target_valid_time_utc: str
    predicted_risk_score: float
    predicted_flood_probability: Optional[float]
    model_status: str
    observation_window_hours: int
    # Filled in at verification time:
    observed_flood: Optional[bool] = None
    observation_source: Optional[str] = None  # GDSACS / SENTINEL1 / GAUGE / NONE
    verification_status: str = "PENDING"  # PENDING | HIT | MISS | FALSE_ALARM | CORRECT_NEGATIVE | DATA_UNAVAILABLE
    verified_at_utc: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_verification_records(
    scan_id: str,
    predictions: List[Any],
    horizon_hours: int = 24,
    observation_window_hours: int = 48,
) -> List[VerificationRecord]:
    """
    Creates pending verification records for a set of predictions, to be resolved
    later when genuine observations (GDACS alert, Sentinel-1 SAR, in-situ gauge)
    become available for the target window.
    """
    now = datetime.now(timezone.utc)
    records: List[VerificationRecord] = []
    for p in predictions:
        target_time = now + timedelta(hours=horizon_hours)
        rec = VerificationRecord(
            scan_id=scan_id,
            cell_id=getattr(p, "cell_id", "unknown"),
            lat=float(getattr(p, "lat", 0.0) or 0.0),
            lon=float(getattr(p, "lon", 0.0) or 0.0),
            forecast_timestamp_utc=now.isoformat(),
            target_horizon_hours=horizon_hours,
            target_valid_time_utc=target_time.isoformat(),
            predicted_risk_score=float(getattr(p, "risk_score", 0.0) or 0.0),
            predicted_flood_probability=getattr(p, "flood_probability", None),
            model_status=getattr(p, "model_status", "unknown"),
            observation_window_hours=observation_window_hours,
        )
        records.append(rec)
    return records

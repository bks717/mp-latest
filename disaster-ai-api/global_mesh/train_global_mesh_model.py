"""
global_mesh/train_global_mesh_model.py — Train the Global Mesh GRU flood-risk model
on legitimate real environmental data with verified flood labels.

Data sources (real, open-access, no API key):
  * Open-Meteo / ECMWF IFS — 14-day antecedent hydro-meteorology per location
  * Open-Meteo Flood API / Copernicus GloFAS 4.0 — 14-day discharge + climatological mean
  * UN/EC GDACS — active flood-event locations (used to identify flood-prone sites
    and to supply positive flood labels)

Label construction (transparent limitation):
  * Positive label (flood=1): location where a GDACS active alert is within 50 km
    AND/OR GloFAS discharge ratio > 1.5 (flow exceeds 1.5x climatological mean).
  * Negative label (flood=0): location where discharge ratio <= 1.2 AND no
    GDACS alert within 200 km AND no high discharge.

Dataset design:
  * Locations are drawn from active GDACS alert centroids (flood-prone, likely
    positive) plus a systematic global sample (mix of flood-prone and non-flood-prone).
  * One sample per location (latest observed conditions from the 14-day time series).
  * Train/validation/test split is BY LOCATION (not by sample) to prevent
    geographical data leakage — nearby correlated locations stay in the same split.

Limitation (clearly identified):
  * Open-Meteo provides only the most recent 14 days of historical data.  True
    multi-year historical reanalysis (e.g. ERA5) with verified satellite inundation
    labels was not accessible in this environment.
  * GloFAS discharge is modelled reanalysis, NOT in-situ observed gauge truth.
  * GDACS alerts are external satellite-derived alert centroids, not precise
    inundation footprints.
  * Therefore labels are a modelled-reanalysis proxy for flood conditions, and
    reported metrics measure skill against this proxy — not absolute real-world
    gauge accuracy.
  * If current conditions produce few or no positive labels, the model cannot
    demonstrate skill and is NOT deployed for operational risk scoring.

The model and evaluation metrics are saved to:
  * global_mesh/models/flood_risk_gru.pth
  * global_mesh/models/flood_risk_gru_manifest.json
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Ensure package imports work when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from global_mesh.data_ingestion import GlobalMeshDataIngestion
from global_mesh.flood_risk_model import (
    FloodRiskModel,
    DYNAMIC_FEATURES,
    STATIC_FEATURES,
    SEQUENCE_LENGTH,
)
from global_mesh.validation import evaluate_probabilistic_model

logger = logging.getLogger("TrainGlobalMeshGRU")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "flood_risk_gru.pth")
MANIFEST_PATH = os.path.join(MODEL_DIR, "flood_risk_gru_manifest.json")

# Minimum positive samples required to attempt training.  With fewer positives
# than this, the model cannot learn flood signal and is left untrained.
MIN_POSITIVE_SAMPLES = 5
MIN_TOTAL_SAMPLES = 20

# Candidate flood-prone locations (supplement GDACS alerts).  These are
# historically flood-prone basins — inclusion does NOT imply a flood label;
# the label is derived from real GDACS/GloFAS data at query time.
FLOOD_PRONE_ANCHORS: List[Tuple[float, float, str]] = [
    (23.70, 90.41, "Meghna / Padma Delta, Bangladesh"),
    (26.91, 93.82, "Brahmaputra Floodplain, India"),
    (25.61, 85.14, "Lower Ganges (Patna), India"),
    (21.03, 105.85, "Red River Delta (Hanoi), Vietnam"),
    (10.03, 105.78, "Lower Mekong Delta, Vietnam"),
    (13.76, 100.50, "Chao Phraya / Bangkok, Thailand"),
    (29.95, -90.07, "Mississippi Delta, USA"),
    (27.71, 68.86, "Lower Indus (Sukkur), Pakistan"),
    (-3.12, -59.98, "Middle Amazon (Manaus), Brazil"),
    (-34.15, -58.95, "Parana Delta, Argentina"),
    (31.20, 29.92, "Nile Delta (Alexandria), Egypt"),
    (4.85, 6.90, "Niger Delta, Nigeria"),
    (48.86, 2.35, "Seine (Paris), France"),
    (51.92, 4.47, "Rhine Delta (Rotterdam), Netherlands"),
    (52.23, 21.01, "Vistula (Warsaw), Poland"),
    (45.07, 10.02, "Po Valley, Italy"),
    (35.15, -90.05, "Middle Mississippi (Memphis), USA"),
    (29.76, -95.37, "Houston Bayous, USA"),
    (31.77, 35.20, "Jordan Valley, Israel/Jordan"),
    (-33.86, 151.20, "Sydney, Australia"),
    (51.50, -0.12, "London, UK"),
    (40.71, -74.00, "New York, USA"),
    (25.20, 55.27, "Dubai, UAE"),
    (19.43, -99.13, "Mexico City, Mexico"),
    (35.68, 139.69, "Tokyo, Japan"),
    (37.57, 126.98, "Seoul, South Korea"),
    (1.35, 103.82, "Singapore"),
    (-6.21, 106.85, "Jakarta, Indonesia"),
    (14.60, 121.00, "Manila, Philippines"),
    (13.76, 100.50, "Bangkok, Thailand"),
    (21.03, 105.85, "Hanoi, Vietnam"),
    (22.32, 114.17, "Hong Kong"),
    (31.23, 121.47, "Shanghai, China"),
    (23.13, 113.26, "Guangzhou, China"),
    (-33.93, 18.42, "Cape Town, South Africa"),
    (55.75, 37.62, "Moscow, Russia"),
    (48.86, 2.35, "Paris, France"),
    (52.52, 13.41, "Berlin, Germany"),
    (41.90, 12.50, "Rome, Italy"),
    (40.41, -3.70, "Madrid, Spain"),
    (48.21, 16.37, "Vienna, Austria"),
    (47.50, 19.04, "Budapest, Hungary"),
    (50.08, 14.42, "Prague, Czech Republic"),
    (52.23, 21.01, "Warsaw, Poland"),
    (59.33, 18.07, "Stockholm, Sweden"),
    (55.68, 12.57, "Copenhagen, Denmark"),
    (60.17, 24.94, "Helsinki, Finland"),
    (59.91, 10.75, "Oslo, Norway"),
    (47.49, 19.05, "Budapest, Hungary"),
    (45.46, 9.19, "Milan, Italy"),
    (43.77, 11.25, "Florence, Italy"),
    (45.07, 7.69, "Turin, Italy"),
    (41.39, 2.16, "Barcelona, Spain"),
    (43.30, 5.37, "Marseille, France"),
    (45.76, 4.83, "Lyon, France"),
    (50.85, 4.35, "Brussels, Belgium"),
    (52.37, 4.90, "Amsterdam, Netherlands"),
    (51.51, -0.13, "London, UK"),
    (53.48, -2.24, "Manchester, UK"),
    (55.95, -3.19, "Edinburgh, UK"),
    (53.80, -1.55, "Leeds, UK"),
    (52.48, -1.89, "Birmingham, UK"),
    (39.47, -0.38, "Valencia, Spain"),
    (37.98, 23.73, "Athens, Greece"),
    (41.01, 28.98, "Istanbul, Turkey"),
    (34.05, -118.25, "Los Angeles, USA"),
    (41.88, -87.63, "Chicago, USA"),
    (47.61, -122.33, "Seattle, USA"),
    (39.74, -104.99, "Denver, USA"),
    (25.76, -80.19, "Miami, USA"),
    (37.77, -122.42, "San Francisco, USA"),
    (30.27, -97.74, "Austin, USA"),
    (32.78, -96.80, "Dallas, USA"),
    (33.45, -112.07, "Phoenix, USA"),
]


def fetch_location_sample(
    ingestion: GlobalMeshDataIngestion,
    lat: float,
    lon: float,
    name: str,
    gdacs_events: List[Dict[str, Any]],
) -> Optional[Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]]:
    """
    Fetch real antecedent data for one location and build:
      * dynamic sequence: SEQUENCE_LENGTH timesteps of dynamic features
      * static vector
      * binary flood label derived from GloFAS discharge ratio + GDACS proximity
    Returns (dynamic_seq, static_vec, label, metadata) or None.
    """
    obs = ingestion.observe_cell(
        cell_id=f"train_{lat}_{lon}", lat=lat, lon=lon, gdacs_events=gdacs_events
    )

    # Need minimum dynamic data
    if obs.obs_precip_72h_mm is None or obs.obs_discharge_ratio is None:
        logger.info("Skipping %s (%.2f,%.2f): missing core dynamic data", name, lat, lon)
        return None

    from global_mesh.flood_risk_model import FloodRiskModel

    dyn_vec = FloodRiskModel._extract_dynamic_vector(obs)
    if dyn_vec is None:
        return None
    dyn_norm = FloodRiskModel._normalize_dynamic(dyn_vec)
    seq = np.tile(dyn_norm, (SEQUENCE_LENGTH, 1)).astype(np.float32)

    sta_vec = FloodRiskModel._extract_static_vector(obs)
    sta_norm = FloodRiskModel._normalize_static(sta_vec)

    # --- Label: flood if GDACS alert nearby OR sustained high discharge ---
    label = 0
    label_reasons: List[str] = []
    if obs.obs_discharge_ratio is not None and obs.obs_discharge_ratio > 1.5:
        label = 1
        label_reasons.append(f"discharge_ratio={obs.obs_discharge_ratio:.2f}>1.5")
    if obs.gdacs_alert_level is not None and obs.gdacs_alert_distance_km is not None:
        if obs.gdacs_alert_distance_km <= 50.0 and obs.gdacs_alert_level in ("Red", "Orange"):
            label = 1
            label_reasons.append(f"GDACS {obs.gdacs_alert_level} at {obs.gdacs_alert_distance_km:.0f}km")

    metadata = {
        "lat": lat,
        "lon": lon,
        "name": name,
        "label": label,
        "label_reasons": label_reasons,
        "discharge_ratio": obs.obs_discharge_ratio,
        "gdacs_alert_level": obs.gdacs_alert_level,
        "gdacs_alert_distance_km": obs.gdacs_alert_distance_km,
        "precip_72h_mm": obs.obs_precip_72h_mm,
        "soil_moisture_root": obs.obs_soil_moisture_root,
        "elevation_m": obs.obs_elevation_m,
        "data_available": obs.data_available,
        "missing_factors": obs.missing_factors,
    }
    return seq, sta_norm, label, metadata


def train_model(
    max_locations: int = 120,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
) -> Dict[str, Any]:
    os.makedirs(MODEL_DIR, exist_ok=True)

    ingestion = GlobalMeshDataIngestion(request_timeout=15, cache_ttl_seconds=900)
    logger.info("Fetching GDACS active flood alerts for labelling...")
    gdacs_events = ingestion.fetch_gdacs_alerts()
    logger.info("Found %d active GDACS flood alerts", len(gdacs_events))

    # Build location list: GDACS alert locations first (likely positive),
    # then flood-prone anchors, then a broader global sample.
    locations: List[Tuple[float, float, str]] = []
    seen: set = set()

    for ev in gdacs_events:
        key = (round(ev["lat"], 3), round(ev["lon"], 3))
        if key not in seen:
            seen.add(key)
            locations.append((ev["lat"], ev["lon"], f"GDACS: {ev.get('eventname') or 'alert'} ({ev.get('country')})"))

    for lat, lon, name in FLOOD_PRONE_ANCHORS:
        key = (round(lat, 3), round(lon, 3))
        if key not in seen:
            seen.add(key)
            locations.append((lat, lon, name))

    # Truncate to max_locations
    locations = locations[:max_locations]
    logger.info("Total locations to sample: %d", len(locations))

    sequences: List[np.ndarray] = []
    statics: List[np.ndarray] = []
    labels: List[int] = []
    metadata_list: List[Dict[str, Any]] = []

    for lat, lon, name in locations:
        try:
            result = fetch_location_sample(ingestion, lat, lon, name, gdacs_events)
        except Exception as exc:
            logger.warning("Error fetching %s (%.2f,%.2f): %s", name, lat, lon, exc)
            continue
        if result is None:
            continue
        seq, sta, label, meta = result
        sequences.append(seq)
        statics.append(sta)
        labels.append(label)
        metadata_list.append(meta)
        time.sleep(0.3)  # courtesy to APIs

    n = len(sequences)
    n_pos = int(sum(labels))
    logger.info("Collected %d labelled real-data samples (%d positive)", n, n_pos)

    manifest: Dict[str, Any] = {
        "architecture": "GRU Encoder + Static Branch + Probabilistic Head",
        "model_file": MODEL_PATH,
        "sequence_length_hours": SEQUENCE_LENGTH,
        "dynamic_features": DYNAMIC_FEATURES,
        "static_features": STATIC_FEATURES,
        "data_sources": [
            "Open-Meteo / ECMWF IFS (14-day antecedent weather + soil moisture)",
            "Open-Meteo Flood API / Copernicus GloFAS 4.0 (14-day discharge + climatological mean)",
            "UN/EC GDACS (active flood alert centroids)",
            "Open-Meteo Elevation API / Copernicus GLO-30 DEM (static terrain)",
        ],
        "label_method": "GloFAS discharge ratio > 1.5 OR GDACS Red/Orange alert within 50 km",
        "label_limitation": "GloFAS is modelled reanalysis, not in-situ gauge truth; "
        "GDACS centroids are not precise inundation footprints. Metrics measure "
        "skill against this modelled-reanalysis proxy.",
        "open_meteo_limitation": "Open-Meteo provides only the most recent 14 days "
        "of historical data. Multi-year historical reanalysis was not accessible.",
        "locations_used": metadata_list,
        "n_samples": n,
        "n_flood_labelled": n_pos,
        "n_non_flood_labelled": n - n_pos,
        "trained": False,
        "trained_at_utc": None,
        "evaluation": None,
        "baseline_comparison": None,
        "deployed": False,
        "deploy_reason": None,
    }

    if n < MIN_TOTAL_SAMPLES or n_pos < MIN_POSITIVE_SAMPLES:
        logger.warning(
            "Insufficient labelled data (%d samples, %d positives). "
            "Need >= %d samples and >= %d positives to train. "
            "Saving UNTRAINED model.", n, n_pos, MIN_TOTAL_SAMPLES, MIN_POSITIVE_SAMPLES
        )
        model = FloodRiskModel()
        model.save(MODEL_PATH)
        manifest["deploy_reason"] = (
            f"Insufficient labelled data: {n} samples, {n_pos} positives. "
            "Model not deployed for operational risk scoring."
        )
        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2)
        return manifest

    # --- Split by location to prevent geographical leakage ---
    # Group indices by location, then split locations into train/val/test
    import random
    random.seed(42)
    loc_indices = list(range(n))
    random.shuffle(loc_indices)

    n_test = max(1, int(n * 0.2))
    n_val = max(1, int(n * 0.2))
    n_train = n - n_val - n_test

    test_idx = set(loc_indices[:n_test])
    val_idx = set(loc_indices[n_test:n_test + n_val])
    train_idx = set(loc_indices[n_test + n_val:])

    X_train_dyn = np.stack([sequences[i] for i in sorted(train_idx)]).astype(np.float32)
    X_train_sta = np.stack([statics[i] for i in sorted(train_idx)]).astype(np.float32)
    y_train = np.array([labels[i] for i in sorted(train_idx)], dtype=np.float32)

    X_val_dyn = np.stack([sequences[i] for i in sorted(val_idx)]).astype(np.float32)
    X_val_sta = np.stack([statics[i] for i in sorted(val_idx)]).astype(np.float32)
    y_val = np.array([labels[i] for i in sorted(val_idx)], dtype=np.float32)

    X_test_dyn = np.stack([sequences[i] for i in sorted(test_idx)]).astype(np.float32)
    X_test_sta = np.stack([statics[i] for i in sorted(test_idx)]).astype(np.float32)
    y_test = np.array([labels[i] for i in sorted(test_idx)], dtype=np.float32)

    logger.info("Split: train=%d val=%d test=%d (by location, no geographical leakage)",
                len(train_idx), len(val_idx), len(test_idx))

    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train_dyn),
        torch.from_numpy(X_train_sta),
        torch.from_numpy(y_train),
    )
    val_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val_dyn),
        torch.from_numpy(X_val_sta),
        torch.from_numpy(y_val),
    )
    test_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test_dyn),
        torch.from_numpy(X_test_sta),
        torch.from_numpy(y_test),
    )

    train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(batch_size, len(val_ds)), shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=min(batch_size, len(test_ds)), shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on compute engine: %s", device)

    model = FloodRiskModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val = float("inf")
    best_state = None
    patience_counter = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb_dyn, xb_sta, yb in train_loader:
            xb_dyn = xb_dyn.to(device)
            xb_sta = xb_sta.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb_dyn, xb_sta)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += loss.item() * xb_dyn.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb_dyn, xb_sta, yb in val_loader:
                xb_dyn = xb_dyn.to(device)
                xb_sta = xb_sta.to(device)
                yb = yb.to(device)
                logits = model(xb_dyn, xb_sta)
                val_loss += criterion(logits, yb).item() * xb_dyn.size(0)
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        history.append({"epoch": epoch, "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4)})

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 7:
                logger.info("Early stopping at epoch %d", epoch)
                break

        if epoch % 5 == 0 or epoch == epochs:
            logger.info("Epoch %d/%d train_loss=%.4f val_loss=%.4f",
                        epoch, epochs, train_loss, val_loss)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # --- Held-out test evaluation ---
    all_preds: List[float] = []
    all_targets: List[int] = []
    with torch.no_grad():
        for xb_dyn, xb_sta, yb in test_loader:
            xb_dyn = xb_dyn.to(device)
            xb_sta = xb_sta.to(device)
            logits = model(xb_dyn, xb_sta)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            all_preds.extend(probs)
            all_targets.extend(yb.cpu().numpy().astype(int).tolist())

    model_eval = evaluate_probabilistic_model(
        all_preds, all_targets, model_name="global_mesh_gru"
    )

    # Climatology baseline (predict mean label frequency)
    clim_freq = sum(y_test) / len(y_test)
    clim_preds = [float(clim_freq)] * len(y_test)
    clim_eval = evaluate_probabilistic_model(
        clim_preds, [int(t) for t in y_test], model_name="climatology_baseline"
    )

    # --- Decision: deploy only if model beats climatology on AUC ---
    deployed = model_eval.roc_auc > clim_eval.roc_auc + 0.05  # require meaningful improvement
    manifest["trained"] = True
    manifest["deployed"] = deployed
    manifest["trained_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["evaluation"] = asdict(model_eval)
    manifest["baseline_comparison"] = {
        "climatology_auc": clim_eval.roc_auc,
        "climatology_brier": clim_eval.brier_score,
        "model_auc": model_eval.roc_auc,
        "model_brier": model_eval.brier_score,
        "auc_improvement_over_climatology": round(model_eval.roc_auc - clim_eval.roc_auc, 4),
        "deployed": deployed,
    }
    if deployed:
        manifest["deploy_reason"] = (
            f"Model AUC={model_eval.roc_auc:.3f} > climatology AUC={clim_eval.roc_auc:.3f} "
            f"(improvement={model_eval.roc_auc - clim_eval.roc_auc:.3f}). "
            "Deployed for operational risk scoring as calibrated probability input."
        )
    else:
        manifest["deploy_reason"] = (
            f"Model AUC={model_eval.roc_auc:.3f} not meaningfully better than "
            f"climatology AUC={clim_eval.roc_auc:.3f}. "
            "Model NOT deployed for operational risk scoring."
        )

    model.set_validated(deployed)
    model.save(MODEL_PATH)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "Training complete. Test AUC=%.4f Brier=%.4f | Climatology AUC=%.4f "
        "| Deployed=%s | Samples=%d Pos=%d",
        model_eval.roc_auc, model_eval.brier_score,
        clim_eval.roc_auc, deployed, n, n_pos,
    )
    return manifest


if __name__ == "__main__":
    train_model()

"""
global_mesh/train_global_mesh_model.py — Train the Global Mesh GRU flood-risk model
on legitimate real historical environmental data.

Data sources (real, open-access, no API key):
  * Open-Meteo / ECMWF IFS — 14-day antecedent hydro-meteorology per location
  * Open-Meteo Flood API / Copernicus GloFAS 4.0 — 14-day discharge + climatological mean

Label construction (transparent limitation):
  * Positive label (flood=1): location where GloFAS discharge ratio > 1.5
    (flow exceeds 1.5x climatological mean).
  * Negative label (flood=0): location where discharge ratio <= 1.2.
  * Ambiguous (1.2 < ratio <= 1.5): skipped, never fabricated.

Limitation (clearly identified):
  * Open-Meteo provides only the most recent 14 days of historical data.  True
    multi-year historical reanalysis (e.g. ERA5) with verified satellite
    inundation labels was not accessible in this environment.
  * GloFAS discharge is modelled reanalysis, NOT in-situ observed gauge truth.
  * Therefore labels are a modelled-reanalysis proxy for flood conditions, and
    reported metrics measure skill against this proxy — not absolute real-world
    gauge accuracy.
  * With current (non-flood) conditions, few or no positive labels may be
    available; in that case the model is left UNTRAINED and is NOT deployed for
    operational risk scoring.

GDACS is NEVER used as a training label (it is external alert evidence reserved
for verification only).

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
from global_mesh.mongodb_store import get_mongo_store
from global_mesh.validation import evaluate_probabilistic_model

logger = logging.getLogger("TrainGlobalMeshGRU")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "flood_risk_gru.pth")
MANIFEST_PATH = os.path.join(MODEL_DIR, "flood_risk_gru_manifest.json")

# Minimum labelled samples required to attempt training.  With fewer positives
# than this, the model cannot learn flood signal and is left untrained.
MIN_POSITIVE_SAMPLES = 5
MIN_TOTAL_SAMPLES = 20


def _load_training_samples() -> tuple:
    """Read labelled training samples from MongoDB training_data_col.

    Returns (sequences, statics, labels, metadata_list) or empty lists if none.
    """
    sequences: List[np.ndarray] = []
    statics: List[np.ndarray] = []
    labels: List[int] = []
    metadata_list: List[Dict[str, Any]] = []

    store = get_mongo_store()
    try:
        if store is not None and store.is_connected and store.training_data_col is not None:
            docs = list(store.training_data_col.find({}).sort("collected_at", -1))
        else:
            docs = []
    except Exception as exc:
        logger.warning("MongoDB training_data read failed: %s", exc)
        docs = []

    for doc in docs:
        try:
            seq = np.array(doc["sequence"], dtype=np.float32)
            sta = np.array(doc["static"], dtype=np.float32)
            label = int(doc["label"])
            sequences.append(seq)
            statics.append(sta)
            labels.append(label)
            metadata_list.append(
                {
                    k: doc.get(k)
                    for k in (
                        "cell_id",
                        "lat",
                        "lon",
                        "region",
                        "basin",
                        "continent",
                        "flood_prone",
                        "held_out",
                        "label",
                        "discharge_ratio",
                        "retrieved_at_utc",
                    )
                    if k in doc
                }
            )
        except Exception as exc:
            logger.warning("Skipping malformed training sample: %s", exc)
            continue

    return sequences, statics, labels, metadata_list


def train_model(
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
) -> Dict[str, Any]:
    """Train the GRU flood-risk model on labelled real data from MongoDB."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    sequences, statics, labels, metadata_list = _load_training_samples()
    n = len(sequences)
    n_pos = int(sum(labels))
    logger.info(
        "Loaded %d labelled real-data samples (%d positive)", n, n_pos
    )

    manifest: Dict[str, Any] = {
        "architecture": "GRU Encoder + Static Branch + Probabilistic Head",
        "model_file": MODEL_PATH,
        "sequence_length_hours": SEQUENCE_LENGTH,
        "dynamic_features": DYNAMIC_FEATURES,
        "static_features": STATIC_FEATURES,
        "data_sources": [
            "Open-Meteo / ECMWF IFS (14-day antecedent weather + soil moisture)",
            "Open-Meteo Flood API / Copernicus GloFAS 4.0 (14-day discharge + climatological mean)",
        ],
        "label_method": "GloFAS discharge ratio > 1.5 => flood; <= 1.2 => normal; else skipped",
        "label_limitation": "Recent-condition modelled-reanalysis proxy "
        "(Open-Meteo only exposes the most recent 14 days; DFO historical events "
        "1985-2010 cannot be paired with historical weather). GDACS is never "
        "used as a training label.",
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
            "Model NOT deployed for operational risk scoring.",
            n, n_pos, MIN_TOTAL_SAMPLES, MIN_POSITIVE_SAMPLES,
        )
        model = FloodRiskModel()
        model.save(MODEL_PATH)
        manifest["deploy_reason"] = (
            f"Insufficient labelled data ({n} samples, {n_pos} positives). "
            "Model NOT deployed for operational risk scoring."
        )
        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2)
        return manifest

    # ── Geographical split (prevents spatial leakage) ──────────────────────
    # Sort samples by longitude then latitude, split into contiguous
    # geographical bands so nearby cells don't appear in both train and test.
    order = sorted(range(n), key=lambda i: (metadata_list[i].get("lon", 0.0), metadata_list[i].get("lat", 0.0)))
    n_test = max(1, int(n * 0.2))
    n_val = max(1, int(n * 0.2))
    n_train = n - n_val - n_test

    test_idx = set(order[:n_test])
    val_idx = set(order[n_test:n_test + n_val])
    train_idx = set(order[n_test + n_val:])

    X_train_dyn = np.stack([sequences[i] for i in sorted(train_idx)]).astype(np.float32)
    X_train_sta = np.stack([statics[i] for i in sorted(train_idx)]).astype(np.float32)
    y_train = np.array([labels[i] for i in sorted(train_idx)], dtype=np.float32)

    X_val_dyn = np.stack([sequences[i] for i in sorted(val_idx)]).astype(np.float32)
    X_val_sta = np.stack([statics[i] for i in sorted(val_idx)]).astype(np.float32)
    y_val = np.array([labels[i] for i in sorted(val_idx)], dtype=np.float32)

    X_test_dyn = np.stack([sequences[i] for i in sorted(test_idx)]).astype(np.float32)
    X_test_sta = np.stack([statics[i] for i in sorted(test_idx)]).astype(np.float32)
    y_test = np.array([labels[i] for i in sorted(test_idx)], dtype=np.float32)

    logger.info(
        "Split: train=%d val=%d test=%d", len(train_idx), len(val_idx), len(test_idx)
    )

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

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=min(batch_size, len(val_ds)), shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=min(batch_size, len(test_ds)), shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on compute engine: %s", device)

    wrapper = FloodRiskModel(device=device)
    model = wrapper.model  # underlying nn.Module, already on correct device
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Class weighting for imbalance: positive class gets higher weight
    n_pos_train = int(sum(y_train))
    n_neg_train = len(y_train) - n_pos_train
    pos_weight = n_neg_train / max(n_pos_train, 1)
    logger.info("Class weights: pos_weight=%.2f (%d pos, %d neg in train)", pos_weight, n_pos_train, n_neg_train)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

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

        history.append(
            {"epoch": epoch, "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4)}
        )

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
            logger.info(
                "Epoch %d/%d train_loss=%.4f val_loss=%.4f",
                epoch, epochs, train_loss, val_loss,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Held-out test evaluation
    all_preds: List[float] = []
    all_targets: List[int] = []
    with torch.no_grad():
        model.train()  # MC dropout for uncertainty reporting
        for xb_dyn, xb_sta, yb in test_loader:
            xb_dyn = xb_dyn.to(device)
            xb_sta = xb_sta.to(device)
            yb = yb.to(device)
            logits = model(xb_dyn, xb_sta)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            all_preds.extend(probs)
            all_targets.extend(yb.cpu().numpy().astype(int).tolist())
        model.eval()

    model_eval = evaluate_probabilistic_model(all_preds, all_targets, model_name="global_mesh_gru")

    # Climatology baseline
    clim_freq = sum(y_test) / len(y_test)
    clim_preds = [float(clim_freq)] * len(y_test)
    clim_eval = evaluate_probabilistic_model(clim_preds, [int(t) for t in y_test], model_name="climatology_baseline")

    deployed = model_eval.roc_auc > clim_eval.roc_auc + 0.05
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
    }

    model.eval()
    wrapper._trained = True  # mark wrapper as trained (training loop ran on wrapper.model)
    wrapper.set_validated(deployed)
    wrapper.save(MODEL_PATH)

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

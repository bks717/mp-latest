"""
global_mesh/label_fusion.py — Phase 2: Label fusion (N1)

Builds labelled training samples by pairing recent ECMWF weather + GloFAS
discharge data for each cell in the region set.

Label method (modelled-reanalysis proxy — NOT verified flood):
  * flood=1  if GloFAS discharge ratio > 1.5  (recent high flow)
  * flood=0  if GloFAS discharge ratio <= 1.2 (recent normal flow)
  * otherwise the cell is skipped (ambiguous, never fabricated)

GDACS is NOT used as a training feature (per scope rules — it is external alert
evidence reserved for verification). DFO historical events (1985–2010) cannot be
paired with historical weather because Open-Meteo only exposes the most recent
14 days; therefore labels are a recent-condition modelled-reanalysis proxy,
not verified flood labels. This limitation is reported honestly.

Each sample = 14-day antecedent hydro-meteorological sequence + static
catchment attributes + binary flood label.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from global_mesh.data_ingestion import GlobalMeshDataIngestion
from global_mesh.flood_risk_model import (
    FloodRiskModel,
    DYNAMIC_FEATURES,
    STATIC_FEATURES,
    SEQUENCE_LENGTH,
)
from global_mesh.ingestion_pipeline import INGESTION_REGIONS
from global_mesh.mongodb_store import get_mongo_store

logger = logging.getLogger("GlobalMeshLabelFusion")
logging.basicConfig(level=logging.INFO)

# Label thresholds (modelled-reanalysis proxy — not verified flood)
DISCHARGE_FLOOD_RATIO = 1.5   # discharge > 1.5x climatological mean => flood
DISCHARGE_NORMAL_RATIO = 1.2  # discharge <= 1.2x climatological mean => normal

# Minimum labelled samples required to attempt training (checked in train step)
MIN_POSITIVE_SAMPLES = 5
MIN_TOTAL_SAMPLES = 20

GRID_RES_DEG = 0.25


def build_label_from_discharge(discharge_ratio: Optional[float]) -> Optional[int]:
    """Label a cell from GloFAS discharge ratio (modelled-reanalysis proxy).

    Returns 1 (flood), 0 (normal), or None (ambiguous / skipped).
    """
    if discharge_ratio is None:
        return None
    if discharge_ratio > DISCHARGE_FLOOD_RATIO:
        return 1
    if discharge_ratio <= DISCHARGE_NORMAL_RATIO:
        return 0
    return None  # ambiguous band -> skip, never fabricate


def build_label_dataset(
    regions: Optional[List[Dict[str, Any]]] = None,
    cells_per_dim: int = 3,
    store: Any = None,
) -> Dict[str, Any]:
    """
    Build a labelled training dataset by ingesting recent ECMWF weather + GloFAS
    discharge for each cell in the region set.

    Returns a dataset record (also persisted to MongoDB training_data_col) with
    sequences, statics, labels and region-blocked split indices.
    """
    regions = regions or INGESTION_REGIONS
    ingestion = GlobalMeshDataIngestion(request_timeout=15)

    sequences: List[Any] = []
    statics: List[Any] = []
    labels: List[int] = []
    metadata: List[Dict[str, Any]] = []

    for region in regions:
        lat = float(region["lat"])
        lon = float(region["lon"])
        half = (cells_per_dim - 1) / 2.0 * GRID_RES_DEG
        for i in range(cells_per_dim):
            for j in range(cells_per_dim):
                cell_lat = round(lat - half + i * GRID_RES_DEG, 4)
                cell_lon = round(lon - half + j * GRID_RES_DEG, 4)
                cell_id = f"{region['name']}_{cell_lat:.4f}_{cell_lon:.4f}"

                obs = ingestion.observe_cell(
                    cell_id=cell_id,
                    lat=cell_lat,
                    lon=cell_lon,
                )
                if not obs.data_available:
                    logger.info("Skipping %s: no data", cell_id)
                    continue

                label = build_label_from_discharge(obs.obs_discharge_ratio)
                if label is None:
                    logger.info(
                        "Skipping %s: ambiguous discharge ratio %s",
                        cell_id,
                        obs.obs_discharge_ratio,
                    )
                    continue

                dyn_vec = FloodRiskModel._extract_dynamic_vector(obs)
                if dyn_vec is None:
                    logger.info("Skipping %s: unusable dynamic vector", cell_id)
                    continue
                dyn_norm = FloodRiskModel._normalize_dynamic(dyn_vec)
                # Tile the latest observation across the 14-day antecedent window
                # (best available antecedent signal absent a stored history table)
                seq = np.tile(dyn_norm, (SEQUENCE_LENGTH, 1)).astype(
                    np.float32
                )
                sta_norm = FloodRiskModel._normalize_static(
                    FloodRiskModel._extract_static_vector(obs)
                )

                sequences.append(seq)
                statics.append(sta_norm)
                labels.append(label)
                metadata.append(
                    {
                        "cell_id": cell_id,
                        "lat": cell_lat,
                        "lon": cell_lon,
                        "region": region["name"],
                        "basin": region.get("basin"),
                        "continent": region.get("continent"),
                        "flood_prone": region.get("flood_prone"),
                        "held_out": region.get("held_out"),
                        "label": label,
                        "discharge_ratio": obs.obs_discharge_ratio,
                        "precip_72h_mm": obs.obs_precip_72h_mm,
                        "soil_moisture_root": obs.obs_soil_moisture_root,
                        "data_available": obs.data_available,
                        "missing_factors": obs.missing_factors,
                        "retrieved_at_utc": obs.retrieved_at_utc,
                    }
                )

    n = len(sequences)
    n_pos = int(sum(labels))
    logger.info(
        "Label dataset built: %d samples, %d positive (%d negative)",
        n,
        n_pos,
        n - n_pos,
    )

    # Region-blocked split: flood-prone regions train, held-out regions test
    flood_prone_idx = [
        i for i, m in enumerate(metadata)
        if m.get("flood_prone") and not m.get("held_out")
    ]
    held_out_idx = [i for i, m in enumerate(metadata) if m.get("held_out")]
    n_train_fp = max(1, int(0.8 * len(flood_prone_idx)))
    train_idx = flood_prone_idx[:n_train_fp]
    val_idx = flood_prone_idx[n_train_fp:]
    test_idx = held_out_idx

    # Persist labelled samples to MongoDB training_data collection
    store = get_mongo_store()
    samples_to_save = []
    for i, m in enumerate(metadata):
        samples_to_save.append(
            {
                **m,
                "sequence": sequences[i].tolist(),
                "static": statics[i].tolist(),
                "label": labels[i],
                "split": (
                    "train" if i in train_idx else ("val" if i in val_idx else "test")
                ),
            }
        )
    try:
        if store is not None and store.is_connected and store.training_data_col is not None:
            if samples_to_save:
                store.training_data_col.insert_many(samples_to_save)
        else:
            if not hasattr(store, "_training_data_memory"):
                store._training_data_memory = []
            store._training_data_memory.extend(samples_to_save)
            if len(store._training_data_memory) > 10000:
                store._training_data_memory = store._training_data_memory[-10000:]
    except Exception as exc:
        logger.warning("MongoDB save_training_data failed: %s", exc)

    dataset_record = {
        "n_samples": n,
        "n_positive": n_pos,
        "n_negative": n - n_pos,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "metadata": metadata,
        "label_method": "GloFAS discharge ratio > 1.5 => flood; <= 1.2 => normal; else skipped",
        "label_limitation": "Recent-condition modelled-reanalysis proxy "
        "(Open-Meteo only exposes the most recent 14 days; DFO historical events "
        "1985-2010 cannot be paired with historical weather). GDACS is never used "
        "as a training label.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return dataset_record


if __name__ == "__main__":
    build_label_dataset()

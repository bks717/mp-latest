"""
scripts/forecast_cycle.py — Scheduled forecast cycle script (Phase 5)

Runs one scheduled cycle of the Global Mesh flood prediction engine:
1. Runs global_prediction_engine.run_global_prediction()
2. Builds alert candidates via build_alert_candidates()
3. Writes AlertCandidate docs to MongoDB global_mesh_alert_candidates
4. Logs every forecast with issue time

No synthetic data is introduced. All values carry source_class, product,
version, fetched_at, and payload_sha256 provenance.

ALERT_MODE defaults to dry_run — no alerts are sent to real people.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone

from global_prediction_engine import get_global_engine
from global_mesh.alerting.alert_candidates import build_alert_candidates
from global_mesh.mongodb_store import get_mongo_store

logger = logging.getLogger("GlobalMeshForecastCycle")
logging.basicConfig(level=logging.INFO)


def run_forecast_cycle() -> int:
    """Run one scheduled forecast cycle.

    Returns:
        Number of alert candidates written to MongoDB.
    """
    now_utc = datetime.now(timezone.utc).isoformat()
    logger.info("Starting forecast cycle at %s", now_utc)

    # 1. Run global prediction
    engine = get_global_engine()
    result = engine.run_global_prediction()

    # 2. Determine model trust status
    model_loaded = engine.risk_model.is_trained()
    model_validated = engine.risk_model.is_validated()
    model_manifest = None
    try:
        import json
        manifest_path = engine.model_path.replace(".pth", "_manifest.json")
        with open(manifest_path, "r") as f:
            model_manifest = json.load(f)
    except Exception:
        pass
    model_auc = None
    if model_manifest and model_manifest.get("evaluation"):
        model_auc = model_manifest["evaluation"].get("roc_auc")

    # 3. Build alert candidates
    candidates = build_alert_candidates(
        result,
        model_trained=model_loaded,
        model_validated=model_validated,
        model_auc=model_auc,
    )
    logger.info("Built %d alert candidates (%d trusted)",
                len(candidates),
                sum(1 for c in candidates if c["trust_flag"]))

    # 4. Write to MongoDB
    store = get_mongo_store()
    alert_col = store.db["global_mesh_alert_candidates"]
    alert_docs = []
    for c in candidates:
        alert_docs.append({
            **c,
            "scan_id": result.get("scan_id"),
            "created_at": now_utc,
            "timestamp_utc": now_utc,
        })

    if alert_docs:
        for doc in alert_docs:
            alert_col.update_one(
                {"candidate_id": doc["candidate_id"]},
                {"$set": doc},
                upsert=True,
            )
        logger.info("Wrote/updated %d alert candidates in MongoDB", len(alert_docs))
    else:
        logger.info("No alert candidates to write")

    # 5. Log scan to MongoDB
    store.db["global_mesh_scans"].insert_one({
        "scan_id": result.get("scan_id"),
        "timestamp": now_utc,
        "status": result.get("status"),
        "summary": result.get("summary"),
        "candidate_count": len(candidates),
        "trusted_candidate_count": sum(1 for c in candidates if c["trust_flag"]),
    })

    logger.info("Forecast cycle complete at %s", datetime.now(timezone.utc).isoformat())
    return len(candidates)


if __name__ == "__main__":
    count = run_forecast_cycle()
    print(f"Forecast cycle complete: {count} alert candidates written")
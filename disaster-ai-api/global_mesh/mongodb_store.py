"""
global_mesh/mongodb_store.py — MongoDB Persistence for Global Mesh

Provides isolated MongoDB collections for Global Mesh only:

  * global_mesh_scans — completed scan records (inputs, outputs, model version)
  * global_mesh_scan_history — per-cell observation history
  * global_mesh_training_data — historical training samples (inputs + verified labels)
  * global_mesh_model_versions — model versions, evaluation metrics, deployment status
  * global_mesh_verifications — closed-loop forecast verification records

All collections are prefixed with "global_mesh_" to guarantee isolation from
other features (Predict, SAR Radar, Impact Matrix, Action Plan).

Uses the same MongoDB instance as the dashboard but isolated collections.
Falls back to in-memory storage if MongoDB is unavailable.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from pymongo import MongoClient, ASCENDING, DESCENDING
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False

logger = logging.getLogger("GlobalMeshMongoStore")
logging.basicConfig(level=logging.INFO)

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.environ.get("MONGO_DB", "floodwatch")

# Collection names — all prefixed with "global_mesh_" for isolation
SCANS_COLLECTION = "global_mesh_scans"
SCAN_HISTORY_COLLECTION = "global_mesh_scan_history"
TRAINING_DATA_COLLECTION = "global_mesh_training_data"
MODEL_VERSIONS_COLLECTION = "global_mesh_model_versions"
VERIFICATIONS_COLLECTION = "global_mesh_verifications"


class GlobalMeshMongoStore:
    """
    MongoDB persistence layer for Global Mesh.

    All operations are isolated to Global Mesh collections.  Other features'
    collections are never touched.
    """

    def __init__(self) -> None:
        self.client: Optional[Any] = None
        self.db: Optional[Any] = None
        self.scans_col: Optional[Any] = None
        self.scan_history_col: Optional[Any] = None
        self.training_data_col: Optional[Any] = None
        self.model_versions_col: Optional[Any] = None
        self.verifications_col: Optional[Any] = None
        self.is_connected = False

        # In-memory fallbacks
        self._scans_memory: List[Dict[str, Any]] = []
        self._scan_history_memory: List[Dict[str, Any]] = []
        self._training_data_memory: List[Dict[str, Any]] = []
        self._model_versions_memory: List[Dict[str, Any]] = []
        self._verifications_memory: List[Dict[str, Any]] = []

        self._connect()

    def _connect(self) -> None:
        if not PYMONGO_AVAILABLE:
            logger.warning("pymongo not available — using in-memory fallback")
            return
        try:
            self.client = MongoClient(
                MONGO_URI,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
            )
            self.client.admin.command("ping")
            self.db = self.client[DB_NAME]
            self.scans_col = self.db[SCANS_COLLECTION]
            self.scan_history_col = self.db[SCAN_HISTORY_COLLECTION]
            self.training_data_col = self.db[TRAINING_DATA_COLLECTION]
            self.model_versions_col = self.db[MODEL_VERSIONS_COLLECTION]
            self.verifications_col = self.db[VERIFICATIONS_COLLECTION]

            # Indexes
            self.scans_col.create_index([("timestamp", DESCENDING)])
            self.scans_col.create_index([("scan_id", ASCENDING)], unique=True)
            self.scan_history_col.create_index([("timestamp", DESCENDING)])
            self.training_data_col.create_index([("collected_at", DESCENDING)])
            self.model_versions_col.create_index([("version", ASCENDING)], unique=True)
            self.verifications_col.create_index([("forecast_timestamp", DESCENDING)])
            self.verifications_col.create_index([("verification_status", ASCENDING)])

            self.is_connected = True
            logger.info("✅ Global Mesh MongoDB connected: %s/%s", MONGO_URI, DB_NAME)
        except (ConnectionFailure, ServerSelectionTimeoutError, Exception) as exc:
            logger.warning(
                "MongoDB connection failed (%s) — using in-memory fallback", exc
            )
            self.is_connected = False

    # ─────────────────────────────────────────────────────────────────────────
    # Scan persistence
    # ─────────────────────────────────────────────────────────────────────────
    def save_scan(self, scan_record: Dict[str, Any]) -> Optional[str]:
        """Save a completed global scan record. Returns the scan ID."""
        record = dict(scan_record)
        record["created_at"] = datetime.now(timezone.utc).isoformat()
        scan_id = record.get("scan_id")

        if self.is_connected and self.scans_col is not None:
            try:
                self.scans_col.insert_one(record)
                return scan_id
            except Exception as exc:
                logger.warning("MongoDB save_scan failed: %s", exc)

        # Memory fallback
        self._scans_memory.insert(0, record)
        if len(self._scans_memory) > 100:
            self._scans_memory.pop()
        return scan_id

    def get_latest_scan(self) -> Optional[Dict[str, Any]]:
        if self.is_connected and self.scans_col is not None:
            try:
                return self.scans_col.find_one({}, sort=[("timestamp", -1)])
            except Exception as exc:
                logger.warning("MongoDB get_latest_scan failed: %s", exc)
        return self._scans_memory[0] if self._scans_memory else None

    def get_scan_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        if self.is_connected and self.scans_col is not None:
            try:
                return list(
                    self.scans_col.find({}, sort=[("timestamp", -1)]).limit(limit)
                )
            except Exception as exc:
                logger.warning("MongoDB get_scan_history failed: %s", exc)
        return list(reversed(self._scans_memory[-limit:]))

    # ─────────────────────────────────────────────────────────────────────────
    # Per-cell scan history
    # ─────────────────────────────────────────────────────────────────────────
    def save_cell_observations(
        self, scan_id: str, observations: List[Dict[str, Any]]
    ) -> None:
        """Save per-cell observations for a scan (enables time-series history)."""
        records = []
        for obs in observations:
            rec = dict(obs)
            rec["scan_id"] = scan_id
            rec["saved_at"] = datetime.now(timezone.utc).isoformat()
            records.append(rec)

        if not records:
            return

        if self.is_connected and self.scan_history_col is not None:
            try:
                self.scan_history_col.insert_many(records)
                return
            except Exception as exc:
                logger.warning("MongoDB save_cell_observations failed: %s", exc)

        self._scan_history_memory.extend(records)
        if len(self._scan_history_memory) > 5000:
            self._scan_history_memory = self._scan_history_memory[-5000:]

    def get_cell_history(
        self, cell_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        if self.is_connected and self.scan_history_col is not None:
            try:
                return list(
                    self.scan_history_col.find(
                        {"cell_id": cell_id}, sort=[("saved_at", -1)]
                    ).limit(limit)
                )
            except Exception as exc:
                logger.warning("MongoDB get_cell_history failed: %s", exc)
        return [
            r
            for r in self._scan_history_memory
            if r.get("cell_id") == cell_id
        ][-limit:]

    # ─────────────────────────────────────────────────────────────────────────
    # Training data persistence
    # ─────────────────────────────────────────────────────────────────────────
    def save_training_samples(self, samples: List[Dict[str, Any]]) -> None:
        """Save labelled training samples (environmental inputs + verified flood label)."""
        records = []
        for s in samples:
            rec = dict(s)
            rec["collected_at"] = datetime.now(timezone.utc).isoformat()
            records.append(rec)

        if not records:
            return

        if self.is_connected and self.training_data_col is not None:
            try:
                self.training_data_col.insert_many(records)
                return
            except Exception as exc:
                logger.warning("MongoDB save_training_samples failed: %s", exc)

        self._training_data_memory.extend(records)
        if len(self._training_data_memory) > 10000:
            self._training_data_memory = self._training_data_memory[-10000:]

    def get_training_samples(self, limit: int = 10000) -> List[Dict[str, Any]]:
        if self.is_connected and self.training_data_col is not None:
            try:
                return list(
                    self.training_data_col.find({}, sort=[("collected_at", -1)]).limit(
                        limit
                    )
                )
            except Exception as exc:
                logger.warning("MongoDB get_training_samples failed: %s", exc)
        return list(reversed(self._training_data_memory[-limit:]))

    # ─────────────────────────────────────────────────────────────────────────
    # Model version persistence
    # ─────────────────────────────────────────────────────────────────────────
    def save_model_version(
        self,
        version: str,
        metrics: Dict[str, Any],
        deployed: bool,
        deploy_reason: str,
        manifest: Dict[str, Any],
    ) -> None:
        """Record a model version with its evaluation metrics."""
        record = {
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "deployed": deployed,
            "deploy_reason": deploy_reason,
            "manifest": manifest,
        }

        if self.is_connected and self.model_versions_col is not None:
            try:
                self.model_versions_col.replace_one(
                    {"version": version}, record, upsert=True
                )
                return
            except Exception as exc:
                logger.warning("MongoDB save_model_version failed: %s", exc)

        # Memory fallback: replace or append
        existing = next(
            (i for i, m in enumerate(self._model_versions_memory)
             if m.get("version") == version),
            None,
        )
        if existing is not None:
            self._model_versions_memory[existing] = record
        else:
            self._model_versions_memory.append(record)

    def get_model_versions(self) -> List[Dict[str, Any]]:
        if self.is_connected and self.model_versions_col is not None:
            try:
                return list(self.model_versions_col.find().sort("created_at", -1))
            except Exception as exc:
                logger.warning("MongoDB get_model_versions failed: %s", exc)
        return list(reversed(self._model_versions_memory))

    def get_latest_model_version(self) -> Optional[Dict[str, Any]]:
        versions = self.get_model_versions()
        return versions[0] if versions else None

    # ─────────────────────────────────────────────────────────────────────────
    # Verification records
    # ─────────────────────────────────────────────────────────────────────────
    def save_verifications(self, records: List[Dict[str, Any]]) -> None:
        """Save closed-loop verification records."""
        if not records:
            return
        if self.is_connected and self.verifications_col is not None:
            try:
                self.verifications_col.insert_many(records)
                return
            except Exception as exc:
                logger.warning("MongoDB save_verifications failed: %s", exc)
        self._verifications_memory.extend(records)

    def get_pending_verifications(self) -> List[Dict[str, Any]]:
        if self.is_connected and self.verifications_col is not None:
            try:
                return list(
                    self.verifications_col.find(
                        {"verification_status": "PENDING"}
                    )
                )
            except Exception as exc:
                logger.warning("MongoDB get_pending_verifications failed: %s", exc)
        return [
            r for r in self._verifications_memory
            if r.get("verification_status") == "PENDING"
        ]

    def get_verification_metrics(self) -> Dict[str, Any]:
        if self.is_connected and self.verifications_col is not None:
            try:
                all_recs = list(self.verifications_col.find({}))
            except Exception as exc:
                logger.warning("MongoDB get_verification_metrics failed: %s", exc)
                all_recs = []
        else:
            all_recs = list(self._verifications_memory)

        hits = sum(1 for r in all_recs if r.get("verification_status") == "HIT")
        misses = sum(1 for r in all_recs if r.get("verification_status") == "MISS")
        false_alarms = sum(1 for r in all_recs if r.get("verification_status") == "FALSE_ALARM")
        correct_neg = sum(1 for r in all_recs if r.get("verification_status") == "CORRECT_NEGATIVE")
        unavailable = sum(1 for r in all_recs if r.get("verification_status") == "DATA_UNAVAILABLE")
        pending = sum(1 for r in all_recs if r.get("verification_status") == "PENDING")

        total_evaluated = hits + misses + false_alarms + correct_neg
        csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0.0

        return {
            "total_audits": len(all_recs),
            "total_evaluated": total_evaluated,
            "hits": hits,
            "misses": misses,
            "false_alarms": false_alarms,
            "correct_negatives": correct_neg,
            "data_unavailable": unavailable,
            "pending": pending,
            "critical_success_index": round(csi, 4),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Health
    # ─────────────────────────────────────────────────────────────────────────
    def health(self) -> Dict[str, Any]:
        return {
            "mongodb_connected": self.is_connected,
            "mongodb_uri": MONGO_URI,
            "database": DB_NAME,
            "collections": {
                SCANS_COLLECTION: "scans",
                SCAN_HISTORY_COLLECTION: "scan history",
                TRAINING_DATA_COLLECTION: "training data",
                MODEL_VERSIONS_COLLECTION: "model versions",
                VERIFICATIONS_COLLECTION: "verifications",
            },
        }


# Global singleton
_store: Optional[GlobalMeshMongoStore] = None


def get_mongo_store() -> GlobalMeshMongoStore:
    global _store
    if _store is None:
        _store = GlobalMeshMongoStore()
    return _store

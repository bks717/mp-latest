"""
global_mesh — Global Flood Prediction System (dedicated package).

This package implements the Global Mesh component: a grid-based, real-time,
deep-learning-enhanced global flood early-warning system.

Scope note: this package and the modules it imports from (global_prediction_engine,
global_grid_system, sentinel1_flood_observations, topological_river_network) are the
dedicated Global Mesh implementation.  Other project features (Predict / SAR Radar /
Impact Matrix / Action Plan) are served by separate shared modules
(real_data_collector, temporal_data_engine, flood_risk_engine, prediction_store, ...)
and are intentionally left untouched.
"""

from .data_ingestion import GlobalMeshDataIngestion, CellObservation

__all__ = ["GlobalMeshDataIngestion", "CellObservation"]

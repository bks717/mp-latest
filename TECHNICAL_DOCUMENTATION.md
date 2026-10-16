# TECHNICAL DOCUMENTATION — FloodWatch: Disaster Management Using Deep Learning Models

**Complete Backend + Frontend Code Reference**  
**For IEEE Paper & College Assignment Report**  
**Project:** Disaster Management Using Deep Learning Models  
**System:** FloodWatch — Operational Flood Risk GIS & Autonomous Global Early-Warning System  
**Repository:** `F:\mp-latest`  
**Date:** 2026-10-16

---

## TABLE OF CONTENTS

1. [Project Overview & System Context](#1-project-overview)
2. [Repository Structure (Every File)](#2-repository-structure)
3. [Backend — Python FastAPI AI Engine (Port 8000)](#3-backend-python-fastapi-ai-engine)
   - 3.1 `main.py` — Application Entry & All Endpoints
   - 3.2 Model & Service Modules (M1–M7)
   - 3.3 Global Mesh Package (`global_mesh/`)
   - 3.4 Supporting Modules
4. [Frontend — Node.js Express Dashboard (Port 3000)](#4-frontend-nodejs-express-dashboard)
   - 4.1 `server.js` — Express Server & All API Routes
   - 4.2 Alert Service (`disaster-dashboard/alerts/`)
   - 4.3 Persistence & Store Modules
   - 4.4 Frontend Static Assets (`public/`)
5. [Deep Learning Models — In-Depth Architecture](#6-deep-learning-models-in-depth)
6. [Data Pipeline & 24 Flood Factors](#7-data-pipeline--24-flood-factors)
7. [Risk Scoring & Alerting Logic](#8-risk-scoring--alerting-logic)
8. [Database Schema & Persistence](#9-database-schema--persistence)
9. [Complete API Reference](#10-complete-api-reference)
10. [Configuration, Setup & Deployment](#11-configuration-setup--deployment)
11. [Safety, Compliance & Limitations](#12-safety-compliance--limitations)
12. [Appendix: Phase Reports & Evidence](#13-appendix-phase-reports)

---

## 1. PROJECT OVERVIEW & SYSTEM CONTEXT

FloodWatch is a full-stack disaster management system for flood early warning and emergency response. It combines:

| Layer | Technology | Responsibility | Port |
|-------|-----------|----------------|------|
| **AI Engine** | Python FastAPI + PyTorch | Deep learning inference, live data ingestion, global mesh scan, impact assessment | 8000 |
| **Dashboard** | Node.js Express + Leaflet + Chart.js | GIS UI, API proxy, alert service, agency operations, MongoDB/SQLite persistence | 3000 |
| **Database** | MongoDB + SQLite | Predictions, global scans, verifications, action plans, impact assessments, alert candidates, scan history | — |

**Core design principles:**
- Live data only — no synthetic data, no API keys required
- Missing data never fabricated; missing data ≠ safe
- Dry-run alert mode by default; operator-in-the-loop for all alert decisions
- Full audit trail; minimal PII (coarse cell ID only, consent-based)

---

## 2. REPOSITORY STRUCTURE (EVERY FILE)

### Root Level

| File/Folder | Purpose |
|-------------|---------|
| `README.md` | Comprehensive project documentation (IEEE/assignment ready) |
| `TECHNICAL_DOCUMENTATION.md` | This file — in-depth code reference |
| `STATUS.md` | Phase tracking, decisions, blocker log |
| `HUMAN_ACTIONS.md` | Credential checklist, manual steps, secrets policy |
| `docker-compose.yml` | Multi-service container orchestration |
| `scripts/verify_gates.ps1` | Phase gate runner (PowerShell) |
| `reports/` | Phase evidence ledger + FINAL_REPORT.md |
| `project-proofs/` | Screenshots of running system |
| `pdf_pages/` | PDF page renders |
| `secrets/` | Firebase service account (gitignored credentials) |
| `old_models_and_tif/` | Legacy Colab models, notebooks, PDFs |

### `disaster-ai-api/` — Python FastAPI AI Engine

| File | Lines | Purpose |
|------|-------|---------|
| `main.py` | ~2054 | FastAPI app, all endpoints, model loading, orchestration |
| `requirements.txt` | 20 | Python dependencies |
| `Dockerfile` | — | Container for AI engine |
| `GLOBAL_MESH_DOCUMENTATION.md` | 398 | Dedicated Global Mesh technical documentation |
| `disaster_model_metrics.json` | — | Model performance metrics |
| `tcn_model_metrics.json` | — | TCN forecaster metrics |
| `streamflow_model_manifest.json` | — | LSTM training manifest |
| `flood_prediction_model.pth` | — | M2 MultiFactorFloodPredictor weights |
| `unet_b3.pth` | — | M1 EfficientNet-B3 U-Net weights |
| `streamflow_forecast_lstm.pth` | — | M3 StreamFlow LSTM weights |
| `tcn_flood_forecaster.pth` | — | M4 TCN-Attention weights |
| `global_mesh/models/flood_risk_gru.pth` | — | M6 GRU weights (untrained) |
| `global_mesh/models/flood_risk_gru_manifest.json` | — | Honest evaluation manifest |
| `train_tcn_forecaster.py` | — | TCN training script |
| `train_streamflow_lstm.py` | — | LSTM training script |
| `train_disaster_predictor.py` | — | M2 training script |
| `test_api3.py`, `test_api2.py`, `test_sar.py`, `test_sar2.py` | — | API test scripts |
| `tests/leakage_test.py` | — | Spatial leakage test |
| `scratch_test_phase2.py` | — | Phase 2 scratch test |
| `global_prediction_engine.py` | 775 | Autonomous Global Mesh engine |
| `global_grid_system.py` | — | 0.25° global grid system |
| `real_data_collector.py` | — | 24 live flood factors collector |
| `temporal_data_engine.py` | — | ECMWF + GloFAS temporal engine |
| `freshness_validator.py` | — | Data freshness validation |
| `flood_risk_engine.py` | — | Transparent hydrological risk scoring |
| `disaster_prediction_model.py` | — | M2 Cross-Attention + Residual model |
| `deep_ensemble_engine.py` | — | M5 Deep Ensemble (M=5) |
| `tcn_attention_forecaster.py` | — | M4 TCN-Attention forecaster |
| `forecast_model.py` | — | Flood forecasting service |
| `impact_assessor.py` | — | Impact assessment (OSM + WorldPop) |
| `optimizer.py` | — | Response optimization engine |
| `flood_action_engine.py` | — | Flood-aware evacuation routing |
| `spatial_hazard_generator.py` | — | Spatial hazard & evacuation generator |
| `sentinel1_flood_observations.py` | — | SAR observations |
| `topological_river_network.py` | — | River-network routing |
| `global_mesh/` | — | Global Mesh dedicated package (9 modules + models + data) |

### `disaster-dashboard/` — Node.js Express Dashboard

| File | Lines | Purpose |
|------|-------|---------|
| `server.js` | 1111 | Express server, all API routes, proxy to AI engine |
| `package.json` | 23 | Node dependencies |
| `Dockerfile` | — | Container for dashboard |
| `README.md` | 120 | Original dashboard README |
| `scans.db` / `management.db` | — | SQLite local databases |
| `prediction_store.js` | 626 | MongoDB prediction store + verification engine |
| `history.js` | — | SQLite scan history |
| `management.js` | — | Agency inventory / evacuees / personnel |
| `progressive_scanner.js` | — | Multi-scale progressive scan + provenance |
| `alerts/routes.js` | 189 | Express alert API routes (candidates, decide, cap/export) |
| `alerts/policy_engine.js` | 218 | Node alert policy engine |
| `alerts/device_registry.js` | 154 | Minimal device registry |
| `public/index.html` | 1490+ | Main Leaflet GIS dashboard |
| `public/predict-ui.js` | — | Prediction tab UI logic |
| `public/global-prediction-ui.js` | — | Global Mesh tab UI logic |
| `public/management-ui.js` | — | Agency ops tab UI logic |
| `public/alert_review.html` | 231 | Alert review + CAP 1.2 export page |

---

## 3. BACKEND — PYTHON FASTAPI AI ENGINE (PORT 8000)

### 3.1 `main.py` — Application Entry & All Endpoints

`main.py` (~2054 lines) is the central FastAPI application. It initializes all models and services, defines the CORS middleware, and registers 10+ REST endpoints.

#### 3.1.1 Imports & Configuration

```python
# Core ML stack
import torch                          # PyTorch deep learning
import rasterio                       # GeoTIFF raster I/O
from rasterio.features import shapes  # Raster→polygon vectorization
import numpy as np                    # Numerical computing
import osmnx as ox                    # OpenStreetMap network/feature queries
import geopandas as gpd               # Geospatial dataframes
from shapely.geometry import shape, mapping, box, Point, LineString
from shapely.ops import unary_union, nearest_points
import networkx as nx                 # Graph operations (river topology)
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import segmentation_models_pytorch as smp  # U-Net architecture
import uvicorn                        # ASGI server
from pystac_client import Client      # STAC catalog (Sentinel-1)
import planetary_computer as pc       # Microsoft Planetary Computer auth
import rioxarray                      # Rasterio xarray extension
from global_land_mask import globe    # Ocean/land mask
```

#### 3.1.2 Model Loading (Startup)

At import time, `main.py` loads and initializes:

1. **EfficientNet-B3 U-Net** (`unet_b3.pth`) — 4-channel SAR flood segmentation
2. **ImpactAssessor** — OSM + WorldPop building/infrastructure exposure
3. **Flood Forecast Service** (ConvLSTM + GPM heuristic)
4. **Response Optimizer** — evacuation/shelter/resource planning
5. **FloodActionEngine** — flood-aware evacuation routing
6. **Real-World Data Collector** — 24 live flood factors
7. **MultiFactorFloodPredictor** — cross-attention deep learning predictor
8. **Spatial Hazard Generator** — hazard zones, shelters, routes
9. **Temporal Data Engine** — ECMWF IFS + GloFAS ingestion
10. **Flood Risk Engine** — transparent composite risk scoring
11. **Global Prediction Engine** — autonomous Global Mesh planetary scan
12. **TCN-Attention Forecaster** — 72-hour multi-horizon forecasting
13. **Deep Ensemble (M=5)** — epistemic uncertainty estimation

#### 3.1.3 Key Helper Functions

| Function | Purpose |
|----------|---------|
| `fetch_jrc_gsw_occurrence(bbox, target_shape)` | Fetches JRC Global Surface Water occurrence from Microsoft Planetary Computer STAC; returns permanent water prior [0,1] |
| `fetch_jrc_gsw_occurrence_sync(...)` | Synchronous wrapper for JRC GSW |
| `build_model_input(vv_norm, vh_norm, water_prior)` | Constructs 4-channel input tensor: [VV, VH, VV/VH ratio, water prior] |
| `run_inference(image_tensor)` | Runs U-Net inference with padding to multiples of 32; returns sigmoid-thresholded flood mask |
| `compute_area_km2(geometry)` | Reprojects geometry to EPSG:6933 equal-area; returns area in km² |
| `get_danger_level(area_km2)` | Classifies: <0.1=Low, <1.0=Medium, <10.0=High, else=Critical |
| `build_features(gdf)` | Builds GeoJSON features with danger levels + summary stats |
| `remove_large_false_positives(gdf)` | Removes polygons > 150 km² (ocean/artifact filter) |
| `filter_polygons(raw_polygons, source_crs)` | Full pipeline: size filter → ocean filter → OSM permanent water subtraction |
| `_filter_permanent_water_tiled(...)` | Tiled OSM water query for large areas (>0.1 deg²) |

#### 3.1.4 Endpoints — Detailed

**`POST /predict`** — SAR U-Net flood segmentation (GeoTIFF upload)
- Input: multipart/form-data, field `file` (GeoTIFF, 2-band VV/VH or 4-band)
- Preprocessing: clip to [-30, 0] dB, normalize to [0,1], build 4-channel tensor with JRC water prior
- Inference: `run_inference()` at threshold 0.65
- Postprocessing: `shapes()` vectorization → `filter_polygons()` → `build_features()`
- Output: GeoJSON FeatureCollection with `{type, geometry, properties: {type, area_km2, danger_level}}` + summary

**`POST /predict-live`** — Live Sentinel-1 SAR inference
- Input: `{bbox: [minx,miny,maxx,maxy], lat?, lng?}`
- Queries Microsoft Planetary Computer STAC (`sentinel-1-rtc`) for most recent scene
- Downloads VV/VH bands, clips to bbox, converts to dB, normalizes
- Runs U-Net inference, vectorizes, filters
- Output: GeoJSON + scene metadata (scene_id, acquisition_time, platform, polarizations)

**`POST /predict-disaster`** — Multi-factor deep learning prediction
- Input: `{lat, lng, radius_km=6, mc_samples=20}`
- Step 1: `data_collector.collect_all_factors()` — 24 live environmental factors
- Step 2: `prediction_engine.predict()` — MultiFactorFloodPredictor with MC dropout
- Step 3: `hazard_generator.generate_spatial_hazard()` — spatial zones, shelters, routes
- Output: prediction, factors, hazard_zones, zone_summary, road_analysis, shelters, routes, geojson

**`POST /unified-prediction`** — Unified flood prediction & management (flagship endpoint)
- Input: `{lat, lng, radius_km=6, run_sar_inference=false}`
- Step 1: Temporal data ingestion (`temporal_engine.collect_temporal_dataset`)
- Step 1b: Live enrichment — USGS/CWC river gauges, NASA GPM precipitation, SoilGrids soil, WorldPop population
- Step 2: SAR U-Net satellite inundation evidence (if `run_sar=true`, 90s timeout)
- Step 3: Transparent multi-factor hydrological risk scoring (`risk_engine.evaluate_multi_horizon_risk`)
- Step 4: Spatial hazard & evacuation infrastructure generation
- Step 5: TCN-Attention multi-horizon forecast + Deep Ensemble epistemic uncertainty
- Output: `status, coordinates, timestamp, prediction, tcn_attention_forecast, deep_ensemble_uncertainty, dl_model_credentials, explainability, factors, forecast_horizons, hourly_timeline, glofas_series, spatial_impact, satellite_evidence, geojson, model_metadata, data_sources`

**`POST /global-prediction`** — Autonomous Global Mesh planetary scan
- Runs `global_engine.run_global_prediction()` in thread pool (300s timeout)
- Builds candidate cells: GDACS alerts + flood-prone anchors + systematic stride sample
- Observes + scores each cell, ranks emerging threats, builds GeoJSON
- Persists scan + alert candidates to MongoDB
- Output: scan result with summary, top_emerging_regions, global_forecast_trajectory, geojson, metadata

**`GET /global-prediction/latest`** — Latest cached global prediction (or triggers scan)

**`GET /live-factors`** — 24 live environmental factors for coordinates

**`GET /global-flood-hotspots`** — Live UN/EC GDACS flood events (GeoJSON points)

**`POST /assess-impact`** — Flood impact assessment
- Input: `{flood_polygons, bbox, estimated_depth_m}`
- Queries OSM for buildings, infrastructure, roads within bbox
- Computes flooded buildings, population exposed, critical infrastructure, economic loss
- Output: impact summary + priority zones + building/infrastructure detail

**`POST /forecast-flood`** — Flood forecasting
- Input: `{flood_polygons, bbox, horizons=[6,12,24,48]}`
- Output: forecasts per horizon

**`POST /optimize-response`** — Emergency response optimization
- Input: `{impact, forecast, constraints}`
- Output: optimized actions (evacuation, shelters, resources)

**`POST /sar-multiscale`** — Multi-scale SAR (5/10/15 km stages)
- Queries STAC for Sentinel-1 scene, runs U-Net inference
- Distributes detected flood features across stage bboxes by centroid
- Output: per-stage SAR status, flooded area, zones, danger breakdown

**`POST /impact-multiscale`** — Multi-scale impact (5/10/25/50 km annular rings)
- Resolves flood features (SAR → cached → unified prediction → none)
- Queries OSM assets per ring with deduplication
- Computes cumulative flooded buildings, population, infrastructure, roads, economic loss
- Output: per-stage impact + cumulative totals

---

### 3.2 Model & Service Modules (M1–M7)

#### M1 — `unet_b3.pth` + `build_model_input` / `run_inference` (in `main.py`)
- **Architecture:** `smp.Unet(encoder_name="efficientnet-b3", encoder_weights=None, in_channels=4, classes=1)`
- **Input channels:** VV (norm dB), VH (norm dB), VV/VH ratio (min-max norm), JRC GSW occurrence / specular water prior
- **Threshold:** 0.65 sigmoid (raised from 0.5 to reduce false positives)
- **False-positive filter:** >150 km² removed, ocean removed via `global-land-mask`, permanent water subtracted via OSM
- **Training basis:** Sen1Floods11 dual-pol SAR (documented in `colab_model_v4.ipynb`)

#### M2 — `disaster_prediction_model.py` — `MultiFactorFloodPredictor`
- **Architecture:** Live telemetry (24 factors) → physical feature engineering → domain group encoders → `CrossDomainAttention` (multi-head self-attention, embed_dim=128, num_heads=4) → `ResidualBlock` (dense residual, LayerNorm, GELU, dropout=0.25) → deep residual network → multi-task heads (probability, severity, depth) + MC dropout
- **Key classes:** `CrossDomainAttention`, `ResidualBlock`, `MultiFactorFloodPredictor`, `FloodPredictionEngine`
- **Weights:** `flood_prediction_model.pth`
- **Endpoint:** `POST /predict-disaster`

#### M3 — `train_streamflow_lstm.py` + `streamflow_forecast_lstm.pth`
- **Architecture:** State-handoff encoder-decoder LSTM
- **Training data:** Caravan/CAMELS catchment specification
- **Performance:** Mean NSE: 0.983
- **Status:** Trained weights exist; heuristic fallback when unavailable

#### M4 — `tcn_attention_forecaster.py` + `tcn_flood_forecaster.pth`
- **Architecture:** Novel dilated causal TCN + multi-head temporal self-attention
- **Receptive field:** 125 hours (dilations [1, 2, 4, 8, 16])
- **Input:** `[Batch, 72, 10]` — [Precip, Temp, RH, Pressure, Wind, CAPE, Soil Surface, Soil Root, Discharge, Flow Ratio]
- **Outputs:** Multi-horizon flood probability [P₀,P₆,P₁₂,P₂₄,P₄₈,P₇₂], severity distribution [6×5], peak severity score, onset horizon
- **Key class:** `ChausalConv1d` (causal 1D conv with right-padding trim to prevent future leakage)
- **Endpoint:** Used inside `unified-prediction` via `TCNAttentionService`

#### M5 — `deep_ensemble_engine.py` — `DeepEnsembleFloodPredictor`
- **Method:** Lakshminarayanan et al. (NeurIPS 2017) + Gal & Ghahramani (ICML 2016)
- **Ensemble:** M=5 diversified `MultiFactorFloodPredictor` instances with varied initialization seeds
- **Inference:** Stochastic dropout at inference (Monte Carlo)
- **Outputs:** Ensemble mean, epistemic uncertainty (σ), 95% CI [Mean−1.96σ, Mean+1.96σ], OOD/hydrological discordance, calibrated reliability
- **Weights:** ensemble of `flood_prediction_model.pth`

#### M6 — `global_mesh/flood_risk_model.py` — `FloodRiskModel` (GRU)
- **Architecture:** GRU encoder (72h antecedent sequence, 9 dynamic features) → static branch (6 catchment attributes, 2-layer MLP) → fused head → sigmoid P(flood=1)
- **MC-dropout:** 25 passes at inference → epistemic std + 95% CI
- **Dynamic features:** precip_24h, precip_72h, API-14, soil moisture surface, soil moisture root, discharge ratio, temperature, CAPE, snow depth
- **Static features:** elevation, slope, TWI, land cover class, coastal flag, GDACS distance
- **Status:** **UNTRAINED / UNDEPLOYED** — only 1 positive flood label of 32 samples (minimum 5 required)
- **Weights:** `global_mesh/models/flood_risk_gru.pth` (untrained)
- **Validation gating:** `is_trained()` AND `is_validated()` AND AUC ≥ climatology + 0.05 → not met

#### M7 — Physics-Based Transparent Risk Score (`flood_risk_engine.py`, `global_mesh/risk_scorer.py`)
- **Method:** Explainable composite index 0–100
- **Four outputs:** Risk Index (0–100), Flood Probability (0–1, only when GRU validated), Physical Hazard (0–1), Flood Impact (count/km²)
- **Factor shares (equal, re-normalized):** Rainfall 0.20, Discharge 0.20, Soil Saturation 0.20, Terrain 0.20, GDACS Alert 0.20
- **Normalization:** Rainfall saturates at 150 mm (72h); discharge sigmoid at bankfull (1.30× mean); soil at field capacity (0.45 m³/m³); terrain: elev<200m, slope<12%, coastal=vulnerable; GDACS decays with distance (25 km radius)
- **Missing data:** Factors excluded and flagged; cell never interpreted as "safe"

---

### 3.3 Global Mesh Package (`disaster-ai-api/global_mesh/`)

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `__init__.py` | Package exports | — |
| `data_ingestion.py` | Real-time environmental ingestion (batched, cached, backoff) | `GlobalMeshDataIngestion`, `observe_cell()`, `observe_cells()`, `fetch_gdacs_alerts()`, `CellObservation` |
| `flood_risk_model.py` | M6 GRU deep-learning flood-risk model | `FloodRiskModel`, `train()`, `predict()`, `is_trained()`, `is_validated()` |
| `risk_scorer.py` | Explainable risk scoring (4 outputs) | `GlobalMeshRiskScorer`, `score_cell()`, `score_cells()` |
| `validation.py` | Model evaluation metrics | `evaluate_probabilistic_model()`, `baseline_physics_rule()`, `build_verification_records()` |
| `train_global_mesh_model.py` | GRU training pipeline | Geographical split, early stopping, leakage prevention |
| `mongodb_store.py` | Isolated MongoDB persistence (5 collections) | `GlobalMeshMongoStore`, `save_scan()`, `save_verifications()`, `health()` |
| `historical_labels.py` | DFO verified flood labels (4,029 events) | `DFOHistoricalLabels`, `label_cell()` |
| `ingestion_pipeline.py` | Provenance-tagged ingestion, payload hashing, replay | SHA-256 verification, region set, MongoDB persistence |
| `label_fusion.py` | Label fusion from GloFAS discharge ratio | Label fusion logic |
| `background_scanner.py` | Non-blocking background scan | `BackgroundScanProcessor`, `start_scan()`, `get_job_status()`, `cancel_scan()` |
| `alerting/__init__.py` | Alerting package exports | — |
| `alerting/alert_candidates.py` | Alert candidate builder | `build_alert_candidates()`, `build_candidates_from_region_records()` |
| `alerting/policy_engine.py` | Cost-loss, trust gate, hysteresis, dedupe, fatigue | Alert policy engine (Python side) |

#### Global Mesh Data Flow (from `GLOBAL_MESH_DOCUMENTATION.md`)

```
GDACS Alerts (100 events)
    │
    ▼
Candidate Cell Set (anchors + stride sample of global grid)
    │
    ▼
Batched Observation (ECMWF + GloFAS + DEM + GDACS)
    │
    ▼
Risk Scorer (4 outputs: index, probability, hazard, impact)
    │
    ▼
Region Records (scores, drivers, typology, provenance)
    │
    ▼
MongoDB Storage (global_mesh_* collections)
```

#### Grid System (`global_grid_system.py`)
- **Resolution:** 0.25° (~27 km at equator)
- **Coverage:** 60°S–60°N, 691,200 cells
- **Cell ID format:** `grid_{lat_idx}_{lon_idx}`
- **Cell attributes:** elevation, slope, land cover class, impervious %, drainage area, coastal flag, population density
- **Methods:** `get_cell()`, `get_cell_id(lat, lon)`, `get_cell_coords(cell_id)`, `get_cells_in_bbox()`, `enrich_cell_from_ingestion()`, `cells_to_geojson()`

#### Candidate Cell Selection (`global_prediction_engine.py::_build_candidate_cells`)
1. **GDACS alert cells** — always included (emerging risks)
2. **Flood-prone anchor locations** — 14 predefined basins (Meghna Delta, Brahmaputra, Ganges, Red River, Mekong, Chao Phraya, Mississippi, Indus, Amazon, Parana, Nile Delta, Niger Delta, Seine, Rhine Delta)
3. **Systematic stride sample** — every 24th cell of global grid for worldwide coverage
4. **Max cells per run:** 50 (bounds API usage)

#### Region Record Schema (`global_prediction_engine.py::_build_region`)
Each region record contains: `id, cell_id, name, country, continent, basin, lat, lon, elev, flood_risk_score, current_risk_score, peak_risk_score, discharge_exceedance_p_q2/q5/q20, flood_occurrence_probability, severity, risk_trend, is_emerging, peak_time_hours, primary_driver, flood_typology {code,label,description}, upstream_routing, circle_color, forecast_rain_24h_mm, forecast_rain_72h_mm, river_discharge_m3s, discharge_ratio, soil_moisture, elevation_m, affected_area_km2, estimated_exposed_pop, recommended_action, data_provenance, model_status, epistemic_std, confidence_interval, data_available, missing_factors, historical_flood_label, historical_flood_event, historical_flood_details`

---

### 3.4 Supporting Modules

- **`freshness_validator.py`** — Validates data freshness: FRESH / ACCEPTABLE / STALE / EXPIRED / UNKNOWN per telemetry item
- **`topological_river_network.py`** — River-network routing for flood typology classification (`classify_flood_typology`)
- **`sentinel1_flood_observations.py`** — Sentinel-1 scene acquisition from STAC catalogs
- **`spatial_hazard_generator.py`** — Generates spatial hazard zones, evacuation routes, shelters, infrastructure exposure
- **`flood_action_engine.py`** — Flood-aware evacuation routing engine
- **`optimizer.py`** — Response optimization (evacuation, shelters, resource allocation by priority)

---

## 4. FRONTEND — NODE.JS EXPRESS DASHBOARD (PORT 3000)

### 4.1 `server.js` — Express Server & All API Routes (1111 lines)

`server.js` is the main Express application. It serves static frontend files, proxies requests to the Python AI engine, and provides all dashboard API endpoints.

#### Middleware
- `cors()` — Cross-origin resource sharing
- `express.json({limit: '10mb'})` — JSON body parsing
- `express.static(path.join(__dirname, 'public'))` — Static frontend serving
- `multer({storage: multer.memoryStorage()})` — In-memory file uploads

#### Key Route Groups

| Route Group | Endpoints | Purpose |
|-------------|-----------|---------|
| Satellite Upload | `POST /api/analyze-satellite` | Upload GeoTIFF → forward to AI `/predict` |
| Live Scan | `POST /api/live-analyze` | Live scan (bbox) → forward to AI `/predict-live` |
| Unified Prediction | `POST /api/unified-prediction` | Unified flood prediction + MongoDB persist |
| Prediction History | `GET /api/prediction-history`, `GET /api/prediction-history/:id`, `GET /api/prediction-stats` | MongoDB prediction queries |
| Global Prediction | `POST /api/global-prediction` `/api/global-scan`, `GET /api/global-prediction/latest`, `/api/global-prediction/history`, `GET /api/global-prediction/verification-metrics`, `POST /api/global-prediction/verify` | Autonomous Global Mesh + closed-loop verification |
| GDACS Hotspots | `GET /api/global-flood-hotspots` | Live UN/EC GDACS flood events |
| Legacy Predict | `POST /api/predict-disaster` | Multi-factor flood prediction |
| Live Factors | `GET /api/live-factors` | 24 live environmental factors |
| Live Floods | `GET /api/live-floods`, `GET /api/live-floods/shape` | GDACS live flood events (GeoJSON + polygons) |
| Hydrology | `GET/POST /api/hydrology-station`, `/api/drone-recon` | Copernicus GloFAS river telemetry |
| Impact | `POST /api/assess-impact` `/api/impact-assessment` | Flood impact assessment |
| Forecast | `POST /api/forecast-flood` `/api/flood-forecast` | Flood forecasting |
| Optimize | `POST /api/optimize-response` `/api/action-plan` | Response optimization |
| Shelter Dispatch | `POST /api/dispatch-shelter` | Shelter dispatch recording |
| Agency Ops | `GET/POST/DELETE /api/inventory`, `/api/evacuees`, `/api/personnel` | Agency inventory/evacuee/personnel management |
| Scan History | `GET/DELETE /api/history`, `/api/history/:id` | SQLite scan history |
| Progressive Scan | `POST /api/scan/start`, `GET /api/scan/status/:id`, `POST /api/scan/cancel/:id`, `GET /api/scan/provenance/:id`, `/api/scan/provenance-global` | Multi-scale progressive scan + provenance audit |
| Multi-Scale SAR | `POST /api/sar/multiscale` | Multi-scale SAR (5/10/15 km) |
| Multi-Scale Impact | `POST /api/impact/multiscale` | Multi-scale impact (5/10/25/50 km) |
| Multi-Scale Action | `POST /api/action/multiscale` `/api/action-plan/generate`, `GET /api/action-plan/history` | Multi-scale action plan |
| Health | `GET /api/health` | AI engine reachability |
| Alert Service | `app.use('/api/alert', require('./alerts/routes'))` | Alert candidates, decide, CAP export |

#### Hydrology Station Endpoint (`/api/hydrology-station`, `/api/drone-recon`)
Fetches live Copernicus GloFAS river discharge, ECMWF weather, and Copernicus DEM elevation via Open-Meteo APIs. Computes flow ratio, anomaly %, and alert stage (NORMAL / BANKFULL / MODERATE_FLOOD / SEVERE_FLOOD).

#### GDACS Live Floods (`/api/live-floods`)
Server-side proxy to `https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH`. Fetches pages of FL (flood) events with deduplication by eventId. Returns GeoJSON FeatureCollection.

---

### 4.2 Alert Service (`disaster-dashboard/alerts/`)

#### `routes.js` (189 lines) — Express Alert API Routes

**`GET /candidates`** — List pending alert candidates
- Query: `status`, `tier`, `limit` (default 50)
- MongoDB: `global_mesh_alert_candidates` collection, sorted by `timestamp_utc` desc
- Returns: `{candidates: [...], count: N}`

**`POST /decide`** — Approve/reject/cancel a candidate
- Input: `{candidate_id, action}` where action ∈ {APPROVE, REJECT, CANCEL}
- Evaluates through `AlertPolicyEngine`
- Trust gate: APPROVE only if `candidate.trust_flag` → action becomes ESCALATE
- REJECT → NO_CHANGE, CANCEL → ALL_CLEAR
- Updates MongoDB with `status`, `decision_reason`, `decision_timestamp`, `policy_decision`

**`POST /cap/export`** — Export approved alerts as CAP 1.2 XML
- Input: `{candidate_ids}` or all APPROVED
- Generates valid CAP 1.2 XML with identifier, sender, sent, status, msgType, scope
- Info blocks: language, category, event (Flood Alert), responseType, priority, urgency, severity, certainty, headline, description, instruction, area (areaDesc, polygon, altitude, ceiling)
- Returns: `{count, cap_xml}`

#### `policy_engine.js` (218 lines) — Node Alert Policy Engine

**Class:** `AlertPolicyEngine`

**Parameters:**
- `costOfAction = 1.0`, `lossIfMissed = 10.0` → `costLossThreshold = 0.10`
- `maxAlertsPerCellPerDay = 3`
- `escalateConfirmations = 2`, `deescalateConfirmations = 3`

**Method:** `evaluate(candidate, nowUtc)`
1. Gets/creates cell state (currentTier, confirmations, alertsToday, lastAlertDate, activeAlert)
2. Effective tier after trust gating: untrusted WARNING/IMMINENT → downgraded to WATCH/WATCH
3. Cost-loss: `expectedUtility = prob × loss − cost`; alert only if > 0 (P > 0.10)
4. Hysteresis: escalation needs 2 confirmations, de-escalation needs 3
5. Fatigue budget: max 3 alerts/cell/day
6. Returns decision: `{candidateId, cellId, action, reason, riskTier, trustFlag, costLossRatio, predictedProbability, previousState, newState, timestampUtc, fatigueBudgetRemaining}`
7. Actions: NO_CHANGE, ESCALATE, DEESCALATE, ALL_CLEAR

#### `device_registry.js` (154 lines) — Minimal Device Registry

**Class:** `DeviceRegistry`
- MongoDB collection: `global_mesh_devices` (indexes: deviceId unique, cellId)
- In-memory fallback if MongoDB unavailable
- Stores: `deviceId, cellId, lat, lon, consent, platform, token, registeredAt, updatedAt`
- **No precise location beyond cell centroid** (privacy by design)
- Methods: `registerDevice()`, `getDevicesForCell()` (consent only), `getAllDevices()`, `deactivateDevice()`, `countConsentedDevices()`

---

### 4.3 Persistence & Store Modules

#### `prediction_store.js` (626 lines) — MongoDB Prediction Store

**Collections:**
- `predictions` — unified prediction snapshots
- `global_predictions` — global scan records
- `forecast_verifications` — closed-loop verification (HIT/MISS/FA/CN/DU)
- `action_plans` — emergency response action plans
- `impact_assessments` — multi-scale impact assessments

**Key Functions:**
- `savePrediction(doc)` — persists unified prediction with timestamp
- `getHistory(query)` — retrieval by coordinates + limit
- `getPredictionById(id)` — ObjectId lookup
- `getStats()` — total predictions, high-risk count, recent samples
- `saveGlobalPrediction(doc)` — persists global scan + registers verification checkpoints
- `getLatestGlobalPrediction()` / `getGlobalHistory(limit)` — global scan queries
- `auditAndEvaluateVerifications(observedMeshMap)` — closed-loop audit: HIT/MISS/FA/CN/DU based on genuine satellite/gauge observations; DATA_UNAVAILABLE excluded from metrics
- `getVerificationMetrics()` — CSI, Hit Rate, FAR, counts
- `saveActionPlan(doc)` / `getActionPlanHistory(query)` — action plan persistence
- `saveImpactAssessment(doc)` / `getImpactAssessmentHistory(query)` — impact persistence
- **In-memory fallback** for all collections if MongoDB unavailable

#### `history.js` — SQLite Scan History
- `logScan({scan_type, filename/lat/lng, summary})` — logs scans to `scans.db`
- `getHistory()` — returns last 100 scans
- `deleteScan(id)` / `clearHistory()` — deletion operations

#### `management.js` — Agency Operations
- `addInventory()` / `listInventory()` / `removeInventory()` — supply inventory
- `addEvacuee()` / `listEvacuees()` / `removeEvacuee()` — evacuee tracking
- `addPersonnel()` / `listPersonnel()` — personnel management

#### `progressive_scanner.js` — Multi-Scale Progressive Scan
- Stages: 5×5 → 10×10 → 25×25 → 50×50 km
- Non-blocking scan with job ID, progress tracking, provenance logs
- Methods: `startProgressiveScan()`, `getScanStatus()`, `cancelScan()`, `getScanProvenance()`, `getGlobalProvenanceLogs()`

---

### 4.4 Frontend Static Assets (`disaster-dashboard/public/`)

#### `index.html` (1490+ lines) — Main Leaflet GIS Dashboard

**Design System:** Muted slate dark theme (`--bg-0: #080c14`), semantic risk colors (normal=emerald, warning=amber, high=orange, critical=crimson), Inter + JetBrains Mono fonts.

**Layout:** CSS Grid shell — header (46px), sidebar (480px collapsible), map viewport (Leaflet).

**10 Tabs:**
| Tab | Icon | Purpose |
|-----|------|---------|
| Predict | ⚡ | Real-time multi-factor flood prediction |
| Global Mesh | 🌐 | Autonomous global early-warning scan |
| River Gauges | 🌊 | Copernicus GloFAS hydrology telemetry |
| 72h Timeline | ⏱️ | TCN-Attention multi-horizon forecast |
| SAR Radar | 🛰️ | Sentinel-1 SAR inundation mapping |
| Impact Matrix | 📊 | Flood impact assessment (buildings, population, loss) |
| Action Plan | 🎯 | Emergency response optimization |
| Agency Ops | 🏢 | Inventory, evacuees, personnel, shelter dispatch |
| Live Floods | ⚠️ | GDACS live flood events |
| GeoTIFF Upload | 📁 | Satellite imagery upload for SAR segmentation |
| History | 📜 | Scan history |

**Key UI Components:**
- Command bar header with engine status dot (online/offline), GDACS basin hotspot selector, sidebar toggle
- Scientific telemetry metric cards with status pills (available/unavailable)
- Formula evaluation box showing risk calculation
- Predict gauge with probability value + severity badge
- Leaflet map with basemap toggle, rainfall toggle, legend, cursor coordinates
- Source/destination/shelter/action pins with pulse animations
- Route cards with status pills (clear/partial/flooded)
- Provenance & factors modals with tables
- Toasts (success/error/warn)
- File drop area for GeoTIFF upload

**JavaScript Modules:**
- `predict-ui.js` — Prediction tab logic (coordinate input, target selection, prediction display)
- `global-prediction-ui.js` — Global Mesh tab logic (global scan, region cards, trajectory, GeoJSON)
- `management-ui.js` — Agency ops tab logic (inventory, evacuees, personnel, shelter dispatch)

#### `alert_review.html` (231 lines) — Alert Review + CAP 1.2 Export

**Table Columns:** Cell ID, Region, Lat/Lng, P(Flood), Tier, Trust, Severity, Primary Driver, Exposed Pop., Decision, Actions

**Features:**
- Fetches `GET /api/alert/candidates` on load + every 30s
- Refresh Candidates button
- Risk tier badges (WATCH/WARNING/IMMINENT/NORMAL) with semantic colors
- Trust badges (Trusted/Untrusted)
- Decision column showing model_status (pending review)
- Approve button disabled for untrusted candidates (trust gate)
- Reject / Cancel buttons always available
- CAP 1.2 Export section (initially hidden): Download CAP XML, Copy CAP XML
- Toast notifications for operations

**Fixed issues (lat/lon null handling, P(Flood) fallback):**
- Lat/Lng: shows "—" when null instead of "0.0000, 0.0000"
- P(Flood): falls back to `peak_risk_score/100` when `predicted_flood_probability` is 0/null

---

## 5. DEEP LEARNING MODELS — IN-DEPTH ARCHITECTURE

(See Section 3.2 above for model inventory. Below are architectural details.)

### Model M1 — EfficientNet-B3 U-Net (SAR Segmentation)

**File locations:** `main.py` (model loading, `build_model_input`, `run_inference`), `unet_b3.pth`

```python
model = smp.Unet(
    encoder_name="efficientnet-b3",
    encoder_weights=None,
    in_channels=4,   # VV, VH, VV/VH ratio, water prior
    classes=1,
)
model.load_state_dict(torch.load("unet_b3.pth", map_location=device))
model.eval()
```

**4-channel input construction:**
```
Channel 1: VV (normalised dB, [-30,0] → [0,1])
Channel 2: VH (normalised dB, [-30,0] → [0,1])
Channel 3: ratio = VV / VH (min-max normalized)
Channel 4: JRC Global Surface Water occurrence / specular water prior [0,1]
```

**Inference pipeline:**
1. Pad input to multiples of 32 (U-Net requirement)
2. Run model → sigmoid → threshold at 0.65
3. `rasterio.features.shapes()` → vectorize flood mask to polygons
4. `filter_polygons()` → remove >150 km² false positives → remove ocean → subtract OSM permanent water
5. `build_features()` → compute area (EPSG:6933), danger level, GeoJSON

**Training basis:** Sen1Floods11 dual-polarization SAR (VV+VH), documented in `colab_model_v4.ipynb` and `colabmodel_code.ipynb` in `old_models_and_tif/`

### Model M2 — MultiFactorFloodPredictor (Cross-Attention + Residual)

**File:** `disaster_prediction_model.py`

```python
class CrossDomainAttention(nn.Module):
    """Multi-Head Self-Attention over fused domain features"""
    def __init__(self, embed_dim=128, num_heads=4):
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

class ResidualBlock(nn.Module):
    """Dense residual block with LayerNorm and Dropout"""
    def __init__(self, dim=128, dropout_rate=0.25):
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_rate)

class MultiFactorFloodPredictor(nn.Module):
    """Cross-Attention + Residual Multi-Task Deep Predictor"""
```

**Pipeline:** 24 live factors → physical feature engineering → domain group encoders → CrossDomainAttention fusion → ResidualBlock stack → multi-task heads (probability, severity, depth) + MC dropout

**Outputs:** Flood probability, severity class (Normal/Advisory/Moderate/High/Critical), inundation depth, epistemic uncertainty

**Weights:** `flood_prediction_model.pth`  
**Reported ROC-AUC:** 1.0 (in `dl_model_credentials`)

### Model M3 — StreamFlow LSTM

**File:** `train_streamflow_lstm.py`, weights `streamflow_forecast_lstm.pth`

**Architecture:** State-handoff encoder-decoder LSTM  
**Training data:** Caravan/CAMELS catchment specification  
**Task:** 7-day river streamflow (discharge) forecasting  
**Performance:** Mean NSE: 0.983  
**Manifest:** `streamflow_model_manifest.json`

### Model M4 — TCN-Attention Multi-Horizon Forecaster

**File:** `tcn_attention_forecaster.py`, weights `tcn_flood_forecaster.pth`

```python
class ChausalConv1d(nn.Module):
    """Causal 1D Convolution: trims right-side padding to prevent future data leakage"""
```

**Architecture:**
- Input: `[Batch, Sequence_Length=72, Features=10]`
- Features: [Precip, Temp, RH, Pressure, Wind, CAPE, Soil_Surface, Soil_Root, Discharge, Flow_Ratio]
- Dilated Causal Convolutional Residual Blocks (dilations: 1, 2, 4, 8, 16)
- Multi-Head Temporal Self-Attention across critical precipitation concentration hours
- Receptive field: 125 hours

**Outputs:**
- Multi-horizon flood probability vector: [P₀, P₆, P₁₂, P₂₄, P₄₈, P₇₂]
- Multi-horizon severity distribution: [6 horizons × 5 classes]
- Peak flood severity score & onset horizon prediction

**Metrics:** `tcn_model_metrics.json`

### Model M5 — Deep Ensemble (M=5) Epistemic Uncertainty

**File:** `deep_ensemble_engine.py`

**Method:** Lakshminarayanan et al. (NeurIPS 2017) + Gal & Ghahramani (ICML 2016)

```python
class DeepEnsembleFloodPredictor:
    def __init__(self, ensemble_size=5, base_weights_path=None):
        # M=5 diversified MultiFactorFloodPredictor instances
        # Varied parameter initialization seeds + stochastic dropout inference
```

**Outputs:**
1. Ensemble mean prediction across M=5 models
2. Epistemic uncertainty (model variance / standard deviation σ)
3. 95% Confidence Intervals [Mean − 1.96σ, Mean + 1.96σ]
4. Out-of-Distribution (OOD) & hydrological discordance detection
5. Calibrated reliability metric

### Model M6 — Global Mesh GRU (Untrained/Undepployed)

**File:** `global_mesh/flood_risk_model.py`, weights `global_mesh/models/flood_risk_gru.pth`

**Architecture:**
- **Encoder:** Single-layer GRU (bidirectional) ingests 72-hour antecedent sequence (9 dynamic features)
- **Static branch:** Catchment attributes (6 static features) projected via 2-layer MLP
- **Head:** Fused representation → sigmoid probability P(flood=1)
- **Uncertainty:** MC-dropout at inference (25 passes) → epistemic std + 95% CI

**Dynamic features:** precip_24h, precip_72h, API-14, soil moisture surface, soil moisture root, discharge ratio, temperature, CAPE, snow depth  
**Static features:** elevation, slope, TWI, land cover class, coastal flag, GDACS distance

**Validation gating:**
- `is_trained()` — weights loaded
- `is_validated()` — demonstrated skill on held-out data
- Model probability NOT used in risk scoring unless validated
- Deployment threshold: AUC > climatology + 0.05

**Status:** NOT TRAINED / NOT DEPLOYED — only 1 positive flood label of 32 samples (minimum 5 required)

### Model M7 — Physics-Based Transparent Risk Score

**Files:** `flood_risk_engine.py`, `global_mesh/risk_scorer.py`

**Formula:** Risk = 100 × Hazard × (0.50 × Exposure + 0.50 × Vulnerability)

**Four outputs:**
| Output | Type | Range | Basis |
|--------|------|-------|-------|
| Risk Index | Uncalibrated composite | 0–100 | Normalized physical factors |
| Flood Probability | Calibrated P(flood=1) | 0–1 | Only when GRU validated |
| Physical Hazard | Physically-based estimate | 0–1 | Discharge + soil + rainfall |
| Flood Impact | Exposed population / area | count / km² | Not a probability |

**Factor shares (equal, re-normalized over available):**
- Rainfall: 0.20 | Discharge: 0.20 | Soil Saturation: 0.20 | Terrain: 0.20 | GDACS Alert: 0.20

---

## 6. DATA PIPELINE & 24 FLOOD FACTORS

### Data Flow Diagram

```
External Live Data Sources
    │
    ├─ ECMWF IFS via Open-Meteo      (precip, temp, soil moisture, CAPE, snow)
    ├─ Copernicus GloFAS 4.0         (river discharge + climatology)
    ├─ Copernicus GLO-30 DEM         (elevation, slope, TWI)
    ├─ UN/EC GDACS                   (live satellite flood alerts)
    ├─ Sentinel-1 SAR (STAC)         (VV+VH backscatter)
    ├─ NASA GPM / MERRA-2            (satellite precipitation)
    ├─ USGS NWIS / CWC India         (river gauges)
    ├─ ISRIC SoilGrids 250m          (soil texture)
    ├─ WorldPop                      (population density)
    ├─ OpenStreetMap                 (buildings, roads, infrastructure, water)
    ├─ JRC Global Surface Water      (permanent water prior)
    └─ Dartmouth Flood Observatory   (4,029 historical labels)
    │
    ▼
Real Data Collector (24 factors) → Freshness Validation
    │
    ▼
Feature Engineering → Deep Learning Inference (M1–M5) + Physics Score (M7)
    │
    ▼
Risk Classification → Spatial Hazard → Impact Assessment → Response Optimization
    │
    ▼
Global Mesh Scan → Alert Candidates → Operator Review → CAP 1.2 Export
    │
    ▼
Closed-Loop Verification (HIT/MISS/FA/CN/DATA_UNAVAILABLE)
```

### 24 Live Flood Factors (collected by `real_data_collector.py`)

| # | Factor | Source | Type | Provenance Tag |
|---|--------|--------|------|----------------|
| 1 | Precipitation 24h | ECMWF IFS via Open-Meteo | Observation | MODEL_ANALYSIS |
| 2 | Precipitation 72h | ECMWF IFS via Open-Meteo | Forecast | MODEL_FORECAST |
| 3 | Antecedent Precipitation Index 14d | ECMWF IFS | Observation | MODEL_ANALYSIS |
| 4 | Soil Moisture Surface | ECMWF IFS | Observation | MODEL_ANALYSIS |
| 5 | Soil Moisture Root | ECMWF IFS | Observation | MODEL_ANALYSIS |
| 6 | River Discharge | Copernicus GloFAS 4.0 | Modelled Reanalysis | MODELLED_REANALYSIS |
| 7 | Discharge Ratio | GloFAS vs climatology | Derived | MODELLED_REANALYSIS |
| 8 | Elevation | Copernicus GLO-30 DEM | Static GIS | OBSERVED_GIS |
| 9 | Slope | Copernicus GLO-30 DEM | Static GIS | OBSERVED_GIS |
| 10 | Topographic Wetness Index | GLO-30 DEM | Derived | OBSERVED_GIS |
| 11 | Temperature | ECMWF IFS | Observation | MODEL_ANALYSIS |
| 12 | CAPE | ECMWF IFS | Observation | MODEL_ANALYSIS |
| 13 | Snow Depth | ECMWF IFS | Observation | MODEL_ANALYSIS |
| 14 | GDACS Alert Level | UN/EC GDACS | External Alert | EXTERNAL_ALERT_EVIDENCE |
| 15 | GDACS Alert Distance | UN/EC GDACS | External Alert | EXTERNAL_ALERT_EVIDENCE |
| 16 | Land Cover Class | Static reference | Static | STATIC_REFERENCE |
| 17 | Coastal Flag | Static reference | Static | STATIC_REFERENCE |
| 18 | NASA GPM Precipitation | NASA GPM / MERRA-2 | Satellite Retrieval | SATELLITE_RETRIEVAL |
| 19 | USGS/CWC River Gauge | USGS NWIS / CWC India | Observed Sensor | OBSERVED_SENSOR |
| 20 | Soil Texture (clay/sand/silt) | ISRIC SoilGrids 250m | Static Reference | STATIC_REFERENCE |
| 21 | Infiltration Rate | ISRIC SoilGrids | Derived | STATIC_REFERENCE |
| 22 | Population Density | WorldPop | Static Reference | STATIC_REFERENCE |
| 23 | Catchment Population | WorldPop | Static Reference | STATIC_REFERENCE |
| 24 | Permanent Water Occurrence | JRC Global Surface Water | Static Reference | STATIC_REFERENCE |

**Freshness hierarchy:** Every field carries `FRESH` / `ACCEPTABLE` / `STALE` / `EXPIRED` / `UNKNOWN`. SAR freshness: ≤48h FRESH, ≤168h STALE, >168h EXPIRED.

**Missing data policy:** Unavailable factors excluded and flagged in `missing_factors` list; cell never interpreted as "safe."

---

## 7. RISK SCORING & ALERTING LOGIC

### Risk Classification Bands

| Band | Score | Color | Hex | Recommended Action |
|------|-------|-------|-----|-------------------|
| CRITICAL | ≥ 75 | Crimson | `#ef4444` | Activate inter-agency emergency operations & issue downstream evacuation warnings |
| HIGH | 50–74 | Safety Orange | `#f97316` | Pre-position flood-fighting assets & inspect primary levees |
| MODERATE | 25–49 | Amber | `#eab308` | Monitor river telemetry & clear critical drainage culverts |
| LOW | < 25 | Emerald | `#10b981` | Routine satellite & hydrometeorological surveillance |

### Flood Typology

| Code | Label | Description |
|------|-------|-------------|
| `FLUVIAL_RIVERINE` | Riverine Flood | River discharge exceeds channel capacity |
| `PLUVIAL_CONVECTIVE` | Pluvial Flash | High-intensity cloudburst exceeds infiltration |
| `NIVAL_SNOWMELT` | Nival / Snowmelt | Snowmelt-driven discharge surge |
| `COMPOUND_COASTAL` | Compound Coastal | Storm surge + precipitation + high tide |
| `BASELINE_NORMAL` | Normal Hydrological Regime | Within seasonal carrying capacity |

### Alerting Layer (Phase 5 + 6 + 7)

#### Alert Candidate Schema (MongoDB: `global_mesh_alert_candidates`)

```
candidate_id, cell_id, lat, lon, region_name, country, basin,
risk_tier, trust_flag, model_status, data_available,
predicted_flood_probability, peak_risk_score, severity, primary_driver,
flood_typology {code, label, description}, data_provenance,
forecast_rain_72h_mm, river_discharge_m3s, discharge_ratio,
soil_moisture, elevation_m, estimated_exposed_pop,
epistemic_std, confidence_interval, scan_id, created_at
```

#### Policy Engine (Node.js: `alerts/policy_engine.js`)

**Cost-loss expected utility:**
```
Expected Utility = P(flood) × LossIfMissed − CostOfAction
Alert if Expected Utility > 0  →  P(flood) > Cost/Loss = 1.0/10.0 = 0.10
```

**Trust gate:** Only trained+validated model output trusted. Untrusted WARNING/IMMINENT → downgraded to WATCH.

**Hysteresis:**
- Escalation: 2 consecutive confirmations required
- De-escalation: 3 consecutive confirmations required

**Deduplication:** One active alert per cell at a time.

**Fatigue budget:** Maximum 3 alerts per cell per day.

**Actions:** NO_CHANGE / ESCALATE / DEESCALATE / ALL_CLEAR

#### Device Registry (Node.js: `alerts/device_registry.js`)

- Stores: `deviceId, cellId, lat, lon, consent, platform, token`
- **Privacy:** No precise location beyond cell centroid; consent-required
- MongoDB-backed with in-memory fallback

#### CAP 1.2 Export (Node.js: `alerts/routes.js`)

Valid CAP 1.2 XML structure:
```xml
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>URN:OASIS:EMERGENCY:CAP:GLOBAL_MESH:...</identifier>
  <sender>global-mesh@floodwatch.ai</sender>
  <sent>...</sent>
  <status>Actual</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
  <info>
    <language>en</language>
    <category>Met</category>
    <event>Flood Alert</event>
    <responseType>Evacuate|Prepare|Monitor</responseType>
    <priority>Immediate|Expected|Minor</priority>
    <urgency>Immediate|Expected|Minor</urgency>
    <severity>...</severity>
    <certainty>Observed|Likely</certainty>
    <headline>Flood Alert — {region}</headline>
    <description>...</description>
    <instruction>...</instruction>
    <area>
      <areaDesc>{country}, {basin}</areaDesc>
      <polygon>{lat} {lon}</polygon>
      <altitude>0</altitude>
      <ceiling>0</ceiling>
    </area>
  </info>
</alert>
```

All exported CAP files validated against CAP 1.2 schema structure.

---

## 8. DATABASE SCHEMA & PERSISTENCE

### MongoDB Collections (`floodwatch` database)

| Collection | Purpose | Key Fields |
|------------|---------|------------|
| `predictions` | Unified prediction snapshots | location, timestamp, prediction, explainability, factors, spatial_impact, satellite_evidence, model_metadata |
| `global_predictions` | Global scan records | scan_id, timestamp, summary, top_emerging_regions, global_forecast_trajectory, geojson, metadata |
| `forecast_verifications` | Closed-loop verification | scan_id, region_id, forecast_timestamp, target_horizon_hours, predicted, observed, verification_status (HIT/MISS/FA/CN/DU) |
| `action_plans` | Emergency response plans | coordinates, budget, user_profile, actions, priority_zones, timestamp |
| `impact_assessments` | Multi-scale impact | coordinates, depth_m, stages, cumulative_total, priority_zones, data_provenance |
| `global_mesh_alert_candidates` | Alert candidates | candidate_id, cell_id, risk_tier, trust_flag, probability, severity, data_provenance, scan_id, created_at |
| `global_mesh_scans` | Global Mesh scan records | scan_id, timestamp, summary, top_emerging_regions, geojson, metadata |
| `global_mesh_scan_history` | Per-cell observation time-series | scan_id, cell_id, observation, timestamp |
| `global_mesh_training_data` | Labelled training samples | inputs, verified_flood_label, cell_id, date |
| `global_mesh_model_versions` | Model versions & metrics | version, metrics, deployed, deploy_reason, manifest |
| `global_mesh_verifications` | Global Mesh forecast verification | scan_id, region_id, predicted, observed, status |
| `global_mesh_devices` | Device registry (alerts) | deviceId, cellId, consent, platform, token |

### SQLite Databases

| Database | Purpose | Tables |
|----------|---------|--------|
| `scans.db` | Scan history | scans (id, scan_type, filename/lat/lng, summary, timestamp) |
| `management.db` | Agency operations | inventory, evacuees, personnel |

---

## 9. COMPLETE API REFERENCE

### Node.js Dashboard (Port 3000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /api/health` | GET | AI engine reachability |
| `POST /api/analyze-satellite` | POST | GeoTIFF upload → AI `/predict` |
| `POST /api/live-analyze` | POST | Live scan bbox → AI `/predict-live` |
| `POST /api/unified-prediction` | POST | Unified flood prediction + MongoDB persist |
| `GET /api/prediction-history` | GET | Prediction history |
| `GET /api/prediction-history/:id` | GET | Prediction by ID |
| `GET /api/prediction-stats` | GET | Prediction statistics |
| `POST /api/global-prediction` \| `POST /api/global-scan` | POST | Autonomous Global Mesh scan |
| `GET /api/global-prediction/latest` | GET | Latest global prediction |
| `GET /api/global-prediction/history` | GET | Global prediction history |
| `GET /api/global-flood-hotspots` | GET | GDACS live flood hotspots |
| `GET /api/global-prediction/verification-metrics` | GET | CSI, Hit Rate, FAR |
| `POST /api/global-prediction/verify` | POST | Trigger verification audit |
| `POST /api/predict-disaster` | POST | Multi-factor flood prediction |
| `GET /api/live-factors` | GET | 24 live environmental factors |
| `GET /api/live-floods` | GET | GDACS live floods (GeoJSON) |
| `GET /api/live-floods/shape` | GET | Flood event polygon |
| `GET/POST /api/hydrology-station` | — | Copernicus GloFAS river telemetry |
| `POST /api/drone-recon` | POST | Drone recon (alias for hydrology) |
| `POST /api/assess-impact` \| `/api/impact-assessment` | POST | Flood impact assessment |
| `POST /api/forecast-flood` \| `/api/flood-forecast` | POST | Flood forecasting |
| `POST /api/optimize-response` \| `/api/action-plan` | POST | Response optimization |
| `POST /api/dispatch-shelter` | POST | Shelter dispatch |
| `GET/POST/DELETE /api/inventory` | — | Agency inventory |
| `GET/POST/DELETE /api/evacuees` | — | Evacuee management |
| `GET/POST /api/personnel` | — | Personnel management |
| `GET/DELETE /api/history` / `/api/history/:id` | — | SQLite scan history |
| `POST /api/scan/start` | POST | Start progressive scan |
| `GET /api/scan/status/:scanId` | GET | Scan status |
| `POST /api/scan/cancel/:scanId` | POST | Cancel scan |
| `GET /api/scan/provenance/:scanId` | GET | Scan provenance logs |
| `GET /api/scan/provenance-global` | GET | Global provenance logs |
| `POST /api/sar/multiscale` | POST | Multi-scale SAR (5/10/15 km) |
| `POST /api/impact/multiscale` | POST | Multi-scale impact (5/10/25/50 km) |
| `POST /api/action/multiscale` \| `/api/action-plan/generate` | POST | Multi-scale action plan |
| `GET /api/action-plan/history` | GET | Action plan history |
| `GET /api/impact/history` | GET | Impact assessment history |
| `POST /api/progressive/stage` | POST | Progressive stage execution |
| `GET /api/alert/candidates` | GET | List alert candidates |
| `POST /api/alert/decide` | POST | Approve/reject/cancel |
| `POST /api/alert/cap/export` | POST | CAP 1.2 XML export |

### Python AI Engine (Port 8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /predict` | POST | SAR U-Net flood segmentation (GeoTIFF) |
| `POST /predict-live` | POST | Live Sentinel-1 SAR inference |
| `POST /predict-disaster` | POST | Multi-factor deep learning prediction |
| `POST /unified-prediction` \| `/api/unified-prediction` | POST | Unified prediction + management |
| `POST /global-prediction` \| `/api/global-prediction` | POST | Autonomous Global Mesh scan |
| `GET /global-prediction/latest` | GET | Latest global prediction |
| `GET /live-factors` | GET | 24 live factors |
| `GET /global-flood-hotspots` | GET | GDACS hotspots |
| `POST /assess-impact` | POST | Impact assessment |
| `POST /forecast-flood` | POST | Flood forecast |
| `POST /optimize-response` | POST | Response optimization |
| `POST /sar-multiscale` \| `/api/sar-multiscale` | POST | Multi-scale SAR |
| `POST /impact-multiscale` \| `/api/impact-multiscale` | POST | Multi-scale impact |
| `GET /docs` | GET | OpenAPI/Swagger UI |

---

## 10. CONFIGURATION, SETUP & DEPLOYMENT

### Prerequisites
- Python 3.9+
- Node.js 18+
- MongoDB (optional; in-memory fallback)
- Git

### Python AI Engine Setup
```powershell
cd disaster-ai-api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### Node.js Dashboard Setup
```powershell
cd disaster-dashboard
npm install
node server.js
```

### Environment Variables
| Variable | Default | Purpose |
|----------|---------|---------|
| `PORT` | 3000 | Dashboard port |
| `API_URL` | http://127.0.0.1:8000 | AI engine URL |
| `MONGODB_URI` | mongodb://127.0.0.1:27017 | MongoDB connection |
| `MONGODB_DB` | floodwatch | Database name |
| `ALERT_MODE` | dry_run | Alert safety mode (never change to live) |

---

## 11. SAFETY, COMPLIANCE & LIMITATIONS

### Safety Compliance
| Rule | Implementation |
|------|---------------|
| ALERT_MODE = dry_run | Default, never changed |
| Operator approval mandatory | AlertReview page requires manual approve/reject/cancel |
| Trust gate | Only trained+validated model output trusted (currently 0 trusted) |
| Fatigue budget | Max 3 alerts per cell per day |
| Kill-switch | Policy engine supports de-escalation/all-clear |
| Audit trail | decision_timestamp, decision_reason, policy_decision stored in MongoDB |
| CAP 1.2 export | Valid CAP 1.2 XML with all required fields |
| Minimal data storage | Device token, coarse cell ID, consent flags only — no precise location |
| Controlled templates | CAP message text from controlled templates only |

### Honest Limitations
| # | Limitation | Disclosure |
|---|-----------|------------|
| L1 | GRU model untrained/undeployed | Only 1 positive label of 32 samples |
| L2 | Open-Meteo limited to 14 days history | Cannot pair DFO labels with historical weather |
| L3 | DFO archive covers 1985–2010 only | No recent flood events for validation |
| L4 | GloFAS is modelled reanalysis, not gauge truth | Proxy for ground truth |
| L5 | GDACS centroids are approximate | Not precise inundation footprints |
| L6 | MC-dropout uncertainty not independently validated | Epistemic estimate only |
| L7 | Grid sampling cap (50 cells/run) | Multiple scans needed for full coverage |
| L8 | P4 calibration/trust not done | No trained model to calibrate |
| L9 | Ablation studies not done | Cannot ablate untrained model |
| L10 | Closed-loop scoring (N7) not done | Requires live alert delivery feedback |
| L11 | Sentinel-1 SAR flood labels not integrated | Placeholder only |
| L12 | FCM real delivery not enabled | dry_run only; FCM mocked |
| L13 | CPU-only inference | Slower than GPU; fine for educational use |

---

## 12. APPENDIX: PHASE REPORTS & EVIDENCE

| Phase | Name | File | Status |
|-------|------|------|--------|
| P0 | Audit | `reports/phase0/endpoint_audit.md` | DONE / PASSED |
| P1 | Evidence ledger + ingestion | `reports/phase1/evidence.md` | DONE / PASSED |
| P2 | Label fusion (N1) | `reports/phase2/dataset_report.md`, `evidence.md` | DONE / PASSED |
| P3 | Models | `reports/phase3/metrics_table.md`, `bootstrap_ci.md` | DONE / PASSED (gated off) |
| P4 | Calibration and trust (N4) | `reports/phase4/reliability_diagram.md`, `conformal_coverage.md` | NOT DONE |
| P5 | Forecast service | — | DONE — alert candidates to MongoDB |
| P6 | Node alert service (N6) | `disaster-dashboard/alerts/` | DONE |
| P7 | Operator review + CAP (N8) | `reports/phase7/cap_schema_validation.md` | DONE |
| P8 | Closed loop + final report | `reports/FINAL_REPORT.md` | DONE |

**Final Report:** `reports/FINAL_REPORT.md`

---

**End of Technical Documentation**

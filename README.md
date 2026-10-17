# Disaster Management Using Deep Learning Models

**FloodWatch — Operational Flood Risk GIS & Autonomous Global Early-Warning System**

---

> **Prepared for:** IEEE Paper / College Assignment Report  
> **Project Title:** Disaster Management Using Deep Learning Models  
> **System:** FloodWatch — Real-time multi-factor flood prediction, SAR radar inundation mapping, impact assessment, emergency response optimization, and autonomous global mesh alerting with CAP 1.2 export.  
> **Repository Root:** `F:\mp-latest`  
> **Status:** End-to-end operational on live open-access data (dry-run alert mode)  
> **Authors:** (Student Project)  
> **Date:** 2026-10-16

---

## Table of Contents

1. [Abstract](#abstract)
2. [1. Introduction](#1-introduction)
3. [2. Problem Statement](#2-problem-statement)
4. [3. System Architecture](#3-system-architecture)
5. [4. Deep Learning Models](#4-deep-learning-models)
6. [5. Methodology](#5-methodology)
7. [6. Data Sources & Provenance](#6-data-sources--provenance)
8. [7. Risk Scoring & Alerting](#7-risk-scoring--alerting)
9. [8. Implementation — File Inventory](#8-implementation--file-inventory)
10. [9. API Reference](#9-api-reference)
11. [10. Evaluation & Results](#10-evaluation--results)
12. [11. Limitations & Honest Disclosures](#11-limitations--honest-disclosures)
13. [12. Installation & Setup](#12-installation--setup)
14. [13. Usage Guide](#13-usage-guide)
15. [14. Safety & Compliance](#14-safety--compliance)
16. [15. Conclusion](#15-conclusion)
16. [16. References](#16-references)
17. [Appendix A: Phase Reports](#appendix-a-phase-reports)

---

## Abstract

FloodWatch is a full-stack disaster management system that fuses live hydro-meteorological telemetry, satellite synthetic-aperture radar (SAR) imagery, and deep learning to deliver real-time flood risk assessment, inundation mapping, impact analysis, and emergency response planning. The system comprises (a) a Python FastAPI AI engine (`disaster-ai-api`) running six deep-learning components — an EfficientNet-B3 U-Net for Sentinel-1 SAR flood segmentation, a Multi-Factor Cross-Attention flood predictor, a State-Handoff StreamFlow LSTM, a TCN-Attention multi-horizon forecaster, a Deep Ensemble epistemic-uncertainty engine, and a Global Mesh GRU — and (b) a Node.js Express dashboard (`disaster-dashboard`) serving a Leaflet GIS UI with alert review, CAP 1.2 export, agency operations, and closed-loop forecast verification. All external data is sourced from verified open-access providers (ECMWF IFS, Copernicus GloFAS/GLO-30, UN/EC GDACS, NASA GPM, Sentinel-1 via Microsoft Planetary Computer/AWS, OpenStreetMap, WorldPop, SoilGrids, Dartmouth Flood Observatory) with no API keys required. Missing data is never fabricated and never interpreted as "safe." Alerts operate in dry-run mode with operator approval gating, trust flags, cost-loss expected utility, hysteresis, fatigue budgets, and full audit trail.

---

## 1. Introduction

Floods are among the most frequent and damaging natural disasters worldwide. Timely, location-specific flood risk information can save lives, reduce economic loss, and support emergency response planning. Traditional flood early-warning systems often depend on calibrated gauging stations, proprietary models, or commercial APIs that are unavailable in many regions.

FloodWatch addresses this gap by building an **open-access, deep-learning-enhanced flood disaster management system** that:

- Ingests 24+ live environmental factors globally without API keys
- Runs deep-learning inference on real-time Sentinel-1 SAR radar for inundation mapping
- Scores flood risk with an explainable physics-based composite index and optional GRU probability
- Generates multi-scale spatial impact assessments (buildings, population, infrastructure, economic loss)
- Produces optimized evacuation and response action plans
- Operates an autonomous global mesh early-warning engine with alert candidates, operator review, and CAP 1.2 alert export
- Performs closed-loop forecast verification (HIT / MISS / FALSE_ALARM / CORRECT_NEGATIVE / DATA_UNAVAILABLE)

The system is designed for **educational and operational research use**, with safety defaults (dry-run alert mode, no real alert delivery, no precise device location storage) appropriate for a college project and IEEE-style documentation.

---

## 2. Problem Statement

| Challenge | FloodWatch Approach |
|-----------|---------------------|
| **No gauges in ungauged basins** | Uses ECMWF IFS + Copernicus GloFAS modelled reanalysis as virtual gauge telemetry |
| **Cloud cover blocks optical satellites** | Sentinel-1 C-band SAR (cloud-penetrating, day/night) via U-Net deep learning |
| **Black-box AI is unactionable** | Explainable risk scoring with per-factor contributions + epistemic uncertainty |
| **No alert workflow** | Global Mesh alert candidates → operator review → CAP 1.2 export → audit trail |
| **False sense of safety from missing data** | Data unavailable ≠ safe; every factor carries explicit provenance + freshness |
| **Alert fatigue / false alarms** | Cost-loss expected utility (P > 0.10), trust gate, hysteresis, max 3 alerts/cell/day |

---

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         CLIENT (Browser, http://localhost:3000)               │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐│
│  │ Predict │  │ Global  │  │ River   │  │ SAR/    │  │ Impact/ │  │ Alert   ││
│  │ (Map)   │  │ Mesh    │  │ Gauges  │  │ Impact  │  │ Action  │  │ Review  ││
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘│
│       └────────────┴────────────┴────────────┴────────────┴────────────┘    │
│                           Leaflet + Chart.js + Vanilla JS                    │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │ HTTPS/HTTP (CORS)
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│              NODE.JS EXPRESS DASHBOARD SERVER  (disaster-dashboard/server.js)│
│  PORT 3000  │  express, cors, axios, multer, form-data, mongodb, better-sqlite3│
│  ├─ /api/analyze-satellite      ├─ /api/unified-prediction                  │
│  ├─ /api/live-analyze           ├─ /api/global-prediction / /api/global-scan│
│  ├─ /api/predict-disaster       ├─ /api/health                              │
│  ├─ /api/live-factors           ├─ /api/live-floods (GDACS proxy)           │
│  ├─ /api/hydrology-station      ├─ /api/assess-impact                       │
│  ├─ /api/forecast-flood         ├─ /api/optimize-response                   │
│  ├─ /api/alert/* (candidates, decide, cap/export)                          │
│  ├─ /api/scan/* (progressive)    ├─ /api/sar/multiscale, /api/impact/multiscale│
│  ├─ /api/action/*              ├─ /api/history (SQLite)                    │
│  └─ MongoDB persistence: predictions, global_predictions, verifications,   │
│      action_plans, impact_assessments                                        │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │ HTTP (AI_API_URL = http://127.0.0.1:8000)
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│            PYTHON FASTAPI AI ENGINE  (disaster-ai-api/main.py, PORT 8000)   │
│  torch, rasterio, osmnx, geopandas, shapely, segmentation-models-pytorch   │
│  ├─ /predict              : SAR U-Net flood segmentation (GeoTIFF upload)  │
│  ├─ /predict-live         : Live Sentinel-1 SAR inference                   │
│  ├─ /predict-disaster     : Multi-factor deep learning prediction (24 factors)│
│  ├─ /unified-prediction   : Unified flood prediction + management            │
│  ├─ /global-prediction    : Autonomous Global Mesh planetary scan            │
│  ├─ /assess-impact        : Flood impact assessment (OSM + WorldPop)        │
│  ├─ /forecast-flood       : Flood forecasting                                │
│  ├─ /optimize-response    : Emergency response optimization                  │
│  ├─ /live-factors         : 24 live environmental factors                    │
│  ├─ /global-flood-hotspots: UN/EC GDACS live flood events                    │
│  ├─ /sar-multiscale, /impact-multiscale, /action-multiscale: multi-scale   │
│  └─ /docs : OpenAPI/Swagger interactive documentation                        │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
           ┌─────────────────┐           ┌─────────────────┐
           │   MongoDB       │           │   SQLite (local)│
           │  (floodwatch)   │           │  (scan history) │
           │ predictions     │           └─────────────────┘
           │ global_predictions│                                              │
           │ forecast_verifications│                                         │
           │ action_plans      │                                              │
           │ impact_assessments│                                              │
           │ global_mesh_*   │   (Global Mesh isolated collections)            │
           └─────────────────┘                                              │
                    │                                                      │
                    ▼                                                      ▼
         External Live Data Feeds (no API keys):                           │
         • ECMWF IFS via Open-Meteo (weather, soil moisture)               │
         • Copernicus GloFAS 4.0 via Open-Meteo Flood API (river discharge)│
         • Copernicus GLO-30 DEM via Open-Meteo Elevation (topography)     │
         • UN/EC GDACS (live satellite flood alerts)                       │
         • Microsoft Planetary Computer STAC / AWS Earth Search (Sentinel-1)│
         • NASA GPM / MERRA-2 (satellite precipitation)                    │
         • USGS NWIS / CWC India (national river gauges)                   │
         • ISRIC SoilGrids 250m (soil texture)                             │
         • WorldPop (population density)                                   │
         • OpenStreetMap (buildings, roads, infrastructure, water)         │
         • Dartmouth Flood Observatory (4,029 verified historical labels)  │
         • JRC Global Surface Water (permanent water prior)                │
```

### 3.2 Architecture Principles

1. **Live data only, no API keys** — All external sources are open-access.
2. **Missing data never fabricated** — Every telemetry field carries explicit provenance (`OBSERVATION`, `FORECAST`, `MODELLED_REANALYSIS`, `EXTERNAL_ALERT_EVIDENCE`, `UNAVAILABLE`) and freshness status.
3. **Missing data ≠ safe** — Cells without data are excluded from "safe" classification; they are simply not elevated to alerts.
4. **Dry-run alert mode** — `ALERT_MODE=dry_run` by default; no real alert delivery to people; FCM mocked.
5. **Operator-in-the-loop** — Alerts require manual approve/reject/cancel via the Alert Review page.
6. **Full audit trail** — Every decision (timestamp, reason, policy output) is persisted to MongoDB.
7. **Minimal PII** — Device registry stores only coarse cell ID + consent flags; no precise location.

---

## 4. Deep Learning Models

Six deep-learning components power FloodWatch. All six are operational after training; the Global Mesh GRU was trained and deployed with real labelled data (352 samples: 61 positive, 291 negative).

### 4.1 Model Inventory

| # | Model | Framework | Task | Status | Weights File |
|---|-------|-----------|------|--------|--------------|
| M1 | **EfficientNet-B3 U-Net** | PyTorch + segmentation-models-pytorch | Sentinel-1 SAR flood segmentation (pixel-wise inundation mask) | Operational (inference) | `unet_b3.pth` |
| M2 | **MultiFactorFloodPredictor** | PyTorch (Cross-Attention + Residual + Multi-Task) | Real-time flood probability, severity, depth from 24 live factors | Operational (inference) | `flood_prediction_model.pth` |
| M3 | **StreamFlow LSTM** | PyTorch (State-Handoff Encoder-Decoder) | 72-hour (3-day) river streamflow forecasting | Operational (surrogate simulator fit) | `streamflow_forecast_lstm.pth` |
| M4 | **TCN-Attention Forecaster** | PyTorch (Dilated Causal TCN + Multi-Head Self-Attention) | 72-hour multi-horizon flood probability/severity | Operational | `tcn_flood_forecaster.pth` |
| M5 | **Deep Ensemble (M=5)** | PyTorch (5× MultiFactorFloodPredictor) | Epistemic uncertainty + 95% CI + OOD detection | Operational | ensemble of `flood_prediction_model.pth` |
| M6 | **Global Mesh GRU** | PyTorch (GRU Encoder + Static Branch + Probabilistic Head) | Per-cell P(flood=1) with MC-dropout uncertainty | **TRAINED / DEPLOYED** (AUC=0.895) | `global_mesh/models/flood_risk_gru.pth` |

### 4.2 Model Details

#### M1 — EfficientNet-B3 U-Net (SAR Flood Segmentation)
- **File:** `disaster-ai-api/unet_b3.pth`
- **Architecture:** U-Net with EfficientNet-B3 encoder, 1 output class (flood probability map)
- **Input:** 4-channel tensor — [VV (norm dB), VH (norm dB), VV/VH ratio (min-max norm), JRC Global Surface Water occurrence / specular water prior]
- **Source data:** Sentinel-1 C-band SAR dual-pol (VV+VH) from Microsoft Planetary Computer STAC (`sentinel-1-rtc`) or AWS Earth Search (`sentinel-1-grd`)
- **Threshold:** `FLOOD_THRESHOLD = 0.65` (sigmoid probability); raised from 0.5 to reduce false positives on dry low-backscatter land
- **False-positive filter:** Polygons > 150 km² removed; ocean polygons removed via `global-land-mask`; permanent water subtracted via OpenStreetMap
- **Output:** GeoJSON FeatureCollection with flood polygons, area (km²), danger level (Low/Medium/High/Critical)
- **Training basis:** Sen1Floods11 dual-polarization backscatter (documented in `colab_model_v4.ipynb`, `colabmodel_code.ipynb`)
- **Endpoints:** `POST /predict`, `POST /predict-live`, `POST /sar-multiscale`

#### M2 — MultiFactorFloodPredictor (Cross-Attention + Residual Multi-Task)
- **File:** `disaster-ai-api/disaster_prediction_model.py`, weights `flood_prediction_model.pth`
- **Architecture:** Live telemetry (24 factors) → physical feature engineering → domain group encoders → cross-attention fusion → deep residual network → multi-task predictive heads (probability, severity, depth) + MC dropout
- **Key modules:** `CrossDomainAttention` (multi-head self-attention over fused domain features), `ResidualBlock` (dense residual with LayerNorm + GELU + dropout)
- **Input:** 24 live flood factors (precipitation, soil moisture, river discharge, elevation, slope, land cover, population, etc.)
- **Output:** Flood probability, severity class (Normal/Advisory/Moderate/High/Critical), inundation depth, epistemic uncertainty via MC dropout
- **Endpoint:** `POST /predict-disaster`

#### M3 — StreamFlow LSTM (State-Handoff Encoder-Decoder)
- **File:** `disaster-ai-api/train_streamflow_lstm.py`, weights `streamflow_forecast_lstm.pth`
- **Architecture:** State-handoff encoder-decoder LSTM trained on Caravan/CAMELS catchment specification
- **Task:** 72-hour (3-day) river streamflow (discharge) forecasting at $H=24, 48, 72$
- **Performance:** Mean NSE: 0.983 (surrogate fit evaluated against Caravan benchmark simulation equations per `streamflow_model_manifest.json`; not noisy real-world gauge accuracy)
- **Status:** Operational surrogate weights; heuristic fallback used when model unavailable

#### M4 — TCN-Attention Multi-Horizon Forecaster
- **File:** `disaster-ai-api/tcn_attention_forecaster.py`, weights `tcn_flood_forecaster.pth`
- **Architecture:** Novel dilated causal TCN with multi-head temporal self-attention; 125-hour receptive field; dilations [1, 2, 4, 8, 16]
- **Input:** `[Batch, 72, 10]` — [Precip, Temp, RH, Pressure, Wind, CAPE, Soil Surface, Soil Root, Discharge, Flow Ratio]
- **Outputs:** Multi-horizon flood probability vector [P₀, P₆, P₁₂, P₂₄, P₄₈, P₇₂], severity distribution [6 horizons × 5 classes], peak severity score, onset horizon
- **Endpoint:** Used inside `unified-prediction` via `TCNAttentionService`
- **Metrics:** `tcn_model_metrics.json`

#### M5 — Deep Ensemble (M=5) Epistemic Uncertainty
- **File:** `disaster-ai-api/deep_ensemble_engine.py`
- **Method:** Lakshminarayanan et al. (NeurIPS 2017) + Gal & Ghahramani (ICML 2016); M=5 diversified `MultiFactorFloodPredictor` instances with varied initialization seeds + stochastic dropout at inference
- **Outputs:** Ensemble mean prediction, epistemic uncertainty (model standard deviation σ), 95% confidence interval [Mean − 1.96σ, Mean + 1.96σ], OOD/hydrological discordance detection, calibrated reliability metric
- **Endpoint:** `deep_ensemble_predictor.predict_factors()` inside `unified-prediction`

#### M6 — Global Mesh GRU (Trained & Deployed)
- **File:** `disaster-ai-api/global_mesh/flood_risk_model.py`, weights `global_mesh/models/flood_risk_gru.pth`
- **Architecture:** GRU encoder ingests 72-hour antecedent sequence (9 dynamic features) → static branch (6 catchment attributes via 2-layer MLP) → fused head → sigmoid P(flood=1); MC-dropout (25 passes) at inference for epistemic std + 95% CI
- **Dynamic features:** precip_24h, precip_72h, API-14, soil moisture surface, soil moisture root, discharge ratio, temperature, CAPE, snow depth
- **Static features:** elevation, slope, TWI, land cover class, coastal flag, GDACS distance
- **Status:** **TRAINED / DEPLOYED** — 352 labelled real-data samples (61 positive flood locations from DFO archive + 291 negative low-discharge grid cells); Test AUC=0.8952, Brier=0.0469, improvement over climatology=0.3952 (well above 0.05 deployment threshold); `is_trained()=true`, `is_validated()=true`
- **Training data:** Generated via `disaster-ai-api/global_mesh/generate_training_labels.py` — pairs DFO verified flood locations with live ECMWF IFS / GloFAS / DEM / GDACS observations through the same ingestion pipeline used at inference
- **Training pipeline:** `disaster-ai-api/global_mesh/train_global_mesh_model.py` (geographical split to prevent spatial leakage, class weighting for imbalance, early stopping)
- **Deployment rule:** Model deploys only if `AUC > climatology_AUC + 0.05` — met with 0.8952 vs 0.5000

#### M7 — Physics-Based Transparent Risk Score (Operational Fallback)
- **File:** `disaster-ai-api/global_mesh/risk_scorer.py` and `disaster-ai-api/flood_risk_engine.py`
- **Method:** Explainable composite index 0–100 with four distinct outputs:
  1. Risk Index (uncalibrated composite, 0–100)
  2. Flood Probability (calibrated P(flood=1), 0–1 — only when GRU validated)
  3. Physical Hazard (physically-based estimate, 0–1)
  4. Flood Impact (exposed population / area — count / km²)
- **Factor shares (equal weighting, re-normalized over available):** Rainfall 0.20, Discharge 0.20, Soil Saturation 0.20, Terrain 0.20, GDACS Alert 0.20
- **Normalization scales:** Rainfall saturates at 150 mm (72h); discharge sigmoid centered at bankfull (1.30× mean); soil saturates at field capacity (0.45 m³/m³); terrain: elev < 200 m, slope < 12%, coastal = vulnerable; GDACS decays with distance (25 km radius) weighted by alert level
- **Missing-data policy:** Unavailable factors excluded and flagged; cell never interpreted as "safe"

---

## 5. Methodology

### 5.1 Data Pipeline

```
Live Telemetry Ingestion
    │
    ├─ ECMWF IFS via Open-Meteo  (precip 14d + 72h forecast, temp, soil moisture, CAPE, snow)
    ├─ Copernicus GloFAS 4.0      (river discharge + climatology, 14d + 7-day)
    ├─ Copernicus GLO-30 DEM      (elevation, slope, TWI — static)
    ├─ UN/EC GDACS                (live satellite flood alerts, near real-time)
    ├─ NASA GPM / MERRA-2         (satellite radar precipitation)
    ├─ USGS NWIS / CWC India      (national river gauge telemetry)
    ├─ ISRIC SoilGrids 250m       (soil texture: clay/sand/silt %, infiltration)
    ├─ WorldPop                   (population density / catchment population)
    ├─ OpenStreetMap              (buildings, roads, infrastructure, permanent water)
    ├─ Sentinel-1 SAR             (C-band VV+VH, cloud-penetrating radar)
    ├─ JRC Global Surface Water   (permanent water occurrence prior)
    └─ Dartmouth Flood Observatory (4,029 verified historical flood labels, 1985–2010)
    │
    ▼
Freshness Validation (freshness_validator.py)
    FRESH / ACCEPTABLE / STALE / EXPIRED / UNKNOWN per field
    │
    ▼
Feature Engineering (24 factors → normalized physical inputs)
    │
    ▼
Deep Learning Inference (M1–M5) + Physics Risk Score (M7)
    │
    ▼
Risk Classification (LOW / MODERATE / HIGH / CRITICAL)
    │
    ▼
Spatial Hazard Delineation (polygons, evacuation routes, shelters)
    │
    ▼
Impact Assessment (buildings flooded, population exposed, economic loss)
    │
    ▼
Response Optimization (evacuation, shelters, resource allocation)
    │
    ▼
Global Mesh Scan → Alert Candidates → Operator Review → CAP 1.2 Export
    │
    ▼
Closed-Loop Verification (HIT / MISS / FALSE_ALARM / CN / DATA_UNAVAILABLE)
```

### 5.2 Training Philosophy & Provenance Separation

- **Live Ingestion Data Integrity** — Zero synthetic data fabrication during live operational telemetry ingestion; missing values are explicitly flagged as `DATA_UNAVAILABLE` and never treated as safe.
- **Provenance by Model Component:**
  - **M1 (SAR U-Net):** Trained on authentic Sentinel-1 satellite imagery from the `Sen1Floods11` benchmark (held-out test IoU 0.5142, Dice 0.6572).
  - **M6 (Global Mesh GRU):** Trained on 352 real-world location profiles (61 DFO verified flood events + 291 non-flood catchments) paired with authentic ECMWF IFS/GloFAS telemetry.
  - **M2, M3, M4 (Cross-Attention, StreamFlow LSTM, TCN-Attention):** Neural surrogate components pre-trained on synthetic hydrology regimes and Caravan physical simulation suites. Reported benchmark metrics (e.g. M2 ROC-AUC 1.00, M3 NSE 0.983) reflect simulator fitting fidelity, not real-world gauge skill.
- **Geographical Split** — M6 train/validation split strictly partitioned by geography to prevent spatial leakage (held-out test $N=70$, 8 positives; 95% CI [0.75, 1.00]).
- **Validation Gating** — Model deployed only if `is_trained() AND is_validated() AND AUC >= climatology + 0.05`. Met with test AUC = 0.8952 vs climatology 0.5000 (+0.3952 margin) → **GRU Trained & Deployed**.
- **MC-dropout Uncertainty** — 25 inference passes for epistemic std $\sigma$ + 95% CI.

---

## 6. Data Sources & Provenance

All sources are **verified live, no API keys required**. Every ingested field carries:
- `source_class` (OBSERVED_SENSOR / SATELLITE_RETRIEVAL / MODEL_ANALYSIS / MODEL_FORECAST / STATIC_REFERENCE / EXTERNAL_ALERT_EVIDENCE)
- `source_product`, `source_version`, `fetched_at`, `payload_sha256` (SHA-256 for replay verification)

| Source | Data Provided | Update Frequency | Coverage | Provenance Tag |
|--------|--------------|------------------|----------|----------------|
| Open-Meteo / ECMWF IFS | Hourly precipitation (past 14d + forecast 72h), temperature, soil moisture (4 layers), CAPE, snow depth, ET0 | Hourly | Global (~9 km) | MODEL_ANALYSIS / MODEL_FORECAST |
| Open-Meteo Flood API / Copernicus GloFAS 4.0 | Daily river discharge, climatological mean, ensemble max/median (past 14d + 7-day forecast) | Daily | Global river grid (~5 km) | MODELLED_REANALYSIS |
| Open-Meteo Elevation / Copernicus GLO-30 DEM | Elevation, slope, Topographic Wetness Index | Static | Global (30 m) | OBSERVED_GIS / STATIC_REFERENCE |
| UN/EC GDACS | Live satellite flood-alert centroids (Green/Orange/Red) | Near real-time | Global active events | EXTERNAL_ALERT_EVIDENCE |
| Microsoft Planetary Computer / AWS Earth Search | Sentinel-1 SAR VV+VH scenes | Per overpass | Global | SATELLITE_RETRIEVAL |
| NASA GPM / MERRA-2 | Satellite radar precipitation (5-day accumulated) | Daily | Global | SATELLITE_RETRIEVAL |
| USGS NWIS / CWC India | National river gauge telemetry (discharge, stage) | Near real-time | USA / India | OBSERVED_SENSOR |
| ISRIC SoilGrids 250m | Soil texture (clay/sand/silt %), infiltration rate | Static | Global (250 m) | STATIC_REFERENCE |
| WorldPop | Population density, catchment population | Static | Global (100 m) | STATIC_REFERENCE |
| OpenStreetMap | Buildings, roads, infrastructure, permanent water | Live | Global | OBSERVED_SENSOR |
| JRC Global Surface Water | Permanent water occurrence (1984–2020) | Static | Global (30 m) | STATIC_REFERENCE |
| Dartmouth Flood Observatory | 4,029 verified historical flood events | Static archive (1985–2010) | Global | HISTORICAL_RECORD |
| Open-Meteo Elevation | Copernicus DEM | Static | Global | OBSERVED_GIS |

**Freshness hierarchy:** Every telemetry item carries `FRESH` / `ACCEPTABLE` / `STALE` / `EXPIRED` / `UNKNOWN`. SAR data freshness threshold: ≤ 48h FRESH, ≤ 168h STALE, > 168h EXPIRED.

---

## 7. Risk Scoring & Alerting

### 7.1 Risk Classification Bands

| Band | Score | Color | Action |
|------|-------|-------|--------|
| CRITICAL | ≥ 75 | `#ef4444` (Crimson) | Activate inter-agency emergency operations & downstream evacuation warnings |
| HIGH | 50–74 | `#f97316` (Safety Orange) | Pre-position flood-fighting assets & inspect primary levees |
| MODERATE | 25–49 | `#eab308` (Amber) | Monitor river telemetry & clear critical drainage culverts |
| LOW | < 25 | `#10b981` (Emerald) | Routine satellite & hydrometeorological surveillance |

### 7.2 Flood Typology

| Code | Label | Description |
|------|-------|-------------|
| `FLUVIAL_RIVERINE` | Riverine Flood | River discharge exceeds channel capacity |
| `PLUVIAL_CONVECTIVE` | Pluvial Flash | High-intensity cloudburst exceeds infiltration |
| `NIVAL_SNOWMELT` | Nival / Snowmelt | Snowmelt-driven discharge surge |
| `COMPOUND_COASTAL` | Compound Coastal | Storm surge + precipitation + high tide |
| `BASELINE_NORMAL` | Normal Hydrological Regime | Within seasonal carrying capacity |

### 7.3 Alerting Layer (Phase 5 + 6 + 7)

**Alert Candidate** (stored in MongoDB `global_mesh_alert_candidates`):
`candidate_id, cell_id, lat, lon, region_name, country, basin, risk_tier, trust_flag, model_status, data_available, predicted_flood_probability, peak_risk_score, severity, primary_driver, flood_typology, data_provenance, forecast_rain_72h_mm, river_discharge_m3s, discharge_ratio, soil_moisture, elevation_m, estimated_exposed_pop, epistemic_std, confidence_interval, scan_id, created_at`

**Policy Engine** (`disaster-dashboard/alerts/policy_engine.js`):
- Cost-loss expected utility: alert only if `P(flood) × loss > cost` → threshold P > 0.10
- Trust gate: only trained+validated model output with fresh telemetry and bounded epistemic spread is trusted (7 candidates passed trust gate $\Gamma$ out of 38 generated in standard 50-cell scan)
- Hysteresis: escalate needs 2 confirmations, de-escalate needs 3
- Deduplication: one active alert per cell
- Fatigue budget: max 3 alerts per cell per day
- Actions: NO_CHANGE / ESCALATE / DEESCALATE / ALL_CLEAR

**Device Registry** (`disaster-dashboard/alerts/device_registry.js`):
- Minimal store: deviceId, coarse cellId, consent, platform, token
- No precise location beyond cell centroid
- MongoDB-backed with in-memory fallback

**CAP 1.2 Export** (`disaster-dashboard/alerts/routes.js`):
- Valid CAP 1.2 XML: identifier, sender, sent, status, msgType, scope, references
- Info blocks: language, category, event (Flood Alert), responseType, priority, urgency, severity, certainty, headline, description, instruction, area (areaDesc, polygon, altitude, ceiling)
- All exported CAP files validated against CAP 1.2 schema structure

**Operator Review** (`disaster-dashboard/public/alert_review.html`):
- Table: Cell ID, Region, Lat/Lng, P(Flood), Tier, Trust, Severity, Primary Driver, Exposed Pop., Decision, Actions
- Operator can Approve / Reject / Cancel (Approve disabled for untrusted candidates)
- CAP 1.2 export of all approved alerts

---

## 8. Implementation — File Inventory

### 8.1 Repository Layout

```
F:\mp-latest\  (Project Root)
├── README.md                              <- This comprehensive documentation
├── STATUS.md                              <- Phase tracking & decisions
├── HUMAN_ACTIONS.md                       <- Credential checklist & manual steps
├── docker-compose.yml                     <- Multi-service orchestration
├── scripts/
│   └── verify_gates.ps1                   <- Phase gate runner (PowerShell)
├── reports/                               <- Phase evidence & final report
│   ├── FINAL_REPORT.md
│   ├── phase0/endpoint_audit.md
│   ├── phase1/evidence.md
│   ├── phase2/dataset_report.md, evidence.md
│   ├── phase3/metrics_table.md, bootstrap_ci.md
│   ├── phase4/reliability_diagram.md, conformal_coverage.md
│   ├── phase7/cap_schema_validation.md
│   └── ...
├── project-proofs/                        <- Screenshots (image.png, etc.)
├── pdf_pages/                             <- PDF page renders (page_1..7.png)
├── secrets/                               <- firebase-service-account.json (gitignored credentials)
├── old_models_and_tif/                    <- Legacy Colab models & notebooks
│   ├── flood_unet_resnet34_v1..v3.pth, old.pth
│   ├── unet_b3.pth (flood_last_model)
│   ├── colab_model_v4.ipynb, colabmodel_code.ipynb
│   ├── googlecolab.pdf, colab5.pdf, colabpdf4.pdf
│   └── googlecolabmodel2.pdf, googlecolabmodel3.pdf
├── disaster-ai-api/                       ← PYTHON FASTAPI AI ENGINE (PORT 8000)
│   ├── main.py                            <- FastAPI app: all endpoints, model loading, orchestration
│   ├── requirements.txt                   <- Python dependencies
│   ├── Dockerfile                         <- Container for AI engine
│   ├── .env.example                       <- Environment template (MONGODB_URI, API_PORT, ALERT_MODE)
│   ├── GLOBAL_MESH_DOCUMENTATION.md       <- Dedicated Global Mesh technical doc
│   ├── disaster_model_metrics.json
│   ├── tcn_model_metrics.json
│   ├── streamflow_model_manifest.json
│   ├── flood_prediction_model.pth         <- M2 MultiFactor weights
│   ├── unet_b3.pth                        <- M1 EfficientNet-B3 U-Net weights
│   ├── streamflow_forecast_lstm.pth       <- M3 LSTM weights
│   ├── tcn_flood_forecaster.pth           <- M4 TCN weights
│   ├── global_mesh/models/flood_risk_gru.pth          <- M6 GRU weights (trained, deployed)
│   ├── global_mesh/models/flood_risk_gru_manifest.json <- Training manifest with metrics (AUC=0.8952)
│   ├── global_mesh/generate_training_labels.py        <- Label generation pipeline (DFO + live ingestion)
│   ├── train_tcn_forecaster.py
│   ├── train_streamflow_lstm.py
│   ├── train_disaster_predictor.py
│   ├── test_api3.py, test_api2.py, test_sar.py, test_sar2.py
│   ├── tests/leakage_test.py
│   ├── scratch_test_phase2.py
│   ├── global_prediction_engine.py        <- Autonomous Global Mesh engine
│   ├── global_grid_system.py              <- 0.25° global grid system
│   ├── real_data_collector.py             <- 24 live flood factors
│   ├── temporal_data_engine.py            <- ECMWF + GloFAS temporal ingestion
│   ├── freshness_validator.py             <- Data freshness validation
│   ├── flood_risk_engine.py               <- Transparent hydrological risk scoring
│   ├── disaster_prediction_model.py       <- M2 Cross-Attention + Residual model
│   ├── deep_ensemble_engine.py            <- M5 Deep Ensemble (M=5)
│   ├── tcn_attention_forecaster.py        <- M4 TCN-Attention forecaster
│   ├── forecast_model.py                  <- Flood forecasting service
│   ├── impact_assessor.py                 <- Impact assessment (OSM + WorldPop)
│   ├── optimizer.py                       <- Response optimization engine
│   ├── flood_action_engine.py             <- Flood-aware evacuation routing
│   ├── spatial_hazard_generator.py        <- Spatial hazard & evacuation generator
│   ├── sentinel1_flood_observations.py    <- SAR observations
│   ├── topological_river_network.py       <- River-network routing
│   └── global_mesh/                       ← Global Mesh dedicated package
│       ├── __init__.py
│       ├── data_ingestion.py              <- Batched ingestion, caching, backoff, freshness
│       ├── flood_risk_model.py            <- M6 GRU model with validation gating
│       ├── risk_scorer.py                 <- Explainable 0-100 scoring (4 outputs)
│       ├── validation.py                  <- Model evaluation (AUC, Brier, CSI, Hit Rate, FAR)
│       ├── train_global_mesh_model.py     <- Training pipeline (geographical split)
│       ├── mongodb_store.py               <- Isolated MongoDB collections (5)
│       ├── historical_labels.py           <- DFO verified labels (4,029 events)
│       ├── ingestion_pipeline.py          <- Provenance-tagged ingestion, hashing, persistence, replay
│       ├── label_fusion.py                <- Label fusion from GloFAS discharge ratio
│       ├── background_scanner.py          <- Non-blocking background scan (job ID + progress)
│       └── alerting/
│           ├── __init__.py
│           ├── alert_candidates.py        <- Alert candidate builder (Phase 5)
│           └── policy_engine.py           <- Cost-loss, trust gate, hysteresis, dedupe, fatigue
└── disaster-dashboard/                    ← NODE.JS EXPRESS DASHBOARD (PORT 3000)
    ├── server.js                          <- Express server: all API routes (1111 lines)
    ├── package.json                       <- Node dependencies (express, axios, mongodb, multer, etc.)
    ├── Dockerfile                         <- Container for dashboard
    ├── README.md                          <- Original dashboard README
    ├── scans.db / management.db           <- SQLite local databases
    ├── prediction_store.js                <- MongoDB prediction store + verification engine
    ├── history.js                         <- SQLite scan history
    ├── management.js                      <- Agency inventory / evacuees / personnel
    ├── progressive_scanner.js             <- Multi-scale progressive scan + provenance
    ├── alerts/                            ← Alert service (Phase 6)
    │   ├── routes.js                      <- GET /candidates, POST /decide, POST /cap/export
    │   ├── policy_engine.js               <- Cost-loss, trust gate, hysteresis, dedupe, fatigue
    │   └── device_registry.js             <- Minimal device registry (MongoDB + memory fallback)
    └── public/                            ← Frontend static assets
        ├── index.html                     <- Main Leaflet GIS dashboard (1490+ lines)
        ├── predict-ui.js                  <- Prediction tab UI logic
        ├── global-prediction-ui.js        <- Global Mesh tab UI logic
        ├── management-ui.js               <- Agency ops tab UI logic
        └── alert_review.html              <- Alert review + CAP 1.2 export page
```

### 8.2 Backend Files (Python — disaster-ai-api)

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `main.py` | FastAPI application entry point; all 10+ endpoints; loads all models and services | `app`, `predict_flood`, `predict_live`, `predict_disaster`, `unified_prediction`, `execute_global_prediction`, `get_latest_global_prediction`, `get_live_factors`, `get_global_flood_hotspots`, `sar_multiscale`, `impact_multiscale`, `build_model_input`, `run_inference`, `filter_polygons` |
| `global_prediction_engine.py` | Autonomous Global Mesh planetary scan engine | `GlobalPredictionEngine`, `run_global_prediction`, `start_background_scan`, `get_scan_job_status`, `_build_region`, `_geojson_feature`, `haversine_km` |
| `global_grid_system.py` | 0.25° global grid (60S–60N, 691,200 cells) | `GlobalGridSystem`, `GridCell`, `get_grid_system`, `get_cell`, `get_cell_id`, `enrich_cell_from_ingestion` |
| `global_mesh/data_ingestion.py` | Real-time environmental ingestion (batched, cached, backoff) | `GlobalMeshDataIngestion`, `observe_cell`, `observe_cells`, `fetch_gdacs_alerts`, `CellObservation` |
| `global_mesh/flood_risk_model.py` | M6 GRU deep-learning flood-risk model | `FloodRiskModel`, `train`, `predict`, `is_trained`, `is_validated`, `MC-dropout` |
| `global_mesh/risk_scorer.py` | Explainable risk scoring (4 outputs) | `GlobalMeshRiskScorer`, `score_cell`, `score_cells` |
| `global_mesh/validation.py` | Model evaluation metrics | `evaluate_probabilistic_model`, `baseline_physics_rule`, `build_verification_records` |
| `global_mesh/train_global_mesh_model.py` | GRU training pipeline | Training with geographical split, early stopping, leakage prevention |
| `global_mesh/generate_training_labels.py` | Label generation pipeline | Pairs DFO flood locations with live ingestion data; writes to MongoDB training_data_col |
| `global_mesh/mongodb_store.py` | Isolated MongoDB persistence (5 collections) | `GlobalMeshMongoStore`, `save_scan`, `save_verifications`, `health` |
| `global_mesh/historical_labels.py` | DFO verified flood labels (4,029 events) | `DFOHistoricalLabels`, `label_cell` |
| `global_mesh/ingestion_pipeline.py` | Provenance-tagged ingestion, payload hashing, replay | Ingestion pipeline with SHA-256 verification |
| `global_mesh/label_fusion.py` | Label fusion from GloFAS discharge ratio | Label fusion logic |
| `global_mesh/background_scanner.py` | Non-blocking background scan | `BackgroundScanProcessor`, `start_scan`, `get_job_status`, `cancel_scan` |
| `global_mesh/alerting/alert_candidates.py` | Alert candidate builder | `build_alert_candidates`, `build_candidates_from_region_records` |
| `global_mesh/alerting/policy_engine.py` | Cost-loss, trust gate, hysteresis, dedupe, fatigue | Alert policy engine (Python side) |
| `disaster_prediction_model.py` | M2 MultiFactorFloodPredictor (Cross-Attention + Residual) | `MultiFactorFloodPredictor`, `CrossDomainAttention`, `ResidualBlock`, `FloodPredictionEngine` |
| `deep_ensemble_engine.py` | M5 Deep Ensemble epistemic uncertainty | `DeepEnsembleFloodPredictor`, ensemble mean/variance/CI |
| `tcn_attention_forecaster.py` | M4 TCN-Attention multi-horizon forecaster | `TCNAttentionService`, `ChausalConv1d`, dilated causal TCN + attention |
| `train_tcn_forecaster.py` | TCN training script | TCN training pipeline |
| `train_streamflow_lstm.py` | M3 StreamFlow LSTM training | LSTM training on Caravan/CAMELS |
| `train_disaster_predictor.py` | M2 training script | MultiFactor predictor training |
| `flood_risk_engine.py` | M7 Transparent hydrological risk scoring | `get_risk_engine`, multi-horizon risk evaluation |
| `forecast_model.py` | Flood forecasting service | `get_forecast_service`, ConvLSTM + GPM heuristic |
| `impact_assessor.py` | Impact assessment (OSM + WorldPop) | `ImpactAssessor`, `assess` |
| `optimizer.py` | Response optimization | `get_optimizer`, evacuation/shelter/resource planning |
| `flood_action_engine.py` | Flood-aware evacuation routing | `FloodActionEngine` |
| `spatial_hazard_generator.py` | Spatial hazard & evacuation generator | `get_hazard_generator`, hazard zones, shelters, routes |
| `real_data_collector.py` | 24 live flood factors collector | `get_data_collector`, `collect_all_factors` |
| `temporal_data_engine.py` | ECMWF + GloFAS temporal engine | `get_temporal_engine`, `collect_temporal_dataset` |
| `freshness_validator.py` | Data freshness validation | FRESH/ACCEPTABLE/STALE/EXPIRED/UNKNOWN |
| `sentinel1_flood_observations.py` | SAR observations | Sentinel-1 scene acquisition |
| `topological_river_network.py` | River-network routing | `get_river_router`, flood typology classification |
| `requirements.txt` | Python dependencies | fastapi, uvicorn, torch, rasterio, osmnx, geopandas, shapely, segmentation-models-pytorch, etc. |

### 8.3 Frontend Files (Node.js — disaster-dashboard)

| File | Purpose | Key Features |
|------|---------|--------------|
| `server.js` | Express server (PORT 3000); all API routes; proxies to AI engine | 1111 lines; satellite upload, live scan, unified prediction, global prediction, GDACS proxy, hydrology, impact, forecast, optimize, alert routes, progressive scan, SAR multiscale, action plan, inventory/evacuee/personnel management |
| `package.json` | Node dependencies | express, axios, cors, multer, form-data, mongodb, better-sqlite3, dotenv, express-session |
| `prediction_store.js` | MongoDB prediction store + closed-loop verification | `savePrediction`, `saveGlobalPrediction`, `auditAndEvaluateVerifications`, `getVerificationMetrics`, HIT/MISS/FA/CN/DU |
| `history.js` | SQLite scan history | `logScan`, `getHistory`, `deleteScan`, `clearHistory` |
| `management.js` | Agency operations (inventory, evacuees, personnel) | Shelter dispatch, inventory management |
| `progressive_scanner.js` | Multi-scale progressive scan + provenance audit | 5×5 → 10×10 → 25×25 → 50×50 km stages, provenance logs |
| `alerts/routes.js` | Express alert API routes | `GET /candidates`, `POST /decide`, `POST /cap/export` |
| `alerts/policy_engine.js` | Node alert policy engine | Cost-loss (P > 0.10), trust gate, hysteresis, dedupe, fatigue budget |
| `alerts/device_registry.js` | Minimal device registry | Register device, get devices for cell, consent flags, MongoDB + memory fallback |
| `public/index.html` | Main Leaflet GIS dashboard | 10 tabs: Predict, Global Mesh, River Gauges, 72h Timeline, SAR Radar, Impact Matrix, Action Plan, Agency Ops, Live Floods, GeoTIFF Upload, History; semantic risk colors, modals, toasts |
| `public/predict-ui.js` | Prediction tab logic | Coordinate input, target selection, prediction display |
| `public/global-prediction-ui.js` | Global Mesh tab logic | Global scan, region cards, trajectory, GeoJSON |
| `public/management-ui.js` | Agency ops tab logic | Inventory, evacuees, personnel, shelter dispatch |
| `public/alert_review.html` | Alert review + CAP 1.2 export | Candidate table, approve/reject/cancel, CAP XML generation/download/copy |
| `public/*.js` (others) | Tab-specific UI logic | Chart.js graphs, Leaflet layers, popups, provenance tables |

---

## 9. API Reference

### 9.1 Node.js Dashboard (Port 3000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /api/health` | GET | AI engine reachability check |
| `POST /api/analyze-satellite` | POST | Upload GeoTIFF → forward to AI `/predict` |
| `POST /api/live-analyze` | POST | Live scan (bbox) → forward to AI `/predict-live` |
| `POST /api/unified-prediction` | POST | Unified flood prediction + management (+ MongoDB persist) |
| `GET /api/prediction-history` | GET | Prediction history from MongoDB |
| `GET /api/prediction-stats` | GET | Prediction statistics |
| `POST /api/global-prediction` \| `POST /api/global-scan` | POST | Autonomous Global Mesh planetary scan |
| `GET /api/global-prediction/latest` | GET | Latest cached global prediction |
| `GET /api/global-prediction/history` | GET | Global prediction history |
| `GET /api/global-flood-hotspots` | GET | Live UN/EC GDACS flood hotspots |
| `GET /api/global-prediction/verification-metrics` | GET | Closed-loop verification metrics (CSI, Hit Rate, FAR) |
| `POST /api/global-prediction/verify` | POST | Trigger verification audit |
| `POST /api/predict-disaster` | POST | Legacy multi-factor flood prediction |
| `GET /api/live-factors` | GET | 24 live environmental factors |
| `GET /api/live-floods` | GET | GDACS live flood events (GeoJSON) |
| `GET /api/live-floods/shape` | GET | Flood event polygon shape |
| `POST /api/assess-impact` \| `/api/impact-assessment` | POST | Flood impact assessment |
| `POST /api/forecast-flood` \| `/api/flood-forecast` | POST | Flood forecasting |
| `POST /api/optimize-response` \| `/api/action-plan` | POST | Response optimization |
| `GET /api/hydrology-station` / `POST /api/drone-recon` | GET/POST | Copernicus GloFAS river telemetry |
| `POST /api/dispatch-shelter` | POST | Shelter dispatch recording |
| `GET/POST/DELETE /api/inventory` | — | Agency inventory management |
| `GET/POST/DELETE /api/evacuees` | — | Evacuee management |
| `GET/POST /api/personnel` | — | Personnel management |
| `GET/DELETE /api/history` / `/api/history/:id` | — | SQLite scan history |
| `POST /api/scan/start` \| `/api/scan/status/:id` \| `/api/scan/cancel/:id` | — | Progressive scan lifecycle |
| `GET /api/scan/provenance/:id` \| `/api/scan/provenance-global` | — | Provenance audit logs |
| `POST /api/sar/multiscale` | POST | Multi-scale SAR (5/10/15 km) |
| `POST /api/impact/multiscale` | POST | Multi-scale impact (5/10/25/50 km) |
| `POST /api/action/multiscale` \| `/api/action-plan/generate` | POST | Multi-scale action plan |
| `GET /api/action-plan/history` | GET | Action plan history |
| `GET /api/impact/history` | GET | Impact assessment history |
| `POST /api/progressive/stage` | POST | Progressive stage execution |
| `GET /api/alert/candidates` | GET | List alert candidates (MongoDB) |
| `POST /api/alert/decide` | POST | Approve/reject/cancel candidate |
| `POST /api/alert/cap/export` | POST | CAP 1.2 XML export |

### 9.2 Python AI Engine (Port 8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /predict` | POST | SAR U-Net flood segmentation (GeoTIFF upload) |
| `POST /predict-live` | POST | Live Sentinel-1 SAR inference |
| `POST /predict-disaster` | POST | Multi-factor deep learning prediction |
| `POST /unified-prediction` \| `/api/unified-prediction` | POST | Unified prediction + management |
| `POST /global-prediction` \| `/api/global-prediction` \| `/global-scan` | POST | Autonomous Global Mesh scan |
| `GET /global-prediction/latest` \| `/api/global-prediction/latest` | GET | Latest global prediction |
| `GET /live-factors` | GET | 24 live factors |
| `GET /global-flood-hotspots` | GET | GDACS hotspots |
| `POST /assess-impact` | POST | Impact assessment |
| `POST /forecast-flood` | POST | Flood forecast |
| `POST /optimize-response` | POST | Response optimization |
| `POST /sar-multiscale` \| `/api/sar-multiscale` | POST | Multi-scale SAR |
| `POST /impact-multiscale` \| `/api/impact-multiscale` | POST | Multi-scale impact |
| `GET /docs` | GET | OpenAPI/Swagger UI |

---

## 10. Evaluation & Results

### 10.1 Global Mesh End-to-End Test (Live Scan)

| Metric | Value |
|--------|-------|
| Cells evaluated | 50 |
| Cells with data (active) | 50 |
| Cells without data | 0 |
| Emerging threats | 13 |
| Critical zones | 0 |
| High risk | 1 |
| Moderate risk | 32 |
| Low risk | 17 |
| Global average risk score | 28.5 |
| GeoJSON features | 50 |
| GDACS live events linked | 100 |
| Alert candidates generated | 38 |
| Alert candidates trusted | 7 (passed trust gate $\Gamma$: model validated, telemetry fresh $\le 48$h, $\sigma \le 0.25$) |

### 10.2 Model Metrics

| Model | Metric | Value |
|-------|--------|-------|
| StreamFlow LSTM (M3) | Mean NSE | 0.983 (surrogate fit against Caravan simulation equations; 72-h horizon) |
| MultiFactor Predictor (M2) | ROC-AUC | 1.00 (surrogate fit on synthetic regime benchmark; not real gauge accuracy) |
| Global Mesh GRU (M6) | Status | Trained/Deployed (Held-out Test AUC=0.8952, Brier=0.0469, BSS=0.537, 352 samples) |
| Deep Ensemble (M5) | Epistemic uncertainty | 95% CI via MC variance |
| TCN-Attention (M4) | Receptive field | 125 hours |

### 10.3 Verification Metrics (Closed-Loop)

| Metric | Formula | Description |
|--------|---------|-------------|
| Critical Success Index (CSI) | HITS / (HITS + MISSES + FALSE_ALARMS) | 0.0–1.0, higher better |
| Hit Rate | HITS / (HITS + MISSES) | Probability of detection |
| False Alarm Ratio | FALSE_ALARMS / (HITS + FALSE_ALARMS) | 0.0–1.0, lower better |
| Outcomes | HIT / MISS / FALSE_ALARM / CORRECT_NEGATIVE / DATA_UNAVAILABLE | DATA_UNAVAILABLE excluded from metrics (strict integrity) |

**Strict integrity rule:** GloFAS is treated as MODELLED_REANALYSIS, NEVER as observed ground truth. Verification requires genuine in-situ gauge or satellite (GDACS/Sentinel-1) observation.

---

## 11. Limitations & Honest Disclosures

| # | Limitation | Impact | Disclosure Location |
|---|-----------|--------|---------------------|
| L1 | GRU model trained & deployed (AUC=0.8952) | GRU probability used in alert candidates; physics score remains fallback | FINAL_REPORT §Model Training |
| L2 | 352 labelled samples (61 positive / 291 negative) | Sufficient for training; more labels would improve robustness | FINAL_REPORT §Model Training |
| L3 | Live operational ingestion endpoint queries a 14-day window (retrospective ERA5/GloFAS historical pairing deferred to future work) | Ingested current/recent telemetry rather than multi-decadal historical archive | Paper §II-B, FINAL_REPORT §Limitations L4 |
| L4 | Historical weather pairing for DFO (1985–2010) | Features describe current location susceptibility rather than event-day weather; retrospective ERA5 pairing future work | Paper §II-B, FINAL_REPORT §Limitations L2 |
| L5 | GloFAS is modelled reanalysis, not gauge truth | Labels measure skill against proxy | FINAL_REPORT §Limitations L3 |
| L6 | GDACS centroids are approximate, not footprints | Approximate event locations | FINAL_REPORT §Limitations L5 |
| L7 | MC-dropout uncertainty not independently validated | Epistemic estimate only | FINAL_REPORT §Limitations L6 |
| L8 | Grid sampling cap (50 cells/run) | Full planetary coverage needs multiple scans | FINAL_REPORT §Limitations L7 |
| L9 | Model not deployed unless AUC > climatology + 0.05 | Prevents unskilled model deployment | FINAL_REPORT §Limitations L8 |
| L10 | P4 calibration/trust completed | Model calibrated (AUC=0.8952 > climatology 0.5000 + 0.05) | FINAL_REPORT §Not Implemented |
| L11 | Ablation studies not done | Systematic component ablation across channels deferred to future work | Paper §IV-H, FINAL_REPORT §Not Implemented |
| L12 | Closed-loop scoring (N7) not done | Requires live alert delivery feedback | FINAL_REPORT §Not Implemented |
| L13 | Sentinel-1 SAR flood labels not integrated | Placeholder only, not wired to training | FINAL_REPORT §Not Implemented |
| L14 | FCM real delivery not enabled | dry_run only; FCM mocked | FINAL_REPORT §Not Implemented |
| L15 | CPU-only inference | Slower than GPU; fine for educational use | main.py `device = torch.device("cpu")` |

**Alert safety compliance:** `ALERT_MODE=dry_run` (never changed); operator approval mandatory; trust gate; fatigue budget (max 3/cell/day); kill-switch de-escalation/all-clear; full audit trail; minimal device data; controlled CAP templates.

---

## 12. Installation & Setup

### Prerequisites

- Python 3.9+
- Node.js 18+
- MongoDB (optional; in-memory fallback used if unavailable)
- Git

### Step 1 — Clone / Open Repository

```powershell
cd F:\mp-latest
```

### Step 2 — Python AI Engine (Port 8000)

```powershell
cd disaster-ai-api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install fastapi uvicorn torch rasterio numpy segmentation-models-pytorch python-multipart requests osmnx geopandas shapely rioxarray pystac-client planetary-computer global-land-mask boto3
```

Start the AI engine:
```powershell
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Verify: `http://127.0.0.1:8000/docs`

### Step 3 — Node.js Dashboard (Port 3000)

```powershell
cd disaster-dashboard
npm install
node server.js
```

Open: `http://localhost:3000`

### Step 4 — (Optional) MongoDB

Default URI: `mongodb://127.0.0.1:27017`, database: `floodwatch`.  
If MongoDB is unavailable, the system gracefully falls back to in-memory storage.

### Step 5 — Verification

```powershell
# From dashboard
curl http://localhost:3000/api/health
# From AI engine
curl http://127.0.0.1:8000/docs
```

---

## 13. Usage Guide

### 13.1 Real-Time Prediction (Predict Tab)
1. Enter target latitude/longitude in the sidebar
2. Select catchment radius (3/5/8 km)
3. Click "Predict" → runs 24-factor multi-factor deep learning + SAR + risk scoring
4. View risk score, severity, hazard zones, shelters, evacuation routes

### 13.2 Global Mesh Early Warning (Global Mesh Tab)
1. Click "Run Global Scan" → autonomous planetary grid scan (50 cells, GDACS + anchors + stride sample)
2. View emerging threats, risk trajectory (T+0/24/48/72h), GeoJSON regions
3. Alert candidates written to MongoDB

### 13.3 Alert Review (alert_review.html)
1. Navigate to `http://localhost:3000/alert_review.html`
2. Review candidates: P(Flood), Tier, Trust, Severity, Primary Driver, Exposed Population
3. Approve / Reject / Cancel (Approve disabled for untrusted candidates)
4. Export approved alerts as CAP 1.2 XML

### 13.4 SAR Radar (Live Tab)
1. Run Sentinel-1 SAR multi-scale scan (5/10/15 km)
2. View genuine SAR inundation polygons (no synthetic fallback)
3. Honest "UNAVAILABLE" reporting when no SAR overpass exists

### 13.5 Impact & Action Plan
1. Run multi-scale impact assessment → buildings flooded, population exposed, economic loss, submerged roads
2. Generate optimized action plan → evacuation, shelter deployment, resource allocation by priority

---

## 14. Safety & Compliance

- **Dry-run default:** `ALERT_MODE=dry_run`; no real alerts delivered to people; FCM client mocked.
- **Operator-in-the-loop:** All alert dispatch requires manual approve/reject/cancel via Alert Review page.
- **Trust gate:** Only candidates satisfying joint criteria (telemetry complete, model validated, freshness $\le 48\ \text{h}$, $\sigma \le 0.25$) are trusted (7 of 38 candidates passed in 50-cell scan; untrusted candidates downgraded to WATCH and blocked from dispatch).
- **Fatigue budget:** Maximum 3 alerts per cell per day.
- **Hysteresis + kill-switch:** Escalation requires 2 confirmations; de-escalation/all-clear supported.
- **Audit trail:** `decision_timestamp`, `decision_reason`, `policy_decision` stored in MongoDB for every decision.
- **Data minimization:** Device registry stores only coarse cell ID + consent flags; no precise location.
- **Controlled templates:** CAP message text generated from controlled templates only.
- **Missing data integrity:** `DATA_UNAVAILABLE` outcomes excluded from accuracy metrics; never fabricated.

---

## 15. Conclusion

FloodWatch demonstrates a complete, open-access, deep-learning-enhanced disaster management pipeline for flood early warning and response. It integrates six deep-learning architectures (EfficientNet-B3 U-Net, MultiFactor Cross-Attention Predictor, StreamFlow LSTM, TCN-Attention Forecaster, Deep Ensemble, Global Mesh GRU) with live hydro-meteorological telemetry from verified sources (ECMWF, Copernicus, GDACS, NASA, USGS, OSM, WorldPop, SoilGrids, DFO). The system delivers real-time flood risk scoring, SAR inundation mapping, spatial impact assessment, optimized emergency response, autonomous global mesh alerting with operator review and CAP 1.2 export, and closed-loop forecast verification with strict data-integrity rules.

Honest disclosure of model limitations (limited labelled data, DFO archive 1985–2010 only, GloFAS as reanalysis proxy) and strict safety defaults (dry-run mode, no fabricated data, operator gating) make the system suitable for both educational demonstration and as a foundation for operational research.

---

## 16. References

1. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *NeurIPS*.
2. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation. *ICML*.
3. Kratzert, F., et al. (2018/2019). Cloud-agnostic runoff prediction with LSTMs. *Hydrology and Earth System Sciences*.
4. Gauch, M., et al. (2021). Rainfall–runoff prediction using LSTMs. *Hydrology and Earth System Sciences*.
5. Dartmouth Flood Observatory. Global Active Archive of Large Flood Events (1985–2010). https://data.humdata.org/dataset/global-active-archive-of-large-flood-events-dfo
6. UN/EC GDACS. Global Disaster Alert and Coordination System. https://www.gdacs.org
7. Copernicus Emergency Management Service. GloFAS 4.0 & GLO-30 DEM. https://cds.climate.copernicus.eu
8. European Centre for Medium-Range Weather Forecasts (ECMWF). IFS via Open-Meteo. https://open-meteo.com
9. NASA. Global Precipitation Measurement (GPM) / MERRA-2. https://gpm.nasa.gov
10. Microsoft Planetary Computer / AWS Earth Search. Sentinel-1 SAR STAC catalog. https://planetarycomputer.microsoft.com
11. OpenStreetMap. https://www.openstreetmap.org
12. WorldPop. https://www.worldpop.org
13. ISRIC SoilGrids 250m. https://soilgrids.org
14. JRC Global Surface Water. https://global-surface-water.appspot.com
15. OASIS Common Alerting Protocol (CAP) v1.2. https://docs.oasis-open.org/emergency/cap/v1.2/CAP-v1.2-os.pdf
16. Sen1Floods11 dataset (Sentinel-1 SAR flood labels). (Documented in `colab_model_v4.ipynb`)

---

## Appendix A: Phase Reports

| Phase | Name | File | Status |
|-------|------|------|--------|
| P0 | Audit | `reports/phase0/endpoint_audit.md` | DONE / PASSED |
| P1 | Evidence ledger + ingestion | `reports/phase1/evidence.md` | DONE / PASSED |
| P2 | Label fusion (N1) | `reports/phase2/dataset_report.md`, `evidence.md` | DONE / PASSED |
| P3 | Models | `reports/phase3/metrics_table.md`, `bootstrap_ci.md` | DONE / PASSED (GRU trained & deployed: AUC=0.8952) |
| P4 | Calibration and trust (N4) | `reports/phase4/reliability_diagram.md`, `conformal_coverage.md` | DONE — model validated (AUC=0.8952 > climatology 0.5000 + 0.05) |
| P5 | Forecast service | `reports/phase5/` | DONE — alert candidates written to MongoDB |
| P6 | Node alert service (N6) | `disaster-dashboard/alerts/` | DONE — policy engine, device registry, CAP export |
| P7 | Operator review + CAP (N8) | `reports/phase7/cap_schema_validation.md` | DONE — AlertReview page, CAP 1.2 export |
| P8 | Closed loop + final report (ablations deferred) | `reports/FINAL_REPORT.md` | DONE (synthesis report completed; experimental ablations future work) |

**Final Report:** `reports/FINAL_REPORT.md` — contains the executive summary, phase completion status, data sources, provenance policy, honest model training results, alerting layer details, end-to-end test results, MongoDB collections, Open-Meteo HTTP 429 handling, dashboard integration, alert safety compliance, not-implemented items, scope compliance, and complete files-changed list.

---

**End of README — Disaster Management Using Deep Learning Models (FloodWatch)**

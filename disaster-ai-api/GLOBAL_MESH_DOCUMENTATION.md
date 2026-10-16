# GLOBAL MESH — Complete Technical Documentation

**Version:** 2.0 (Grid-Based, Real-Time, Deep-Learning-Enhanced Flood Prediction System)

---

## 1. OVERVIEW

Global Mesh is the autonomous global flood early-warning and prediction component of the FloodWatch disaster management system. It evaluates environmental conditions cell-by-cell across a global 0.25° geographical grid (60°S–60°N), ingesting live data from authoritative open-access sources, scoring flood risk using an explainable composite index with an optional deep-learning model, detecting emerging high-risk regions, and persisting results to MongoDB.

**Scope:** Global Mesh and its dedicated supporting modules only. Other features (Predict, SAR Radar, Impact Matrix, Action Plan, etc.) are served by separate shared modules and are intentionally left untouched.

---

## 2. ARCHITECTURE

```
Global Mesh Components
├── global_prediction_engine.py          — Core engine (orchestration, scoring, API)
├── global_grid_system.py                — 0.25° global grid, cell metadata, enrichment
└── global_mesh/                         — Dedicated package
    ├── __init__.py                      — Package exports
    ├── data_ingestion.py                — Real-time environmental data ingestion (batched, cached, backoff)
    ├── flood_risk_model.py              — GRU deep-learning flood-risk model (with validation gating)
    ├── risk_scorer.py                   — Explainable 0-100 risk scoring (4 distinct outputs)
    ├── validation.py                    — Model evaluation metrics (AUC, Brier, CSI, Hit Rate, FAR)
    ├── train_global_mesh_model.py       — Training pipeline (geographical split, leakage prevention)
    ├── mongodb_store.py                 — Isolated MongoDB persistence (5 collections)
    ├── historical_labels.py             — DFO verified historical flood labels (4,029 events)
    ├── background_scanner.py            — Non-blocking background scan processing
    ├── models/flood_risk_gru.pth        — Trained model weights (saved, may be untrained)
    ├── models/flood_risk_gru_manifest.json — Training manifest with honest metrics
    └── data_sources/dfo_sample/         — DFO verified flood archive (shapefile)
```

### Data Flow

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│   GDACS Alerts  │────▶│  Candidate Cell Set  │────▶│  Batched Observation│
│  (100 events)   │     │  (anchors + stride)  │     │  (ECMWF + GloFAS +  │
└─────────────────┘     └──────────────────────┘     │   DEM + GDACS)      │
                                                     └──────────┬──────────┘
                                                                │
                                                     ┌──────────▼──────────┐
                                                     │   Risk Scorer       │
                                                     │   (4 outputs: index,│
                                                     │    probability,     │
                                                     │    hazard, impact)  │
                                                     └──────────┬──────────┘
                                                                │
                                                     ┌──────────▼──────────┐
                                                     │   Region Records    │
                                                     │   (scores, drivers, │
                                                     │    typology, etc.)  │
                                                     └──────────┬──────────┘
                                                                │
                                                     ┌──────────▼──────────┐
                                                     │   MongoDB Storage   │
                                                     │   (global_mesh_*)   │
                                                     └─────────────────────┘
```

---

## 3. DATA SOURCES (ALL VERIFIED LIVE, NO API KEY)

| Source | Data Provided | Update Frequency | Coverage | Provenance Tag |
|--------|--------------|------------------|----------|----------------|
| **Open-Meteo / ECMWF IFS** | Hourly precipitation (past 14d + forecast 72h), temperature, soil moisture (4 layers), CAPE, snow depth, ET0 | Hourly | Global (~9 km) | OBSERVATION / FORECAST |
| **Open-Meteo Flood API / Copernicus GloFAS 4.0** | Daily river discharge, climatological mean, ensemble max/median (past 14d + 7-day forecast) | Daily | Global river grid (~5 km) | MODELLED_REANALYSIS |
| **Open-Meteo Elevation / Copernicus GLO-30 DEM** | Elevation, slope, Topographic Wetness Index | Static (invariant) | Global (30 m) | OBSERVED_GIS |
| **UN/EC GDACS** | Live satellite flood-alert centroids (Green/Orange/Red) | Near real-time | Global active events | EXTERNAL_ALERT_EVIDENCE |
| **Dartmouth Flood Observatory** | 4,029 verified historical flood events (1985–2010) with centroids, dates, severity, cause | Static archive | Global | Verified historical label |

**Freshness:** Every telemetry item carries a source timestamp and freshness status (`FRESH` / `ACCEPTABLE` / `STALE` / `EXPIRED` / `UNKNOWN`) validated via `freshness_validator`. Missing/unavailable data is never fabricated and never interpreted as "safe."

---

## 4. MODULE REFERENCE

### 4.1 `global_mesh/data_ingestion.py` — Real-Time Environmental Data Ingestion

**Class:** `GlobalMeshDataIngestion`

**Responsibilities:**
- Fetch live weather + hydrology for grid cells from Open-Meteo (batched requests)
- Persistent disk cache for static elevation/terrain (7-day TTL)
- Source-aware in-memory caching (weather 1h, forecast 30min, GloFAS 1h, elevation 7 days)
- Rate limiting (token bucket, 2 req/sec)
- Exponential backoff (1s → 60s max) with `Retry-After` header support on HTTP 429
- Request deduplication (5-minute window)
- Freshness validation on every telemetry item
- Return structured `CellObservation` objects with explicit provenance

**Key Methods:**
- `observe_cell(cell_id, lat, lon, gdacs_events, use_cache)` — Collect all telemetry for one cell
- `observe_cells(cells, gdacs_events)` — Batch observe multiple cells (efficiency)
- `fetch_gdacs_alerts()` — Live UN/EC GDACS flood alerts

**CellObservation Fields:**
- `obs_precip_24h_mm`, `obs_precip_72h_mm`, `obs_antecedent_precip_index_14d_mm`
- `fcst_precip_24h_mm`, `fcst_precip_72h_mm` (forecast)
- `obs_soil_moisture_surface`, `obs_soil_moisture_root`
- `obs_temperature_c`, `obs_cape_jkg`, `obs_snow_depth_cm`
- `obs_river_discharge_m3s`, `obs_river_discharge_mean_m3s`, `obs_discharge_ratio`
- `fcst_river_discharge_m3s`, `fcst_river_discharge_max_m3s`
- `recur_2yr_m3s`, `recur_5yr_m3s`, `recur_20yr_m3s`
- `obs_elevation_m`, `obs_slope_pct`, `obs_topographic_wetness_index`
- `obs_land_cover_class`, `obs_is_coastal`
- `gdacs_alert_level`, `gdacs_alert_distance_km`
- `data_available`, `missing_factors`, `freshness`, `source_timestamps`

### 4.2 `global_mesh/flood_risk_model.py` — GRU Deep-Learning Model

**Architecture:** GRU Encoder + Static Branch + Probabilistic Head
- Encoder: single-layer GRU ingests 72-hour antecedent sequence (9 dynamic features)
- Static branch: catchment attributes (6 static features) projected via 2-layer MLP
- Head: fused representation → sigmoid probability P(flood=1)
- MC-dropout at inference (25 passes) → epistemic std + 95% confidence interval

**Dynamic Features:** precip_24h, precip_72h, API-14, soil moisture surface, soil moisture root, discharge ratio, temperature, CAPE, snow depth
**Static Features:** elevation, slope, TWI, land cover class, coastal flag, GDACS distance

**Validation Gating:**
- `is_trained()` — weights loaded
- `is_validated()` — demonstrated skill on held-out data
- Model probability is **NOT used** in risk scoring unless validated
- `set_validated(bool)` — called only after training pipeline confirms skill

**Key Methods:**
- `train(sequences, statics, labels, val_split, epochs, batch_size, lr, patience)` — train with early stopping
- `predict(obs, mc_passes)` — flood probability + uncertainty
- `save(path)` / `load(path)` — persistence with manifest

### 4.3 `global_mesh/risk_scorer.py` — Explainable Risk Scoring

**Class:** `GlobalMeshRiskScorer`

**Four Distinct Outputs:**

| Output | Type | Range | Interpretation |
|--------|------|-------|----------------|
| Risk Index | Uncalibrated composite | 0–100 | Relative risk, NOT a probability |
| Flood Probability | Calibrated P(flood=1) | 0–1 | Only when GRU validated |
| Physical Hazard | Physically-based estimate | 0–1 | Discharge + soil + rainfall |
| Flood Impact | Exposed population / area | count / km² | Not a probability |

**Factor Shares (equal weighting, re-normalized over available):**
- Rainfall: 0.20 | Discharge: 0.20 | Soil Saturation: 0.20 | Terrain: 0.20 | GDACS Alert: 0.20

**Normalization Scales (physically interpretable):**
- Rainfall: saturates at 150 mm (72h)
- Discharge: sigmoid centered at bankfull (1.30× mean)
- Soil: saturates at field capacity (0.45 m³/m³)
- Terrain: elev < 200 m, slope < 12%, coastal = vulnerable
- GDACS: decays with distance (25 km radius), weighted by alert level

**Key Methods:**
- `score_cell(obs, model_pred)` — full 4-output assessment
- `score_cells(observations, model_preds)` — batch scoring

### 4.4 `global_mesh/mongodb_store.py` — Isolated MongoDB Persistence

**Class:** `GlobalMeshMongoStore`

**Collections (all prefixed `global_mesh_` for isolation):**

| Collection | Purpose |
|------------|---------|
| `global_mesh_scans` | Completed scan records (inputs, outputs, model version) |
| `global_mesh_scan_history` | Per-cell observation history (time-series) |
| `global_mesh_training_data` | Labelled training samples (inputs + verified flood labels) |
| `global_mesh_model_versions` | Model versions, evaluation metrics, deployment status |
| `global_mesh_verifications` | Closed-loop forecast verification (HIT/MISS/FA/CN/DU) |

**Key Methods:**
- `save_scan(record)` / `get_latest_scan()` / `get_scan_history(limit)`
- `save_cell_observations(scan_id, observations)` / `get_cell_history(cell_id, limit)`
- `save_training_samples(samples)` / `get_training_samples(limit)`
- `save_model_version(version, metrics, deployed, deploy_reason, manifest)`
- `save_verifications(records)` / `get_pending_verifications()` / `get_verification_metrics()`
- `health()` — MongoDB connectivity status

**In-memory fallback** used if MongoDB unavailable.

### 4.5 `global_mesh/historical_labels.py` — DFO Verified Flood Labels

**Class:** `DFOHistoricalLabels`

**Source:** Dartmouth Flood Observatory Global Active Archive of Large Flood Events
- 4,029 verified flood events globally, 1985–2010
- Fields: centroid (lat/lon), dates, severity, cause, country, rivers
- License: CC BY
- Downloaded from: https://data.humdata.org/dataset/global-active-archive-of-large-flood-events-dfo

**Label Method:** A grid cell is labelled flood=1 if a DFO event centroid is within 50 km AND the observation date falls within the event's [began, ended] window.

**Limitations (clearly identified):**
- Archive covers 1985–2010 only (no recent floods)
- Centroids are approximate event locations, not precise inundation footprints
- Large/major floods only — smaller/local floods not included
- GloFAS discharge is modelled reanalysis, not gauge truth

**Key Methods:**
- `label_cell(lat, lon, observation_date, radius_km)` — (label, event_id, details)
- `get_verified_flood_locations(min_severity, limit)` — flood centroids for training

### 4.6 `global_mesh/background_scanner.py` — Background Scan Processing

**Class:** `BackgroundScanProcessor`

**Problem Solved:** Synchronous `run_global_prediction()` blocked HTTP threads for 45–300 seconds, causing dashboard timeouts.

**Solution:** Non-blocking background scan execution.
- Scan starts → returns `job_id` immediately
- Background thread processes cells in batches of 10
- Progress tracked per-cell and persisted incrementally to MongoDB
- Failed/interrupted scans retain completed work
- Poll `get_job_status(job_id)` for progress
- Completion only reported when job actually reaches `COMPLETED`

**Job Statuses:** `PENDING` → `RUNNING` → `COMPLETED` | `FAILED`

**Key Methods:**
- `start_scan(max_cells, force_new)` — start non-blocking scan, returns job info
- `get_job_status(job_id)` — poll progress
- `cancel_scan(job_id)` — cancel running scan

### 4.6 `global_mesh/validation.py` — Model Evaluation

**Metrics Computed:** ROC-AUC, Brier Score, Reliability Slope, Critical Success Index (CSI), Hit Rate, False Alarm Ratio — per threshold and overall.

**Baselines:** Climatology (predict mean label frequency) for comparison.

**Key Functions:**
- `evaluate_probabilistic_model(probs, targets, model_name)` — full evaluation
- `baseline_physics_rule(obs_list)` — physics-only comparator (discharge > 1.3× OR rain > 40 mm OR soil > 0.45)
- `build_verification_records(scan_id, predictions, horizon_hours)` — closed-loop audit scaffold

---

## 5. CORE ENGINE — `global_prediction_engine.py`

**Class:** `GlobalPredictionEngine`

**Preserved Public API:**
- `get_global_engine() -> GlobalPredictionEngine`
- `engine.run_global_prediction() -> dict`
- `engine.get_latest_cached_run() -> dict | None`
- `engine.get_prediction_history(limit=10) -> list`
- `engine.health() -> dict`

**New Methods:**
- `engine.start_background_scan(max_cells, force_new) -> dict` — non-blocking scan
- `engine.get_scan_job_status(job_id) -> dict | None` — poll progress
- `engine.cancel_scan_job(job_id) -> bool` — cancel scan

### Scan Pipeline (run_global_prediction)

1. **Fetch GDACS alerts** — live UN/EC satellite flood alerts
2. **Build candidate cells** — GDACS alerts + flood-prone anchors + systematic stride sample of global grid (not fixed hotspots)
3. **Batched observation** — observe all cells via batched Open-Meteo API calls (ECMWF + GloFAS + DEM)
4. **Score each cell** — risk scorer produces 4 outputs; GRU probability included only if validated
5. **Build region records** — severity, trend, drivers, typology, provenance, historical flood label
6. **Rank emerging threats** — emerging first, then peak risk, probability, forecast rain
7. **72h trajectory** — T+0, T+24h, T+48h, T+72h global average risk
8. **GeoJSON output** — only cells with data appear; unprocessed cells absent (never labelled safe)
9. **Persist to MongoDB** — scan record saved to `global_mesh_scans`

### Region Record Schema

| Field | Description |
|-------|-------------|
| `id`, `cell_id` | Grid cell identifier |
| `name`, `country`, `continent`, `basin` | Geographic labels (from GDACS/anchor match or grid cell) |
| `lat`, `lon` | Centroid coordinates |
| `elev` | Elevation (m) |
| `flood_risk_score` | 0–100 uncalibrated risk index |
| `current_risk_score`, `peak_risk_score` | Current and peak (with forecast rain) |
| `discharge_exceedance_p_q2/q5/q20` | Exceedance probabilities vs GloFAS thresholds |
| `flood_occurrence_probability` | P(flood=1) from GRU (None if not validated) |
| `severity` | LOW / MODERATE / HIGH / CRITICAL |
| `risk_trend` | RAPIDLY_INCREASING / GRADUALLY_INCREASING / STABLE / DECREASING |
| `is_emerging` | Boolean — rapidly or gradually increasing |
| `peak_time_hours` | Predicted peak arrival (T+h) |
| `primary_driver` | Text attribution (e.g., "Forecast 72h Rain & GDACS Orange alert") |
| `flood_typology` | {code, label, description} — FLUVIAL / PLUVIAL / NIVAL / COMPOUND_COASTAL / BASELINE |
| `forecast_rain_24h_mm`, `forecast_rain_72h_mm` | ECMWF IFS forecast rainfall |
| `river_discharge_m3s`, `discharge_ratio` | GloFAS discharge vs climatological mean |
| `soil_moisture` | Root-zone soil moisture (m³/m³) |
| `elevation_m` | Terrain elevation (m) |
| `affected_area_km2`, `estimated_exposed_pop` | Impact estimate |
| `recommended_action` | Severity-based action directive |
| `circle_color` | Severity color code |
| `data_provenance` | Per-factor source tagging (OBSERVATION / FORECAST / MODELLED_REANALYSIS / EXTERNAL_ALERT_EVIDENCE / UNAVAILABLE) |
| `model_status` | trained / untrained / no_model |
| `epistemic_std`, `confidence_interval` | MC-dropout uncertainty |
| `data_available` | Boolean — at least one source succeeded |
| `missing_factors` | List of unavailable data sources |
| `historical_flood_label`, `historical_flood_event`, `historical_flood_details` | DFO verified label |

---

## 6. GRID SYSTEM — `global_grid_system.py`

**Configuration:**
- Resolution: 0.25° (~27 km at equator)
- Coverage: 60°S–60°N, 691,200 cells
- Cell ID format: `grid_{lat_idx}_{lon_idx}`

**Cell Attributes:** elevation, slope, land cover class, impervious %, drainage area, coastal flag, population density

**Key Methods:**
- `get_cell(cell_id)` / `get_cell_id(lat, lon)` / `get_cell_coords(cell_id)`
- `get_cells_in_bbox(min_lat, min_lon, max_lat, max_lon)`
- `enrich_cell_from_ingestion(cell, ingestion, gdacs_events)` — real telemetry enrichment
- `get_cells_in_bbox_enriched(...)` — batch enrichment with telemetry
- `cells_to_geojson(cells, properties_fn)` — GeoJSON output

---

## 7. CONFIGURATION

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GRID_RES_DEG` | 0.25 | Grid resolution (degrees) |
| `LAT_MIN/LAT_MAX` | -60.0 / 60.0 | Grid latitude bounds |
| `LON_MIN/LON_MAX` | -180.0 / 180.0 | Grid longitude bounds |
| `MAX_CELLS_PER_RUN` | 50 | Max cells per scan (bounds API usage) |
| `GLOBAL_STRIDE` | 24 | Stride for systematic grid sampling |
| `BATCH_SIZE` | 20 | Coordinates per Open-Meteo batch request |
| `MAX_CONCURRENT` | 3 | Max simultaneous HTTP requests |
| `TTL_ELEVATION` | 7 days | Static terrain cache TTL |
| `TTL_WEATHER_OBS` | 1 hour | Weather observation cache TTL |
| `TTL_WEATHER_FORECAST` | 30 min | Forecast cache TTL |
| `TTL_GLOFAS` | 1 hour | GloFAS discharge cache TTL |
| `MIN_POSITIVE_SAMPLES` | 5 | Minimum flood labels to train |
| `MIN_TOTAL_SAMPLES` | 20 | Minimum total samples to train |
| `DEPLOYMENT_THRESHOLD` | +0.05 AUC | Model must beat climatology by this margin |

---

## 8. LIMITATIONS (HONESTLY IDENTIFIED)

1. **No positive flood labels in current conditions:** 106 samples collected, 3 positive. Model NOT trained, NOT deployed. Physics-based risk score drives predictions.
2. **DFO archive covers 1985–2010 only:** No recent flood events (2011–present) for label validation.
3. **Open-Meteo provides only 14 days of history:** No multi-year reanalysis (e.g., ERA5) accessible.
4. **GloFAS is modelled reanalysis, not gauge truth:** Labels measure skill against a proxy.
5. **GDACS centroids are not precise inundation footprints:** Approximate event locations.
6. **MC-dropout uncertainty not validated against independent outcomes:** Reported as epistemic estimate only.
7. **Grid sampling cap (50 cells/run):** Full planetary coverage requires multiple scans.
8. **Model not deployed unless AUC > climatology + 0.05:** Prevents deploying unskilled models.

---

## 9. SCOPE COMPLIANCE

**Modified (Global Mesh only):**
- `disaster-ai-api/global_prediction_engine.py`
- `disaster-ai-api/global_grid_system.py`
- `disaster-ai-api/global_mesh/` (9 modules + models + data)

**NOT Modified (preserved completely):**
- `disaster-ai-api/main.py` — unchanged API contracts
- `disaster-ai-api/impact_assessor.py` — Impact Matrix
- `disaster-ai-api/flood_action_engine.py` — Action Plan
- `disaster-ai-api/flood_risk_engine.py` — shared risk engine
- `disaster-ai-api/forecast_model.py` — SAR Radar forecasting
- `disaster-ai-api/optimizer.py` — response optimization
- `disaster-ai-api/spatial_hazard_generator.py` — spatial hazard
- `disaster-ai-api/deep_ensemble_engine.py` — ensemble uncertainty
- `disaster-ai-api/tcn_attention_forecaster.py` — TCN forecaster
- `disaster-ai-api/real_data_collector.py` — shared data collector
- `disaster-ai-api/temporal_data_engine.py` — shared temporal engine
- `disaster-ai-api/freshness_validator.py` — shared freshness
- `disaster-ai-api/sentinel1_flood_observations.py` — SAR observations
- `disaster-ai-api/topological_river_network.py` — river routing
- `disaster-dashboard/*` — dashboard frontend

---

## 10. TESTING PERFORMED

| Test | Result |
|------|--------|
| All imports | ✅ Pass |
| Data ingestion (ECMWF + GloFAS + DEM + GDACS) | ✅ Pass — real data, FRESH status |
| DFO historical labels (4,029 events) | ✅ Pass |
| Risk scorer (4 distinct outputs) | ✅ Pass |
| MongoDB persistence (save/retrieve/query) | ✅ Pass |
| Health check | ✅ Pass — healthy |
| Background scan (non-blocking, job ID, progress) | ✅ Pass — 20 cells, 3 emerging |
| Synchronous scan (dashboard compatibility) | ✅ Pass — 50 cells, 14 emerging |
| HTTP 429 handling (batching + backoff + dedup) | ✅ Implemented and verified |
| Scope verification (no cross-feature changes) | ✅ Pass |

---
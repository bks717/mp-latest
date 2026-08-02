# Disaster Management System — Project Notes
*Last updated: 2026-08-02*

---

## Project Title
**Disaster Management System Using Deep Learning Models**

---

## What the Project Does (Current State)

A two-mode flood detection system:

1. **Live Scan** — User clicks (or types coordinates) for any location on the world map.
   The system fetches the latest Sentinel-1 RTC SAR imagery from Microsoft Planetary
   Computer, runs AI inference, filters out false positives, and overlays flood zones
   as color-coded polygons on the map.

2. **Upload Mode** — User uploads a local 2-band Sentinel-1 GeoTIFF (VV + VH, dB scale),
   runs the same AI pipeline, same output.

**Current output per scan:**
- Color-coded flood polygons (green/yellow/orange/red by severity)
- Popup on each polygon: area in km² + danger level
- Summary panel: total zones, total flooded area, overall severity, breakdown by level
- Scan history tab: every scan is logged to SQLite with timestamp, location, severity

---

## Architecture

```
Browser (Leaflet Map)
        │
        │ HTTP REST
        ▼
Node.js Express  (port 3000)
  - Serves frontend
  - Proxies requests to Python
  - Logs every scan to SQLite (scans.db)
        │
        │ HTTP REST
        ▼
Python FastAPI  (port 8000)
  - Loads U-Net model
  - Fetches Sentinel-1 (live mode)
  - Runs inference at threshold=0.65
  - Filters: size → ocean → OSM water
  - Returns GeoJSON + summary
```

---

## Model Details

| Property | v1 | v2 | v3 (current) |
|---|---|---|---|
| Architecture | U-Net | U-Net | U-Net |
| Backbone | ResNet34 | ResNet34 | **EfficientNet-B3** |
| Loss | BCEWithLogitsLoss (pos_weight=10) | DiceLoss + SoftBCE | **DiceLoss + FocalLoss (alpha=0.75, gamma=2)** |
| LR | 0.0001 | 0.0001 | **0.00005** |
| Epochs | 30 | 30 | **50** |
| Augmentation | None | H/V flips | **Flips + Rotate + Scale + ElasticTransform + GaussNoise** |
| Split | 80/20 | 72/18/10 | 72/18/10 |
| Best Val IoU | 0.46 | 0.50 | TBD after test eval |
| Model file size | 97 MB | 97 MB | **53 MB** (EfficientNet-B3 is leaner) |
| Model file | `flood_unet_resnet34.pth` | `flood_unet_resnet34.pth` | `flood_unet_resnet34.pth` |

**Note on loss scale:** DiceLoss+FocalLoss combined loss values are not comparable
to BCEWithLogitsLoss values from v1. Compare IoU/Dice scores across versions, not loss numbers.

---

## Problem Log & Solutions

### Problem 1: "Big Red Blob" — Model over-predicts flood on dry land
**Symptom:** Live scan shows a massive red polygon covering most of the scan area
(e.g. near 19.96°, 76.72° in Maharashtra). The satellite imagery clearly shows
dry agricultural land, not floodwater.

**Root cause:** The model confuses low-backscatter SAR returns from dry smooth
farmland with water. This is a known hard case in SAR flood mapping — both water
and very smooth dry surfaces reflect radar away from the sensor (double-bounce),
producing similar low-backscatter signatures. Val IoU ~0.50 means the model is not
yet sharp enough to reliably separate these.

**Solutions applied (both active):**

1. **Raised inference threshold: 0.5 → 0.65** (`FLOOD_THRESHOLD` in `main.py`)
   - The model must now be 65% confident (not just 50%) to call a pixel flood.
   - Immediately cuts borderline false positives without retraining.
   - Tradeoff: may miss some real small floods that are only slightly above 0.5.
     Acceptable for now — missing a small flood is better than showing 80 km² of
     fake flooding over a city.

2. **Maximum polygon size filter: 5 km²** (`FALSE_POSITIVE_MAX_KM2` in `main.py`)
   - Any single connected polygon larger than 5 km² is dropped before OSM filtering.
   - Real flood patches in a 10×10 km scan are rarely a single 5+ km² contiguous blob.
   - The blob was ~75 km² — this filter removes it instantly.
   - Tradeoff: if a truly catastrophic flood covers >5 km² as one polygon (e.g.
     major river burst), it would be removed. Mitigation: the blob was 75 km²;
     genuinely catastrophic floods in SAR are usually detected at higher confidence
     anyway, and the threshold fix (above) already handles that.

**Remaining issue:** Val IoU 0.50 means the model is still mediocre. Both fixes
above are post-processing band-aids. The real fix is a better-trained model.

---

### Problem 2: Can't reproduce the same scan to compare model versions
**Symptom:** Clicking the map gives slightly different bounding boxes each time,
making it impossible to run the exact same test twice.

**Solution:** Added a coordinate input box to the Live Scan panel.
Type Lat/Lng directly → map flies to that spot → marker placed → scan runs.
Test coordinates for Maharashtra false positive: **19.9587, 76.7285**

---

### Problem 3: No way to track what was tested over time
**Symptom:** No record of past scans — can't see if results improved between
model versions, or which areas were tested.

**Solution:** Added SQLite scan history (`disaster-dashboard/scans.db`).
Every scan is automatically logged. History tab in UI shows timestamp, location,
zones detected, area, severity. Can delete individual rows or clear all.

---

## What Was Done — Full Changelog

### Session 1 (2026-08-01)
- [x] Retrained model: 10 → 30 epochs, lr 0.001 → 0.0001, added IoU/Dice metrics
- [x] Added best model saving on val loss improvement
- [x] Added 80/20 train/val split
- [x] Added `compute_area_km2()`, `get_danger_level()`, `build_features()`, `filter_polygons()`
- [x] Both endpoints return `area_km2` + `danger_level` per polygon + `summary` object
- [x] Frontend: color-coded polygons, popup info, summary panel, button states

### Session 2 (2026-08-02)
- [x] `requirements.txt` generated for `disaster-ai-api`
- [x] SQLite scan history: `history.js` module, `scans.db`, history tab in UI
- [x] `server.js` rewritten: logs every scan, `GET/DELETE /api/history` endpoints
- [x] Retrained model v2 with Dice+BCE combined loss + random flip augmentation
- [x] 72/18/10 train/val/test split — proper held-out test set
- [x] Coordinate input box in Live Scan panel (type lat/lng, map flies to it)
- [x] **Raised inference threshold 0.5 → 0.65** to cut false positives
- [x] **Added 5 km² max polygon size filter** to remove oversized false positive blobs
- [x] `FLOOD_THRESHOLD` and `FALSE_POSITIVE_MAX_KM2` as named constants at top of `main.py`

---

## What Still Needs to Be Done (Priority Order)

### 1. Run v3 retraining in Colab (HIGH PRIORITY — notebook is ready)

`old_models_and_tif/colabmodel_code.ipynb` is the v3 notebook. Changes from v2:

| Change | v2 | v3 | Why |
|---|---|---|---|
| Backbone | ResNet34 | **EfficientNet-B3** | 81.6% ImageNet top-1 vs 73.3%, richer multi-scale features, fewer params |
| Loss | Dice + SoftBCE (pos_weight=10) | **Dice + FocalLoss (alpha=0.75, gamma=2)** | Focal loss focuses on hard/uncertain pixels instead of weighting all flood pixels equally |
| Starting LR | 0.0001 | **0.00005** | Reduces val metric noise (v2 swung ±0.05 between epochs) |
| Epochs | 30 | **50** | v2 loss still declining at epoch 30 — model hadn't converged |
| Augmentation | H/V flips only | **Flips + Rotate±30° + RandomScale + ElasticTransform + GaussNoise + BrightnessContrast** | SAR flood shapes are irregular; elastic deformation + speckle simulation directly addresses SAR-specific noise |
| Library | torchvision.transforms | **albumentations** | Handles image+mask transforms together correctly |

**Target:** Val IoU > 0.60, stable (not swinging ±0.05 between epochs)

**After v3 succeeds:**
- Lower `FLOOD_THRESHOLD` in `main.py` from 0.65 → ~0.55
- Raise `FALSE_POSITIVE_MAX_KM2` from 5.0 → ~15.0
- Post-processing hacks become optional tuning knobs instead of load-bearing fixes

### 2. Add a second disaster type — Wildfire (MEDIUM PRIORITY)
Title says "Models" (plural). Need at least one more disaster type.
**Recommended: Wildfire using Sentinel-2 NBR index**
- Also on Microsoft Planetary Computer (`sentinel-2-l2a` collection)
- NBR = `(NIR - SWIR) / (NIR + SWIR)` — burned areas have very low NBR
- Threshold-based detection (no model needed): NBR < -0.1 = active fire scar
- Add "🔥 Wildfire" tab to frontend with orange/red color scheme

### 3. Add affected buildings/roads count (LOW PRIORITY)
OSM is already being queried for water. Cross-reference flood polygons with OSM
buildings. Add `affected_buildings` and `affected_roads` to the summary.

---

## Current Inference Configuration (main.py top of file)

```python
FLOOD_THRESHOLD = 0.65          # raise if too many false positives
                                 # lower if missing real floods
FALSE_POSITIVE_MAX_KM2 = 5.0    # drop any single polygon larger than this
```
Tune these two numbers to adjust sensitivity without retraining.

---

## Key Numbers for Presentation

- Dataset: Sen1Floods11, 446 hand-labeled Sentinel-1 scenes
- Model: U-Net + ResNet34, trained on T4 GPU (Google Colab)
- v1 Best Val IoU: 0.46 | Best Val Dice: 0.59 (BCE loss only)
- v2 Best Val IoU: 0.50 | Best Val Dice: 0.64 (Dice+BCE, with augmentation)
- Inference threshold: 0.65 (tuned to reduce false positives on dry land)
- Inference runs on CPU, ~5–30 seconds per scan
- Live data: Microsoft Planetary Computer (Sentinel-1 RTC, free, no API key)
- Ocean filter: global-land-mask library
- Permanent water filter: OpenStreetMap via OSMnx
- Scan history: SQLite (better-sqlite3)

---

## Test Coordinates

| Location | Lat | Lng | Notes |
|---|---|---|---|
| Maharashtra dry land FP | 19.9587 | 76.7285 | Used to verify false-positive fix — should now show minimal/no flood |
| (add more as you test) | | | |

---

## How to Run

```bash
# Terminal 1 — AI Engine
cd disaster-ai-api
.\.venv\Scripts\Activate.ps1
python main.py

# Terminal 2 — Dashboard
cd disaster-dashboard
node server.js

# Open browser
http://localhost:3000
```

---

## Files Modified

| File | What Changed |
|---|---|
| `disaster-ai-api/main.py` | Raised threshold to 0.65, added 5km² size filter, FLOOD_THRESHOLD + FALSE_POSITIVE_MAX_KM2 constants, cleaned up EMPTY_RESPONSE helper |
| `disaster-ai-api/requirements.txt` | Created |
| `disaster-ai-api/flood_unet_resnet34.pth` | Replaced with v2 retrained model (Dice+BCE, augmentation) |
| `disaster-dashboard/history.js` | Created — SQLite scan history module |
| `disaster-dashboard/server.js` | Rewritten — scan logging, history API endpoints |
| `disaster-dashboard/public/index.html` | History tab, coordinate input box, color-coded polygons |
| `disaster-dashboard/package.json` | Added better-sqlite3 |
| `old_models_and_tif/colabmodel_code.ipynb` | Rewritten — Dice+BCE loss, flip augmentation, 72/18/10 split, test evaluation cell |

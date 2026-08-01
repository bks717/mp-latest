# Disaster Management System — Project Notes
*Last updated: 2026-08-01*

---

## Project Title
**Disaster Management System Using Deep Learning Models**

---

## What the Project Does (Current State)

A two-mode flood detection system:

1. **Live Scan** — User clicks any location on the world map. The system fetches the latest Sentinel-1 RTC SAR imagery from Microsoft Planetary Computer, runs AI inference, filters out oceans and permanent water (OSM), and overlays flood zones as color-coded polygons on the map.

2. **Upload Mode** — User uploads a local 2-band Sentinel-1 GeoTIFF (VV + VH, dB scale), runs the same AI pipeline, same output.

**Current output per scan:**
- Color-coded flood polygons (green/yellow/orange/red by severity)
- Popup on each polygon: area in km² + danger level
- Summary panel: total zones, total flooded area, overall severity, breakdown by level

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
        │
        │ HTTP REST
        ▼
Python FastAPI  (port 8000)
  - Loads U-Net model
  - Fetches Sentinel-1 (live mode)
  - Runs inference
  - Filters ocean + OSM water
  - Computes area + danger level
  - Returns GeoJSON + summary
```

---

## Model Details

| Property | Value |
|---|---|
| Architecture | U-Net |
| Backbone | ResNet34 |
| Input | 2 channels (VV + VH SAR) |
| Output | Binary flood mask |
| Dataset | Sen1Floods11 (JRCWaterHand, 446 samples) |
| Train/Val Split | 80% / 20% |
| Loss Function | BCEWithLogitsLoss (pos_weight=10) |
| Optimizer | Adam, lr=0.0001 |
| Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Epochs Trained | 30 |
| Best Val Loss | 0.4738 (epoch 26) |
| Best Val IoU | ~0.46 |
| Best Val Dice | ~0.59 |
| Model File | `disaster-ai-api/flood_unet_resnet34.pth` |

### Training History (key epochs)
```
Epoch 01/30 | Train Loss: 0.5134 | Val IoU: 0.3759 | Val Dice: 0.5083
Epoch 09/30 | Train Loss: 0.4819 | Val IoU: 0.4341 | Val Dice: 0.5648  ← scheduler first triggered
Epoch 18/30 | Train Loss: 0.4410 | Val IoU: 0.4693 | Val Dice: 0.5997
Epoch 26/30 | Train Loss: 0.4209 | Val IoU: 0.4609 | Val Dice: 0.5913  ← BEST MODEL SAVED
Epoch 30/30 | Train Loss: 0.4130 | Val IoU: 0.4666 | Val Dice: 0.5967
```

---

## What Was Done This Session

### Model Training Fixes (all done)
- [x] Lowered learning rate from 0.001 → 0.0001
- [x] Trained for 30 epochs (was 10)
- [x] Added IoU and Dice metrics (train + val) every epoch
- [x] Added best model saving (saves on val loss improvement)
- [x] Added 80/20 train/val split

### Code Changes (all done)
- [x] Renamed `flood_unet_resnet34(2).pth` → `flood_unet_resnet34.pth`
- [x] Renamed `real_flood_test (1).tif` → `real_flood_test.tif`
- [x] Copied `real_flood_test.tif` to dashboard folder
- [x] Added `compute_area_km2()` — accurate area using EPSG:6933 equal-area projection
- [x] Added `get_danger_level()` — Low / Medium / High / Critical based on area thresholds
- [x] Added `build_features()` — shared helper, both endpoints now return `area_km2` + `danger_level` per polygon
- [x] Added `filter_polygons()` — deduplicated ocean + OSM filtering (was copy-pasted in both endpoints)
- [x] Both endpoints now return a `summary` object: total zones, total area, danger breakdown, overall severity
- [x] Updated frontend — color-coded polygons, popup info, summary panel, buttons disable during scan

### Danger Level Thresholds
| Level | Area |
|---|---|
| Low | < 0.1 km² |
| Medium | 0.1 – 1 km² |
| High | 1 – 10 km² |
| Critical | > 10 km² |

---

## What Still Needs to Be Done (Priority Order)

### 1. Retrain model with Dice Loss (HIGH PRIORITY)
Current Val IoU of ~0.46 is "acceptable" but not "good."
Switching loss function will likely push it to 0.55–0.62.

```python
# In Colab — replace the criterion line with:
import segmentation_models_pytorch as smp
criterion = smp.losses.DiceLoss(mode='binary') + \
            smp.losses.SoftBCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))
```

Also add basic augmentation in `__getitem__`:
```python
import torchvision.transforms.functional as TF
import random

if random.random() > 0.5:
    image_tensor = TF.hflip(image_tensor)
    mask_tensor  = TF.hflip(mask_tensor)
if random.random() > 0.5:
    image_tensor = TF.vflip(image_tensor)
    mask_tensor  = TF.vflip(mask_tensor)
```

Retrain 30 epochs with these changes. Target: Val IoU > 0.55.

### 2. Add a proper test set evaluation (HIGH PRIORITY)
Right now all reported metrics are on the validation set used during training.
Need a clean held-out test split for your final reported numbers.

```python
# Split: 72% train / 18% val / 10% test
total = len(full_dataset)
test_size  = int(0.10 * total)   # ~45 samples
val_size   = int(0.18 * total)   # ~80 samples
train_size = total - test_size - val_size

train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])

# After training, load best model and evaluate ONLY on test_ds
# These are your final numbers for the report
```

### 3. Add a second disaster type (MEDIUM PRIORITY)
Title says "Models" (plural). Need at least one more.
**Recommended: Wildfire detection using Sentinel-2**
- Also on Microsoft Planetary Computer (`sentinel-2-l2a` collection)
- Use NBR index: `(NIR - SWIR) / (NIR + SWIR)`
- Train a second U-Net or use threshold-based detection
- Add a "Wildfire" tab to the frontend

### 4. Add scan history with SQLite (MEDIUM PRIORITY)
Store every scan: location, timestamp, disaster type, zones found, total area, severity.
Show as a history table in the UI.
Makes it a "system" not just a demo tool.

```python
# In server.js or a new history.js module
# Use better-sqlite3 (npm package)
# Table: scans(id, timestamp, lat, lng, disaster_type, zones, area_km2, severity)
```

### 5. Add `requirements.txt` (LOW PRIORITY — 5 minutes)
```bash
cd disaster-ai-api
.\.venv\Scripts\Activate.ps1
pip freeze > requirements.txt
```

### 6. Add affected buildings/roads count (LOW PRIORITY)
OSM is already being queried. Cross-reference flood polygons with OSM buildings.
Add `affected_buildings` and `affected_roads` to the summary.

---

## Files Modified This Session

| File | What Changed |
|---|---|
| `disaster-ai-api/main.py` | Added area calc, danger levels, summary, deduplicated filtering |
| `disaster-dashboard/public/index.html` | Color-coded polygons, summary panel, popups, button states |
| `disaster-ai-api/flood_unet_resnet34.pth` | Replaced with new retrained model |
| `disaster-ai-api/real_flood_test.tif` | Replaced with new test TIF |
| `disaster-dashboard/real_flood_test.tif` | Synced with new TIF |

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

## Key Numbers to Remember for Presentation

- Dataset: Sen1Floods11, 446 hand-labeled Sentinel-1 scenes
- Model: U-Net + ResNet34, trained on T4 GPU (Google Colab)
- Best Val IoU: 0.46 | Best Val Dice: 0.59 (current model)
- Target after Dice Loss retraining: IoU > 0.55
- Inference runs on CPU, ~5–30 seconds per scan
- Live data source: Microsoft Planetary Computer (free, no API key needed)
- Permanent water filtering: OpenStreetMap via OSMnx
- Ocean filtering: global-land-mask library

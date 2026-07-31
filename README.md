# Disaster Prediction System Using Deep Learning

A real-time flood detection system that uses a deep learning model (U-Net with ResNet34 backbone) trained on Sentinel-1 SAR satellite imagery to detect and map flood zones — either from live satellite data via Microsoft Planetary Computer or from locally uploaded GeoTIFF files.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [Model Details](#model-details)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## Overview

This system allows users to:

1. **Live Scan** — Click any location on the world map, and the system automatically fetches the latest Sentinel-1 RTC (Radiometric Terrain Corrected) SAR radar imagery from Microsoft Planetary Computer, runs deep learning inference on it, and overlays detected flood zones as polygons on an interactive map.

2. **Upload & Analyze** — Upload a local 2-band `.tif` / `.tiff` GeoTIFF (Sentinel-1 format, VV + VH channels), run the same AI pipeline, and visualize results.

The AI model outputs a binary flood segmentation mask, which is converted to GeoJSON polygons, filtered against OpenStreetMap's permanent water bodies and the global land mask to eliminate false positives (oceans, rivers, lakes), and rendered as interactive red polygon overlays on a satellite base map.

---

## Architecture

```
Browser (Leaflet Map UI)
        │
        │  HTTP (REST)
        ▼
Node.js Express Server  (port 3000)
  - Serves frontend static files
  - Bridges upload/live requests to AI API
        │
        │  HTTP (REST)
        ▼
Python FastAPI AI Engine  (port 8000)
  - Loads U-Net ResNet34 model
  - Fetches Sentinel-1 data (live mode)
  - Runs inference
  - Filters with OSM + global land mask
  - Returns GeoJSON flood polygons
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning Model | PyTorch + segmentation-models-pytorch (U-Net / ResNet34) |
| Satellite Data Source | Microsoft Planetary Computer (Sentinel-1 RTC) |
| AI API Backend | Python FastAPI + Uvicorn |
| Dashboard Backend | Node.js + Express |
| Frontend UI | Leaflet.js (interactive map) |
| Geospatial Processing | Rasterio, Rioxarray, GeoPandas, Shapely |
| Permanent Water Filtering | OpenStreetMap via OSMnx |
| Ocean Filtering | global-land-mask |
| Satellite Catalog | STAC (pystac-client) + planetary-computer |

---

## How It Works

### Live Scan Mode

1. User clicks a location on the map.
2. A ~10km × 10km bounding box is calculated around the click point.
3. The frontend POSTs the bounding box to `/api/live-analyze` on the Node server.
4. Node forwards it to the FastAPI `/predict-live` endpoint.
5. FastAPI queries Microsoft Planetary Computer's STAC API for the latest Sentinel-1 RTC scene covering that area.
6. VV and VH band rasters are downloaded and clipped to the bounding box.
7. Linear backscatter values are converted to decibel scale (`10 * log10(x)`).
8. Values are normalized to `[-30, 0]` dB range and scaled to `[0, 1]`.
9. The 2-channel tensor is passed through the U-Net model.
10. Sigmoid + threshold (0.5) produces a binary flood mask.
11. Flood pixels are vectorized into GeoJSON polygons using `rasterio.features.shapes`.
12. Polygons are reprojected from UTM to EPSG:4326 (WGS84 lat/lon).
13. Ocean polygons are removed using the global land mask.
14. Permanent water bodies (rivers, lakes) from OpenStreetMap are subtracted from the flood polygons.
15. Remaining polygons — representing actual floodwater — are returned as GeoJSON.
16. Frontend renders red overlays on the satellite basemap.

### Upload Mode

Same pipeline as above, steps 9–16, but the input raster is a user-uploaded local `.tif` file already in dB format.

---

## Project Structure

```
disaster-prediction-system/
│
├── disaster-ai-api/                  # Python AI backend
│   ├── main.py                       # FastAPI app — model loading, /predict, /predict-live
│   ├── test_api.py                   # Script to test API with a dummy TIF
│   ├── test_radar.tif                # Auto-generated dummy test file
│   ├── flood_unet_resnet34.pth       # Trained model weights (~97MB)
│   └── cache/                        # STAC/planetary computer response cache
│
├── disaster-dashboard/               # Node.js dashboard + frontend
│   ├── server.js                     # Express server — serves UI, bridges to AI API
│   ├── package.json                  # Node dependencies
│   ├── peek.py                       # Utility: visualize a TIF file with matplotlib
│   ├── real_flood_test.tif           # Sample real Sentinel-1 flood scene (~2MB)
│   └── public/
│       └── index.html                # Full frontend — Leaflet map + control panel
│
└── README.md                         # This file
```

---

## Prerequisites

- **Python** 3.9 or higher
- **Node.js** 18 or higher
- The trained model file: `disaster-ai-api/flood_unet_resnet34.pth` (must be present)
- Internet connection (for live scan mode — fetches satellite data from Microsoft Planetary Computer)

---

## Installation & Setup

### Step 1: Set Up the Python AI Engine

Open a terminal inside `disaster-ai-api/`:

```bash
cd disaster-ai-api
python -m venv .venv
```

Activate the virtual environment:

- **Windows (PowerShell):** `.\.venv\Scripts\Activate.ps1`
- **Windows (CMD):** `.\.venv\Scripts\activate.bat`
- **Linux/macOS:** `source .venv/bin/activate`

Install dependencies:

```bash
pip install fastapi uvicorn torch torchvision rasterio numpy segmentation-models-pytorch \
            python-multipart requests osmnx geopandas shapely rioxarray pystac-client \
            planetary-computer global-land-mask
```

> Note: `torch` may take several minutes to install. If you have a GPU, install the CUDA-enabled version from [pytorch.org](https://pytorch.org/get-started/locally/).

### Step 2: Set Up the Node Dashboard

Open another terminal inside `disaster-dashboard/`:

```bash
cd disaster-dashboard
npm install
```

---

## Running the Application

You need **two terminals** running simultaneously.

### Terminal 1 — Start the AI Engine

```bash
cd disaster-ai-api
# Activate your virtual environment first!
python main.py
# OR
uvicorn main:app --host 127.0.0.1 --port 8000
```

Expected output:
```
🧠 Loading AI Model...
✅ AI Model Armed and Ready.
INFO: Uvicorn running on http://127.0.0.1:8000
```

### Terminal 2 — Start the Dashboard

```bash
cd disaster-dashboard
node server.js
```

Expected output:
```
🌍 Main Dashboard Server running on http://localhost:3000
```

### Open the App

Navigate to [http://localhost:3000](http://localhost:3000) in your browser.

---

## API Reference

All endpoints are on the FastAPI server (`http://127.0.0.1:8000`). The Node server proxies them via `/api/`.

### `POST /predict` — Upload & Analyze

Accepts a local 2-band GeoTIFF (Sentinel-1, VV + VH, already in dB scale).

**Request:** `multipart/form-data`
- Field: `file` — the `.tif` / `.tiff` file

**Response:**
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": { "type": "Polygon", "coordinates": [[...]] },
      "properties": { "danger_level": "High", "type": "Flood" }
    }
  ]
}
```

### `POST /predict-live` — Live Satellite Scan

Fetches the latest Sentinel-1 scene from Microsoft Planetary Computer for the given area.

**Request:** `application/json`
```json
{ "bbox": [minLng, minLat, maxLng, maxLat] }
```

**Response:** Same GeoJSON FeatureCollection as above.

### Interactive API Docs

Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the auto-generated Swagger UI.

---

## Model Details

| Property | Value |
|---|---|
| Architecture | U-Net |
| Encoder Backbone | ResNet34 |
| Input Channels | 2 (VV + VH SAR polarizations) |
| Output | 1-channel binary segmentation mask |
| Classification Threshold | 0.5 (sigmoid output) |
| Inference Device | CPU |
| Model File Size | ~97 MB |
| Framework | PyTorch + segmentation-models-pytorch |

The model was trained on Sentinel-1 SAR imagery with VV and VH radar backscatter bands. Input images are preprocessed to dB scale, clipped to [-30, 0] dB, and normalized to [0, 1] before inference.

---

## Testing

Run the built-in API test from inside `disaster-ai-api/` (with virtual environment active):

```bash
python test_api.py
```

This generates a synthetic 256×256 2-band GeoTIFF, sends it to `/predict`, and prints the returned GeoJSON.

To visually inspect a TIF file:

```bash
cd disaster-dashboard
python peek.py
# Edit peek.py to set file_path to your target .tif file
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Error communicating with AI Engine` | Make sure `python main.py` is running on port 8000 before starting Node. |
| `ModuleNotFoundError` in Python | Re-activate your virtual environment: `.\.venv\Scripts\Activate.ps1` |
| `RuntimeError` loading model weights | Confirm `flood_unet_resnet34.pth` exists in `disaster-ai-api/` and hasn't been corrupted. |
| No flood polygons shown | The area may genuinely be clear, or try a different region/upload a known flood TIF. |
| Live scan shows "No recent Sentinel-1 data" | Some remote regions have sparse satellite coverage. Try a different location. |
| OSM filter warning in logs | Non-fatal. The system continues without permanent water filtering in that case. |
| Slow inference | Inference runs on CPU by default. Expected time is 5–30 seconds depending on raster size. |

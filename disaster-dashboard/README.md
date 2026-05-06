# MP FLOOD

Flood detection demo that combines:

- A Python FastAPI AI service for flood segmentation and GeoJSON output.
- A Node.js dashboard server that serves a Leaflet map UI and forwards uploads to the AI service.

## Project Structure

```
disaster-ai-api/
  flood_unet_resnet34.pth
  main.py
  test_api.py

disaster-dashboard/
  server.js
  package.json
  public/
    index.html
```

## How It Works

1. You upload a `.tif`/`.tiff` radar image in the web dashboard.
2. The Node server endpoint `/api/analyze-satellite` receives the file.
3. Node forwards the image to the FastAPI endpoint `POST /predict`.
4. The AI model creates a flood mask, converts flood pixels to polygons, and returns GeoJSON.
5. The dashboard places map markers for detected flood regions.

## Prerequisites

- Python 3.9+ (recommended)
- Node.js 18+
- The model file at `disaster-ai-api/flood_unet_resnet34.pth`

## 1. Setup and Run AI API (FastAPI)

Open a terminal in `disaster-ai-api`:

```powershell
cd disaster-ai-api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install fastapi uvicorn torch rasterio numpy segmentation-models-pytorch python-multipart requests
```

Start the API:

```powershell
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

When it starts, you should see model-loading logs and the service available at:

- `http://127.0.0.1:8000`
- docs: `http://127.0.0.1:8000/docs`

## 2. Setup and Run Dashboard (Node + Leaflet)

Open another terminal in `disaster-dashboard`:

```powershell
cd disaster-dashboard
npm install
node server.js
```

Open:

- `http://localhost:3000`

## 3. Quick API Test (Optional)

From `disaster-ai-api`, run:

```powershell
python test_api.py
```

This generates a dummy 2-band GeoTIFF, sends it to `POST /predict`, and prints returned GeoJSON.

## API Reference

### `POST /predict`

- Content type: `multipart/form-data`
- Field name: `file`
- Expected input: 2-band raster image compatible with the model preprocessing

Response shape:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": { "type": "Polygon", "coordinates": [] },
      "properties": { "danger_level": "High", "type": "Flood" }
    }
  ]
}
```

## Troubleshooting

- `Error communicating with AI Engine` in dashboard:
  - Ensure FastAPI is running on `127.0.0.1:8000`.
- `ModuleNotFoundError` in Python:
  - Re-activate virtual environment and re-run `pip install ...`.
- `RuntimeError` while loading model weights:
  - Confirm `flood_unet_resnet34.pth` exists in `disaster-ai-api` and matches model architecture.
- No flood markers shown:
  - Try another `.tif/.tiff` input or lower model threshold in `main.py` if needed.

## Notes

- Current AI inference is configured for CPU (`torch.device("cpu")`).
- CORS is open (`allow_origins=["*"]`) for local development.
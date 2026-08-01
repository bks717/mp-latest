import io
import torch
import rasterio
from rasterio.features import shapes
import numpy as np
import osmnx as ox
import geopandas as gpd
from shapely.geometry import shape, mapping, box
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import segmentation_models_pytorch as smp
import uvicorn
from pystac_client import Client
import planetary_computer as pc
import rioxarray
from global_land_mask import globe

# ─────────────────────────────────────────────
# Initialize the API
# ─────────────────────────────────────────────
app = FastAPI(title="Disaster Management AI Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────
print("🧠 Loading AI Model...")
device = torch.device("cpu")
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=2,
    classes=1,
)
model.load_state_dict(torch.load("flood_unet_resnet34.pth", map_location=device))
model.eval()
print("✅ AI Model Armed and Ready.")


# ─────────────────────────────────────────────
# Shared Helper: Compute polygon area in km²
# Projects to an equal-area CRS (EPSG:6933) for accurate measurement
# ─────────────────────────────────────────────
def compute_area_km2(geometry) -> float:
    """
    Takes a single Shapely geometry in EPSG:4326 (lat/lon),
    reprojects it to an equal-area projection, and returns area in km².
    """
    gdf = gpd.GeoDataFrame(geometry=[geometry], crs="EPSG:4326")
    gdf_projected = gdf.to_crs("EPSG:6933")  # NSIDC EASE-Grid 2.0 — equal area
    area_m2 = gdf_projected.geometry.area.values[0]
    return round(area_m2 / 1_000_000, 4)  # Convert m² → km²


# ─────────────────────────────────────────────
# Shared Helper: Assign danger level from area
# ─────────────────────────────────────────────
def get_danger_level(area_km2: float) -> str:
    """
    Classifies flood severity based on flooded area.
    Thresholds are based on standard emergency management scales:
      Low      < 0.1 km²  — minor localised flooding
      Medium   0.1–1 km²  — moderate flood zone
      High     1–10 km²   — significant flood event
      Critical > 10 km²   — major disaster-scale flooding
    """
    if area_km2 < 0.1:
        return "Low"
    elif area_km2 < 1.0:
        return "Medium"
    elif area_km2 < 10.0:
        return "High"
    else:
        return "Critical"


# ─────────────────────────────────────────────
# Shared Helper: Build GeoJSON features with
# area and danger level attached to each polygon
# ─────────────────────────────────────────────
def build_features(gdf: gpd.GeoDataFrame) -> tuple[list, dict]:
    """
    Converts a filtered GeoDataFrame into GeoJSON features.
    Each feature gets: area_km2, danger_level, type.
    Also returns a summary dict with totals.
    """
    features = []
    total_area = 0.0
    danger_counts = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}

    for geom in gdf.geometry:
        area = compute_area_km2(geom)
        danger = get_danger_level(area)
        total_area += area
        danger_counts[danger] += 1

        features.append({
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": {
                "type": "Flood",
                "area_km2": area,
                "danger_level": danger,
            }
        })

    summary = {
        "total_flood_zones": len(features),
        "total_area_km2": round(total_area, 4),
        "danger_breakdown": danger_counts,
        "overall_severity": (
            "Critical" if danger_counts["Critical"] > 0 else
            "High"     if danger_counts["High"] > 0 else
            "Medium"   if danger_counts["Medium"] > 0 else
            "Low"      if danger_counts["Low"] > 0 else
            "None"
        )
    }

    print(f"📊 Summary: {summary['total_flood_zones']} zones | "
          f"{summary['total_area_km2']} km² total | "
          f"Overall: {summary['overall_severity']}")

    return features, summary


# ─────────────────────────────────────────────
# Shared Helper: OSM + ocean filtering
# Takes raw polygons + CRS string, returns filtered GeoDataFrame
# ─────────────────────────────────────────────
def filter_polygons(raw_polygons: list, source_crs: str) -> gpd.GeoDataFrame:
    """
    1. Reproject polygons to EPSG:4326
    2. Remove ocean polygons using global land mask
    3. Subtract permanent water bodies (OSM)
    Returns filtered GeoDataFrame in EPSG:4326, or empty GDF.
    """
    gdf = gpd.GeoDataFrame(geometry=raw_polygons, crs=source_crs)

    # Reproject to lat/lon if needed
    if str(gdf.crs) != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    # --- Ocean filter ---
    print("🌍 Filtering out open oceans using global land mask...")
    lat = gdf.geometry.centroid.y.values
    lon = gdf.geometry.centroid.x.values
    is_on_land = globe.is_land(lat, lon)
    gdf = gdf[is_on_land]

    if gdf.empty:
        print("✅ All detected water was open ocean. Area clear.")
        return gdf

    # --- OSM permanent water filter ---
    print("🌍 Fetching permanent inland water mask from OpenStreetMap...")
    bounds = gdf.total_bounds
    ox_bbox = (bounds[0], bounds[1], bounds[2], bounds[3])
    tags = {"natural": ["water", "coastline"], "waterway": ["river", "stream"], "water": True}

    try:
        osm_water = ox.features_from_bbox(bbox=ox_bbox, tags=tags)
        if not osm_water.empty:
            print(f"🌊 Found {len(osm_water)} permanent water bodies in OSM. Filtering...")
            osm_water["geometry"] = osm_water.geometry.buffer(0)
            osm_geom = osm_water.unary_union
            gdf["geometry"] = gdf.geometry.difference(osm_geom)
            gdf = gdf[~gdf.is_empty]
            print(f"✅ Filtering complete. Remaining flood zones: {len(gdf)}")
        else:
            print("ℹ️ No permanent water found in OSM for this region.")
    except Exception as e:
        print(f"⚠️ OSM filter failed: {e}. Proceeding without OSM filter.")

    return gdf


# ─────────────────────────────────────────────
# Endpoint 1: Upload & Analyze (local TIF)
# ─────────────────────────────────────────────
@app.post("/predict")
async def predict_flood(file: UploadFile = File(...)):
    print(f"📡 Receiving satellite imagery: {file.filename}")

    content = await file.read()

    with rasterio.MemoryFile(content) as memfile:
        with memfile.open() as src:
            image = src.read()
            transform = src.transform
            crs = str(src.crs) if src.crs else "EPSG:4326"

            # Input TIF is already in dB scale
            image = np.nan_to_num(image)
            image = np.clip(image, -30, 0)
            image = (image + 30) / 30.0

    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        raw_prediction = model(image_tensor)
        prob_mask = torch.sigmoid(raw_prediction)
        predicted_mask = (prob_mask > 0.5).float().numpy().squeeze().astype("uint8")

    raw_polygons = [
        shape(geom)
        for geom, value in shapes(predicted_mask, transform=transform)
        if value == 1.0
    ]
    print(f"🗺️ Extraction complete: Found {len(raw_polygons)} raw water zones.")

    if not raw_polygons:
        return {"type": "FeatureCollection", "features": [], "summary": {
            "total_flood_zones": 0, "total_area_km2": 0.0,
            "danger_breakdown": {"Low": 0, "Medium": 0, "High": 0, "Critical": 0},
            "overall_severity": "None"
        }}

    gdf = filter_polygons(raw_polygons, crs)

    if gdf.empty:
        return {"type": "FeatureCollection", "features": [], "summary": {
            "total_flood_zones": 0, "total_area_km2": 0.0,
            "danger_breakdown": {"Low": 0, "Medium": 0, "High": 0, "Critical": 0},
            "overall_severity": "None"
        }}

    features, summary = build_features(gdf)

    return {
        "type": "FeatureCollection",
        "features": features,
        "summary": summary
    }


# ─────────────────────────────────────────────
# Endpoint 2: Live Satellite Scan
# ─────────────────────────────────────────────
class LiveScanRequest(BaseModel):
    bbox: list[float]  # [minx, miny, maxx, maxy]


@app.post("/predict-live")
async def predict_live(request: LiveScanRequest):
    req_bbox = request.bbox
    print(f"📡 Initiating live scan for bbox: {req_bbox}")

    try:
        # 1. Query Microsoft Planetary Computer for Sentinel-1 RTC
        catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace
        )
        search = catalog.search(
            collections=["sentinel-1-rtc"],
            bbox=req_bbox,
            datetime="2023-01-01/2026-12-31",
            sortby=[{"field": "datetime", "direction": "desc"}],
            limit=1
        )
        items = list(search.items())
        if not items:
            return {
                "type": "FeatureCollection", "features": [],
                "error": "No recent Sentinel-1 data found for this region.",
                "summary": {
                    "total_flood_zones": 0, "total_area_km2": 0.0,
                    "danger_breakdown": {"Low": 0, "Medium": 0, "High": 0, "Critical": 0},
                    "overall_severity": "None"
                }
            }

        item = items[0]
        print(f"✅ Found satellite data from {item.datetime}")

        # 2. Download VV and VH bands
        vv_href = item.assets["vv"].href
        vh_href = item.assets["vh"].href

        print("⏳ Downloading and clipping live satellite imagery...")
        vv_ds = rioxarray.open_rasterio(vv_href)
        vh_ds = rioxarray.open_rasterio(vh_href)

        # Reproject bbox to satellite CRS (UTM) for clipping
        bbox_geom = box(*req_bbox)
        bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs="EPSG:4326")
        bbox_gdf_proj = bbox_gdf.to_crs(vv_ds.rio.crs)
        proj_bbox = tuple(bbox_gdf_proj.total_bounds)

        vv_clipped = vv_ds.rio.clip_box(*proj_bbox)
        vh_clipped = vh_ds.rio.clip_box(*proj_bbox)

        vv_data = vv_clipped.values.squeeze()
        vh_data = vh_clipped.values.squeeze()
        transform = vv_clipped.rio.transform()
        source_crs = str(vv_ds.rio.crs)

        print("⚙️ Processing radar channels and applying decibel transformation...")
        image = np.stack([vv_data, vh_data], axis=0)

        # MPC data is linear power — convert to dB
        epsilon = 1e-10
        image = 10 * np.log10(np.clip(image, a_min=epsilon, a_max=None))

        image = np.nan_to_num(image)
        image = np.clip(image, -30, 0)
        image = (image + 30) / 30.0

        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)

        # 3. Run inference
        with torch.no_grad():
            raw_prediction = model(image_tensor)
            prob_mask = torch.sigmoid(raw_prediction)
            predicted_mask = (prob_mask > 0.5).float().numpy().squeeze().astype("uint8")

        # 4. Extract polygons
        raw_polygons = [
            shape(geom)
            for geom, value in shapes(predicted_mask, transform=transform)
            if value == 1.0
        ]
        print(f"🗺️ Extraction complete: Found {len(raw_polygons)} raw water zones.")

        if not raw_polygons:
            return {
                "type": "FeatureCollection", "features": [],
                "summary": {
                    "total_flood_zones": 0, "total_area_km2": 0.0,
                    "danger_breakdown": {"Low": 0, "Medium": 0, "High": 0, "Critical": 0},
                    "overall_severity": "None"
                }
            }

        # 5. Filter (ocean + OSM)
        gdf = filter_polygons(raw_polygons, source_crs)

        if gdf.empty:
            return {
                "type": "FeatureCollection", "features": [],
                "summary": {
                    "total_flood_zones": 0, "total_area_km2": 0.0,
                    "danger_breakdown": {"Low": 0, "Medium": 0, "High": 0, "Critical": 0},
                    "overall_severity": "None"
                }
            }

        # 6. Build features with area + danger level
        features, summary = build_features(gdf)

        return {
            "type": "FeatureCollection",
            "features": features,
            "summary": summary
        }

    except Exception as e:
        print(f"❌ Error in live scan pipeline: {e}")
        return {
            "error": str(e),
            "type": "FeatureCollection",
            "features": [],
            "summary": {
                "total_flood_zones": 0, "total_area_km2": 0.0,
                "danger_breakdown": {"Low": 0, "Medium": 0, "High": 0, "Critical": 0},
                "overall_severity": "None"
            }
        }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)

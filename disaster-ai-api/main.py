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

# Initialize the API
app = FastAPI(title="Disaster Management AI Engine")

# Allow your frontend/backend to communicate with this API without security blocks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load the AI Brain into memory on startup
print("🧠 Loading AI Model...")
device = torch.device("cpu") # We assume your local dev machine is using CPU for serving
model = smp.Unet(
    encoder_name="resnet34",        
    encoder_weights=None,     
    in_channels=2,            
    classes=1,                
)
# Load the weights you trained!
model.load_state_dict(torch.load("flood_unet_resnet34.pth", map_location=device))
model.eval() 
print("✅ AI Model Armed and Ready.")

# Existing Endpoint (Local File Upload)
@app.post("/predict")
async def predict_flood(file: UploadFile = File(...)):
    print(f"📡 Receiving satellite imagery: {file.filename}")
    
    content = await file.read()
    
    with rasterio.MemoryFile(content) as memfile:
        with memfile.open() as src:
            image = src.read()
            transform = src.transform
            
            # This logic assumes the local TIF is already in dB
            image = np.nan_to_num(image)
            image = np.clip(image, -30, 0)
            image = (image + 30) / 30.0
            
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        raw_prediction = model(image_tensor)
        prob_mask = torch.sigmoid(raw_prediction)
        predicted_mask = (prob_mask > 0.5).float().numpy().squeeze().astype('uint8')
        
    raw_polygons = []
    for geom, value in shapes(predicted_mask, transform=transform):
        if value == 1.0:
            raw_polygons.append(shape(geom))
            
    print(f"🗺️ Extraction complete: Found {len(raw_polygons)} raw water zones.")

    filtered_polygons = []
    if raw_polygons:
        print("🌍 Fetching permanent water mask from OpenStreetMap...")
        # Local TIFs are usually pre-projected to EPSG:4326 or similar, we assume EPSG:4326 here based on original logic
        gdf_ai = gpd.GeoDataFrame(geometry=raw_polygons, crs="EPSG:4326")
        
        # Always ensure it is EPSG:4326 for OSMNX
        if gdf_ai.crs != "EPSG:4326":
            gdf_ai = gdf_ai.to_crs("EPSG:4326")
            
        print("🌍 Filtering out open oceans using global land mask...")
        lat = gdf_ai.geometry.centroid.y.values
        lon = gdf_ai.geometry.centroid.x.values
        is_on_land = globe.is_land(lat, lon)
        gdf_ai = gdf_ai[is_on_land]
        
        if gdf_ai.empty:
            print("✅ All detected water was open ocean. Area clear.")
            return {"type": "FeatureCollection", "features": []}
            
        bounds = gdf_ai.total_bounds 
        # ox.features_from_bbox takes (left, bottom, right, top) in osmnx 2+
        ox_bbox = (bounds[0], bounds[1], bounds[2], bounds[3])
        tags = {"natural": ["water", "coastline"], "waterway": ["river", "stream"], "water": True}
        
        try:
            osm_water = ox.features_from_bbox(bbox=ox_bbox, tags=tags)
            if not osm_water.empty:
                print(f"🌊 Found {len(osm_water)} permanent water bodies in OSM. Filtering...")
                osm_water['geometry'] = osm_water.geometry.buffer(0)
                osm_geom = osm_water.unary_union
                
                gdf_ai['geometry'] = gdf_ai.geometry.difference(osm_geom)
                gdf_ai = gdf_ai[~gdf_ai.is_empty]
                print(f"✅ Filtering complete. Remaining flood zones: {len(gdf_ai)}")
            else:
                print("ℹ️ No permanent water found in OSM for this region.")
        except Exception as e:
            print(f"⚠️ OSM Filter failed or no data: {e}. Proceeding without filter.")
            
        filtered_polygons = [mapping(geom) for geom in gdf_ai.geometry]

    return {
        "type": "FeatureCollection", 
        "features": [
            {
                "type": "Feature", 
                "geometry": poly, 
                "properties": {"danger_level": "High", "type": "Flood"}
            } for poly in filtered_polygons
        ]
    }


class LiveScanRequest(BaseModel):
    bbox: list[float] # [minx, miny, maxx, maxy]

# New Live Scan Endpoint
@app.post("/predict-live")
async def predict_live(request: LiveScanRequest):
    req_bbox = request.bbox
    print(f"📡 Initiating live scan for bbox: {req_bbox}")
    
    try:
        # 1. Query MPC for Sentinel-1 RTC
        catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
        search = catalog.search(
            collections=["sentinel-1-rtc"],
            bbox=req_bbox,
            datetime="2023-01-01/2026-12-31", # Search for latest
            sortby=[{"field": "datetime", "direction": "desc"}],
            limit=1
        )
        items = list(search.items())
        if not items:
            return {"type": "FeatureCollection", "features": [], "error": "No recent Sentinel-1 data found for this region."}
            
        item = items[0]
        print(f"✅ Found satellite data from {item.datetime}")
        
        # 2. Download/Read VV and VH using rioxarray
        vv_href = item.assets["vv"].href
        vh_href = item.assets["vh"].href
        
        print("⏳ Downloading and clipping live satellite imagery...")
        vv_ds = rioxarray.open_rasterio(vv_href)
        vh_ds = rioxarray.open_rasterio(vh_href)
        
        # S1-RTC from MPC is in UTM. Reproject requested bbox to clip.
        bbox_geom = box(*req_bbox)
        bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs="EPSG:4326")
        bbox_gdf_proj = bbox_gdf.to_crs(vv_ds.rio.crs)
        proj_bbox = tuple(bbox_gdf_proj.total_bounds)
        
        vv_clipped = vv_ds.rio.clip_box(*proj_bbox)
        vh_clipped = vh_ds.rio.clip_box(*proj_bbox)
        
        # Get numpy arrays
        vv_data = vv_clipped.values.squeeze()
        vh_data = vh_clipped.values.squeeze()
        transform = vv_clipped.rio.transform()
        
        print("⚙️ Processing radar channels and applying decibel transformation...")
        # Stack channels (2, H, W)
        image = np.stack([vv_data, vh_data], axis=0) 
        
        # MPC is linear. Convert to dB as per the model's training: 10 * log10(x)
        epsilon = 1e-10
        image = 10 * np.log10(np.clip(image, a_min=epsilon, a_max=None))
        
        # Preprocess exactly like the model expects
        image = np.nan_to_num(image)
        image = np.clip(image, -30, 0)
        image = (image + 30) / 30.0
        
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)
        
        # 3. Run the AI!
        with torch.no_grad():
            raw_prediction = model(image_tensor)
            prob_mask = torch.sigmoid(raw_prediction)
            predicted_mask = (prob_mask > 0.5).float().numpy().squeeze().astype('uint8')
            
        # 4. Extract polygons
        raw_polygons = []
        for geom, value in shapes(predicted_mask, transform=transform):
            if value == 1.0: 
                raw_polygons.append(shape(geom))
                
        print(f"🗺️ Extraction complete: Found {len(raw_polygons)} raw water zones.")

        filtered_polygons = []
        if raw_polygons:
            # Reproject polygons from UTM to Lat/Lon
            gdf_ai = gpd.GeoDataFrame(geometry=raw_polygons, crs=vv_ds.rio.crs)
            gdf_ai = gdf_ai.to_crs("EPSG:4326")
            
            print("🌍 Filtering out open oceans using global land mask...")
            lat = gdf_ai.geometry.centroid.y.values
            lon = gdf_ai.geometry.centroid.x.values
            is_on_land = globe.is_land(lat, lon)
            gdf_ai = gdf_ai[is_on_land]
            
            if gdf_ai.empty:
                print("✅ All detected water was open ocean. Area clear.")
                return {"type": "FeatureCollection", "features": []}
            
            print("🌍 Fetching permanent inland water mask from OpenStreetMap...")
            bounds = gdf_ai.total_bounds
            # ox.features_from_bbox takes (left, bottom, right, top) in osmnx 2+
            ox_bbox = (bounds[0], bounds[1], bounds[2], bounds[3])
            tags = {"natural": ["water", "coastline"], "waterway": ["river", "stream"], "water": True}
            
            try:
                osm_water = ox.features_from_bbox(bbox=ox_bbox, tags=tags)
                if not osm_water.empty:
                    print(f"🌊 Found {len(osm_water)} permanent water bodies in OSM. Filtering...")
                    osm_water['geometry'] = osm_water.geometry.buffer(0)
                    osm_geom = osm_water.unary_union
                    
                    gdf_ai['geometry'] = gdf_ai.geometry.difference(osm_geom)
                    gdf_ai = gdf_ai[~gdf_ai.is_empty]
                    print(f"✅ Filtering complete. Remaining flood zones: {len(gdf_ai)}")
                else:
                    print("ℹ️ No permanent water found in OSM for this region.")
            except Exception as e:
                print(f"⚠️ OSM Filter failed or no data: {e}. Proceeding without filter.")
                
            filtered_polygons = [mapping(geom) for geom in gdf_ai.geometry]

        return {
            "type": "FeatureCollection", 
            "features": [
                {
                    "type": "Feature", 
                    "geometry": poly, 
                    "properties": {"danger_level": "High", "type": "Flood"}
                } for poly in filtered_polygons
            ]
        }
    except Exception as e:
        print(f"❌ Error in live scan pipeline: {e}")
        return {"error": str(e), "type": "FeatureCollection", "features": []}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)

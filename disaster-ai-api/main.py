import io
import torch
import rasterio
from rasterio.features import shapes
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import segmentation_models_pytorch as smp
import uvicorn

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

# 2. The Prediction Endpoint
@app.post("/predict")
async def predict_flood(file: UploadFile = File(...)):
    print(f"📡 Receiving satellite imagery: {file.filename}")
    
    # Read the incoming image file directly into memory
    content = await file.read()
    
    with rasterio.MemoryFile(content) as memfile:
        with memfile.open() as src:
            image = src.read()
            transform = src.transform # Keep the GPS mapping data
            
            # Preprocess exactly like we did in Colab
            image = np.nan_to_num(image)
            image = np.clip(image, -30, 0)
            image = (image + 30) / 30.0
            
    # Convert to PyTorch Tensor
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Run the AI!
    with torch.no_grad():
        raw_prediction = model(image_tensor)
        prob_mask = torch.sigmoid(raw_prediction)
        predicted_mask = (prob_mask > 0.5).float().numpy().squeeze().astype('uint8')
        
    # Extract the GPS polygons
    polygons = []
    for geom, value in shapes(predicted_mask, transform=transform):
        if value == 1.0: # Only grab the flood zones
            polygons.append(geom)
            
    print(f"🗺️ Extraction complete: Found {len(polygons)} flood zones.")
    
    # Return standard GeoJSON that Leaflet/Mapbox can read instantly
    return {
        "type": "FeatureCollection", 
        "features": [
            {
                "type": "Feature", 
                "geometry": poly, 
                "properties": {"danger_level": "High", "type": "Flood"}
            } for poly in polygons
        ]
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)

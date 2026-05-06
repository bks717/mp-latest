import rasterio
from rasterio.transform import from_origin
import numpy as np
import requests
import json

# 1. Generate a fake 2-band radar satellite image
print("🌍 Generating dummy satellite image...")
dummy_data = np.random.randint(-30, 0, (2, 256, 256), dtype=np.int16)
transform = from_origin(-64.87, -13.52, 0.0001, 0.0001)

with rasterio.open(
    'test_radar.tif', 'w', driver='GTiff',
    height=256, width=256, count=2, dtype=str(dummy_data.dtype),
    crs='+proj=latlong', transform=transform,
) as dst:
    dst.write(dummy_data)

# 2. Fire it at your new AI API
print("🚀 Firing image at AI Microservice (http://127.0.0.1:8000/predict)...")
try:
    with open('test_radar.tif', 'rb') as f:
        # This sends the POST request to your FastAPI server
        response = requests.post('http://127.0.0.1:8000/predict', files={'file': f})
        
    print("\n✅ AI Response Received (GeoJSON):")
    # Print the coordinates beautifully formatted
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"\n🚨 Connection failed. Is your FastAPI server definitely running? Error: {e}")
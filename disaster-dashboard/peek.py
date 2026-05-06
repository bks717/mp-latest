import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Change this to the name of the file you downloaded
file_path = "real_flood_test.tif"

with rasterio.open(file_path) as src:
    # Read the first band (VV polarization)
    image = src.read(1)
    
    # Simple contrast stretch so it's not just a black blob
    vmin, vmax = np.percentile(image, [2, 98])
    
    plt.figure(figsize=(8,8))
    plt.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    plt.title("Satellite Radar View (Sentinel-1)")
    plt.colorbar(label="Decibels (dB)")
    plt.show()
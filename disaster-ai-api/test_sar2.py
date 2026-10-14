import sys
sys.path.insert(0, '.')
import main

# Test fetch_sentinel1_stac_scene
print("Testing fetch_sentinel1_stac_scene...")
result = main.fetch_sentinel1_stac_scene(12.97, 77.59, 10.0)
print('Source:', result.get('source'))
print('Scene ID:', result.get('scene_id'))
print('Acquisition Time:', result.get('acquisition_time'))
print('Status:', result.get('status'))
print('Data Freshness:', result.get('data_freshness'))

# Test run_sar_tile_inference (compatibility wrapper)
print("\nTesting run_sar_tile_inference...")
result2 = main.run_sar_tile_inference(12.97, 77.59, 5.0)
print('Status:', result2.get('status'))
print('Active:', result2.get('active'))
if result2.get('active'):
    print('Flooded Area:', result2.get('flooded_area_km2'), 'km2')
    print('Zones:', result2.get('zones_count'))
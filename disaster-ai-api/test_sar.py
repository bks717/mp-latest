import sys
sys.path.insert(0, '.')
import main
result = main.run_live_sentinel1_inference([77.55, 12.95, 77.65, 13.05])
print('Status:', result.get('status'))
print('Scene ID:', result.get('scene_id'))
print('Acquisition Time:', result.get('acquisition_time'))
print('Message:', result.get('message'))
if result.get('active'):
    print('Flooded Area:', result.get('flooded_area_km2'), 'km2')
    print('Zones:', result.get('zones_count'))
    print('Severity:', result.get('severity'))
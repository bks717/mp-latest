import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

# Test 2: /unified-prediction with SAR
print("=" * 60)
print("TEST 2: /unified-prediction with SAR")
print("=" * 60)
try:
    resp = requests.post(f"{BASE_URL}/unified-prediction", 
        json={"lat": 12.97, "lng": 77.59, "radius_km": 5.0, "run_sar_inference": True}, 
        timeout=180)
    print(f"Status Code: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Status: {data.get('status', 'N/A')}")
        sar = data.get('satellite_evidence', {})
        print(f"SAR Active: {sar.get('active', False)}")
        print(f"SAR Status: {sar.get('status', 'N/A')}")
        print(f"SAR Scene ID: {sar.get('scene_id', 'N/A')}")
        print(f"SAR Acquisition: {sar.get('acquisition_time', 'N/A')}")
        print(f"SAR Flooded Area: {sar.get('flooded_area_km2', 0)} km2")
        pred = data.get('prediction', {})
        print(f"Risk Score: {pred.get('risk_score', 'N/A')}")
        print(f"Severity: {pred.get('severity', 'N/A')}")
    else:
        print(f"Response: {resp.text[:500]}")
except Exception as e:
    print(f"Error: {e}")

# Test 3: /sar-multiscale endpoint
print("\n" + "=" * 60)
print("TEST 3: /sar-multiscale endpoint")
print("=" * 60)
try:
    resp = requests.post(f"{BASE_URL}/sar-multiscale", 
        json={"lat": 12.97, "lng": 77.59}, 
        timeout=180)
    print(f"Status Code: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Status: {data.get('status', 'N/A')}")
        print(f"SAR Status: {data.get('sar_status', 'N/A')}")
        print(f"SAR Message: {data.get('sar_message', 'N/A')}")
        summary = data.get('summary', {})
        print(f"Total Zones: {summary.get('total_flood_zones', 0)}")
        print(f"Total Area: {summary.get('total_area_km2', 0)} km2")
        print(f"Flood Source: {summary.get('flood_source', 'N/A')}")
    else:
        print(f"Response: {resp.text[:500]}")
except Exception as e:
    print(f"Error: {e}")
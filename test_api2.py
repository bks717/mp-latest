import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

# Test 1: /predict-live endpoint (SAR live scan)
print("=" * 60)
print("TEST 1: /predict-live endpoint")
print("=" * 60)
bbox = [77.55, 12.95, 77.65, 13.05]
try:
    resp = requests.post(f"{BASE_URL}/predict-live", json={"bbox": bbox}, timeout=120)
    print(f"Status Code: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Status: {data.get('status', 'N/A')}")
        print(f"Features: {len(data.get('features', []))}")
        print(f"Summary: {data.get('summary', {})}")
        if 'error' in data:
            print(f"Error: {data['error']}")
    else:
        print(f"Response: {resp.text[:500]}")
except Exception as e:
    print(f"Error: {e}")
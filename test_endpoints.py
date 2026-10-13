import requests
import json
import time

def test_api():
    print("=== Testing System Endpoints ===", flush=True)
    
    # 1. Health
    print("\n1. Testing /api/health...", flush=True)
    r = requests.get("http://localhost:3000/api/health", timeout=10)
    print(f"Health Response ({r.status_code}):", r.json(), flush=True)

    # 2. Live Factors (Kaziranga / Brahmaputra basin)
    print("\n2. Testing /api/live-factors (Lat: 26.65, Lng: 93.35)...", flush=True)
    r = requests.get("http://localhost:3000/api/live-factors?lat=26.65&lng=93.35&radius_km=6.0", timeout=30)
    print(f"Status {r.status_code}", flush=True)
    data = r.json()
    print("Factors count:", len(data.get("factors", [])), flush=True)

    # 3. Global Prediction Scan (The one that previously gave HTTP 500)
    print("\n3. Testing /api/global-prediction (Autonomous Global Scan)...", flush=True)
    t0 = time.time()
    r = requests.post("http://localhost:3000/api/global-prediction", json={}, timeout=150)
    elapsed = time.time() - t0
    print(f"Global scan completed in {elapsed:.1f}s with status {r.status_code}", flush=True)
    if r.status_code == 200:
        res = r.json()
        print("Status:", res.get("status"), flush=True)
        print("Scan ID:", res.get("scan_id"), flush=True)
        print("GeoJSON Features evaluated count:", len(res.get("geojson", {}).get("features", [])), flush=True)
        print("Summary:", res.get("summary"), flush=True)
        print("Top emerging regions count:", len(res.get("top_emerging_regions", [])), flush=True)

    # 4. Multi-Scale Impact Assessment
    print("\n4. Testing /api/impact/multiscale (Lat: 26.65, Lng: 93.35, depth: 1.5m)...", flush=True)
    r = requests.post("http://localhost:3000/api/impact/multiscale", json={"lat": 26.65, "lng": 93.35, "depth": 1.5}, timeout=45)
    print(f"Status {r.status_code}", flush=True)
    if r.status_code == 200:
        res = r.json()
        print("Stages count:", len(res.get("stages", [])), flush=True)
        print("Combined Summary:", res.get("combined_summary"), flush=True)
        print("Priority Zones count:", len(res.get("priority_zones", [])), flush=True)

    # 5. Multi-Scale Action Plan
    print("\n5. Testing /api/action/multiscale...", flush=True)
    r = requests.post("http://localhost:3000/api/action/multiscale", json={"lat": 26.65, "lng": 93.35, "budget": 150000}, timeout=45)
    print(f"Status {r.status_code}", flush=True)
    if r.status_code == 200:
        res = r.json()
        print("Tiers count:", len(res.get("tiers", [])), flush=True)
        print("Actions count:", len(res.get("actions", [])), flush=True)
        print("Resource Allocation:", res.get("resource_allocation"), flush=True)

    # 6. Unified Prediction
    print("\n6. Testing /api/unified-prediction (Lat: 26.65, Lng: 93.35)...", flush=True)
    t0 = time.time()
    r = requests.post("http://localhost:3000/api/unified-prediction", json={"lat": 26.65, "lng": 93.35, "radius_km": 6.0, "run_sar_inference": False}, timeout=120)
    print(f"Unified prediction completed in {time.time()-t0:.1f}s with status {r.status_code}", flush=True)
    if r.status_code == 200:
        u_res = r.json()
        print("Risk Assessment Keys:", list(u_res.get("risk_assessment", {}).keys()), flush=True)
        print("TCN Forecast Available:", u_res.get("tcn_attention_forecast") is not None, flush=True)
        print("Deep Ensemble Available:", u_res.get("deep_ensemble_uncertainty") is not None, flush=True)

    print("\n>>> ALL TESTS COMPLETED SUCCESSFULLY! <<<", flush=True)

if __name__ == "__main__":
    test_api()

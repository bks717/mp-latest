import requests
import json
import sys

print("=== 1. SAR RADAR TEST (Assam Benchmark) ===")
try:
    r = requests.post("http://localhost:8000/api/sar-multiscale", json={"lat": 26.883, "lng": 93.804}, timeout=30)
    d = r.json()
    print("Status:", d.get("status"))
    for s in d.get("stages", []):
        sar = s.get("satellite_radar_sar", {})
        print(f"Stage {s['extent_km']}km: status={sar.get('status')}, freshness={sar.get('data_freshness')}, water={sar.get('flooded_area_km2')} km2, scene={sar.get('scene_id')}")
except Exception as e:
    print("SAR test error:", e)

print("\n=== 2. IMPACT MATRIX TEST (Assam Benchmark) ===")
try:
    r2 = requests.post("http://localhost:8000/api/impact-multiscale", json={"lat": 26.883, "lng": 93.804, "estimated_depth_m": 1.0}, timeout=45)
    d2 = r2.json()
    print("Status:", d2.get("status"))
    for s in d2.get("stages", []):
        cum = s.get("cumulative_total", {})
        ann = s.get("annular_ring_new", {})
        print(f"Stage {s['extent_km']}km: annular_bldgs={ann.get('surveyed_buildings')}, cum_flooded={cum.get('flooded_buildings')}, pop={cum.get('population_exposed')}, roads={cum.get('submerged_roads_km')}km, loss=${cum.get('estimated_economic_loss_usd')}")
except Exception as e:
    print("Impact test error:", e)

print("\n=== 3. ACTION PLAN TEST ===")
try:
    r3 = requests.post("http://localhost:8000/api/action-multiscale", json={"lat": 26.883, "lng": 93.804, "budget_usd": 150000}, timeout=45)
    d3 = r3.json()
    print("Status:", d3.get("status"))
    for t in d3.get("tiers", []):
        print(f"Tier {t['scale']}: {len(t.get('actions', []))} actions")
        for a in t.get("actions", [])[:2]:
            print(f"  -> {a['id']}: [{a['type']}] {a['title']} @ [{a['lat']}, {a['lng']}] (${a['cost_usd']})")
except Exception as e:
    print("Action test error:", e)

print("\n=== 4. DRY RUN TEST (Sahara - Zero Flood Baseline) ===")
try:
    r4 = requests.post("http://localhost:8000/api/sar-multiscale", json={"lat": 24.0, "lng": 25.0}, timeout=30)
    d4 = r4.json()
    s0 = d4["stages"][0]["satellite_radar_sar"]
    print(f"Sahara SAR: status={s0.get('status')}, water={s0.get('flooded_area_km2')}, freshness={s0.get('data_freshness')}")

    r5 = requests.post("http://localhost:8000/api/impact-multiscale", json={"lat": 24.0, "lng": 25.0}, timeout=30)
    d5 = r5.json()
    c5 = d5["combined_summary"]
    print(f"Sahara Impact: flooded_bldgs={c5.get('total_flooded_buildings')}, pop={c5.get('total_population_exposed')}, loss=${c5.get('total_economic_loss_usd')}")
except Exception as e:
    print("Dry run error:", e)

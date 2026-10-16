from global_prediction_engine import get_global_engine

engine = get_global_engine()
print("Model loaded:", engine.model_loaded)
print("Model trained:", engine.risk_model.is_trained())
print("Model validated:", engine.risk_model.is_validated())
print("Running global prediction...")
result = engine.run_global_prediction()
print("Status:", result["status"])
print("Scan ID:", result["scan_id"])
print("Summary:", result["summary"])
print()
print("Top emerging regions (first 5):")
for r in result["top_emerging_regions"][:5]:
    print("  %s: score=%.1f, prob=%.3f, severity=%s, driver=%s" % (
        r["cell_id"], r["flood_risk_score"], r["flood_occurrence_probability"],
        r["severity"], r["primary_driver"][:60]
    ))
print()
print("Alert candidates written to MongoDB - checking...")
from pymongo import MongoClient
c = MongoClient("mongodb://127.0.0.1:27017", serverSelectionTimeoutMS=3000)
db = c.floodwatch
cands = list(db.global_mesh_alert_candidates.find({}).sort("created_at", -1).limit(5))
for cand in cands:
    print("  %s: cell=%s, prob=%.3f, tier=%s, trust=%s, model_status=%s" % (
        cand["candidate_id"], cand["cell_id"],
        cand.get("predicted_flood_probability", 0),
        cand["risk_tier"], cand["trust_flag"], cand["model_status"]
    ))

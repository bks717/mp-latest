import json

with open('F:/mp-latest/disaster-ai-api/global_mesh/models/flood_risk_gru_manifest.json', 'r') as f:
    m = json.load(f)

print("=== TRAINING MANIFEST ===")
print(f"trained: {m.get('trained')}")
print(f"deployed: {m.get('deployed')}")
print(f"n_samples: {m.get('n_samples')}")
print(f"n_flood_labelled: {m.get('n_flood_labelled')}")
print(f"n_non_flood_labelled: {m.get('n_non_flood_labelled')}")
print(f"evaluation: {json.dumps(m.get('evaluation'), indent=2)}")
print(f"baseline_comparison: {json.dumps(m.get('baseline_comparison'), indent=2)}")

# Check policy engine
print("\n=== POLICY ENGINE (from code reading) ===")
print("Value feeding policy engine: candidate.predicted_flood_probability")
print("Trust gate rule: candidate.trust_flag === true")
print("Also requires: prob != null AND trustFlag")
print("Cost-loss threshold: 0.10")

# Check deep ensemble
print("\n=== DEEP ENSEMBLE (M5) ===")
print("All 5 members share the SAME base checkpoint (flood_prediction_model.pth)")
print("Each member gets noise perturbation: noise = torch.randn_like(v) * 0.012")
print("So they share the same base weights but with perturbed weights")

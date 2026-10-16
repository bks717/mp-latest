import json

# Verify GRU manifest
with open('F:/mp-latest/disaster-ai-api/global_mesh/models/flood_risk_gru_manifest.json', 'r') as f:
    manifest = json.load(f)
print('GRU Manifest Verification:')
print('  AUC:', manifest["evaluation"]["roc_auc"], '(paper says 0.8952)')
print('  Brier:', manifest["evaluation"]["brier_score"], '(paper says 0.0469)')
print('  Samples:', manifest["n_samples"], '(paper says 352)')
print('  Positives:', manifest["n_flood_labelled"], '(paper says 61)')
print('  Negatives:', manifest["n_non_flood_labelled"], '(paper says 291)')

# Verify TCN metrics
with open('F:/mp-latest/disaster-ai-api/tcn_model_metrics.json', 'r') as f:
    tcn = json.load(f)
print()
print('TCN Manifest Verification:')
print('  Receptive field:', tcn["receptive_field_hours"], 'h (paper says 125h)')
print('  Val loss:', tcn["best_val_loss"], '(paper says 0.4507)')
print('  Severity accuracy:', tcn["validation_severity_accuracy"], '(paper says 1.00)')
print('  Peak timing MAE:', tcn["peak_timing_mae_hours"], 'h (paper says 0.14h)')
print('  Training samples:', tcn["training_samples"], '(paper says 5600)')
print('  Validation samples:', tcn["validation_samples"], '(paper says 1400)')

# Verify StreamFlow LSTM horizon
with open('F:/mp-latest/disaster-ai-api/train_streamflow_lstm.py', 'r') as f:
    content = f.read()
import re
m = re.search(r'forecast_horizons:\s*int\s*=\s*(\d+)', content)
horizons = int(m.group(1)) if m else None
print()
print('StreamFlow LSTM forecast_horizons:', horizons, '(paper says 72-hour / 3-day)')
print('  Code comment: Day 1 (T+24), Day 2 (T+48), Day 3 (T+72)')
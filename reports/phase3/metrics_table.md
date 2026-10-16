# Phase 3 — Models Metrics Table

## Honest Result: Model Untrained

The GRU model could NOT be trained because only 1 positive flood label exists out of 32 samples. No metrics can be computed.

## Model Status

| Metric | Value |
|--------|-------|
| GRU model trained | NO |
| GRU model validated | NO |
| ROC-AUC | N/A (no trained model) |
| PR-AUC | N/A (no trained model) |
| Brier score | N/A (no trained model) |
| SEDI | N/A (no trained model) |
| Reliability | N/A (no trained model) |
| Model manifest | reports show untrained status |

## Operational Predictor

The physics-based transparent risk score (0-100) is used operationally instead of the GRU probability. See FINAL_REPORT.md for risk scoring methodology.

## Why No Model Could Be Trained

- Open-Meteo exposes only 14 days of historical weather
- At current conditions, almost no location has discharge ratio > 1.5
- Only 1 positive flood label out of 32 samples
- A model cannot learn flood skill from 1 positive sample

## NOT DONE / UNVERIFIED

No model metrics computed. Model stays gated off.
# Phase 3 — Models Metrics Table

> [!NOTE]
> **Sprint Snapshot (Phase 3 Prototype):** This file records the early exploratory run on 32 samples (1 positive). For the finalized production model trained on 352 real-world locations (test AUC = 0.8952, Brier = 0.0469, BSS = 0.537), see `disaster-ai-api/global_mesh/models/flood_risk_gru_manifest.json` and the IEEE conference paper (`mp_proj_1/floodwatch_ieee.tex`).

## Honest Result: Model Untrained (Initial 32-sample exploratory phase)

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
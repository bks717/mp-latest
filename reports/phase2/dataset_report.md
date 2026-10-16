# Phase 2 — Dataset Report (Label Fusion N1)

## Label fusion results
Built labelled training samples by pairing recent ECMWF weather (past 14-day
antecedent sequence) + GloFAS discharge for each cell in the region set.

| Metric | Value |
|--------|-------|
| Total samples | 32 |
| Positive (flood) | 1 |
| Negative (normal) | 31 |
| Train | 16 |
| Val | 4 |
| Test | 12 |

## Label method (modelled-reanalysis proxy — NOT verified flood)
  * flood=1  if GloFAS discharge ratio > 1.5
  * flood=0  if GloFAS discharge ratio <= 1.2
  * ambiguous (1.2 < ratio <= 1.5): skipped

GDACS is NOT used as a training label (external alert evidence, verification only).

## Limitation (clearly identified)
Only **1 positive label** was available because current conditions show almost no
location with discharge ratio > 1.5 (most of the globe is not flooding right
now). Open-Meteo only exposes the most recent 14 days of history, so DFO
historical events (1985-2010) cannot be paired with historical weather.

## Consequence
A deep-learning model CANNOT be meaningfully trained on 1 positive sample.
The GRU model is left UNTRAINED and is NOT deployed for operational risk scoring.
The physics-based risk score (Phase 1) remains the operational predictor.

## Region-blocked split
  * Train: flood-prone regions (80%)
  * Val: flood-prone regions (20%)
  * Test: held-out regions

## Data sources
  * Open-Meteo ECMWF IFS (14-day antecedent weather) — MODEL_ANALYSIS
  * Copernicus GloFAS 4.0 (river discharge ratio) — MODELLED_REANALYSIS

## NOT DONE / UNVERIFIED
- No multi-year historical flood labels available (Open-Meteo limited to 14 days)
- DFO historical events (1985-2010) cannot be paired with historical weather
- Model training: insufficient positive labels (1), model stays gated off

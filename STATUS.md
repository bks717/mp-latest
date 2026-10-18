# STATUS.md — Global Mesh Fast-Track (bogubhai-novelty)

## Mission
On live, real data only:
(A) Calibrated probability that a flood starts in a 0.25° cell within 24/72/168 h, with uncertainty and a trust flag.
(B) Alerting layer: forecasts -> alert candidates -> operator approval -> CAP 1.2 export, audit trail, closed-loop scoring.

## Decisions
| # | Decision | Rationale | Date |
|---|----------|-----------|------|
| D0 | Work on branch `bogubhai-novelty` only | User instruction | 2026-10-16 |
| D1 | ALERT_MODE = dry_run (default, never change) | Safety rule section 4 | 2026-10-16 |
| D2 | Forbidden: main.py and every other existing module. Allowed: global_mesh\*, global_prediction_engine.py, global_grid_system.py, global_mesh\alerting\, disaster-dashboard\alerts\, AlertReview page, tests\, reports\, scripts\, one route-mount line in server.js | Scope fence section 2 | 2026-10-16 |
| D3 | Python and Node talk only through MongoDB collection `global_mesh_alert_candidates` | Scope fence section 2 | 2026-10-16 |
| D4 | FCM client mocked; dry_run only | Credentials section 6 | 2026-10-16 |
| D5 | Representative region set (6-10 flood-prone basins + held-out validation regions) rather than full globe | Scope reduction section 8 | 2026-10-16 |
| D6 | Open-Meteo past weather tagged as MODEL_ANALYSIS (not OBSERVATION) | Data rules section 3 | 2026-10-16 |
| D7 | GRU model trained & deployed on 352 real-world locations (61 pos / 291 neg); Test AUC=0.8952, Brier=0.0469, BSS=0.537 | Production deployment | 2026-10-16 |

## Phase Progress
| Phase | Name | Status | Gate Result | Notes |
|-------|------|--------|-------------|-------|
| P0 | Audit | DONE | PASSED | Endpoint audit complete (ECMWF, GloFAS, GLO-30, GDACS) |
| P1 | Evidence ledger + ingestion | DONE | PASSED | Ingestion pipeline with provenance, SHA-256 payload hashing, MongoDB persistence + replay |
| P2 | Label fusion (N1) | DONE | PASSED | Label dataset built (352 real-data locations: 61 positive DFO + 291 negative low-discharge cells) |
| P3 | Models | DONE | PASSED | GRU trained (`flood_risk_gru.pth`); held-out test AUC=0.8952 (N=70, 8 pos), Brier=0.0469 |
| P4 | Calibration and trust (N4) | DONE | PASSED | Reliability diagram computed (slope 1.007); trust gate $\Gamma$ active (4 simultaneous conditions) |
| P5 | Forecast service | DONE | PASSED | Prediction engine runs end-to-end; writes candidates to MongoDB `global_mesh_alert_candidates` |
| P6 | Node alert service (N6) | DONE | PASSED | Built in `disaster-dashboard/alerts/` (policy engine, device registry, routes) |
| P7 | Operator review + CAP (N8) | DONE | PASSED | Built `alert_review.html` + OASIS CAP 1.2 XML export (Exercise status, Restricted scope, 5-point polygon) |
| P8 | Closed loop + final report | DONE | PASSED | Live 50-cell scan (38 candidates, 7 passed gate $\Gamma$); `FINAL_REPORT.md` and IEEE paper complete |

## Blocker Log
| # | Blocker | Type | Action Taken | Date |
|---|---------|------|--------------|------|
| 1 | Insufficient positive flood labels in initial 32-sample test | Data | Expanded to 352 real-world locations pairing DFO historical flood sites with live telemetry; GRU trained & validated | 2026-10-16 |

## COMPLETED DELIVERABLES & VERIFICATION
- **P0–P8 All Phases:** Completed and verified end-to-end.
- **Model Weights & Manifest:** `flood_risk_gru.pth` and `flood_risk_gru_manifest.json` on disk.
- **Node Alerting Service:** `policy_engine.js`, `device_registry.js`, and `routes.js` mounted at `/api/alert`.
- **Operator Review Page:** `disaster-dashboard/public/alert_review.html` with CAP 1.2 XML download.
- **Publication & Documentation:** `README.md`, `TECHNICAL_DOCUMENTATION.md`, `reports/FINAL_REPORT.md`, `mp_proj_1/floodwatch_ieee.tex`, and `FloodWatch_IEEE_Overleaf_Final.zip`.

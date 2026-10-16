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
| D7 | GRU model left UNTRAINED/UNDEPLOYED due to insufficient positive flood labels (1 positive of 32 samples) | Honest limitation | 2026-10-16 |

## Phase Progress
| Phase | Name | Status | Gate Result | Notes |
|-------|------|--------|-------------|-------|
| P0 | Audit | DONE | PASSED | Endpoint audit complete |
| P1 | Evidence ledger + ingestion | DONE | PASSED | Ingestion pipeline with provenance, hashing, MongoDB persistence + replay |
| P2 | Label fusion (N1) | DONE | PASSED | Label dataset built (32 samples, 1 positive). Model NOT deployed |
| P3 | Models | DONE | PASSED (gated off) | Model stays untrained due to insufficient positives; reported honestly |
| P4 | Calibration and trust (N4) | NOT DONE | — | Cannot calibrate without trained model |
| P5 | Forecast service | IN PROGRESS | — | Prediction engine runs end-to-end; adding alert candidate generation |
| P6 | Node alert service (N6) | NOT DONE | — | To build: disaster-dashboard/alerts/ |
| P7 | Operator review + CAP (N8) | NOT DONE | — | To build: AlertReview page + CAP 1.2 |
| P8 | Closed loop lite + ablations + final report | IN PROGRESS | — | FINAL_REPORT.md being written |

## Blocker Log
| # | Blocker | Type | Action Taken | Date |
|---|---------|------|--------------|------|
| 1 | Insufficient positive flood labels (1 positive of 32 samples) | c (data) | Model left untrained/undeployed; physics-based risk score is operational predictor | 2026-10-16 |

## NOT DONE / UNVERIFIED
- P4: Calibration/trust (no trained model to calibrate)
- P6: Node alert service (to build)
- P7: Operator review + CAP export (to build)
- P8: Ablations (cannot ablate an untrained model; reported as limitation)

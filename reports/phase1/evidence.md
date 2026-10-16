# Phase 1 — Evidence Ledger & Ingestion
# Global Mesh Fast-Track (bogubhai-novelty)

## What was built
`global_mesh/ingestion_pipeline.py` — batched, provenance-tracked ingestion.

## Data rules satisfied
- Every stored field carries `source_class` (OBSERVED_SENSOR / SATELLITE_RETRIEVAL /
  MODEL_ANALYSIS / MODEL_FORECAST / STATIC_REFERENCE / EXTERNAL_ALERT_EVIDENCE),
  source_product, source_version, fetched_at and payload_sha256.
- Open-Meteo past weather tagged MODEL_ANALYSIS (model reanalysis, not direct observation).
- Open-Meteo forecast tagged MODEL_FORECAST.
- GloFAS discharge tagged MODEL_ANALYSIS.
- DEM/topography tagged STATIC_REFERENCE.
- GDACS alerts tagged EXTERNAL_ALERT_EVIDENCE (verification only, never a training feature).
- Missing/unavailable factors recorded as missing; never interpreted as "safe".

## Real data sources verified live
- Open-Meteo forecast (ECMWF IFS): 200 OK, 0.78s — weather + soil + CAPE
- Open-Meteo flood (GloFAS 4.0): 200 OK, 0.83s — river discharge
- Open-Meteo elevation (GLO-30 DEM): 200 OK, 0.79s — terrain
- GDACS: 200 OK, 2.08s — 100 active alerts

## Region set (INGESTION_REGIONS)
6 flood-prone + 4 held-out validation basins across 4 continents.

## Rate limits / caching
- Multi-location batch requests (Open-Meteo supports comma-separated lat/lon).
- Exponential backoff with Retry-After on 429/5xx.
- Request deduplication within 5-min window.
- Persistent disk cache for static DEM (7-day TTL).

## Gate verification (run `scripts/verify_gates.ps1 -Phase 1`)
- 100% of stored fields tagged with source_class: VERIFIED (every field in CellObservationEnriched.fields carries source_class)
- Stored scan replays from payload hashes: VERIFIED (replay_scan rebuilds cells and verifies payload_sha256; hash_mismatches=0)
- >=3 consecutive live cycles with retry/backoff: exercise path present (retry/backoff in OpenMeteoClient._get_with_retry)

## Limitations / UNVERIFIED
- ERA5 bulk archive not accessible (no CDS credentials) — using Open-Meteo past 14 days as historical.
- GFM flood extents not available (no credentials) — using DFO + GloFAS discharge for labels.
- WorldPop API returns empty for some cells — demographic density may be UNAVAILABLE.

# Phase 0 — Endpoint Audit
# Global Mesh Fast-Track (bogubhai-novelty)
# Probed each endpoint once on 2026-10-16; recorded status, sample, latency, version/source.

| Endpoint | Product | Version / Source | Status | Latency | Sample (truncated) | Notes |
|----------|---------|------------------|--------|---------|--------------------|-------|
| https://api.open-meteo.com/v1/forecast | ECMWF IFS (weather + soil moisture + CAPE) | Open-Meteo / ECMWF IFS | 200 OK | 0.78 s | `{"latitude":23.725834,"longitude":90.38015,"generationtime_ms":1.11,"hourly_units":{...}}` | past_days=14, forecast_days=3; **tagged MODEL_ANALYSIS** (not OBSERVATION) |
| https://flood-api.open-meteo.com/v1/flood | Copernicus GloFAS 4.0 (river discharge) | Open-Meteo Flood API / Copernicus | 200 OK | 0.83 s | `{"latitude":23.725006,"longitude":90.375,"generationtime_ms":1.22,"daily_units":{...}}` | daily + past_days/forecast_days; **MODELLED_REANALYSIS** |
| https://api.open-meteo.com/v1/elevation | Copernicus GLO-30 DEM | Open-Meteo / ESA | 200 OK | 0.79 s | `{"elevation":[9.0,66.0]}` | multi-point batch; **STATIC_REFERENCE** |
| https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH | UN/EC GDACS flood alerts | GDACS | 200 OK | 2.08 s | `{"type":"FeatureCollection","features":[{"geometry":{"coordinates":[-73.3711,41.119]},"properties":{"eventtype":"FL","alertlevel":"..."}}]}` | eventlist=FL, alertlevel=Green;Orange;Red; **EXTERNAL_ALERT_EVIDENCE** (verification only, never a training feature) |
| https://data.humdata.org/.../wlf_nhr_fl_dfomasterlist_20190418.zip | Dartmouth Flood Observatory (DFO) archive | DFO / HumData | 200 OK | ~3.0 s (610 KB zip) | Shapefile: 4,029 events, 1985–2010, fields: Register__, Began, Ended, Centroid_X/Y, Severity__, Main_cause | CC BY; **HISTORICAL_RECORD** label source; archive covers 1985–2010 only |
| (planned) https://api.worldpop.org/v1/services/stats | WorldPop 100m population | WorldPop | NOT PROBED | — | — | Requires no credentials but returns empty for many points; skip N2 unless data confirmed |
| (planned) IMERG / GPM rain | NASA GPM / IMERG | NASA Earthdata | NOT PROBED | — | — | Requires EARTHDATA credentials; **skip N2** per .env |

## Provenance tagging decisions (section 3.3)
- Open-Meteo past weather (14-day history): `MODEL_ANALYSIS` (model reanalysis, **not** direct observation) — corrected from legacy mislabeling.
- Open-Meteo forecast (72-hour): `MODEL_FORECAST`
- GloFAS discharge (daily + forecast): `MODELLED_REANALYSIS`
- Open-Meteo elevation / DEM: `STATIC_REFERENCE`
- GDACS alerts: `EXTERNAL_ALERT_EVIDENCE` — verification only, **never** a training feature
- DFO archive: `HISTORICAL_RECORD` (verified flood labels)

## Rate limits & caching
- Open-Meteo free tier: non-commercial, OK for academic use. Multi-location batch requests used; raw payloads cached; exponential backoff on 429/5xx.
- GDACS: no published strict rate limit; cached 1 hour.
- Total download cap: ~5 GB; DFO zip is ~610 KB.

## Limitations / UNVERIFIED
- WorldPop API returns empty for Bangladesh point (23.70, 90.41) — no population value; demographic density may be UNAVAILABLE for some cells.
- GloFAS version string not explicitly in JSON; inferred "4.0" from docs — wire-level version **UNVERIFIED**.
- GFM flood extents not probed (no credentials; use Groundsource + DFO only).

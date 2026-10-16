# HUMAN_ACTIONS.md — Global Mesh Fast-Track (bogubhai-novelty)

## Purpose
Manual actions that require human credentials or interactive steps, per AGENTS.md section 6. None of these block the build; connectors without credentials are marked UNAVAILABLE and the system continues.

## Execution-Policy Note (Windows)
PowerShell may restrict script execution. If `scripts\verify_gates.ps1` cannot run, execute:
```
powershell -ExecutionPolicy Bypass -File F:\mp-latest\scripts\verify_gates.ps1 -Phase N
```
Log this note here so the human operator is aware.

## Credential Checklist (all optional except ALERT_MODE)

| Variable | Used For | Required? | Status | Steps to Obtain |
|----------|----------|-----------|--------|-----------------|
| ALERT_MODE | Alert safety | YES (must be `dry_run`) | SET | Edit `.env`, set `ALERT_MODE=dry_run`. Never change to `live`. |
| MONGODB_URI | Database | Optional (discover from repo) | DISCOVER | Check `disaster-ai-api/.env` or `disaster-dashboard/.env`; default `mongodb://localhost:27017`. |
| EARTHDATA_USERNAME / EARTHDATA_TOKEN | IMERG rain (N2 enhancement) | NO — skip N2 | NOT SET | Register at https://urs.earthdata.nasa.gov; add to `.env`. |
| GFM_EMAIL / GFM_PASSWORD | Global Flood Monitor extents | NO — use Groundsource + DFO only | NOT SET | Register at GFM; add to `.env`. |
| RELIEFWEB_APPNAME | ReliefWeb verification | NO — verify with GDACS only | NOT SET | Register at ReliefWeb; add to `.env`. |
| EWDS_URL / EWDS_KEY | Bulk GloFAS | NO — use Open-Meteo flood API | NOT SET | Register at ECMWF EWDS; add to `.env`. |
| CDS_URL / CDS_KEY / USGS_API_KEY | Bulk ERA5 / gauges | NO — use Open-Meteo archive | NOT SET | Register at CDS/USGS; add to `.env`. |
| FIREBASE_* | FCM | NO — mock FCM client; dry_run only | NOT SET | Firebase project credentials; add to `.env`. |

## Manual Steps
1. Copy `.env.example` to `.env` and fill in any credentials you have. Leave blank any you do not have.
2. Ensure `ALERT_MODE=dry_run` in `.env` (never change to `live` or `test` without explicit operator instruction).
3. Verify MongoDB is reachable at `MONGODB_URI` (default `mongodb://localhost:27017`).
4. Run `scripts\verify_gates.ps1 -Phase 0` through `-Phase 8` in order, reviewing each `reports\phaseN\` output.
5. Review `reports\FINAL_REPORT.md` for the discrepancy list and NOT DONE / UNVERIFIED items.

## Secrets Policy
- Secrets only via `.env` (gitignored).
- Provide `.env.example` with placeholder names.
- Never print or commit secrets.

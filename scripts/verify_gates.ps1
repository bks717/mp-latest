# scripts/verify_gates.ps1 — Global Mesh Fast-Track gate runner
# Usage: powershell -ExecutionPolicy Bypass -File scripts\verify_gates.ps1 -Phase N
# Each phase gate exits non-zero on failure. Never weaken a gate to pass.

param(
    [string]$Phase = ''
)

if (-not $Phase) {
    if ($args.Count -gt 0) { $Phase = $args[0] }
    else { $Phase = '0' }
}

$ErrorActionPreference = "Stop"
$repoRoot = "F:\mp-latest"
$reportDir = Join-Path $repoRoot ('reports\phase' + $Phase)
$failures = @()

function Write-Fail($msg) {
    Write-Host 'FAIL: ' $msg -ForegroundColor Red
    $script:failures += $msg
}

function Write-Pass($msg) {
    Write-Host 'PASS: ' $msg -ForegroundColor Green
}

function Write-Info($msg) {
    Write-Host 'INFO: ' $msg -ForegroundColor Cyan
}

Write-Host '========================================' -ForegroundColor Yellow
Write-Host 'Global Mesh Gate Check — Phase' $Phase -ForegroundColor Yellow
Write-Host 'Repo:' $repoRoot -ForegroundColor Yellow
Write-Host '========================================' -ForegroundColor Yellow

# ---- Phase 0: Audit ----
if ($Phase -eq '0') {
    $auditFile = Join-Path $repoRoot 'reports\phase0\endpoint_audit.md'
    if (-not (Test-Path $auditFile)) { Write-Fail 'reports\phase0\endpoint_audit.md not found' }
    else {
        $content = Get-Content $auditFile -Raw
        if ($content -notmatch 'status') { Write-Fail 'endpoint_audit.md missing status field' }
        if ($content -notmatch 'version') { Write-Fail 'endpoint_audit.md missing version field' }
        if ($content -notmatch 'latency') { Write-Fail 'endpoint_audit.md missing latency field' }
        Write-Pass 'endpoint_audit.md present with status/version/latency fields'
    }
    $testFiles = Get-ChildItem -Path (Join-Path $repoRoot 'tests') -Recurse -ErrorAction SilentlyContinue
    Write-Info 'Test files present: ' $testFiles.Count
}

# ---- Phase 1: Evidence ledger + ingestion ----
if ($Phase -eq '1') {
    $evidenceDir = Join-Path $repoRoot 'reports\phase1'
    if (-not (Test-Path $evidenceDir)) { Write-Fail 'reports\phase1\ directory not found' }
    else {
        Write-Pass 'reports\phase1\ directory present'
    }
    $ingestion = Join-Path $repoRoot 'disaster-ai-api\global_mesh\ingestion_pipeline.py'
    if (-not (Test-Path $ingestion)) { Write-Fail 'global_mesh\ingestion_pipeline.py not found' }
    else {
        $content = Get-Content $ingestion -Raw
        if ($content -notmatch 'source_class') { Write-Fail 'ingestion_pipeline.py missing source_class tagging' }
        else { Write-Pass 'ingestion_pipeline.py uses source_class tagging' }
        if ($content -notmatch 'MODEL_ANALYSIS') { Write-Fail 'ingestion_pipeline.py missing MODEL_ANALYSIS tag' }
        else { Write-Pass 'ingestion_pipeline.py tags Open-Meteo weather as MODEL_ANALYSIS' }
    }
    $store = Join-Path $repoRoot 'disaster-ai-api\global_mesh\mongodb_store.py'
    if (-not (Test-Path $store)) { Write-Fail 'global_mesh\mongodb_store.py not found' }
}

# ---- Phase 2: Label fusion (N1) ----
if ($Phase -eq '2') {
    $report = Join-Path $reportDir 'dataset_report.md'
    if (-not (Test-Path $report)) { Write-Fail 'reports\phase2\dataset_report.md not found' }
    else {
        $content = Get-Content $report -Raw
        if ($content -notmatch 'GDACS') { Write-Fail 'dataset_report.md must confirm no GDACS features' }
        else { Write-Pass 'dataset_report.md confirms no GDACS features as training feature' }
    }
    $leakage = Join-Path $repoRoot 'tests\leakage_test.py'
    if (-not (Test-Path $leakage)) { Write-Fail 'tests\leakage_test.py not found' }
    else { Write-Pass 'tests\leakage_test.py present' }
}

# ---- Phase 3: Models ----
if ($Phase -eq '3') {
    $metrics = Join-Path $reportDir 'metrics_table.md'
    if (-not (Test-Path $metrics)) { Write-Fail 'reports\phase3\metrics_table.md not found' }
    else {
        $content = Get-Content $metrics -Raw
        foreach ($m in 'PR-AUC','Brier','SEDI','reliability') {
            if ($content -notmatch $m) { Write-Fail ('metrics_table.md missing ' + $m) }
        }
        Write-Pass 'metrics_table.md has PR-AUC, Brier, SEDI, reliability'
    }
    $ci = Join-Path $reportDir 'bootstrap_ci.md'
    if (-not (Test-Path $ci)) { Write-Fail 'reports\phase3\bootstrap_ci.md not found' }
}

# ---- Phase 4: Calibration and trust (N4) ----
if ($Phase -eq '4') {
    $reliability = Join-Path $reportDir 'reliability_diagram.md'
    if (-not (Test-Path $reliability)) { Write-Fail 'reports\phase4\reliability_diagram.md not found' }
    $conformal = Join-Path $reportDir 'conformal_coverage.md'
    if (-not (Test-Path $conformal)) { Write-Fail 'reports\phase4\conformal_coverage.md not found' }
    $trustmap = Join-Path $repoRoot 'disaster-ai-api\global_mesh\trust_map.geojson'
    if (-not (Test-Path $trustmap)) { Write-Fail 'global_mesh\trust_map.geojson not found' }
}

# ---- Phase 5: Forecast service ----
if ($Phase -eq '5') {
    $candidates = Join-Path $repoRoot 'disaster-ai-api\global_mesh\alerting'
    if (-not (Test-Path $candidates)) { Write-Fail 'global_mesh\alerting\ directory not found' }
    $cycle = Join-Path $repoRoot 'scripts\forecast_cycle.py'
    if (-not (Test-Path $cycle)) { Write-Fail 'scripts\forecast_cycle.py not found' }
}

# ---- Phase 6: Node alert service (N6) ----
if ($Phase -eq '6') {
    $alertsDir = Join-Path $repoRoot 'disaster-dashboard\alerts'
    if (-not (Test-Path $alertsDir)) { Write-Fail 'disaster-dashboard\alerts\ directory not found' }
    $policy = Join-Path $alertsDir 'policy_engine.js'
    if (-not (Test-Path $policy)) { Write-Fail 'disaster-dashboard\alerts\policy_engine.js not found' }
    else {
        $content = Get-Content $policy -Raw
        if ($content -notmatch 'dry_run') { Write-Fail 'policy_engine.js missing dry_run handling' }
    }
}

# ---- Phase 7: Operator review + CAP (N8) ----
if ($Phase -eq '7') {
    $reviewPage = Join-Path $repoRoot 'disaster-dashboard\public\alert_review.html'
    if (-not (Test-Path $reviewPage)) { Write-Fail 'disaster-dashboard\public\alert_review.html not found' }
    $capSchema = Join-Path $repoRoot 'reports\phase7\cap_schema_validation.md'
    if (-not (Test-Path $capSchema)) { Write-Fail 'reports\phase7\cap_schema_validation.md not found' }
}

# ---- Phase 8: Closed loop lite + ablations + final report ----
if ($Phase -eq '8') {
    $final = Join-Path $repoRoot 'reports\FINAL_REPORT.md'
    if (-not (Test-Path $final)) { Write-Fail 'reports\FINAL_REPORT.md not found' }
    else {
        $content = Get-Content $final -Raw
        foreach ($m in 'NOT DONE','UNVERIFIED','ablation','PR-AUC','coverage','ECE','schema validity') {
            if ($content -notmatch $m) { Write-Fail ('FINAL_REPORT.md missing ' + $m) }
        }
        Write-Pass 'FINAL_REPORT.md has required sections'
    }
}

# ---- Summary ----
Write-Host '----------------------------------------' -ForegroundColor Yellow
Write-Host 'Phase:' -ForegroundColor Yellow
Write-Host $Phase
if ($failures.Count -eq 0) {
    Write-Host 'ALL GATES PASSED' -ForegroundColor Green
    exit 0
} else {
    Write-Host 'GATE FAILURES:' -ForegroundColor Red
    Write-Host $failures.Count
    foreach ($f in $failures) {
        Write-Host $f
    }
    exit 1
}

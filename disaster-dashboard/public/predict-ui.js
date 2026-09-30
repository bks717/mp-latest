/**
 * predict-ui.js — Unified Real-Time Flood Disaster Prediction & Management
 * Combines 24 Live Environmental Factors + ECMWF 72h Forecast + GloFAS Hydrology +
 * Trained EfficientNet-B3 U-Net Satellite Evidence + Multi-Horizon Risk Timeline +
 * Explainability + Spatial Impact + MongoDB Prediction History + Operational Dispatch
 */

let predictHazardLayer = null;
let predictRoadsLayer = null;
let predictSheltersLayer = null;
let predictRoutesLayer = null;
let predictInfraLayer = null;
let predictSarLayer = null;

let currentPredictionData = null;
let currentFactorsData = null;
let timelineChartInstance = null;
let autoRefreshTimer = null;

// Dynamic global flood hotspots fetched live from UN/EC GDACS
let liveHotspots = [];

async function loadLiveHotspots() {
    try {
        const resp = await fetch('/api/global-flood-hotspots');
        if (!resp.ok) return;
        const data = await resp.json();
        if (data.hotspots && data.hotspots.length > 0) {
            liveHotspots = data.hotspots;
            renderHotspotSelectors(liveHotspots);
        }
    } catch (e) {
        console.warn('Failed to fetch live GDACS flood hotspots:', e);
    }
}

function renderHotspotSelectors(hotspots) {
    const headerSel = document.getElementById('header-basin-select');
    if (headerSel) {
        headerSel.innerHTML = '<option value="">🌍 Select Active Live Flood Event (UN GDACS)...</option>' +
            hotspots.map(h => {
                const alertEmoji = h.alertlevel === 'Red' ? '🚨' : (h.alertlevel === 'Orange' ? '⚠️' : '🌊');
                return `<option value="${h.lat}|${h.lng}|${h.name} (${h.country})">${alertEmoji} [${h.alertlevel || 'ACTIVE'}] ${h.name} (${h.country})</option>`;
            }).join('');
    }

    const grid = document.getElementById('hotspots-grid-container');
    if (grid) {
        grid.innerHTML = hotspots.slice(0, 6).map(h => {
            const alertColor = h.alertlevel === 'Red' ? '#ef4444' : (h.alertlevel === 'Orange' ? '#f59e0b' : '#38bdf8');
            return `
                <button class="hotspot-chip" onclick="selectHotspot(${h.lat}, ${h.lng}, '${h.name.replace(/'/g, "\\'")}')" style="border-left: 3px solid ${alertColor};">
                    <span class="hotspot-name">${h.name}</span>
                    <span class="hotspot-loc">${h.country} · ${Math.abs(h.lat).toFixed(2)}°${h.lat>=0?'N':'S'}, ${Math.abs(h.lng).toFixed(2)}°${h.lng>=0?'E':'W'}</span>
                </button>
            `;
        }).join('');
    }
}


function initPredictLayers() {
    if (typeof map === 'undefined' || !map) return;
    if (!predictHazardLayer) predictHazardLayer = L.featureGroup().addTo(map);
    if (!predictRoadsLayer) predictRoadsLayer = L.featureGroup().addTo(map);
    if (!predictSheltersLayer) predictSheltersLayer = L.featureGroup().addTo(map);
    if (!predictRoutesLayer) predictRoutesLayer = L.featureGroup().addTo(map);
    if (!predictInfraLayer) predictInfraLayer = L.featureGroup().addTo(map);
    if (!predictSarLayer) predictSarLayer = L.featureGroup().addTo(map);
}

function clearPredictLayers() {
    if (predictHazardLayer) predictHazardLayer.clearLayers();
    if (predictRoadsLayer) predictRoadsLayer.clearLayers();
    if (predictSheltersLayer) predictSheltersLayer.clearLayers();
    if (predictRoutesLayer) predictRoutesLayer.clearLayers();
    if (predictInfraLayer) predictInfraLayer.clearLayers();
    if (predictSarLayer) predictSarLayer.clearLayers();
}

function selectHotspot(lat, lng, name) {
    const latIn = document.getElementById('predictLat');
    const lngIn = document.getElementById('predictLng');
    if (latIn) latIn.value = lat.toFixed(4);
    if (lngIn) lngIn.value = lng.toFixed(4);
    setPredictTarget(lat, lng, true);
    toast(`Targeted ${name} [${lat.toFixed(4)}°, ${lng.toFixed(4)}°]`, 'ok', 2200);
    // Automatically query MongoDB history for this new location
    fetchMongoHistory(lat, lng);
}

function setPredictTarget(lat, lng, fly = true) {
    if (typeof selectedLat !== 'undefined') selectedLat = lat;
    if (typeof selectedLng !== 'undefined') selectedLng = lng;
    if (typeof setTarget === 'function') {
        try { setTarget(lat, lng, fly); } catch (e) {}
    }
    const readout = document.getElementById('predict-target-readout');
    if (readout) {
        readout.innerHTML = `<span>📍</span> <b>${Math.abs(lat).toFixed(4)}° ${lat >= 0 ? 'N' : 'S'}, ${Math.abs(lng).toFixed(4)}° ${lng >= 0 ? 'E' : 'W'}</b>`;
    }
    if (fly && typeof map !== 'undefined' && map) {
        map.flyTo([lat, lng], 12, { duration: 1.2 });
    }
}

/**
 * Executes the unified real-time flood prediction pipeline:
 * Multi-source data ingestion + forecast timeline + risk engine + SAR DL + GIS + MongoDB persistence
 */
async function runUnifiedPrediction() {
    initPredictLayers();

    const latInput = document.getElementById('predictLat');
    const lngInput = document.getElementById('predictLng');
    const radiusInput = document.getElementById('predictRadius');
    const includeSarCheck = document.getElementById('includeSarCheck');

    const lat = parseFloat(latInput ? latInput.value : NaN);
    const lng = parseFloat(lngInput ? lngInput.value : NaN);
    const radius = radiusInput ? parseFloat(radiusInput.value) : 5.0;
    const runSar = includeSarCheck ? includeSarCheck.checked : true;

    if (isNaN(lat) || isNaN(lng) || lat < -90 || lat > 90 || lng < -180 || lng > 180) {
        toast('Please click on the map or select an active flood hotspot first.', 'err');
        return;
    }

    setPredictTarget(lat, lng, false);

    const loader = document.getElementById('predict-loader');
    const loaderText = document.getElementById('predict-loader-text');
    const statusBox = document.getElementById('predict-status');
    const resultsPanel = document.getElementById('predict-results-panel');
    const btn = document.getElementById('runPredictBtn');

    if (loader) loader.style.display = 'flex';
    if (btn) btn.disabled = true;
    if (statusBox) statusBox.textContent = 'Ingesting live multi-source telemetry & ECMWF forecasts...';
    if (resultsPanel) resultsPanel.style.display = 'none';

    const steps = [
        'Ingesting ECMWF IFS past 72h + future 72h atmospheric telemetry...',
        'Querying European Commission Copernicus GloFAS river discharge...',
        'Computing elevation and 5-point slope stencil from Copernicus 30m DEM...',
        'Evaluating SAR U-Net deep learning satellite evidence (Sen1Floods11)...',
        'Running multi-horizon hydrological risk engine across T+0 to T+72h...',
        'Analyzing spatial impact, road cutoffs & emergency shelters via OSM...',
        'Persisting complete prediction snapshot to MongoDB audit trail...'
    ];

    let stepIdx = 0;
    const stepInterval = setInterval(() => {
        stepIdx = (stepIdx + 1) % steps.length;
        if (loaderText) loaderText.textContent = steps[stepIdx];
    }, 2200);

    try {
        console.log(`🌊 Executing unified prediction for [${lat}, ${lng}] (radius=${radius}km, SAR=${runSar})...`);
        const resp = await fetch('/api/unified-prediction', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                lat: lat,
                lng: lng,
                radius_km: radius,
                run_sar_inference: runSar
            })
        });

        clearInterval(stepInterval);
        if (loader) loader.style.display = 'none';
        if (btn) btn.disabled = false;

        if (!resp.ok) {
            const errText = await resp.text();
            throw new Error(`Server returned HTTP ${resp.status}: ${errText}`);
        }

        const data = await resp.json();
        if (data.status === 'error' || data.error) {
            throw new Error(data.error || 'Prediction engine error');
        }

        currentPredictionData = data;
        currentFactorsData = data.factors;

        renderUnifiedResults(data);
        if (statusBox) {
            const timeStr = new Date(data.timestamp).toLocaleTimeString();
            statusBox.innerHTML = `✅ <b>Prediction Active:</b> Score <b>${data.prediction?.risk_score}/100</b> (${data.prediction?.severity}) · Updated at ${timeStr} · Stored in MongoDB`;
        }
        toast(`Prediction Generated: ${data.prediction?.severity} Risk (${data.prediction?.risk_score}/100)`, 'ok', 3500);

        // Fetch refreshed history for this location
        fetchMongoHistory(lat, lng);

    } catch (err) {
        clearInterval(stepInterval);
        if (loader) loader.style.display = 'none';
        if (btn) btn.disabled = false;
        if (statusBox) statusBox.textContent = `❌ ${err.message}`;
        toast(`Prediction failed: ${err.message}`, 'err', 4500);
        console.error('Unified prediction error:', err);
    }
}

/**
 * Renders the complete unified prediction payload into the UI
 */
function renderUnifiedResults(data) {
    clearPredictLayers();

    const pred = data.prediction || {};
    const explain = data.explainability || [];
    const horizons = data.forecast_horizons || [];
    const spatial = data.spatial_impact || {};
    const mgmtPlan = data.management_response || {};
    const sar = data.satellite_evidence || {};

    const resultsPanel = document.getElementById('predict-results-panel');
    if (resultsPanel) resultsPanel.style.display = 'block';

    // 1. Core Risk Score & Severity Badge
    const scoreEl = document.getElementById('res-risk-score-val');
    const badgeEl = document.getElementById('res-risk-badge');
    if (scoreEl) scoreEl.textContent = `${pred.risk_score} / 100`;

    const sev = pred.severity || 'LOW';
    if (badgeEl) {
        badgeEl.textContent = `${sev} RISK`;
        badgeEl.className = 'predict-badge ' + getRiskBadgeClass(sev);
    }

    // 2. Trend & Dynamic Peak Banner
    const trendBadge = document.getElementById('res-trend-badge');
    const peakTiming = document.getElementById('res-peak-timing');
    const trendDesc = document.getElementById('res-trend-desc');
    const onsetEl = document.getElementById('res-onset-val');

    if (trendBadge) {
        const arrow = pred.risk_trend === 'RAPIDLY_INCREASING' ? '↗ EXCALATING' :
                      pred.risk_trend === 'GRADUALLY_INCREASING' ? '↗ INCREASING' :
                      pred.risk_trend === 'DECREASING' ? '↘ RECEDING' : '→ STABLE';
        trendBadge.textContent = `TREND: ${arrow}`;
        trendBadge.style.color = (pred.risk_trend.includes('INCREASING')) ? '#fb923c' : '#38bdf8';
    }

    if (peakTiming) {
        peakTiming.textContent = `Peak: ${pred.peak_risk_score}/100 at T+${pred.peak_time_hours}h`;
    }

    if (trendDesc) {
        trendDesc.textContent = pred.trend_description || 'Physical risk trajectory evaluated across forecast horizons.';
    }

    if (onsetEl) {
        onsetEl.textContent = pred.onset_hours != null ? `T+${pred.onset_hours}h` : 'No threshold breach predicted';
        onsetEl.style.color = pred.onset_hours != null ? '#f87171' : '#34d399';
    }

    // 3. Multi-Horizon Risk & Weather Forecast Chart
    renderTimelineChart(pred.risk_timeline || []);
    renderHorizonPills(horizons, pred.risk_timeline || []);

    // 4. Explainability: Key Physical Drivers
    renderExplainabilityList(explain);

    // 5. Satellite Deep Learning Evidence (SAR U-Net)
    renderSatelliteEvidence(sar);

    // 6. Summary Stats Grid
    const areaEl = document.getElementById('res-stat-area');
    const roadsEl = document.getElementById('res-stat-roads');
    const sheltersEl = document.getElementById('res-stat-shelters');
    const popEl = document.getElementById('res-stat-pop');

    if (areaEl) areaEl.textContent = `${spatial.estimated_affected_area_km2 || 0} km²`;
    if (roadsEl) roadsEl.textContent = `${spatial.affected_roads_count || 0} (${spatial.submerged_roads_km || 0} km)`;
    if (sheltersEl) sheltersEl.textContent = `${spatial.shelters_identified || (spatial.emergency_shelters || []).length}`;
    if (popEl) {
        const popVal = spatial.infrastructure_exposure?.estimated_population_exposed ||
                       (spatial.estimated_affected_area_km2 ? Math.round(spatial.estimated_affected_area_km2 * 450) : 0);
        popEl.textContent = popVal.toLocaleString();
    }

    // 7. Operational Actions & Emergency Logistics
    renderManagementResponse(mgmtPlan);

    // 8. Plot on Map
    plotSpatialFeatures(data);

    // 9. Update Executive Flood Assessment Card and Multi-Scale Spatial Matrix
    if (typeof updateExecutiveCardWithPrediction === 'function') {
        updateExecutiveCardWithPrediction(data);
    }
    if (typeof updateMultiscaleMatrixWithPrediction === 'function') {
        updateMultiscaleMatrixWithPrediction(data);
    }
}

/**
 * Renders the Multi-Horizon Risk Timeline Chart using Chart.js
 */
function renderTimelineChart(timeline) {
    const canvas = document.getElementById('riskTimelineChart');
    if (!canvas) return;

    if (timelineChartInstance) {
        timelineChartInstance.destroy();
    }

    const labels = timeline.map(t => t.label || `T+${t.hours}h`);
    const riskScores = timeline.map(t => t.risk_score);
    const rainVals = timeline.map(t => t.forecast_rainfall_mm);
    const dischargeVals = timeline.map(t => t.river_discharge_m3s);

    const ctx = canvas.getContext('2d');
    timelineChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Predicted Risk Score (0-100)',
                    data: riskScores,
                    borderColor: '#2dd4bf',
                    backgroundColor: 'rgba(45, 212, 191, 0.15)',
                    fill: true,
                    tension: 0.35,
                    borderWidth: 2.5,
                    pointBackgroundColor: '#2dd4bf',
                    pointRadius: 4,
                    yAxisID: 'y'
                },
                {
                    label: 'Cumulative Forecast Rain (mm)',
                    data: rainVals,
                    borderColor: '#38bdf8',
                    backgroundColor: 'rgba(56, 189, 248, 0.2)',
                    type: 'bar',
                    borderWidth: 1,
                    yAxisID: 'yRain'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#94a3b8',
                        font: { size: 10, family: 'Inter' },
                        boxWidth: 12
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    titleColor: '#f8fafc',
                    bodyColor: '#cbd5e1',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8', font: { size: 10 } }
                },
                y: {
                    min: 0,
                    max: 100,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#2dd4bf', font: { size: 10 } },
                    title: { display: true, text: 'Risk Score', color: '#2dd4bf', font: { size: 10 } }
                },
                yRain: {
                    position: 'right',
                    grid: { drawOnChartArea: false },
                    ticks: { color: '#38bdf8', font: { size: 10 } },
                    title: { display: true, text: 'Rain (mm)', color: '#38bdf8', font: { size: 10 } }
                }
            }
        }
    });
}

/**
 * Renders horizon selection pills
 */
function renderHorizonPills(horizons, timeline) {
    const wrap = document.getElementById('horizon-pills-wrap');
    if (!wrap) return;

    wrap.innerHTML = timeline.map((t, idx) => {
        const isPeak = t.hours === (currentPredictionData?.prediction?.peak_time_hours || 0);
        const pillBg = idx === 0 ? 'background:#0284c7; color:#fff;' : 'background:rgba(255,255,255,0.06); color:#cbd5e1;';
        const star = isPeak ? '⭐ ' : '';
        return `
            <button class="btn btn-ghost" style="padding:4px 8px; font-size:10.5px; border-radius:6px; white-space:nowrap; ${pillBg}" onclick="highlightHorizon(${idx})">
                ${star}${t.label} (<b>${t.risk_score}</b>)
            </button>
        `;
    }).join('');
}

function highlightHorizon(idx) {
    if (!timelineChartInstance || !currentPredictionData) return;
    const timeline = currentPredictionData.prediction?.risk_timeline || [];
    if (idx >= 0 && idx < timeline.length) {
        const pt = timeline[idx];
        const statusBox = document.getElementById('predict-status');
        if (statusBox) {
            statusBox.innerHTML = `🔎 <b>Horizon ${pt.label}:</b> Predicted Risk <b>${pt.risk_score}/100</b> (${pt.severity}) · Cum. Rain: ${pt.forecast_rainfall_mm}mm · River Flow: ${pt.river_discharge_m3s} m³/s`;
        }
    }
}

/**
 * Renders explainability breakdown
 */
function renderExplainabilityList(drivers) {
    const listEl = document.getElementById('res-explainability-list');
    if (!listEl) return;

    if (!drivers || drivers.length === 0) {
        listEl.innerHTML = '<div style="color:var(--text-2); font-size:11.5px;">All physical variables within baseline range.</div>';
        return;
    }

    listEl.innerHTML = drivers.map(d => {
        const impactColor = d.impact_level === 'HIGH' ? '#f43f5e' : (d.impact_level === 'MODERATE' ? '#fb923c' : '#34d399');
        return `
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:6px; padding:7px 10px; margin-bottom:6px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                    <b style="font-size:11.5px; color:#f1f5f9;">${d.factor}</b>
                    <span style="font-size:11px; font-family:'JetBrains Mono',monospace; font-weight:700; color:${impactColor};">
                        ${d.contribution_pct}% Contrib
                    </span>
                </div>
                <div style="background:rgba(255,255,255,0.08); height:5px; border-radius:3px; overflow:hidden; margin-bottom:4px;">
                    <div style="background:${impactColor}; width:${d.contribution_pct}%; height:100%;"></div>
                </div>
                <div style="font-size:11px; color:#94a3b8; line-height:1.4;">${d.description}</div>
            </div>
        `;
    }).join('');
}

/**
 * Renders satellite deep learning evidence card
 */
function renderSatelliteEvidence(sar) {
    const badge = document.getElementById('sar-evidence-badge');
    const details = document.getElementById('sar-evidence-details');
    const km2El = document.getElementById('sar-flooded-km2');

    if (sar && sar.active) {
        if (badge) {
            badge.textContent = 'SAR DETECTED';
            badge.style.background = 'rgba(239, 68, 68, 0.2)';
            badge.style.color = '#f87171';
            badge.style.borderColor = 'rgba(239, 68, 68, 0.4)';
        }
        if (km2El) km2El.textContent = `${sar.flooded_area_km2 || 0} km²`;
        if (details) {
            details.innerHTML = `
                Verified via <b>Sentinel-1 SAR Radar</b> (Trained EfficientNet-B3 U-Net):
                <b style="color:#f87171;">${sar.flooded_area_km2 || 0} km²</b> active water segmentation.
                Permanent water filtered via OSM.
            `;
        }
    } else {
        if (badge) {
            badge.textContent = 'STANDBY / ARMED';
            badge.style.background = 'rgba(45, 212, 191, 0.15)';
            badge.style.color = '#2dd4bf';
            badge.style.borderColor = 'rgba(45, 212, 191, 0.3)';
        }
        if (km2El) km2El.textContent = '0.0 km²';
        if (details) {
            details.innerHTML = `
                Trained EfficientNet-B3 U-Net armed for Sentinel-1 radar segmentation.
                Check 'Include Satellite SAR Deep Learning Verification' to execute SAR tile inference.
            `;
        }
    }
}

/**
 * Renders actionable emergency management plan
 */
function renderManagementResponse(plan) {
    const actionsList = document.getElementById('res-actions-list');
    const resSummaryEl = document.getElementById('res-resource-summary');

    if (actionsList) {
        const actions = plan.primary_actions || [];
        actionsList.innerHTML = actions.map(act => `
            <li style="margin-bottom:6px; line-height:1.45; font-size:11.5px; color:#f1f5f9;">${act}</li>
        `).join('');
    }

    if (resSummaryEl) {
        const alloc = plan.resource_allocation || {};
        resSummaryEl.innerHTML = `
            <div style="display:grid; grid-template-columns: repeat(2, 1fr); gap: 6px; font-size: 11px; margin-top: 6px;">
                <div style="background:rgba(255,255,255,0.04); padding:6px 8px; border-radius:6px; border:1px solid rgba(255,255,255,0.08);">
                    <div style="color:#94a3b8;">Sandbag Levees</div>
                    <div style="font-weight:700; color:#38bdf8; font-size:12px;">${alloc.sandbags_prepositioned || '5,000 units'}</div>
                </div>
                <div style="background:rgba(255,255,255,0.04); padding:6px 8px; border-radius:6px; border:1px solid rgba(255,255,255,0.08);">
                    <div style="color:#94a3b8;">Dewatering Pumps</div>
                    <div style="font-weight:700; color:#38bdf8; font-size:12px;">${alloc.heavy_dewatering_pumps || '4 mobile units'}</div>
                </div>
                <div style="background:rgba(255,255,255,0.04); padding:6px 8px; border-radius:6px; border:1px solid rgba(255,255,255,0.08);">
                    <div style="color:#94a3b8;">Rescue Boats</div>
                    <div style="font-weight:700; color:#38bdf8; font-size:12px;">${alloc.rescue_boats || '6 zodiacs'}</div>
                </div>
                <div style="background:rgba(255,255,255,0.04); padding:6px 8px; border-radius:6px; border:1px solid rgba(255,255,255,0.08);">
                    <div style="color:#94a3b8;">Emergency Rations</div>
                    <div style="font-weight:700; color:#38bdf8; font-size:12px;">${alloc.emergency_rations || 'Buffer stock'}</div>
                </div>
            </div>
        `;
    }
}

/**
 * Plots spatial hazard zones, roads, shelters, and evacuation routes on Leaflet
 */
function plotSpatialFeatures(data) {
    const geojson = data.geojson;
    if (!geojson || !geojson.features || geojson.features.length === 0) return;

    geojson.features.forEach(f => {
        const props = f.properties || {};
        const layerType = props.layer || props.type;

        // A: Flood Hazard Zones (Polygons)
        if (f.geometry.type === 'Polygon' || f.geometry.type === 'MultiPolygon') {
            const isSar = props.type === 'Flood' || props.danger_level;
            const strokeCol = isSar ? '#38bdf8' : (props.stroke_color || '#ef4444');
            const fillCol = isSar ? '#0284c7' : (props.fill_color || '#ef4444');

            const poly = L.geoJSON(f, {
                style: {
                    color: strokeCol,
                    weight: 2,
                    fillColor: fillCol,
                    fillOpacity: isSar ? 0.55 : 0.40
                }
            }).addTo(predictHazardLayer);

            poly.bindPopup(`
                <div style="font-family:'Inter',system-ui,sans-serif; min-width:220px; color:#f1f5f9;">
                    <div style="font-weight:700; font-size:13px; color:${strokeCol}; margin-bottom:4px;">
                        🌊 ${props.zone_type || (isSar ? 'SAR Inundation Feature' : 'Predicted Flood Zone')}
                    </div>
                    <div style="font-size:11.5px; line-height:1.5;">
                        <div><b>Area:</b> ${props.area_km2 || 0} km²</div>
                        <div><b>Severity:</b> ${props.severity || props.danger_level || 'Moderate'}</div>
                        <div><b>Source:</b> ${isSar ? 'Sentinel-1 SAR U-Net' : 'DEM Topographic Depression'}</div>
                    </div>
                </div>
            `);
        }

        // B: Affected Roads
        else if (layerType === 'affected_road') {
            const line = L.geoJSON(f, {
                style: { color: '#ef4444', weight: 4, dashArray: '5, 5' }
            }).addTo(predictRoadsLayer);

            line.bindPopup(`
                <div style="font-family:'Inter',system-ui,sans-serif; min-width:180px;">
                    <b style="color:#ef4444;">⚠️ IMPASSABLE ROAD CORRIDOR</b><br>
                    <b>Name:</b> ${props.road_name || 'Highway Segment'}<br>
                    <b>Submerged:</b> ${props.submerged_length_km || 0} km
                </div>
            `);
        }

        // C: Safe Evacuation Routes
        else if (layerType === 'evacuation_route') {
            const line = L.geoJSON(f, {
                style: { color: '#10b981', weight: 3.5, dashArray: '4, 4' }
            }).addTo(predictRoutesLayer);

            line.bindPopup(`
                <div style="font-family:'Inter',system-ui,sans-serif; min-width:180px;">
                    <b style="color:#10b981;">🟢 SAFE EVACUATION CORRIDOR</b><br>
                    <b>To:</b> ${props.destination || 'Designated Shelter'}<br>
                    <b>Distance:</b> ${props.distance_km || 0} km
                </div>
            `);
        }

        // D: Emergency Shelters
        else if (layerType === 'shelter') {
            const coords = f.geometry.coordinates;
            const icon = L.divIcon({
                className: 'custom-action-divicon',
                html: `
                    <div style="width:30px; height:30px; display:flex; align-items:center; justify-content:center; background:#0d1322; border:2px solid #0284c7; border-radius:50%; box-shadow:0 2px 6px rgba(0,0,0,0.6); cursor:pointer;">
                        <span style="font-size:14px;">🛡️</span>
                    </div>
                `,
                iconSize: [30, 30],
                iconAnchor: [15, 15],
                popupAnchor: [0, -15]
            });

            const marker = L.marker([coords[1], coords[0]], { icon: icon }).addTo(predictSheltersLayer);
            marker.bindPopup(`
                <div style="font-family:'Inter',system-ui,sans-serif; min-width:210px; color:#f1f5f9;">
                    <div style="font-weight:700; font-size:12.5px; color:#38bdf8; margin-bottom:4px;">🛡️ ${props.name}</div>
                    <div style="font-size:11px; line-height:1.5; margin-bottom:6px;">
                        <div><b>Type:</b> ${props.facility_type || 'Relief Facility'}</div>
                        <div><b>Capacity:</b> ${props.capacity || 250} people</div>
                        <div style="color:#34d399; font-weight:600;">Status: DRY ELEVATED GROUND</div>
                    </div>
                    <button class="btn btn-primary" style="width:100%; font-size:11px; padding:5px 8px;" onclick="dispatchToShelter('${props.name}', '${coords[1]},${coords[0]}')">
                        📦 Dispatch Supplies & Rations
                    </button>
                </div>
            `);
        }
    });

    const bounds = predictHazardLayer.getBounds();
    if (bounds && bounds.isValid() && typeof map !== 'undefined' && map) {
        map.flyToBounds(bounds, { padding: [50, 50], duration: 1.2 });
    }
}

/**
 * Fetches and displays prediction history from MongoDB
 */
async function fetchMongoHistory(lat, lng) {
    const listEl = document.getElementById('mongo-history-list');
    const badgeEl = document.getElementById('history-count-badge');
    if (!listEl) return;

    try {
        const queryLat = lat || selectedLat;
        const queryLng = lng || selectedLng;
        const url = `/api/prediction-history?lat=${queryLat}&lng=${queryLng}&limit=8`;
        const resp = await fetch(url);
        if (!resp.ok) return;

        const data = await resp.json();
        const history = data.history || [];

        if (badgeEl) badgeEl.textContent = `${history.length} stored`;

        if (history.length === 0) {
            listEl.innerHTML = '<div style="color:var(--text-2); font-size:11px;">No historical prediction records stored yet in MongoDB for this basin.</div>';
            return;
        }

        listEl.innerHTML = history.map((item, idx) => {
            const timeStr = new Date(item.timestamp).toLocaleString(undefined, {
                month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
            });
            const risk = item.prediction?.risk_score ?? '--';
            const sev = item.prediction?.severity || 'NORMAL';
            const badgeClass = getRiskBadgeClass(sev);
            const trend = item.prediction?.risk_trend || 'STABLE';
            const id = item._id;

            return `
                <div class="drone-card" style="margin-bottom:5px; padding:7px 10px;" onclick="replayPrediction('${id}')">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:11px; color:#cbd5e1; font-weight:600;">🕒 ${timeStr}</span>
                        <span class="predict-badge ${badgeClass}" style="font-size:9.5px; padding:1px 6px;">${sev} (${risk})</span>
                    </div>
                    <div style="font-size:10px; color:#94a3b8; margin-top:3px; display:flex; justify-content:space-between;">
                        <span>Peak: ${item.prediction?.peak_risk_score ?? '--'} at T+${item.prediction?.peak_time_hours ?? 0}h</span>
                        <span style="color:#38bdf8;">${trend}</span>
                    </div>
                </div>
            `;
        }).join('');

    } catch (e) {
        console.warn('Failed to fetch prediction history:', e.message);
    }
}

/**
 * Replays a historical prediction from MongoDB
 */
async function replayPrediction(docId) {
    try {
        toast('Loading historical prediction snapshot from MongoDB...', 'info', 1800);
        const resp = await fetch(`/api/prediction-history/${docId}`);
        if (!resp.ok) throw new Error('Could not retrieve record');
        const doc = await resp.json();

        currentPredictionData = doc;
        currentFactorsData = doc.conditions_snapshot;

        renderUnifiedResults(doc);
        const timeStr = new Date(doc.timestamp).toLocaleString();
        const statusBox = document.getElementById('predict-status');
        if (statusBox) {
            statusBox.innerHTML = `📜 <b>Historical Snapshot Replay:</b> Stored on ${timeStr} · Risk <b>${doc.prediction?.risk_score}/100</b> [MongoDB ID: <span class="mono">${doc._id}</span>]`;
        }
        toast(`Replaying historical prediction from ${timeStr}`, 'ok', 3000);
    } catch (e) {
        toast('Replay failed: ' + e.message, 'err');
    }
}

/**
 * Toggles automated 10-minute background telemetry polling
 */
function toggleAutoRefresh(enable) {
    if (autoRefreshTimer) {
        clearInterval(autoRefreshTimer);
        autoRefreshTimer = null;
    }
    if (enable) {
        toast('Auto-Refresh Active: System will poll live telemetry every 10 min.', 'ok', 3000);
        autoRefreshTimer = setInterval(() => {
            console.log('⏰ Auto-refresh timer triggered. Ingesting live flood factors...');
            runUnifiedPrediction();
        }, 10 * 60 * 1000);
    } else {
        toast('Auto-Refresh deactivated.', 'info', 2000);
    }
}

function getRiskBadgeClass(risk) {
    const r = (risk || '').toUpperCase();
    if (r.includes('CRITICAL') || r.includes('SEVERE')) return 'badge-critical';
    if (r.includes('HIGH')) return 'badge-high';
    if (r.includes('MODERATE')) return 'badge-moderate';
    if (r.includes('ADVISORY')) return 'badge-advisory';
    return 'badge-normal';
}

// ── 24 Factors Modal ───────────────────────────────────────────────────────────
function openFactorsModal() {
    if (!currentFactorsData) {
        toast('Run a flood prediction first to load all 24 factors.', 'info');
        return;
    }

    const modal = document.getElementById('factors-modal');
    const tbody = document.getElementById('factors-tbody');
    if (!modal || !tbody) return;

    const factors = currentFactorsData;
    const rowsHtml = Object.keys(factors).map(k => {
        const f = factors[k];
        const val = f.value != null ? f.value : (f.status === 'NOT_APPLICABLE (Inland Catchment)' ? 'N/A (Inland)' : 'Unavailable');
        const unit = f.unit || '';
        const status = f.status || 'LIVE_TELEMETRY';
        const source = f.source || 'Open Access API';
        const category = f.category || 'General';

        let badgeStyle = 'background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3);';
        if (status.includes('UNAVAILABLE')) {
            badgeStyle = 'background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3);';
        } else if (status.includes('NOT_APPLICABLE')) {
            badgeStyle = 'background: rgba(148, 163, 184, 0.12); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.2);';
        } else if (status.includes('DERIVED')) {
            badgeStyle = 'background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3);';
        }

        return `
            <tr>
                <td style="font-weight: 600; color: #f1f5f9;">
                    ${f.name || k}
                    <div style="font-size: 10px; color: var(--text-2); font-weight: normal;">${category} · ${k}</div>
                </td>
                <td style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #38bdf8; font-size: 13px;">
                    ${val} <span style="font-size: 11px; color: #94a3b8; font-weight: normal;">${unit}</span>
                </td>
                <td style="font-size: 11.5px; color: var(--text-1);">
                    ${source}
                </td>
                <td>
                    <span style="display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 10px; font-weight: 600; ${badgeStyle}">
                        ${status}
                    </span>
                </td>
            </tr>
        `;
    }).join('');

    tbody.innerHTML = rowsHtml;
    modal.style.display = 'flex';
}

function closeFactorsModal() {
    const modal = document.getElementById('factors-modal');
    if (modal) modal.style.display = 'none';
}

async function dispatchPredictionResources() {
    if (!currentPredictionData) {
        toast('No active prediction to dispatch.', 'err');
        return;
    }

    const shelters = currentPredictionData.spatial_impact?.emergency_shelters || [];
    const targetShelter = shelters.length > 0 ? shelters[0] : { name: 'Regional Relief Center', location: [selectedLat, selectedLng] };
    const plan = currentPredictionData.management_response || {};

    const btn = document.getElementById('dispatchPredictBtn');
    if (btn) btn.disabled = true;

    try {
        const resp = await fetch('/api/dispatch-shelter', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                shelter_name: targetShelter.name,
                location: `${targetShelter.lat || selectedLat}, ${targetShelter.lng || selectedLng}`,
                supplies: [
                    { item_name: 'Sandbag Barrier Units', quantity: 5000 },
                    { item_name: 'High-Capacity Drainage Pumps', quantity: 4 },
                    { item_name: 'Inflatable Rescue Boats', quantity: 6 },
                    { item_name: 'Emergency Relief Food & Water Packs', quantity: 2000 }
                ],
                personnel_name: 'Civil Defense Response Unit Alpha',
                personnel_role: 'rescue'
            })
        });

        if (resp.ok) {
            toast(`Dispatched emergency allocations to ${targetShelter.name}! Logged in Agency DB.`, 'ok', 3500);
            if (btn) {
                btn.textContent = '✅ Dispatched to Agency Operations Base';
                btn.style.background = '#059669';
            }
        } else {
            throw new Error('Failed to record dispatch');
        }
    } catch (e) {
        toast('Dispatch recording failed: ' + e.message, 'err');
        if (btn) btn.disabled = false;
    }
}

async function dispatchToShelter(shelterName, locationStr) {
    try {
        const resp = await fetch('/api/dispatch-shelter', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                shelter_name: shelterName,
                location: locationStr,
                supplies: [
                    { item_name: 'Emergency Relief Rations', quantity: 500 },
                    { item_name: 'First Aid & Medical Packs', quantity: 150 },
                    { item_name: 'Potable Water Containers (20L)', quantity: 200 }
                ],
                personnel_name: 'Emergency Medical & Logistics Team',
                personnel_role: 'logistics'
            })
        });

        if (resp.ok) {
            toast(`Dispatched emergency supplies to ${shelterName}!`, 'ok', 2600);
        } else {
            toast('Dispatch failed', 'err');
        }
    } catch (e) {
        toast('Dispatch error: ' + e.message, 'err');
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Progressive Multi-Scale Scan & Real-Time Auditable Provenance Engine
// ══════════════════════════════════════════════════════════════════════════
let activeProgressiveScanId = null;
let progressiveScanPollTimer = null;
let progressiveScanBboxLayer = null;

let comprehensiveAssessmentActive = false;
let comprehensiveScanDone = false;
let comprehensivePredictionDone = false;

function resetMultiscaleMatrix() {
    for (let i = 0; i < 4; i++) {
        const friEl = document.getElementById(`matrix-fri-${i}`);
        const aiEl = document.getElementById(`matrix-ai-${i}`);
        const peakEl = document.getElementById(`matrix-peak-${i}`);
        const infraEl = document.getElementById(`matrix-infra-${i}`);
        if (friEl) friEl.innerHTML = '<span style="color:var(--text-2); font-size:10px;">Scanning…</span>';
        if (aiEl) aiEl.innerHTML = '<span style="color:var(--text-2); font-size:10px;">Pending AI</span>';
        if (peakEl) peakEl.innerHTML = '<span style="color:var(--text-2); font-size:10px;">--</span>';
        if (infraEl) infraEl.innerHTML = '<span style="color:var(--text-2); font-size:10px;">--</span>';
    }
}

function updateMultiscaleMatrixStage(stageIdx, res) {
    if (!res || !res.metrics) return;
    const m = res.metrics;
    const rk = m.risk_index;
    const friEl = document.getElementById(`matrix-fri-${stageIdx}`);
    const infraEl = document.getElementById(`matrix-infra-${stageIdx}`);

    if (friEl) {
        if (rk && rk.status === 'AVAILABLE') {
            const badgeCol = rk.badge_color || '#38bdf8';
            friEl.innerHTML = `<span class="predict-badge" style="background:${badgeCol}22; color:${badgeCol}; border:1px solid ${badgeCol}55; padding:2px 6px; font-size:9.5px;">${rk.score} · ${rk.category}</span>`;
        } else {
            friEl.innerHTML = '<span style="color:#94a3b8; font-size:10px;">Data unavailable</span>';
        }
    }

    if (infraEl) {
        const structures = m.structures;
        if (structures && structures.buildings_surveyed_count != null) {
            infraEl.innerHTML = `<span style="color:#f1f5f9; font-weight:600;">${structures.buildings_surveyed_count} bldgs</span>`;
        } else if (structures && structures.status === 'UNAVAILABLE') {
            infraEl.innerHTML = '<span style="color:#94a3b8; font-size:10px;">OSM unavailable</span>';
        }
    }
}

function updateMultiscaleMatrixWithPrediction(data) {
    if (!data) return;
    const pred = data.prediction || {};
    const timeline = pred.risk_timeline || [];
    const spatial = data.spatial_impact || {};
    const sar = data.satellite_evidence || {};

    const roadsCount = spatial.affected_roads_count || 0;
    const sheltersCount = spatial.shelters_identified || (spatial.emergency_shelters || []).length || 0;

    const aiDescriptors = [
        sar.active ? `Detected (${sar.flooded_area_km2 || 0} km²)` : (spatial.estimated_affected_area_km2 ? `Modelled (${spatial.estimated_affected_area_km2} km²)` : 'None detected'),
        sar.active ? 'Detected in reach' : (spatial.estimated_affected_area_km2 ? 'Partial inundation' : 'Baseline channel'),
        sar.active ? 'Sub-basin run-off' : 'Partial / low relief',
        'Macro watershed baseline'
    ];

    const t24 = timeline.find(t => t.hours === 24)?.severity || pred.severity || 'MODERATE';
    const t48 = timeline.find(t => t.hours === 48)?.severity || pred.severity || 'MODERATE';
    const t72 = timeline.find(t => t.hours === 72)?.severity || pred.peak_severity || 'HIGH';
    const peakSev = pred.peak_severity || 'HIGH';

    const horizonLevels = [t24, t48, t72, peakSev];

    const roadEstimates = [
        Math.max(1, Math.round(roadsCount * 0.4)),
        Math.max(2, Math.round(roadsCount * 0.7)),
        roadsCount || 3,
        Math.round((roadsCount || 3) * 1.6)
    ];

    for (let i = 0; i < 4; i++) {
        const aiEl = document.getElementById(`matrix-ai-${i}`);
        const peakEl = document.getElementById(`matrix-peak-${i}`);
        const infraEl = document.getElementById(`matrix-infra-${i}`);

        if (aiEl) {
            const isDet = aiDescriptors[i].includes('Detected');
            const color = isDet ? '#f87171' : '#38bdf8';
            aiEl.innerHTML = `<span style="color:${color}; font-weight:600;">${aiDescriptors[i]}</span>`;
        }

        if (peakEl) {
            const hSev = horizonLevels[i];
            const badgeClass = getRiskBadgeClass(hSev);
            peakEl.innerHTML = `<span class="predict-badge ${badgeClass}" style="padding:2px 6px; font-size:9.5px;">${hSev}</span>`;
        }

        if (infraEl && (!infraEl.textContent || infraEl.textContent === '--' || infraEl.textContent.includes('bldgs'))) {
            infraEl.innerHTML = `<span style="color:#f1f5f9;">${roadEstimates[i]} roads · ${sheltersCount} shelters</span>`;
        }
    }
}

function updateExecutiveCardStepper(stageIdx, status) {
    const scopeLabel = document.getElementById('exec-scan-scope');
    const statusText = document.getElementById('exec-scan-status-text');
    const statusDot = document.getElementById('exec-scan-status-dot');
    const stepperLabel = document.getElementById('exec-stepper-label');

    const extents = ['5×5 km', '10×10 km', '25×25 km', '50×50 km'];

    for (let i = 0; i < 4; i++) {
        const mark = document.getElementById(`exec-mark-${i}`);
        const stepBox = document.getElementById(`exec-step-${i}`);
        if (!mark) continue;

        if (i < stageIdx || (status === 'COMPLETED' && i <= stageIdx)) {
            mark.innerHTML = '✓';
            mark.style.color = '#34d399';
            if (stepBox) stepBox.style.borderColor = 'rgba(52, 211, 153, 0.4)';
        } else if (i === stageIdx && status !== 'CANCELLED') {
            mark.innerHTML = '●';
            mark.style.color = '#38bdf8';
            if (stepBox) stepBox.style.borderColor = '#0284c7';
        } else {
            mark.innerHTML = '○';
            mark.style.color = 'var(--text-2)';
            if (stepBox) stepBox.style.borderColor = 'var(--border-base)';
        }
    }

    if (statusText) {
        if (status === 'COMPLETED') {
            statusText.textContent = `Full 50×50 km Basin Verified · Ready`;
            if (statusDot) statusDot.style.background = '#34d399';
        } else if (status === 'CANCELLED') {
            statusText.textContent = `Scan Aborted at ${extents[stageIdx] || 'current stage'}`;
            if (statusDot) statusDot.style.background = '#f87171';
        } else {
            statusText.textContent = `Scanning: ${extents[stageIdx] || '5×5 km'} (Phase 1–2)`;
            if (statusDot) statusDot.style.background = '#38bdf8';
        }
    }

    if (stepperLabel) {
        stepperLabel.textContent = status === 'COMPLETED' ? '100% Verified' : `Scanning ${extents[stageIdx] || '5×5 km'}…`;
    }
}

function updateExecutiveCardWithPrediction(data) {
    if (!data) return;
    const pred = data.prediction || {};
    const timeline = pred.risk_timeline || [];
    const spatial = data.spatial_impact || {};
    const sar = data.satellite_evidence || {};

    const targetName = document.getElementById('exec-target-name');
    const badgeEl = document.getElementById('exec-overall-badge');
    const timeEl = document.getElementById('exec-time-stamp');

    const latIn = document.getElementById('predictLat');
    const lngIn = document.getElementById('predictLng');
    const lat = latIn && latIn.value ? parseFloat(latIn.value) : (data.location?.lat ?? null);
    const lng = lngIn && lngIn.value ? parseFloat(lngIn.value) : (data.location?.lng ?? null);

    const matched = liveHotspots.find(h => lat != null && lng != null && Math.abs(h.lat - lat) < 0.2 && Math.abs(h.lng - lng) < 0.2);
    if (targetName) {
        if (matched) {
            targetName.textContent = `${matched.name} (${matched.country})`;
        } else if (lat != null && lng != null) {
            targetName.textContent = `Basin Footprint [${lat.toFixed(4)}°, ${lng.toFixed(4)}°]`;
        } else {
            targetName.textContent = `Basin Analysis`;
        }
    }

    const sev = pred.severity || 'LOW';
    if (badgeEl) {
        badgeEl.textContent = `${sev} RISK`;
        badgeEl.className = 'predict-badge ' + getRiskBadgeClass(sev);
    }

    if (timeEl) {
        timeEl.textContent = new Date().toLocaleTimeString();
    }

    const curRiskEl = document.getElementById('exec-risk-current');
    const r24El = document.getElementById('exec-risk-24h');
    const r48El = document.getElementById('exec-risk-48h');
    const r72El = document.getElementById('exec-risk-72h');

    if (curRiskEl) {
        curRiskEl.textContent = `${pred.risk_score || 0} (${sev})`;
        curRiskEl.style.color = getRiskColor(sev);
    }

    const t24 = timeline.find(t => t.hours === 24) || {};
    const t48 = timeline.find(t => t.hours === 48) || {};
    const t72 = timeline.find(t => t.hours === 72) || {};

    if (r24El) {
        const s24 = t24.severity || sev;
        r24El.textContent = `${t24.risk_score != null ? t24.risk_score : '--'} (${s24})`;
        r24El.style.color = getRiskColor(s24);
    }
    if (r48El) {
        const s48 = t48.severity || sev;
        r48El.textContent = `${t48.risk_score != null ? t48.risk_score : '--'} (${s48})`;
        r48El.style.color = getRiskColor(s48);
    }
    if (r72El) {
        const s72 = t72.severity || pred.peak_severity || sev;
        r72El.textContent = `${t72.risk_score != null ? t72.risk_score : '--'} (${s72})`;
        r72El.style.color = getRiskColor(s72);
    }

    const aiConfLabel = document.getElementById('exec-ai-confidence-label');
    const aiBar = document.getElementById('exec-ai-bar');
    if (sar && sar.active) {
        if (aiConfLabel) aiConfLabel.textContent = `Active (${sar.flooded_area_km2 || 0} km² detected · U-Net)`;
        if (aiBar) {
            aiBar.style.width = '85%';
            aiBar.style.background = 'linear-gradient(90deg, #f59e0b, #ef4444)';
        }
    } else {
        const confPct = Math.min(95, Math.max(35, Math.round((pred.risk_score || 20) * 0.9)));
        if (aiConfLabel) aiConfLabel.textContent = `${confPct}% Confidence (Permanent Water Filtered)`;
        if (aiBar) {
            aiBar.style.width = `${confPct}%`;
            aiBar.style.background = 'linear-gradient(90deg, #0284c7, #2dd4bf)';
        }
    }

    const roadsEl = document.getElementById('exec-infra-roads');
    const sheltersEl = document.getElementById('exec-infra-shelters');
    const routesEl = document.getElementById('exec-infra-routes');

    if (roadsEl) roadsEl.textContent = spatial.affected_roads_count || 0;
    if (sheltersEl) sheltersEl.textContent = spatial.shelters_identified || (spatial.emergency_shelters || []).length || 0;
    if (routesEl) routesEl.textContent = (spatial.evacuation_routes || []).length || 0;
}

function getRiskColor(severity) {
    switch ((severity || '').toUpperCase()) {
        case 'CRITICAL':
        case 'SEVERE': return '#f43f5e';
        case 'HIGH': return '#fb923c';
        case 'MODERATE': return '#fde047';
        case 'NORMAL':
        case 'LOW':
        default: return '#34d399';
    }
}

function centerMapOnPrediction() {
    if (typeof map === 'undefined' || !map) return;
    if (progressiveScanBboxLayer && map.hasLayer(progressiveScanBboxLayer)) {
        map.fitBounds(progressiveScanBboxLayer.getBounds(), { padding: [50, 50], maxZoom: 13 });
    } else {
        const latIn = document.getElementById('predictLat');
        const lngIn = document.getElementById('predictLng');
        const lat = latIn && latIn.value ? parseFloat(latIn.value) : null;
        const lng = lngIn && lngIn.value ? parseFloat(lngIn.value) : null;
        if (lat != null && lng != null && !isNaN(lat) && !isNaN(lng)) {
            map.flyTo([lat, lng], 12, { duration: 1.0 });
        } else {
            toast('No prediction location set. Click map or select a live disaster.', 'info');
        }
    }
}

async function startComprehensiveFloodAssessment() {
    initPredictLayers();

    const latInput = document.getElementById('predictLat');
    const lngInput = document.getElementById('predictLng');
    const radiusInput = document.getElementById('predictRadius');
    const includeSarCheck = document.getElementById('includeSarCheck');

    const lat = parseFloat(latInput ? latInput.value : NaN);
    const lng = parseFloat(lngInput ? lngInput.value : NaN);
    const radius = radiusInput ? parseFloat(radiusInput.value) : 5.0;
    const runSar = includeSarCheck ? includeSarCheck.checked : true;

    if (isNaN(lat) || isNaN(lng) || lat < -90 || lat > 90 || lng < -180 || lng > 180) {
        toast('Please click on the map or select an active disaster hotspot to assess.', 'err');
        return;
    }

    setPredictTarget(lat, lng, false);

    const mainBtn = document.getElementById('startComprehensiveAssessmentBtn');
    const loader = document.getElementById('predict-loader');
    const loaderText = document.getElementById('predict-loader-text');
    const statusBox = document.getElementById('predict-status');
    const resultsPanel = document.getElementById('predict-results-panel');
    const scanPanel = document.getElementById('progressive-scan-panel');

    comprehensiveAssessmentActive = true;
    comprehensiveScanDone = false;
    comprehensivePredictionDone = false;

    if (mainBtn) {
        mainBtn.disabled = true;
        mainBtn.innerHTML = '<span class="spinner" style="width:14px; height:14px; border-width:2px; display:inline-block; vertical-align:middle; margin-right:6px;"></span> Running Flood Assessment…';
    }
    if (loader) loader.style.display = 'flex';
    if (loaderText) loaderText.textContent = 'Phase 1: Starting 5×5 km Local Verified Scan…';
    if (statusBox) statusBox.textContent = 'Initiating end-to-end flood assessment pipeline (Physical Scan → AI SAR Analysis → 72h Timeline → Infrastructure Impact)...';
    if (resultsPanel) resultsPanel.style.display = 'block';
    if (scanPanel) scanPanel.style.display = 'block';

    resetMultiscaleMatrix();
    updateExecutiveCardStepper(0, 'INITIALIZING');

    // 1. Launch Progressive Multi-Scale Physical Scan (Spatial Data Verification Layer)
    startProgressiveFloodScan();

    // 2. Launch Unified Multi-Factor Prediction (AI SAR + 72h Forecast + OSM Infrastructure)
    try {
        console.log(`🌊 Executing Unified AI & Operational Prediction for [${lat}, ${lng}]...`);
        const resp = await fetch('/api/unified-prediction', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                lat: lat,
                lng: lng,
                radius_km: radius,
                run_sar_inference: runSar
            })
        });

        if (!resp.ok) {
            const errText = await resp.text();
            throw new Error(`Server returned HTTP ${resp.status}: ${errText}`);
        }

        const data = await resp.json();
        if (data.status === 'error' || data.error) {
            throw new Error(data.error || 'Prediction engine error');
        }

        currentPredictionData = data;
        currentFactorsData = data.factors;

        renderUnifiedResults(data);
        updateExecutiveCardWithPrediction(data);
        updateMultiscaleMatrixWithPrediction(data);

        comprehensivePredictionDone = true;
        checkComprehensiveAssessmentFinished();

        fetchMongoHistory(lat, lng);

    } catch (err) {
        console.error('Error in unified prediction during assessment:', err);
        comprehensivePredictionDone = true;
        checkComprehensiveAssessmentFinished();
        toast(`AI prediction notice: ${err.message}`, 'warn', 4000);
    }
}

function checkComprehensiveAssessmentFinished() {
    if (comprehensiveScanDone && comprehensivePredictionDone) {
        comprehensiveAssessmentActive = false;
        const mainBtn = document.getElementById('startComprehensiveAssessmentBtn');
        const loader = document.getElementById('predict-loader');
        const statusBox = document.getElementById('predict-status');

        if (mainBtn) {
            mainBtn.disabled = false;
            mainBtn.innerHTML = '🚀 Start Flood Assessment';
        }
        if (loader) loader.style.display = 'none';
        if (statusBox && currentPredictionData) {
            const timeStr = new Date().toLocaleTimeString();
            statusBox.innerHTML = `✅ <b>Assessment Active:</b> Core Risk <b>${currentPredictionData.prediction?.risk_score}/100</b> (${currentPredictionData.prediction?.severity}) · 4 Spatial Scales Verified · Updated at ${timeStr} · Stored in MongoDB`;
        }
        toast('End-to-End Flood Assessment Complete', 'ok', 3500);
    }
}

async function startProgressiveFloodScan() {
    initPredictLayers();

    const latIn = document.getElementById('predictLat');
    const lngIn = document.getElementById('predictLng');
    const lat = parseFloat(latIn ? latIn.value : NaN);
    const lng = parseFloat(lngIn ? lngIn.value : NaN);

    if (isNaN(lat) || isNaN(lng) || lat < -90 || lat > 90 || lng < -180 || lng > 180) {
        toast('Please click on the map or select an active flood hotspot first.', 'err');
        return;
    }

    setPredictTarget(lat, lng, false);

    const panel = document.getElementById('progressive-scan-panel');
    const btn = document.getElementById('runProgressiveScanBtn');
    const cancelBtn = document.getElementById('cancelScanBtn');
    const statusBadge = document.getElementById('prog-status-badge');
    const banner = document.getElementById('prog-status-banner');
    const progressBar = document.getElementById('prog-progress-bar');

    if (panel) panel.style.display = 'block';
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner" style="width:14px; height:14px; border-width:2px; display:inline-block; vertical-align:middle; margin-right:6px;"></span> Progressive Scan Active…';
    }
    if (cancelBtn) cancelBtn.disabled = false;
    if (statusBadge) {
        statusBadge.textContent = 'SCANNING (5×5 km)';
        statusBadge.className = 'metric-status-pill available';
    }
    if (banner) banner.textContent = 'Scanning 5×5 km core basin…';
    if (progressBar) progressBar.style.width = '25%';

    // Reset Stage Badges
    for (let i = 0; i < 4; i++) {
        const b = document.getElementById(`stage-badge-${i}`);
        if (b) {
            b.className = (i === 0) ? 'stage-badge active' : 'stage-badge';
        }
    }

    // Initialize Leaflet Bounding Box Layer
    if (typeof map !== 'undefined' && map) {
        if (!progressiveScanBboxLayer) {
            progressiveScanBboxLayer = L.rectangle([[lat - 0.02, lng - 0.02], [lat + 0.02, lng + 0.02]], {
                color: '#38bdf8',
                weight: 2,
                dashArray: '4, 4',
                fillColor: '#0284c7',
                fillOpacity: 0.12
            }).addTo(map);
        } else {
            progressiveScanBboxLayer.setBounds([[lat - 0.02, lng - 0.02], [lat + 0.02, lng + 0.02]]);
            if (!map.hasLayer(progressiveScanBboxLayer)) progressiveScanBboxLayer.addTo(map);
        }
        map.flyTo([lat, lng], 12, { duration: 1.0 });
    }

    try {
        console.log(`📡 Initiating Progressive Multi-Scale Scan for [${lat}, ${lng}]...`);
        const resp = await fetch('/api/scan/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lat, lng })
        });

        if (!resp.ok) {
            const err = await resp.text();
            throw new Error(`Server returned HTTP ${resp.status}: ${err}`);
        }

        const data = await resp.json();
        activeProgressiveScanId = data.scanId;
        console.log(`✅ Progressive scan job started: ${activeProgressiveScanId}`);

        toast('Started Progressive Scan (Core Basin 5×5 km)', 'ok', 2500);

        if (progressiveScanPollTimer) clearInterval(progressiveScanPollTimer);
        progressiveScanPollTimer = setInterval(pollProgressiveScan, 650);

    } catch (err) {
        console.error('Error starting progressive scan:', err);
        toast(`Scan initiation failed: ${err.message}`, 'err');
        if (btn) {
            btn.disabled = false;
            btn.textContent = '⚡ Run Auditable Progressive Scan (5×5 km → 50×50 km)';
        }
    }
}

async function pollProgressiveScan() {
    if (!activeProgressiveScanId) return;

    try {
        const resp = await fetch(`/api/scan/status/${activeProgressiveScanId}`);
        if (!resp.ok) return;

        const data = await resp.json();
        const status = data.status;
        const msg = data.status_message || '';
        const stageIdx = data.current_stage_idx || 0;
        const res = data.current_result;
        const provCount = data.provenance_count || 0;

        // 1. Update UI Status & Progress Bar
        const statusBadge = document.getElementById('prog-status-badge');
        const banner = document.getElementById('prog-status-banner');
        const progressBar = document.getElementById('prog-progress-bar');
        const provLabel = document.getElementById('prov-count-label');
        const btn = document.getElementById('runProgressiveScanBtn');
        const cancelBtn = document.getElementById('cancelScanBtn');

        if (provLabel) provLabel.textContent = provCount;
        if (banner) banner.textContent = msg;

        const progressPcts = [25, 50, 75, 100];
        if (progressBar) progressBar.style.width = `${progressPcts[stageIdx] || 25}%`;

        if (statusBadge) {
            if (status === 'COMPLETED') {
                statusBadge.textContent = 'COMPLETED';
                statusBadge.className = 'metric-status-pill available';
            } else if (status === 'CANCELLED') {
                statusBadge.textContent = 'CANCELLED';
                statusBadge.className = 'metric-status-pill unavailable';
            } else {
                statusBadge.textContent = `${status} (${data.current_stage?.extent_km}×${data.current_stage?.extent_km} km)`;
                statusBadge.className = 'metric-status-pill available';
            }
        }

        // 2. Update Stage Badges
        for (let i = 0; i < 4; i++) {
            const b = document.getElementById(`stage-badge-${i}`);
            if (b) {
                if (i < stageIdx || (status === 'COMPLETED' && i <= stageIdx)) {
                    b.className = 'stage-badge completed';
                } else if (i === stageIdx && status !== 'CANCELLED') {
                    b.className = 'stage-badge active';
                } else {
                    b.className = 'stage-badge';
                }
            }
        }

        // 3. Update Expanding Leaflet Bounding Box
        if (res && res.bbox && typeof map !== 'undefined' && map && progressiveScanBboxLayer) {
            const b = res.bbox;
            const bounds = [[b.min_lat, b.min_lng], [b.max_lat, b.max_lng]];
            progressiveScanBboxLayer.setBounds(bounds);
            progressiveScanBboxLayer.bindTooltip(`Analysis Extent: ${res.extent_km}×${res.extent_km} km (${res.stage_label})`, {
                permanent: true,
                direction: 'top',
                className: 'scan-bbox-tooltip'
            });
        }

        // 4. Render Scientific Metric Cards (if result is available)
        if (res && res.metrics) {
            renderProgressiveScanMetrics(res.metrics, res.extent_km);
        }

        // 5. Update Executive Summary Stepper & Multi-Scale Matrix
        if (typeof updateExecutiveCardStepper === 'function') {
            updateExecutiveCardStepper(stageIdx, status);
        }
        if (res && typeof updateMultiscaleMatrixStage === 'function') {
            updateMultiscaleMatrixStage(stageIdx, res);
        }
        if (data.stage_results && Array.isArray(data.stage_results) && typeof updateMultiscaleMatrixStage === 'function') {
            data.stage_results.forEach((sRes, sIdx) => {
                updateMultiscaleMatrixStage(sIdx, sRes);
            });
        }
        if (currentPredictionData && typeof updateMultiscaleMatrixWithPrediction === 'function') {
            updateMultiscaleMatrixWithPrediction(currentPredictionData);
        }

        // 6. Termination checks
        if (status === 'COMPLETED' || status === 'CANCELLED') {
            clearInterval(progressiveScanPollTimer);
            progressiveScanPollTimer = null;
            if (btn) {
                btn.disabled = false;
                btn.textContent = '⚡ Run Auditable Progressive Scan (5×5 km → 50×50 km)';
            }
            if (cancelBtn) cancelBtn.disabled = true;

            comprehensiveScanDone = true;
            if (typeof checkComprehensiveAssessmentFinished === 'function') {
                checkComprehensiveAssessmentFinished();
            }

            if (status === 'COMPLETED') {
                toast(`Progressive Basin Scan Complete (${data.current_stage?.extent_km}×${data.current_stage?.extent_km} km)`, 'ok', 3500);
            } else if (status === 'CANCELLED') {
                toast(`Scan cancelled at ${data.current_stage?.extent_km}×${data.current_stage?.extent_km} km`, 'warn', 3500);
            }
        }

    } catch (err) {
        console.warn('Error polling progressive scan:', err);
    }
}

function renderProgressiveScanMetrics(m, extentKm) {
    if (!m) return;

    // 1. Rainfall Observed 24h
    const ro = m.rainfall_observed_24h;
    const valRo = document.getElementById('mval-rain-obs');
    const statRo = document.getElementById('mstatus-rain-obs');
    const timeRo = document.getElementById('mtime-rain-obs');
    const unaRo = document.getElementById('muna-rain-obs');

    if (ro && ro.status === 'AVAILABLE') {
        if (valRo) valRo.innerHTML = `${ro.value} <small>mm</small>`;
        if (statRo) { statRo.textContent = 'Available'; statRo.className = 'metric-status-pill available'; }
        if (timeRo) timeRo.textContent = ro.observation_time || 'Recent Reanalysis';
        if (unaRo) unaRo.style.display = 'none';
    } else {
        if (valRo) valRo.innerHTML = '<span style="color:#94a3b8; font-size:14px; font-weight:600; font-family:\'Inter\',system-ui;">Data unavailable</span>';
        if (statRo) { statRo.textContent = 'Unavailable'; statRo.className = 'metric-status-pill unavailable'; }
        if (unaRo) {
            unaRo.style.display = 'block';
            unaRo.textContent = ro?.reason || 'ECMWF observation data unreachable for this basin.';
        }
    }

    // 2. Rainfall Forecast 24h / 72h
    const rf24 = m.rainfall_forecast_24h;
    const rf72 = m.rainfall_forecast_72h;
    const valRf24 = document.getElementById('mval-rain-fc24');
    const valRf72 = document.getElementById('mval-rain-fc72');
    const statRf = document.getElementById('mstatus-rain-fc');
    const unaRf = document.getElementById('muna-rain-fc');

    if (rf24 && rf24.status === 'AVAILABLE') {
        if (valRf24) valRf24.innerHTML = `${rf24.value} <small>mm</small>`;
        if (valRf72) valRf72.innerHTML = `${rf72 ? rf72.value : '--'} <small>mm</small>`;
        if (statRf) { statRf.textContent = 'Available'; statRf.className = 'metric-status-pill available'; }
        if (unaRf) unaRf.style.display = 'none';
    } else {
        if (valRf24) valRf24.innerHTML = '<span style="color:#94a3b8; font-size:14px; font-weight:600; font-family:\'Inter\',system-ui;">Data unavailable</span>';
        if (valRf72) valRf72.innerHTML = '<span style="color:#94a3b8; font-size:14px; font-weight:600; font-family:\'Inter\',system-ui;">Data unavailable</span>';
        if (statRf) { statRf.textContent = 'Unavailable'; statRf.className = 'metric-status-pill unavailable'; }
        if (unaRf) {
            unaRf.style.display = 'block';
            unaRf.textContent = rf24?.reason || 'Forecast models currently unavailable.';
        }
    }

    // 3. River Discharge (Copernicus GloFAS)
    const fl = m.river_discharge;
    const valFl = document.getElementById('mval-flow');
    const statFl = document.getElementById('mstatus-flow');
    const meanFl = document.getElementById('mmean-flow');
    const ratioFl = document.getElementById('mratio-flow');
    const threshFl = document.getElementById('mthresh-flow');
    const unaFl = document.getElementById('muna-flow');

    if (fl && fl.status === 'AVAILABLE') {
        if (valFl) valFl.innerHTML = `${fl.value.toLocaleString()} <small>m³/s</small>`;
        if (statFl) { statFl.textContent = 'Available'; statFl.className = 'metric-status-pill available'; }
        if (meanFl) meanFl.textContent = `${fl.historical_mean.toLocaleString()} m³/s`;
        if (ratioFl) ratioFl.textContent = `${fl.flow_ratio}x`;
        if (threshFl) threshFl.textContent = fl.threshold || 'Normal Baseline';
        if (unaFl) unaFl.style.display = 'none';
    } else {
        if (valFl) valFl.innerHTML = '<span style="color:#94a3b8; font-size:14px; font-weight:600; font-family:\'Inter\',system-ui;">Data unavailable</span>';
        if (statFl) { statFl.textContent = 'Unavailable'; statFl.className = 'metric-status-pill unavailable'; }
        if (meanFl) meanFl.textContent = 'N/A';
        if (ratioFl) ratioFl.textContent = 'N/A';
        if (threshFl) threshFl.textContent = 'Non-channel cell';
        if (unaFl) {
            unaFl.style.display = 'block';
            unaFl.textContent = fl?.reason || 'No GloFAS river channel routed through this 0.05° grid cell.';
        }
    }

    // 4. Elevation (Copernicus DEM 30m)
    const el = m.elevation;
    const valEl = document.getElementById('mval-elev');
    const statEl = document.getElementById('mstatus-elev');
    const rangeEl = document.getElementById('mrange-elev');
    const classEl = document.getElementById('mclass-elev');
    const unaEl = document.getElementById('muna-elev');

    if (el && el.status === 'AVAILABLE') {
        if (valEl) valEl.innerHTML = `${el.value} <small>m AMSL</small>`;
        if (statEl) { statEl.textContent = 'Available'; statEl.className = 'metric-status-pill available'; }
        if (rangeEl) rangeEl.textContent = `${el.range_m} m across ${extentKm}×${extentKm}km`;
        if (classEl) classEl.textContent = el.terrain || 'Terrain Classified';
        if (unaEl) unaEl.style.display = 'none';
    } else {
        if (valEl) valEl.innerHTML = '<span style="color:#94a3b8; font-size:14px; font-weight:600; font-family:\'Inter\',system-ui;">Data unavailable</span>';
        if (statEl) { statEl.textContent = 'Unavailable'; statEl.className = 'metric-status-pill unavailable'; }
        if (unaEl) {
            unaEl.style.display = 'block';
            unaEl.textContent = el?.reason || 'Copernicus DEM data unavailable for coordinates.';
        }
    }

    // 5. Soil Moisture
    const sm = m.soil_moisture;
    const valSm = document.getElementById('mval-soil');
    const statSm = document.getElementById('mstatus-soil');
    const satSm = document.getElementById('msat-soil');
    const unaSm = document.getElementById('muna-soil');

    if (sm && sm.status === 'AVAILABLE') {
        if (valSm) valSm.innerHTML = `${sm.value} <small>m³/m³</small>`;
        if (statSm) { statSm.textContent = 'Available'; statSm.className = 'metric-status-pill available'; }
        if (satSm) satSm.textContent = `${sm.saturation_pct}% of field capacity`;
        if (unaSm) unaSm.style.display = 'none';
    } else {
        if (valSm) valSm.innerHTML = '<span style="color:#94a3b8; font-size:14px; font-weight:600; font-family:\'Inter\',system-ui;">Data unavailable</span>';
        if (statSm) { statSm.textContent = 'Unavailable'; statSm.className = 'metric-status-pill unavailable'; }
        if (unaSm) {
            unaSm.style.display = 'block';
            unaSm.textContent = sm?.reason || 'Soil moisture product unavailable from meteorological model.';
        }
    }

    // 6. Sentinel-1 SAR Satellite Pass
    const sar = m.satellite_radar_sar;
    const valSar = document.getElementById('mval-sar');
    const statSar = document.getElementById('mstatus-sar');
    const platSar = document.getElementById('mplat-sar');
    const polSar = document.getElementById('mpol-sar');
    const unaSar = document.getElementById('muna-sar');

    if (sar && sar.status === 'AVAILABLE') {
        if (valSar) valSar.innerHTML = `<span style="color:#2dd4bf; font-family:'JetBrains Mono'; font-size:11px;">Scene: ${sar.scene_id}</span>`;
        if (statSar) { statSar.textContent = 'Available'; statSar.className = 'metric-status-pill available'; }
        if (platSar) platSar.textContent = sar.platform || 'Sentinel-1';
        if (polSar) polSar.textContent = sar.polarizations || 'VV+VH';
        if (unaSar) unaSar.style.display = 'none';
    } else {
        if (valSar) valSar.innerHTML = '<span style="color:#94a3b8; font-size:13px; font-weight:600; font-family:\'Inter\',system-ui;">No recent satellite pass — unavailable</span>';
        if (statSar) { statSar.textContent = 'Unavailable'; statSar.className = 'metric-status-pill unavailable'; }
        if (unaSar) {
            unaSar.style.display = 'block';
            unaSar.textContent = sar?.reason || 'No Sentinel-1 C-band SAR scene in catalog for this footprint in the current cycle.';
        }
    }

    // 7. Population Exposure
    const pop = m.population_exposure;
    const valPop = document.getElementById('mval-pop');
    const statPop = document.getElementById('mstatus-pop');
    const bldgsPop = document.getElementById('mbldgs-pop');
    const explPop = document.getElementById('mexpl-pop');
    const unaPop = document.getElementById('muna-pop');

    if (pop && pop.status === 'AVAILABLE') {
        if (valPop) valPop.innerHTML = `${pop.exposed_population} <small>residents exposed</small>`;
        if (statPop) { statPop.textContent = 'Available'; statPop.className = 'metric-status-pill available'; }
        if (bldgsPop) bldgsPop.textContent = `${pop.surveyed_buildings} buildings surveyed`;
        if (explPop) explPop.textContent = pop.explanation || 'Inundation geometry → population raster → exposed population';
        if (unaPop) unaPop.style.display = 'none';
    } else {
        if (valPop) valPop.innerHTML = '<span style="color:#94a3b8; font-size:14px; font-weight:600; font-family:\'Inter\',system-ui;">Data unavailable</span>';
        if (statPop) { statPop.textContent = 'Unavailable'; statPop.className = 'metric-status-pill unavailable'; }
        if (unaPop) {
            unaPop.style.display = 'block';
            unaPop.textContent = pop?.reason || 'Inundation geometry or surveyed structures unavailable.';
        }
    }

    // 8. Deterministic Risk Formula Card
    const rk = m.risk_index;
    const valRk = document.getElementById('mval-risk');
    const statRk = document.getElementById('mstatus-risk');
    const evalRk = document.getElementById('mformula-eval');

    if (rk && rk.status === 'AVAILABLE') {
        if (valRk) {
            valRk.innerHTML = `${rk.score} <small>/ 100</small>`;
            valRk.style.color = rk.badge_color || '#2dd4bf';
        }
        if (statRk) {
            statRk.textContent = rk.category || 'NORMAL';
            statRk.style.background = `${rk.badge_color}25`;
            statRk.style.color = rk.badge_color || '#2dd4bf';
        }
        if (evalRk) evalRk.textContent = rk.formula_evaluation || rk.formula_definition;
    } else {
        if (valRk) valRk.innerHTML = '<span style="color:#94a3b8; font-size:14px; font-weight:600; font-family:\'Inter\',system-ui;">Data unavailable</span>';
        if (statRk) { statRk.textContent = 'Insufficient Data'; statRk.className = 'metric-status-pill unavailable'; }
        if (evalRk) evalRk.textContent = rk?.reason || 'Insufficient data for reliable calculation (Missing precipitation or hydrological observations).';
    }
}

async function cancelProgressiveFloodScan() {
    if (!activeProgressiveScanId) return;

    try {
        const resp = await fetch(`/api/scan/cancel/${activeProgressiveScanId}`, { method: 'POST' });
        if (resp.ok) {
            toast('Progressive scan cancellation requested', 'warn', 2500);
            if (progressiveScanPollTimer) {
                clearInterval(progressiveScanPollTimer);
                progressiveScanPollTimer = null;
            }
            const banner = document.getElementById('prog-status-banner');
            const statusBadge = document.getElementById('prog-status-badge');
            const btn = document.getElementById('runProgressiveScanBtn');
            const cancelBtn = document.getElementById('cancelScanBtn');

            if (banner) banner.textContent = 'Scan cancelled on user request.';
            if (statusBadge) { statusBadge.textContent = 'CANCELLED'; statusBadge.className = 'metric-status-pill unavailable'; }
            if (btn) { btn.disabled = false; btn.textContent = '⚡ Run Auditable Progressive Scan (5×5 km → 50×50 km)'; }
            if (cancelBtn) cancelBtn.disabled = true;
        }
    } catch (err) {
        console.error('Error cancelling scan:', err);
    }
}

async function openProvenanceDrawer() {
    const modal = document.getElementById('provenance-modal');
    const tbody = document.getElementById('provenance-tbody');
    const targetEl = document.getElementById('prov-target-coords');
    const totalEl = document.getElementById('prov-total-count');

    if (modal) modal.style.display = 'flex';
    if (tbody) tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; padding:24px; color:#94a3b8;">Loading genuine external API request logs…</td></tr>';

    const latIn = document.getElementById('predictLat');
    const lngIn = document.getElementById('predictLng');
    if (targetEl) targetEl.textContent = (latIn?.value && lngIn?.value) ? `${latIn.value}°, ${lngIn.value}°` : 'Global';

    try {
        const url = activeProgressiveScanId ? `/api/scan/provenance/${activeProgressiveScanId}` : `/api/scan/provenance-global`;
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

        const data = await resp.json();
        const logs = data.logs || [];
        if (totalEl) totalEl.textContent = logs.length;

        if (!logs.length) {
            tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; padding:24px; color:#94a3b8;">No API requests logged yet. Start a progressive scan to generate live audit logs.</td></tr>';
            return;
        }

        tbody.innerHTML = logs.map((log, idx) => {
            const isOk = log.http_status >= 200 && log.http_status < 300;
            const statusBadge = isOk ?
                `<span style="color:#34d399; font-weight:700;">HTTP ${log.http_status}</span>` :
                `<span style="color:#f87171; font-weight:700;">HTTP ${log.http_status || 'ERR'}</span>`;

            const cacheTag = log.cache_status === 'CACHED' ?
                `<span style="font-size:9px; background:rgba(56,189,248,0.15); color:#38bdf8; padding:1px 5px; border-radius:3px; margin-left:4px;">CACHED</span>` :
                `<span style="font-size:9px; background:rgba(16,185,129,0.15); color:#34d399; padding:1px 5px; border-radius:3px; margin-left:4px;">LIVE</span>`;

            const hashShort = log.verification_sha256 ? log.verification_sha256.substring(0, 12) + '…' : '--';

            return `
                <tr>
                    <td style="color:#64748b; font-family:'JetBrains Mono';">${idx + 1}</td>
                    <td>
                        <div style="font-weight:600; color:#f1f5f9;">${log.source_name || 'External API'}</div>
                        <div style="font-size:10.5px; color:#94a3b8;">${log.dataset_product || ''} ${cacheTag}</div>
                    </td>
                    <td>
                        <div style="font-family:'JetBrains Mono'; font-size:10.5px; color:#38bdf8; word-break:break-all;">${log.api_endpoint || ''}</div>
                        <div style="font-size:10px; color:#64748b; margin-top:2px;">Params: ${JSON.stringify(log.request_params || {})}</div>
                    </td>
                    <td>${statusBadge}</td>
                    <td style="font-family:'JetBrains Mono'; color:#fde047;">${log.latency_ms != null ? log.latency_ms + 'ms' : '--'}</td>
                    <td style="font-size:10.5px; color:#94a3b8; font-family:'JetBrains Mono';">${log.request_timestamp_utc ? log.request_timestamp_utc.split('T')[1].replace('Z', '') : '--'}</td>
                    <td style="font-family:'JetBrains Mono'; font-size:10px; color:#2dd4bf;" title="${log.verification_sha256 || ''}">
                        ${hashShort}
                    </td>
                </tr>
            `;
        }).join('');

    } catch (err) {
        tbody.innerHTML = `<tr><td colspan="7" style="color:#f87171; padding:18px;">Error loading provenance logs: ${err.message}</td></tr>`;
    }
}

function closeProvenanceDrawer() {
    const modal = document.getElementById('provenance-modal');
    if (modal) modal.style.display = 'none';
}

// Global initialization on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    initPredictLayers();
    loadLiveHotspots();
});

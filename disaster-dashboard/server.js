const express  = require('express');
const cors     = require('cors');
const multer   = require('multer');
const axios    = require('axios');
const FormData = require('form-data');
const path     = require('path');
const fs       = require('fs');
const { logScan, getHistory, deleteScan, clearHistory } = require('./history');
const mgmt = require('./management');
const predictionStore = require('./prediction_store');
const progressiveScanner = require('./progressive_scanner');

const app  = express();
const PORT = 3000;
const AI_API_URL = process.env.API_URL || 'http://127.0.0.1:8000';

// ── Middleware ────────────────────────────────────────────────────────────────
app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// Multer: keep uploaded files in memory
const upload = multer({ storage: multer.memoryStorage() });


// ── Upload & Analyze ──────────────────────────────────────────────────────────
// POST /api/analyze-satellite
// Receives a TIF from the frontend, forwards to Python AI, logs the result.
app.post('/api/analyze-satellite', upload.single('satelliteImage'), async (req, res) => {
    try {
        console.log(`🛰️  Received upload: ${req.file.originalname} — forwarding to AI Engine...`);

        const formData = new FormData();
        formData.append('file', req.file.buffer, { filename: req.file.originalname });

        const aiResponse = await axios.post(AI_API_URL + '/predict', formData, {
            headers: { ...formData.getHeaders() },
            timeout: 300_000,   // 5-minute cap — inference can be slow on CPU
        });

        const data = aiResponse.data;
        console.log('✅ AI Engine replied. Logging to history...');

        // Log to SQLite
        if (data.summary) {
            logScan({
                scan_type: 'upload',
                filename:  req.file.originalname,
                summary:   data.summary,
            });
        }

        res.json(data);

    } catch (error) {
        console.error('🚨 Error communicating with AI Engine (upload):', error.message);
        res.status(500).json({ error: 'Failed to process image through AI.' });
    }
});



// ── Live Satellite Scan ───────────────────────────────────────────────────────
// POST /api/live-analyze
// Receives bbox + centre point from frontend, forwards to Python AI, logs result.
app.post('/api/live-analyze', async (req, res) => {
    try {
        console.log('🛰️  Received live scan request — forwarding to AI Engine...', req.body);

        const { bbox, lat, lng } = req.body;

        const aiResponse = await axios.post(
            AI_API_URL + '/predict-live',
            { bbox },
            { timeout: 600_000 }  // 10-minute cap — live scan fetches satellite data
        );

        const data = aiResponse.data;
        console.log('✅ AI Engine replied. Logging to history...');

        // Log to SQLite — store the centre point so the history table is informative
        if (data.summary) {
            logScan({
                scan_type: 'live',
                lat:       lat  ?? null,
                lng:       lng  ?? null,
                summary:   data.summary,
            });
        }

        res.json(data);

    } catch (error) {
        console.error('🚨 Error communicating with AI Engine (live scan):', error.message);
        res.status(500).json({ error: 'Failed to run live scan through AI.' });
    }
});


// ── Unified Real-Time Flood Prediction & Disaster Management (Single Intelligent Feature) ──
// POST /api/unified-prediction
// Combines Live Telemetry (24 factors) + ECMWF Forecasts + GloFAS Hydrology +
// SAR U-Net Satellite Evidence + Spatial Impact + Emergency Management + MongoDB Storage
app.post('/api/unified-prediction', async (req, res) => {
    try {
        const { lat, lng, radius_km, run_sar_inference } = req.body;
        console.log(`🌊 Unified Prediction Request for [${lat}, ${lng}] (radius=${radius_km || 6}km, SAR=${!!run_sar_inference})...`);

        if (lat == null || lng == null) {
            return res.status(400).json({ error: 'lat and lng coordinates are required.' });
        }

        const aiResponse = await axios.post(
            AI_API_URL + '/unified-prediction',
            {
                lat: parseFloat(lat),
                lng: parseFloat(lng),
                radius_km: parseFloat(radius_km || 6.0),
                run_sar_inference: !!run_sar_inference
            },
            { timeout: 120_000 }
        );

        const data = aiResponse.data;
        if (data.status === 'success') {
            // Save complete prediction snapshot to MongoDB
            const savedDoc = await predictionStore.savePrediction({
                location: {
                    lat: parseFloat(lat),
                    lng: parseFloat(lng),
                    radius_km: parseFloat(radius_km || 6.0)
                },
                timestamp: data.timestamp || new Date().toISOString(),
                prediction: data.prediction,
                explainability: data.explainability,
                conditions_snapshot: data.factors,
                forecast_horizons: data.forecast_horizons,
                spatial_impact: data.spatial_impact,
                management_response: data.management_response,
                satellite_evidence: data.satellite_evidence,
                model_metadata: data.model_metadata
            });

            data._id = savedDoc._id;
            console.log(`💾 Persisted prediction to MongoDB (ID: ${savedDoc._id}, Risk: ${data.prediction?.risk_score}/100 [${data.prediction?.severity}])`);
        }

        res.json(data);
    } catch (error) {
        console.error('🚨 Error in unified prediction proxy:', error.message);
        res.status(500).json({ error: 'Failed to generate unified prediction: ' + error.message });
    }
});

// GET /api/prediction-history
app.get('/api/prediction-history', async (req, res) => {
    try {
        const { lat, lng, limit } = req.query;
        const history = await predictionStore.getHistory({ lat, lng, limit: limit || 20 });
        res.json({ history });
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch prediction history: ' + error.message });
    }
});

// GET /api/prediction-history/:id
app.get('/api/prediction-history/:id', async (req, res) => {
    try {
        const doc = await predictionStore.getPredictionById(req.params.id);
        if (!doc) return res.status(404).json({ error: 'Prediction record not found' });
        res.json(doc);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch prediction record: ' + error.message });
    }
});

// GET /api/prediction-stats
app.get('/api/prediction-stats', async (req, res) => {
    try {
        const stats = await predictionStore.getStats();
        res.json(stats);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch prediction stats: ' + error.message });
    }
});

// ── Autonomous Global Flood Prediction & Emerging Risk Detection ───────────────
// POST /api/global-prediction
// Triggers autonomous global catchment scan, batch weather/hydrology ingestion,
// dynamic ranking of top emerging risk zones, and saves snapshot to MongoDB.
app.post('/api/global-prediction', async (req, res) => {
    try {
        console.log('🌍 Autonomous Global Prediction Triggered...');
        const aiResponse = await axios.post(
            AI_API_URL + '/global-prediction',
            {},
            { timeout: 120_000 }
        );

        const data = aiResponse.data;
        if (data.status === 'success') {
            const savedDoc = await predictionStore.saveGlobalPrediction(data);
            data._id = savedDoc._id;
            console.log(`💾 Persisted Global Prediction to MongoDB (Scan ID: ${data.scan_id}, Emerging: ${data.summary?.emerging_threats_count}, Critical: ${data.summary?.critical_zones_count})`);
        }

        res.json(data);
    } catch (error) {
        console.error('🚨 Error in global prediction proxy:', error.message);
        res.status(500).json({ error: 'Failed to execute global prediction: ' + error.message });
    }
});

// GET /api/global-prediction/latest
// Returns the most recent global scan from MongoDB (or triggers a scan if none exist)
app.get('/api/global-prediction/latest', async (req, res) => {
    try {
        const latest = await predictionStore.getLatestGlobalPrediction();
        if (latest) {
            return res.json(latest);
        }

        // If no cached run exists in DB, trigger through AI engine
        console.log('🌍 No cached global prediction found in DB, requesting initial scan from AI Engine...');
        const aiResponse = await axios.get(
            AI_API_URL + '/global-prediction/latest',
            { timeout: 120_000 }
        );

        const data = aiResponse.data;
        if (data.status === 'success') {
            const savedDoc = await predictionStore.saveGlobalPrediction(data);
            data._id = savedDoc._id;
        }

        res.json(data);
    } catch (error) {
        console.error('🚨 Error fetching latest global prediction:', error.message);
        res.status(500).json({ error: 'Failed to retrieve latest global prediction: ' + error.message });
    }
});

// GET /api/global-prediction/history
// Returns global prediction history to observe how planetary flood risk evolves over time
app.get('/api/global-prediction/history', async (req, res) => {
    try {
        const limit = req.query.limit || 10;
        const history = await predictionStore.getGlobalHistory(limit);
        res.json({ history });
    } catch (error) {
        console.error('🚨 Error fetching global prediction history:', error.message);
        res.status(500).json({ error: 'Failed to retrieve global prediction history: ' + error.message });
    }
});
// GET /api/global-flood-hotspots
// Proxies live UN/EC GDACS Active Floods telemetry directly from AI API
app.get('/api/global-flood-hotspots', async (req, res) => {
    try {
        const aiResponse = await axios.get(AI_API_URL + '/global-flood-hotspots', { timeout: 30_000 });
        res.json(aiResponse.data);
    } catch (error) {
        console.error('🚨 Error fetching live flood hotspots from GDACS:', error.message);
        res.status(500).json({ error: 'Failed to retrieve live flood hotspots: ' + error.message });
    }
});

// ── Closed-Loop Forecast Verification Pipeline ─────────────────────────────────
// GET /api/global-prediction/verification-metrics
// Returns live verification metrics: Hits, Misses, False Alarms, CSI, Hit Rate
app.get('/api/global-prediction/verification-metrics', async (req, res) => {
    try {
        const metrics = await predictionStore.getVerificationMetrics();
        res.json(metrics);
    } catch (error) {
        console.error('🚨 Error fetching verification metrics:', error.message);
        res.status(500).json({ error: 'Failed to retrieve verification metrics: ' + error.message });
    }
});

// POST /api/global-prediction/verify
// Triggers an audit of past forecasts against latest observed river discharge & satellite alerts
app.post('/api/global-prediction/verify', async (req, res) => {
    try {
        console.log('🔍 Executing Closed-Loop Forecast Verification Audit...');
        // Ingest latest observed mesh state
        const latestScan = await predictionStore.getLatestGlobalPrediction();
        const observedMap = {};
        if (latestScan && latestScan.geojson?.features) {
            latestScan.geojson.features.forEach(f => {
                const p = f.properties || {};
                const hasGdacs = p.data_provenance?.satellite_alerts?.includes('OBSERVED') || (p.gdacs_alert != null);
                
                // Only provide genuine observations (satellite observation or in-situ gauge)
                // GloFAS is modelled reanalysis, so it is strictly excluded from observed ground truth
                if (hasGdacs) {
                    observedMap[p.id] = {
                        has_gdacs_alert: true,
                        satellite_pass_confirmed: true,
                        alert_level: p.gdacs_alert?.alertlevel || 'ORANGE'
                    };
                } else if (p.in_situ_gauge_stage_m !== undefined) {
                    observedMap[p.id] = {
                        in_situ_gauge_stage_m: p.in_situ_gauge_stage_m,
                        in_situ_gauge_flood: p.in_situ_gauge_flood || false
                    };
                }
                // If neither satellite pass nor in-situ gauge is available for this catchment,
                // observedMap[p.id] is left undefined, which causes it to be marked DATA_UNAVAILABLE
                // and excluded from accuracy metrics.
            });
        }

        const auditedCount = await predictionStore.auditAndEvaluateVerifications(observedMap);
        const updatedMetrics = await predictionStore.getVerificationMetrics();
        res.json({
            status: 'success',
            audited_checkpoints: auditedCount,
            metrics: updatedMetrics
        });
    } catch (error) {
        console.error('🚨 Error running verification audit:', error.message);
        res.status(500).json({ error: 'Failed to execute verification audit: ' + error.message });
    }
});

// ── Legacy Multi-Factor Flood Disaster Prediction (Backwards Compatibility) ──
app.post('/api/predict-disaster', async (req, res) => {
    try {
        const { lat, lng, radius_km, mc_samples } = req.body;
        console.log(`🔮 Real-Time Flood Prediction Request for [${lat}, ${lng}] (radius=${radius_km || 6}km, MC=${mc_samples || 20})...`);

        if (lat == null || lng == null) {
            return res.status(400).json({ error: 'lat and lng coordinates are required.' });
        }

        const aiResponse = await axios.post(
            AI_API_URL + '/predict-disaster',
            { lat: parseFloat(lat), lng: parseFloat(lng), radius_km: parseFloat(radius_km || 6.0), mc_samples: parseInt(mc_samples || 20, 10) },
            { timeout: 60_000 }
        );

        const data = aiResponse.data;
        console.log(`✅ Prediction Complete: Risk=${data.prediction?.risk_level} (${data.prediction?.flood_probability_pct}%), Zones=${data.zone_summary?.total_zones}`);

        // Log to SQLite scan history
        if (data.zone_summary) {
            logScan({
                scan_type: 'prediction',
                lat: parseFloat(lat),
                lng: parseFloat(lng),
                summary: {
                    total_flood_zones: data.zone_summary.total_zones,
                    total_area_km2: data.zone_summary.total_flooded_area_km2,
                    overall_severity: data.prediction?.risk_level || 'Normal',
                    danger_breakdown: {
                        Low: data.prediction?.risk_level === 'Normal' ? 1 : 0,
                        Medium: data.prediction?.risk_level === 'Advisory' ? 1 : 0,
                        High: data.prediction?.risk_level === 'Moderate' ? 1 : 0,
                        Critical: (data.prediction?.risk_level === 'High' || data.prediction?.risk_level === 'Critical') ? 1 : 0
                    }
                }
            });
        }

        res.json(data);
    } catch (error) {
        console.error('🚨 Error communicating with AI Engine (predict-disaster):', error.message);
        res.status(500).json({ error: 'Failed to generate flood prediction: ' + error.message });
    }
});

// GET /api/live-factors
app.get('/api/live-factors', async (req, res) => {
    try {
        const { lat, lng, radius_km } = req.query;
        if (!lat || !lng) return res.status(400).json({ error: 'lat and lng query params are required' });
        const aiResponse = await axios.get(AI_API_URL + '/live-factors', {
            params: { lat, lng, radius_km: radius_km || 6.0 },
            timeout: 25_000
        });
        res.json(aiResponse.data);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch live factors: ' + error.message });
    }
});

// GET /api/global-flood-hotspots
app.get('/api/global-flood-hotspots', async (req, res) => {
    try {
        const aiResponse = await axios.get(AI_API_URL + '/global-flood-hotspots', { timeout: 8000 });
        res.json(aiResponse.data);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch global hotspots: ' + error.message });
    }
});


// ── Health Check ────────────────────────────────────────────────────────────
// GET /api/health — reports whether the Python AI engine is reachable.
// Used by the frontend status dot.
app.get('/api/health', async (req, res) => {
    try {
        await axios.get(AI_API_URL + '/docs', { timeout: 3000 });
        res.json({ ai_engine: 'online' });
    } catch {
        res.status(503).json({ ai_engine: 'offline' });
    }
});


// ── Live Data (real-world floods from GDACS) ──────────────────────────────────
// GDACS = Global Disaster Alert & Coordination System (UN/EU). Free, no API key.
// We proxy it server-side so the browser doesn't hit CORS restrictions.
const GDACS_UA = { 'User-Agent': 'FloodWatch-AI/1.0 (educational project)' };

// GET /api/live-floods — list of currently active floods worldwide (as GeoJSON points).
app.get('/api/live-floods', async (req, res) => {
    try {
        const now = new Date();
        const todate = req.query.todate || now.toISOString().split('T')[0];
        
        // Dynamic default: past 90 days for recent active events
        const defaultFrom = new Date(now);
        defaultFrom.setDate(defaultFrom.getDate() - 90);
        const fromdate = req.query.fromdate || defaultFrom.toISOString().split('T')[0];
        
        const alertlevel = req.query.alertlevel || 'Green;Orange;Red';
        const pagesize = parseInt(req.query.pagesize, 10) || 100;
        const maxPages = parseInt(req.query.maxpages, 10) || 5;

        console.log(`🌐 Fetching live floods from GDACS (Date: ${fromdate} to ${todate}, AlertLevels: ${alertlevel}, maxPages: ${maxPages})...`);

        const allFeatures = [];
        const seenEventIds = new Set();
        const eventIdsReceived = [];

        for (let page = 1; page <= maxPages; page++) {
            try {
                const response = await axios.get(
                    'https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH',
                    {
                        params: {
                            eventlist: 'FL',
                            alertlevel,
                            fromdate,
                            todate,
                            pagesize,
                            pagenumber: page
                        },
                        headers: GDACS_UA,
                        timeout: 25_000
                    }
                );

                const features = response.data?.features || [];
                console.log(`   📄 Page ${page}: received ${features.length} features`);

                if (!features.length) break;

                for (const feat of features) {
                    const eventId = feat.properties?.eventid;
                    if (eventId && !seenEventIds.has(eventId)) {
                        seenEventIds.add(eventId);
                        eventIdsReceived.push(eventId);
                        allFeatures.push(feat);
                    }
                }

                if (features.length < pagesize) break;
            } catch (pageErr) {
                console.error(`   ⚠️ Page ${page} fetch error:`, pageErr.message);
                if (page === 1) throw pageErr;
                break;
            }
        }

        // Server-side diagnostic logging
        const target1103972 = allFeatures.find(f => f.properties?.eventid == 1103972);
        console.log(`📊 GDACS Flood Summary: ${allFeatures.length} unique events received worldwide.`);
        console.log(`   📋 Event IDs (${eventIdsReceived.length}): ${eventIdsReceived.slice(0, 15).join(', ')}${eventIdsReceived.length > 15 ? '...' : ''}`);
        if (target1103972) {
            const p = target1103972.properties || {};
            const coords = target1103972.geometry?.coordinates;
            console.log(`   ✅ Event 1103972 found -> Name: "${p.name}", Country: "${p.country}", Alert: "${p.alertlevel}", Dates: ${p.fromdate} -> ${p.todate}, Coords: [${coords ? coords.join(', ') : 'N/A'}]`);
        } else {
            console.log('   ℹ️ Event 1103972 was not found in the current fetched dataset.');
        }

        res.json({
            type: 'FeatureCollection',
            features: allFeatures
        });
    } catch (err) {
        console.error('🚨 GDACS list fetch failed:', err.message);
        res.status(502).json({ error: 'Could not reach GDACS live-flood feed.' });
    }
});

// GET /api/live-floods/shape?eventid=&episodeid= — real "affected area" polygon
// for one flood event, so we can draw its actual shape on the map.
app.get('/api/live-floods/shape', async (req, res) => {
    const { eventid, episodeid } = req.query;
    if (!eventid || !episodeid) {
        return res.status(400).json({ error: 'eventid and episodeid are required.' });
    }
    try {
        const r = await axios.get(
            'https://www.gdacs.org/gdacsapi/api/polygons/getgeometry',
            { params: { eventtype: 'FL', eventid, episodeid }, headers: GDACS_UA, timeout: 25_000 }
        );
        res.json(r.data);
    } catch (err) {
        console.error('🚨 GDACS shape fetch failed:', err.message);
        res.status(502).json({ error: 'Could not fetch flood shape from GDACS.' });
    }
});


// ── History Endpoints ─────────────────────────────────────────────────────────

// GET /api/history — return last 100 scans as JSON
app.get('/api/history', (req, res) => {
    try {
        const rows = getHistory();
        res.json(rows);
    } catch (err) {
        console.error('🚨 Failed to read history:', err.message);
        res.status(500).json({ error: 'Could not read scan history.' });
    }
});

// DELETE /api/history/:id — remove one scan
app.delete('/api/history/:id', (req, res) => {
    const id = parseInt(req.params.id, 10);
    if (isNaN(id)) return res.status(400).json({ error: 'Invalid id.' });

    const deleted = deleteScan(id);
    if (deleted) {
        res.json({ success: true });
    } else {
        res.status(404).json({ error: 'Scan not found.' });
    }
});

// DELETE /api/history — wipe all scans
app.delete('/api/history', (req, res) => {
    try {
        const count = clearHistory();
        res.json({ success: true, deleted: count });
    } catch (err) {
        console.error('🚨 Failed to clear history:', err.message);
        res.status(500).json({ error: 'Could not clear history.' });
    }
});


// ── Impact Assessment ───────────────────────────────────────────────────────────
// POST /api/assess-impact — forwards flood polygons to Python for impact analysis
app.post('/api/assess-impact', async (req, res) => {
    try {
        console.log('📊 Impact assessment request — forwarding to AI Engine...');
        
        const { flood_polygons, bbox, estimated_depth_m } = req.body;
        
        if (!flood_polygons || !Array.isArray(flood_polygons) || !bbox) {
            return res.status(400).json({ error: 'flood_polygons (array) and bbox are required.' });
        }

        const aiResponse = await axios.post(
            AI_API_URL + '/assess-impact',
            { flood_polygons, bbox, estimated_depth_m: estimated_depth_m ?? 1.0 },
            { timeout: 300_000 }  // 5 minutes — OSM queries can be slow
        );

        console.log('✅ AI Engine replied with impact assessment.');
        res.json(aiResponse.data);

    } catch (error) {
        console.error('🚨 Error in impact assessment:', error.message);
        res.status(500).json({ error: 'Failed to run impact assessment.' });
    }
});


// ── Flood Forecasting ───────────────────────────────────────────────────────────
// POST /api/forecast-flood — forwards flood polygons to Python for forecasting
app.post('/api/forecast-flood', async (req, res) => {
    try {
        console.log('🌧️ Flood forecast request — forwarding to AI Engine...');
        
        const { flood_polygons, bbox, horizons } = req.body;
        
        if (!flood_polygons || !Array.isArray(flood_polygons) || !bbox) {
            return res.status(400).json({ error: 'flood_polygons (array) and bbox are required.' });
        }

        const aiResponse = await axios.post(
            AI_API_URL + '/forecast-flood',
            { flood_polygons, bbox, horizons: horizons ?? [6, 12, 24, 48] },
            { timeout: 300_000 }  // 5 minutes
        );

        console.log('✅ AI Engine replied with flood forecast.');
        res.json(aiResponse.data);

    } catch (error) {
        console.error('🚨 Error in flood forecast:', error.message);
        res.status(500).json({ error: 'Failed to run flood forecast.' });
    }
});


// ── Response Optimization ────────────────────────────────────────────────────────
// POST /api/optimize-response — forwards impact + forecast to Python for action planning
app.post('/api/optimize-response', async (req, res) => {
    try {
        console.log('🎯 Response optimization request — forwarding to AI Engine...');
        
        const { impact, forecast, constraints } = req.body;
        
        if (!impact || !forecast) {
            return res.status(400).json({ error: 'impact and forecast are required.' });
        }

        const aiResponse = await axios.post(
            AI_API_URL + '/optimize-response',
            { impact, forecast, constraints: constraints ?? {} },
            { timeout: 60_000 }
        );

        console.log('✅ AI Engine replied with action plan.');
        res.json(aiResponse.data);

    } catch (error) {
        console.error('🚨 Error in response optimization:', error.message);
        res.status(500).json({ error: 'Failed to run response optimization.' });
    }
});

// Aliases for frontend & test convenience
app.post('/api/impact-assessment', (req, res) => res.redirect(307, '/api/assess-impact'));
app.post('/api/flood-forecast', (req, res) => res.redirect(307, '/api/forecast-flood'));
app.post('/api/action-plan', (req, res) => res.redirect(307, '/api/optimize-response'));

function extractCenter(geometry) {
    if (!geometry) return null;
    try {
        const coords = geometry.coordinates;
        if (!coords) return null;
        const flat = [];
        function walk(arr) {
            if (Array.isArray(arr) && arr.length >= 2 && typeof arr[0] === 'number') {
                flat.push(arr);
            } else if (Array.isArray(arr)) {
                arr.forEach(walk);
            }
        }
        walk(coords);
        if (flat.length > 0) {
            const sumLng = flat.reduce((s, p) => s + p[0], 0);
            const sumLat = flat.reduce((s, p) => s + p[1], 0);
            return [sumLat / flat.length, sumLng / flat.length];
        }
    } catch (e) {}
    return null;
}

// ── Copernicus GloFAS River Gauges & Hydrological Telemetry ────────────────
app.all(['/api/hydrology-station', '/api/drone-recon'], async (req, res) => {
    try {
        let lat = parseFloat(req.query?.lat || req.body?.lat);
        let lng = parseFloat(req.query?.lng || req.body?.lng);

        if (isNaN(lat) || isNaN(lng)) {
            const flood_features = req.body?.flood_features;
            if (Array.isArray(flood_features) && flood_features.length > 0) {
                let totalLat = 0, totalLng = 0, count = 0;
                flood_features.forEach(f => {
                    const c = f.properties?.center || extractCenter(f.geometry);
                    if (c && Array.isArray(c) && c.length >= 2) {
                        totalLat += c[0]; totalLng += c[1]; count++;
                    }
                });
                if (count > 0) { lat = totalLat / count; lng = totalLng / count; }
            }
        }

        if (isNaN(lat) || isNaN(lng)) {
            return res.status(400).json({ error: 'Valid latitude and longitude are required for live Copernicus GloFAS telemetry.' });
        }

        console.log(`🌊 Fetching live Copernicus GloFAS river discharge & telemetry for [${lat.toFixed(4)}, ${lng.toFixed(4)}]...`);

        // 1. Copernicus GloFAS River Discharge via Open-Meteo Flood API
        const glofasUrl = `https://flood-api.open-meteo.com/v1/flood?latitude=${lat.toFixed(4)}&longitude=${lng.toFixed(4)}&daily=river_discharge,river_discharge_mean,river_discharge_max,river_discharge_median&forecast_days=7`;
        
        // 2. ECMWF Atmospheric Weather & Soil Moisture
        const weatherUrl = `https://api.open-meteo.com/v1/forecast?latitude=${lat.toFixed(4)}&longitude=${lng.toFixed(4)}&current=precipitation,rain,wind_speed_10m&hourly=precipitation,soil_moisture_0_to_10cm&forecast_days=2`;

        // 3. Copernicus DEM Elevation
        const demUrl = `https://api.open-meteo.com/v1/elevation?latitude=${lat.toFixed(4)}&longitude=${lng.toFixed(4)}`;

        const [glofasRes, weatherRes, demRes] = await Promise.allSettled([
            axios.get(glofasUrl, { timeout: 8000, headers: { 'User-Agent': 'FloodWatch/2.0' } }),
            axios.get(weatherUrl, { timeout: 8000, headers: { 'User-Agent': 'FloodWatch/2.0' } }),
            axios.get(demUrl, { timeout: 6000, headers: { 'User-Agent': 'FloodWatch/2.0' } })
        ]);

        const glofasData = glofasRes.status === 'fulfilled' ? glofasRes.value.data : null;
        const weatherData = weatherRes.status === 'fulfilled' ? weatherRes.value.data : null;
        const demData = demRes.status === 'fulfilled' ? demRes.value.data : null;

        const dates = glofasData?.daily?.time || [];
        const discharges = glofasData?.daily?.river_discharge || [];
        const means = glofasData?.daily?.river_discharge_mean || [];
        const maxes = glofasData?.daily?.river_discharge_max || [];

        const currentDischarge = discharges[0] != null ? discharges[0] : 0.0;
        const currentMean = means[0] != null ? means[0] : (discharges[0] ?? 0.0);
        const peakDischarge = discharges.length > 0 ? Math.max(...discharges.filter(v => v != null), currentDischarge) : currentDischarge;

        const soilMoisture = weatherData?.hourly?.soil_moisture_0_to_10cm?.[0] ?? 0.25;
        const currentPrecip = weatherData?.current?.precipitation ?? 0.0;
        const elevationM = demData?.elevation?.[0] ?? 0.0;

        const flowRatio = currentMean > 0 ? (currentDischarge / currentMean) : 1.0;
        const flowAnomalyPct = ((currentDischarge - currentMean) / (currentMean || 1)) * 100;

        let alertStage = 'NORMAL';
        let alertBadge = 'Normal Seasonal Flow';
        let alertColor = '#10b981';
        let alertDescription = 'River flow is within normal historical baseline for this catchment basin.';

        if (flowRatio >= 1.85) {
            alertStage = 'SEVERE_FLOOD';
            alertBadge = 'Severe Flood Warning (5-Yr+ Recurrence)';
            alertColor = '#ef4444';
            alertDescription = 'River discharge significantly exceeds critical threshold. High probability of extensive floodplain inundation.';
        } else if (flowRatio >= 1.45) {
            alertStage = 'MODERATE_FLOOD';
            alertBadge = 'Flood Advisory (2-Yr Recurrence)';
            alertColor = '#f97316';
            alertDescription = 'River discharge exceeds 2-year recurrence capacity. Embankments and low-lying agrarian zones at risk.';
        } else if (flowRatio >= 1.15) {
            alertStage = 'BANKFULL';
            alertBadge = 'Bankfull / High Flow Alert';
            alertColor = '#eab308';
            alertDescription = 'River channel flowing near full capacity. Minor bank overtopping expected in localized depression zones.';
        }

        const forecastSeries = dates.map((date, idx) => ({
            date,
            discharge_m3s: discharges[idx] != null ? parseFloat(discharges[idx].toFixed(1)) : null,
            mean_m3s: means[idx] != null ? parseFloat(means[idx].toFixed(1)) : null,
            max_m3s: maxes[idx] != null ? parseFloat(maxes[idx].toFixed(1)) : null
        }));

        const responseData = {
            station: {
                name: `Copernicus GloFAS River Gauging Station`,
                lat: parseFloat(lat.toFixed(4)),
                lng: parseFloat(lng.toFixed(4)),
                elevation_m: elevationM,
                coordinates: `${Math.abs(lat).toFixed(4)}° ${lat >= 0 ? 'N' : 'S'}, ${Math.abs(lng).toFixed(4)}° ${lng >= 0 ? 'E' : 'W'}`
            },
            hydrology: {
                current_discharge_m3s: parseFloat(currentDischarge.toFixed(1)),
                historical_mean_m3s: parseFloat(currentMean.toFixed(1)),
                peak_forecast_m3s: parseFloat(peakDischarge.toFixed(1)),
                flow_anomaly_pct: parseFloat(flowAnomalyPct.toFixed(1)),
                alert_stage: alertStage,
                alert_badge: alertBadge,
                alert_color: alertColor,
                alert_description: alertDescription,
                soil_moisture_m3m3: parseFloat(soilMoisture.toFixed(3)),
                precipitation_rate_mmh: parseFloat(currentPrecip.toFixed(1))
            },
            forecast_7day: forecastSeries,
            data_source: {
                river_model: 'European Commission Copernicus GloFAS (ECMWF)',
                atmospheric_model: 'ECMWF Integrated Forecasting System (IFS)',
                elevation_model: 'Copernicus 30m Global DEM',
                is_live_telemetry: true,
                attribution: 'Open Access / European Centre for Medium-Range Weather Forecasts'
            },
            // Compatibility mappings
            sector: {
                name: `Copernicus GloFAS Telemetry Basin`,
                coordinates: `${Math.abs(lat).toFixed(4)}° ${lat >= 0 ? 'N' : 'S'}, ${Math.abs(lng).toFixed(4)}° ${lng >= 0 ? 'E' : 'W'}`
            },
            recommendation: alertDescription
        };

        res.json(responseData);

    } catch (error) {
        console.error('🚨 Error in hydrology station:', error);
        res.status(500).json({ error: 'Failed to fetch Copernicus GloFAS river telemetry: ' + error.message });
    }
});

// Operational Dispatch to Designated Shelter / Community Facility
app.post('/api/dispatch-shelter', (req, res) => {
    try {
        const { shelter_name, location, supplies, personnel_name, personnel_role } = req.body;
        console.log(`📦 Operational Dispatch to: ${shelter_name} at [${location}]`);

        if (supplies && Array.isArray(supplies)) {
            supplies.forEach(s => {
                mgmt.addInventory({
                    item_name: s.item_name || 'Emergency Rations',
                    quantity: s.quantity || 100,
                    location: `${shelter_name}`
                });
            });
        }

        if (personnel_name) {
            mgmt.addPersonnel({
                name: personnel_name,
                role: personnel_role || 'rescue',
                status: 'deployed',
                assigned_zone: shelter_name
            });
        }

        res.json({
            success: true,
            shelter: shelter_name,
            timestamp: new Date().toISOString()
        });
    } catch (err) {
        res.status(500).json({ error: 'Failed to record shelter dispatch: ' + err.message });
    }
});


// ── Agency Management Endpoints ───────────────────────────────────────────────

app.get('/api/inventory', (req, res) => res.json(mgmt.listInventory()));
app.post('/api/inventory', (req, res) => res.json({ id: mgmt.addInventory(req.body) }));
app.delete('/api/inventory/:id', (req, res) => res.json({ success: mgmt.removeInventory(req.params.id) }));

app.get('/api/evacuees', (req, res) => res.json(mgmt.listEvacuees()));
app.post('/api/evacuees', (req, res) => res.json({ id: mgmt.addEvacuee(req.body) }));
app.delete('/api/evacuees/:id', (req, res) => res.json({ success: mgmt.removeEvacuee(req.params.id) }));

app.get('/api/personnel', (req, res) => res.json(mgmt.listPersonnel()));
app.post('/api/personnel', (req, res) => res.json({ id: mgmt.addPersonnel(req.body) }));
// ── Progressive Scan & Provenance Audit Endpoints ─────────────────────────
app.post('/api/scan/start', (req, res) => {
    try {
        const { lat, lng } = req.body;
        const latNum = parseFloat(lat);
        const lngNum = parseFloat(lng);
        if (isNaN(latNum) || isNaN(lngNum) || latNum < -90 || latNum > 90 || lngNum < -180 || lngNum > 180) {
            return res.status(400).json({ error: 'Valid latitude (-90..90) and longitude (-180..180) are required.' });
        }
        const scanId = progressiveScanner.startProgressiveScan(latNum, lngNum);
        console.log(`📡 Started progressive scan [${scanId}] for coords [${latNum.toFixed(4)}, ${lngNum.toFixed(4)}]`);
        res.json({
            success: true,
            scanId: scanId,
            message: 'Progressive scan initialized (Stage 1: 5×5 km Core Basin)'
        });
    } catch (err) {
        console.error('🚨 Error starting progressive scan:', err);
        res.status(500).json({ error: 'Failed to initiate progressive scan: ' + err.message });
    }
});

app.get('/api/scan/status/:scanId', (req, res) => {
    try {
        const status = progressiveScanner.getScanStatus(req.params.scanId);
        if (!status) {
            return res.status(404).json({ error: 'Scan job not found' });
        }
        res.json(status);
    } catch (err) {
        res.status(500).json({ error: 'Failed to retrieve scan status: ' + err.message });
    }
});

app.post('/api/scan/cancel/:scanId', (req, res) => {
    try {
        const success = progressiveScanner.cancelScan(req.params.scanId);
        if (!success) {
            return res.status(404).json({ error: 'Scan job not found' });
        }
        console.log(`⏹️ Cancelled progressive scan [${req.params.scanId}]`);
        res.json({ success: true, message: 'Scan cancelled successfully' });
    } catch (err) {
        res.status(500).json({ error: 'Failed to cancel scan: ' + err.message });
    }
});

app.get('/api/scan/provenance/:scanId', (req, res) => {
    try {
        const logs = progressiveScanner.getScanProvenance(req.params.scanId);
        res.json({ scanId: req.params.scanId, count: logs.length, logs });
    } catch (err) {
        res.status(500).json({ error: 'Failed to fetch provenance logs: ' + err.message });
    }
});

app.get('/api/scan/provenance-global', (req, res) => {
    try {
        const logs = progressiveScanner.getGlobalProvenanceLogs();
        res.json({ count: logs.length, logs });
    } catch (err) {
        res.status(500).json({ error: 'Failed to fetch global provenance logs: ' + err.message });
    }
});

// ── Multi-Scale Progressive Analysis Endpoints (5x5 -> 10x10 -> 25x25 -> 50x50 km) ──
app.post('/api/sar/multiscale', async (req, res) => {
    try {
        const { lat, lng } = req.body;
        if (lat == null || lng == null) {
            return res.status(400).json({ error: 'lat and lng coordinates are required.' });
        }
        console.log(`🛰️ Multi-Scale SAR scan request for [${lat}, ${lng}]...`);
        const aiResponse = await axios.post(
            AI_API_URL + '/api/sar-multiscale',
            { lat: parseFloat(lat), lng: parseFloat(lng) },
            { timeout: 180_000 }
        );
        res.json(aiResponse.data);
    } catch (error) {
        console.error('🚨 Error in multi-scale SAR proxy:', error.message);
        res.status(500).json({ error: 'Failed to execute multi-scale SAR scan: ' + error.message });
    }
});

app.post('/api/impact/multiscale', async (req, res) => {
    try {
        const { lat, lng } = req.body;
        if (lat == null || lng == null) {
            return res.status(400).json({ error: 'lat and lng coordinates are required.' });
        }
        console.log(`📊 Multi-Scale Impact assessment request for [${lat}, ${lng}]...`);
        const aiResponse = await axios.post(
            AI_API_URL + '/api/impact-multiscale',
            { lat: parseFloat(lat), lng: parseFloat(lng) },
            { timeout: 180_000 }
        );
        res.json(aiResponse.data);
    } catch (error) {
        console.error('🚨 Error in multi-scale Impact proxy:', error.message);
        res.status(500).json({ error: 'Failed to execute multi-scale impact assessment: ' + error.message });
    }
});

app.post('/api/action/multiscale', async (req, res) => {
    try {
        const { lat, lng, budget } = req.body;
        if (lat == null || lng == null) {
            return res.status(400).json({ error: 'lat and lng coordinates are required.' });
        }
        console.log(`🎯 Multi-Scale Action Plan request for [${lat}, ${lng}] (budget=$${budget})...`);
        const aiResponse = await axios.post(
            AI_API_URL + '/api/action-multiscale',
            { lat: parseFloat(lat), lng: parseFloat(lng), budget: budget ? parseFloat(budget) : 100000.0 },
            { timeout: 180_000 }
        );
        res.json(aiResponse.data);
    } catch (error) {
        console.error('🚨 Error in multi-scale Action Plan proxy:', error.message);
        res.status(500).json({ error: 'Failed to execute multi-scale action plan: ' + error.message });
    }
});

app.post('/api/progressive/stage', async (req, res) => {
    try {
        const { lat, lng, stage_idx, feature, estimated_depth_m, budget_usd } = req.body;
        if (lat == null || lng == null) {
            return res.status(400).json({ error: 'lat and lng coordinates are required.' });
        }
        const aiResponse = await axios.post(
            AI_API_URL + '/api/progressive-stage',
            {
                lat: parseFloat(lat),
                lng: parseFloat(lng),
                stage_idx: parseInt(stage_idx || 0, 10),
                feature: feature || 'all',
                estimated_depth_m: parseFloat(estimated_depth_m || 1.0),
                budget_usd: parseFloat(budget_usd || 100000.0)
            },
            { timeout: 90_000 }
        );
        res.json(aiResponse.data);
    } catch (error) {
        console.error('🚨 Error in progressive stage proxy:', error.message);
        res.status(500).json({ error: 'Failed to execute progressive stage: ' + error.message });
    }
});

// ── Start ─────────────────────────────────────────────────────────────────────
app.listen(PORT, async () => {
    console.log(`🌍 Main Dashboard Server running on http://localhost:${PORT}`);
    try {
        await predictionStore.initMongo();
    } catch (e) {
        console.warn('MongoDB initialization notice:', e.message);
    }
});

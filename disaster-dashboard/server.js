const express  = require('express');
const cors     = require('cors');
const multer   = require('multer');
const axios    = require('axios');
const FormData = require('form-data');
const path     = require('path');
const { logScan, getHistory, deleteScan, clearHistory } = require('./history');

const app  = express();
const PORT = 3000;

// ── Middleware ────────────────────────────────────────────────────────────────
app.use(cors());
app.use(express.json());
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

        const aiResponse = await axios.post('http://127.0.0.1:8000/predict', formData, {
            headers: { ...formData.getHeaders() },
            timeout: 120_000,   // 2-minute cap — inference can be slow on CPU
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
            'http://127.0.0.1:8000/predict-live',
            { bbox },
            { timeout: 180_000 }  // 3-minute cap — live scan fetches satellite data
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


// ── Health Check ────────────────────────────────────────────────────────────
// GET /api/health — reports whether the Python AI engine is reachable.
// Used by the frontend status dot.
app.get('/api/health', async (req, res) => {
    try {
        await axios.get('http://127.0.0.1:8000/docs', { timeout: 3000 });
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
        console.log('🌐 Fetching live floods from GDACS...');
        const r = await axios.get(
            'https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH?eventlist=FL',
            { headers: GDACS_UA, timeout: 25_000 }
        );
        res.json(r.data);
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


// ── Start ─────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
    console.log(`🌍 Main Dashboard Server running on http://localhost:${PORT}`);
});

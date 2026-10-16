/**
 * disaster-dashboard/alerts/routes.js — Alert API routes (Phase 6)
 *
 * Express routes for the Global Mesh alert service:
 *   GET  /api/alert/candidates        - List pending alert candidates
 *   POST  /api/alert/decide           - Approve/reject/cancel a candidate
 *   POST  /api/alert/cap/export       - Export approved alerts as CAP 1.2
 *
 * Backed by MongoDB collection global_mesh_alert_candidates (populated by the
 * Python AI engine) and local device registry for delivery targeting.
 */

const express = require('express');
const { MongoClient } = require('mongodb');
const { DeviceRegistry } = require('./device_registry');
const { AlertPolicyEngine } = require('./policy_engine');

const router = express.Router();

const MONGO_URI = process.env.MONGO_URI || 'mongodb://127.0.0.1:27017';
const DB_NAME = process.env.MONGO_DB || 'floodwatch';

let client, db, alertsCol, deviceRegistry, policyEngine;
let initialized = false;

async function init() {
  if (initialized) return;
  client = new MongoClient(MONGO_URI, {
    serverSelectionTimeoutMS: 5000,
    connectTimeoutMS: 5000,
  });
  await client.connect();
  db = client.db(DB_NAME);
  alertsCol = db.collection('global_mesh_alert_candidates');
  deviceRegistry = new DeviceRegistry();
  await deviceRegistry.connect();
  policyEngine = new AlertPolicyEngine();
  initialized = true;
  console.log('✅ Alert service initialized');
}

// GET /api/alert/candidates - list pending alert candidates
router.get('/candidates', async (req, res) => {
  try {
    await init();
    const { status, tier, limit = 50 } = req.query;
    const filter = {};
    if (status) filter.status = status;
    if (tier) filter.risk_tier = tier;
    const candidates = await alertsCol.find(filter).sort({ timestamp_utc: -1 }).limit(parseInt(limit)).toArray();
    res.json({ candidates, count: candidates.length });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// POST /api/alert/decide - approve/reject/cancel a candidate
router.post('/decide', async (req, res) => {
  try {
    await init();
    const { candidate_id, action } = req.body;
    if (!candidate_id || !action) {
      return res.status(400).json({ error: 'candidate_id and action are required' });
    }
    if (!['APPROVE', 'REJECT', 'CANCEL'].includes(action)) {
      return res.status(400).json({ error: 'Invalid action: ' + action });
    }

    // Find the candidate
    const candidate = await alertsCol.findOne({ candidate_id });
    if (!candidate) {
      return res.status(404).json({ error: 'Candidate not found' });
    }

    // Evaluate through policy engine
    const decision = policyEngine.evaluate(candidate);

    // Override with user decision if they want to override
    let finalAction = decision.action;
    let reason = decision.reason;
    if (action === 'APPROVE' && candidate.trust_flag) {
      finalAction = 'ESCALATE';
      reason = 'Manually approved by operator';
    } else if (action === 'REJECT') {
      finalAction = 'NO_CHANGE';
      reason = 'Manually rejected by operator';
    } else if (action === 'CANCEL') {
      finalAction = 'ALL_CLEAR';
      reason = 'Manually cancelled by operator';
    }

    // Update the candidate in MongoDB
    await alertsCol.updateOne(
      { candidate_id },
      {
        $set: {
          status: finalAction,
          decision_reason: reason,
          decision_timestamp: new Date().toISOString(),
          policy_decision: decision,
        }
      }
    );

    res.json({
      candidate_id,
      action: finalAction,
      reason,
      policy_decision: decision,
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// POST /api/alert/cap/export - export approved alerts as CAP 1.2
router.post('/cap/export', async (req, res) => {
  try {
    await init();
    const { candidate_ids } = req.body;
    const filter = candidate_ids ? { candidate_id: { $in: candidate_ids }, status: 'APPROVED' } : { status: 'APPROVED' };
    const approved = await alertsCol.find(filter).toArray();

    if (!approved.length) {
      return res.status(400).json({ error: 'No approved alerts to export' });
    }

    const now = new Date().toISOString();
    const alertIdentifier = `URN:OASIS:EMERGENCY:CAP:GLOBAL_MESH:${now.slice(0,13).replace(/[-:]/g,'')}`;

    const info = approved.map(a => ({
      language: 'en',
      category: 'Met',
      event: 'Flood Alert',
      responseType: a.risk_tier === 'IMMINENT' ? 'Evacuate' : a.risk_tier === 'WARNING' ? 'Prepare' : 'Monitor',
      priority: a.risk_tier === 'IMMINENT' ? 'Immediate' : a.risk_tier === 'WARNING' ? 'Expected' : 'Minor',
      urgency: a.risk_tier === 'IMMINENT' ? 'Immediate' : a.risk_tier === 'WARNING' ? 'Expected' : 'Minor',
      severity: a.severity || 'Moderate',
      certainty: a.trust_flag ? 'Observed' : 'Likely',
      headline: `Flood Alert — ${a.region_name || a.cell_id}`,
      description: `Flood risk predicted for ${a.region_name || a.cell_id} (basin: ${a.basin}). Probability: ${(a.predicted_flood_probability || 0).toFixed(0)}%. ${a.primary_driver || ''}`,
      instruction: 'Monitor local conditions and follow official evacuation instructions.',
      area: [{
        areaDesc: `${a.country || 'Unknown'}, ${a.basin || 'Unknown'}`,
        polygon: `${(a.lat||0).toFixed(4)} ${(a.lon||0).toFixed(4)}`,
        altitude: '0',
        ceiling: '0',
      }],
    }));

    const capXml = `<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>${alertIdentifier}</identifier>
  <sender>global-mesh@floodwatch.ai</sender>
  <sent>${now}</sent>
  <status>Actual</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
  <references></references>
  <info>
${approved.map(a => `    <info>
      <language>en</language>
      <category>Met</category>
      <event>Flood Alert</event>
      <responseType>${a.risk_tier === 'IMMINENT' ? 'Evacuate' : 'Prepare'}</responseType>
      <priority>${a.risk_tier === 'IMMINENT' ? 'Immediate' : 'Expected'}</priority>
      <urgency>${a.risk_tier === 'IMMINENT' ? 'Immediate' : 'Expected'}</urgency>
      <severity>${a.severity || 'Moderate'}</severity>
      <certainty>${a.trust_flag ? 'Observed' : 'Likely'}</certainty>
      <headline>Flood Alert — ${a.region_name || a.cell_id}</headline>
      <description>Flood risk predicted for ${a.region_name || a.cell_id} (basin: ${a.basin}). Probability: ${(a.predicted_flood_probability || 0).toFixed(0)}%. ${a.primary_driver || ''}</description>
      <instruction>Monitor local conditions and follow official evacuation instructions.</instruction>
      <area>
        <areaDesc>${a.country || 'Unknown'}, ${a.basin || 'Unknown'}</areaDesc>
        <polygon>${(a.lat||0).toFixed(4)} ${(a.lon||0).toFixed(4)}</polygon>
        <altitude>0</altitude>
        <ceiling>0</ceiling>
      </area>
    </info>`).join('\n  ')}
  </info>
</alert>`;

    res.json({ count: approved.length, cap_xml });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

module.exports = router;

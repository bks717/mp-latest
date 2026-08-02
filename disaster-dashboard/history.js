/**
 * history.js — SQLite scan history module
 * Uses better-sqlite3 (synchronous, no async drama)
 *
 * Table: scans
 *   id            INTEGER PRIMARY KEY
 *   timestamp     TEXT    — ISO 8601 UTC string
 *   scan_type     TEXT    — 'live' or 'upload'
 *   lat           REAL    — centre lat (null for uploads)
 *   lng           REAL    — centre lng (null for uploads)
 *   filename      TEXT    — original filename (null for live scans)
 *   zones         INTEGER — total flood zones detected
 *   area_km2      REAL    — total flooded area in km²
 *   severity      TEXT    — overall severity label
 *   danger_low    INTEGER — count of Low zones
 *   danger_medium INTEGER — count of Medium zones
 *   danger_high   INTEGER — count of High zones
 *   danger_critical INTEGER — count of Critical zones
 */

const Database = require('better-sqlite3');
const path     = require('path');

// Database lives next to server.js
const DB_PATH = path.join(__dirname, 'scans.db');

// Open (or create) the database
const db = new Database(DB_PATH);

// Enable WAL mode for better concurrent read performance
db.pragma('journal_mode = WAL');

// Create the table if it doesn't already exist
db.exec(`
    CREATE TABLE IF NOT EXISTS scans (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp        TEXT    NOT NULL,
        scan_type        TEXT    NOT NULL,
        lat              REAL,
        lng              REAL,
        filename         TEXT,
        zones            INTEGER NOT NULL DEFAULT 0,
        area_km2         REAL    NOT NULL DEFAULT 0.0,
        severity         TEXT    NOT NULL DEFAULT 'None',
        danger_low       INTEGER NOT NULL DEFAULT 0,
        danger_medium    INTEGER NOT NULL DEFAULT 0,
        danger_high      INTEGER NOT NULL DEFAULT 0,
        danger_critical  INTEGER NOT NULL DEFAULT 0
    )
`);

console.log(`📦 Scan history DB ready: ${DB_PATH}`);

// ── Prepared statements (compiled once, reused) ──────────────────────────────

const insertScan = db.prepare(`
    INSERT INTO scans
        (timestamp, scan_type, lat, lng, filename, zones, area_km2,
         severity, danger_low, danger_medium, danger_high, danger_critical)
    VALUES
        (@timestamp, @scan_type, @lat, @lng, @filename, @zones, @area_km2,
         @severity, @danger_low, @danger_medium, @danger_high, @danger_critical)
`);

const selectAll = db.prepare(`
    SELECT * FROM scans ORDER BY id DESC LIMIT 100
`);

const selectById = db.prepare(`
    SELECT * FROM scans WHERE id = ?
`);

const deleteById = db.prepare(`
    DELETE FROM scans WHERE id = ?
`);

const clearAll = db.prepare(`
    DELETE FROM scans
`);

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Log a completed scan to the database.
 *
 * @param {object} opts
 * @param {'live'|'upload'} opts.scan_type
 * @param {number|null}     opts.lat       — centre latitude  (live scans)
 * @param {number|null}     opts.lng       — centre longitude (live scans)
 * @param {string|null}     opts.filename  — original TIF name (upload scans)
 * @param {object}          opts.summary   — the summary object from the AI response
 * @returns {number} the new row id
 */
function logScan({ scan_type, lat = null, lng = null, filename = null, summary = {} }) {
    const breakdown = summary.danger_breakdown || {};
    const row = {
        timestamp:       new Date().toISOString(),
        scan_type,
        lat,
        lng,
        filename:        filename || null,
        zones:           summary.total_flood_zones  ?? 0,
        area_km2:        summary.total_area_km2     ?? 0.0,
        severity:        summary.overall_severity   ?? 'None',
        danger_low:      breakdown.Low              ?? 0,
        danger_medium:   breakdown.Medium           ?? 0,
        danger_high:     breakdown.High             ?? 0,
        danger_critical: breakdown.Critical         ?? 0,
    };
    const result = insertScan.run(row);
    return result.lastInsertRowid;
}

/**
 * Return the last 100 scans, newest first.
 * @returns {object[]}
 */
function getHistory() {
    return selectAll.all();
}

/**
 * Delete a single scan by id.
 * @param {number} id
 * @returns {boolean} true if a row was deleted
 */
function deleteScan(id) {
    const result = deleteById.run(id);
    return result.changes > 0;
}

/**
 * Wipe all history (used by the "Clear All" button in the UI).
 * @returns {number} number of rows deleted
 */
function clearHistory() {
    const result = clearAll.run();
    return result.changes;
}

module.exports = { logScan, getHistory, deleteScan, clearHistory };

const Database = require('better-sqlite3');
const path     = require('path');

const DB_PATH = path.join(__dirname, 'management.db');
const db = new Database(DB_PATH);
db.pragma('journal_mode = WAL');

// ── Table Initialization ──────────────────────────────────────────────────
db.exec(`
    CREATE TABLE IF NOT EXISTS inventory (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        item_name    TEXT NOT NULL,
        quantity     INTEGER NOT NULL DEFAULT 0,
        location     TEXT NOT NULL,
        last_updated TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS evacuees (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        name         TEXT NOT NULL,
        age          INTEGER,
        status       TEXT NOT NULL,
        shelter_id   TEXT
    );

    CREATE TABLE IF NOT EXISTS personnel (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        name          TEXT NOT NULL,
        role          TEXT NOT NULL,
        status        TEXT NOT NULL,
        assigned_zone TEXT
    );
`);
console.log(`📦 Management DB ready: ${DB_PATH}`);

// Seed baseline emergency resources if newly created
const countInv = db.prepare(`SELECT count(*) as count FROM inventory`).get().count;
if (countInv === 0) {
    const seedInv = db.prepare(`INSERT INTO inventory (item_name, quantity, location, last_updated) VALUES (?, ?, ?, ?)`);
    const now = new Date().toISOString();
    seedInv.run('Inflatable Motorized Rescue Boats (Zodiac 40HP)', 12, 'District EOC Depot', now);
    seedInv.run('Submersible High-Volume De-watering Pumps (5000 L/min)', 8, 'Flood Control Base', now);
    seedInv.run('FEMA/WHO Emergency First Aid Trauma Kits', 450, 'Red Cross Medical Cache', now);
    seedInv.run('Potable Water Purification Sachets (10,000L capacity)', 2500, 'Civil Supplies Store', now);
    seedInv.run('Woven Polypropylene Flood Sandbags (Filled)', 6000, 'Irrigation & Flood Control Base', now);
    seedInv.run('Ready-to-Eat Emergency Ration Packs (MRE)', 3200, 'State Disaster Response Depot', now);
}

const countPers = db.prepare(`SELECT count(*) as count FROM personnel`).get().count;
if (countPers === 0) {
    const seedPers = db.prepare(`INSERT INTO personnel (name, role, status, assigned_zone) VALUES (?, ?, ?, ?)`);
    seedPers.run('Commander Rajesh Sharma', 'Incident Commander', 'Active / Operational', 'District EOC Command');
    seedPers.run('Dr. Ananya Baruah', 'Chief Medical Officer', 'On Standby', 'District General Hospital');
    seedPers.run('Inspector Bikram Das', 'Water Rescue Lead', 'Deployed', 'Riverine Sector Alpha');
    seedPers.run('Officer Priya Sen', 'Logistics Coordinator', 'Active', 'Central Relief Depot');
}

// ── Inventory CRUD ───────────────────────────────────────────────────────
const insertInv = db.prepare(`INSERT INTO inventory (item_name, quantity, location, last_updated) VALUES (@item_name, @quantity, @location, @last_updated)`);
const updateInv = db.prepare(`UPDATE inventory SET item_name=@item_name, quantity=@quantity, location=@location, last_updated=@last_updated WHERE id=@id`);
const getInv = db.prepare(`SELECT * FROM inventory ORDER BY id DESC`);
const deleteInv = db.prepare(`DELETE FROM inventory WHERE id = ?`);

function addInventory(item) { return insertInv.run({ ...item, last_updated: new Date().toISOString() }).lastInsertRowid; }
function editInventory(item) { return updateInv.run({ ...item, last_updated: new Date().toISOString() }).changes > 0; }
function listInventory() { return getInv.all(); }
function removeInventory(id) { return deleteInv.run(id).changes > 0; }

// ── Evacuees CRUD ────────────────────────────────────────────────────────
const insertEvac = db.prepare(`INSERT INTO evacuees (name, age, status, shelter_id) VALUES (@name, @age, @status, @shelter_id)`);
const updateEvac = db.prepare(`UPDATE evacuees SET name=@name, age=@age, status=@status, shelter_id=@shelter_id WHERE id=@id`);
const getEvac = db.prepare(`SELECT * FROM evacuees ORDER BY id DESC`);
const deleteEvac = db.prepare(`DELETE FROM evacuees WHERE id = ?`);

function addEvacuee(person) { return insertEvac.run(person).lastInsertRowid; }
function editEvacuee(person) { return updateEvac.run(person).changes > 0; }
function listEvacuees() { return getEvac.all(); }
function removeEvacuee(id) { return deleteEvac.run(id).changes > 0; }

// ── Personnel CRUD ───────────────────────────────────────────────────────
const insertPers = db.prepare(`INSERT INTO personnel (name, role, status, assigned_zone) VALUES (@name, @role, @status, @assigned_zone)`);
const updatePers = db.prepare(`UPDATE personnel SET name=@name, role=@role, status=@status, assigned_zone=@assigned_zone WHERE id=@id`);
const getPers = db.prepare(`SELECT * FROM personnel ORDER BY id DESC`);
const deletePers = db.prepare(`DELETE FROM personnel WHERE id = ?`);

function addPersonnel(person) { return insertPers.run(person).lastInsertRowid; }
function editPersonnel(person) { return updatePers.run(person).changes > 0; }
function listPersonnel() { return getPers.all(); }
function removePersonnel(id) { return deletePers.run(id).changes > 0; }

module.exports = {
    addInventory, editInventory, listInventory, removeInventory,
    addEvacuee, editEvacuee, listEvacuees, removeEvacuee,
    addPersonnel, editPersonnel, listPersonnel, removePersonnel
};

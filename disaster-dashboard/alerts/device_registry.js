/**
 * disaster-dashboard/alerts/device_registry.js — Device registry for alert delivery (Phase 6)
 *
 * Manages registered devices that can receive flood alerts. In a production
 * system this would be backed by a database; here it uses an in-memory store
 * backed by MongoDB when available.
 *
 * Stores minimal data only: device token, coordinates (coarse cell), consent
 * flags. No precise location beyond cell centroid.
 */

const { MongoClient } = require("mongodb");

const MONGO_URI = process.env.MONGODB_URI || "mongodb://127.0.0.1:27017";
const DB_NAME = process.env.MONGODB_DB || "floodwatch";
const DEVICES_COLLECTION = "global_mesh_devices";

class DeviceRegistry {
  constructor() {
    this.client = null;
    this.db = null;
    this.devicesCol = null;
    this.isConnected = false;
    this._memoryDevices = new Map(); // deviceId -> device
  }

  async connect() {
    try {
      this.client = new MongoClient(MONGO_URI, {
        serverSelectionTimeoutMS: 5000,
        connectTimeoutMS: 5000,
      });
      await this.client.connect();
      this.db = this.client.db(DB_NAME);
      this.devicesCol = this.db.collection(DEVICES_COLLECTION);
      await this.devicesCol.createIndex({ deviceId: 1 }, { unique: true });
      await this.devicesCol.createIndex({ cellId: 1 });
      this.isConnected = true;
      console.log("✅ Device registry MongoDB connected:", MONGO_URI);
    } catch (exc) {
      console.warn("⚠️ Device registry MongoDB failed:", exc.message, "— using in-memory fallback");
      this.isConnected = false;
    }
  }

  async disconnect() {
    if (this.client) {
      try {
        await this.client.close();
      } catch (e) {
        /* ignore */
      }
    }
  }

  /**
   * Register or update a device.
   * device: { deviceId, cellId, lat, lon, consent: boolean, platform, token }
   */
  async registerDevice(device) {
    const rec = {
      deviceId: device.deviceId,
      cellId: device.cellId,
      lat: device.lat,
      lon: device.lon,
      consent: !!device.consent,
      platform: device.platform || "web",
      token: device.token || null,
      registeredAt: device.registeredAt || new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };

    if (this.isConnected && this.devicesCol) {
      try {
        await this.devicesCol.updateOne(
          { deviceId: rec.deviceId },
          { $set: rec },
          { upsert: true }
        );
      } catch (exc) {
        console.warn("Device registry save failed:", exc.message);
        this._memoryDevices.set(rec.deviceId, rec);
      }
    } else {
      this._memoryDevices.set(rec.deviceId, rec);
    }
    return rec;
  }

  /**
   * Get all devices for a cell (used to notify when an alert fires).
   * Only returns devices with consent = true.
   */
  async getDevicesForCell(cellId) {
    if (this.isConnected && this.devicesCol) {
      try {
        return await this.devicesCol.find({
          cellId,
          consent: true,
        }).toArray();
      } catch (exc) {
        console.warn("Device registry read failed:", exc.message);
      }
    }
    return Array.from(this._memoryDevices.values()).filter(
      (d) => d.cellId === cellId && d.consent
    );
  }

  /**
   * Get all registered devices (for admin UI).
   */
  async getAllDevices() {
    if (this.isConnected && this.devicesCol) {
      try {
        return await this.devicesCol.find({}).toArray();
      } catch (exc) {
        console.warn("Device registry read failed:", exc.message);
      }
    }
    return Array.from(this._memoryDevices.values());
  }

  /**
   * Deactivate a device (revoke consent).
   */
  async deactivateDevice(deviceId) {
    if (this.isConnected && this.devicesCol) {
      try {
        await this.devicesCol.updateOne(
          { deviceId },
          { $set: { consent: false, updatedAt: new Date().toISOString() } }
        );
      } catch (exc) {
        console.warn("Device registry deactivate failed:", exc.message);
      }
    }
    const d = this._memoryDevices.get(deviceId);
    if (d) {
      d.consent = false;
      d.updatedAt = new Date().toISOString();
    }
  }

  /**
   * Count consented devices for a cell.
   */
  async countConsentedDevices(cellId) {
    const devices = await this.getDevicesForCell(cellId);
    return devices.length;
  }
}

module.exports = { DeviceRegistry };

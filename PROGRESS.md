# 📈 Development Progress — FloodWatch AI

A staged log of what was built, in order. Most recent stage last.

---

## Stage 1 — Baseline system (earlier sessions)

The starting point before this round of work:

- **AI model v1–v3**: U-Net flood segmentation trained on Sen1Floods11, evolving
  from ResNet34 → EfficientNet-B3 backbone, BCE → Dice+Focal loss.
- **Python FastAPI engine** (`main.py`): `/predict` (upload) and `/predict-live`
  (fetch Sentinel-1 from Microsoft Planetary Computer + infer) endpoints, with
  post-processing (threshold, size filter, ocean + OpenStreetMap water filters).
- **Node/Express dashboard** (`server.js`): serves the frontend, proxies to Python,
  logs every scan to SQLite (`scans.db`) via `history.js`.
- **Leaflet frontend** (`index.html`): world map, click-to-scan, upload, and a
  history tab.

---

## Stage 2 — v4 model integration (get it running)

**Problem found:** the API could not load the new model at all — two separate bugs:

1. **Filename mismatch** — `main.py` loaded `flood_unet_resnet34.pth`, but the new
   model was renamed to `unet_b3.pth`.
2. **Channel mismatch** — the v4 model (`colab_model_v4.ipynb`) was trained on
   **4 input channels** (`VV`, `VH`, `VV/VH ratio`, JRC permanent-water mask), but
   the API built the model with `in_channels=2` and only fed 2 channels. Confirmed
   by inspecting the weights directly (`_conv_stem.weight` shape = `(40, 4, 3, 3)`).

**Fixes applied to `main.py`:**
- Load `unet_b3.pth` with `in_channels=4`, EfficientNet-B3.
- Added `build_model_input()` — reconstructs the missing channels the same way the
  notebook did: `ratio = VV/VH` (min-max normalized) + a zeros placeholder for the
  permanent-water channel (not available at inference for arbitrary locations; the
  OSM water filter covers it downstream).
- Both `/predict` and `/predict-live` now use this builder.
- Fixed the osmnx 2.0 call signature and replaced deprecated `.unary_union` with `.union_all()`.

**Verified:** model loads, runs inference on `real_flood_test_v4_tif_file.tif`,
detects ~5,940 flood pixels at threshold 0.65.

---

## Stage 3 — Dashboard redesign (look & feel)

Full rewrite of `public/index.html` into a cohesive design system.

**Added / improved:**
- App header with **FloodWatch AI** branding and a **live "AI Engine online/offline"
  status dot** (polls `/api/health` every 15 s).
- Clean sidebar layout, gradient buttons, monospace coordinate inputs.
- Always-visible **severity legend** (Low → Critical color scale).
- **Basemap toggle** — Satellite ↔ Streets (dark labeled map).
- **Scan-box preview** — dashed rectangle showing the ~10 km area before scanning.
- Redesigned **summary card** (big area number, severity pill, danger chips).
- **Toast notifications** replacing raw `alert()` popups.
- **Loading spinner** with rotating step text ("Downloading imagery… Running model…").
- History rebuilt as **cards** with click-to-refly and hover-to-delete.
- Responsive layout for narrow screens.

**Backend support:** added `GET /api/health` to `server.js` (pings the Python engine).

**Verified in browser:** page renders, zero console errors, tab switching, history
loads from SQLite, basemap toggle all working.

---

## Stage 4 — Live Data feature (real-world floods)

A new mode showing floods **actually happening right now**, independent of the AI model.

**Data source decision:** GDACS (UN/EU Global Disaster Alert System) — free, no API
key. (Gemini summaries were considered but the provided key's Google account had
zero generate-content quota, so the feature was built on GDACS's own text, which
is sufficient.)

**Backend — `server.js`:**
- `GET /api/live-floods` — proxies the GDACS active-floods feed (avoids browser CORS).
- `GET /api/live-floods/shape?eventid=&episodeid=` — proxies one event's real
  affected-area polygon.

**Frontend — new 🌐 Live Data tab:**
- Colored markers per flood, keyed to GDACS alert level (🔴 Red / 🟠 Orange / 🟢 Green).
- Sidebar list with text: event name, country, coordinates, date range, source.
- **Click an event → draws its real affected-area polygon** on the map in the alert
  color and flies to it.

**Verified in browser:** loaded 24 real active floods worldwide; clicking the
"Red Flood in China" event drew its true affected-area polygon (red, dashed) over
the satellite terrain. Same behavior will surface India/Assam whenever it's in the feed.

---

## Stage 5 — Documentation

- Rewrote `README.md` for the v4 model, four-mode dashboard, GDACS Live Data, full
  endpoint list, project structure, and run instructions.
- Created this `PROGRESS.md`.

---

## Known limitations / next ideas

- **4th model channel is a zeros placeholder.** Faithful option would be to fetch
  the JRC Global Surface Water tile per scan; current approach is reliable and fast,
  and the OSM filter compensates on the visible map.
- **No Python venv committed** — must be created locally to run the AI engine.
- **Possible future work:** a second disaster type (wildfire via Sentinel-2 NBR —
  needs no model), affected-buildings/roads count from OpenStreetMap, and optional
  LLM-generated flood summaries if a working API key becomes available.

#!/usr/bin/env python3
"""
Generate Clean Academic Diagram for [SCREENSHOT PLACEHOLDER: 3.3]
Data flow from 24 input factors through M1 SAR, M6 GRU Scanner,
M2-M5 Ensemble, and M7 Hydraulic Engine into Safety Containment.
Publication quality, pure white background, perfectly legible, zero overlaps.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_diagram():
    # Setup canvas: spacious, high DPI
    fig, ax = plt.subplots(figsize=(18.8, 10.2), dpi=300)
    ax.set_xlim(0, 18.8)
    ax.set_ylim(0, 10.2)
    ax.axis('off')
    fig.patch.set_facecolor('#ffffff')

    # Top Header Banner
    header_box = patches.FancyBboxPatch(
        (0.6, 9.35), 17.6, 0.70,
        boxstyle="round,pad=0.04",
        facecolor="#f8fafc",
        edgecolor="#cbd5e1",
        lw=1.0
    )
    ax.add_patch(header_box)
    ax.text(9.4, 9.77, "FloodWatch: End-to-End Predictive Pipeline & Safety Containment Architecture",
            fontsize=14.0, fontweight='bold', color="#0f172a", ha='center', va='center', fontfamily='sans-serif')
    ax.text(9.4, 9.50, "Figure 3.3: Systematic Data Flow from 24 Ingestion Factors through ML/Physics Engines (M1-M7) into Gated Operational Safety Containment",
            fontsize=9.2, color="#475569", ha='center', va='center', fontfamily='sans-serif')

    # 4 Main Architectural Column Headers
    col_defs = [
        {"x": 0.6, "w": 3.8, "title": "STAGE 1: 24 INPUT FACTORS", "sub": "Raw Ingestion & Feature Engineering", "color": "#1e3a8a"},
        {"x": 5.0, "w": 4.0, "title": "STAGE 2: ANALYTICAL & DL ENGINES", "sub": "Multi-Model Inference (M1 - M7)", "color": "#1e40af"},
        {"x": 9.6, "w": 4.1, "title": "STAGE 3: TRUST & DECISION LOGIC", "sub": "Cost-Loss Gate & Anti-Flapping Policy", "color": "#991b1b"},
        {"x": 14.3, "w": 3.9, "title": "STAGE 4: SAFETY CONTAINMENT", "sub": "Dry-Run Barrier & CAP 1.2 Export", "color": "#166534"}
    ]

    for col in col_defs:
        ch_box = patches.FancyBboxPatch(
            (col["x"], 8.65), col["w"], 0.55,
            boxstyle="round,pad=0.03",
            facecolor="#ffffff",
            edgecolor=col["color"],
            lw=1.5
        )
        ax.add_patch(ch_box)
        ax.text(col["x"] + col["w"]/2, 8.98, col["title"],
                fontsize=8.8, fontweight='bold', color=col["color"], ha='center', va='center')
        ax.text(col["x"] + col["w"]/2, 8.78, col["sub"],
                fontsize=7.0, color="#64748b", ha='center', va='center')

    # =========================================================================
    # COLUMN 1: 24 INPUT FACTORS (Aligned 1-to-1 with M1, M6, M2-M5, M7)
    # =========================================================================
    c1_cards = [
        {
            # Aligned with M1: SAR Inundation Engine
            "y": 6.60, "h": 1.90, "title": "SAR & Priors (4 Factors)", "tag": "Sentinel-1 & JRC",
            "border": "#7c3aed",
            "items": [
                "sar_vv_backscatter (Sentinel-1 VV Polarized dB)",
                "sar_vh_backscatter (Sentinel-1 VH Cross-Pol dB)",
                "sar_cross_ratio (VH / VV Backscatter Ratio)",
                "jrc_permanent_water (Permanent Water Mask)"
            ]
        },
        {
            # Aligned with M6: Global Mesh GRU Scanner
            "y": 4.55, "h": 1.90, "title": "Atmospheric & Met (8 Factors)", "tag": "ECMWF IFS (0.1 deg)",
            "border": "#2563eb",
            "items": [
                "precip_1h, precip_6h, precip_24h (Rainfall Acc.)",
                "temp_2m (2m Surface Air Temperature)",
                "dewpoint_2m (Humidity / Saturation)",
                "pressure_msl (Mean Sea Level Pressure)",
                "wind_u10, wind_v10 (10m Vector Wind Field)"
            ]
        },
        {
            # Aligned with M2-M5: Predictive Ensemble
            "y": 2.50, "h": 1.90, "title": "Hydrological & Soil (6 Factors)", "tag": "GloFAS & Gauges",
            "border": "#059669",
            "items": [
                "soil_moisture_layer1 (0-7 cm Topsoil Moisture)",
                "soil_moisture_layer2 (7-28 cm Root-Zone)",
                "runoff_surface (Direct Overland Runoff)",
                "runoff_subsurface (Interflow & Infiltration)",
                "discharge_volume (River Discharge m3/s)",
                "gauge_stage_height (In-Situ Stage Height m)"
            ]
        },
        {
            # Aligned with M7: Hydraulic Engine
            "y": 0.45, "h": 1.90, "title": "Topography & Drainage (6 Factors)", "tag": "Copernicus & OSM",
            "border": "#d97706",
            "items": [
                "elevation_dem (GLO-30 30m Copernicus DEM)",
                "slope_radians (Topographic Gradient)",
                "topographic_wetness_index (TWI = ln(a / tan b))",
                "flow_accumulation (Upstream Catchment Area)",
                "distance_to_river_osm (Proximity to Channel)",
                "river_channel_density (Drainage Density / km2)"
            ]
        }
    ]

    for card in c1_cards:
        box = patches.FancyBboxPatch(
            (0.6, card["y"]), 3.8, card["h"],
            boxstyle="round,pad=0.04",
            facecolor="#ffffff",
            edgecolor=card["border"],
            lw=1.2
        )
        ax.add_patch(box)
        # Header text
        ax.text(0.75, card["y"] + card["h"] - 0.24, card["title"],
                fontsize=8.1, fontweight='bold', color=card["border"], va='center')
        ax.text(4.25, card["y"] + card["h"] - 0.24, card["tag"],
                fontsize=6.8, color="#64748b", ha='right', va='center')
        # Separator line
        ax.plot([0.75, 4.25], [card["y"] + card["h"] - 0.38, card["y"] + card["h"] - 0.38],
                color="#e2e8f0", lw=0.8)
        # Items
        start_y = card["y"] + card["h"] - 0.58
        for idx, item in enumerate(card["items"]):
            ax.text(0.78, start_y - idx * 0.25, "*  " + item,
                    fontsize=6.5, color="#334155", va='center')

    # =========================================================================
    # COLUMN 2: ANALYTICAL & DEEP LEARNING ENGINES (M1, M6, M2-M5, M7)
    # =========================================================================
    c2_cards = [
        {
            "y": 6.60, "h": 1.90, "title": "M1: SAR Inundation Engine", "badge": "2D U-Net (ResNet-34)",
            "border": "#7c3aed",
            "lines": [
                ("Input:", "Sentinel-1 VV/VH backscatter + JRC prior mask"),
                ("Architecture:", "2D U-Net pixel-level flood segmentation"),
                ("Performance:", "Dice = 0.7812, IoU = 0.6410 on held-out floods"),
                ("Output:", "Observed water mask & vector flood polygons")
            ]
        },
        {
            "y": 4.55, "h": 1.90, "title": "M6: Global Mesh GRU Scanner", "badge": "Spatio-Temporal Sequence",
            "border": "#2563eb",
            "lines": [
                ("Input:", "24-factor sequence vectors over 0.1 deg cells"),
                ("Architecture:", "2-layer GRU (hidden dim 64) + recurrent dropout"),
                ("Performance:", "Test AUC = 0.8952 (Held-out Test Partition)"),
                ("Output:", "Candidate alert screening & preliminary P(flood)")
            ]
        },
        {
            "y": 2.50, "h": 1.90, "title": "M2 - M5: Predictive Ensemble", "badge": "Multi-Horizon ML & GNN",
            "border": "#059669",
            "lines": [
                ("M2 (XGB/LGBM):", "Gradient Boosted Trees (Test AUC = 0.8841)"),
                ("M3 (ST-GNN):", "River topology graph network (Macro F1 = 0.742)"),
                ("M4 (TCN-Attn):", "Multi-horizon trajectory forecast (0 - 72h)"),
                ("M5 (MC Ensemble):", "Epistemic uncertainty bound: sigma_p <= 0.25")
            ]
        },
        {
            "y": 0.45, "h": 1.90, "title": "M7: Deterministic Hydraulic Engine", "badge": "Physics Simulation",
            "border": "#d97706",
            "lines": [
                ("Input:", "GloFAS discharge Q, channel slope, GLO-30 DEM"),
                ("Physics Engine:", "1D/2D Saint-Venant shallow water equations"),
                ("Rating Curves:", "Manning equation -> peak stage & flow velocity"),
                ("Output:", "Hydrodynamic flood depth & composite risk index")
            ]
        }
    ]

    for card in c2_cards:
        box = patches.FancyBboxPatch(
            (5.0, card["y"]), 4.0, card["h"],
            boxstyle="round,pad=0.04",
            facecolor="#ffffff",
            edgecolor=card["border"],
            lw=1.3
        )
        ax.add_patch(box)
        # Header
        ax.text(5.15, card["y"] + card["h"] - 0.24, card["title"],
                fontsize=8.3, fontweight='bold', color=card["border"], va='center')
        ax.text(8.85, card["y"] + card["h"] - 0.24, card["badge"],
                fontsize=6.8, color="#64748b", ha='right', va='center')
        # Separator
        ax.plot([5.15, 8.85], [card["y"] + card["h"] - 0.38, card["y"] + card["h"] - 0.38],
                color="#e2e8f0", lw=0.8)
        # Content
        start_y = card["y"] + card["h"] - 0.58
        for idx, (label, val) in enumerate(card["lines"]):
            ax.text(5.18, start_y - idx * 0.27, label,
                    fontsize=6.6, fontweight='bold', color="#0f172a", va='center')
            ax.text(6.25, start_y - idx * 0.27, val,
                    fontsize=6.4, color="#334155", va='center')

    # =========================================================================
    # COLUMN 3: DECISION LOGIC & TRUST GATE
    # =========================================================================
    # Top Card: Multi-Factor Trust Gate Gamma
    tg_box = patches.FancyBboxPatch(
        (9.6, 4.55), 4.1, 3.95,
        boxstyle="round,pad=0.04",
        facecolor="#ffffff",
        edgecolor="#b91c1c",
        lw=1.4
    )
    ax.add_patch(tg_box)
    ax.text(9.75, 8.25, "Multi-Factor Trust Gate (Gamma)", fontsize=8.8, fontweight='bold', color="#991b1b")
    ax.text(9.75, 7.98, "Operational requirement: All 4 conditions must pass simultaneously:", fontsize=6.8, color="#475569")
    ax.plot([9.75, 13.55], [7.82, 7.82], color="#fee2e2", lw=0.8)

    gamma_rules = [
        ("1. Data Availability Integrity", "Mandatory dynamic telemetry present; missing values", "never imputed to artificially inflate hazard risk."),
        ("2. Empirical Model Validity", "Active model trained & verified: Test AUC > AUC_clim + 0.05", "(Observed Test AUC = 0.8952 vs 0.50 climatology)."),
        ("3. Telemetry Freshness Window", "All dynamic telemetry validated: FRESH (<=24h) or", "ACCEPTABLE (<=48h). Stale observations rejected."),
        ("4. Epistemic Uncertainty Bound", "Ensemble standard deviation strictly bounded:", "sigma_p <= 0.25 (evaluated via M5 MC ensemble variance).")
    ]

    for idx, (head, l1, l2) in enumerate(gamma_rules):
        yy = 7.50 - idx * 0.70
        ax.text(9.78, yy, "[OK] " + head, fontsize=6.8, fontweight='bold', color="#b91c1c")
        ax.text(10.00, yy - 0.22, l1, fontsize=6.3, color="#334155")
        ax.text(10.00, yy - 0.40, l2, fontsize=6.3, color="#334155")

    # Bottom Card: Cost-Loss & Anti-Flapping Policy
    cl_box = patches.FancyBboxPatch(
        (9.6, 0.45), 4.1, 3.95,
        boxstyle="round,pad=0.04",
        facecolor="#ffffff",
        edgecolor="#991b1b",
        lw=1.3
    )
    ax.add_patch(cl_box)
    ax.text(9.75, 4.15, "Cost-Loss & Anti-Flapping Policy", fontsize=8.8, fontweight='bold', color="#991b1b")
    ax.text(9.75, 3.88, "Economic decision thresholding & operational fatigue guards:", fontsize=6.8, color="#475569")
    ax.plot([9.75, 13.55], [3.72, 3.72], color="#fee2e2", lw=0.8)

    policy_rules = [
        ("Economic Threshold:", "Alert raised iff P(flood) > C/L = 1/10 = 0.10"),
        ("Cost-Loss Criterion:", "Expected loss exceeds action cost: P * Loss > Cost"),
        ("Escalation Guard:", "Requires n_up >= 2 consecutive evaluations > 0.10"),
        ("De-escalation Guard:", "Requires n_down >= 3 consecutive evaluations < 0.10"),
        ("Cell Deduplication:", "Strictly 1 active candidate alert per spatial grid cell"),
        ("Anti-Fatigue Budget:", "Maximum 3 alerts per grid cell within any 24h cycle"),
        ("Untrusted Action:", "If Gamma = 0, candidate is withheld for monitoring only")
    ]

    for idx, (label, val) in enumerate(policy_rules):
        yy = 3.42 - idx * 0.40
        ax.text(9.78, yy, label, fontsize=6.6, fontweight='bold', color="#7f1d1d")
        ax.text(11.05, yy, val, fontsize=6.3, color="#334155")

    # =========================================================================
    # COLUMN 4: SAFETY CONTAINMENT & AUDIT
    # =========================================================================
    # Top Card: Dry-Run Containment Barrier
    sc1_box = patches.FancyBboxPatch(
        (14.3, 4.55), 3.9, 3.95,
        boxstyle="round,pad=0.04",
        facecolor="#ffffff",
        edgecolor="#16a34a",
        lw=1.4
    )
    ax.add_patch(sc1_box)
    ax.text(14.45, 8.25, "Dry-Run Containment Barrier", fontsize=8.8, fontweight='bold', color="#15803d")
    ax.text(14.45, 7.98, "Human-in-the-Loop review before any public dissemination:", fontsize=6.8, color="#475569")
    ax.plot([14.45, 18.05], [7.82, 7.82], color="#dcfce7", lw=0.8)

    containment_items = [
        ("ALERT_MODE = dry_run", "Hardcoded safe default; public broadcasts blocked.", "Ensures zero unverified alerts leave the system."),
        ("FCM Push Delivery Gated", "Mocked/gated transport; unsolicited push disabled.", "Prevents accidental notification to public devices."),
        ("Operator Review Interface", "Interactive portal (alert_review.html) for humans.", "Displays multi-model metrics & spatial polygon."),
        ("Mandatory Human Action", "Explicit click required: APPROVE | REJECT | CANCEL.", "Autonomous alert dispatch is strictly prohibited."),
        ("Untrusted State Lockdown", "If Gamma = 0: candidate shown for situational awareness;", "APPROVE button disabled until criteria are resolved.")
    ]

    for idx, (title, l1, l2) in enumerate(containment_items):
        yy = 7.50 - idx * 0.58
        ax.text(14.48, yy, "*  " + title, fontsize=6.8, fontweight='bold', color="#166534")
        ax.text(14.68, yy - 0.20, l1, fontsize=6.3, color="#334155")
        ax.text(14.68, yy - 0.36, l2, fontsize=6.1, color="#64748b")

    # Bottom Card: CAP 1.2 Protocol & Verification
    sc2_box = patches.FancyBboxPatch(
        (14.3, 0.45), 3.9, 3.95,
        boxstyle="round,pad=0.04",
        facecolor="#ffffff",
        edgecolor="#15803d",
        lw=1.3
    )
    ax.add_patch(sc2_box)
    ax.text(14.45, 4.15, "CAP 1.2 Protocol & Verification", fontsize=8.8, fontweight='bold', color="#15803d")
    ax.text(14.45, 3.88, "OASIS international standard export & audit trail:", fontsize=6.8, color="#475569")
    ax.plot([14.45, 18.05], [3.72, 3.72], color="#dcfce7", lw=0.8)

    cap_items = [
        ("CAP 1.2 Status:", "<status>Exercise</status> (safety containment)"),
        ("Routing Scope:", "<scope>Restricted</scope> (internal testing)"),
        ("Spatial Area:", "<polygon> 5-point GeoJSON bounding box"),
        ("Device Registry:", "Coarse cell ID only; zero GPS tracking"),
        ("Immutable Log:", "MongoDB: ISO-8601 UTC + SHA-256 hash"),
        ("Operator Signature:", "Records operator ID, review time & rationale"),
        ("Verification Score:", "HIT | MISS | FALSE_ALARM | CORRECT_NEG"),
        ("Missing Data Guard:", "DATA_UNAVAILABLE recorded, not imputed")
    ]

    for idx, (label, val) in enumerate(cap_items):
        yy = 3.42 - idx * 0.40
        ax.text(14.48, yy, label, fontsize=6.5, fontweight='bold', color="#14532d")
        ax.text(15.55, yy, val, fontsize=6.3, color="#334155")

    # =========================================================================
    # CONNECTING ARROWS & DATA FLOW
    # =========================================================================
    # From Column 1 to Column 2 (Perfect 1-to-1 horizontal alignment)
    # SAR factors (Col 1 Card 1) -> M1: SAR Inundation (Col 2 Card 1)
    ax.annotate('', xy=(4.95, 7.55), xytext=(4.45, 7.55),
                arrowprops=dict(arrowstyle="-|>", color="#7c3aed", lw=1.6))
    # Met factors (Col 1 Card 2) -> M6: GRU Scanner (Col 2 Card 2)
    ax.annotate('', xy=(4.95, 5.50), xytext=(4.45, 5.50),
                arrowprops=dict(arrowstyle="-|>", color="#2563eb", lw=1.6))
    # Hydro factors (Col 1 Card 3) -> M2-M5: Ensemble (Col 2 Card 3)
    ax.annotate('', xy=(4.95, 3.45), xytext=(4.45, 3.45),
                arrowprops=dict(arrowstyle="-|>", color="#059669", lw=1.6))
    # Topo factors (Col 1 Card 4) -> M7: Hydraulic Engine (Col 2 Card 4)
    ax.annotate('', xy=(4.95, 1.40), xytext=(4.45, 1.40),
                arrowprops=dict(arrowstyle="-|>", color="#d97706", lw=1.6))

    # Cross-feeding telemetry:
    # 24-feature vector bus to M6 (GRU takes all factors)
    ax.annotate('', xy=(4.95, 5.25), xytext=(4.45, 3.45),
                arrowprops=dict(arrowstyle="-|>", color="#2563eb", lw=1.0, linestyle="--", connectionstyle="arc3,rad=-0.12"))
    ax.annotate('', xy=(4.95, 5.75), xytext=(4.45, 7.55),
                arrowprops=dict(arrowstyle="-|>", color="#2563eb", lw=1.0, linestyle="--", connectionstyle="arc3,rad=0.12"))

    # From Column 2 to Column 3 (Decision & Trust Gate)
    # M1 SAR observed mask -> Trust Gate validation
    ax.annotate('', xy=(9.55, 7.20), xytext=(9.05, 7.55),
                arrowprops=dict(arrowstyle="-|>", color="#7c3aed", lw=1.4, connectionstyle="arc3,rad=-0.08"))
    # M6 candidate alert -> Trust Gate Gamma
    ax.annotate('', xy=(9.55, 6.00), xytext=(9.05, 5.50),
                arrowprops=dict(arrowstyle="-|>", color="#2563eb", lw=1.6, connectionstyle="arc3,rad=0.08"))
    # M2-M5 ensemble P(flood) -> Cost-Loss Threshold
    ax.annotate('', xy=(9.55, 3.00), xytext=(9.05, 3.45),
                arrowprops=dict(arrowstyle="-|>", color="#059669", lw=1.6, connectionstyle="arc3,rad=-0.08"))
    # M7 deterministic rating -> Cost-Loss corroboration
    ax.annotate('', xy=(9.55, 1.80), xytext=(9.05, 1.40),
                arrowprops=dict(arrowstyle="-|>", color="#d97706", lw=1.4, connectionstyle="arc3,rad=0.08"))

    # Internal link in Col 3: Trust Gate condition feeds into candidate qualification
    ax.annotate('', xy=(11.65, 4.45), xytext=(11.65, 4.55),
                arrowprops=dict(arrowstyle="-|>", color="#b91c1c", lw=1.5))

    # From Column 3 to Column 4 (Safety Containment)
    # Qualified candidate (Gamma = 1) -> Operator Review Portal
    ax.annotate('', xy=(14.25, 6.52), xytext=(13.75, 6.52),
                arrowprops=dict(arrowstyle="-|>", color="#dc2626", lw=1.8))
    # Decision action approved -> CAP 1.2 Export
    ax.annotate('', xy=(14.25, 2.42), xytext=(13.75, 2.42),
                arrowprops=dict(arrowstyle="-|>", color="#166534", lw=1.8))

    # Internal link in Col 4: Human Operator click triggers CAP export
    ax.annotate('', xy=(16.25, 4.45), xytext=(16.25, 4.55),
                arrowprops=dict(arrowstyle="-|>", color="#15803d", lw=1.5))

    # Save to all target paths
    paths = [
        r"F:\mp-latest\model_pipeline_safety_containment.jpg",
        r"F:\mp-latest\mp_proj_1\figures\model_pipeline_safety_containment.jpg",
        r"F:\mp-latest\mp_proj_1\figures\model_pipeline_safety_containment.png",
        r"F:\mp-latest\project-proofs\pipeline_data_flow_containment.jpg"
    ]

    plt.tight_layout()
    for p in paths:
        if p.endswith('.jpg'):
            plt.savefig(p, dpi=300, bbox_inches='tight', pil_kwargs={'quality': 96})
        else:
            plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f"Saved: {p}")

    plt.close()

if __name__ == '__main__':
    generate_diagram()

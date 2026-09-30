"""
topological_river_network.py — HydroBASINS Topological River Routing Graph Engine

Implements Directed Acyclic Graph (DAG) routing along continental river networks:
- Upstream headwaters -> Middle corridor reaches -> Delta outlets.
- Connects catchment nodes via directed edges (downstream pointers: NEXT_DOWN).
- Computes hydraulic travel delays (tau = L / v_celerity) based on river length and flow velocity (~1.5 m/s).
- Routes upstream precipitation and runoff pulses downstream, allowing the system to predict
  downstream flood peaks hours or days before local storms arrive.
- Classifies flood typology: Fluvial (Riverine), Pluvial (Flash), Nival (Snowmelt), or Compound Coastal.
"""

import math
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger("TopologicalRiverNetwork")

# Typical mean flood wave celerity in natural alluvial rivers: ~1.5 m/s (~130 km/day)
DEFAULT_CELERITY_M_PER_S = 1.6
KM_PER_DAY_CELERITY = (DEFAULT_CELERITY_M_PER_S * 86400) / 1000.0  # ~138.2 km/day

# Directed River Network Topology Map
# Each node specifies:
#   id: node identifier
#   downstream_id: NEXT_DOWN pointer to immediate receiving reach (None for ocean sink / terminal lake)
#   dist_to_downstream_km: river reach distance
#   is_coastal_outlet: True if estuarine/delta outlet subject to tidal/storm surge backwater
RIVER_TOPOLOGY_GRAPH = {
    # ── Brahmaputra / Ganges / Bengal Delta System ──
    "in_brahmaputra_upper": {"name": "Upper Brahmaputra (Arunachal/Upper Assam)", "downstream_id": "in_brahmaputra", "dist_km": 180, "is_coastal": False},
    "in_brahmaputra":       {"name": "Brahmaputra Floodplain (Kaziranga Reach)", "downstream_id": "bd_meghna", "dist_km": 340, "is_coastal": False},
    "in_ganges_upper":      {"name": "Upper Ganges (Kanpur/Varanasi Reach)", "downstream_id": "in_ganges", "dist_km": 280, "is_coastal": False},
    "in_ganges":            {"name": "Lower Ganges Basin (Patna Reach)", "downstream_id": "bd_meghna", "dist_km": 390, "is_coastal": False},
    "bd_meghna":            {"name": "Meghna / Padma Delta (Bay of Bengal Outlet)", "downstream_id": None, "dist_km": 0, "is_coastal": True},

    # ── Indus River Basin ──
    "pk_indus_upper":       {"name": "Upper Indus / Tarbela Gorge", "downstream_id": "pk_indus_punjab", "dist_km": 260, "is_coastal": False},
    "pk_indus_punjab":      {"name": "Punjab Five-Rivers Convergence", "downstream_id": "pk_indus_sindh", "dist_km": 420, "is_coastal": False},
    "pk_indus_sindh":       {"name": "Lower Indus Floodplain (Sukkur Reach)", "downstream_id": "pk_indus_delta", "dist_km": 310, "is_coastal": False},
    "pk_indus_delta":       {"name": "Indus Delta (Arabian Sea)", "downstream_id": None, "dist_km": 0, "is_coastal": True},

    # ── Yangtze & Pearl Corridors (China) ──
    "cn_yangtze_upper":     {"name": "Upper Yangtze / Three Gorges", "downstream_id": "cn_yangtze_mid", "dist_km": 320, "is_coastal": False},
    "cn_yangtze_mid":       {"name": "Middle Yangtze / Dongting Lake", "downstream_id": "cn_yangtze_delta", "dist_km": 580, "is_coastal": False},
    "cn_yangtze_delta":     {"name": "Yangtze Delta / Shanghai Outlet", "downstream_id": None, "dist_km": 0, "is_coastal": True},
    "cn_pearl_upper":       {"name": "West River (Xijiang) Reach", "downstream_id": "cn_pearl", "dist_km": 210, "is_coastal": False},
    "cn_pearl":             {"name": "Pearl River Delta (Guangdong)", "downstream_id": None, "dist_km": 0, "is_coastal": True},

    # ── Southeast Asia: Chao Phraya & Mekong ──
    "th_chaophraya_upper":  {"name": "Ping/Nan River Confluence (Nakhon Sawan)", "downstream_id": "th_chaophraya", "dist_km": 240, "is_coastal": False},
    "th_chaophraya":        {"name": "Chao Phraya Central Basin & Bangkok", "downstream_id": None, "dist_km": 0, "is_coastal": True},
    "vn_mekong_upper":      {"name": "Mekong Tonle Sap Inflow", "downstream_id": "vn_mekong_delta", "dist_km": 290, "is_coastal": False},
    "vn_mekong_delta":      {"name": "Lower Mekong Delta (Can Tho Reach)", "downstream_id": None, "dist_km": 0, "is_coastal": True},

    # ── Mississippi River System (USA) ──
    "us_mississippi_upper": {"name": "Upper Mississippi / Missouri Confluence", "downstream_id": "us_mississippi_mid", "dist_km": 310, "is_coastal": False},
    "us_ohio_river":        {"name": "Ohio River Confluence (Cairo)", "downstream_id": "us_mississippi_mid", "dist_km": 260, "is_coastal": False},
    "us_mississippi_mid":   {"name": "Middle Mississippi / Memphis Reach", "downstream_id": "us_mississippi_delta", "dist_km": 520, "is_coastal": False},
    "us_mississippi_delta": {"name": "Lower Mississippi Basin / New Orleans Delta", "downstream_id": None, "dist_km": 0, "is_coastal": True},

    # ── Rhine River Basin (Europe) ──
    "ch_rhine_upper":       {"name": "Alpine Rhine / Lake Constance", "downstream_id": "de_rhine_upper", "dist_km": 160, "is_coastal": False},
    "de_rhine_upper":       {"name": "Upper Rhine / Karlsruhe Reach", "downstream_id": "de_rhine_lower", "dist_km": 230, "is_coastal": False},
    "de_rhine_lower":       {"name": "Lower Rhine Corridor (Cologne Reach)", "downstream_id": "nl_rhine_delta", "dist_km": 210, "is_coastal": False},
    "nl_rhine_delta":       {"name": "Rhine-Meuse Delta (Rotterdam Outlet)", "downstream_id": None, "dist_km": 0, "is_coastal": True},

    # ── Danube River Basin (Europe) ──
    "de_danube_upper":      {"name": "Bavarian Danube / Passau", "downstream_id": "at_danube_upper", "dist_km": 190, "is_coastal": False},
    "at_danube_upper":      {"name": "Upper Danube Basin (Vienna Reach)", "downstream_id": "hu_danube_mid", "dist_km": 240, "is_coastal": False},
    "hu_danube_mid":        {"name": "Middle Danube / Budapest Reach", "downstream_id": "ro_danube_lower", "dist_km": 480, "is_coastal": False},
    "ro_danube_lower":      {"name": "Lower Danube / Black Sea Delta", "downstream_id": None, "dist_km": 0, "is_coastal": True},

    # ── South America: Amazon & Rio Grande do Sul ──
    "br_amazon_upper":      {"name": "Solimões / Upper Amazon Basin", "downstream_id": "br_amazon_mid", "dist_km": 800, "is_coastal": False},
    "br_amazon_mid":        {"name": "Middle Amazon / Manaus Reach", "downstream_id": "br_amazon_delta", "dist_km": 950, "is_coastal": False},
    "br_amazon_delta":      {"name": "Amazon Delta & Estuary (Macapá)", "downstream_id": None, "dist_km": 0, "is_coastal": True},
    "br_rs_jacui":          {"name": "Jacuí River Upper Basin", "downstream_id": "br_rs_portoalegre", "dist_km": 170, "is_coastal": False},
    "br_rs_portoalegre":    {"name": "Guaíba River / Porto Alegre Lagoon", "downstream_id": None, "dist_km": 0, "is_coastal": True},

    # ── Africa: Nile & Niger ──
    "et_blue_nile":         {"name": "Blue Nile Headwaters (Ethiopia)", "downstream_id": "sd_nile_khartoum", "dist_km": 450, "is_coastal": False},
    "sd_white_nile":        {"name": "White Nile Sudd Catchment", "downstream_id": "sd_nile_khartoum", "dist_km": 550, "is_coastal": False},
    "sd_nile_khartoum":     {"name": "Nile Confluence (Khartoum Reach)", "downstream_id": "eg_nile_delta", "dist_km": 980, "is_coastal": False},
    "eg_nile_delta":        {"name": "Nile Delta (Alexandria/Damietta)", "downstream_id": None, "dist_km": 0, "is_coastal": True},
    "ng_niger_upper":       {"name": "Upper Niger Inland Delta (Mali)", "downstream_id": "ng_niger_delta", "dist_km": 720, "is_coastal": False},
    "ng_niger_delta":       {"name": "Niger Delta Oil & Mangrove Reach", "downstream_id": None, "dist_km": 0, "is_coastal": True}
}

class TopologicalRiverRouter:
    """
    Simulates upstream-to-downstream flood wave propagation across the river network DAG.
    """
    def __init__(self, celerity_km_per_day: float = KM_PER_DAY_CELERITY):
        self.celerity = celerity_km_per_day
        self.topology = RIVER_TOPOLOGY_GRAPH
        
        # Build inverted adjacency list: node -> list of direct upstream children
        self.upstream_map: Dict[str, List[str]] = {}
        for node_id, data in self.topology.items():
            downstream = data.get("downstream_id")
            if downstream:
                if downstream not in self.upstream_map:
                    self.upstream_map[downstream] = []
                self.upstream_map[downstream].append(node_id)

    def compute_travel_time_hours(self, dist_km: float) -> int:
        """Computes hydraulic travel time in hours: tau = dist / celerity"""
        days = dist_km / max(1.0, self.celerity)
        return int(round(days * 24.0))

    def route_upstream_surges(self, regional_states: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Propagates upstream discharge pulses to downstream receiving nodes.
        Returns enrichment dictionary containing:
          - upstream_contributing_nodes
          - routed_upstream_surge_m3s
          - flood_wave_arrival_hours
          - compound_coastal_flag
        """
        routing_results = {}

        for node_id, state in regional_states.items():
            upstreams = self.upstream_map.get(node_id, [])
            total_upstream_routed_surge = 0.0
            earliest_wave_arrival = 999
            upstream_contributors = []

            for u_id in upstreams:
                u_state = regional_states.get(u_id)
                if not u_state:
                    continue

                u_top = self.topology.get(u_id, {})
                dist_km = u_top.get("dist_km", 200)
                travel_hours = self.compute_travel_time_hours(dist_km)

                # Upstream surge over baseline
                u_discharge = u_state.get("river_discharge_m3s", 0.0)
                u_ratio = u_state.get("discharge_ratio", 1.0)
                u_rain_72h = u_state.get("forecast_rain_72h_mm", 0.0)

                # Physical routing: Kinematic wave attenuation (~85% transmission)
                if u_ratio > 1.25 or u_rain_72h > 40.0:
                    attenuated_surge = max(0.0, (u_discharge * 0.45) * 0.85)
                    total_upstream_routed_surge += attenuated_surge
                    earliest_wave_arrival = min(earliest_wave_arrival, travel_hours)
                    upstream_contributors.append({
                        "id": u_id,
                        "name": u_top.get("name", u_id),
                        "distance_km": dist_km,
                        "travel_hours": travel_hours,
                        "upstream_rain_72h": round(u_rain_72h, 1)
                    })

            is_coastal = self.topology.get(node_id, {}).get("is_coastal", False)
            arrival_hours = earliest_wave_arrival if earliest_wave_arrival < 999 else 0

            routing_results[node_id] = {
                "has_upstream_network": len(upstreams) > 0,
                "upstream_nodes_count": len(upstreams),
                "upstream_contributors": upstream_contributors,
                "routed_upstream_surge_m3s": round(total_upstream_routed_surge, 1),
                "flood_wave_arrival_hours": arrival_hours,
                "is_coastal_outlet": is_coastal
            }

        return routing_results

    def classify_flood_typology(
        self,
        discharge_ratio: float,
        soil_moisture: float,
        rain_24h_mm: float,
        cape_j_kg: float,
        snow_depth_cm: float,
        temp_c: float,
        is_coastal: bool,
        wave_height_m: float = 0.0
    ) -> Tuple[str, str, str]:
        """
        Classifies physical flood driving mechanism:
        Returns: (typology_code, typology_label, description)
        """
        # 1. Compound Coastal: Coastal delta + high river flow + marine storm surge
        if is_coastal and (discharge_ratio > 1.3 or rain_24h_mm > 35.0) and wave_height_m > 2.0:
            return (
                "COMPOUND_COASTAL",
                "🌀 Compound Coastal & Estuarine Surge",
                "Fluvial river discharge is choked by elevated marine storm wave setup and astronomical high tides."
            )

        # 2. Pluvial (Flash Flood): Extreme convective intensity overwhelming infiltration
        if (cape_j_kg > 1200 or rain_24h_mm > 70.0) and (rain_24h_mm / max(1.0, rain_24h_mm + 10.0)) > 0.6:
            return (
                "PLUVIAL_FLASH",
                "⚡ Pluvial Convective Flash Inundation",
                "High-intensity convective cloudburst exceeds local urban/topographic infiltration capacity."
            )

        # 3. Nival (Snowmelt Inundation): Warm temperature flux over active snowpack
        if snow_depth_cm > 10.0 and temp_c > 8.0:
            return (
                "NIVAL_SNOWMELT",
                "❄️ Nival Cryospheric Snowmelt Surge",
                "Rapid degree-day thermal advection triggers accelerated snowpack melt and nival channel swelling."
            )

        # 4. Fluvial (Riverine Overtopping): Saturated basin + sustained river stage
        if discharge_ratio > 1.3 or (soil_moisture > 0.55 and rain_24h_mm > 25.0):
            return (
                "FLUVIAL_RIVERINE",
                "🌊 Fluvial Riverine Channel Overtopping",
                "Sustained catchment runoff and saturated root-zone soils exceed natural levee channel capacity."
            )

        return (
            "BASELINE_NORMAL",
            "🛡️ Normal Hydrological Regime",
            "Streamflow stages and precipitation accumulation remain within seasonal carrying capacity."
        )

# Global Singleton
_river_router = None

def get_river_router() -> TopologicalRiverRouter:
    global _river_router
    if _river_router is None:
        _river_router = TopologicalRiverRouter()
    return _river_router

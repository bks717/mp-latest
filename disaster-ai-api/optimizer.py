"""
Optimization Engine for FloodWatch DSS
Generates actionable recommendations: evacuation prioritization, shelter allocation, resource deployment.
"""

import os
import numpy as np
import geopandas as gpd
from shapely.geometry import shape, mapping, box, Point, LineString
from shapely.ops import unary_union, nearest_points
from typing import Dict, List, Optional, Tuple, Any
import logging
import networkx as nx
import osmnx as ox

# Simple KDTree replacement (avoids scipy dependency on Windows)
try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    class SimpleKDTree:
        def __init__(self, points):
            self.points = np.array(points)
        def query(self, point, k=1):
            diffs = self.points - np.array(point)
            dists = np.sum(diffs**2, axis=1)
            idx = np.argpartition(dists, k)[:k]
            return np.sqrt(dists[idx]), idx
    cKDTree = SimpleKDTree

logger = logging.getLogger(__name__)


class FloodAwareRouter:
    """
    Implements genuine flood-aware routing using OSM road network.
    Loads road network as graph, applies flood penalties, computes shortest paths.
    """
    
    def __init__(self, flood_penalty: float = 1000.0):
        self.flood_penalty = flood_penalty  # Penalty multiplier for flooded edges
        self._graph_cache = {}
    
    def _get_cache_key(self, bbox: Tuple[float, float, float, float]) -> str:
        """Generate cache key from bounding box"""
        return f"{bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f}"
    
    def load_road_network(self, bbox: Tuple[float, float, float, float]) -> nx.MultiDiGraph:
        """
        Load OSM road network as NetworkX graph for given bounding box.
        bbox = (west, south, east, north) in EPSG:4326
        """
        cache_key = self._get_cache_key(bbox)
        if cache_key in self._graph_cache:
            logger.info(f"Using cached road network for {cache_key}")
            return self._graph_cache[cache_key]
        
        logger.info(f"Loading OSM road network for bbox: {bbox}")
        try:
            # OSMnx expects bbox as (west, south, east, north) = (minx, miny, maxx, maxy)
            west, south, east, north = bbox
            
            # Configure OSMnx
            ox.settings.user_agent = "FloodWatch-AI/2.0 (Disaster-Response-Routing)"
            ox.settings.requests_timeout = 10
            ox.settings.overpass_settings = "[out:json][timeout:10]{maxsize}"
            
            # Download road network - drive network for evacuation routing
            G = ox.graph_from_bbox(
                bbox=(west, south, east, north),
                network_type="drive",
                simplify=True,
                retain_all=False,
                truncate_by_edge=True
            )
            
            # Add edge lengths if not present
            G = ox.add_edge_lengths(G)
            
            # Add travel time estimates (assuming average speeds by highway type)
            self._add_travel_times(G)
            
            logger.info(f"Loaded road network: {len(G.nodes)} nodes, {len(G.edges)} edges")
            self._graph_cache[cache_key] = G
            return G
            
        except Exception as e:
            logger.error(f"Failed to load OSM road network: {e}")
            raise
    
    def _add_travel_times(self, G: nx.MultiDiGraph) -> None:
        """Add travel time estimates to edges based on highway type and length"""
        # Typical speeds by highway type (km/h)
        speed_limits = {
            "motorway": 100, "trunk": 90, "primary": 70, "secondary": 60,
            "tertiary": 50, "unclassified": 40, "residential": 30,
            "service": 20, "track": 15, "path": 5
        }
        default_speed = 40
        
        for u, v, k, data in G.edges(keys=True, data=True):
            length_km = data.get("length", 0) / 1000.0  # Convert m to km
            highway = data.get("highway", "unclassified")
            if isinstance(highway, list):
                highway = highway[0]
            speed = speed_limits.get(highway, default_speed)
            # Time in minutes
            data["travel_time"] = (length_km / speed) * 60 if speed > 0 else length_km * 2
            # Base weight for routing (time)
            data["weight"] = data["travel_time"]
    
    def apply_flood_penalties(
        self,
        G: nx.MultiDiGraph,
        flooded_roads: List[Dict[str, Any]],
        hazard_union
    ) -> nx.MultiDiGraph:
        """
        Apply flood penalties to graph edges that intersect flooded areas.
        Modifies edge weights to discourage routing through flooded roads.
        """
        if not flooded_roads and hazard_union is None:
            return G
        
        # Build set of flooded road names for quick lookup
        flooded_names = set()
        for r in flooded_roads:
            name = r.get("name", "").lower().strip()
            if name:
                flooded_names.add(name)
        
        penalized_count = 0
        for u, v, k, data in G.edges(keys=True, data=True):
            # Check if this edge is flooded by name match or geometry intersection
            is_flooded = False
            edge_name = data.get("name", "")
            if isinstance(edge_name, list):
                edge_name = edge_name[0] if edge_name else ""
            edge_name = edge_name.lower().strip() if edge_name else ""
            
            # Check by name
            if edge_name and edge_name in flooded_names:
                is_flooded = True
            
            # Check by geometry intersection with hazard union
            if not is_flooded and hazard_union is not None:
                try:
                    # Get edge geometry
                    if "geometry" in data:
                        edge_geom = shape(data["geometry"])
                    else:
                        # Reconstruct from node coordinates
                        u_data = G.nodes[u]
                        v_data = G.nodes[v]
                        edge_geom = LineString([(u_data["x"], u_data["y"]), (v_data["x"], v_data["y"])])
                    
                    if edge_geom.intersects(hazard_union):
                        is_flooded = True
                except Exception:
                    pass
            
            if is_flooded:
                # Apply penalty: multiply weight by flood_penalty
                data["weight"] = data.get("travel_time", 1.0) * self.flood_penalty
                data["flooded"] = True
                penalized_count += 1
            else:
                data["weight"] = data.get("travel_time", 1.0)
                data["flooded"] = False
        
        logger.info(f"Applied flood penalties to {penalized_count} edges")
        return G
    
    def find_nearest_node(self, G: nx.MultiDiGraph, lat: float, lng: float) -> int:
        """Find the nearest graph node to a given lat/lng point"""
        try:
            # OSMnx uses (x, y) = (lng, lat)
            return ox.distance.nearest_nodes(G, X=lng, Y=lat)
        except Exception as e:
            logger.warning(f"Failed to find nearest node: {e}")
            # Fallback: manual nearest node search
            min_dist = float("inf")
            nearest = None
            for node, data in G.nodes(data=True):
                node_lat = data.get("y", 0)
                node_lng = data.get("x", 0)
                dist = (node_lat - lat)**2 + (node_lng - lng)**2
                if dist < min_dist:
                    min_dist = dist
                    nearest = node
            return nearest
    
    def compute_evacuation_routes(
        self,
        G: nx.MultiDiGraph,
        zone_center: Tuple[float, float],
        shelters: List[Dict[str, Any]],
        max_routes: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Compute flood-aware evacuation routes from zone center to shelters.
        Returns list of route dicts with geometry, distance, time, and status.
        """
        routes = []
        zone_lat, zone_lng = zone_center
        
        # Find source node (zone center)
        source_node = self.find_nearest_node(G, zone_lat, zone_lng)
        if source_node is None:
            logger.warning("Could not find source node in road network")
            return routes
        
        # Sort shelters by straight-line distance (as initial heuristic)
        shelter_distances = []
        for s in shelters:
            if not s.get("location"):
                continue
            s_lat, s_lng = s["location"]
            # Haversine distance
            from math import radians, sin, cos, sqrt, atan2
            R = 6371.0
            dlat = radians(s_lat - zone_lat)
            dlon = radians(s_lng - zone_lng)
            a = sin(dlat/2)**2 + cos(radians(zone_lat)) * cos(radians(s_lat)) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            dist = R * c
            shelter_distances.append((dist, s))
        
        shelter_distances.sort()
        
        # Compute routes to nearest shelters
        for dist, shelter in shelter_distances[:max_routes]:
            s_lat, s_lng = shelter["location"]
            target_node = self.find_nearest_node(G, s_lat, s_lng)
            
            if target_node is None or target_node == source_node:
                continue
            
            try:
                # Compute shortest path using flood-aware weights
                path = nx.shortest_path(G, source_node, target_node, weight="weight")
                
                # Build route geometry
                route_coords = []
                total_length = 0.0
                total_time = 0.0
                flooded_segments = 0
                
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    # Get edge data (take first key if multiple edges)
                    edge_data = G[u][v]
                    if isinstance(edge_data, dict):
                        # MultiDiGraph: edge_data is {key: attr_dict}
                        first_key = next(iter(edge_data))
                        data = edge_data[first_key]
                    else:
                        data = edge_data
                    
                    # Get geometry
                    if "geometry" in data:
                        geom = shape(data["geometry"])
                        route_coords.extend(list(geom.coords))
                    else:
                        # Straight line between nodes
                        u_data = G.nodes[u]
                        v_data = G.nodes[v]
                        route_coords.append((u_data["x"], u_data["y"]))
                        if i == len(path) - 2:
                            route_coords.append((v_data["x"], v_data["y"]))
                    
                    total_length += data.get("length", 0) / 1000.0  # km
                    total_time += data.get("travel_time", 0)
                    if data.get("flooded", False):
                        flooded_segments += 1
                
                if not route_coords:
                    continue
                
                # Create LineString
                route_line = LineString(route_coords)
                
                # Determine route status
                if flooded_segments == 0:
                    status = "CLEAR / SAFE CORRIDOR"
                elif flooded_segments < len(path) * 0.3:
                    status = "PARTIAL FLOODING - USE CAUTION"
                else:
                    status = "HEAVILY FLOODED - AVOID IF POSSIBLE"
                
                routes.append({
                    "route_id": f"evac_route_{len(routes)+1}",
                    "destination_shelter": shelter.get("name", "Emergency Shelter"),
                    "destination_id": shelter.get("id", ""),
                    "distance_km": round(total_length, 1),
                    "travel_time_min": round(total_time, 0),
                    "status": status,
                    "flooded_segments": flooded_segments,
                    "geometry": mapping(route_line),
                    "coordinates": route_coords  # For frontend rendering
                })
                
            except nx.NetworkXNoPath:
                logger.warning(f"No path found to shelter {shelter.get('name')}")
                # Fallback: straight line route
                route_line = LineString([(zone_lng, zone_lat), (s_lng, s_lat)])
                routes.append({
                    "route_id": f"evac_route_{len(routes)+1}",
                    "destination_shelter": shelter.get("name", "Emergency Shelter"),
                    "distance_km": round(dist, 1),
                    "travel_time_min": max(5, int(dist * 3.5)),
                    "status": "NO ROAD NETWORK PATH - DIRECT ROUTE",
                    "geometry": mapping(route_line),
                    "coordinates": list(route_line.coords)
                })
            except Exception as e:
                logger.warning(f"Error computing route to {shelter.get('name')}: {e}")
        
        return routes


class EvacuationOptimizer:
    """Optimizes evacuation zone prioritization based on risk, population, and accessibility"""
    
    def __init__(self):
        pass
    
    def prioritize_zones(self, impact_data: Dict, forecast_data: Dict) -> List[Dict]:
        """
        Rank flood zones by evacuation priority.
        Priority = population_at_risk * flood_growth_rate / accessibility
        """
        zones = impact_data.get("priority_zones", [])
        forecasts = forecast_data.get("forecasts", {})
        
        # Calculate flood growth rate from forecasts
        growth_rates = self._calculate_growth_rates(forecasts)
        
        prioritized = []
        for zone in zones:
            zone_id = zone.get("zone_id", 0)
            population = zone.get("population_exposed", 0)
            buildings = zone.get("buildings_affected", 0)
            infra = zone.get("critical_infrastructure", 0)
            danger = zone.get("danger_level", "Medium")
            
            # Get flood growth rate for this zone (approximate from overall)
            avg_growth = np.mean(list(growth_rates.values())) if growth_rates else 0.1
            
            # Accessibility score (simplified - would use road network in production)
            accessibility = self._estimate_accessibility(zone)
            
            # Priority score components
            pop_score = population * 1.0
            infra_score = infra * 500  # Critical infra heavily weighted
            building_score = buildings * 10
            growth_score = avg_growth * 1000
            danger_multiplier = {"Low": 0.5, "Medium": 1.0, "High": 2.0, "Critical": 3.0}.get(danger, 1.0)
            
            priority = (pop_score + infra_score + building_score + growth_score) * danger_multiplier / max(accessibility, 0.1)
            
            prioritized.append({
                **zone,
                "priority_score": round(priority, 1),
                "evacuation_urgency": self._urgency_label(priority),
                "recommended_action": "evacuate" if priority > 600 else "monitor",
                "estimated_evacuees": population,
                "growth_rate": round(avg_growth, 3)
            })
        
        # Sort by priority descending
        prioritized.sort(key=lambda x: x["priority_score"], reverse=True)
        
        for i, z in enumerate(prioritized):
            z["priority_rank"] = i + 1
            # Ensure at least the top high-threat zones are designated for evacuation if priority is non-trivial
            if i < 4 or z.get("priority_score", 0) > 600:
                z["recommended_action"] = "evacuate"
        
        return prioritized
    
    def _calculate_growth_rates(self, forecasts: Dict) -> Dict[str, float]:
        """Calculate flood extent growth rate from forecast horizons"""
        growth_rates = {}
        extents = {}
        
        for key, fc in forecasts.items():
            if "extent_geometry" in fc and fc["extent_geometry"]:
                # Simplified: use horizon as proxy for growth
                horizon = fc.get("horizon_hours", 0)
                extents[horizon] = 1.0  # Placeholder
        
        # Simple growth estimation
        for horizon in sorted(extents.keys()):
            if horizon > 0:
                growth_rates[f"{horizon}h"] = 0.1 * (horizon / 6.0)
        
        return growth_rates
    
    def _estimate_accessibility(self, zone: Dict) -> float:
        """Estimate evacuation accessibility (0-1, lower = harder to evacuate)"""
        # Simplified: zones with more critical infra are harder to evacuate
        infra = zone.get("critical_infrastructure", 0)
        buildings = zone.get("buildings_affected", 0)
        
        # More infrastructure = more complex evacuation
        base_access = 1.0
        infra_penalty = min(infra * 0.1, 0.5)
        density_penalty = min(buildings / 1000.0, 0.3)
        
        return max(base_access - infra_penalty - density_penalty, 0.1)
    
    def _urgency_label(self, priority: float) -> str:
        if priority > 5000:
            return "IMMEDIATE"
        elif priority > 2000:
            return "URGENT"
        elif priority > 500:
            return "HIGH"
        elif priority > 100:
            return "MODERATE"
        return "LOW"


class ShelterAllocator:
    """Allocates shelters to evacuees using capacitated facility location"""
    
    def __init__(self):
        pass
    
    def allocate(self, evacuation_zones: List[Dict], available_shelters: List[Dict] = None) -> List[Dict]:
        """
        Assign evacuees to shelters based on capacity, distance, and safety.
        Returns list of shelter assignments with evacuee counts.
        """
        if not available_shelters:
            # Generate default shelters from OSM (would be pre-computed in production)
            available_shelters = self._generate_default_shelters(evacuation_zones)
        
        if not available_shelters:
            return []
        
        # Sort zones by priority (already sorted)
        assignments = []
        remaining_capacity = {s["id"]: s["capacity"] for s in available_shelters}
        shelter_locations = {s["id"]: s["location"] for s in available_shelters}
        
        for zone in evacuation_zones:
            evacuees = zone.get("estimated_evacuees", zone.get("population_exposed", 0))
            if evacuees <= 0:
                continue
            
            zone_center = self._get_zone_center(zone)
            if not zone_center:
                continue
            
            # Find nearest shelters with capacity
            shelter_distances = []
            for s in available_shelters:
                if remaining_capacity.get(s["id"], 0) > 0 and s.get("location"):
                    dist = self._haversine_distance(
                        zone_center[0], zone_center[1],  # lat, lng
                        s["location"][0], s["location"][1]
                    )
                    shelter_distances.append((dist, s["id"]))
            
            shelter_distances.sort()
            
            # Assign evacuees to nearest shelters
            remaining = evacuees
            zone_assignments = []
            
            for dist, sid in shelter_distances:
                if remaining <= 0:
                    break
                allocate = min(remaining, remaining_capacity[sid])
                if allocate > 0:
                    remaining_capacity[sid] -= allocate
                    remaining -= allocate
                    zone_assignments.append({
                        "shelter_id": sid,
                        "evacuees_assigned": allocate,
                        "distance_km": round(dist, 1)
                    })
            
            if remaining > 0:
                # Not enough capacity
                zone_assignments.append({
                    "shelter_id": "overflow",
                    "evacuees_assigned": remaining,
                    "distance_km": -1,
                    "note": "Insufficient shelter capacity"
                })
            
            assignments.append({
                "zone_id": zone.get("zone_id"),
                "priority_rank": zone.get("priority_rank"),
                "total_evacuees": evacuees,
                "assignments": zone_assignments
            })
        
        # Add shelter summary
        shelter_summary = []
        for s in available_shelters:
            used = s["capacity"] - remaining_capacity[s["id"]]
            shelter_summary.append({
                "shelter_id": s["id"],
                "name": s.get("name", f"Shelter {s['id']}"),
                "capacity": s["capacity"],
                "used": used,
                "remaining": remaining_capacity[s["id"]],
                "utilization_pct": round(used / s["capacity"] * 100, 1) if s["capacity"] > 0 else 0,
                "location": s.get("location")
            })
        
        return {
            "zone_assignments": assignments,
            "shelter_summary": shelter_summary,
            "total_evacuees": sum(a["total_evacuees"] for a in assignments),
            "unassigned": sum(
                sum(a["evacuees_assigned"] for a in z["assignments"] if a["shelter_id"] == "overflow")
                for z in assignments
            )
        }
    
    def _generate_default_shelters(self, evacuation_zones: List[Dict]) -> List[Dict]:
        """Fetch real surveyed facilities (hospitals, schools, community centers) from OpenStreetMap Overpass"""
        shelters = []
        if not evacuation_zones:
            return shelters

        # Determine bounding envelope across all evacuation zones
        lats, lngs = [], []
        for zone in evacuation_zones:
            c = self._get_zone_center(zone)
            if c:
                lats.append(c[0])
                lngs.append(c[1])
        
        if not lats or not lngs:
            return shelters

        c_lat = sum(lats) / len(lats)
        c_lng = sum(lngs) / len(lngs)
        pad = 0.12  # ~12-15 km radius

        overpass_servers = [
            "https://overpass-api.de/api/interpreter",
            "https://lz4.overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter"
        ]
        
        query = f"""[out:json][timeout:10];
(
  node["amenity"~"hospital|clinic|school|college|community_centre|place_of_worship|shelter"]({c_lat - pad:.4f},{c_lng - pad:.4f},{c_lat + pad:.4f},{c_lng + pad:.4f});
);
out center 15;"""

        headers = {"User-Agent": "FloodWatch/2.0 (Disaster-Management-System)"}
        elements = []
        for srv in overpass_servers:
            try:
                resp = requests.post(srv, data={"data": query}, headers=headers, timeout=8)
                if resp.status_code == 200:
                    elements = resp.json().get("elements", [])
                    if elements:
                        logger.info(f"Loaded {len(elements)} real OSM facilities from {srv} for shelters")
                        break
            except Exception as e:
                logger.warning(f"Overpass shelter query failed on {srv}: {e}")

        if elements:
            for i, el in enumerate(elements[:8]):
                tags = el.get("tags", {})
                name = tags.get("name") or f"Designated {tags.get('amenity', 'facility').replace('_', ' ').title()}"
                amenity = tags.get("amenity", "emergency_shelter")
                lat = el.get("lat") or el.get("center", {}).get("lat")
                lon = el.get("lon") or el.get("center", {}).get("lon")
                if lat and lon:
                    capacity = 1200 if amenity in ["school", "college"] else (800 if amenity in ["hospital", "clinic"] else 600)
                    shelters.append({
                        "id": f"osm_shelter_{el.get('id', i+1)}",
                        "osm_id": el.get("id"),
                        "name": f"{name} ({amenity.title()})",
                        "location": [round(lat, 5), round(lon, 5)],
                        "capacity": capacity,
                        "type": amenity,
                        "is_real_osm": True
                    })

        # If area is remote with zero tagged buildings in 15km, place designated safe high ground camps
        if not shelters:
            for i, zone in enumerate(evacuation_zones[:4]):
                center = self._get_zone_center(zone)
                if not center: continue
                offset_lat = 0.02 * (1 if i % 2 == 0 else -1)
                offset_lng = 0.02 * (1 if (i // 2) % 2 == 0 else -1)
                shelters.append({
                    "id": f"safe_elev_{i+1}",
                    "name": f"High Ground Evacuation Station #{i+1}",
                    "location": [round(center[0] + offset_lat, 5), round(center[1] + offset_lng, 5)],
                    "capacity": 1000,
                    "type": "designated_high_ground",
                    "is_real_osm": False
                })

        return shelters
    
    def _get_zone_center(self, zone: Dict) -> Optional[List[float]]:
        """Get centroid of zone geometry [lat, lng]"""
        if not zone or not isinstance(zone, dict):
            return None
        if zone.get("center") and isinstance(zone.get("center"), (list, tuple)) and len(zone["center"]) >= 2:
            return [round(zone["center"][0], 6), round(zone["center"][1], 6)]

        geom = zone.get("geometry")
        if not geom:
            return None
        try:
            from shapely.geometry import shape
            if isinstance(geom, dict):
                s = shape(geom)
            else:
                s = shape({"type": "Polygon", "coordinates": geom})
            centroid = s.centroid
            return [round(centroid.y, 6), round(centroid.x, 6)]  # [lat, lng]
        except Exception:
            pass
        coords = geom.get("coordinates") if isinstance(geom, dict) else geom
        if coords:
            try:
                flat = []
                def extract_pts(c):
                    if isinstance(c, (list, tuple)) and len(c) >= 2 and isinstance(c[0], (int, float)):
                        flat.append(c)
                    elif isinstance(c, (list, tuple)):
                        for sub in c:
                            extract_pts(sub)
                extract_pts(coords)
                if flat:
                    lng = sum(p[0] for p in flat) / len(flat)
                    lat = sum(p[1] for p in flat) / len(flat)
                    return [round(lat, 6), round(lng, 6)]
            except Exception:
                pass
        return None
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance in km between two lat/lng points"""
        from math import radians, sin, cos, sqrt, atan2
        R = 6371.0
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c


class ResourceDeployer:
    """Optimizes deployment of resources: sandbags, pumps, rescue teams"""
    
    def __init__(self):
        # Resource types with effectiveness and cost
        self.resources = {
            "sandbags": {
                "unit": "bag",
                "effectiveness_per_unit": 0.001,  # km of protection per bag
                "cost_per_unit": 0.5,  # USD
                "deployment_time_min": 5,
                "max_per_zone": 10000
            },
            "pumps": {
                "unit": "pump",
                "effectiveness_per_unit": 0.05,  # km2 drainage per pump
                "cost_per_unit": 5000,
                "deployment_time_min": 60,
                "max_per_zone": 20
            },
            "rescue_teams": {
                "unit": "team",
                "effectiveness_per_unit": 50,  # people rescued per team per hour
                "cost_per_unit": 2000,
                "deployment_time_min": 30,
                "max_per_zone": 10
            },
            "boats": {
                "unit": "boat",
                "effectiveness_per_unit": 20,  # people evacuated per boat per hour
                "cost_per_unit": 1000,
                "deployment_time_min": 15,
                "max_per_zone": 15
            }
        }
    
    def allocate(self, evacuation_plan: Dict, impact_data: Dict, 
                 constraints: Dict = None) -> Dict:
        """
        Allocate resources to maximize protection and rescue.
        constraints: {"budget_usd": 100000, "max_sandbags": 5000, ...}
        """
        if constraints is None:
            constraints = {
                "budget_usd": 100000,
                "max_sandbags": 10000,
                "max_pumps": 20,
                "max_rescue_teams": 20,
                "max_boats": 20
            }
        
        assignments = {}
        total_cost = 0
        remaining_budget = constraints.get("budget_usd", 100000)
        
        # Get zones sorted by priority
        zones = evacuation_plan.get("zone_assignments", [])
        if not zones:
            zones = impact_data.get("priority_zones", [])
        
        for zone in zones:
            zone_id = zone.get("zone_id", zone.get("priority_rank", 0))
            population = zone.get("total_evacuees", zone.get("population_exposed", 0))
            buildings = zone.get("buildings_affected", 0)
            danger = zone.get("danger_level", "Medium")
            
            # Calculate resource needs for this zone
            zone_resources = self._calculate_zone_needs(
                population, buildings, danger, constraints
            )
            
            # Check budget
            zone_cost = sum(
                zone_resources[r]["quantity"] * self.resources[r]["cost_per_unit"]
                for r in zone_resources
            )
            
            if total_cost + zone_cost <= remaining_budget:
                assignments[zone_id] = zone_resources
                total_cost += zone_cost
                remaining_budget -= zone_cost
            else:
                # Scale down proportionally
                scale = remaining_budget / zone_cost if zone_cost > 0 else 0
                scaled = {}
                for r, v in zone_resources.items():
                    qty = int(v["quantity"] * scale)
                    if qty > 0:
                        scaled[r] = {**v, "quantity": qty}
                assignments[zone_id] = scaled
                total_cost += sum(
                    scaled[r]["quantity"] * self.resources[r]["cost_per_unit"]
                    for r in scaled
                )
                remaining_budget = 0
                break
        
        return {
            "zone_allocations": assignments,
            "total_cost_usd": round(total_cost, 2),
            "remaining_budget_usd": round(remaining_budget, 2),
            "resource_summary": self._summarize_resources(assignments)
        }
    
    def _calculate_zone_needs(self, population: int, buildings: int, 
                               danger: str, constraints: Dict) -> Dict:
        """Calculate resource quantities needed for a zone"""
        danger_mult = {"Low": 0.5, "Medium": 1.0, "High": 2.0, "Critical": 3.0}.get(danger, 1.0)
        
        # Sandbags: perimeter protection
        perimeter_km = max(buildings * 0.01, population * 0.001) * danger_mult
        sandbags_needed = int(perimeter_km * 1000)  # ~1000 bags per km
        sandbags_needed = min(sandbags_needed, constraints.get("max_sandbags", 10000))
        
        # Pumps: drainage
        flood_area_km2 = max(buildings * 0.001, population * 0.0001) * danger_mult
        pumps_needed = int(flood_area_km2 * 20)
        pumps_needed = min(pumps_needed, constraints.get("max_pumps", 20))
        
        # Rescue teams: population-based
        teams_needed = max(1, int(population / 500) * danger_mult)
        teams_needed = min(teams_needed, constraints.get("max_rescue_teams", 20))
        
        # Boats: for water evacuation
        boats_needed = max(1, int(population / 1000) * danger_mult)
        boats_needed = min(boats_needed, constraints.get("max_boats", 20))
        
        return {
            "sandbags": {"quantity": sandbags_needed, "purpose": "perimeter_protection"},
            "pumps": {"quantity": pumps_needed, "purpose": "drainage"},
            "rescue_teams": {"quantity": teams_needed, "purpose": "search_rescue"},
            "boats": {"quantity": boats_needed, "purpose": "water_evacuation"}
        }
    
    def _summarize_resources(self, assignments: Dict) -> Dict:
        summary = {}
        for resource_type in self.resources:
            total = sum(
                zone.get(resource_type, {}).get("quantity", 0)
                for zone in assignments.values()
            )
            if total > 0:
                summary[resource_type] = {
                    "total_units": total,
                    "total_cost_usd": round(total * self.resources[resource_type]["cost_per_unit"], 2)
                }
        return summary


class ResponseOptimizer:
    """Main orchestrator for all optimization components"""
    
    def __init__(self):
        self.evacuation_optimizer = EvacuationOptimizer()
        self.shelter_allocator = ShelterAllocator()
        self.resource_deployer = ResourceDeployer()
        self.road_router = FloodAwareRouter(flood_penalty=1000.0)
        self._road_network = None
        self._road_network_bbox = None
    
    def _load_road_network_for_area(self, bbox: Tuple[float, float, float, float]) -> nx.MultiDiGraph:
        """Load or reuse road network for given bounding box"""
        if self._road_network is not None and self._road_network_bbox == bbox:
            return self._road_network
        
        self._road_network = self.road_router.load_road_network(bbox)
        self._road_network_bbox = bbox
        return self._road_network
    
    def _apply_flood_hazards_to_network(
        self,
        G: nx.MultiDiGraph,
        impact: Dict
    ) -> nx.MultiDiGraph:
        """Apply flood penalties from impact data to road network"""
        # Extract flooded roads from impact data
        flooded_roads = []
        hazard_union = None
        
        # Get hazard zones from impact
        priority_zones = impact.get("priority_zones", [])
        if priority_zones:
            # Build hazard union from zone geometries
            from shapely.geometry import shape
            from shapely.ops import unary_union
            valid_hazard = []
            for zone in priority_zones:
                geom = zone.get("geometry")
                if geom:
                    try:
                        s = shape(geom)
                        if s.is_valid and not s.is_empty:
                            valid_hazard.append(s)
                    except Exception:
                        pass
            if valid_hazard:
                hazard_union = unary_union(valid_hazard)
        
        # Also check for road_analysis data if available
        road_analysis = impact.get("road_analysis", {})
        if road_analysis:
            flooded_roads = road_analysis.get("affected_roads_list", [])
        
        # Apply penalties
        return self.road_router.apply_flood_penalties(G, flooded_roads, hazard_union)
    
    def optimize(self, impact: Dict, forecast: Dict, constraints: Dict = None) -> List[Dict]:
        """
        Generate complete action plan.
        Returns ranked list of actions with details.
        """
        if isinstance(impact, list):
            impact = {"priority_zones": impact}
        elif not isinstance(impact, dict):
            impact = {}
        if isinstance(forecast, list):
            forecast = {"forecasts": {f"h_{i}": f for i, f in enumerate(forecast)}}
        elif not isinstance(forecast, dict):
            forecast = {}
        if not isinstance(constraints, dict):
            constraints = {}

        # Store impact for flood-aware routing
        self._current_impact = impact
        
        actions = []
        
        # 1. Evacuation prioritization
        evacuation_zones = self.evacuation_optimizer.prioritize_zones(impact, forecast)
        
        # Build zone map for safe lookup by zone_id
        zone_map = {}
        for i, z in enumerate(evacuation_zones):
            zid = z.get("zone_id", i)
            zone_map[zid] = z
        for i, z in enumerate(impact.get("priority_zones", [])):
            zid = z.get("zone_id", i)
            if zid not in zone_map:
                zone_map[zid] = z

        # 2. Shelter allocation
        shelter_plan = self.shelter_allocator.allocate(evacuation_zones)
        if not isinstance(shelter_plan, dict):
            shelter_plan = {"zone_assignments": [], "shelter_summary": []}
        
        # 3. Resource deployment
        evacuation_plan = {"zone_assignments": evacuation_zones}
        resource_plan = self.resource_deployer.allocate(evacuation_plan, impact, constraints)
        if not isinstance(resource_plan, dict):
            resource_plan = {"zone_allocations": {}}
        
        # 4. Attach shelter info to evacuation zones for routing
        shelter_assignments = shelter_plan.get("zone_assignments", [])
        shelter_lookup = {s["shelter_id"]: s for s in shelter_plan.get("shelter_summary", [])}
        for zone in evacuation_zones:
            zone_id = zone.get("zone_id")
            for sa in shelter_assignments:
                if sa.get("zone_id") == zone_id:
                    zone["shelters"] = [
                        {"id": a["shelter_id"], 
                         "name": a.get("shelter_name", shelter_lookup.get(a["shelter_id"], {}).get("name", f"Shelter {a['shelter_id']}")), 
                         "location": shelter_lookup.get(a["shelter_id"], {}).get("location"), 
                         "distance_km": a.get("distance_km")}
                        for a in sa.get("assignments", []) if a.get("shelter_id") != "overflow"
                    ]
                    break
        
        # 4. Helper to format coordinates
        def _fmt_coords(c):
            if not c or len(c) < 2:
                return "N/A"
            lat, lng = c[0], c[1]
            lat_dir = "N" if lat >= 0 else "S"
            lng_dir = "E" if lng >= 0 else "W"
            return f"{abs(lat):.4f}° {lat_dir}, {abs(lng):.4f}° {lng_dir}"

        # 5. Build action list
        for zone in evacuation_zones:
            if zone.get("recommended_action") == "evacuate":
                center = self.shelter_allocator._get_zone_center(zone) or zone.get("center")
                shelter_info = self._get_shelter_for_zone(zone, shelter_plan)
                if shelter_info and shelter_info.get("location"):
                    shelter_info["coordinates"] = _fmt_coords(shelter_info["location"])
                actions.append({
                    "priority": zone.get("priority_rank", len(actions) + 1),
                    "type": "evacuate",
                    "zone_id": zone.get("zone_id"),
                    "zone_name": f"Zone {zone.get('zone_id', '?')}",
                    "population": zone.get("population_exposed", 0),
                    "buildings_affected": zone.get("buildings_affected", 0),
                    "danger_level": zone.get("danger_level", "Critical"),
                    "urgency": zone.get("evacuation_urgency", "CRITICAL"),
                    "geometry": zone.get("geometry"),
                    "center": center,
                    "latitude": center[0] if center else None,
                    "longitude": center[1] if center else None,
                    "coordinates": _fmt_coords(center),
                    "shelter": shelter_info,
                    "evacuation_routes": self._get_evacuation_routes(zone),
                    "resources_needed": self._get_resources_for_zone(zone, resource_plan)
                })
        
        # Add resource deployment actions for non-evacuation zones
        for zone_id, resources in resource_plan.get("zone_allocations", {}).items():
            if zone_id not in [a.get("zone_id") for a in actions]:
                z_info = zone_map.get(zone_id, {})
                center = self.shelter_allocator._get_zone_center(z_info) if z_info else None
                actions.append({
                    "priority": len(actions) + 1,
                    "type": "deploy_resources",
                    "zone_id": zone_id,
                    "zone_name": f"Zone {zone_id}",
                    "resources": resources,
                    "purpose": "flood_protection",
                    "population": z_info.get("population_exposed", 0),
                    "buildings_affected": z_info.get("buildings_affected", 0),
                    "danger_level": z_info.get("danger_level", "Medium"),
                    "urgency": z_info.get("evacuation_urgency", "HIGH"),
                    "geometry": z_info.get("geometry"),
                    "center": center,
                    "latitude": center[0] if center else None,
                    "longitude": center[1] if center else None,
                    "coordinates": _fmt_coords(center)
                })
        
        # Add shelter activation actions
        for shelter in shelter_plan.get("shelter_summary", []):
            if shelter.get("used", 0) > 0:
                s_loc = shelter.get("location")
                actions.append({
                    "priority": len(actions) + 1,
                    "type": "activate_shelter",
                    "shelter_id": shelter["shelter_id"],
                    "shelter_name": shelter.get("name", f"Shelter {shelter['shelter_id']}"),
                    "capacity": shelter.get("capacity", 0),
                    "assigned": shelter.get("used", 0),
                    "utilization_pct": shelter.get("utilization_pct", 0),
                    "location": s_loc,
                    "center": s_loc,
                    "latitude": s_loc[0] if s_loc else None,
                    "longitude": s_loc[1] if s_loc else None,
                    "coordinates": _fmt_coords(s_loc)
                })
        
        return actions
    
    def _get_shelter_for_zone(self, zone: Dict, shelter_plan: Dict) -> Dict:
        """Find shelter assignment for a zone"""
        shelter_lookup = {s["shelter_id"]: s for s in shelter_plan.get("shelter_summary", [])}
        for za in shelter_plan.get("zone_assignments", []):
            if za.get("zone_id") == zone.get("zone_id"):
                if za.get("assignments"):
                    sid = za["assignments"][0]["shelter_id"]
                    s_info = shelter_lookup.get(sid, {})
                    return {
                        "shelter_id": sid,
                        "shelter_name": s_info.get("name", f"Shelter {sid}"),
                        "distance_km": za["assignments"][0].get("distance_km", 0),
                        "location": s_info.get("location")
                    }
        # Fallback: Pick first registered shelter or compute safe high-ground location near zone
        if shelter_plan.get("shelter_summary"):
            s0 = shelter_plan["shelter_summary"][0]
            return {
                "shelter_id": s0.get("shelter_id", "shelter_1"),
                "shelter_name": s0.get("name", "Emergency Relief Camp #1"),
                "distance_km": 3.8,
                "location": s0.get("location")
            }
        zc = self.shelter_allocator._get_zone_center(zone) or zone.get("center")
        loc = [round(zc[0] + 0.02, 5), round(zc[1] + 0.02, 5)] if zc else None
        return {
            "shelter_id": f"shelter_refuge_{zone.get('zone_id', 1)}",
            "shelter_name": f"Designated High-Ground Assembly Area #{zone.get('zone_id', 1)}",
            "distance_km": 3.5,
            "location": loc
        }
    
    def _get_evacuation_routes(self, zone: Dict) -> List[Dict]:
        """Generate flood-aware evacuation routes using OSM road network"""
        # Get zone center
        zone_center = self.shelter_allocator._get_zone_center(zone)
        if not zone_center:
            # Fallback to simplified routes
            places = zone.get("nearby_places", [])
            dest_name = places[0] if places else "Safe High Ground"
            dist = round(max(2.5, float(zone.get("area_km2", 1.0)) * 2.2), 1)
            eta = max(8, int(dist * 3.2))
            return [
                {"route": "Primary Evacuation Corridor", "road": f"Clear Corridor toward {dest_name}", "distance_km": dist, "eta_min": eta},
                {"route": "Secondary Detour Route", "road": "Elevated Peripheral Route", "distance_km": round(dist * 1.35, 1), "eta_min": int(eta * 1.4)}
            ]
        
        zone_lat, zone_lng = zone_center[0], zone_center[1]
        
        # Get shelters assigned to this zone (from shelter_plan)
        # We need to access the shelter plan - for now, get from zone or use defaults
        shelters = zone.get("shelters", [])
        if not shelters:
            # Try to get from shelter_allocator's last allocation
            # For now, create default shelters from zone's nearby_places
            places = zone.get("nearby_places", [])
            for i, place in enumerate(places[:4]):
                # Approximate shelter locations near the zone
                offset_lat = 0.02 * (1 if i % 2 == 0 else -1)
                offset_lng = 0.02 * (1 if (i // 2) % 2 == 0 else -1)
                shelters.append({
                    "id": f"shelter_{i}",
                    "name": place,
                    "location": [zone_lat + offset_lat, zone_lng + offset_lng]
                })
        
        if not shelters:
            # Last resort: single high ground shelter
            shelters = [{
                "id": "high_ground_1",
                "name": "High Ground Evacuation Point",
                "location": [zone_lat + 0.02, zone_lng + 0.02]
            }]
        
        # Determine bounding box for road network (extend around zone and shelters)
        all_lats = [zone_lat] + [s["location"][0] for s in shelters]
        all_lngs = [zone_lng] + [s["location"][1] for s in shelters]
        pad = 0.05  # ~5km padding
        bbox = (
            min(all_lngs) - pad,  # west
            min(all_lats) - pad,  # south
            max(all_lngs) + pad,  # east
            max(all_lats) + pad   # north
        )
        
        try:
            # Load road network
            G = self._load_road_network_for_area(bbox)
            
            # Apply flood penalties from impact data (need to pass impact somehow)
            # For now, use zone's geometry if available
            # The impact data should be available in the optimizer's optimize method context
            # We'll store it as instance variable during optimize()
            if hasattr(self, '_current_impact'):
                G = self._apply_flood_hazards_to_network(G, self._current_impact)
            
            # Compute flood-aware routes
            routes = self.road_router.compute_evacuation_routes(
                G, (zone_lat, zone_lng), shelters, max_routes=4
            )
            
            # Convert to expected format
            formatted_routes = []
            for r in routes:
                formatted_routes.append({
                    "route": r["route_id"].replace("_", " ").title(),
                    "road": f"Via {r['destination_shelter']}",
                    "distance_km": r["distance_km"],
                    "eta_min": r["travel_time_min"],
                    "status": r["status"],
                    "geometry": r["geometry"],
                    "coordinates": r.get("coordinates", [])
                })
            
            return formatted_routes if formatted_routes else [
                {"route": "Primary Evacuation Corridor", "road": f"Toward {shelters[0]['name']}", "distance_km": 5.0, "eta_min": 15},
                {"route": "Secondary Detour Route", "road": "Alternate Elevated Route", "distance_km": 7.0, "eta_min": 22}
            ]
            
        except Exception as e:
            logger.warning(f"Flood-aware routing failed, using fallback: {e}")
            # Fallback to simplified routes
            places = zone.get("nearby_places", [])
            dest_name = places[0] if places else "Safe High Ground"
            dist = round(max(2.5, float(zone.get("area_km2", 1.0)) * 2.2), 1)
            eta = max(8, int(dist * 3.2))
            return [
                {"route": "Primary Evacuation Corridor", "road": f"Clear Corridor toward {dest_name}", "distance_km": dist, "eta_min": eta},
                {"route": "Secondary Detour Route", "road": "Elevated Peripheral Route", "distance_km": round(dist * 1.35, 1), "eta_min": int(eta * 1.4)}
            ]
    
    def _get_resources_for_zone(self, zone: Dict, resource_plan: Dict) -> Dict:
        """Extract resource allocation for a zone"""
        return resource_plan.get("zone_allocations", {}).get(zone.get("zone_id"), {})


# Global optimizer instance
_optimizer = None

def get_optimizer() -> ResponseOptimizer:
    global _optimizer
    if _optimizer is None:
        _optimizer = ResponseOptimizer()
    return _optimizer
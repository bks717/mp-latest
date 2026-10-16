"""
global_grid_system.py — Global Grid-Based Flood Prediction System

Replaces hardcoded basin mesh with a scalable geographical grid covering eligible global regions.
Automatically assesses grid cells without depending on fixed flood hotspots.
Aligns real environmental data with appropriate grid cells and river catchments.

Grid Design:
- Resolution: ~0.25° (~27km at equator) — matches GloFAS/ECMWF native resolution
- Coverage: Global land areas between 60°S and 60°N (excluding polar regions)
- Cell ID format: "grid_{lat_idx}_{lon_idx}" where lat_idx, lon_idx are integer grid indices
- Each cell stores: centroid coordinates, elevation, slope, land cover, catchment area
"""

import math
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import os

logger = logging.getLogger("GlobalGridSystem")
logging.basicConfig(level=logging.INFO)

# Grid Configuration
GRID_RESOLUTION_DEG = 0.25  # ~27km at equator
LAT_MIN = -60.0
LAT_MAX = 60.0
LON_MIN = -180.0
LON_MAX = 180.0

# Derived grid dimensions
N_LAT = int((LAT_MAX - LAT_MIN) / GRID_RESOLUTION_DEG)
N_LON = int((LON_MAX - LON_MIN) / GRID_RESOLUTION_DEG)

@dataclass
class GridCell:
    """Represents a single grid cell in the global prediction grid."""
    cell_id: str
    lat_idx: int
    lon_idx: int
    lat: float          # Centroid latitude
    lon: float          # Centroid longitude
    elevation_m: float  # Mean elevation (from Copernicus DEM)
    slope_pct: float    # Mean slope percentage
    land_cover_class: int  # ESA WorldCover class (0-10)
    impervious_pct: float  # Urban impervious surface percentage
    drainage_area_km2: float  # Upstream catchment area
    is_coastal: bool    # Within 50km of coastline
    is_active: bool     # Has sufficient data for prediction
    pop_density: float  # Population density (people/km²)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class GlobalGridSystem:
    """
    Manages the global prediction grid, cell metadata, and spatial operations.
    """
    
    def __init__(self, resolution_deg: float = GRID_RESOLUTION_DEG):
        self.resolution = resolution_deg
        self.cells: Dict[str, GridCell] = {}
        self.cell_array: List[GridCell] = []  # Flat list for iteration
        self._initialized = False
        
        # Pre-computed grid boundaries
        self.lat_edges = np.arange(LAT_MIN, LAT_MAX + resolution_deg, resolution_deg)
        self.lon_edges = np.arange(LON_MIN, LON_MAX + resolution_deg, resolution_deg)
        
    def initialize(self, force_rebuild: bool = False) -> None:
        """
        Initialize the global grid by creating all cells and populating with
        static geographical data (elevation, slope, land cover, etc.).
        
        In production, this would load from pre-computed tiles or a database.
        Here we compute on-the-fly using available APIs.
        """
        if self._initialized and not force_rebuild:
            return
            
        logger.info(f"Initializing global grid: {N_LAT}×{N_LON} = {N_LAT * N_LON} cells at {self.resolution}° resolution")
        
        # For now, create cells on-demand rather than pre-populating all 172,800 cells
        # We'll use lazy initialization: cells are created when first accessed
        self._initialized = True
        logger.info("Global grid system ready (lazy initialization mode)")
        
    def get_cell_id(self, lat: float, lon: float) -> str:
        """Get the cell ID for a given coordinate."""
        lat_idx = int((lat - LAT_MIN) / self.resolution)
        lon_idx = int((lon - LON_MIN) / self.resolution)
        lat_idx = max(0, min(lat_idx, N_LAT - 1))
        lon_idx = max(0, min(lon_idx, N_LON - 1))
        return f"grid_{lat_idx}_{lon_idx}"
    
    def get_cell_coords(self, cell_id: str) -> Tuple[float, float]:
        """Get centroid coordinates from cell ID."""
        parts = cell_id.split('_')
        if len(parts) != 3:
            return 0.0, 0.0
        lat_idx = int(parts[1])
        lon_idx = int(parts[2])
        lat = LAT_MIN + (lat_idx + 0.5) * self.resolution
        lon = LON_MIN + (lon_idx + 0.5) * self.resolution
        return lat, lon
    
    def get_cell(self, cell_id: str) -> Optional[GridCell]:
        """Get or create a grid cell."""
        if cell_id in self.cells:
            return self.cells[cell_id]
        
        # Parse cell ID
        parts = cell_id.split('_')
        if len(parts) != 3:
            return None
        lat_idx = int(parts[1])
        lon_idx = int(parts[2])
        
        # Compute centroid
        lat = LAT_MIN + (lat_idx + 0.5) * self.resolution
        lon = LON_MIN + (lon_idx + 0.5) * self.resolution
        
        # Create cell with default values (will be enriched later)
        cell = GridCell(
            cell_id=cell_id,
            lat_idx=lat_idx,
            lon_idx=lon_idx,
            lat=lat,
            lon=lon,
            elevation_m=0.0,
            slope_pct=1.0,
            land_cover_class=0,
            impervious_pct=0.0,
            drainage_area_km2=100.0,
            is_coastal=False,
            is_active=False,
            pop_density=50.0
        )
        
        self.cells[cell_id] = cell
        return cell
    
    def get_cells_in_bbox(self, min_lat: float, min_lon: float, 
                          max_lat: float, max_lon: float) -> List[GridCell]:
        """Get all cells within a bounding box."""
        min_lat_idx = max(0, int((min_lat - LAT_MIN) / self.resolution))
        max_lat_idx = min(N_LAT - 1, int((max_lat - LAT_MIN) / self.resolution))
        min_lon_idx = max(0, int((min_lon - LON_MIN) / self.resolution))
        max_lon_idx = min(N_LON - 1, int((max_lon - LON_MIN) / self.resolution))
        
        cells = []
        for lat_idx in range(min_lat_idx, max_lat_idx + 1):
            for lon_idx in range(min_lon_idx, max_lon_idx + 1):
                cell_id = f"grid_{lat_idx}_{lon_idx}"
                cell = self.get_cell(cell_id)
                if cell:
                    cells.append(cell)
        return cells
    
    def get_active_cells(self, limit: Optional[int] = None) -> List[GridCell]:
        """Get all cells marked as active (have sufficient data)."""
        active = [c for c in self.cells.values() if c.is_active]
        if limit:
            active = active[:limit]
        return active
    
    def enrich_cell_from_data(self, cell: GridCell, 
                               elevation_data: Dict = None,
                               glofas_data: Dict = None,
                               osm_data: Dict = None,
                               worldpop_data: Dict = None) -> None:
        """Enrich a grid cell with real environmental data."""
        if elevation_data:
            cell.elevation_m = elevation_data.get("elevation_m", cell.elevation_m)
            cell.slope_pct = elevation_data.get("slope_pct", cell.slope_pct)
            cell.is_coastal = cell.elevation_m < 10.0 and abs(cell.lat) < 60.0
        
        if glofas_data:
            # Estimate drainage area from GloFAS mean discharge
            mean_q = glofas_data.get("historical_mean_m3s", 100.0)
            # Rough empirical relationship: drainage_area ~ mean_q * 0.5 (very approximate)
            cell.drainage_area_km2 = max(10.0, mean_q * 0.5)
        
        if osm_data:
            cell.impervious_pct = osm_data.get("impervious_pct", cell.impervious_pct)
            cell.pop_density = osm_data.get("population_density_km2", cell.pop_density)
            # Simple land cover classification from OSM
            if cell.impervious_pct > 30:
                cell.land_cover_class = 10  # Urban
            elif osm_data.get("waterway_density_km_km2", 0) > 1.0:
                cell.land_cover_class = 2   # Wetland/riverine
            else:
                cell.land_cover_class = 3   # Vegetated/agricultural
        
        if worldpop_data:
            cell.pop_density = worldpop_data.get("population_density_km2", cell.pop_density)
        
        # Mark as active if we have minimum required data
        cell.is_active = (cell.elevation_m > 0 or cell.drainage_area_km2 > 10) and cell.pop_density > 0

    def enrich_cell_from_ingestion(
        self,
        cell: GridCell,
        ingestion: Any,
        gdacs_events: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Enrich a grid cell with real environmental telemetry from the Global Mesh
        data ingestion module (ECMWF weather, GloFAS discharge, GDACS alerts, DEM).

        Returns a report dict with data availability and freshness.  The cell is
        marked active only when at least one observation source succeeded.
        """
        # Lazy import to keep this module importable without the ingestion stack
        from global_mesh.data_ingestion import GlobalMeshDataIngestion

        if not isinstance(ingestion, GlobalMeshDataIngestion):
            logger.warning("enrich_cell_from_ingestion: not a GlobalMeshDataIngestion instance")
            cell.is_active = False
            return {"cell_id": cell.cell_id, "data_available": False, "error": "invalid ingestion"}

        obs = ingestion.observe_cell(
            cell_id=cell.cell_id, lat=cell.lat, lon=cell.lon, gdacs_events=gdacs_events
        )

        # Map observation -> GridCell fields
        if obs.obs_elevation_m is not None:
            cell.elevation_m = float(obs.obs_elevation_m)
        if obs.obs_slope_pct is not None:
            cell.slope_pct = float(obs.obs_slope_pct)
        cell.is_coastal = bool(obs.obs_is_coastal)
        if obs.obs_land_cover_class is not None:
            cell.land_cover_class = int(obs.obs_land_cover_class)
        cell.is_active = bool(obs.data_available)

        report = {
            "cell_id": cell.cell_id,
            "lat": cell.lat,
            "lon": cell.lon,
            "data_available": bool(obs.data_available),
            "is_active": cell.is_active,
            "elevation_m": cell.elevation_m,
            "slope_pct": cell.slope_pct,
            "is_coastal": cell.is_coastal,
            "land_cover_class": cell.land_cover_class,
            "missing_factors": obs.missing_factors,
            "source_timestamps": obs.source_timestamps,
            "freshness": obs.freshness,
            "retrieved_at_utc": obs.retrieved_at_utc,
        }
        logger.info(
            "Enriched cell %s (%.2f, %.2f): active=%s elevation=%.1f",
            cell.cell_id, cell.lat, cell.lon, cell.is_active, cell.elevation_m,
        )
        return report

    def get_cells_in_bbox_enriched(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        ingestion: Any,
        gdacs_events: Optional[List[Dict[str, Any]]] = None,
        max_cells: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return all grid cells in a bbox, each enriched with real telemetry.
        Respects max_cells to bound API usage for planetary-scale scans.
        """
        cells = self.get_cells_in_bbox(min_lat, min_lon, max_lat, max_lon)
        if max_cells is not None:
            cells = cells[:max_cells]
        reports: List[Dict[str, Any]] = []
        for cell in cells:
            try:
                report = self.enrich_cell_from_ingestion(cell, ingestion, gdacs_events)
                reports.append(report)
            except Exception as exc:
                logger.warning("Failed to enrich cell %s: %s", cell.cell_id, exc)
                reports.append({
                    "cell_id": cell.cell_id,
                    "lat": cell.lat,
                    "lon": cell.lon,
                    "data_available": False,
                    "is_active": False,
                    "error": str(exc),
                })
        return reports

    def get_neighbors(self, cell_id: str, radius_cells: int = 1) -> List[GridCell]:
        """Get neighboring cells within radius (for spatial context)."""
        parts = cell_id.split('_')
        if len(parts) != 3:
            return []
        lat_idx = int(parts[1])
        lon_idx = int(parts[2])
        
        neighbors = []
        for dlat in range(-radius_cells, radius_cells + 1):
            for dlon in range(-radius_cells, radius_cells + 1):
                if dlat == 0 and dlon == 0:
                    continue
                n_lat_idx = lat_idx + dlat
                n_lon_idx = lon_idx + dlon
                if 0 <= n_lat_idx < N_LAT and 0 <= n_lon_idx < N_LON:
                    n_id = f"grid_{n_lat_idx}_{n_lon_idx}"
                    neighbor = self.get_cell(n_id)
                    if neighbor:
                        neighbors.append(neighbor)
        return neighbors
    
    def cells_to_geojson(self, cells: List[GridCell], 
                        properties_fn=None) -> Dict[str, Any]:
        """Convert cells to GeoJSON FeatureCollection."""
        features = []
        for cell in cells:
            if not cell.is_active:
                continue
            # Create a small polygon for the cell (square)
            half_res = self.resolution / 2.0
            coords = [[
                [cell.lon - half_res, cell.lat - half_res],
                [cell.lon + half_res, cell.lat - half_res],
                [cell.lon + half_res, cell.lat + half_res],
                [cell.lon - half_res, cell.lat + half_res],
                [cell.lon - half_res, cell.lat - half_res]
            ]]
            
            props = {
                "cell_id": cell.cell_id,
                "lat": cell.lat,
                "lon": cell.lon,
                "elevation_m": cell.elevation_m,
                "slope_pct": cell.slope_pct,
                "drainage_area_km2": cell.drainage_area_km2,
                "pop_density": cell.pop_density,
                "is_coastal": cell.is_coastal
            }
            if properties_fn:
                props.update(properties_fn(cell))
            
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": coords},
                "properties": props
            })
        
        return {"type": "FeatureCollection", "features": features}


# Global singleton
_grid_system = None

def get_grid_system() -> GlobalGridSystem:
    global _grid_system
    if _grid_system is None:
        _grid_system = GlobalGridSystem()
        _grid_system.initialize()
    return _grid_system


# Utility functions for coordinate conversion
def lat_lon_to_grid_idx(lat: float, lon: float, resolution: float = GRID_RESOLUTION_DEG) -> Tuple[int, int]:
    """Convert lat/lon to grid indices."""
    lat_idx = int((lat - LAT_MIN) / resolution)
    lon_idx = int((lon - LON_MIN) / resolution)
    lat_idx = max(0, min(lat_idx, N_LAT - 1))
    lon_idx = max(0, min(lon_idx, N_LON - 1))
    return lat_idx, lon_idx


def grid_idx_to_lat_lon(lat_idx: int, lon_idx: int, resolution: float = GRID_RESOLUTION_DEG) -> Tuple[float, float]:
    """Convert grid indices to centroid lat/lon."""
    lat = LAT_MIN + (lat_idx + 0.5) * resolution
    lon = LON_MIN + (lon_idx + 0.5) * resolution
    return lat, lon


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
    return 2 * R * math.asin(math.sqrt(a))


if __name__ == "__main__":
    # Quick test
    grid = GlobalGridSystem()
    grid.initialize()
    
    # Test cell creation
    cell = grid.get_cell("grid_120_720")  # Around equator, prime meridian
    print(f"Cell: {cell.cell_id} at ({cell.lat:.3f}, {cell.lon:.3f})")
    
    # Test bbox query
    cells = grid.get_cells_in_bbox(10, 100, 11, 101)
    print(f"Cells in bbox: {len(cells)}")
    
    # Test neighbors
    neighbors = grid.get_neighbors("grid_120_720", radius_cells=1)
    print(f"Neighbors: {len(neighbors)}")
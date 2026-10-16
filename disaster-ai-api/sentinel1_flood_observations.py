"""
sentinel1_flood_observations.py — Sentinel-1 SAR Flood Observations for Training & Verification

Integrates Sentinel-1 SAR-derived flood extent from Copernicus Data Space / Microsoft Planetary Computer
for:
1. Training labels (historical flood inundation maps)
2. Independent verification of predictions
3. Near-real-time flood detection for data assimilation

Data Sources:
- Microsoft Planetary Computer STAC: Sentinel-1 RTC (Radiometrically Terrain Corrected)
- Copernicus Data Space Ecosystem: Sentinel-1 GRD + flood mapping processors
- NASA OPERA Sentinel-1 Flood Products (when available)

Processing:
- Change detection: Compare VV/VH backscatter to reference (dry) conditions
- Thresholding: Otsu's method or fixed dB thresholds for water detection
- Temporal compositing: Multi-temporal filtering to reduce noise
- Geometric correction: Project to grid system
"""

import asyncio
import logging
import math
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import requests

try:
    from pystac_client import Client
    import planetary_computer as pc
    import rioxarray
    import rasterio
    from rasterio.warp import reproject, Resampling
    STAC_AVAILABLE = True
except ImportError:
    STAC_AVAILABLE = False

logger = logging.getLogger("Sentinel1FloodObs")
logging.basicConfig(level=logging.INFO)

# Sentinel-1 Collections
SENTINEL1_RTC_COLLECTION = "sentinel-1-rtc"
SENTINEL1_GRD_COLLECTION = "sentinel-1-grd"

# Flood detection thresholds (dB change from reference)
VV_FLOOD_THRESHOLD_DB = -3.0  # VV drop of 3dB indicates flooding
VH_FLOOD_THRESHOLD_DB = -2.0  # VH drop of 2dB indicates flooding
RATIO_THRESHOLD = 1.5         # VV/VH ratio increase

# Reference period for "dry" conditions (days before event)
REFERENCE_WINDOW_DAYS = 30
MAX_REFERENCE_IMAGES = 5

@dataclass
class FloodObservation:
    """Represents a Sentinel-1 derived flood observation."""
    observation_id: str
    cell_id: str
    lat: float
    lon: float
    acquisition_time: datetime
    flood_fraction: float        # Fraction of cell flooded (0-1)
    flooded_area_km2: float      # Absolute flooded area
    confidence: float            # Detection confidence (0-1)
    method: str                  # "change_detection", "thresholding", "ml_classifier"
    vv_change_db: float          # Mean VV change from reference
    vh_change_db: float          # Mean VH change from reference
    source_scene_ids: List[str]  # Sentinel-1 scene IDs used
    processing_level: str        # "L2_RTC", "L1_GRD"
    quality_flags: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "cell_id": self.cell_id,
            "lat": self.lat,
            "lon": self.lon,
            "acquisition_time": self.acquisition_time.isoformat(),
            "flood_fraction": self.flood_fraction,
            "flooded_area_km2": self.flooded_area_km2,
            "confidence": self.confidence,
            "method": self.method,
            "vv_change_db": self.vv_change_db,
            "vh_change_db": self.vh_change_db,
            "source_scene_ids": self.source_scene_ids,
            "processing_level": self.processing_level,
            "quality_flags": self.quality_flags
        }


class Sentinel1FloodObserver:
    """
    Fetches and processes Sentinel-1 SAR data for flood detection.
    Uses Microsoft Planetary Computer STAC API for discovery and access.
    """
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "FloodWatch-AI/1.0 (Sentinel-1 Flood Observer)"
        })
        self._catalog = None
        
    def _get_catalog(self):
        """Get or create STAC catalog client."""
        if not STAC_AVAILABLE:
            return None
        if self._catalog is None:
            try:
                self._catalog = Client.open(
                    "https://planetarycomputer.microsoft.com/api/stac/v1",
                    modifier=pc.sign_inplace,
                    timeout=15.0
                )
            except Exception as e:
                logger.warning(f"Failed to initialize STAC catalog: {e}")
                self._catalog = None
        return self._catalog
    
    async def search_scenes(self, 
                           bbox: List[float], 
                           start_date: datetime, 
                           end_date: datetime,
                           max_items: int = 10) -> List[Dict[str, Any]]:
        """
        Search for Sentinel-1 RTC scenes covering a bbox and time range.
        Returns list of scene metadata with asset URLs.
        """
        catalog = self._get_catalog()
        if not catalog:
            return []
        
        try:
            # Format datetime range for STAC
            datetime_range = f"{start_date.isoformat()}/{end_date.isoformat()}"
            
            search = catalog.search(
                collections=[SENTINEL1_RTC_COLLECTION],
                bbox=bbox,
                datetime=datetime_range,
                query={
                    "sar:instrument_mode": {"eq": "IW"},  # Interferometric Wide swath
                    "sat:orbit_state": {"in": ["ascending", "descending"]}
                },
                sortby=[{"field": "datetime", "direction": "desc"}],
                limit=max_items
            )
            
            items = list(search.items())
            scenes = []
            for item in items:
                assets = item.assets
                vv_asset = assets.get("vv") or assets.get("VV")
                vh_asset = assets.get("vh") or assets.get("VH")
                
                if vv_asset and vh_asset:
                    scenes.append({
                        "scene_id": item.id,
                        "datetime": item.datetime,
                        "bbox": item.bbox,
                        "geometry": item.geometry,
                        "vv_href": vv_asset.href,
                        "vh_href": vh_asset.href,
                        "orbit_state": item.properties.get("sat:orbit_state"),
                        "instrument_mode": item.properties.get("sar:instrument_mode"),
                        "polarisation": item.properties.get("sar:polarisation"),
                        "resolution": item.properties.get("sar:resolution_range", 10)
                    })
            
            logger.info(f"Found {len(scenes)} Sentinel-1 RTC scenes for bbox {bbox}")
            return scenes
            
        except Exception as e:
            logger.error(f"STAC search failed: {e}")
            return []
    
    async def get_reference_scenes(self, 
                                  bbox: List[float], 
                                  before_date: datetime,
                                  max_scenes: int = MAX_REFERENCE_IMAGES) -> List[Dict[str, Any]]:
        """
        Get 'dry' reference scenes from a period before the event.
        Uses a 30-day window ending REFERENCE_WINDOW_DAYS before the event.
        """
        ref_end = before_date - timedelta(days=1)
        ref_start = ref_end - timedelta(days=REFERENCE_WINDOW_DAYS)
        return await self.search_scenes(bbox, ref_start, ref_end, max_scenes)
    
    async def get_event_scenes(self, 
                              bbox: List[float], 
                              event_date: datetime,
                              window_days: int = 3) -> List[Dict[str, Any]]:
        """Get scenes during/after a flood event."""
        start = event_date - timedelta(days=1)
        end = event_date + timedelta(days=window_days)
        return await self.search_scenes(bbox, start, end, max_items=10)
    
    def compute_flood_fraction(self, 
                              vv_data: np.ndarray, 
                              vh_data: np.ndarray,
                              ref_vv: np.ndarray, 
                              ref_vh: np.ndarray,
                              water_mask: np.ndarray = None) -> Tuple[float, float, float, float]:
        """
        Compute flood fraction from change detection.
        
        Returns: (flood_fraction, vv_change_db, vh_change_db, confidence)
        """
        # Convert to dB if not already (RTC is typically in linear power)
        epsilon = 1e-10
        vv_db = 10 * np.log10(np.clip(vv_data, epsilon, None))
        vh_db = 10 * np.log10(np.clip(vh_data, epsilon, None))
        ref_vv_db = 10 * np.log10(np.clip(ref_vv, epsilon, None))
        ref_vh_db = 10 * np.log10(np.clip(ref_vh, epsilon, None))
        
        # Change detection
        vv_change = vv_db - ref_vv_db
        vh_change = vh_db - ref_vh_db
        
        # Mean changes (negative = decrease = potential flooding)
        mean_vv_change = float(np.nanmean(vv_change))
        mean_vh_change = float(np.nanmean(vh_change))
        
        # Flood detection: significant decrease in backscatter
        # Water shows specular reflection -> low backscatter
        flood_vv = vv_change < VV_FLOOD_THRESHOLD_DB
        flood_vh = vh_change < VH_FLOOD_THRESHOLD_DB
        
        # Combined detection (both polarizations agree)
        flood_mask = flood_vv & flood_vh
        
        # Apply permanent water mask if available
        if water_mask is not None:
            flood_mask = flood_mask & (~water_mask.astype(bool))
        
        # Compute fraction
        valid_pixels = np.isfinite(vv_change) & np.isfinite(vh_change)
        if water_mask is not None:
            valid_pixels = valid_pixels & (~water_mask.astype(bool))
        
        if np.sum(valid_pixels) == 0:
            return 0.0, mean_vv_change, mean_vh_change, 0.0
        
        flood_fraction = float(np.sum(flood_mask & valid_pixels) / np.sum(valid_pixels))
        
        # Confidence based on:
        # 1. Magnitude of change
        # 2. Spatial coherence (connected components)
        # 3. Agreement between VV and VH
        vv_magnitude = min(1.0, abs(mean_vv_change) / 10.0)
        vh_magnitude = min(1.0, abs(mean_vh_change) / 8.0)
        polarization_agreement = float(np.sum(flood_vv & flood_vh & valid_pixels) / 
                                       max(1, np.sum(flood_vv & valid_pixels)))
        
        confidence = (vv_magnitude + vh_magnitude + polarization_agreement) / 3.0
        
        return flood_fraction, mean_vv_change, mean_vh_change, confidence
    
    async def process_cell_observation(self, 
                                      cell_id: str,
                                      lat: float, 
                                      lon: float,
                                      event_date: datetime,
                                      grid_resolution: float = 0.25) -> Optional[FloodObservation]:
        """
        Process a single grid cell for flood observation at a specific time.
        
        This is the main entry point for getting flood observations for training/verification.
        """
        # Define bbox for the cell (with small buffer)
        half_res = grid_resolution / 2.0
        buffer = 0.01  # ~1km buffer
        bbox = [
            lon - half_res - buffer,
            lat - half_res - buffer,
            lon + half_res + buffer,
            lat + half_res + buffer
        ]
        
        # Get event scenes
        event_scenes = await self.get_event_scenes(bbox, event_date)
        if not event_scenes:
            logger.warning(f"No Sentinel-1 scenes found for cell {cell_id} at {event_date}")
            return None
        
        # Get reference scenes
        ref_scenes = await self.get_reference_scenes(bbox, event_date)
        if not ref_scenes:
            logger.warning(f"No reference scenes found for cell {cell_id}")
            # Still process with single scene using threshold method
            return await self._process_single_scene_threshold(cell_id, lat, lon, event_scenes[0], bbox)
        
        # For now, use the most recent event scene and median reference
        # In production, would do multi-temporal compositing
        event_scene = event_scenes[0]
        
        # TODO: Download and process actual SAR data
        # This requires rioxarray/rasterio to fetch and clip the COGs
        # For now, return a placeholder with metadata
        
        return FloodObservation(
            observation_id=f"s1_{cell_id}_{event_date.strftime('%Y%m%d')}",
            cell_id=cell_id,
            lat=lat,
            lon=lon,
            acquisition_time=event_scene["datetime"],
            flood_fraction=0.0,  # Placeholder - would be computed from actual data
            flooded_area_km2=0.0,
            confidence=0.0,
            method="change_detection_pending",
            vv_change_db=0.0,
            vh_change_db=0.0,
            source_scene_ids=[event_scene["scene_id"]] + [s["scene_id"] for s in ref_scenes[:3]],
            processing_level="L2_RTC",
            quality_flags={
                "reference_scenes_count": len(ref_scenes),
                "event_scenes_count": len(event_scenes),
                "orbit_state": event_scene["orbit_state"],
                "processing_note": "Full SAR processing not implemented - placeholder"
            }
        )
    
    async def _process_single_scene_threshold(self, 
                                             cell_id: str,
                                             lat: float,
                                             lon: float,
                                             scene: Dict[str, Any],
                                             bbox: List[float]) -> FloodObservation:
        """
        Fallback: Single-scene threshold-based flood detection.
        Uses VV/VH ratio and absolute thresholds.
        """
        # This would download the scene and apply thresholds
        # Placeholder for now
        return FloodObservation(
            observation_id=f"s1_{cell_id}_{scene['datetime'].strftime('%Y%m%d')}_single",
            cell_id=cell_id,
            lat=lat,
            lon=lon,
            acquisition_time=scene["datetime"],
            flood_fraction=0.0,
            flooded_area_km2=0.0,
            confidence=0.0,
            method="single_scene_threshold_pending",
            vv_change_db=0.0,
            vh_change_db=0.0,
            source_scene_ids=[scene["scene_id"]],
            processing_level="L2_RTC",
            quality_flags={
                "processing_note": "Single-scene threshold method - requires SAR download"
            }
        )
    
    async def get_historical_flood_labels(self, 
                                         cell_ids: List[str],
                                         grid_system,
                                         start_date: datetime,
                                         end_date: datetime,
                                         min_confidence: float = 0.5) -> Dict[str, List[FloodObservation]]:
        """
        Get historical flood observations for a list of cells over a date range.
        Used for training label generation.
        
        Returns: Dict mapping cell_id -> list of FloodObservation
        """
        # In production, this would query a pre-computed flood database
        # or process Sentinel-1 time series for each cell
        # For now, return empty - would be populated by offline processing
        logger.info(f"Historical flood label query for {len(cell_ids)} cells from {start_date} to {end_date}")
        return {cell_id: [] for cell_id in cell_ids}
    
    def download_and_process_scene(self, 
                                   vv_href: str, 
                                   vh_href: str, 
                                   bbox: List[float],
                                   target_shape: Tuple[int, int] = (256, 256)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Download and clip Sentinel-1 RTC scene to bbox.
        Returns (vv_data, vh_data) arrays.
        """
        if not STAC_AVAILABLE:
            raise RuntimeError("rioxarray/rasterio not available for SAR processing")
        
        try:
            # Open VV
            vv_ds = rioxarray.open_rasterio(vv_href)
            vh_ds = rioxarray.open_rasterio(vh_href)
            
            # Clip to bbox
            from shapely.geometry import box
            import geopandas as gpd
            bbox_geom = box(*bbox)
            bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs="EPSG:4326")
            bbox_gdf_proj = bbox_gdf.to_crs(vv_ds.rio.crs)
            proj_bbox = tuple(bbox_gdf_proj.total_bounds)
            
            vv_clipped = vv_ds.rio.clip_box(*proj_bbox)
            vh_clipped = vh_ds.rio.clip_box(*proj_bbox)
            
            # Resample to target shape
            vv_resampled = vv_clipped.rio.reproject(
                vv_clipped.rio.crs,
                shape=target_shape,
                resampling=Resampling.bilinear
            )
            vh_resampled = vh_clipped.rio.reproject(
                vh_clipped.rio.crs,
                shape=target_shape,
                resampling=Resampling.bilinear
            )
            
            return vv_resampled.values.squeeze(), vh_resampled.values.squeeze()
            
        except Exception as e:
            logger.error(f"Failed to download/process SAR scene: {e}")
            raise


# Synchronous wrapper for non-async contexts
class SyncSentinel1Observer:
    """Synchronous wrapper for Sentinel1FloodObserver."""
    
    def __init__(self):
        self.async_observer = Sentinel1FloodObserver()
    
    def search_scenes(self, bbox, start_date, end_date, max_items=10):
        return asyncio.run(self.async_observer.search_scenes(bbox, start_date, end_date, max_items))
    
    def get_historical_flood_labels(self, cell_ids, grid_system, start_date, end_date):
        return asyncio.run(self.async_observer.get_historical_flood_labels(
            cell_ids, grid_system, start_date, end_date))


# Global singleton
_sentinel1_observer = None

def get_sentinel1_observer() -> Sentinel1FloodObserver:
    global _sentinel1_observer
    if _sentinel1_observer is None:
        _sentinel1_observer = Sentinel1FloodObserver()
    return _sentinel1_observer


def get_sync_sentinel1_observer() -> SyncSentinel1Observer:
    return SyncSentinel1Observer()


if __name__ == "__main__":
    # Quick test
    import asyncio
    
    async def test():
        observer = Sentinel1FloodObserver()
        
        # Test search for a known flood area (Bangladesh, 2022 floods)
        bbox = [88.0, 23.0, 91.0, 26.0]
        start = datetime(2022, 6, 1, tzinfo=timezone.utc)
        end = datetime(2022, 6, 30, tzinfo=timezone.utc)
        
        scenes = await observer.search_scenes(bbox, start, end, max_items=5)
        print(f"Found {len(scenes)} scenes")
        for s in scenes[:3]:
            print(f"  {s['scene_id']} - {s['datetime']} - {s['orbit_state']}")
    
    asyncio.run(test())
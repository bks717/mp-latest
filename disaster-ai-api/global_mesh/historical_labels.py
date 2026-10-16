"""
global_mesh/historical_labels.py — Historical Flood Label Integration

Integrates the Dartmouth Flood Observatory (DFO) Global Active Archive of Large
Flood Events as verified historical flood labels for training and verification.

Source (verified):
  * DFO Global Active Archive of Large Flood Events — 4029 real flood events
    globally, 1985–2010, with dates, centroids (lat/lon), severity, cause.
    Downloaded from: https://data.humdata.org/dataset/global-active-archive-of-large-flood-events-dfo
    License: CC BY (Creative Commons Attribution International)
    File: wlf_nhr_fl_dfomasterlist_20190418.zip (ESRI Shapefile)

Limitations (clearly identified):
  * The archive covers 1985–2010 only — it does not contain recent floods
    (2011–present).
  * Centroids are approximate event locations, not precise inundation footprints.
  * Events are large/major floods only — smaller/local floods are not included.
  * Labels derived from this archive are historical verified flood observations,
    but they are not a substitute for in-situ gauge truth.

This module:
  * Loads the DFO archive from the extracted sample directory.
  * Matches historical flood events to grid cells (within a radius).
  * Provides verified flood labels (flood=1 within event date window).
  * Generates negative labels (flood=0) from non-flood locations.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("GlobalMeshHistoricalLabels")
logging.basicConfig(level=logging.INFO)

# Path to the extracted DFO sample archive
DFO_SAMPLE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data_sources", "dfo_sample"
)
DFO_SHAPEFILE = os.path.join(DFO_SAMPLE_DIR, "wlf_nhr_fl_dfomasterlist_20190418.shp")

# Label window: a grid cell is "flooded" if a DFO event centroid is within
# EVENT_RADIUS_KM and the cell's observation date falls within [began, ended].
EVENT_RADIUS_KM = 50.0


@dataclass
class DFOEvent:
    """A verified historical flood event from the DFO archive."""
    register_id: float
    annual_dfo: Optional[float]
    glide: Optional[str]
    country: Optional[str]
    rivers: Optional[str]
    began: Optional[str]
    ended: Optional[str]
    duration_days: Optional[float]
    dead: Optional[float]
    displaced: Optional[float]
    main_cause: Optional[str]
    severity: Optional[float]
    centroid_x: Optional[float]  # longitude
    centroid_y: Optional[float]  # latitude
    geometry_wkt: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DFOHistoricalLabels:
    """
    Loads and queries the DFO historical flood archive.

    Provides verified flood labels for grid cells: a cell is labelled flood=1
    if a DFO event centroid falls within EVENT_RADIUS_KM and the observation
    date falls within the event's [began, ended] window.
    """

    def __init__(self) -> None:
        self.events: List[DFOEvent] = []
        self._loaded = False
        self._load_archive()

    def _load_archive(self) -> None:
        """Load the DFO shapefile archive."""
        if not os.path.exists(DFO_SHAPEFILE):
            logger.warning(
                "DFO shapefile not found at %s — historical labels unavailable. "
                "Run the DFO download/extraction step first.",
                DFO_SHAPEFILE,
            )
            self._loaded = False
            return

        try:
            import geopandas as gpd

            gdf = gpd.read_file(DFO_SHAPEFILE)
            logger.info("Loaded DFO archive: %d events", len(gdf))

            for _, row in gdf.iterrows():
                try:
                    centroid_x = row.get("Centroid_X")
                    centroid_y = row.get("Centroid_Y")
                    if centroid_x is None or centroid_y is None:
                        continue
                    # Parse dates
                    began = row.get("Began")
                    ended = row.get("Ended")
                    began_str = began.isoformat() if hasattr(began, "isoformat") else str(began) if began else None
                    ended_str = ended.isoformat() if hasattr(ended, "isoformat") else str(ended) if ended else None

                    event = DFOEvent(
                        register_id=float(row.get("Register__", 0)) if row.get("Register__") is not None else 0.0,
                        annual_dfo=float(row.get("Annual_DFO")) if row.get("Annual_DFO") is not None else None,
                        glide=str(row.get("Glide__")) if row.get("Glide__") else None,
                        country=str(row.get("Country__c")) if row.get("Country__c") else None,
                        rivers=str(row.get("Rivers")) if row.get("Rivers") else None,
                        began=began_str,
                        ended=ended_str,
                        duration_days=float(row.get("Duration_i")) if row.get("Duration_i") else None,
                        dead=float(row.get("Dead")) if row.get("Dead") else None,
                        displaced=float(row.get("Displaced")) if row.get("Displaced") else None,
                        main_cause=str(row.get("Main_cause")) if row.get("Main_cause") else None,
                        severity=float(row.get("Severity__")) if row.get("Severity__") else None,
                        centroid_x=float(centroid_x),
                        centroid_y=float(centroid_y),
                        geometry_wkt=str(row.get("geometry")) if row.get("geometry") else None,
                    )
                    self.events.append(event)
                except Exception as exc:
                    logger.debug("Skipping DFO event row: %s", exc)
                    continue

            self._loaded = True
            logger.info(
                "DFO labels ready: %d verified flood events (%s)",
                len(self.events),
                self._date_range(),
            )
        except ImportError:
            logger.warning("geopandas not available — cannot load DFO shapefile")
            self._loaded = False
        except Exception as exc:
            logger.warning("Failed to load DFO archive: %s", exc)
            self._loaded = False

    def _date_range(self) -> str:
        if not self.events:
            return "no events"
        dates = [
            e.began for e in self.events if e.began
        ]
        if not dates:
            return "no dates"
        return f"{min(dates)} to {max(dates)}"

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        )
        return 2 * R * math.asin(math.sqrt(a))

    def is_loaded(self) -> bool:
        return self._loaded

    def event_count(self) -> int:
        return len(self.events)

    def label_cell(
        self,
        lat: float,
        lon: float,
        observation_date: Optional[str] = None,
        radius_km: float = EVENT_RADIUS_KM,
    ) -> Tuple[int, Optional[str], Optional[Dict[str, Any]]]:
        """
        Label a grid cell based on DFO historical flood events.

        Returns: (label, matched_event_id, event_details)
          label: 1 if a verified flood event is within radius_km and the
                 observation date falls within the event window; 0 otherwise.
          matched_event_id: the DFO register ID if matched, else None.
          event_details: dict with event info if matched, else None.
        """
        if not self._loaded:
            return 0, None, None

        obs_dt = None
        if observation_date:
            try:
                obs_dt = datetime.fromisoformat(observation_date.replace("Z", "+00:00"))
            except Exception:
                obs_dt = None

        for ev in self.events:
            dist = self._haversine_km(lat, lon, ev.centroid_y, ev.centroid_x)
            if dist > radius_km:
                continue

            # Check date window if we have an observation date
            if obs_dt is not None and ev.began and ev.ended:
                try:
                    began_dt = datetime.fromisoformat(ev.began.replace("Z", "+00:00"))
                    ended_dt = datetime.fromisoformat(ev.ended.replace("Z", "+00:00"))
                    if obs_dt < began_dt or obs_dt > ended_dt:
                        continue  # event not active at observation time
                except Exception:
                    pass  # if we can't parse dates, accept spatial match

            return (
                1,
                str(ev.register_id),
                {
                    "register_id": ev.register_id,
                    "country": ev.country,
                    "rivers": ev.rivers,
                    "began": ev.began,
                    "ended": ev.ended,
                    "main_cause": ev.main_cause,
                    "severity": ev.severity,
                    "distance_km": round(dist, 1),
                    "centroid_lat": ev.centroid_y,
                    "centroid_lon": ev.centroid_x,
                },
            )

        return 0, None, None

    def get_verified_flood_locations(
        self,
        min_severity: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return verified flood event locations (centroids) for training label generation.
        """
        results = []
        for ev in self.events:
            if min_severity is not None and (ev.severity is None or ev.severity < min_severity):
                continue
            results.append(
                {
                    "lat": ev.centroid_y,
                    "lon": ev.centroid_x,
                    "register_id": ev.register_id,
                    "country": ev.country,
                    "began": ev.began,
                    "ended": ev.ended,
                    "main_cause": ev.main_cause,
                    "severity": ev.severity,
                    "source": "DFO Global Active Archive (verified historical flood)",
                }
            )
            if limit and len(results) >= limit:
                break
        return results


# Global singleton
_labels: Optional[DFOHistoricalLabels] = None


def get_historical_labels() -> DFOHistoricalLabels:
    global _labels
    if _labels is None:
        _labels = DFOHistoricalLabels()
    return _labels

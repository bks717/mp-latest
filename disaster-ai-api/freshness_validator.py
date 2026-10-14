"""
Data Freshness Validator for Flood Disaster Management System
Validates observation/forecast timestamps against configurable freshness thresholds.
Enforces 24-48 hour freshness for dynamic environmental data.
"""

import math
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("FreshnessValidator")
logging.basicConfig(level=logging.INFO)


class FreshnessStatus(Enum):
    FRESH = "FRESH"           # Within optimal threshold (24h for most dynamic data)
    ACCEPTABLE = "ACCEPTABLE"  # Within extended threshold (48h)
    STALE = "STALE"           # Beyond extended but within maximum (7 days)
    EXPIRED = "EXPIRED"       # Beyond maximum acceptable age
    UNKNOWN = "UNKNOWN"       # No timestamp available


@dataclass
class FreshnessConfig:
    """Freshness thresholds by data type"""
    optimal_hours: float      # Green zone
    extended_hours: float     # Yellow zone  
    maximum_hours: float      # Red zone (hard rejection)
    description: str


# Default freshness policies by data source/type
DEFAULT_POLICIES = {
    # Meteorological observations (ECMWF IFS)
    "ecmwf_observation": FreshnessConfig(24, 48, 168, "ECMWF observed weather"),
    "ecmwf_forecast": FreshnessConfig(6, 12, 72, "ECMWF forecast (issue time)"),
    
    # Hydrological (GloFAS)
    "glofas_discharge": FreshnessConfig(24, 48, 168, "GloFAS river discharge"),
    "glofas_forecast": FreshnessConfig(6, 24, 72, "GloFAS discharge forecast"),
    
    # Satellite SAR
    "sentinel1_sar": FreshnessConfig(24, 48, 168, "Sentinel-1 SAR acquisition"),
    
    # Satellite precipitation
    "nasa_gpm": FreshnessConfig(48, 72, 168, "NASA GPM satellite precipitation"),
    
    # River gauges
    "usgs_gauge": FreshnessConfig(6, 24, 72, "USGS real-time gauge"),
    "cwc_gauge": FreshnessConfig(6, 24, 72, "CWC India gauge"),
    
    # Soil moisture (modelled)
    "soil_moisture": FreshnessConfig(24, 48, 168, "ECMWF soil moisture"),
    
    # Static reference data
    "copernicus_dem": FreshnessConfig(8760, 8760, 8760, "Copernicus DEM (static)"),
    "soilgrids": FreshnessConfig(8760, 8760, 8760, "ISRIC SoilGrids (static)"),
    "worldpop": FreshnessConfig(8760, 8760, 8760, "WorldPop demographics (annual)"),
    "osm_infrastructure": FreshnessConfig(168, 720, 2160, "OSM infrastructure (weeks)"),
    
    # Disaster alerts
    "gdacs_alert": FreshnessConfig(24, 72, 168, "GDACS flood alert"),
}


class FreshnessValidator:
    """
    Validates data freshness across all environmental data sources.
    Provides detailed provenance including timestamps, age, and status.
    """
    
    def __init__(self, policies: Dict[str, FreshnessConfig] = None):
        self.policies = policies or DEFAULT_POLICIES.copy()
    
    def validate(self, 
                 data_type: str, 
                 observation_time: Optional[str] = None,
                 forecast_issue_time: Optional[str] = None,
                 forecast_valid_time: Optional[str] = None,
                 retrieval_time: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate freshness of a data record.
        
        Args:
            data_type: Key into policies dict (e.g., "ecmwf_observation")
            observation_time: When the observation was made (ISO format)
            forecast_issue_time: When forecast was issued (ISO format)
            forecast_valid_time: When forecast is valid for (ISO format)
            retrieval_time: When data was retrieved (ISO format, defaults to now)
        
        Returns:
            Dict with freshness status, age, and metadata
        """
        policy = self.policies.get(data_type)
        if not policy:
            logger.warning(f"No freshness policy for data_type: {data_type}")
            policy = FreshnessConfig(24, 48, 168, "Unknown data type")
        
        now = datetime.now(timezone.utc)
        if retrieval_time:
            try:
                now = datetime.fromisoformat(retrieval_time.replace('Z', '+00:00'))
            except Exception:
                pass
        
        result = {
            "data_type": data_type,
            "policy": policy.description,
            "thresholds": {
                "optimal_hours": policy.optimal_hours,
                "extended_hours": policy.extended_hours,
                "maximum_hours": policy.maximum_hours
            },
            "timestamps": {
                "observation_time": observation_time,
                "forecast_issue_time": forecast_issue_time,
                "forecast_valid_time": forecast_valid_time,
                "retrieval_time": now.isoformat()
            },
            "age_hours": None,
            "status": FreshnessStatus.UNKNOWN.value,
            "is_valid_for_operations": False,
            "warnings": []
        }
        
        # Determine which timestamp to use for age calculation
        ref_time = None
        ref_label = ""
        
        def _parse_to_utc(dt_str: str):
            """Parse ISO format string to timezone-aware UTC datetime."""
            try:
                dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                # If naive, assume UTC
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception as e:
                result["warnings"].append(f"Invalid timestamp format: {e}")
                return None
        
        if observation_time:
            ref_time = _parse_to_utc(observation_time)
            ref_label = "observation"
        elif forecast_issue_time:
            ref_time = _parse_to_utc(forecast_issue_time)
            ref_label = "forecast_issue"
        elif forecast_valid_time:
            ref_time = _parse_to_utc(forecast_valid_time)
            ref_label = "forecast_valid"
        
        if ref_time is None:
            result["status"] = FreshnessStatus.UNKNOWN.value
            result["warnings"].append("No valid timestamp provided for freshness validation")
            return result
        
        # Calculate age
        age_delta = now - ref_time
        age_hours = age_delta.total_seconds() / 3600
        result["age_hours"] = round(age_hours, 1)
        result["reference_timestamp_type"] = ref_label
        
        # Determine status
        if age_hours < 0:
            # Future timestamp - possible forecast
            if forecast_valid_time and ref_label == "forecast_issue":
                # This is a forecast; check validity period
                try:
                    valid_time = datetime.fromisoformat(forecast_valid_time.replace('Z', '+00:00'))
                    lead_hours = (valid_time - ref_time).total_seconds() / 3600
                    result["forecast_lead_hours"] = round(lead_hours, 1)
                    result["status"] = FreshnessStatus.FRESH.value
                    result["is_valid_for_operations"] = True
                    result["note"] = f"Forecast issued {abs(age_hours):.1f}h ago, valid for +{lead_hours:.1f}h"
                except Exception:
                    result["status"] = FreshnessStatus.FRESH.value
                    result["is_valid_for_operations"] = True
            else:
                result["warnings"].append(f"Future timestamp detected ({age_hours:.1f}h in future)")
                result["status"] = FreshnessStatus.FRESH.value
                result["is_valid_for_operations"] = True
        elif age_hours <= policy.optimal_hours:
            result["status"] = FreshnessStatus.FRESH.value
            result["is_valid_for_operations"] = True
        elif age_hours <= policy.extended_hours:
            result["status"] = FreshnessStatus.ACCEPTABLE.value
            result["is_valid_for_operations"] = True
            result["warnings"].append(f"Data age ({age_hours:.1f}h) exceeds optimal threshold ({policy.optimal_hours}h) but within extended ({policy.extended_hours}h)")
        elif age_hours <= policy.maximum_hours:
            result["status"] = FreshnessStatus.STALE.value
            result["is_valid_for_operations"] = False
            result["warnings"].append(f"Data age ({age_hours:.1f}h) exceeds extended threshold ({policy.extended_hours}h) — STALE")
        else:
            result["status"] = FreshnessStatus.EXPIRED.value
            result["is_valid_for_operations"] = False
            result["warnings"].append(f"Data age ({age_hours:.1f}h) exceeds maximum threshold ({policy.maximum_hours}h) — REJECTED")
        
        return result
    
    def validate_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate freshness for multiple records."""
        return [self.validate(**r) for r in records]
    
    def get_summary(self, validations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics from validation results."""
        if not validations:
            return {"total": 0, "fresh": 0, "acceptable": 0, "stale": 0, "expired": 0, "unknown": 0}
        
        counts = {"FRESH": 0, "ACCEPTABLE": 0, "STALE": 0, "EXPIRED": 0, "UNKNOWN": 0}
        for v in validations:
            counts[v.get("status", "UNKNOWN")] += 1
        
        return {
            "total": len(validations),
            "fresh": counts["FRESH"],
            "acceptable": counts["ACCEPTABLE"],
            "stale": counts["STALE"],
            "expired": counts["EXPIRED"],
            "unknown": counts["UNKNOWN"],
            "freshness_rate": round((counts["FRESH"] + counts["ACCEPTABLE"]) / len(validations) * 100, 1),
            "rejection_rate": round((counts["STALE"] + counts["EXPIRED"]) / len(validations) * 100, 1)
        }


# Global validator instance
_validator = None

def get_freshness_validator() -> FreshnessValidator:
    global _validator
    if _validator is None:
        _validator = FreshnessValidator()
    return _validator


def validate_data_freshness(data_type: str, **timestamps) -> Dict[str, Any]:
    """Convenience function for single validation."""
    return get_freshness_validator().validate(data_type, **timestamps)


def is_data_fresh(data_type: str, **timestamps) -> bool:
    """Quick boolean check for operational validity."""
    result = validate_data_freshness(data_type, **timestamps)
    return result.get("is_valid_for_operations", False)
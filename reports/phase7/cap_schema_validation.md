# Phase 7 — CAP 1.2 Schema Validation

## CAP 1.2 Export Validation

The alert service exports approved alerts as CAP 1.2 XML via `POST /api/alert/cap/export`.

## CAP 1.2 Required Elements Present

| Element | Present | Notes |
|---------|---------|-------|
| `alert` root element | Yes | With xmlns="urn:oasis:names:tc:emergency:cap:1.2" |
| `identifier` | Yes | URN format: URN:OASIS:EMERGENCY:CAP:GLOBAL_MESH:YYYYMMDDTHH |
| `sender` | Yes | global-mesh@floodwatch.ai |
| `sent` | Yes | ISO 8601 timestamp |
| `status` | Yes | "Actual" |
| `msgType` | Yes | "Alert" |
| `scope` | Yes | "Public" |
| `references` | Yes | Empty string |
| `info` | Yes | One info block per alert |
| `info/language` | Yes | "en" |
| `info/category` | Yes | "Met" |
| `info/event` | Yes | "Flood Alert" |
| `info/responseType` | Yes | Evacuate/Prepare/Monitor |
| `info/priority` | Yes | Immediate/Expected/Minor |
| `info/urgency` | Yes | Immediate/Expected/Minor |
| `info/severity` | Yes | From candidate severity field |
| `info/certainty` | Yes | "Observed" if trusted, "Likely" otherwise |
| `info/headline` | Yes | "Flood Alert — {region}" |
| `info/description` | Yes | Risk probability and driver |
| `info/instruction` | Yes | Evacuation instructions |
| `info/area` | Yes | One area per alert |
| `info/area/areaDesc` | Yes | Country + basin |
| `info/area/polygon` | Yes | Lat/Lng coordinates |
| `info/area/altitude` | Yes | "0" |
| `info/area/ceiling` | Yes | "0" |

## Validation Method

CAP 1.2 XML is generated programmatically and structurally validated against the CAP 1.2 schema requirements:
- All required elements present
- Correct namespace declaration
- Proper XML structure
- All mandatory info/area sub-elements present

## NOT DONE / UNVERIFIED

No automated XSD schema validation performed. CAP XML structure verified manually against CAP 1.2 specification requirements.
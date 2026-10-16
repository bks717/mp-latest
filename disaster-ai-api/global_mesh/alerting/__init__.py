"""
global_mesh/alerting/__init__.py — Alerting package for Global Mesh.

Exports the alert candidate builder and policy engine used by the forecast
service and the dashboard alert service.
"""

from global_mesh.alerting.alert_candidates import build_alert_candidates
from global_mesh.alerting.policy_engine import AlertPolicyEngine, AlertDecision

__all__ = [
    "build_alert_candidates",
    "AlertPolicyEngine",
    "AlertDecision",
]

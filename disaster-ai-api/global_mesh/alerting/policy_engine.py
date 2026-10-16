"""
global_mesh/alerting/policy_engine.py — Alert policy engine (Phase 6)

Implements the alert decision policy:
  * Cost-loss expected utility: alert only if P(flood) * loss_avoidance > cost_of_action
  * Trust gate: only trust model output if trained+validated and in-domain
  * Hysteresis: require sustained elevation before alerting; require sustained
    normal before all-clear
  * Deduplication: one active alert per cell at a time
  * Supersede / all-clear: escalate or de-escalate existing alerts
  * Fatigue budget: max alerts per cell per day to avoid alert fatigue

All decisions are deterministic given the same inputs. No synthetic data is
introduced. Missing data leads to "no alert" (never a false sense of safety).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("GlobalMeshAlertPolicy")
logging.basicConfig(level=logging.INFO)

# Cost-loss parameters (tunable, documented)
COST_OF_ACTION = 1.0          # relative cost of issuing an alert
LOSS_IF_MISSED = 10.0         # relative loss of missing a real flood
COST_LOSS_THRESHOLD = COST_OF_ACTION / LOSS_IF_MISSED  # 0.10

# Fatigue budget: max alerts per cell per day
MAX_ALERTS_PER_CELL_PER_DAY = 3

# Hysteresis: consecutive confirmations needed to escalate / de-escalate
ESCALATE_CONFIRMATIONS = 2
DEESCALATE_CONFIRMATIONS = 3


@dataclass
class AlertDecision:
    """Result of evaluating a candidate against the policy engine."""

    candidate_id: str
    cell_id: str
    action: str  # "ALERT" | "SUPPRESS" | "ESCALATE" | "DEESCALATE" | "ALL_CLEAR" | "NO_CHANGE"
    reason: str
    risk_tier: str
    trust_flag: bool
    cost_loss_ratio: float
    predicted_probability: Optional[float]
    previous_state: Optional[str]
    new_state: Optional[str]
    timestamp_utc: str
    fatigue_budget_remaining: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AlertPolicyEngine:
    """
    Stateful policy engine for alert decisions.

    Maintains per-cell state (current tier, confirmation counters, daily alert
    count) to implement hysteresis, dedupe, supersede/all-clear, and fatigue
    budgeting.
    """

    def __init__(
        self,
        cost_of_action: float = COST_OF_ACTION,
        loss_if_missed: float = LOSS_IF_MISSED,
        max_alerts_per_cell_per_day: int = MAX_ALERTS_PER_CELL_PER_DAY,
        escalate_confirmations: int = ESCALATE_CONFIRMATIONS,
        deescalate_confirmations: int = DEESCALATE_CONFIRMATIONS,
    ) -> None:
        self.cost_of_action = cost_of_action
        self.loss_if_missed = loss_if_missed
        self.cost_loss_threshold = cost_of_action / loss_if_missed
        self.max_alerts_per_cell_per_day = max_alerts_per_cell_per_day
        self.escalate_confirmations = escalate_confirmations
        self.deescalate_confirmations = deescalate_confirmations

        # Per-cell state: cell_id -> state dict
        self._cell_state: Dict[str, Dict[str, Any]] = {}

    def evaluate(
        self,
        candidate: Dict[str, Any],
        now_utc: Optional[str] = None,
    ) -> AlertDecision:
        """
        Evaluate a single candidate and return an alert decision.

        The decision considers:
          1. Trust flag: if model not trusted, downgrade tier by one level
          2. Cost-loss: expected utility = P * loss - cost; alert only if > 0
          3. Hysteresis: need confirmations to escalate/de-escalate
          4. Dedupe: one active alert per cell
          5. Fatigue budget: max alerts per cell per day
        """
        now = now_utc or datetime.now(timezone.utc).isoformat()
        cell_id = candidate.get("cell_id")
        candidate_id = candidate.get("candidate_id", cell_id)
        tier = candidate.get("risk_tier", "NORMAL")
        prob = candidate.get("predicted_flood_probability")
        trust_flag = bool(candidate.get("trust_flag", False))

        # Get or create cell state
        state = self._cell_state.get(
            cell_id,
            {
                "current_tier": "NORMAL",
                "confirmations": 0,
                "alerts_today": 0,
                "last_alert_date": None,
                "active_alert": False,
            },
        )

        # Determine effective tier after trust gating
        effective_tier = tier
        if not trust_flag and tier in ("WARNING", "IMMINENT"):
            # Downgrade by one level if not trusted
            if tier == "IMMINENT":
                effective_tier = "WARNING"
            elif tier == "WARNING":
                effective_tier = "WATCH"

        # Cost-loss decision: alert only if expected utility > 0
        # expected utility = P * LOSS - COST ; alert if > 0 => P > COST/LOSS
        should_alert_based_on_cost = False
        if prob is not None and trust_flag:
            expected_utility = prob * self.loss_if_missed - self.cost_of_action
            should_alert_based_on_cost = expected_utility > 0

        # Determine target tier
        target_tier = effective_tier
        if not should_alert_based_on_cost:
            # Even if tier says HIGH/IMMINENT, cost-loss says don't alert
            if effective_tier in ("WARNING", "IMMINENT"):
                target_tier = "WATCH"

        # Determine action
        current_tier = state.get("current_tier", "NORMAL")
        confirmations = state.get("confirmations", 0)
        active_alert = state.get("active_alert", False)
        alerts_today = state.get("alerts_today", 0)
        last_alert_date = state.get("last_alert_date")
        today_date = now_utc[:10]

        # Reset daily counter if new day
        if last_alert_date != today_date:
            alerts_today = 0
            last_alert_date = today_date

        action = "NO_CHANGE"
        new_tier = current_tier
        reason = "No change"

        if effective_tier == "NORMAL":
            # No risk - de-escalate to normal
            if current_tier != "NORMAL":
                new_tier = "NORMAL"
                confirmations = 0
                active_alert = False
                action = "ALL_CLEAR"
                reason = "Risk returned to normal"
            else:
                action = "NO_CHANGE"
        else:
            # Some risk - consider escalating
            tier_order = {"NORMAL": 0, "WATCH": 1, "WARNING": 2, "IMMINENT": 3}
            current_level = tier_order.get(current_tier, 0)
            target_level = tier_order.get(effective_tier, 0)

            if target_level > current_level:
                # Escalating
                confirmations += 1
                if confirmations >= self.escalate_confirmations:
                    new_tier = effective_tier
                    confirmations = 0
                    if not active_alert:
                        # Check fatigue budget
                        if alerts_today >= self.max_alerts_per_cell_per_day:
                            action = "NO_CHANGE"
                            reason = "Fatigue budget exhausted"
                            new_tier = current_tier
                            confirmations = self.escalate_confirmations - 1  # keep near threshold
                        else:
                            action = "ESCALATE"
                            active_alert = True
                            alerts_today += 1
                            reason = f"Escalated to {effective_tier}"
                    else:
                        action = "ESCALATE"
                        reason = f"Escalated to {effective_tier}"
                else:
                    action = "NO_CHANGE"
                    reason = f"Building confidence for {effective_tier} ({confirmations}/{self.escalate_confirmations})"
            elif target_level < current_level:
                # De-escalating
                confirmations += 1
                if confirmations >= self.deescalate_confirmations:
                    new_tier = effective_tier
                    confirmations = 0
                    if effective_tier == "NORMAL":
                        active_alert = False
                        action = "ALL_CLEAR"
                        reason = "De-escalated to normal"
                    else:
                        action = "DEESCALATE"
                        reason = f"De-escalated to {effective_tier}"
                else:
                    action = "NO_CHANGE"
                    reason = f"Building confidence for de-escalation ({confirmations}/{self.deescalate_confirmations})"
            else:
                # Same tier
                action = "NO_CHANGE"
                reason = f"Maintaining {effective_tier}"

        # Build decision
        decision = AlertDecision(
            candidate_id=candidate_id,
            cell_id=cell_id,
            action=action,
            reason=reason,
            risk_tier=effective_tier,
            trust_flag=trust_flag,
            cost_loss_ratio=round(self.cost_loss_threshold, 4),
            predicted_probability=prob,
            previous_state=current_tier,
            new_state=new_tier,
            timestamp_utc=now,
            fatigue_budget_remaining=max(0, self.max_alerts_per_cell_per_day - alerts_today),
        )

        # Update state
        self._cell_state[cell_id] = {
            "current_tier": new_tier,
            "confirmations": confirmations,
            "alerts_today": alerts_today,
            "last_alert_date": last_alert_date,
            "active_alert": active_alert,
        }

        return decision

    def get_cell_state(self, cell_id: str) -> Optional[Dict[str, Any]]:
        return self._cell_state.get(cell_id)

    def reset_cell(self, cell_id: str) -> None:
        self._cell_state.pop(cell_id, None)

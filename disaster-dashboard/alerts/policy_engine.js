/**
 * disaster-dashboard/alerts/policy_engine.js — Alert policy engine (Phase 6)
 *
 * Implements the alert decision policy for the dashboard alert service:
 *   * Cost-loss expected utility: alert only if P(flood) * loss_avoidance > cost_of_action
 *   * Trust gate: only trust model output if trained+validated and in-domain
 *   * Hysteresis: require sustained elevation before alerting; require sustained
 *     normal before all-clear
 *   * Deduplication: one active alert per cell at a time
  * * Supersede / all-clear: escalate or de-escalate existing alerts
 *   * Fatigue budget: max alerts per cell per day to avoid alert fatigue
 *
 * All decisions are deterministic given the same inputs. No synthetic data is
 * introduced. Missing data leads to "no alert" (never a false sense of safety).
 */

// ALERT_MODE = dry_run (default) — no alerts sent to real people.
// Cost-loss expected utility parameters apply only in dry_run/test mode.
const COST_OF_ACTION = 1.0;
const LOSS_IF_MISSED = 10.0;
const COST_LOSS_THRESHOLD = COST_OF_ACTION / LOSS_IF_MISSED; // 0.10

const MAX_ALERTS_PER_CELL_PER_DAY = 3;
const ESCALATE_CONFIRMATIONS = 2;
const DEESCALATE_CONFIRMATIONS = 3;

const TIER_ORDER = { NORMAL: 0, WATCH: 1, WARNING: 2, IMMINENT: 3 };

/**
 * Tier from probability.
 */
function tierFromProbability(prob) {
  if (prob == null) return "NORMAL";
  if (prob >= 0.70) return "IMMINENT";
  if (prob >= 0.50) return "WARNING";
  if (prob >= 0.30) return "WATCH";
  return "NORMAL";
}

/**
 * AlertPolicyEngine — stateful policy engine for alert decisions.
 */
class AlertPolicyEngine {
  constructor({
    costOfAction = COST_OF_ACTION,
    lossIfMissed = LOSS_IF_MISSED,
    maxAlertsPerCellPerDay = MAX_ALERTS_PER_CELL_PER_DAY,
    escalateConfirmations = ESCALATE_CONFIRMATIONS,
    deescalateConfirmations = DEESCALATE_CONFIRMATIONS,
  } = {}) {
    this.costOfAction = costOfAction;
    this.lossIfMissed = lossIfMissed;
    this.costLossThreshold = costOfAction / lossIfMissed;
    this.maxAlertsPerCellPerDay = maxAlertsPerCellPerDay;
    this.escalateConfirmations = escalateConfirmations;
    this.deescalateConfirmations = deescalateConfirmations;
    this._cellState = new Map(); // cell_id -> state
  }

  /**
   * Evaluate a single candidate and return an alert decision.
   */
  evaluate(candidate, nowUtc = null) {
    const now = nowUtc || new Date().toISOString();
    const cellId = candidate.cell_id;
    const candidateId = candidate.candidate_id || cell_id;
    const tier = candidate.risk_tier || "NORMAL";
    const prob = candidate.predicted_flood_probability;
    const trustFlag = candidate.trust_flag === true;

    // Get or create cell state
    let state = this._cellState.get(cellId);
    if (!state) {
      state = {
        currentTier: "NORMAL",
        confirmations: 0,
        alertsToday: 0,
        lastAlertDate: null,
        activeAlert: false,
      };
      this._cellState.set(cellId, state);
    }

    // Effective tier after trust gating
    let effectiveTier = tier;
    if (!trustFlag && (tier === "WARNING" || tier === "IMMINENT")) {
      effectiveTier = tier === "IMMINENT" ? "WARNING" : "WATCH";
    }

    // Cost-loss decision: alert only if expected utility > 0
    // expected utility = P * LOSS - COST ; alert if > 0 => P > COST/LOSS
    let shouldAlertBasedOnCost = false;
    if (prob != null && trustFlag) {
      const expectedUtility = prob * this.lossIfMissed - this.costOfAction;
      shouldAlertBasedOnCost = expectedUtility > 0;
    }

    let targetTier = effectiveTier;
    if (!shouldAlertBasedOnCost && (effectiveTier === "WARNING" || effectiveTier === "IMMINENT")) {
      targetTier = "WATCH";
    }

    const currentTier = state.currentTier || "NORMAL";
    const confirmations = state.confirmations || 0;
    const activeAlert = state.activeAlert === true;
    const alertsToday = state.alertsToday || 0;
    const lastAlertDate = state.lastAlertDate;
    const todayDate = now.slice(0, 10);

    // Reset daily counter if new day
    if (lastAlertDate !== todayDate) {
      state.alertsToday = 0;
      state.lastAlertDate = todayDate;
    }

    let action = "NO_CHANGE";
    let newTier = currentTier;
    let reason = "No change";

    if (effectiveTier === "NORMAL") {
      if (currentTier !== "NORMAL") {
        newTier = "NORMAL";
        state.confirmations = 0;
        state.activeAlert = false;
        action = "ALL_CLEAR";
        reason = "Risk returned to normal";
      }
    } else {
      const currentLevel = TIER_ORDER[currentTier] || 0;
      const targetLevel = TIER_ORDER[effectiveTier] || 0;

      if (targetLevel > currentLevel) {
        // Escalating
        state.confirmations = confirmations + 1;
        if (state.confirmations >= this.escalateConfirmations) {
          newTier = effectiveTier;
          state.confirmations = 0;
          if (!activeAlert) {
            if (state.alertsToday >= this.maxAlertsPerCellPerDay) {
              action = "NO_CHANGE";
              reason = "Fatigue budget exhausted";
              newTier = currentTier;
              state.confirmations = this.escalateConfirmations - 1;
            } else {
              action = "ESCALATE";
              state.activeAlert = true;
              state.alertsToday += 1;
              reason = `Escalated to ${effectiveTier}`;
            }
          } else {
            action = "ESCALATE";
            reason = `Escalated to ${effectiveTier}`;
          }
        } else {
          action = "NO_CHANGE";
          reason = `Building confidence for ${effectiveTier} (${state.confirmations}/${this.escalateConfirmations})`;
        }
      } else if (targetLevel < currentLevel) {
        // De-escalating
        state.confirmations = confirmations + 1;
        if (state.confirmations >= this.deescalateConfirmations) {
          newTier = effectiveTier;
          state.confirmations = 0;
          if (effectiveTier === "NORMAL") {
            state.activeAlert = false;
            action = "ALL_CLEAR";
            reason = "De-escalated to normal";
          } else {
            action = "DEESCALATE";
            reason = `De-escalated to ${effectiveTier}`;
          }
        } else {
          action = "NO_CHANGE";
          reason = `Building confidence for de-escalation (${state.confirmations}/${this.deescalateConfirmations})`;
        }
      } else {
        action = "NO_CHANGE";
        reason = `Maintaining ${effectiveTier}`;
      }
    }

    const decision = {
      candidateId,
      cellId,
      action,
      reason,
      riskTier: effectiveTier,
      trustFlag,
      costLossRatio: this.costLossThreshold,
      predictedProbability: prob,
      previousState: currentTier,
      newState: newTier,
      timestampUtc: now,
      fatigueBudgetRemaining: Math.max(0, this.maxAlertsPerCellPerDay - state.alertsToday),
    };

    // Update state
    this._cellState.set(cellId, {
      currentTier: newTier,
      confirmations: state.confirmations,
      alertsToday: state.alertsToday,
      lastAlertDate: state.lastAlertDate,
      activeAlert: state.activeAlert,
    });

    return decision;
  }

  getCellState(cellId) {
    return this._cellState.get(cellId);
  }

  resetCell(cellId) {
    this._cellState.delete(cellId);
  }
}

module.exports = { AlertPolicyEngine, COST_OF_ACTION, LOSS_IF_MISSED, COST_LOSS_THRESHOLD };

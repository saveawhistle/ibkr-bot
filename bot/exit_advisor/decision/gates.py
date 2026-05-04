"""Risk gate framework — hard guardrails (non-negotiable safety rules)
plus soft gates (parameterized, tunable bounds).

Every policy decision passes through the gate chain before the harness
acts on it. Gates can VETO a decision (which becomes ``hold``) but
cannot modify it. A vetoed decision still gets logged with its full
chain trace for forensic analysis — the goal is to learn which gates
were near-misses and which were active barriers, not to silently
suppress activity.

Chain order:
1. Hard guardrails first — cheap to evaluate, fail fast on safety violations.
2. ``DrawdownAccelerationGate`` before ``ConfidenceThresholdGate`` —
   it sets a shared flag in ``GateContext`` that the confidence gate
   reads to lower its effective threshold during a drawdown event.
3. Other soft gates in any order.

The "continue evaluating after rejection" choice is deliberate: capturing
which subsequent gates would have rejected (or accepted) is more useful
for calibration than short-circuiting on the first veto. If this turns
out to produce noisy results in practice, switching to short-circuit is
a one-line change in :func:`apply_gate_chain`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol

from .policy import ExitDecision, TradeState

if TYPE_CHECKING:
    from bot.config import ExitGatesConfig


@dataclass
class GateContext:
    """State shared across gates for a single decision evaluation.

    ``drawdown_acceleration_active`` is mutable: ``DrawdownAccelerationGate``
    writes it; ``ConfidenceThresholdGate`` reads it. Order in the chain
    matters for this reason — see :func:`build_default_gate_chain`.
    """

    current_timestamp: datetime
    recent_decisions: list[tuple[datetime, ExitDecision]] = field(default_factory=list)
    events_emitted_this_trade: int = 0
    bar_count_in_trade: int = 0
    drawdown_acceleration_active: bool = False


@dataclass(frozen=True)
class GateResult:
    accepted: bool
    rejection_reason: str = ""
    rejection_detail: dict[str, Any] = field(default_factory=dict)


class Gate(Protocol):
    name: str

    def evaluate(
        self,
        trade_state: TradeState,
        decision: ExitDecision,
        gate_context: GateContext,
    ) -> GateResult:  # pragma: no cover - protocol
        ...


# ============================================================
# Hard guardrails — non-configurable, always active
# ============================================================


@dataclass
class StopProtectionGuardrail:
    """Cannot widen or remove the protective stop. Tightening only."""

    name: str = "stop_protection"

    def evaluate(
        self,
        trade_state: TradeState,
        decision: ExitDecision,
        gate_context: GateContext,
    ) -> GateResult:
        if decision.action != "tighten_stop":
            return GateResult(True)
        if decision.new_stop_price is None:
            return GateResult(
                False,
                "tighten_stop_missing_new_price",
                {"decision": _decision_dict(decision)},
            )
        # Long-only: tightening means moving the stop UP closer to the price.
        if decision.new_stop_price <= trade_state.current_stop:
            return GateResult(
                False,
                "stop_widening_attempted",
                {
                    "current_stop": trade_state.current_stop,
                    "proposed_stop": decision.new_stop_price,
                },
            )
        return GateResult(True)


@dataclass
class NoReentryGuardrail:
    """No actions on a flat position — re-entry is the strategy layer's job."""

    name: str = "no_reentry"

    def evaluate(
        self,
        trade_state: TradeState,
        decision: ExitDecision,
        gate_context: GateContext,
    ) -> GateResult:
        if trade_state.current_position_size == 0:
            return GateResult(
                False,
                "no_action_on_flat_position",
                {"decision_action": decision.action},
            )
        return GateResult(True)


@dataclass
class ProtectedPositionGuardrail:
    """Defense-in-depth: re-affirm the layer-1 sacred-ground rule at the
    gate layer. Policy invocation is already gated on ``is_protected``,
    but if a future code path bypasses the policy gate this guardrail
    catches it."""

    name: str = "protected_position"

    def evaluate(
        self,
        trade_state: TradeState,
        decision: ExitDecision,
        gate_context: GateContext,
    ) -> GateResult:
        if not trade_state.is_protected:
            return GateResult(
                False,
                "position_not_yet_protected",
                {"position_size": trade_state.current_position_size},
            )
        return GateResult(True)


@dataclass
class NakedPositionGuardrail:
    """Exit-partial recommendations cannot leave less than 1 share —
    that would either break protective-stop sizing or be effectively
    a full exit dressed as a partial. Use ``exit_full`` if that's
    what's intended."""

    name: str = "naked_position"

    def evaluate(
        self,
        trade_state: TradeState,
        decision: ExitDecision,
        gate_context: GateContext,
    ) -> GateResult:
        if decision.action != "exit_partial":
            return GateResult(True)
        remaining = trade_state.current_position_size * (1 - decision.partial_pct)
        if remaining < 1:
            return GateResult(
                False,
                "partial_would_leave_no_position",
                {
                    "current_size": trade_state.current_position_size,
                    "partial_pct": decision.partial_pct,
                },
            )
        return GateResult(True)


@dataclass
class MaxHoldTimeGuardrail:
    """Final backstop for the hold-time rule. The
    :class:`policy.MaxHoldTimePolicy` is what *produces* the force-exit
    decision; this guardrail only ensures that force-exits are always
    allowed through regardless of other gate results — the parameter is
    configurable, but the rule (force-exit when reached) is not.
    """

    max_hold_minutes: int
    name: str = "max_hold_time"

    def evaluate(
        self,
        trade_state: TradeState,
        decision: ExitDecision,
        gate_context: GateContext,
    ) -> GateResult:
        # We never *reject* here; the gate's role is to whitelist the
        # specific force-exit reason produced by MaxHoldTimePolicy.
        return GateResult(True)


# ============================================================
# Soft gates — configurable thresholds, can be calibrated
# ============================================================


@dataclass
class ConfidenceThresholdGate:
    """Reject low-confidence non-hold recommendations.

    During a drawdown (signaled by ``GateContext.drawdown_acceleration_active``),
    falls back to a lower threshold so the advisor can act decisively
    when conditions are deteriorating fast.
    """

    threshold: float
    drawdown_reduced_threshold: float | None = None
    name: str = "confidence_threshold"

    def evaluate(
        self,
        trade_state: TradeState,
        decision: ExitDecision,
        gate_context: GateContext,
    ) -> GateResult:
        if decision.action == "hold":
            return GateResult(True)
        effective = self.threshold
        if (
            self.drawdown_reduced_threshold is not None
            and gate_context.drawdown_acceleration_active
        ):
            effective = self.drawdown_reduced_threshold
        if decision.confidence < effective:
            return GateResult(
                False,
                "confidence_below_threshold",
                {
                    "confidence": decision.confidence,
                    "threshold": effective,
                    "drawdown_reduced": gate_context.drawdown_acceleration_active,
                },
            )
        return GateResult(True)


@dataclass
class RecencyThrottleGate:
    """Reject decisions that REVERSE a recent non-hold action within
    the throttle window. Prevents flapping (exit_partial → tighten_stop
    in two consecutive bars). Same-action repeats and pure ``hold``
    decisions are always allowed."""

    throttle_seconds: int
    name: str = "recency_throttle"

    def evaluate(
        self,
        trade_state: TradeState,
        decision: ExitDecision,
        gate_context: GateContext,
    ) -> GateResult:
        if decision.action == "hold":
            return GateResult(True)
        if not gate_context.recent_decisions:
            return GateResult(True)
        last_ts, last_decision = gate_context.recent_decisions[0]
        if last_decision.action == "hold":
            return GateResult(True)
        if last_decision.action == decision.action:
            return GateResult(True)
        elapsed = gate_context.current_timestamp - last_ts
        if elapsed < timedelta(seconds=self.throttle_seconds):
            return GateResult(
                False,
                "recency_throttle_active",
                {
                    "last_action": last_decision.action,
                    "current_action": decision.action,
                    "seconds_since_last": elapsed.total_seconds(),
                },
            )
        return GateResult(True)


def _current_r_multiple(trade_state: TradeState) -> tuple[float | None, float]:
    """Return (R-multiple at current price, risk_per_share) or (None, 0)
    if the risk denominator is degenerate (entry <= initial_stop)."""
    risk = trade_state.entry_price - trade_state.initial_stop
    if risk <= 0:
        return None, 0.0
    return (trade_state.current_price - trade_state.entry_price) / risk, risk


@dataclass
class MinRMultipleForPartialGate:
    """Reject ``exit_partial`` recommendations below the configured R-multiple."""

    min_r: float
    name: str = "min_r_for_partial"

    def evaluate(
        self,
        trade_state: TradeState,
        decision: ExitDecision,
        gate_context: GateContext,
    ) -> GateResult:
        if decision.action != "exit_partial":
            return GateResult(True)
        current_r, risk = _current_r_multiple(trade_state)
        if current_r is None:
            return GateResult(
                False,
                "invalid_risk_per_share",
                {
                    "entry": trade_state.entry_price,
                    "initial_stop": trade_state.initial_stop,
                },
            )
        if current_r < self.min_r:
            return GateResult(
                False,
                "below_min_r_for_partial",
                {"current_r": current_r, "min_r": self.min_r},
            )
        return GateResult(True)


@dataclass
class MinRMultipleForStopTightenGate:
    """Reject ``tighten_stop`` recommendations below the configured R-multiple."""

    min_r: float
    name: str = "min_r_for_stop_tighten"

    def evaluate(
        self,
        trade_state: TradeState,
        decision: ExitDecision,
        gate_context: GateContext,
    ) -> GateResult:
        if decision.action != "tighten_stop":
            return GateResult(True)
        current_r, risk = _current_r_multiple(trade_state)
        if current_r is None:
            return GateResult(
                False,
                "invalid_risk_per_share",
                {
                    "entry": trade_state.entry_price,
                    "initial_stop": trade_state.initial_stop,
                },
            )
        if current_r < self.min_r:
            return GateResult(
                False,
                "below_min_r_for_stop_tighten",
                {"current_r": current_r, "min_r": self.min_r},
            )
        return GateResult(True)


@dataclass
class DrawdownAccelerationGate:
    """Sets ``GateContext.drawdown_acceleration_active`` when the trade's
    R has retraced ``drawdown_pct`` from peak. Doesn't reject decisions
    itself; the confidence-threshold gate consumes the flag downstream.
    """

    drawdown_pct: float
    name: str = "drawdown_acceleration"

    def evaluate(
        self,
        trade_state: TradeState,
        decision: ExitDecision,
        gate_context: GateContext,
    ) -> GateResult:
        risk = trade_state.entry_price - trade_state.initial_stop
        if risk <= 0 or trade_state.peak_price <= trade_state.entry_price:
            gate_context.drawdown_acceleration_active = False
            return GateResult(True)
        peak_r = (trade_state.peak_price - trade_state.entry_price) / risk
        current_r = (trade_state.current_price - trade_state.entry_price) / risk
        if peak_r <= 0:
            gate_context.drawdown_acceleration_active = False
            return GateResult(True)
        drawdown_fraction = (peak_r - current_r) / peak_r
        gate_context.drawdown_acceleration_active = drawdown_fraction >= self.drawdown_pct
        return GateResult(True)


# ============================================================
# Chain composition + application
# ============================================================


def build_default_gate_chain(config: ExitGatesConfig) -> list[Gate]:
    """Canonical order: hard guardrails first, then drawdown gate before
    confidence gate (it sets the shared flag), then remaining soft gates.

    Tweaking this order has subtle implications — the chain isn't fully
    independent. The drawdown→confidence ordering is the only mandatory
    one in v1; everything else is order-independent today.
    """
    return [
        ProtectedPositionGuardrail(),
        NoReentryGuardrail(),
        StopProtectionGuardrail(),
        NakedPositionGuardrail(),
        MaxHoldTimeGuardrail(max_hold_minutes=config.max_hold_minutes),
        DrawdownAccelerationGate(drawdown_pct=config.drawdown_acceleration_pct),
        ConfidenceThresholdGate(
            threshold=config.confidence_threshold,
            drawdown_reduced_threshold=config.drawdown_reduced_confidence_threshold,
        ),
        RecencyThrottleGate(throttle_seconds=config.recency_throttle_seconds),
        MinRMultipleForPartialGate(min_r=config.min_r_for_partial),
        MinRMultipleForStopTightenGate(min_r=config.min_r_for_stop_tighten),
    ]


def apply_gate_chain(
    gates: list[Gate],
    trade_state: TradeState,
    decision: ExitDecision,
    gate_context: GateContext,
) -> tuple[ExitDecision | None, list[tuple[str, GateResult]]]:
    """Apply ``gates`` in order and return ``(final_decision, results)``.

    ``final_decision`` is ``None`` if any gate rejected (harness treats
    that as a hold). All gates are evaluated even after a rejection so
    the forensic record captures the full chain — calibration analysis
    needs to see which downstream gates would also have rejected.

    A gate that raises is recorded as a synthetic ``"error"`` rejection
    rather than crashing the harness — gate misbehavior shouldn't take
    out the replay.
    """
    results: list[tuple[str, GateResult]] = []
    final: ExitDecision | None = decision
    for gate in gates:
        try:
            result = gate.evaluate(trade_state, decision, gate_context)
        except Exception as exc:  # noqa: BLE001 — gate errors must be captured, not crash
            result = GateResult(
                accepted=False,
                rejection_reason="gate_evaluation_error",
                rejection_detail={"gate": gate.name, "error": repr(exc)},
            )
        results.append((gate.name, result))
        if not result.accepted:
            final = None
    return final, results


def _decision_dict(decision: ExitDecision) -> dict[str, Any]:
    """Serializable view of an :class:`ExitDecision` for rejection
    detail — avoids leaking dataclass internals into logs."""
    out = asdict(decision)
    return out

"""Gate framework + exit-policy implementations.

Two related concerns:

* :mod:`bot.exit_advisor.decision.gates` — the gate chain that
  filters/short-circuits exit decisions before they're acted on
  (hard guardrails like StopProtection / NakedPosition + soft gates
  like ConfidenceThreshold / RecencyThrottle).
* :mod:`bot.exit_advisor.decision.policy` — the policy classes that
  *produce* exit decisions (Actual replay, Oracle perfect-knowledge
  baseline, MechanicalTrail, FixedR, StallExit, MaxHoldTime).
"""

from bot.exit_advisor.decision.gates import (
    ConfidenceThresholdGate,
    DrawdownAccelerationGate,
    Gate,
    GateContext,
    GateResult,
    MaxHoldTimeGuardrail,
    MinRMultipleForPartialGate,
    MinRMultipleForStopTightenGate,
    NakedPositionGuardrail,
    NoReentryGuardrail,
    ProtectedPositionGuardrail,
    RecencyThrottleGate,
    StopProtectionGuardrail,
    apply_gate_chain,
    build_default_gate_chain,
)
from bot.exit_advisor.decision.policy import (
    ActualPolicy,
    ExitDecision,
    ExitPolicy,
    FixedRTakeProfit,
    MaxHoldTimePolicy,
    MechanicalTrailPolicy,
    OracleExitPolicy,
    StallExitPolicy,
    TradeState,
)

__all__ = [
    # gates — framework
    "Gate",
    "GateContext",
    "GateResult",
    "build_default_gate_chain",
    "apply_gate_chain",
    # gates — hard guardrails
    "StopProtectionGuardrail",
    "NoReentryGuardrail",
    "ProtectedPositionGuardrail",
    "NakedPositionGuardrail",
    "MaxHoldTimeGuardrail",
    # gates — soft gates
    "ConfidenceThresholdGate",
    "RecencyThrottleGate",
    "MinRMultipleForPartialGate",
    "MinRMultipleForStopTightenGate",
    "DrawdownAccelerationGate",
    # policy — framework
    "ExitDecision",
    "ExitPolicy",
    "TradeState",
    # policy — implementations
    "ActualPolicy",
    "OracleExitPolicy",
    "MechanicalTrailPolicy",
    "FixedRTakeProfit",
    "StallExitPolicy",
    "MaxHoldTimePolicy",
]

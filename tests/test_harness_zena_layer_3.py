"""ZENA layer-3 harness tests — gate chain enabled, ActualPolicy still
reproduces recorded P&L, gate counters and event log populated."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from bot.config import ExitEventsConfig, ExitGatesConfig, Settings
from bot.exit_advisor.core.events import GateChainResult
from bot.exit_advisor.decision.policy import ActualPolicy
from bot.exit_advisor.replay.harness import TradeReplayHarness
from bot.exit_advisor.replay.replay_source import TradeReplayData, load_trade_replay_data

ZENA_DATE = date(2026, 4, 30)
FIXTURES = Path(__file__).parent / "fixtures" / "exit_advisor_zena"


@pytest.fixture(scope="module")
def zena_replay() -> TradeReplayData:
    return load_trade_replay_data("ZENA", ZENA_DATE, cache_dir=FIXTURES)


@pytest.fixture(scope="module")
def events_cfg() -> ExitEventsConfig:
    return Settings().exit_events


@pytest.fixture(scope="module")
def gates_cfg() -> ExitGatesConfig:
    return Settings().exit_gates


def test_zena_actual_policy_pnl_unchanged_with_gates(
    zena_replay: TradeReplayData,
    events_cfg: ExitEventsConfig,
    gates_cfg: ExitGatesConfig,
) -> None:
    """Layer 1+2+2.5 correctness regression with the gate chain in
    place. ActualPolicy's exit decision (exit_full at recorded ts,
    confidence=1.0) should pass every gate."""
    harness = TradeReplayHarness(
        zena_replay,
        ActualPolicy(zena_replay),
        events_cfg,
        gates_config=gates_cfg,
    )
    result = harness.run()
    assert result.exit_price == zena_replay.recorded_exit_price
    assert abs(result.final_pnl - zena_replay.recorded_pnl) < 0.01


def test_zena_decision_passes_all_gates(
    zena_replay: TradeReplayData,
    events_cfg: ExitEventsConfig,
    gates_cfg: ExitGatesConfig,
) -> None:
    """ActualPolicy's exit_full confidence=1.0 has no gate that should
    reject it — the decision exists at the recorded exit time, well
    inside the 60-min hold limit, with full confidence."""
    harness = TradeReplayHarness(
        zena_replay,
        ActualPolicy(zena_replay),
        events_cfg,
        gates_config=gates_cfg,
    )
    result = harness.run()
    assert result.decisions_proposed == 1
    assert result.decisions_accepted == 1
    assert result.decisions_rejected == 0
    assert result.gate_rejections == []


def test_zena_gate_chain_result_logged(
    zena_replay: TradeReplayData,
    events_cfg: ExitEventsConfig,
    gates_cfg: ExitGatesConfig,
) -> None:
    """One GateChainResult event per policy decision processed."""
    harness = TradeReplayHarness(
        zena_replay,
        ActualPolicy(zena_replay),
        events_cfg,
        gates_config=gates_cfg,
    )
    result = harness.run()
    chain_events = [e for e in result.events_emitted if isinstance(e, GateChainResult)]
    assert len(chain_events) == 1
    chain_evt = chain_events[0]
    # Final == original (no veto), and every gate result is accepted.
    assert chain_evt.final_decision is chain_evt.original_decision
    assert all(accepted for _, accepted, _ in chain_evt.gate_results)


def test_zena_gates_disabled_bypasses_chain(
    zena_replay: TradeReplayData, events_cfg: ExitEventsConfig
) -> None:
    """With gates_config.enabled=False, the gate chain is empty and
    no GateChainResult events fire — but P&L correctness still holds."""
    disabled = ExitGatesConfig(enabled=False)
    harness = TradeReplayHarness(
        zena_replay,
        ActualPolicy(zena_replay),
        events_cfg,
        gates_config=disabled,
    )
    result = harness.run()
    assert abs(result.final_pnl - zena_replay.recorded_pnl) < 0.01
    chain_events = [e for e in result.events_emitted if isinstance(e, GateChainResult)]
    assert chain_events == []
    assert result.decisions_proposed == 1
    assert result.decisions_accepted == 1


def test_zena_no_gates_config_falls_back_to_layer_2_behavior(
    zena_replay: TradeReplayData, events_cfg: ExitEventsConfig
) -> None:
    """Backward compatibility: callers that don't pass ``gates_config``
    get the layer-2 behavior (single advisor policy, no gate chain).
    Important so layer-1/2/2.5 tests don't break."""
    harness = TradeReplayHarness(
        zena_replay,
        ActualPolicy(zena_replay),
        events_cfg,
    )
    result = harness.run()
    assert abs(result.final_pnl - zena_replay.recorded_pnl) < 0.01
    chain_events = [e for e in result.events_emitted if isinstance(e, GateChainResult)]
    assert chain_events == []

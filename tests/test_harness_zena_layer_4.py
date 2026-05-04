"""Layer 4 harness regression — confirms the new policies don't
break the harness's behavior on prior-layer correctness oracles."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from bot.config import Settings
from bot.exit_advisor.decision.policy import (
    ActualPolicy,
    FixedRTakeProfit,
    MechanicalTrailPolicy,
    OracleExitPolicy,
    StallExitPolicy,
)
from bot.exit_advisor.replay.harness import TradeReplayHarness
from bot.exit_advisor.replay.replay_source import TradeReplayData, load_trade_replay_data

ZENA_DATE = date(2026, 4, 30)
FIXTURES = Path(__file__).parent / "fixtures" / "exit_advisor_zena"


@pytest.fixture(scope="module")
def zena_replay() -> TradeReplayData:
    return load_trade_replay_data("ZENA", ZENA_DATE, cache_dir=FIXTURES)


def test_actual_policy_still_matches_recorded_pnl(zena_replay: TradeReplayData) -> None:
    settings = Settings()
    harness = TradeReplayHarness(
        zena_replay,
        ActualPolicy(zena_replay),
        settings.exit_events,
        gates_config=settings.exit_gates,
    )
    result = harness.run()
    assert abs(result.final_pnl - zena_replay.recorded_pnl) < 0.01


def test_oracle_policy_runs_and_exits(zena_replay: TradeReplayData) -> None:
    """OracleExitPolicy on ZENA's 1-bar trade window should produce a
    decision (exit at the optimal close) and the harness should treat
    it as a normal exit_full."""
    settings = Settings()
    harness = TradeReplayHarness(
        zena_replay,
        OracleExitPolicy(zena_replay),
        settings.exit_events,
        gates_config=settings.exit_gates,
    )
    result = harness.run()
    assert result.decisions_accepted == 1
    assert result.exit_price is not None
    # Oracle picks the highest-close bar in the trade window. For ZENA's
    # single bar, that's the bar's close (~2.175).
    assert 2.10 <= result.exit_price <= 2.25


def test_mechanical_trail_does_not_propose_below_initial_stop(
    zena_replay: TradeReplayData,
) -> None:
    """ZENA's bracket stop is $0.02 below entry — tighter than any of
    the layer-4 trail parameters. Trail should propose nothing because
    every proposed level is below the initial stop."""
    settings = Settings()
    harness = TradeReplayHarness(
        zena_replay,
        MechanicalTrailPolicy(trail_abs=0.10),
        settings.exit_events,
        gates_config=settings.exit_gates,
    )
    result = harness.run()
    assert result.decisions_proposed == 0


def test_fixed_r_take_profit_does_not_fire_on_zena(
    zena_replay: TradeReplayData,
) -> None:
    """ZENA's trade-window peak (bar high ~2.18) doesn't exceed entry,
    so no R-multiple > 0 — no FixedR target reached."""
    settings = Settings()
    harness = TradeReplayHarness(
        zena_replay,
        FixedRTakeProfit(target_r=1.0),
        settings.exit_events,
        gates_config=settings.exit_gates,
    )
    result = harness.run()
    assert result.decisions_proposed == 0


def test_stall_exit_does_not_fire_on_short_trade(zena_replay: TradeReplayData) -> None:
    """ZENA's trade lasted ~30 seconds — the stall policy's max_minutes=5
    threshold is never approached."""
    settings = Settings()
    harness = TradeReplayHarness(
        zena_replay,
        StallExitPolicy(target_r=1.0, max_minutes=5),
        settings.exit_events,
        gates_config=settings.exit_gates,
    )
    result = harness.run()
    assert result.decisions_proposed == 0

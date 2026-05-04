"""Regression check: layer 1, 2, 2.5, 3, 4, 4.5 still produce the
expected ZENA P&L after layer L2-A's L2 detectors + multi-day cache
extension landed."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from bot.config import Settings
from bot.exit_advisor.decision.policy import ActualPolicy
from bot.exit_advisor.replay.harness import TradeReplayHarness
from bot.exit_advisor.replay.replay_source import TradeReplayData, load_trade_replay_data

ZENA_DATE = date(2026, 4, 30)
FIXTURES = Path(__file__).parent / "fixtures" / "exit_advisor_zena"


@pytest.fixture(scope="module")
def zena_replay() -> TradeReplayData:
    return load_trade_replay_data("ZENA", ZENA_DATE, cache_dir=FIXTURES)


def test_zena_actual_policy_pnl_unchanged_after_l2_a(
    zena_replay: TradeReplayData,
) -> None:
    """ActualPolicy must still reproduce -$2.38 after the L2-A
    machinery landed. L2 detectors don't fire on bar-only replay
    (no L2 stream in TradeReplayData), so the harness behavior
    on this trade is identical to layer 4.5."""
    settings = Settings()
    harness = TradeReplayHarness(
        zena_replay,
        ActualPolicy(zena_replay),
        settings.exit_events,
        gates_config=settings.exit_gates,
    )
    result = harness.run()
    assert abs(result.final_pnl - zena_replay.recorded_pnl) < 0.01


def test_l2_event_classes_are_now_activatable() -> None:
    """Sanity: layer L2-A removes l2 from the deferred-class list,
    so a config setting l2.enabled=true loads cleanly."""
    from bot.config import ExitEventsConfig

    cfg = ExitEventsConfig.model_validate({"l2": {"enabled": True}})
    assert cfg.l2.enabled is True
    # Sub-detector toggles default to True too — the spec wants them
    # operationally consumable out of the box.
    assert cfg.l2.bid_offer_pulls.enabled
    assert cfg.l2.absorption.enabled
    assert cfg.l2.spread_events.enabled
    assert cfg.l2.imbalance.enabled
    assert cfg.l2.print_clusters.enabled
    assert cfg.l2.large_prints.enabled

"""Layer 2.5 ZENA replay tests — runs the harness with the cache-merged
TradeReplayData and verifies the pre-trade backfill, indicator warmup,
and prior-day-derived events.

Uses the synthetic ZENA fixtures in ``tests/fixtures/`` rather than the
real cache. Mocking the cache via fixture files keeps these tests
deterministic regardless of whether an operator has populated the
on-disk cache yet.
"""

from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path

import pytest

from bot.config import ExitEventsConfig, Settings
from bot.exit_advisor.core.events import (
    LevelDataUnavailable,
    LevelTouched,
)
from bot.exit_advisor.decision.policy import ActualPolicy
from bot.exit_advisor.replay.harness import TradeReplayHarness
from bot.exit_advisor.replay.replay_source import TradeReplayData, load_trade_replay_data

ZENA_DATE = date(2026, 4, 30)
FIXTURES = Path(__file__).parent / "fixtures" / "exit_advisor_zena"


@pytest.fixture(scope="module")
def zena_replay_with_cache() -> TradeReplayData:
    return load_trade_replay_data("ZENA", ZENA_DATE, cache_dir=FIXTURES)


@pytest.fixture(scope="module")
def cfg() -> ExitEventsConfig:
    return Settings().exit_events


def test_zena_actual_policy_pnl_unchanged_with_cache(
    zena_replay_with_cache: TradeReplayData, cfg: ExitEventsConfig
) -> None:
    """Layer 1+2 correctness regression: ActualPolicy must still
    reproduce the recorded -$2.38 P&L within $0.01 even with the
    expanded pre-trade bar context."""
    harness = TradeReplayHarness(
        zena_replay_with_cache, ActualPolicy(zena_replay_with_cache), cfg
    )
    result = harness.run()
    assert result.exit_price == zena_replay_with_cache.recorded_exit_price
    assert abs(result.final_pnl - zena_replay_with_cache.recorded_pnl) < 0.01


def test_zena_pre_trade_backfill_grew_with_cache(
    zena_replay_with_cache: TradeReplayData,
) -> None:
    """Layer 2's session-log-only backfill yielded ~10 bars. With the
    cache merged in, the count should be ~22 in the spec's optimistic
    case. The synthetic fixture covers 09:30 → 10:52 ET (82 minutes),
    so the actual count for these tests sits well above the spec's
    floor — what matters is that we're materially larger than 10."""
    n = len(zena_replay_with_cache.pre_trade_bars)
    assert n >= 22, f"expected at least 22 pre-trade bars after cache merge, got {n}"


def test_zena_vwap_warm_full_session(
    zena_replay_with_cache: TradeReplayData, cfg: ExitEventsConfig
) -> None:
    """With pre-subscription bars now available, VWAP at trade start
    reflects the fuller session, not the last-10-min approximation."""
    harness = TradeReplayHarness(
        zena_replay_with_cache, ActualPolicy(zena_replay_with_cache), cfg
    )
    harness.run()
    vwap = harness._history.session_vwap()
    assert vwap is not None
    # Synthetic fixture's bars hover near 2.15; real ZENA was 2.16-2.18.
    # Either way VWAP should sit comfortably in the 2.10-2.25 range.
    assert 2.05 <= vwap <= 2.30


def test_zena_ema_9_warm_full_session(
    zena_replay_with_cache: TradeReplayData, cfg: ExitEventsConfig
) -> None:
    harness = TradeReplayHarness(
        zena_replay_with_cache, ActualPolicy(zena_replay_with_cache), cfg
    )
    harness.run()
    assert harness._ma_detector is not None
    ema = harness._ma_detector.ema_9_value()
    assert ema is not None
    assert 2.05 <= ema <= 2.30


def test_zena_prior_day_levels_compute_when_cache_populated(
    zena_replay_with_cache: TradeReplayData, cfg: ExitEventsConfig
) -> None:
    """Layer 2's LevelDataUnavailable warnings for prior_day_high /
    prior_day_low / prior_day_close should NOT fire when the prior-day
    cache file is present. Instead, those levels become live and may
    fire LevelTouched events when price reaches them."""
    harness = TradeReplayHarness(
        zena_replay_with_cache, ActualPolicy(zena_replay_with_cache), cfg
    )
    result = harness.run()

    unavailable_levels = {
        e.level_name
        for e in result.events_emitted
        if isinstance(e, LevelDataUnavailable)
    }
    # Prior-day-derived levels must NO LONGER appear as unavailable.
    assert "prior_day_high" not in unavailable_levels
    assert "prior_day_low" not in unavailable_levels
    assert "prior_day_close" not in unavailable_levels

    # And the prior-day levels should be available to the detector — at
    # least one of them should produce a LevelTouched event during the
    # pre-trade backfill or trade window (the synthetic fixture's price
    # band overlaps prior-day's, so touches are expected).
    touched_levels = {
        e.level_name
        for e in result.events_emitted
        if isinstance(e, LevelTouched)
    }
    prior_day_levels = {"prior_day_high", "prior_day_low", "prior_day_close"}
    assert touched_levels & prior_day_levels, (
        f"expected at least one prior-day level touch, got {touched_levels}"
    )


def test_zena_event_counts_summary(
    zena_replay_with_cache: TradeReplayData, cfg: ExitEventsConfig
) -> None:
    """Sanity floor — layer 2.5's expanded backfill produces materially
    more events than layer 2's narrow backfill did. Specific counts
    aren't pre-specified; we just assert the rough shape."""
    harness = TradeReplayHarness(
        zena_replay_with_cache, ActualPolicy(zena_replay_with_cache), cfg
    )
    result = harness.run()
    counts = Counter(type(e).__name__ for e in result.events_emitted)
    # Pre-trade bars now warm up the MA detector, so multiple crosses
    # are expected as price oscillates around VWAP and EMA.
    assert counts.get("MovingAverageCross", 0) >= 1
    # 82 pre-trade bars ought to fire several BarShape detections + wicks.
    assert counts.get("BarShapeDetected", 0) + counts.get("WickEvent", 0) >= 1
    # PositionProtected always fires exactly once.
    assert counts.get("PositionProtected", 0) == 1

"""Layer 2 ZENA replay tests.

Re-runs ZENA's 2026-04-30 trade with the full layer-2 detector stack
enabled and checks that:
1. ActualPolicy still produces the same recorded P&L (regression check —
   layer 2 must not break layer 1's correctness oracle)
2. Pre-trade backfill loaded a sensible number of bars
3. VWAP and 9 EMA are non-None at trade entry (warmup actually warmed)
4. Layer 2 event types appear in the emitted event stream
"""

from __future__ import annotations

from collections import Counter
from datetime import date

import pytest

from bot.config import ExitEventsConfig, Settings
from bot.exit_advisor.core.events import (
    RVolDataUnavailable,
)
from bot.exit_advisor.decision.policy import ActualPolicy
from bot.exit_advisor.replay.harness import TradeReplayHarness
from bot.exit_advisor.replay.replay_source import TradeReplayData, load_trade_replay_data

ZENA_DATE = date(2026, 4, 30)


@pytest.fixture(scope="module")
def zena_replay() -> TradeReplayData:
    # Layer-2 tests deliberately exercise the session-log-only path, so we
    # pin cache_dir to a non-existent directory rather than letting the
    # default (operator-populated) cache leak in. With the cache populated,
    # backfill bar counts and indicator values match layer 2.5's metrics
    # instead of layer 2's — those are validated separately in
    # test_harness_zena_layer_2_5.py.
    return load_trade_replay_data("ZENA", ZENA_DATE, cache_dir="/nonexistent")


@pytest.fixture(scope="module")
def cfg() -> ExitEventsConfig:
    return Settings().exit_events


def test_zena_pre_trade_backfill_loaded(zena_replay: TradeReplayData) -> None:
    """ZENA was added to the watchlist at ~10:52 ET (subscription started
    once the catalyst classifier picked it up), and entered at ~11:02 ET.
    Pre-trade backfill should be ~10 bars — fewer than full session-open
    backfill would yield because the bot wasn't subscribed to ZENA from
    09:30. The exact count is asserted as a sanity range."""
    n = len(zena_replay.pre_trade_bars)
    assert 5 <= n <= 30, f"unexpected pre-trade backfill bar count for ZENA: {n}"


def test_zena_actual_policy_pnl_unchanged(
    zena_replay: TradeReplayData, cfg: ExitEventsConfig
) -> None:
    """Layer 2 regression — ActualPolicy must still reproduce the
    recorded -$2.38 within $0.01."""
    harness = TradeReplayHarness(zena_replay, ActualPolicy(zena_replay), cfg)
    result = harness.run()
    assert result.exit_price == zena_replay.recorded_exit_price
    assert abs(result.final_pnl - zena_replay.recorded_pnl) < 0.01


def test_zena_vwap_warm_at_entry(zena_replay: TradeReplayData, cfg: ExitEventsConfig) -> None:
    """After the pre-trade backfill, VWAP must be non-None and within
    the bar range we saw in pre-trade data (rough sanity check)."""
    harness = TradeReplayHarness(zena_replay, ActualPolicy(zena_replay), cfg)
    harness.run()
    vwap = harness._history.session_vwap()
    assert vwap is not None
    # ZENA's pre-trade bars hovered between 2.16 and 2.18; VWAP must sit
    # somewhere in or near that band.
    assert 2.10 <= vwap <= 2.25


def test_zena_ema_9_warm_at_entry(zena_replay: TradeReplayData, cfg: ExitEventsConfig) -> None:
    """ZENA's pre-trade backfill is ~10 bars, just enough to seed the
    9-bar EMA. After replay the EMA value must be set."""
    harness = TradeReplayHarness(zena_replay, ActualPolicy(zena_replay), cfg)
    harness.run()
    assert harness._ma_detector is not None
    ema = harness._ma_detector.ema_9_value()
    assert ema is not None
    assert 2.10 <= ema <= 2.25


def test_zena_layer_2_events_emitted(zena_replay: TradeReplayData, cfg: ExitEventsConfig) -> None:
    """Layer 2 event types must appear in the emitted stream when the
    detectors are wired in. Specific counts depend on the bar sequence
    and aren't pre-specified — this test only asserts the right
    families of events show up."""
    harness = TradeReplayHarness(zena_replay, ActualPolicy(zena_replay), cfg)
    result = harness.run()
    counts = Counter(type(e).__name__ for e in result.events_emitted)

    # ZENA had no prior-day session log, so RVOL warning should fire.
    assert counts.get("RVolDataUnavailable", 0) >= 1
    # Pre-trade bars touch HOD/LOD; at least one LevelTouched fires.
    assert counts.get("LevelTouched", 0) >= 1
    # VWAP/EMA-9 cross at least once during the warmed run.
    assert counts.get("MovingAverageCross", 0) >= 1
    # The narrow ZENA bars produce at least one bar-shape detection or wick.
    assert counts.get("BarShapeDetected", 0) + counts.get("WickEvent", 0) >= 1


def test_zena_data_unavailable_warnings_fire_once(
    zena_replay: TradeReplayData, cfg: ExitEventsConfig
) -> None:
    """RVolDataUnavailable + LevelDataUnavailable each fire exactly once
    per missing input — the latch in each detector must not let multiple
    warnings stack on subsequent bars."""
    harness = TradeReplayHarness(zena_replay, ActualPolicy(zena_replay), cfg)
    result = harness.run()

    rvol_warnings = [e for e in result.events_emitted if isinstance(e, RVolDataUnavailable)]
    assert len(rvol_warnings) == 1

    # LevelDataUnavailable fires once per missing level (prior_day_high,
    # prior_day_low, prior_day_close, gap_fill — 4 missing levels for
    # ZENA's no-prior-day case).
    from bot.exit_advisor.core.events import LevelDataUnavailable

    by_level: dict[str, int] = {}
    for e in result.events_emitted:
        if isinstance(e, LevelDataUnavailable):
            by_level[e.level_name] = by_level.get(e.level_name, 0) + 1
    for level, count in by_level.items():
        assert count == 1, f"{level} warning fired {count} times"

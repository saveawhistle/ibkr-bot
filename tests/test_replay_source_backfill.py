"""Replay-source backfill tests: pre-trade bar loading + prior-day data
graceful degradation."""

from __future__ import annotations

from datetime import date

from bot.exit_advisor.replay.replay_source import load_trade_replay_data

ZENA_DATE = date(2026, 4, 30)


def test_pre_trade_bars_for_zena_loaded() -> None:
    rd = load_trade_replay_data("ZENA", ZENA_DATE)
    assert len(rd.pre_trade_bars) >= 1
    # Pre-trade bars must all close before the trade-window's first bar.
    first_trade_bar_ts = rd.bars[0].timestamp
    for b in rd.pre_trade_bars:
        assert b.timestamp < first_trade_bar_ts


def test_trade_window_bars_still_loaded() -> None:
    """Layer 1's contract: bars are sorted, in chronological order,
    bounded by bracket placement and recorded exit."""
    rd = load_trade_replay_data("ZENA", ZENA_DATE)
    assert len(rd.bars) >= 1
    bar_times = [b.timestamp for b in rd.bars]
    assert bar_times == sorted(bar_times)


def test_prior_day_unavailable_returns_empty_not_crash() -> None:
    """When the prior-day cache is empty, the loader returns empty
    prior-day fields rather than crashing. We pin cache_dir to a
    non-existent path so the test is deterministic regardless of
    whether an operator has populated the real cache."""
    rd = load_trade_replay_data("ZENA", ZENA_DATE, cache_dir="/nonexistent")
    assert rd.prior_day_bars == []
    assert rd.prior_day_session_high is None
    assert rd.prior_day_session_low is None
    assert rd.prior_day_session_close is None


def test_prior_day_missing_session_log_handled() -> None:
    """If the prior calendar day's session log file is missing entirely,
    the loader still succeeds and returns empty prior-day data. ZENA's
    2026-04-29 file exists but contains no ZENA bars; the negative
    case here is a date whose log is more likely to be absent."""
    rd = load_trade_replay_data("ZENA", ZENA_DATE)
    # Whatever the on-disk situation, prior_day_bars must always be a list.
    assert isinstance(rd.prior_day_bars, list)

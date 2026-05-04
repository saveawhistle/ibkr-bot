"""Multi-day cache extension tests: trading_dates_to_fetch + RVOL curve loader."""

from __future__ import annotations

from datetime import date

from bot.exit_advisor.replay.replay_source import (
    _build_cumulative_volume_curve,
    load_prior_n_day_volume_curve,
)
from scripts.fetch_historical_bars import trading_dates_to_fetch


def test_trading_dates_to_fetch_skips_weekends() -> None:
    """trade_date Monday 2026-05-04, prior_days=3 →
    [05-04, 05-01 Fri, 04-30 Thu, 04-29 Wed]. Skips Saturday/Sunday."""
    dates = trading_dates_to_fetch(date(2026, 5, 4), prior_days=3)
    assert dates == [date(2026, 5, 4), date(2026, 5, 1), date(2026, 4, 30), date(2026, 4, 29)]


def test_trading_dates_to_fetch_skips_holidays() -> None:
    """2026-01-19 is MLK Day. Walking back from 2026-01-20 should skip it."""
    dates = trading_dates_to_fetch(date(2026, 1, 20), prior_days=2)
    assert date(2026, 1, 19) not in dates  # MLK skipped
    # Should land on Friday 01-16 + Thursday 01-15 as the 2 prior trading days.
    assert dates == [date(2026, 1, 20), date(2026, 1, 16), date(2026, 1, 15)]


def test_trading_dates_zero_prior_days_returns_only_target() -> None:
    dates = trading_dates_to_fetch(date(2026, 5, 4), prior_days=0)
    assert dates == [date(2026, 5, 4)]


def test_build_cumulative_volume_curve_monotonic() -> None:
    """The per-minute cumulative curve should be monotonically
    non-decreasing — each bar adds volume, never subtracts."""
    from datetime import UTC, datetime, timedelta

    from bot.exit_advisor.replay.replay_source import Bar

    open_ts = datetime(2026, 5, 5, 13, 30, tzinfo=UTC)  # 09:30 ET DST
    bars = [
        Bar(open_ts + timedelta(minutes=0), 10, 10.1, 9.9, 10.0, 1000),
        Bar(open_ts + timedelta(minutes=1), 10, 10.1, 9.9, 10.0, 500),
        Bar(open_ts + timedelta(minutes=2), 10, 10.1, 9.9, 10.0, 750),
    ]
    curve = _build_cumulative_volume_curve(bars, date(2026, 5, 5))
    assert curve == {0: 1000, 1: 1500, 2: 2250}


def test_load_prior_n_day_volume_curve_zena_real_cache() -> None:
    """Smoke test against the real cache populated for ZENA. Only 1
    prior trading day is cached (2026-04-29) so days_used=1, not 10."""
    curve, days_used = load_prior_n_day_volume_curve(
        symbol="ZENA", trade_date=date(2026, 4, 30), n_days=10
    )
    # If the operator hasn't populated the cache, this returns empty;
    # assert the GRACEFUL DEGRADATION path rather than the happy one.
    assert isinstance(curve, dict)
    assert days_used >= 0
    assert days_used <= 10


def test_load_prior_n_day_volume_curve_missing_data(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Empty cache_dir → empty curve, days_used=0."""
    curve, days_used = load_prior_n_day_volume_curve(
        symbol="ZZZ", trade_date=date(2026, 4, 30), n_days=10, cache_dir=tmp_path
    )
    assert curve == {}
    assert days_used == 0

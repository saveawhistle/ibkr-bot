"""RTH boundary tests across DST transitions.

Layer 1 used a hand-coded ``3 <= ts.month <= 11`` DST check that misfires
on transition weeks (e.g. early November is -04:00 until the second Sunday).
Layer 2 routes through ``zoneinfo.ZoneInfo("America/New_York")``; these
tests pin down the corner cases the hand-coded check would have gotten wrong.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from bot.exit_advisor.core.timeutil import rth_close_utc, rth_open_for, rth_open_utc


def test_zena_layer_1_date_unchanged() -> None:
    """Sanity: 2026-04-30 was DST (-04:00). RTH open should be 13:30 UTC,
    matching what the hand-coded check produced for layer 1's ZENA test."""
    assert rth_open_utc(date(2026, 4, 30)) == datetime(2026, 4, 30, 13, 30, tzinfo=UTC)
    assert rth_close_utc(date(2026, 4, 30)) == datetime(2026, 4, 30, 20, 0, tzinfo=UTC)


def test_pre_dst_spring_offset() -> None:
    """Early March is still EST (-05:00). RTH open is 14:30 UTC."""
    assert rth_open_utc(date(2026, 3, 5)) == datetime(2026, 3, 5, 14, 30, tzinfo=UTC)


def test_post_dst_spring_offset() -> None:
    """Late March is EDT (-04:00). RTH open is 13:30 UTC."""
    assert rth_open_utc(date(2026, 3, 25)) == datetime(2026, 3, 25, 13, 30, tzinfo=UTC)


def test_pre_dst_fall_offset() -> None:
    """Late October is EDT (-04:00) right up until the first Sunday in
    November. The layer-1 hand-coded check happened to agree here. In
    2026 DST ends on Sunday Nov 1, so Oct 31 is the last EDT day."""
    assert rth_open_utc(date(2026, 10, 31)) == datetime(2026, 10, 31, 13, 30, tzinfo=UTC)


def test_post_dst_fall_offset() -> None:
    """Late November is EST (-05:00). RTH open is 14:30 UTC.

    THIS is where the layer-1 hand-coded check was wrong: it said any
    November date was DST. ``zoneinfo`` correctly returns -05:00."""
    assert rth_open_utc(date(2026, 11, 25)) == datetime(2026, 11, 25, 14, 30, tzinfo=UTC)


def test_winter_offset() -> None:
    """January is EST (-05:00). RTH open is 14:30 UTC."""
    assert rth_open_utc(date(2026, 1, 15)) == datetime(2026, 1, 15, 14, 30, tzinfo=UTC)


def test_rth_open_for_uses_ny_local_date() -> None:
    """An event at 02:00 UTC on day D = 21:00 ET on day D-1; rth_open_for
    should resolve to 09:30 ET on day D-1, not day D."""
    ts = datetime(2026, 4, 30, 2, 0, tzinfo=UTC)  # = 22:00 ET on 04-29
    assert rth_open_for(ts) == datetime(2026, 4, 29, 13, 30, tzinfo=UTC)

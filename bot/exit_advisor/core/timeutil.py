"""RTH boundaries via ``zoneinfo`` — replaces layer 1's hand-coded DST offset.

The harness needs ``09:30 ET`` and ``16:00 ET`` for any given calendar
date as UTC datetimes (for milestone calculations and pre-trade backfill
windows). Hand-coded DST month checks misfire on transition weeks; using
the IANA tz database via ``zoneinfo`` is correct on every date.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)


def rth_open_utc(d: date) -> datetime:
    """09:30 America/New_York on ``d``, returned as UTC."""
    return datetime.combine(d, RTH_OPEN, tzinfo=NY).astimezone(UTC)


def rth_close_utc(d: date) -> datetime:
    """16:00 America/New_York on ``d``, returned as UTC."""
    return datetime.combine(d, RTH_CLOSE, tzinfo=NY).astimezone(UTC)


def rth_open_for(ts: datetime) -> datetime:
    """09:30 ET on the same NY-local date as ``ts`` (returned as UTC).

    Uses NY-local date so an event at 22:00 UTC on day D (which is
    18:00 ET on day D) returns 09:30 ET on day D, not on day D+1.
    """
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    ny_date = ts.astimezone(NY).date()
    return rth_open_utc(ny_date)

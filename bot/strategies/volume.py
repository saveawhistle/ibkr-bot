"""Recent-window relative-volume helper for strategy signal validation.

The scanner's daily RVOL gate (today's cumulative volume vs 10-day average
daily volume) is a *day-scale* qualification: "is this ticker abnormally
active today vs its baseline". That's appropriate for ticker admission to
the watchlist but insufficient for moment-of-entry signal validation -- a
ticker can clear daily RVOL at 9:35 ET on premarket-driven volume and then
fade on dead air during the actual trading session, with the breakout bar
firing on the same low volume that's been trickling for 20 minutes.

Ross's framework treats moment-of-entry volume as a separate decision:
the breakout bar must show ≥2× the rolling-window average of recent bars
to confirm the entry signal. ``RecentVolumeWindow`` implements that
rolling-window average; strategies call ``relative_volume(candidate_bar)``
at signal generation and suppress the signal if the result is below the
strategy's configured threshold.

Stateful design (per the spec): one instance per ticker, fed bar-by-bar
as bars roll in. Strategies in this codebase get the full bars DataFrame
each evaluation, so the helper class is constructed fresh per ``evaluate``
call from the prior-N bars rather than maintained across calls -- the
two approaches are mathematically equivalent for the same window slice,
but constructing per-call keeps the class stateless across strategy
evaluations and avoids sync issues if bars are ever backfilled.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import structlog

_log = structlog.get_logger("bot.strategies.volume")


@dataclass
class _BarLike:
    """Duck-typed bar protocol: anything with ``.volume`` works.

    Strategies pass plain volume integers via ``add_volume`` for ergonomics;
    when they have full bar objects (e.g. from ``bot.indicators``) the
    helper accepts those too via ``add_bar``.
    """

    volume: float


class RecentVolumeWindow:
    """Rolling N-bar volume window for breakout-bar volume validation.

    Constructed per ticker. Feed prior bars via ``add_bar`` (or volumes via
    ``add_volume``); the window evicts oldest entries beyond ``window_bars``
    capacity. ``average_volume`` returns ``None`` until the window holds at
    least ``window_bars`` entries -- the strategy treats that as
    "insufficient history, suppress signal" per the locked spec
    (default behaviour: conservative suppression rather than partial-window
    proportional thresholds).

    ``relative_volume(candidate_bar)`` returns ``candidate_volume /
    average_volume()``, or ``None`` when the window isn't yet populated.
    The candidate bar is NOT added to the window by this call -- the
    contract is "average is over the *prior* N bars; rvol compares the
    *next* (candidate) bar against that baseline".
    """

    def __init__(self, window_bars: int = 20) -> None:
        if window_bars < 1:
            raise ValueError(
                f"window_bars must be >= 1 (got {window_bars}); a zero-bar "
                "window has no meaningful average."
            )
        self._window_bars = window_bars
        # ``maxlen`` gives us O(1) eviction-on-append for free.
        self._volumes: deque[float] = deque(maxlen=window_bars)

    @property
    def window_bars(self) -> int:
        """Configured maximum window size."""
        return self._window_bars

    @property
    def bars_seen(self) -> int:
        """Current number of volumes in the window (≤ ``window_bars``)."""
        return len(self._volumes)

    @property
    def is_populated(self) -> bool:
        """True iff the window holds at least ``window_bars`` entries."""
        return len(self._volumes) >= self._window_bars

    def add_volume(self, volume: float) -> None:
        """Append one volume reading; evicts the oldest if at capacity."""
        if volume < 0:
            # Defensive: a negative bar volume would corrupt the average and
            # is almost certainly a bad data feed reading. Treat as zero.
            volume = 0.0
        self._volumes.append(float(volume))

    def add_bar(self, bar: _BarLike) -> None:
        """Append one bar's volume reading. Convenience wrapper around add_volume."""
        self.add_volume(bar.volume)

    def extend_from_volumes(self, volumes: Iterable[float]) -> None:
        """Bulk-feed volumes (e.g. from a DataFrame slice) in chronological order."""
        for v in volumes:
            self.add_volume(v)

    def average_volume(self) -> float | None:
        """Return the rolling mean, or None when the window isn't yet populated."""
        if not self.is_populated:
            return None
        return sum(self._volumes) / len(self._volumes)

    def relative_volume(self, candidate_volume: float) -> float | None:
        """Return ``candidate_volume / average_volume()``.

        Returns ``None`` when the window isn't populated (signal should
        be suppressed). Returns ``None`` when the average is zero (a
        legitimate but pathological case -- 20 consecutive zero-volume
        bars; treat as "no baseline" rather than divide-by-zero).
        """
        avg = self.average_volume()
        if avg is None or avg <= 0:
            return None
        return float(candidate_volume) / avg


def check_recent_window_rvol(
    *,
    bars: pd.DataFrame,
    window_bars: int,
    threshold: float,
    symbol: str,
    strategy: str,
    bar_time: datetime | pd.Timestamp,
) -> str | None:
    """Validate the latest bar's volume against the prior-N-bar average.

    Returns ``None`` when the signal should proceed; returns a short
    suppression-reason string (``"window_not_populated"`` /
    ``"low_recent_rvol"``) when the strategy should NOT emit. Logs the
    suppression event with full forensic context as a side effect.

    Contract:

    * The ``bars`` DataFrame's last row is the candidate breakout bar.
    * The prior ``window_bars`` rows are the rolling baseline.
    * If ``bars`` doesn't hold enough rows to populate the window
      (i.e. fewer than ``window_bars + 1`` total rows), the helper
      returns ``"window_not_populated"`` and logs
      ``strategy.signal_suppressed_window_not_populated``. This is the
      conservative spec: suppress until the window is fully populated
      rather than fall back to partial-window proportional thresholds.
    * If the candidate's volume / window-average is below
      ``threshold``, returns ``"low_recent_rvol"`` and logs
      ``strategy.signal_suppressed_recent_rvol``.

    The helper does NOT mutate ``bars`` and does NOT add the candidate
    bar to the rolling window -- the average is over the *prior* N bars
    only, and the candidate is the comparison subject.
    """
    if threshold <= 0:
        # Disabled-via-test path. Production config's validator enforces
        # ``recent_rvol_min > 0``, so this branch is reachable only from
        # tests that opt out by passing ``recent_rvol_min=0.0`` to the
        # strategy constructor directly. Mirrors the
        # ``universe.rvol_min <= 0`` test-disable in the scanner pillar.
        return None
    if "volume" not in bars.columns:
        # Synthetic-frame test path. Bail quietly to None -- pre-12.4
        # strategies emitted in this case without any volume gate.
        return None
    total_rows = len(bars)
    if total_rows < window_bars + 1:
        ts_iso = bar_time.isoformat() if hasattr(bar_time, "isoformat") else str(bar_time)
        _log.info(
            "strategy.signal_suppressed_window_not_populated",
            symbol=symbol,
            strategy=strategy,
            bar_time=ts_iso,
            bars_available=total_rows,
            window_required=window_bars + 1,
        )
        return "window_not_populated"
    window = RecentVolumeWindow(window_bars=window_bars)
    # Prior-N bars: rows [-window_bars-1, -1) -- excludes the candidate.
    prior_volumes = bars["volume"].iloc[-window_bars - 1 : -1]
    window.extend_from_volumes(float(v) for v in prior_volumes)
    candidate_volume = float(bars["volume"].iloc[-1])
    rvol = window.relative_volume(candidate_volume)
    avg = window.average_volume()
    if rvol is None:
        # Window populated but average is zero -- legitimate but
        # pathological. Treat as "no baseline; suppress" rather than
        # divide-by-zero. Use the not_populated event so the operator
        # has one bucket to grep for "couldn't compute rvol".
        ts_iso = bar_time.isoformat() if hasattr(bar_time, "isoformat") else str(bar_time)
        _log.info(
            "strategy.signal_suppressed_window_not_populated",
            symbol=symbol,
            strategy=strategy,
            bar_time=ts_iso,
            bars_available=total_rows,
            window_required=window_bars + 1,
            window_average=avg,
            note="zero_average",
        )
        return "window_not_populated"
    if rvol < threshold:
        ts_iso = bar_time.isoformat() if hasattr(bar_time, "isoformat") else str(bar_time)
        _log.info(
            "strategy.signal_suppressed_recent_rvol",
            symbol=symbol,
            strategy=strategy,
            bar_time=ts_iso,
            candidate_volume=candidate_volume,
            window_average=round(avg, 2) if avg is not None else None,
            rvol=round(rvol, 3),
            threshold=threshold,
            window_bars=window_bars,
        )
        return "low_recent_rvol"
    return None


__all__ = ["RecentVolumeWindow", "check_recent_window_rvol"]

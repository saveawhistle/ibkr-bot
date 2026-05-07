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
from dataclasses import dataclass
from typing import Iterable


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


__all__ = ["RecentVolumeWindow"]

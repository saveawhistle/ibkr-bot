"""Per-day cost tracking for the Phase 12 catalyst classifier.

Mirrors :class:`bot.exit_advisor.advisor.cost_tracker.CostTracker` —
two thresholds (soft warning, hard cap), latched once tripped, with
optional notification callback. Differs in one important way: the
catalyst classifier's caps are **per-day**, not per-session.

Why per-day rather than per-session: the classifier is called every
scanner pass (~every 5 minutes during the morning window) for every
catalyst-bearing ticker. A single session can run from premarket
(~07:00 ET) through 11:30 ET = 4.5 hours. With 5-10 tickers per pass
and a soft cap meant for "warn me when I'm a couple sessions deep
without noticing," day-level rollover keeps the alert actionable.

Implementation: track the current day-of-year (in NY-local time) and
reset accumulators on the first ``record_cost`` call after the day
flips. Tests can inject a clock function to drive deterministic
rollover.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from datetime import date, datetime
from zoneinfo import ZoneInfo

DEFAULT_SOFT_CAP_USD = 5.0
DEFAULT_HARD_CAP_USD = 25.0
_NY = ZoneInfo("America/New_York")


def _today_ny(now_provider: Callable[[], datetime] | None = None) -> date:
    """Return today's NY-local date. Injectable clock for tests."""
    now = now_provider() if now_provider is not None else datetime.now(_NY)
    if now.tzinfo is None:
        now = now.replace(tzinfo=_NY)
    return now.astimezone(_NY).date()


class CatalystCostTracker:
    """Tracks per-day cumulative LLM cost for the catalyst classifier.

    Construction parameters validated; runtime updates not (the classifier
    only feeds non-negative cost deltas reported by the API).

    Soft warning fires once per day (re-arms on day rollover). Hard cap
    latches for the rest of the day; rolls over at NY-local midnight.
    """

    def __init__(
        self,
        soft_cap_usd: float = DEFAULT_SOFT_CAP_USD,
        hard_cap_usd: float = DEFAULT_HARD_CAP_USD,
        notify_callback: Callable[[str], None] | None = None,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        if soft_cap_usd <= 0.0:
            raise ValueError(f"soft_cap_usd must be > 0.0 (got {soft_cap_usd})")
        if hard_cap_usd <= 0.0:
            raise ValueError(f"hard_cap_usd must be > 0.0 (got {hard_cap_usd})")
        if soft_cap_usd >= hard_cap_usd:
            raise ValueError(
                f"soft_cap_usd ({soft_cap_usd}) must be strictly less than "
                f"hard_cap_usd ({hard_cap_usd}); the soft warning must precede the hard cap."
            )
        self._soft_cap_usd = soft_cap_usd
        self._hard_cap_usd = hard_cap_usd
        self._notify_callback = notify_callback
        self._now_provider = now_provider
        self._current_day: date | None = None
        self._cost_today_usd = 0.0
        self._soft_warning_fired_today = False
        self._hard_capped_today = False

    def _maybe_rollover(self) -> None:
        today = _today_ny(self._now_provider)
        if self._current_day is None or today != self._current_day:
            self._current_day = today
            self._cost_today_usd = 0.0
            self._soft_warning_fired_today = False
            self._hard_capped_today = False

    def record_cost(self, cost_usd: float) -> None:
        """Add ``cost_usd`` to today's running total. Triggers cap signals on crossing."""
        if cost_usd < 0.0:
            raise ValueError(f"cost_usd must be non-negative (got {cost_usd})")
        self._maybe_rollover()
        if self._hard_capped_today:
            return
        self._cost_today_usd += cost_usd
        if not self._soft_warning_fired_today and self._cost_today_usd >= self._soft_cap_usd:
            self._soft_warning_fired_today = True
            self._notify(
                f"catalyst-classifier cost soft warning: ${self._cost_today_usd:.2f} "
                f">= soft cap ${self._soft_cap_usd:.2f} (NY date {self._current_day})"
            )
        if not self._hard_capped_today and self._cost_today_usd >= self._hard_cap_usd:
            self._hard_capped_today = True
            self._notify(
                f"catalyst-classifier cost HARD CAP reached: ${self._cost_today_usd:.2f} "
                f">= hard cap ${self._hard_cap_usd:.2f}; classifier returning "
                f"deterministic non-qualify until NY midnight rollover."
            )

    def is_hard_capped(self) -> bool:
        """True iff today's hard cap has tripped. Re-arms automatically on day rollover."""
        self._maybe_rollover()
        return self._hard_capped_today

    def cost_today_usd(self) -> float:
        """Today's running cost total, in USD. Re-armed on day rollover."""
        self._maybe_rollover()
        return self._cost_today_usd

    def soft_warning_fired_today(self) -> bool:
        """True iff today's soft warning has fired (forensics + tests)."""
        self._maybe_rollover()
        return self._soft_warning_fired_today

    def _notify(self, message: str) -> None:
        if self._notify_callback is None:
            return
        with contextlib.suppress(Exception):
            self._notify_callback(message)


__all__ = [
    "DEFAULT_HARD_CAP_USD",
    "DEFAULT_SOFT_CAP_USD",
    "CatalystCostTracker",
]

"""Session-cumulative LLM cost tracking with soft warning + hard cap.

Two thresholds, both configurable:

* **Soft cap** — fires a one-time notification on the first crossing.
  The advisor keeps calling the LLM after this; the warning exists so
  the operator sees that the session's spend is mounting before a hard
  stop ever triggers.
* **Hard cap** — once crossed, the advisor stops calling the LLM and
  returns a deterministic ``hold`` for every subsequent event for the
  rest of the session. The bot's mechanical exit logic continues
  unaffected.

Daily totals reset only by constructing a new tracker; in practice the
bot creates one per session so that's automatic. There is no
across-midnight rollover logic — typical sessions are 9:30 AM to 11:30
AM ET, well clear of midnight.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable


class CostTracker:
    """Tracks session-cumulative LLM cost and enforces caps.

    Construction parameters are validated; runtime updates are not — the
    advisor only feeds non-negative cost deltas reported by the API.

    Soft warning is fired exactly once. Hard cap, once tripped, latches
    True for the lifetime of the tracker.
    """

    def __init__(
        self,
        soft_cap_usd: float = 10.0,
        hard_cap_usd: float = 50.0,
        notify_callback: Callable[[str], None] | None = None,
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
        self._session_cost_usd = 0.0
        self._soft_warning_fired = False
        self._hard_capped = False

    def record_cost(self, cost_usd: float) -> None:
        """Add ``cost_usd`` to the session total. Triggers soft + hard cap signals on crossing."""
        if cost_usd < 0.0:
            # Defensive: reject upstream bugs rather than silently corrupting the running total.
            raise ValueError(f"cost_usd must be non-negative (got {cost_usd})")
        if self._hard_capped:
            # Already capped — record nothing further; hard cap latches.
            return
        self._session_cost_usd += cost_usd
        if not self._soft_warning_fired and self._session_cost_usd >= self._soft_cap_usd:
            self._soft_warning_fired = True
            self._notify(
                f"exit-advisor cost soft warning: ${self._session_cost_usd:.2f} "
                f">= soft cap ${self._soft_cap_usd:.2f}"
            )
        if not self._hard_capped and self._session_cost_usd >= self._hard_cap_usd:
            self._hard_capped = True
            self._notify(
                f"exit-advisor cost HARD CAP reached: ${self._session_cost_usd:.2f} "
                f">= hard cap ${self._hard_cap_usd:.2f}; advisor returning deterministic hold"
            )

    def is_hard_capped(self) -> bool:
        """True once the hard cap has been reached. Latched for the tracker's lifetime."""
        return self._hard_capped

    def session_cost_usd(self) -> float:
        """Total cost recorded so far this session, in USD."""
        return self._session_cost_usd

    def soft_warning_fired(self) -> bool:
        """True iff the soft-cap notification has fired (exposed for forensics + tests)."""
        return self._soft_warning_fired

    def _notify(self, message: str) -> None:
        """Best-effort fire the operator notification; never propagate exceptions."""
        if self._notify_callback is None:
            return
        with contextlib.suppress(Exception):
            self._notify_callback(message)

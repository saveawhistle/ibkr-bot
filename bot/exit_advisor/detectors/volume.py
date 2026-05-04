"""Volume detectors: rolling-baseline bot/exit_advisor/dryup, plus session-cumulative
RVOL milestones against a prior-N-day average.

Within-session rolling baseline (last N bars before the current bar) is
deliberate — bot's setups inherently trade days where today's volume is
abnormally high vs prior days, so the relative comparison that matters
*during the trade* is "is this bar unusual for this trade right now",
not "is today unusual vs prior days". Today-vs-prior-days is what got
the bot into the trade in the first place; layer 2 events are about
within-trade dynamics.

Each (threshold) latch fires once per crossing and re-arms when the
ratio crosses back across the threshold from the other direction.

RVOL milestones require prior-day session logs. Layer 2 ships with
graceful degradation: if no prior data is available, a one-shot
``RVolDataUnavailable`` warning fires and milestones are suppressed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from bot.exit_advisor.core.events import (
    Event,
    RVolDataUnavailable,
    RVolMilestone,
    VolumeDryUp,
    VolumeSpike,
)
from bot.exit_advisor.replay.bar_history import BarHistory
from bot.exit_advisor.replay.replay_source import Bar


@dataclass
class VolumeDetector:
    symbol: str
    spike_threshold_x_avg: float = 2.0
    dryup_threshold_x_avg: float = 0.4
    baseline_window_bars: int = 20
    rvol_milestones: list[float] = field(default_factory=lambda: [1.0, 2.0, 5.0])

    # Per-(threshold) latch keyed by configured value. Spike thresholds
    # could be multi-valued if the config evolves; today both fields are
    # single floats but keeping the shape generic costs nothing.
    _spike_armed: bool = True
    _dryup_armed: bool = True
    _fired_rvol_milestones: set[float] = field(default_factory=set)
    _rvol_warning_emitted: bool = False

    # Prior-N-day cumulative volume curve indexed by minutes-since-open.
    # If empty, RVOL milestones can't be computed.
    prior_day_cum_volume_by_minute: dict[int, float] | None = None
    rvol_prior_days_used: int = 0
    rvol_prior_days_configured: int = 0

    def on_bar(self, bar: Bar, history: BarHistory) -> list[Event]:
        bar_close_ts = bar.timestamp + timedelta(minutes=1)
        events: list[Event] = []

        # Rolling baseline excludes the current bar — recent_bars(N+1) returns
        # up to N+1 bars ending with the current one (since BarHistory.add_bar
        # was called before on_bar). Drop the last to get the baseline window.
        recent = history.recent_bars(self.baseline_window_bars + 1)
        if len(recent) >= self.baseline_window_bars + 1:
            baseline_bars = recent[:-1]
            avg = sum(b.volume for b in baseline_bars) / len(baseline_bars)
            if avg > 0:
                ratio = bar.volume / avg
                if ratio >= self.spike_threshold_x_avg and self._spike_armed:
                    events.append(
                        VolumeSpike(
                            timestamp=bar_close_ts,
                            symbol=self.symbol,
                            bar_volume=bar.volume,
                            rolling_average=avg,
                            ratio=ratio,
                            threshold=self.spike_threshold_x_avg,
                        )
                    )
                    self._spike_armed = False
                elif ratio < self.spike_threshold_x_avg:
                    self._spike_armed = True

                if ratio <= self.dryup_threshold_x_avg and self._dryup_armed:
                    events.append(
                        VolumeDryUp(
                            timestamp=bar_close_ts,
                            symbol=self.symbol,
                            bar_volume=bar.volume,
                            rolling_average=avg,
                            ratio=ratio,
                            threshold=self.dryup_threshold_x_avg,
                        )
                    )
                    self._dryup_armed = False
                elif ratio > self.dryup_threshold_x_avg:
                    self._dryup_armed = True

        # RVOL milestones — only meaningful with prior-day data.
        if self.rvol_milestones:
            self._maybe_emit_rvol(bar, bar_close_ts, history, events)

        return events

    # --- internals ---

    def _maybe_emit_rvol(
        self,
        bar: Bar,
        bar_close_ts: datetime,
        history: BarHistory,
        events: list[Event],
    ) -> None:
        prior = self.prior_day_cum_volume_by_minute
        if not prior:
            if not self._rvol_warning_emitted:
                if self.rvol_prior_days_configured > 0:
                    reason = (
                        f"only {self.rvol_prior_days_used} of "
                        f"{self.rvol_prior_days_configured} prior days available; "
                        "RVOL suppressed"
                    )
                else:
                    reason = "prior-day session logs unavailable; RVOL suppressed"
                events.append(
                    RVolDataUnavailable(
                        timestamp=bar_close_ts,
                        symbol=self.symbol,
                        reason=reason,
                    )
                )
                self._rvol_warning_emitted = True
            return

        # Layer L2-A: when SOME prior days are available but fewer than
        # configured, emit the refined warning once AND continue to fire
        # milestones using whatever data we have. The operator sees
        # both: the partial-data flag and the milestone events.
        if (
            self.rvol_prior_days_configured > 0
            and self.rvol_prior_days_used < self.rvol_prior_days_configured
            and not self._rvol_warning_emitted
        ):
            events.append(
                RVolDataUnavailable(
                    timestamp=bar_close_ts,
                    symbol=self.symbol,
                    reason=(
                        f"only {self.rvol_prior_days_used} of "
                        f"{self.rvol_prior_days_configured} prior days available; "
                        "RVOL milestones still firing on partial data"
                    ),
                )
            )
            self._rvol_warning_emitted = True

        # Bucket by minutes-since-open. ``BarHistory`` doesn't track this
        # mapping itself; the harness should pre-compute the prior-day
        # curve and hand it in. Layer 2 ships RVOL as a graceful-degrade
        # path because none of the test data has a prior-day log for the
        # same symbol — the implementation is correct but rarely fires.
        # When data IS available, the harness populates
        # ``prior_day_cum_volume_by_minute`` and we look up by minute.
        minute_key = self._minutes_since_open(bar)
        if minute_key is None:
            return
        prior_cum = prior.get(minute_key)
        if not prior_cum:
            return
        cum_today = history.cumulative_volume()
        rvol = cum_today / prior_cum
        for milestone in sorted(self.rvol_milestones):
            if rvol >= milestone and milestone not in self._fired_rvol_milestones:
                events.append(
                    RVolMilestone(
                        timestamp=bar_close_ts,
                        symbol=self.symbol,
                        rvol=rvol,
                        milestone=milestone,
                        cumulative_volume_today=cum_today,
                        prior_n_day_average_at_time=prior_cum,
                        prior_days_used=self.rvol_prior_days_used,
                    )
                )
                self._fired_rvol_milestones.add(milestone)

    @staticmethod
    def _minutes_since_open(bar: Bar) -> int | None:
        # Late-binding import to avoid an import cycle at module load.
        from bot.exit_advisor.core.timeutil import rth_open_for

        rth_open = rth_open_for(bar.timestamp)
        delta = (bar.timestamp.astimezone(rth_open.tzinfo) - rth_open).total_seconds() / 60
        if delta < 0:
            return None
        return int(delta)

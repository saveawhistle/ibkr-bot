"""Price-level detectors: HOD, LOD, prior-day high/low/close, gap fill.

LevelTouched fires once per (level, approach direction) per session. After
firing, it re-arms only when price moves at least ``RETOUCH_DIST_PCT``
away from the level on a 1-min close — that's the "meaningful retreat"
threshold called out in the layer 2 spec, encoded numerically rather than
left to interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from bot.exit_advisor.core.events import Event, LevelDataUnavailable, LevelReclaimed, LevelTouched
from bot.exit_advisor.replay.bar_history import BarHistory
from bot.exit_advisor.replay.replay_source import Bar

RETOUCH_DIST_PCT = 0.005
"""0.5% — price must move at least this far from a level before another
LevelTouched of the same direction will fire."""

LevelName = Literal["hod", "lod", "prior_day_high", "prior_day_low", "prior_day_close", "gap_fill"]


@dataclass
class _LevelState:
    fired_from_below: bool = False
    fired_from_above: bool = False
    last_close_side: Literal["above", "below", "unknown"] = "unknown"
    ever_broken_above: bool = False
    ever_broken_below: bool = False


@dataclass
class PriceLevelsDetector:
    symbol: str
    hod_lod_enabled: bool = True
    prior_day_high_low_enabled: bool = True
    prior_day_close_enabled: bool = True
    gap_fill_enabled: bool = True

    prior_day_high: float | None = None
    prior_day_low: float | None = None
    prior_day_close: float | None = None
    today_open: float | None = None
    gap_threshold_pct: float = 0.01

    _states: dict[str, _LevelState] = field(default_factory=dict)
    _data_warnings_emitted: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        for name in (
            "hod",
            "lod",
            "prior_day_high",
            "prior_day_low",
            "prior_day_close",
            "gap_fill",
        ):
            self._states[name] = _LevelState()

    def on_bar(self, bar: Bar, history: BarHistory) -> list[Event]:
        bar_close_ts = bar.timestamp + timedelta(minutes=1)
        events: list[Event] = []

        if self.hod_lod_enabled:
            hod = history.session_high()
            lod = history.session_low()
            self._process_level(bar, bar_close_ts, "hod", hod, events)
            self._process_level(bar, bar_close_ts, "lod", lod, events)

        if self.prior_day_high_low_enabled:
            self._process_static(bar, bar_close_ts, "prior_day_high", self.prior_day_high, events)
            self._process_static(bar, bar_close_ts, "prior_day_low", self.prior_day_low, events)

        if self.prior_day_close_enabled:
            self._process_static(bar, bar_close_ts, "prior_day_close", self.prior_day_close, events)

        if self.gap_fill_enabled:
            gap_level = self._gap_fill_level()
            if gap_level is not None:
                self._process_level(bar, bar_close_ts, "gap_fill", gap_level, events)
            elif self.today_open is None or self.prior_day_close is None:
                self._maybe_emit_data_warning(
                    "gap_fill", "prior-day close or today's open missing", bar_close_ts, events
                )

        return events

    # --- internals ---

    def _gap_fill_level(self) -> float | None:
        if self.today_open is None or self.prior_day_close is None:
            return None
        if self.prior_day_close <= 0:
            return None
        gap_pct = abs(self.today_open - self.prior_day_close) / self.prior_day_close
        if gap_pct < self.gap_threshold_pct:
            return None
        return self.prior_day_close

    def _process_static(
        self,
        bar: Bar,
        bar_close_ts: datetime,
        name: str,
        level: float | None,
        events: list[Event],
    ) -> None:
        if level is None:
            self._maybe_emit_data_warning(
                name, f"prior-day data missing for {name}", bar_close_ts, events
            )
            return
        self._process_level(bar, bar_close_ts, name, level, events)

    def _process_level(
        self,
        bar: Bar,
        bar_close_ts: datetime,
        name: str,
        level: float,
        events: list[Event],
    ) -> None:
        if level <= 0:
            return
        state = self._states[name]

        # ``from_below`` requires the bar to have started at or under the
        # level and reached up to it. ``from_above`` requires the bar to
        # have started at or above the level and reached down to it.
        # Without the open-side guard, every bar would trivially "touch
        # HOD from above" because bar.low <= HOD (which is just bar.high)
        # is always true.
        if bar.high >= level and bar.open <= level:
            if not state.fired_from_below:
                events.append(
                    LevelTouched(
                        timestamp=bar_close_ts,
                        symbol=self.symbol,
                        level_name=name,  # type: ignore[arg-type]
                        level_price=level,
                        current_price=bar.close,
                        direction="from_below",
                    )
                )
                state.fired_from_below = True
        elif bar.close < level * (1 - RETOUCH_DIST_PCT):
            state.fired_from_below = False

        if bar.low <= level and bar.open >= level:
            if not state.fired_from_above:
                events.append(
                    LevelTouched(
                        timestamp=bar_close_ts,
                        symbol=self.symbol,
                        level_name=name,  # type: ignore[arg-type]
                        level_price=level,
                        current_price=bar.close,
                        direction="from_above",
                    )
                )
                state.fired_from_above = True
        elif bar.close > level * (1 + RETOUCH_DIST_PCT):
            state.fired_from_above = False

        # Reclaim tracking.
        if bar.close > level:
            this_side: Literal["above", "below"] = "above"
        elif bar.close < level:
            this_side = "below"
        else:
            return

        prior_side = state.last_close_side
        if prior_side == "unknown":
            state.last_close_side = this_side
            if this_side == "above":
                state.ever_broken_above = True
            else:
                state.ever_broken_below = True
            return

        if prior_side != this_side:
            if this_side == "below" and state.ever_broken_above:
                events.append(
                    LevelReclaimed(
                        timestamp=bar_close_ts,
                        symbol=self.symbol,
                        level_name=name,  # type: ignore[arg-type]
                        level_price=level,
                        direction="above_to_below",
                    )
                )
            elif this_side == "above" and state.ever_broken_below:
                events.append(
                    LevelReclaimed(
                        timestamp=bar_close_ts,
                        symbol=self.symbol,
                        level_name=name,  # type: ignore[arg-type]
                        level_price=level,
                        direction="below_to_above",
                    )
                )
            state.last_close_side = this_side
            if this_side == "above":
                state.ever_broken_above = True
            else:
                state.ever_broken_below = True

    def _maybe_emit_data_warning(
        self,
        name: str,
        reason: str,
        bar_close_ts: datetime,
        events: list[Event],
    ) -> None:
        if name in self._data_warnings_emitted:
            return
        self._data_warnings_emitted.add(name)
        events.append(
            LevelDataUnavailable(
                timestamp=bar_close_ts,
                symbol=self.symbol,
                level_name=name,
                reason=reason,
            )
        )

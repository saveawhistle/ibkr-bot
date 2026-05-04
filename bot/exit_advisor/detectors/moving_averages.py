"""Moving-average cross detectors: VWAP and 9 EMA.

Each MA tracks its own value and the side (above/below/unknown) of the
prior bar's close relative to that value. A cross emits ``MovingAverageCross``
on the bar where the side flips.

VWAP comes from ``BarHistory.session_vwap()`` (cumulative session VWAP).
9 EMA is computed inline here: simple average of the first 9 bars seeds
the value, then standard EMA from bar 10 onward (smoothing factor
``2 / (N + 1) = 0.2`` for N=9).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from bot.exit_advisor.core.events import Event, MovingAverageCross
from bot.exit_advisor.replay.bar_history import BarHistory
from bot.exit_advisor.replay.replay_source import Bar

EMA_PERIOD = 9
EMA_ALPHA = 2.0 / (EMA_PERIOD + 1)


@dataclass
class _MaState:
    last_side: Literal["above", "below", "unknown"] = "unknown"


@dataclass
class MovingAveragesDetector:
    symbol: str
    vwap_enabled: bool = True
    ema_9_enabled: bool = True

    _vwap_state: _MaState = field(default_factory=_MaState)
    _ema_state: _MaState = field(default_factory=_MaState)
    _ema_value: float | None = None
    _ema_seed_closes: list[float] = field(default_factory=list)

    def on_bar(self, bar: Bar, history: BarHistory) -> list[Event]:
        bar_close_ts = bar.timestamp + timedelta(minutes=1)
        events: list[Event] = []

        if self.vwap_enabled:
            vwap = history.session_vwap()
            if vwap is not None:
                self._maybe_fire_cross(
                    name="vwap",
                    ma_value=vwap,
                    bar_close=bar.close,
                    state=self._vwap_state,
                    bar_close_ts=bar_close_ts,
                    events=events,
                )

        if self.ema_9_enabled:
            ema = self._update_ema(bar.close)
            if ema is not None:
                self._maybe_fire_cross(
                    name="ema_9",
                    ma_value=ema,
                    bar_close=bar.close,
                    state=self._ema_state,
                    bar_close_ts=bar_close_ts,
                    events=events,
                )

        return events

    def vwap_value(self, history: BarHistory) -> float | None:
        return history.session_vwap()

    def ema_9_value(self) -> float | None:
        return self._ema_value

    # --- internals ---

    def _update_ema(self, close: float) -> float | None:
        if self._ema_value is None:
            self._ema_seed_closes.append(close)
            if len(self._ema_seed_closes) < EMA_PERIOD:
                return None
            self._ema_value = sum(self._ema_seed_closes) / EMA_PERIOD
            return self._ema_value
        self._ema_value = EMA_ALPHA * close + (1 - EMA_ALPHA) * self._ema_value
        return self._ema_value

    def _maybe_fire_cross(
        self,
        *,
        name: Literal["vwap", "ema_9"],
        ma_value: float,
        bar_close: float,
        state: _MaState,
        bar_close_ts: datetime,
        events: list[Event],
    ) -> None:
        if bar_close > ma_value:
            this_side: Literal["above", "below"] = "above"
        elif bar_close < ma_value:
            this_side = "below"
        else:
            return

        prior_side = state.last_side
        if prior_side == "unknown":
            state.last_side = this_side
            return

        if prior_side != this_side:
            direction: Literal["price_above_to_below", "price_below_to_above"] = (
                "price_above_to_below" if this_side == "below" else "price_below_to_above"
            )
            events.append(
                MovingAverageCross(
                    timestamp=bar_close_ts,
                    symbol=self.symbol,
                    ma_name=name,
                    ma_value=ma_value,
                    direction=direction,
                    bar_close=bar_close,
                )
            )
        state.last_side = this_side

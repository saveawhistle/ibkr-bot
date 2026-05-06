"""Bar-shape detectors: named patterns + WickEvent + ConsecutiveBars.

Pattern definitions (these are the canonical thresholds — six months from
now when someone wants to tune the doji body cutoff, the definition is
right here):

- doji: body / range < 0.10. body = abs(close - open). range = high - low.
- hammer: lower_wick > 2 * body AND upper_wick < body AND body's midpoint
  is in the UPPER 30% of the range. lower_wick = min(open, close) - low;
  upper_wick = high - max(open, close). NOTE: layer 2 spec said "lower
  30%" but a hammer has the body at the top — the long lower wick pulled
  price down and price recovered, leaving body up high. We use the
  canonical (upper 30%) definition; the spec wording was inverted.
- shooting_star: upper_wick > 2 * body AND lower_wick < body AND body's
  midpoint is in the LOWER 30% of the range. Same canonical-vs-spec note
  as hammer — spec said "upper 30%", canonical is lower 30%.
- engulfing: requires the prior bar. Current bar's body fully engulfs
  the prior bar's body (current open is on the far side of prior close,
  current close is on the far side of prior open) AND the two bars are
  opposite colors. Direction-agnostic — ``BarShapeDetected`` carries
  the OHLC so post-hoc analysis can determine bullish/bearish.
- inside_bar: current.high < prior.high AND current.low > prior.low.
- outside_bar: current.high > prior.high AND current.low < prior.low.

WickEvent fires when ``wick_size / total_range >= wick_threshold_pct`` on
either side. Both sides can fire on the same bar (long upper AND lower
wick). Ratio guards against zero-range bars.

ConsecutiveBars tracks a running count of same-direction (green/red)
closes and emits every bar past the threshold, with the updated count.
Bars where close == open break the streak (treated as "no direction").
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from bot.exit_advisor.core.events import BarShapeDetected, ConsecutiveBars, Event, WickEvent
from bot.exit_advisor.replay.bar_history import BarHistory
from bot.exit_advisor.replay.replay_source import Bar

DOJI_BODY_RATIO = 0.10
HAMMER_STAR_BODY_ZONE_PCT = 0.30


@dataclass
class BarShapeDetector:
    symbol: str
    enabled_shapes: tuple[str, ...] = (
        "doji",
        "hammer",
        "shooting_star",
        "engulfing",
        "inside_bar",
        "outside_bar",
    )
    wick_threshold_pct: float = 0.6
    consecutive_bars_threshold: int = 3

    _prior_bar: Bar | None = None
    _streak_direction: Literal["green", "red", "none"] = "none"
    _streak_count: int = 0

    def on_bar(self, bar: Bar, history: BarHistory) -> list[Event]:
        bar_close_ts = bar.timestamp + timedelta(minutes=1)
        events: list[Event] = []

        rng = bar.high - bar.low
        body = abs(bar.close - bar.open)
        upper_wick = bar.high - max(bar.open, bar.close)
        lower_wick = min(bar.open, bar.close) - bar.low

        if rng > 0:
            # --- shape detection ---
            if "doji" in self.enabled_shapes and body / rng < DOJI_BODY_RATIO:
                events.append(self._shape_event(bar, bar_close_ts, "doji"))

            body_midpoint = (bar.open + bar.close) / 2
            body_zone_lo = bar.low + rng * HAMMER_STAR_BODY_ZONE_PCT
            body_zone_hi = bar.high - rng * HAMMER_STAR_BODY_ZONE_PCT

            if (
                "hammer" in self.enabled_shapes
                and body > 0
                and lower_wick > 2 * body
                and upper_wick < body
                and body_midpoint >= body_zone_hi
            ):
                events.append(self._shape_event(bar, bar_close_ts, "hammer"))

            if (
                "shooting_star" in self.enabled_shapes
                and body > 0
                and upper_wick > 2 * body
                and lower_wick < body
                and body_midpoint <= body_zone_lo
            ):
                events.append(self._shape_event(bar, bar_close_ts, "shooting_star"))

            if self._prior_bar is not None:
                pb = self._prior_bar
                if "engulfing" in self.enabled_shapes and self._is_engulfing(pb, bar):
                    events.append(self._shape_event(bar, bar_close_ts, "engulfing"))
                if "inside_bar" in self.enabled_shapes and bar.high < pb.high and bar.low > pb.low:
                    events.append(self._shape_event(bar, bar_close_ts, "inside_bar"))
                if "outside_bar" in self.enabled_shapes and bar.high > pb.high and bar.low < pb.low:
                    events.append(self._shape_event(bar, bar_close_ts, "outside_bar"))

            # --- wick events (separate from named shapes) ---
            for side, wick in (("upper", upper_wick), ("lower", lower_wick)):
                ratio = wick / rng
                if ratio >= self.wick_threshold_pct:
                    events.append(
                        WickEvent(
                            timestamp=bar_close_ts,
                            symbol=self.symbol,
                            wick_side=side,  # type: ignore[arg-type]
                            wick_size=wick,
                            body_size=body,
                            total_range=rng,
                            wick_ratio=ratio,
                        )
                    )

        # --- consecutive-bar streak ---
        if bar.close > bar.open:
            self._extend_streak("green", events, bar_close_ts)
        elif bar.close < bar.open:
            self._extend_streak("red", events, bar_close_ts)
        else:
            self._streak_direction = "none"
            self._streak_count = 0

        self._prior_bar = bar
        return events

    # --- internals ---

    def _shape_event(self, bar: Bar, ts: datetime, shape: str) -> BarShapeDetected:
        return BarShapeDetected(
            timestamp=ts,
            symbol=self.symbol,
            shape=shape,  # type: ignore[arg-type]
            bar_open=bar.open,
            bar_high=bar.high,
            bar_low=bar.low,
            bar_close=bar.close,
        )

    @staticmethod
    def _is_engulfing(prior: Bar, current: Bar) -> bool:
        prior_green = prior.close > prior.open
        current_green = current.close > current.open
        if prior_green == current_green or prior.open == prior.close:
            return False
        if current_green:
            return current.open <= prior.close and current.close >= prior.open
        return current.open >= prior.close and current.close <= prior.open

    def _extend_streak(
        self,
        direction: Literal["green", "red"],
        events: list[Event],
        bar_close_ts: datetime,
    ) -> None:
        if self._streak_direction == direction:
            self._streak_count += 1
        else:
            self._streak_direction = direction
            self._streak_count = 1

        if self._streak_count >= self.consecutive_bars_threshold:
            events.append(
                ConsecutiveBars(
                    timestamp=bar_close_ts,
                    symbol=self.symbol,
                    direction=direction,
                    count=self._streak_count,
                )
            )

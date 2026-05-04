"""Momentum / bull-flag detector for the 9:30–11:30 ET window.

Setup: after the opening impulse, the stock consolidates in a shallow flag
(pullback within ``flag_max_pullback_pct`` off the flag high) and then breaks
the high of day on the latest bar. Entry = breakout bar close, stop = flag
low, target = ``scale_out_multiple × initial_risk`` (default 2R — the 2:1 R:R rule).
"""

from __future__ import annotations

from datetime import datetime, time
from typing import cast

import pandas as pd
import structlog

from bot.indicators import evaluate_extension, evaluate_hod, is_bull_flag, vwap
from bot.strategies.base import (
    _PMH_CAP_TICK,
    Signal,
    Strategy,
    _apply_premarket_high_cap,
    _apply_stop_distance_floor,
)

_WINDOW_START = time(9, 30)
_FLAG_LOOKBACK = 10

_log = structlog.get_logger("bot.strategies.momentum")


class MomentumStrategy(Strategy):
    """HOD breakout out of a shallow bull flag during the momentum window.

    Phase 6.6: ``extended_from_vwap_atr_multiple`` is the configured ATR-multiple
    threshold for the VWAP-distance extension check (default 5.0 — calibrated
    against Day 3 paper trading where 3.0 rejected normal continuation setups).
    ``log_extension_check_passes`` toggles a per-bar
    ``strategy.extension_check_passed`` log so an operator can grep
    ``extension_ratio`` values across a session for further calibration.

    Phase 6.7: window end (``window_end``) is configurable — was hardcoded
    at 11:30 ET. Start remains 09:30 ET (market open).
    """

    name = "momentum"

    def __init__(
        self,
        flag_max_pullback_pct: float = 5.0,
        scale_out_multiple: float = 2.0,
        extended_from_vwap_atr_multiple: float = 5.0,
        log_extension_check_passes: bool = False,
        window_end: time = time(11, 30),
        premarket_high_cap_enabled: bool = True,
        stop_floor_min_abs: float = 0.05,
        stop_floor_min_pct: float = 0.02,
    ) -> None:
        """Store pullback envelope, scale-out R-multiple, extension config, window end.

        Phase 6.7 ``window_end`` replaces the hardcoded 11:30 ET cutoff so
        off-hours testing can widen the evaluation window without code
        edits. Bars with ``ts.time() >= window_end`` silently short-circuit.

        Phase 8.1: ``rr_min`` parameter removed — R:R is pinned to
        ``scale_out_multiple`` by construction. See Strategy-base docstring.

        Phase 8.4: ``premarket_high_cap_enabled`` (default True) caps the
        scale-out at PMH−$0.01 when entry sits below PMH. See
        ``_apply_premarket_high_cap`` for semantics.

        Phase 10.2: ``stop_floor_min_abs`` (default 5¢) and
        ``stop_floor_min_pct`` (default 2%) floor the stop distance to
        ``max(min_abs, entry × min_pct)``. Defends against tight
        consolidation breakouts (2026-04-30 ZENA: $2.18 entry, $2.17
        structural stop, 1¢ risk that microstructure noise would tag
        immediately).
        """
        super().__init__(scale_out_multiple=scale_out_multiple)
        self.flag_max_pullback_pct = flag_max_pullback_pct
        self.extended_from_vwap_atr_multiple = extended_from_vwap_atr_multiple
        self.log_extension_check_passes = log_extension_check_passes
        self.window_end = window_end
        self.premarket_high_cap_enabled = premarket_high_cap_enabled
        self.stop_floor_min_abs = stop_floor_min_abs
        self.stop_floor_min_pct = stop_floor_min_pct

    def evaluate(self, symbol: str, bars: pd.DataFrame) -> Signal | None:
        """Emit a Signal if the latest bar breaks HOD out of a bull flag."""
        if bars.empty:
            return None
        last_ts = bars.index[-1]
        # Window check first — outside-window bars short-circuit silently.
        if not self._within_window(last_ts):
            return None

        if len(bars) < _FLAG_LOOKBACK:
            return self._reject(
                symbol,
                last_ts,
                "setup",
                "insufficient_bars",
                bars_len=len(bars),
                required=_FLAG_LOOKBACK,
            )

        # Phase 7.2: HOD check resets at 09:30 ET so premarket wicks (AUUD
        # Day 4: $10+ premarket prints blocked 37 consecutive bars) no
        # longer contaminate the running max. Phase 9.1: ``by="close"`` so
        # a wick-and-retrace bar is rejected as a failed breakout instead
        # of confirming entry on the wick. ``hod.last_high`` /
        # ``session_hod`` stay high-based so rejection events show both
        # the wick and the close that failed.
        last_close = float(bars["close"].iloc[-1])
        hod = evaluate_hod(bars, by="close")
        if hod is None or not hod.is_new_hod:
            return self._reject(
                symbol,
                last_ts,
                "entry_trigger",
                "not_new_hod",
                last_high=hod.last_high if hod else None,
                last_close=last_close,
                session_hod=hod.session_hod if hod else None,
                bars_in_session=hod.bars_in_session if hod else 0,
            )
        if not is_bull_flag(
            bars,
            max_pullback_pct=self.flag_max_pullback_pct,
            lookback=_FLAG_LOOKBACK,
        ):
            return self._reject(
                symbol,
                last_ts,
                "setup",
                "no_bull_flag",
                max_pullback_pct=self.flag_max_pullback_pct,
                lookback=_FLAG_LOOKBACK,
            )

        vwap_series = vwap(bars)
        if vwap_series.empty:
            return self._reject(symbol, last_ts, "setup", "vwap_unavailable")
        # Momentum intentionally applies VWAP extension check at all bars (no grace period).
        # Phase 5.5 added grace period to gap_and_go only, because gap-and-go's ORB entry
        # legitimately trades extended-from-VWAP setups in the first 15 minutes.
        # Momentum is an ongoing-intraday pattern where extension indicates a stock that
        # has already run too far and should not be chased.
        last_vwap = float(vwap_series.iloc[-1])
        check = evaluate_extension(
            bars, vwap_series, atr_multiple=self.extended_from_vwap_atr_multiple
        )
        if check.extended:
            return self._reject(
                symbol,
                last_ts,
                "entry_trigger",
                "extended_from_vwap",
                last_close=last_close,
                vwap=last_vwap,
                last_atr_value=check.last_atr_value,
                atr_multiple=self.extended_from_vwap_atr_multiple,
                distance_from_vwap=check.distance_from_vwap,
                threshold_distance=check.threshold_distance,
                extension_ratio=check.extension_ratio,
            )
        if self.log_extension_check_passes:
            _log.info(
                "strategy.extension_check_passed",
                symbol=symbol,
                strategy=self.name,
                bar_time=last_ts.isoformat(),
                last_close=last_close,
                vwap=last_vwap,
                last_atr_value=check.last_atr_value,
                atr_multiple=self.extended_from_vwap_atr_multiple,
                distance_from_vwap=check.distance_from_vwap,
                threshold_distance=check.threshold_distance,
                extension_ratio=check.extension_ratio,
            )

        # Phase 7.1: restrict the flag window to market-hours bars on the bar's
        # local date. Pre-fix, `iloc[-_FLAG_LOOKBACK:]` could dip into premarket
        # backfill when evaluating within the first _FLAG_LOOKBACK minutes of
        # RTH — a 1 AM wick would anchor `flag_low` and widen the stop.
        last_date = last_ts.date()
        index = cast("pd.DatetimeIndex", bars.index)
        market_open = pd.Timestamp(datetime.combine(last_date, time(9, 30)))
        if index.tz is not None:
            market_open = market_open.tz_localize(index.tz)
        session = bars.loc[bars.index >= market_open]
        if session.empty:
            return self._reject(symbol, last_ts, "stop_calculation", "no_market_hours_bars")
        flag_window = session.iloc[-_FLAG_LOOKBACK:]
        bars_available_for_lookback = len(flag_window)
        flag_low = float(flag_window["low"].min())
        entry = float(bars["close"].iloc[-1])
        stop = flag_low
        risk = entry - stop
        if risk <= 0:
            return self._reject(
                symbol,
                last_ts,
                "stop_calculation",
                "nonpositive_risk",
                entry=entry,
                stop=stop,
                pullback_low=flag_low,
                pullback_lookback_bars=_FLAG_LOOKBACK,
                bars_available_for_lookback=bars_available_for_lookback,
                vwap_at_entry=last_vwap,
            )
        # Phase 10.2 — minimum stop-distance floor. Applies after the
        # nonpositive_risk screen so broken setups still reject; rescues
        # only the "tight but valid" case (1-2¢ structural risk, ZENA
        # 2026-04-30 precedent). The structural value (``flag_low``)
        # remains in the ``signal.emitted`` log for forensics; ``stop``
        # is the floored value end-to-end so risk sizing, fill-anchored
        # re-protection, and the Phase 10.1 watchdog all see one number.
        stop = _apply_stop_distance_floor(
            entry=entry,
            structural_stop=stop,
            floor_min_abs=self.stop_floor_min_abs,
            floor_min_pct=self.stop_floor_min_pct,
            symbol=symbol,
            strategy=self.name,
            bar_time=last_ts,
        )
        risk = entry - stop
        # Phase 4i: scale-out = entry + scale_out_multiple × initial_risk
        # (default 2.0 — the 2:1 R:R rule). Runner ceiling is left None here;
        # the executor populates it only when runner_target_enabled is true.
        default_scale_out = entry + self.scale_out_multiple * risk
        scale_out, scale_out_cap_reason, capped_target = _apply_premarket_high_cap(
            entry=entry,
            default_scale_out=default_scale_out,
            bars=bars,
            enabled=self.premarket_high_cap_enabled,
        )
        if scale_out_cap_reason == "premarket_high":
            _log.info(
                "strategy.scale_out_capped_premarket_high",
                symbol=symbol,
                strategy=self.name,
                bar_time=last_ts.isoformat(),
                entry=entry,
                default_scale_out=round(default_scale_out, 4),
                capped_scale_out=round(scale_out, 4),
                premarket_high=round(capped_target + _PMH_CAP_TICK, 4),
            )

        # Phase 7.1: observability — see gap_and_go for rationale. Emitted so
        # an operator can correlate the strategy's stop reference with any
        # downstream stop_too_wide rejection in the risk engine.
        _log.info(
            "signal.emitted",
            symbol=symbol,
            strategy=self.name,
            bar_time=last_ts.isoformat(),
            entry=entry,
            stop=stop,
            pullback_low=flag_low,
            pullback_lookback_bars=_FLAG_LOOKBACK,
            bars_available_for_lookback=bars_available_for_lookback,
            vwap_at_entry=last_vwap,
        )
        return Signal(
            symbol=symbol,
            strategy=self.name,
            entry=round(entry, 4),
            stop=round(stop, 4),
            scale_out_price=round(scale_out, 4),
            runner_target_price=None,
            timestamp=last_ts.to_pydatetime(),
            reasons=["bull_flag", "hod_break"],
            recent_bar_volume=_recent_volume(bars),
            pullback_low=flag_low,
            pullback_lookback_bars=_FLAG_LOOKBACK,
            bars_available_for_lookback=bars_available_for_lookback,
            vwap_at_entry=last_vwap,
        )

    def _within_window(self, ts: pd.Timestamp) -> bool:
        """True iff ``ts`` (NY-local) sits in the 09:30–``window_end`` momentum window.

        Phase 6.7 — end is per-instance so tests / operator can widen the
        window at config time without touching the module constant.
        """
        local = ts.time()
        return bool(_WINDOW_START <= local < self.window_end)


def _recent_volume(bars: pd.DataFrame) -> int | None:
    """Best-effort int cast of the latest bar's volume; None on missing column."""
    if "volume" not in bars.columns:
        return None
    raw = bars["volume"].iloc[-1]
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None

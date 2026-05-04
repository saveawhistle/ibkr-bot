"""Gap-and-Go detector: first 30 minutes of RTH, VWAP reclaim + shallow pullback entry.

Setup: stock gapped up (pre-scan filter), opens near HOD, holds above VWAP on
the first 1-min close, then pulls back to (but not through) VWAP on low-range
consolidation. Entry triggers when the next bar pushes a new high over the
pullback consolidation. Stop = VWAP (or the pullback low, whichever is lower).
Target = ``scale_out_multiple × initial_risk`` (default 2R — the 2:1 R:R rule).
"""

from __future__ import annotations

from datetime import datetime, time
from typing import cast

import pandas as pd
import structlog

from bot.indicators import atr, evaluate_extension, evaluate_hod, vwap
from bot.strategies.base import (
    _PMH_CAP_TICK,
    Signal,
    Strategy,
    _apply_premarket_high_cap,
    _apply_stop_distance_floor,
)

_WINDOW_START = time(9, 30)
_MIN_BARS = 3
# Phase 7.1: recent-N-bar pullback low. Session minimum was wrong — a wick or
# premarket print pollutes the reference for the rest of the session. 3 bars
# matches the published "stop below recent consolidation" methodology.
PULLBACK_LOOKBACK_BARS = 3

_log = structlog.get_logger("bot.strategies.gap_and_go")


class GapAndGoStrategy(Strategy):
    """First-pullback entry above reclaimed VWAP during the opening window.

    Phase 5.5: the ``extended_from_vwap`` rejection is bypassed for bars stamped
    within ``vwap_extension_grace_minutes`` of ``trading_start``. During the
    grace period the ORB entry is the gap-and-go entry and the
    VWAP-distance filter is context only. After the grace period the 3× ATR
    check applies as before. ``vwap_extension_grace_minutes=0`` restores
    pre-5.5 behaviour.

    Phase 6.7: window end (``window_end``) is now configurable — was
    hardcoded at 10:00 ET. Start is always 09:30 ET (market open).
    """

    name = "gap_and_go"

    def __init__(
        self,
        scale_out_multiple: float = 2.0,
        vwap_extension_grace_minutes: int = 15,
        trading_start: time = time(9, 30),
        extended_from_vwap_atr_multiple: float = 5.0,
        log_extension_check_passes: bool = False,
        window_end: time = time(10, 0),
        premarket_high_cap_enabled: bool = True,
        stop_floor_min_abs: float = 0.05,
        stop_floor_min_pct: float = 0.02,
    ) -> None:
        """Store scale-out multiple, grace window, extension config, window end.

        ``trading_start`` is the NY-local session open (defaults to 09:30) used
        as the grace-period anchor; the bot measures grace from this wall-clock
        time, not from when the process started.

        Phase 6.6 ``extended_from_vwap_atr_multiple`` is the configured ATR
        multiple for the post-grace extension check (default 5.0; 3.0 was
        structurally too tight for low-float gappers per Day 3 calibration).
        ``log_extension_check_passes`` toggles a per-bar
        ``strategy.extension_check_passed`` log for calibration sweeps.

        Phase 6.7 ``window_end`` replaces the hardcoded 10:00 ET cutoff so
        off-hours testing can widen the evaluation window without code
        edits. Bars with ``ts.time() >= window_end`` silently short-circuit.

        Phase 8.1: ``rr_min`` parameter removed. R:R is pinned to
        ``scale_out_multiple`` by the emission formula, so the floor check
        was tautological. See Strategy-base-class docstring.

        Phase 10.2: ``stop_floor_min_abs`` (default 5¢) and
        ``stop_floor_min_pct`` (default 2%) floor the stop distance to
        ``max(min_abs, entry × min_pct)``. Same pathology as momentum —
        a structural ``min(VWAP, 3-bar pullback low)`` can land 1-2¢
        below entry on tight consolidation breakouts and get tagged on
        microstructure noise.
        """
        super().__init__(scale_out_multiple=scale_out_multiple)
        self.vwap_extension_grace_minutes = vwap_extension_grace_minutes
        self.trading_start = trading_start
        self.extended_from_vwap_atr_multiple = extended_from_vwap_atr_multiple
        self.log_extension_check_passes = log_extension_check_passes
        self.window_end = window_end
        self.premarket_high_cap_enabled = premarket_high_cap_enabled
        self.stop_floor_min_abs = stop_floor_min_abs
        self.stop_floor_min_pct = stop_floor_min_pct

    def evaluate(self, symbol: str, bars: pd.DataFrame) -> Signal | None:
        """Emit a Signal if the latest bar triggers a gap-and-go entry, else None."""
        if bars.empty:
            return None
        last_ts = bars.index[-1]
        # Window check comes first so outside-window bars short-circuit silently.
        if not self._within_window(last_ts):
            return None

        if len(bars) < _MIN_BARS:
            return self._reject(
                symbol,
                last_ts,
                "setup",
                "insufficient_bars",
                bars_len=len(bars),
                required=_MIN_BARS,
            )

        vwap_series = vwap(bars)
        if vwap_series.empty:
            return self._reject(symbol, last_ts, "setup", "vwap_unavailable")

        last_close = float(bars["close"].iloc[-1])
        last_vwap = float(vwap_series.iloc[-1])
        if last_close <= last_vwap:
            return self._reject(
                symbol,
                last_ts,
                "entry_trigger",
                "below_vwap",
                last_close=last_close,
                vwap=last_vwap,
            )

        # Phase 5.5: bypass the extension check during the grace window
        # (default 15 min from trading_start). ORB entries legitimately trade
        # extended-from-VWAP in the first minutes; after the grace window the
        # ATR check applies normally. Momentum intentionally does NOT get
        # this grace — see bot/strategies/momentum.py for the rationale.
        # Phase 6.6: post-grace check uses configured multiple (default 5×)
        # and emits the same enriched diagnostic fields as momentum.
        minutes_since_open = self._minutes_since_open(last_ts)
        in_grace = (
            self.vwap_extension_grace_minutes > 0
            and 0 <= minutes_since_open < self.vwap_extension_grace_minutes
        )
        if in_grace:
            atr_series = atr(bars)
            last_atr = float(atr_series.iloc[-1]) if not atr_series.empty else None
            _log.info(
                "gap_and_go.vwap_extension_bypassed",
                symbol=symbol,
                bar_time=last_ts.isoformat(),
                minutes_since_open=minutes_since_open,
                last_close=last_close,
                vwap=last_vwap,
                atr=last_atr,
            )
        else:
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

        # Phase 7.2: HOD check resets at 09:30 ET (premarket wicks no longer
        # contaminate the market-hours running max). Phase 9.1: ``by="close"``
        # so a wick-and-retrace bar (high above HOD, close back below) is
        # correctly rejected as a failed breakout — RMAX 2026-04-27 09:34
        # entered on exactly that pattern. ``hod.last_high`` / ``session_hod``
        # remain high-based so the rejection event shows both the wick that
        # made HOD and the close that failed to confirm.
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

        # Phase 7.1: restrict stop reference to market-hours bars (>= 09:30 ET on
        # the bar's local date). IBKR backfills premarket bars with the same
        # calendar date, so the previous same-date filter leaked overnight wicks
        # into the pullback reference; a 1 AM $7.87 print polluted TZOO's stop
        # across the entire session on Day 4. `last_ts` is tz-aware NY-local.
        last_date = last_ts.date()
        index = cast("pd.DatetimeIndex", bars.index)
        market_open = pd.Timestamp(datetime.combine(last_date, time(9, 30)))
        if index.tz is not None:
            market_open = market_open.tz_localize(index.tz)
        session = bars.loc[bars.index >= market_open]
        if session.empty:
            return self._reject(symbol, last_ts, "stop_calculation", "no_market_hours_bars")
        # Phase 7.1: pullback reference is the recent N-bar minimum, not the
        # session minimum — session-min accumulated the worst wick of the day
        # and permanently widened the stop. If fewer than N bars are available
        # (e.g. 9:31 evaluation with 1 completed bar), use whatever we have.
        recent_bars = session.iloc[-PULLBACK_LOOKBACK_BARS:]
        bars_available_for_lookback = len(recent_bars)
        pullback_low = float(recent_bars["low"].min())
        stop = float(min(last_vwap, pullback_low))
        entry = last_close
        risk = entry - stop
        if risk <= 0:
            return self._reject(
                symbol,
                last_ts,
                "stop_calculation",
                "nonpositive_risk",
                entry=entry,
                stop=stop,
                pullback_low=pullback_low,
                pullback_lookback_bars=PULLBACK_LOOKBACK_BARS,
                bars_available_for_lookback=bars_available_for_lookback,
                vwap_at_entry=last_vwap,
            )
        # Phase 10.2 — minimum stop-distance floor. See momentum.py for
        # rationale. ``stop`` becomes the floored value end-to-end;
        # ``pullback_low`` keeps the structural reference for forensics.
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
        # Phase 4i: scale-out anchor = entry + scale_out_multiple × initial_risk
        # (default 2.0 — the 2:1 R:R rule). ``runner_target_price`` is left None
        # here: the executor populates it only when
        # ``execution.runner_target_enabled`` is true.
        default_scale_out = entry + self.scale_out_multiple * risk
        scale_out, scale_out_cap_reason, capped_target = _apply_premarket_high_cap(
            entry=entry,
            default_scale_out=default_scale_out,
            bars=bars,
            enabled=self.premarket_high_cap_enabled,
        )

        reasons = ["vwap_hold", "new_hod"]
        if scale_out_cap_reason == "premarket_high":
            reasons.append("scale_out_capped_premarket_high")
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
        # Phase 7.1: observability for the stop calculation. Emitted so a
        # downstream `stop_too_wide` rejection can be correlated with the
        # strategy's reference values (volatility vs. wick vs. bad pick).
        _log.info(
            "signal.emitted",
            symbol=symbol,
            strategy=self.name,
            bar_time=last_ts.isoformat(),
            entry=entry,
            stop=stop,
            pullback_low=pullback_low,
            pullback_lookback_bars=PULLBACK_LOOKBACK_BARS,
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
            reasons=reasons,
            recent_bar_volume=_recent_volume(bars),
            pullback_low=pullback_low,
            pullback_lookback_bars=PULLBACK_LOOKBACK_BARS,
            bars_available_for_lookback=bars_available_for_lookback,
            vwap_at_entry=last_vwap,
        )

    def _minutes_since_open(self, ts: pd.Timestamp) -> int:
        """Minutes between ``ts`` (NY-local) and ``trading_start`` on the same date.

        Returns a negative number if the bar stamp is before ``trading_start``;
        callers must treat those bars as outside the grace window (they also
        fail ``_within_window`` and won't reach the extension check).
        """
        start = self.trading_start
        delta_min = (ts.hour - start.hour) * 60 + (ts.minute - start.minute)
        return int(delta_min)

    def _within_window(self, ts: pd.Timestamp) -> bool:
        """True iff ``ts`` (NY-local) sits in the 09:30–``window_end`` window.

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

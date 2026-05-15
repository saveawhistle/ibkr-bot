"""Momentum strategy: Bull Flag, Micro Pullback, and Flat Top Breakout detection.

Phase 14 overhaul: replaces the single is_bull_flag() heuristic with the
richer analyze_momentum_pattern() detector that covers all three of Cameron's
published entry setups and corrects six structural gaps:

1. Red-candle count enforced (was absent)
2. Pullback depth now pole-relative (< 50% of impulse pole, was % of peak)
3. VWAP hold verified on every consolidation bar (was unchecked)
4. Breakout bar volume compared to consolidation avg (was rolling 20-bar window)
5. Stop scoped to consolidation bars only (was 10-bar RTH minimum, included impulse)
6. Standing STP_LMT order emitted when close < resistance (was close-only entry)
"""

from __future__ import annotations

from datetime import datetime, time
from typing import cast

import pandas as pd
import structlog

from bot.config import EntryQualityConfig
from bot.indicators import (
    MomentumPattern,
    analyze_momentum_pattern,
    evaluate_extension,
    evaluate_hod,
    vwap,
)
from bot.strategies.base import (
    _PMH_CAP_TICK,
    Signal,
    Strategy,
    _apply_premarket_high_cap,
    _apply_stop_distance_floor,
)
from bot.strategies.entry_quality import (
    check_breakout_volume_ratio,
    check_consolidation_tightness,
    check_consolidation_vwap_hold,
    check_halt_detection,
    check_impulse_strength,
    check_volume_contraction,
    check_vwap_extension,
)
from bot.strategies.volume import check_recent_window_rvol

_DEFAULT_WINDOW_START = time(10, 0)
"""Phase 12.6 — default momentum window start. Pre-12.6 was hardcoded at
09:30; raised to 10:00 so the strategy non-overlaps gap-and-go's default
opening window. Operators can still set it to any value >= 09:30 via
``strategies.momentum.window_start``."""

_log = structlog.get_logger("bot.strategies.momentum")


class MomentumStrategy(Strategy):
    """Bull Flag, Micro Pullback, and Flat Top Breakout during the momentum window.

    Phase 14: replaces the single is_bull_flag() heuristic with
    analyze_momentum_pattern(), which detects all three of Cameron's published
    entry setups and corrects structural gaps in red-candle count, pole-relative
    pullback depth, VWAP hold during consolidation, breakout volume confirmation,
    stop scoping, and standing STP_LMT pre-breakout orders.
    """

    name = "momentum"

    def __init__(
        self,
        flag_max_pullback_pct: float = 5.0,
        scale_out_multiple: float = 2.0,
        extended_from_vwap_atr_multiple: float = 5.0,
        log_extension_check_passes: bool = False,
        window_start: time | None = None,
        window_end: time = time(11, 30),
        premarket_high_cap_enabled: bool = True,
        stop_floor_min_abs: float = 0.05,
        stop_floor_min_pct: float = 0.02,
        catalyst_required: bool = False,
        recent_rvol_min: float = 2.0,
        recent_rvol_window_bars: int = 20,
        entry_quality: EntryQualityConfig | None = None,
        bull_flag_min_red_candles: int = 2,
        bull_flag_max_red_candles: int = 3,
        bull_flag_max_pullback_pct_of_pole: float = 0.50,
        micro_pullback_max_range_pct: float = 2.0,
        flat_top_max_high_range_pct: float = 0.5,
        breakout_volume_enabled: bool = True,
        breakout_volume_vs_consolidation_min_ratio: float = 1.5,
    ) -> None:
        """Store pattern-detection knobs, scale-out R-multiple, extension config, window.

        ``flag_max_pullback_pct`` is retained for backward-compatibility with
        existing test fixtures and the orchestrator's config pass-through, but
        is no longer used by evaluate() — the pole-relative
        ``bull_flag_max_pullback_pct_of_pole`` gates pullback depth instead.

        Phase 14 adds:
        * ``bull_flag_min/max_red_candles`` — 2-3 descending red candles required
        * ``bull_flag_max_pullback_pct_of_pole`` — pullback < 50% of impulse pole
        * ``micro_pullback_max_range_pct`` — total consolidation range < 2%
        * ``flat_top_max_high_range_pct`` — consolidation highs cluster within 0.5%
        * ``breakout_volume_enabled`` / ``breakout_volume_vs_consolidation_min_ratio``
          — breakout bar must surge vs. consolidation avg (breakout path only)
        """
        super().__init__(scale_out_multiple=scale_out_multiple)
        self.flag_max_pullback_pct = flag_max_pullback_pct  # legacy, unused
        self.extended_from_vwap_atr_multiple = extended_from_vwap_atr_multiple
        self.log_extension_check_passes = log_extension_check_passes
        self.window_start = window_start if window_start is not None else _DEFAULT_WINDOW_START
        self.window_end = window_end
        self.premarket_high_cap_enabled = premarket_high_cap_enabled
        self.stop_floor_min_abs = stop_floor_min_abs
        self.stop_floor_min_pct = stop_floor_min_pct
        self.catalyst_required = catalyst_required
        self.recent_rvol_min = recent_rvol_min
        self.recent_rvol_window_bars = recent_rvol_window_bars
        self.entry_quality = entry_quality if entry_quality is not None else EntryQualityConfig()
        self.bull_flag_min_red_candles = bull_flag_min_red_candles
        self.bull_flag_max_red_candles = bull_flag_max_red_candles
        self.bull_flag_max_pullback_pct_of_pole = bull_flag_max_pullback_pct_of_pole
        self.micro_pullback_max_range_pct = micro_pullback_max_range_pct
        self.flat_top_max_high_range_pct = flat_top_max_high_range_pct
        self.breakout_volume_enabled = breakout_volume_enabled
        self.breakout_volume_vs_consolidation_min_ratio = breakout_volume_vs_consolidation_min_ratio

    def evaluate(self, symbol: str, bars: pd.DataFrame) -> Signal | None:
        """Emit a Signal if the latest bar fires a valid momentum pattern."""
        if bars.empty:
            return None
        last_ts = bars.index[-1]
        # Window check first — outside-window bars short-circuit silently.
        if not self._within_window(last_ts):
            return None

        eq = self.entry_quality
        required = eq.impulse_window_bars + eq.consolidation_window_bars + 1
        if len(bars) < required:
            return self._reject(
                symbol,
                last_ts,
                "setup",
                "insufficient_bars",
                bars_len=len(bars),
                required=required,
            )

        vwap_series = vwap(bars)
        if vwap_series.empty:
            return self._reject(symbol, last_ts, "setup", "vwap_unavailable")

        # HOD context — determines mode (breakout vs. standing order).
        hod_close = evaluate_hod(bars, by="close")
        hod_high = evaluate_hod(bars, by="high")
        if hod_close is None or hod_high is None:
            return self._reject(symbol, last_ts, "setup", "no_hod_context")

        last_close = float(bars["close"].iloc[-1])
        trigger_level = hod_high.session_hod  # prior session HOD by high = resistance

        if trigger_level is None:
            # First market-hours bar of the session — nothing to exceed yet.
            return self._reject(
                symbol,
                last_ts,
                "entry_trigger",
                "first_session_bar",
                last_close=last_close,
            )

        is_breakout = hod_close.is_new_hod  # close > prior session HOD
        is_standing_order = not is_breakout and last_close < trigger_level

        if not is_breakout and not is_standing_order:
            # close == trigger_level exactly, or some degenerate edge case.
            return self._reject(
                symbol,
                last_ts,
                "entry_trigger",
                "not_new_hod",
                last_high=hod_close.last_high,
                last_close=last_close,
                session_hod=hod_high.session_hod,
                bars_in_session=hod_close.bars_in_session,
            )

        # Wick-and-retrace: high pierced resistance but close failed to confirm.
        # This looks like a standing order candidate but is actually a failed
        # breakout — the bar wicked through trigger then pulled back hard.
        if is_standing_order and hod_high.is_new_hod:
            return self._reject(
                symbol,
                last_ts,
                "entry_trigger",
                "not_new_hod",
                last_high=hod_high.last_high,
                last_close=last_close,
                session_hod=hod_high.session_hod,
                bars_in_session=hod_close.bars_in_session,
            )

        # Pattern analysis — covers all three entry setups.
        pattern: MomentumPattern | None = analyze_momentum_pattern(
            bars=bars,
            vwap_series=vwap_series,
            impulse_window_bars=eq.impulse_window_bars,
            consolidation_window_bars=eq.consolidation_window_bars,
            include_last_bar_in_consolidation=is_standing_order,
            bull_flag_min_red_candles=self.bull_flag_min_red_candles,
            bull_flag_max_red_candles=self.bull_flag_max_red_candles,
            bull_flag_max_pullback_pct_of_pole=self.bull_flag_max_pullback_pct_of_pole,
            micro_pullback_max_range_pct=self.micro_pullback_max_range_pct,
            flat_top_max_high_range_pct=self.flat_top_max_high_range_pct,
        )
        if pattern is None:
            return self._reject(
                symbol,
                last_ts,
                "setup",
                "no_momentum_pattern",
                is_breakout=is_breakout,
                is_standing_order=is_standing_order,
                last_close=last_close,
                trigger_level=round(trigger_level, 4),
            )

        # VWAP extension check at entry bar (no grace period for momentum).
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

        # VWAP hold during consolidation — all consolidation closes must be >= VWAP.
        if check_consolidation_vwap_hold(
            vwap_hold=pattern.vwap_hold,
            pattern_type=pattern.pattern_type,
            consolidation_low=pattern.consolidation_low,
            symbol=symbol,
            strategy=self.name,
            bar_time=last_ts,
        ):
            return None
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

        # Phase 13 entry-quality gates.
        if _apply_entry_quality_gates(
            cfg=eq,
            bars=bars,
            candidate_price=last_close,
            symbol=symbol,
            strategy=self.name,
            bar_time=last_ts,
        ):
            return None

        # Breakout volume gate (only in the breakout path — no surge to verify yet
        # when we're placing a standing order before the trigger fires).
        if is_breakout and self.breakout_volume_enabled:
            vol_result = check_breakout_volume_ratio(
                bars=bars,
                consolidation_window_bars=eq.consolidation_window_bars,
                min_ratio=self.breakout_volume_vs_consolidation_min_ratio,
                symbol=symbol,
                strategy=self.name,
                bar_time=last_ts,
            )
            if vol_result is not None:
                return None

        # Rolling RVOL check (both paths).
        suppression = check_recent_window_rvol(
            bars=bars,
            window_bars=self.recent_rvol_window_bars,
            threshold=self.recent_rvol_min,
            symbol=symbol,
            strategy=self.name,
            bar_time=last_ts,
        )
        if suppression is not None:
            return None

        # Entry and stop.
        entry = trigger_level if is_standing_order else last_close
        stop = pattern.consolidation_low  # scoped to consolidation bars only
        risk = entry - stop
        if risk <= 0:
            return self._reject(
                symbol,
                last_ts,
                "stop_calculation",
                "nonpositive_risk",
                entry=entry,
                stop=stop,
                pullback_low=pattern.consolidation_low,
                pullback_lookback_bars=eq.consolidation_window_bars,
                bars_available_for_lookback=eq.consolidation_window_bars,
                vwap_at_entry=last_vwap,
            )

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

        reasons = [pattern.pattern_type]
        if is_standing_order:
            reasons.append("standing_stp_lmt")
        else:
            reasons.append("hod_break")

        _log.info(
            "signal.emitted",
            symbol=symbol,
            strategy=self.name,
            bar_time=last_ts.isoformat(),
            entry=entry,
            stop=stop,
            pullback_low=pattern.consolidation_low,
            pullback_lookback_bars=eq.consolidation_window_bars,
            bars_available_for_lookback=eq.consolidation_window_bars,
            vwap_at_entry=last_vwap,
            pattern_type=pattern.pattern_type,
            is_standing_order=is_standing_order,
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
            pullback_low=pattern.consolidation_low,
            pullback_lookback_bars=eq.consolidation_window_bars,
            bars_available_for_lookback=eq.consolidation_window_bars,
            vwap_at_entry=last_vwap,
            market_anchor_price=_prior_bar_close(bars),
            preferred_order_type="STP_LMT" if is_standing_order else None,
        )

    def _within_window(self, ts: pd.Timestamp) -> bool:
        """True iff ``ts`` (NY-local) sits in the ``window_start``-``window_end`` window."""
        local = ts.time()
        return bool(self.window_start <= local < self.window_end)


def _recent_volume(bars: pd.DataFrame) -> int | None:
    """Best-effort int cast of the latest bar's volume; None on missing column."""
    if "volume" not in bars.columns:
        return None
    raw = bars["volume"].iloc[-1]
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _prior_bar_close(bars: pd.DataFrame) -> float | None:
    """Phase 12.5 — return the close of the bar immediately before the candidate, or None."""
    if len(bars) < 2 or "close" not in bars.columns:
        return None
    raw = bars["close"].iloc[-2]
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _apply_entry_quality_gates(
    *,
    cfg: EntryQualityConfig,
    bars: pd.DataFrame,
    candidate_price: float,
    symbol: str,
    strategy: str,
    bar_time: pd.Timestamp,
) -> bool:
    """Phase 13 — run enabled entry-quality gates; return True iff any rejected.

    Cheapest-first ordering: halt → impulse → consolidation → volume
    contraction → VWAP extension. First rejecting gate short-circuits.
    """
    if cfg.halt_detection_enabled and check_halt_detection(
        bars=bars,
        max_bar_gap_minutes=cfg.max_bar_gap_minutes,
        rth_only=cfg.halt_detection_rth_only,
        symbol=symbol,
        strategy=strategy,
        bar_time=bar_time,
    ):
        return True
    if cfg.impulse_strength_enabled and check_impulse_strength(
        bars=bars,
        impulse_window_bars=cfg.impulse_window_bars,
        consolidation_window_bars=cfg.consolidation_window_bars,
        impulse_min_pct_move=cfg.impulse_min_pct_move,
        impulse_min_slope_ratio=cfg.impulse_min_slope_ratio,
        symbol=symbol,
        strategy=strategy,
        bar_time=bar_time,
    ):
        return True
    if cfg.consolidation_tightness_enabled and check_consolidation_tightness(
        bars=bars,
        impulse_window_bars=cfg.impulse_window_bars,
        consolidation_window_bars=cfg.consolidation_window_bars,
        consolidation_max_range_pct=cfg.consolidation_max_range_pct,
        symbol=symbol,
        strategy=strategy,
        bar_time=bar_time,
    ):
        return True
    if cfg.volume_contraction_enabled and check_volume_contraction(
        bars=bars,
        impulse_window_bars=cfg.impulse_window_bars,
        consolidation_window_bars=cfg.consolidation_window_bars,
        max_consolidation_to_impulse_volume_ratio=cfg.max_consolidation_to_impulse_volume_ratio,
        symbol=symbol,
        strategy=strategy,
        bar_time=bar_time,
    ):
        return True
    return bool(
        cfg.vwap_extension_enabled
        and check_vwap_extension(
            bars=bars,
            candidate_price=candidate_price,
            max_extension_above_vwap_pct=cfg.max_extension_above_vwap_pct,
            symbol=symbol,
            strategy=strategy,
            bar_time=bar_time,
        )
    )

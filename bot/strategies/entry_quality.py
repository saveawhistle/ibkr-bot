"""Entry-quality gates for momentum and gap-and-go signal validation.

Forensic origin: ``reports/momentum_failures_2026_05_08.md`` analysed three
losing momentum trades (AEHL, TRAW, AIIO) on 2026-05-08 and identified five
distinct structural failure modes that the existing ``is_bull_flag`` heuristic
and Phase 12.4's recent-window RVOL gate did not catch:

* H1 - weak/absent impulse: the "flag" forms after a flat or down move.
* H2 - sloppy consolidation: the "flag" range is wide relative to the impulse,
  reflecting distribution rather than absorption.
* Volume contraction missing: consolidation volume isn't decreasing the way a
  real bull flag's does.
* VWAP extension: entry sits well above VWAP, indicating the move is exhausted
  rather than initiating.
* Halt-related sparse bars: AEHL had an 11-minute gap before its breakout bar
  due to a regulatory halt; the strategies treat the post-halt re-open like
  any other bar even though halt-resume context is markedly different.

These five gates supplement (do not replace) the existing pattern detection
and recent-RVOL checks. Each is independently enable/disable controlled at
config time and emits a distinct rejection event for forensic visibility.

Design pattern follows Phase 12.4's ``check_recent_window_rvol`` helper in
``bot/strategies/volume.py``: each gate is a pure function that returns either
``None`` (proceed) or a short reason string (suppress). Gates log a structured
``strategy.signal_rejected_<reason>`` event on rejection; suppressed signals
never reach the bus.

Insufficient-data policy: every gate that needs N+ bars to compute returns
``None`` when fewer bars are available rather than rejecting. This mirrors
the conservative "permissive on incomplete data" stance from gap-and-go's
PULLBACK_LOOKBACK_BARS=3 fallback (use whatever's available) and avoids
over-filtering early in the session. The recent-RVOL gate's stricter
behaviour (suppress until window is fully populated) is deliberate for that
gate -- a 20-bar volume baseline can't be meaningfully approximated -- and
not the right default for the structural gates here, which can be evaluated
on smaller windows with degraded but still-useful signal.
"""

from __future__ import annotations

from datetime import datetime, time
from typing import cast

import pandas as pd
import structlog

_log = structlog.get_logger("bot.strategies.entry_quality")

_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(16, 0)


def _iso(bar_time: datetime | pd.Timestamp) -> str:
    """ISO-8601 serialisation tolerant of both pd.Timestamp and datetime."""
    if hasattr(bar_time, "isoformat"):
        return bar_time.isoformat()
    return str(bar_time)


def _is_rth(ts: pd.Timestamp | datetime) -> bool:
    """True when ``ts`` (NY-local) sits in [09:30, 16:00) regular trading hours.

    Pre-market and after-hours bars are not RTH; legitimate sparse-bar gaps
    are common outside RTH and the halt-detection gate intentionally skips
    those periods.
    """
    local = ts.time()
    return bool(_MARKET_OPEN <= local < _MARKET_CLOSE)


def check_halt_detection(
    *,
    bars: pd.DataFrame,
    max_bar_gap_minutes: float,
    rth_only: bool,
    symbol: str,
    strategy: str,
    bar_time: datetime | pd.Timestamp,
) -> str | None:
    """AEHL pattern (2026-05-08 forensic). Reject if the last gap between bars exceeds threshold.

    Empirical detection only: looks at the gap between the candidate bar and
    the prior bar. A gap larger than ``max_bar_gap_minutes`` during RTH is
    treated as evidence of a regulatory halt or otherwise pathologically
    thin trading; halt-resume context is meaningfully different from
    continuation context, and we don't want to chase the breakout immediately
    on the re-open.

    When ``rth_only`` is True (default), the gap check is skipped entirely
    if the candidate bar is outside RTH -- legitimate thin-volume gaps are
    common pre-market and after-hours.

    Insufficient-data policy: <2 bars => return None (no gap to measure).
    """
    if len(bars) < 2:
        return None
    if rth_only and not _is_rth(bar_time):
        return None
    candidate_ts = bars.index[-1]
    prior_ts = bars.index[-2]
    if not isinstance(candidate_ts, pd.Timestamp) or not isinstance(prior_ts, pd.Timestamp):
        return None
    gap_seconds = (candidate_ts - prior_ts).total_seconds()
    gap_minutes = gap_seconds / 60.0
    if gap_minutes <= max_bar_gap_minutes:
        return None
    _log.info(
        "strategy.signal_rejected_halt_detected",
        symbol=symbol,
        strategy=strategy,
        bar_time=_iso(bar_time),
        prior_bar_time=_iso(prior_ts),
        gap_minutes=round(gap_minutes, 3),
        max_bar_gap_minutes=max_bar_gap_minutes,
        rth_only=rth_only,
    )
    return "halt_detected"


def check_impulse_strength(
    *,
    bars: pd.DataFrame,
    impulse_window_bars: int,
    consolidation_window_bars: int,
    impulse_min_pct_move: float,
    impulse_min_slope_ratio: float,
    symbol: str,
    strategy: str,
    bar_time: datetime | pd.Timestamp,
) -> str | None:
    """H1 from 2026-05-08 forensic. Reject when the impulse window lacks upward movement.

    The impulse window is the first ``impulse_window_bars`` bars of the lookback
    span (impulse + consolidation + breakout). For a real bull flag the impulse
    must show:

    * ``impulse_pct_move = (impulse_high - first_open) / first_open * 100``
      >= ``impulse_min_pct_move`` (default 1.5%)
    * ``slope_ratio = last_close / first_open`` >= ``impulse_min_slope_ratio``
      (default 1.005 -- i.e. 0.5% positive slope from open to close of impulse)

    Insufficient-data policy: needs >= ``impulse_window_bars + consolidation_window_bars + 1``
    bars to evaluate (impulse + consolidation + candidate breakout). When fewer
    are available, returns None rather than rejecting -- consistent with the
    "permissive on incomplete data" policy above.
    """
    required = impulse_window_bars + consolidation_window_bars + 1
    if len(bars) < required:
        return None
    # The breakout bar is bars.iloc[-1]; consolidation is the prior
    # ``consolidation_window_bars`` rows; impulse is the ``impulse_window_bars``
    # rows immediately before consolidation.
    impulse_end = -(consolidation_window_bars + 1)
    impulse_start = impulse_end - impulse_window_bars
    impulse = bars.iloc[impulse_start:impulse_end]
    if impulse.empty:
        return None
    first_open = float(impulse["open"].iloc[0])
    if first_open <= 0:
        return None
    impulse_high = float(impulse["high"].max())
    last_close = float(impulse["close"].iloc[-1])
    impulse_pct_move = (impulse_high - first_open) / first_open * 100.0
    slope_ratio = last_close / first_open
    if impulse_pct_move < impulse_min_pct_move:
        _log.info(
            "strategy.signal_rejected_weak_impulse",
            symbol=symbol,
            strategy=strategy,
            bar_time=_iso(bar_time),
            reason="weak_impulse_pct",
            impulse_pct_move=round(impulse_pct_move, 4),
            impulse_min_pct_move=impulse_min_pct_move,
            impulse_slope_ratio=round(slope_ratio, 5),
            impulse_min_slope_ratio=impulse_min_slope_ratio,
            impulse_window_bars=impulse_window_bars,
        )
        return "weak_impulse_pct"
    if slope_ratio < impulse_min_slope_ratio:
        _log.info(
            "strategy.signal_rejected_weak_impulse",
            symbol=symbol,
            strategy=strategy,
            bar_time=_iso(bar_time),
            reason="weak_impulse_slope",
            impulse_pct_move=round(impulse_pct_move, 4),
            impulse_min_pct_move=impulse_min_pct_move,
            impulse_slope_ratio=round(slope_ratio, 5),
            impulse_min_slope_ratio=impulse_min_slope_ratio,
            impulse_window_bars=impulse_window_bars,
        )
        return "weak_impulse_slope"
    return None


def check_consolidation_tightness(
    *,
    bars: pd.DataFrame,
    impulse_window_bars: int,
    consolidation_window_bars: int,
    consolidation_max_range_pct: float,
    symbol: str,
    strategy: str,
    bar_time: datetime | pd.Timestamp,
) -> str | None:
    """H2 from 2026-05-08 forensic. Reject when consolidation range is too wide.

    Range = (max_high - min_low) over the consolidation window, expressed as
    a percentage of the impulse high. Real bull flags absorb in a narrow
    range; sloppy consolidations are distribution disguised as absorption.

    Insufficient-data policy: same as ``check_impulse_strength`` --
    >= impulse + consolidation + 1 bars required, else returns None.
    """
    required = impulse_window_bars + consolidation_window_bars + 1
    if len(bars) < required:
        return None
    impulse_end = -(consolidation_window_bars + 1)
    impulse_start = impulse_end - impulse_window_bars
    impulse = bars.iloc[impulse_start:impulse_end]
    consolidation = bars.iloc[impulse_end:-1]
    if impulse.empty or consolidation.empty:
        return None
    impulse_high = float(impulse["high"].max())
    if impulse_high <= 0:
        return None
    cons_high = float(consolidation["high"].max())
    cons_low = float(consolidation["low"].min())
    range_pct = (cons_high - cons_low) / impulse_high * 100.0
    if range_pct <= consolidation_max_range_pct:
        return None
    _log.info(
        "strategy.signal_rejected_loose_consolidation",
        symbol=symbol,
        strategy=strategy,
        bar_time=_iso(bar_time),
        consolidation_range_pct=round(range_pct, 4),
        consolidation_max_range_pct=consolidation_max_range_pct,
        impulse_high=round(impulse_high, 4),
        consolidation_high=round(cons_high, 4),
        consolidation_low=round(cons_low, 4),
        consolidation_window_bars=consolidation_window_bars,
    )
    return "loose_consolidation"


def check_volume_contraction(
    *,
    bars: pd.DataFrame,
    impulse_window_bars: int,
    consolidation_window_bars: int,
    max_consolidation_to_impulse_volume_ratio: float,
    symbol: str,
    strategy: str,
    bar_time: datetime | pd.Timestamp,
) -> str | None:
    """TRAW pattern (2026-05-08 forensic). Reject when consolidation volume is not contracted.

    Real bull flags show decreasing volume during consolidation as supply
    dries up. Distribution disguised as consolidation often shows the
    opposite -- heavy or rising volume that reflects sellers unloading
    into bids.

    Computes ``cons_avg_vol / impulse_avg_vol`` and rejects when the ratio
    exceeds ``max_consolidation_to_impulse_volume_ratio`` (default 0.8 --
    consolidation volume must be at most 80% of impulse volume).

    Insufficient-data policy:
    * <required bars => return None
    * ``volume`` column missing => return None (synthetic test path)
    * impulse avg volume == 0 => return None (legitimate but pathological;
      can't compute a ratio)
    """
    required = impulse_window_bars + consolidation_window_bars + 1
    if len(bars) < required:
        return None
    if "volume" not in bars.columns:
        return None
    impulse_end = -(consolidation_window_bars + 1)
    impulse_start = impulse_end - impulse_window_bars
    impulse = bars.iloc[impulse_start:impulse_end]
    consolidation = bars.iloc[impulse_end:-1]
    if impulse.empty or consolidation.empty:
        return None
    impulse_avg = float(impulse["volume"].mean())
    cons_avg = float(consolidation["volume"].mean())
    if impulse_avg <= 0:
        return None
    ratio = cons_avg / impulse_avg
    if ratio <= max_consolidation_to_impulse_volume_ratio:
        return None
    _log.info(
        "strategy.signal_rejected_no_volume_contraction",
        symbol=symbol,
        strategy=strategy,
        bar_time=_iso(bar_time),
        consolidation_avg_volume=round(cons_avg, 2),
        impulse_avg_volume=round(impulse_avg, 2),
        ratio=round(ratio, 4),
        max_consolidation_to_impulse_volume_ratio=max_consolidation_to_impulse_volume_ratio,
        impulse_window_bars=impulse_window_bars,
        consolidation_window_bars=consolidation_window_bars,
    )
    return "no_volume_contraction"


def _session_vwap(bars: pd.DataFrame) -> float | None:
    """Compute session-anchored VWAP for the *latest* bar.

    Walks back from the candidate bar to the session boundary (most recent
    09:30 ET on the candidate's local date, or first bar in the frame if
    the frame doesn't span 09:30) and sums (typical * volume) / sum(volume)
    across that slice. Mirrors ``bot.indicators.vwap``'s session anchoring
    but returns just the latest bar's value as a scalar.

    Returns None when the cumulative volume across the session slice is
    zero (no volume bars to weight) or when required columns are missing.
    """
    if bars.empty or "volume" not in bars.columns:
        return None
    required = {"high", "low", "close", "volume"}
    if not required.issubset(bars.columns):
        return None
    last_ts = bars.index[-1]
    if isinstance(last_ts, pd.Timestamp):
        index = cast("pd.DatetimeIndex", bars.index)
        local = index.tz_convert("America/New_York") if index.tz is not None else index
        last_local = local[-1]
        candidate_date = last_local.date()
        # Slice to the candidate's session: pre-market bars on the same
        # date are intentionally included (matches indicators.vwap's
        # calendar-date anchoring -- premarket flows into the morning's
        # VWAP per the published playbook).
        same_date = local.date == candidate_date
        slice_mask = pd.Series(same_date, index=bars.index)
        session = bars.loc[slice_mask]
        if session.empty:
            session = bars
    else:
        session = bars
    typical = (session["high"] + session["low"] + session["close"]) / 3.0
    volume = session["volume"].astype(float)
    pv_sum = float((typical * volume).sum())
    v_sum = float(volume.sum())
    if v_sum <= 0:
        return None
    return pv_sum / v_sum


def check_vwap_extension(
    *,
    bars: pd.DataFrame,
    candidate_price: float,
    max_extension_above_vwap_pct: float,
    symbol: str,
    strategy: str,
    bar_time: datetime | pd.Timestamp,
) -> str | None:
    """AIIO pattern (2026-05-08 forensic). Reject when entry price is too far above VWAP.

    Late-in-move entries from extended levels often fail because the move
    is exhausted, not initiating. Computes session-anchored VWAP and rejects
    when ``(candidate_price - vwap) / vwap * 100`` exceeds
    ``max_extension_above_vwap_pct`` (default 5.0%).

    Insufficient-data policy: VWAP undefined (no volume bars, missing
    columns) => return None rather than reject (don't penalise legitimate
    signals on data limitations). Distinct from the existing
    ``evaluate_extension`` ATR-multiple check in ``bot.indicators`` -- that
    one scales with volatility (ATR), this one is a fixed percentage above
    VWAP and catches the late-extension pattern that ATR-scaled checks let
    through when ATR is depressed.
    """
    vwap_value = _session_vwap(bars)
    if vwap_value is None or vwap_value <= 0:
        return None
    extension_pct = (candidate_price - vwap_value) / vwap_value * 100.0
    if extension_pct <= max_extension_above_vwap_pct:
        return None
    _log.info(
        "strategy.signal_rejected_vwap_extension",
        symbol=symbol,
        strategy=strategy,
        bar_time=_iso(bar_time),
        candidate_price=round(candidate_price, 4),
        vwap=round(vwap_value, 4),
        extension_pct=round(extension_pct, 4),
        max_extension_above_vwap_pct=max_extension_above_vwap_pct,
    )
    return "excessive_vwap_extension"


def check_consolidation_vwap_hold(
    *,
    vwap_hold: bool,
    pattern_type: str,
    consolidation_low: float,
    symbol: str,
    strategy: str,
    bar_time: datetime | pd.Timestamp,
) -> str | None:
    """Reject when one or more consolidation bars closed below VWAP.

    Cameron's published rule: real bull flags, micro pullbacks, and flat tops
    all consolidate *above* VWAP. A close below VWAP during the flag signals
    buyers losing control before the breakout triggers.

    ``vwap_hold`` is computed by ``analyze_momentum_pattern`` at parse time;
    this gate wraps the bool in the canonical rejection path so it can be
    disabled by conftest's autouse noop fixture for legacy tests.
    """
    if vwap_hold:
        return None
    _log.info(
        "strategy.signal_rejected_vwap_not_held",
        symbol=symbol,
        strategy=strategy,
        bar_time=_iso(bar_time),
        pattern_type=pattern_type,
        consolidation_low=round(consolidation_low, 4),
    )
    return "vwap_not_held_during_consolidation"


def check_breakout_volume_ratio(
    *,
    bars: pd.DataFrame,
    consolidation_window_bars: int,
    min_ratio: float,
    symbol: str,
    strategy: str,
    bar_time: datetime | pd.Timestamp,
) -> str | None:
    """Reject when the breakout bar's volume is weak relative to the consolidation.

    Cameron's published rule: the breakout candle must show a volume surge
    above the quiet consolidation bars, confirming that buyers are stepping
    in at the trigger rather than the move being a low-conviction drift.
    Computes ``breakout_vol / consolidation_avg_vol`` and rejects when the
    ratio falls below ``min_ratio`` (default 1.5).

    Insufficient-data policy:
    * Fewer bars than needed => return None
    * ``volume`` column absent => return None (synthetic test path)
    * consolidation avg == 0 => return None (pathological but not an error)
    """
    required = consolidation_window_bars + 2
    if len(bars) < required:
        return None
    if "volume" not in bars.columns:
        return None
    breakout_vol = float(bars["volume"].iloc[-1])
    cons_vols = bars["volume"].iloc[-(consolidation_window_bars + 1):-1]
    cons_avg = float(cons_vols.mean())
    if cons_avg <= 0:
        return None
    ratio = breakout_vol / cons_avg
    if ratio >= min_ratio:
        return None
    _log.info(
        "strategy.signal_rejected_insufficient_breakout_volume",
        symbol=symbol,
        strategy=strategy,
        bar_time=_iso(bar_time),
        breakout_volume=round(breakout_vol, 2),
        consolidation_avg_volume=round(cons_avg, 2),
        ratio=round(ratio, 4),
        min_ratio=min_ratio,
    )
    return "insufficient_breakout_volume"


__all__ = [
    "check_breakout_volume_ratio",
    "check_consolidation_tightness",
    "check_consolidation_vwap_hold",
    "check_halt_detection",
    "check_impulse_strength",
    "check_volume_contraction",
    "check_vwap_extension",
]

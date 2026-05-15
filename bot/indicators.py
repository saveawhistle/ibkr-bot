"""Pure-function technical indicators for strategy evaluation.

All functions operate on pandas DataFrames / Series with an ``America/New_York``
timezone-aware index (see ``bot.brokerage.market_data``). Nothing here does any I/O or
has side effects — strategies must remain side-effect-free during ``evaluate``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Literal, NamedTuple, cast

import pandas as pd

_MARKET_OPEN = time(9, 30)


class HodContext(NamedTuple):
    """Phase 7.2 — running-HOD diagnostic for the *latest* bar in a frame.

    ``is_new_hod`` is the same boolean the strategies gate on; the other
    fields let a ``not_new_hod`` rejection event name the exact value the
    current bar needed to exceed. ``session_hod`` is ``None`` when the
    current bar is the first market-hours bar of the session (nothing to
    exceed yet). ``bars_in_session`` counts market-hours bars on the
    current session; it's 0 when the latest bar is premarket.
    """

    is_new_hod: bool
    last_high: float
    session_hod: float | None
    bars_in_session: int


class ExtensionCheck(NamedTuple):
    """Phase 6.6 — diagnostic result of the VWAP-distance extension check.

    All numeric fields can be ``None`` when inputs are missing (empty bars,
    empty VWAP series, or a non-positive ATR — see ``evaluate_extension``).
    Strategies stuff these onto the ``signal.rejected`` event so an operator
    can grep ``extension_ratio`` across sessions to recalibrate the multiple
    without re-deriving the math from raw close + vwap each time.

    ``extension_ratio = distance_from_vwap / threshold_distance``: values
    above 1.0 are rejections, values below 1.0 mean the bar passed the
    threshold. ``None`` when the threshold is unavailable.
    """

    extended: bool
    last_atr_value: float | None
    distance_from_vwap: float | None
    threshold_distance: float | None
    extension_ratio: float | None


@dataclass(frozen=True)
class MomentumPattern:
    """Rich result of ``analyze_momentum_pattern``.

    ``pattern_type`` is one of ``"bull_flag"``, ``"micro_pullback"``, or
    ``"flat_top"``. ``trigger_level`` is the max high of the consolidation
    window — the resistance price where a standing STP_LMT order would sit.
    ``consolidation_low`` is the min low of the consolidation window and
    serves as the structural stop reference (scoped to the flag/pause bars,
    not the broader impulse). ``vwap_hold`` is ``True`` when every
    consolidation bar's close is at or above VWAP at that bar's timestamp.
    """

    pattern_type: str
    trigger_level: float
    consolidation_low: float
    red_candle_count: int
    pullback_pct_of_pole: float
    vwap_hold: bool


def vwap(bars: pd.DataFrame) -> pd.Series:
    """Session-anchored VWAP resetting every day at 09:30 ET.

    Uses the bar's typical price ``(high + low + close) / 3`` weighted by
    volume, cumulatively summed within each trading session. Pre-09:30 bars on
    a given day are grouped with that day's regular session (premarket flows
    into the VWAP for the morning window, matching the published playbook).
    """
    if bars.empty:
        return pd.Series(dtype=float, name="vwap")
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    volume = bars["volume"].astype(float)
    session_key = _session_key(cast("pd.DatetimeIndex", bars.index))
    pv = (typical * volume).groupby(session_key).cumsum()
    cum_v = volume.groupby(session_key).cumsum()
    result = pv / cum_v.where(cum_v > 0)
    result.name = "vwap"
    return result


def ema(series: pd.Series, length: int) -> pd.Series:
    """Standard pandas EMA with ``adjust=False`` (matches TradingView's EMA)."""
    if length <= 0:
        raise ValueError(f"EMA length must be positive, got {length}")
    return series.ewm(span=length, adjust=False).mean()


def atr(bars: pd.DataFrame, length: int = 14) -> pd.Series:
    """Wilder's ATR on 1-min bars — used for stop-distance sanity checks."""
    if length <= 0:
        raise ValueError(f"ATR length must be positive, got {length}")
    high = bars["high"]
    low = bars["low"]
    prev_close = bars["close"].shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # Wilder smoothing = EMA with alpha = 1/length
    return true_range.ewm(alpha=1.0 / length, adjust=False).mean()


def new_high_of_day(
    bars: pd.DataFrame,
    by: Literal["high", "close"] = "high",
) -> pd.Series:
    """Boolean mask: True where the bar's price exceeds all prior session highs.

    Session-aware: HOD resets at 09:30 ET (Phase 7.2). Premarket bars form
    their own group and do NOT contaminate the market-hours running max;
    the first bar of each group is always False (no prior bar to exceed).

    ``by`` selects the comparison column. Running max stays high-based —
    the session HOD itself is defined by the highest high — only the
    value we compare *against* it changes:

    * ``"high"`` (default): bar high vs prior session highs. Use for
      observability, session HOD tracking, identifying any bar that
      touched HOD.
    * ``"close"``: bar close vs prior session highs. Use for breakout
      entry confirmation. A wick-and-retrace bar (high above HOD, close
      below) returns False, correctly identifying a failed breakout.

    Phase 7.2 fixed the calendar-date session-boundary bug. Phase 9.1
    added the ``by`` parameter after observing RMAX 2026-04-27 09:34
    enter on a bar that wicked HOD then closed red.
    """
    if bars.empty:
        return pd.Series(dtype=bool, name="new_hod")
    session_key = _market_hours_session_key(cast("pd.DatetimeIndex", bars.index))
    running_max = bars["high"].groupby(session_key).cummax()
    prior_max = running_max.groupby(session_key).shift(1)
    result = bars[by] > prior_max
    result.name = "new_hod"
    return result.fillna(False).astype(bool)


def evaluate_hod(
    bars: pd.DataFrame,
    by: Literal["high", "close"] = "high",
) -> HodContext | None:
    """Phase 7.2 — full HOD context for the *latest* bar, for rejection logging.

    Returns ``None`` when ``bars`` is empty. For premarket-only frames the
    returned context reflects the premarket group (still no HOD break
    unless exceeded within the group). For market-hours frames, the first
    market-hours bar yields ``session_hod=None`` and ``is_new_hod=False``
    by construction — nothing to exceed.

    ``by`` mirrors ``new_high_of_day``: ``"high"`` (default) compares
    last bar's high to the prior session high (observability); ``"close"``
    compares last bar's close to the prior session high (breakout entry
    confirmation — Phase 9.1). ``last_high`` and ``session_hod`` on the
    returned context remain high-based regardless so the rejection event
    can show both the wick and the close that failed to confirm.
    """
    if bars.empty:
        return None
    index = cast("pd.DatetimeIndex", bars.index)
    session_key = _market_hours_session_key(index)
    last_key = session_key.iloc[-1]
    session_slice = bars.loc[session_key == last_key]
    last_high = float(session_slice["high"].iloc[-1])
    bars_in_session = int(len(session_slice))
    if bars_in_session <= 1:
        return HodContext(
            is_new_hod=False,
            last_high=last_high,
            session_hod=None,
            bars_in_session=bars_in_session,
        )
    prior_max = float(session_slice["high"].iloc[:-1].max())
    comparison_value = float(session_slice[by].iloc[-1])
    return HodContext(
        is_new_hod=comparison_value > prior_max,
        last_high=last_high,
        session_hod=prior_max,
        bars_in_session=bars_in_session,
    )


def is_bull_flag(bars: pd.DataFrame, max_pullback_pct: float = 5.0, lookback: int = 10) -> bool:
    """Heuristic: the recent ``lookback`` bars show a shallow pullback after an impulse.

    The flag is inspected on the *consolidation* window — the bars between the
    initial impulse and the current (breakout) bar. Impulse high is the max of
    the first 3 bars; the consolidation region is the middle slice
    ``[3 : -1]``. If the lowest close in that region is within
    ``max_pullback_pct`` of the impulse high, we're in a flag.
    """
    if len(bars) < lookback:
        return False
    window = bars.iloc[-lookback:]
    impulse_high = float(window["high"].iloc[:3].max())
    if impulse_high <= 0:
        return False
    consolidation = window.iloc[3:-1]
    if consolidation.empty:
        return False
    trough = float(consolidation["close"].min())
    pullback_pct = (impulse_high - trough) / impulse_high * 100.0
    return 0.0 <= pullback_pct <= max_pullback_pct


def analyze_momentum_pattern(
    bars: pd.DataFrame,
    vwap_series: pd.Series,
    *,
    impulse_window_bars: int = 3,
    consolidation_window_bars: int = 6,
    include_last_bar_in_consolidation: bool = False,
    bull_flag_min_red_candles: int = 2,
    bull_flag_max_red_candles: int = 3,
    bull_flag_max_pullback_pct_of_pole: float = 0.50,
    micro_pullback_max_range_pct: float = 2.0,
    flat_top_max_high_range_pct: float = 0.5,
) -> MomentumPattern | None:
    """Detect Bull Flag, Micro Pullback, or Flat Top consolidation pattern.

    Segments the bar history into an impulse leg and a consolidation window
    (parameterised by ``impulse_window_bars`` / ``consolidation_window_bars``).
    Returns ``None`` when insufficient bars are available or when no pattern
    matches the consolidation.

    When ``include_last_bar_in_consolidation=False`` (breakout path, default):
      * consolidation = ``bars.iloc[-(cons+1):-1]``  (trigger bar excluded)
      * impulse       = ``bars.iloc[-(imp+cons+1):-(cons+1)]``

    When ``include_last_bar_in_consolidation=True`` (standing-order path, the
    current bar is still part of the flag):
      * consolidation = ``bars.iloc[-cons:]``
      * impulse       = ``bars.iloc[-(imp+cons):-cons]``

    Pattern priority (first match wins):
      1. **flat_top**      — highs of consolidation cluster within ``flat_top_max_high_range_pct``
      2. **micro_pullback** — total range (high-to-low) within ``micro_pullback_max_range_pct``
      3. **bull_flag**     — ``[min, max]`` red candles AND pullback < ``bull_flag_max_pullback_pct_of_pole`` of pole

    VWAP hold: ``MomentumPattern.vwap_hold`` is ``True`` when every consolidation
    bar's close is >= its VWAP value. Permissive on missing VWAP data (treats
    missing bars as held) so synthetic test frames without full VWAP history
    don't auto-reject.
    """
    required = impulse_window_bars + consolidation_window_bars + 1
    if len(bars) < required:
        return None

    if include_last_bar_in_consolidation:
        consolidation = bars.iloc[-consolidation_window_bars:]
        impulse = bars.iloc[-(impulse_window_bars + consolidation_window_bars):-consolidation_window_bars]
    else:
        consolidation = bars.iloc[-(consolidation_window_bars + 1):-1]
        impulse = bars.iloc[-(impulse_window_bars + consolidation_window_bars + 1):-(consolidation_window_bars + 1)]

    if consolidation.empty or impulse.empty:
        return None

    flag_high = float(consolidation["high"].max())
    flag_low = float(consolidation["low"].min())
    if flag_high <= 0:
        return None

    pole_low = float(impulse["low"].min())
    pole_height = flag_high - pole_low
    if pole_height <= 0:
        return None
    pullback_depth = flag_high - flag_low
    pullback_pct_of_pole = pullback_depth / pole_height

    # Count bars where close < the prior bar's close (down bars in the session).
    # Using prior close rather than bar open because 1-minute bar opens typically
    # equal the prior close intraday; close < prior_close cleanly identifies a
    # descending bar regardless of how the fixture or feed populates the open field.
    prior_closes = bars["close"].shift(1)
    red_candle_count = int(
        (consolidation["close"] < prior_closes.loc[consolidation.index]).sum()
    )
    high_range_pct = (float(consolidation["high"].max()) - float(consolidation["high"].min())) / flag_high * 100.0
    total_range_pct = pullback_depth / flag_high * 100.0

    # VWAP hold: check each consolidation bar's close against its VWAP value.
    vwap_hold = True
    if not vwap_series.empty:
        for ts in consolidation.index:
            if ts in vwap_series.index:
                bar_close = float(consolidation.loc[ts, "close"])
                bar_vwap = float(vwap_series.loc[ts])
                if bar_close < bar_vwap:
                    vwap_hold = False
                    break

    # Pattern classification — first match wins.
    if high_range_pct <= flat_top_max_high_range_pct:
        pattern_type = "flat_top"
    elif total_range_pct <= micro_pullback_max_range_pct:
        pattern_type = "micro_pullback"
    elif bull_flag_min_red_candles <= red_candle_count <= bull_flag_max_red_candles and pullback_pct_of_pole <= bull_flag_max_pullback_pct_of_pole:
        pattern_type = "bull_flag"
    else:
        return None

    return MomentumPattern(
        pattern_type=pattern_type,
        trigger_level=flag_high,
        consolidation_low=flag_low,
        red_candle_count=red_candle_count,
        pullback_pct_of_pole=round(pullback_pct_of_pole, 4),
        vwap_hold=vwap_hold,
    )


def evaluate_extension(
    bars: pd.DataFrame,
    vwap_series: pd.Series,
    *,
    atr_multiple: float = 5.0,
) -> ExtensionCheck:
    """Phase 6.6 — full extension check + diagnostic context for logging.

    Returns ``ExtensionCheck(extended, atr, distance, threshold, ratio)``.
    Strict greater-than against the threshold matches the prior behaviour
    of ``is_extension_bar_atr`` so a distance exactly equal to the
    threshold passes.

    Defaults ``atr_multiple`` to 5.0 — the calibrated value from Day 3
    paper trading. Strategies always pass an explicit value sourced from
    config; the default is documentary only.

    The graceful-fallback semantics (empty bars / vwap / non-positive ATR
    => ``extended=False`` with as many context fields populated as the
    inputs allow) preserve back-compat with the pre-6.6
    ``is_extension_bar_atr`` contract.
    """
    if bars.empty or vwap_series.empty:
        return ExtensionCheck(False, None, None, None, None)
    last_close = float(bars["close"].iloc[-1])
    last_vwap = float(vwap_series.iloc[-1])
    distance = last_close - last_vwap
    last_atr_series = atr(bars)
    if last_atr_series.empty:
        return ExtensionCheck(False, None, distance, None, None)
    last_atr_value = float(last_atr_series.iloc[-1])
    if last_atr_value <= 0:
        return ExtensionCheck(False, last_atr_value, distance, None, None)
    threshold = atr_multiple * last_atr_value
    ratio = distance / threshold if threshold > 0 else None
    return ExtensionCheck(distance > threshold, last_atr_value, distance, threshold, ratio)


def is_extension_bar_atr(
    bars: pd.DataFrame, vwap_series: pd.Series, atr_multiple: float = 3.0
) -> bool:
    """Phase 6.6 — thin back-compat wrapper around ``evaluate_extension``.

    Pre-6.6 callers (and the dedicated indicator unit tests) consume only
    the boolean. Strategies migrated to ``evaluate_extension`` directly
    so they can read ATR + distance + ratio for the enriched
    ``signal.rejected`` event. Default kept at 3.0 so any external caller
    that didn't pass ``atr_multiple`` sees identical behaviour.
    """
    return evaluate_extension(bars, vwap_series, atr_multiple=atr_multiple).extended


def is_extension_bar_dollar(bar: pd.Series, position_shares: int, dollar_threshold: float) -> bool:
    """True when a single bar's unrealized gain clears ``dollar_threshold``.

    the extension bar rule is scale-dependent: a candle that instantly
    puts him up $200-$400 on the position. We approximate "instant gain" as
    ``(high - open) * shares`` — the best-case mark during the bar, measured
    from its open. Returns False on a red bar (``close < open``) so a wick
    through the high doesn't trigger on what turned out to be a rejection.
    """
    if position_shares <= 0 or dollar_threshold <= 0:
        return False
    bar_open = float(bar["open"])
    bar_high = float(bar["high"])
    bar_close = float(bar["close"])
    if bar_close < bar_open:
        return False
    return (bar_high - bar_open) * position_shares >= dollar_threshold


def premarket_high(bars: pd.DataFrame) -> float | None:
    """Phase 8.4 — highest ``high`` across premarket bars on the latest session.

    "Premarket" = bars stamped before 09:30 ET on the same NY-local
    calendar date as the latest bar in the frame. Returns ``None`` when
    the frame is empty, has no datetime index, or contains no premarket
    bars on the latest session (e.g. the bot subscribed mid-session
    after market open and IBKR's backfill didn't include premarket).

    Strategies use this to cap the scale-out target: if a setup fires
    while ``entry < premarket_high``, the take-profit is placed just
    below PMH (well-known intraday resistance) rather than at
    ``entry + N×R``. When ``entry`` already cleared PMH the cap can't
    bind and the strategy falls back to the standard 2R target.
    """
    if bars.empty or not isinstance(bars.index, pd.DatetimeIndex) or len(bars.index) == 0:
        return None
    index = bars.index
    local = index.tz_convert("America/New_York") if index.tz is not None else index
    last_date = local[-1].date()
    same_date = local.date == last_date
    is_premarket = local.time < _MARKET_OPEN
    mask = same_date & is_premarket
    if not mask.any():
        return None
    return float(bars.loc[mask, "high"].max())


def _session_key(index: pd.DatetimeIndex) -> pd.Series:
    """Return a session-grouping key: each NY calendar date is one session.

    Still used by ``vwap``: the intraday VWAP intentionally includes
    premarket volume, so a calendar-date grouping is the correct boundary.
    For HOD-style computations use ``_market_hours_session_key`` instead.
    """
    # Convert to NY for grouping even if already in NY (idempotent).
    local = index.tz_convert("America/New_York") if index.tz is not None else index
    return pd.Series(local.date, index=index, name="session")


def _market_hours_session_key(index: pd.DatetimeIndex) -> pd.Series:
    """Return a session-grouping key that resets at 09:30 ET.

    Phase 7.2: premarket bars (pre-09:30 NY local) are assigned a distinct
    ``<date>-premarket`` group, separate from the ``<date>-market`` group
    that covers 09:30 onward. This lets HOD-style computations reset at
    market open instead of carrying premarket wicks into the running max.

    Contrast with ``_session_key`` which groups all bars on the same
    calendar date together — that function still serves VWAP, where
    premarket inclusion is intentional per the published playbook.
    """
    local = index.tz_convert("America/New_York") if index.tz is not None else index
    is_market_hours = local.time >= _MARKET_OPEN
    date_str = pd.Series(local.date, index=index).astype(str)
    suffix = pd.Series(
        ["market" if mh else "premarket" for mh in is_market_hours],
        index=index,
    )
    return pd.Series(date_str + "-" + suffix, index=index, name="market_session")


__all__ = [
    "ExtensionCheck",
    "HodContext",
    "atr",
    "ema",
    "evaluate_extension",
    "evaluate_hod",
    "is_bull_flag",
    "is_extension_bar_atr",
    "is_extension_bar_dollar",
    "new_high_of_day",
    "premarket_high",
    "vwap",
]

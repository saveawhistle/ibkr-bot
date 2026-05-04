"""Tests for pure-function indicators — VWAP, EMA, ATR, HOD, flag, extension."""

from __future__ import annotations

import pandas as pd
import pytest

from bot.indicators import (
    _market_hours_session_key,
    _session_key,
    atr,
    ema,
    evaluate_hod,
    is_bull_flag,
    is_extension_bar_atr,
    is_extension_bar_dollar,
    new_high_of_day,
    vwap,
)


def _frame(
    times: list[str],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
) -> pd.DataFrame:
    """Build a tiny NY-tz DataFrame matching the MarketData shape."""
    idx = pd.to_datetime(times).tz_localize("America/New_York")
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "vwap": closes,  # fabricated, indicators ignore this column
        },
        index=idx,
    )


def test_vwap_single_session_matches_typical_price_weighting() -> None:
    """Two equal-volume bars with typical prices 10 and 12 → VWAP [10, 11]."""
    bars = _frame(
        times=["2026-04-16 09:30", "2026-04-16 09:31"],
        opens=[10, 11],
        highs=[10, 12],
        lows=[10, 12],
        closes=[10, 12],
        volumes=[100, 100],
    )
    result = vwap(bars).tolist()
    assert result == pytest.approx([10.0, 11.0])


def test_vwap_resets_across_sessions() -> None:
    """A new calendar date starts its own VWAP accumulation."""
    bars = _frame(
        times=["2026-04-15 15:59", "2026-04-16 09:30"],
        opens=[20, 5],
        highs=[20, 5],
        lows=[20, 5],
        closes=[20, 5],
        volumes=[1000, 100],
    )
    result = vwap(bars).tolist()
    assert result[0] == pytest.approx(20.0)
    assert result[1] == pytest.approx(5.0)


def test_ema_monotonic_on_rising_series() -> None:
    """EMA of a strictly rising series must also be strictly rising."""
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    e = ema(s, length=3)
    diffs = e.diff().dropna().tolist()
    assert all(d > 0 for d in diffs)


def test_ema_rejects_nonpositive_length() -> None:
    """EMA length must be positive — a zero/negative length is a config bug."""
    with pytest.raises(ValueError):
        ema(pd.Series([1.0]), length=0)


def test_atr_stabilises_on_constant_range() -> None:
    """On bars with a constant 1.0 true range, ATR converges to 1.0."""
    bars = _frame(
        times=[f"2026-04-16 09:{30 + i:02d}" for i in range(15)],
        opens=[10.0] * 15,
        highs=[11.0] * 15,
        lows=[10.0] * 15,
        closes=[10.5] * 15,
        volumes=[100.0] * 15,
    )
    result = atr(bars, length=14).iloc[-1]
    assert result == pytest.approx(1.0, rel=0.05)


def test_new_high_of_day_resets_each_session() -> None:
    """HOD mask is False on each session's first bar and True on bars that exceed prior session highs."""
    bars = _frame(
        times=[
            "2026-04-15 09:30",
            "2026-04-15 09:31",
            "2026-04-15 09:32",
            "2026-04-16 09:30",
            "2026-04-16 09:31",
        ],
        opens=[1, 1, 1, 1, 1],
        highs=[10, 9, 11, 5, 6],
        lows=[1, 1, 1, 1, 1],
        closes=[1, 1, 1, 1, 1],
        volumes=[1, 1, 1, 1, 1],
    )
    mask = new_high_of_day(bars).tolist()
    # First bar of each session is False; 09:32 on 04-15 breaks 10; 09:31 on 04-16 breaks 5.
    assert mask == [False, False, True, False, True]


# ---------- Phase 7.2: HOD resets at 09:30 ET ---------- #


def test_new_high_of_day_resets_at_market_open() -> None:
    """Premarket highs do not contaminate the market-hours running max.

    Five premarket bars (4:00-9:29) peaking at $15 and five market bars
    (9:30-9:34) with lower locally-rising highs. Only the market-hours bars
    after the first are compared against each other; the $15 premarket high
    is invisible to the market-hours HOD.
    """
    bars = _frame(
        times=[
            "2026-04-16 04:00",
            "2026-04-16 06:00",
            "2026-04-16 08:00",
            "2026-04-16 09:00",
            "2026-04-16 09:29",
            "2026-04-16 09:30",
            "2026-04-16 09:31",
            "2026-04-16 09:32",
            "2026-04-16 09:33",
            "2026-04-16 09:34",
        ],
        opens=[5] * 10,
        highs=[5.0, 10.0, 12.0, 15.0, 14.0, 7.0, 8.0, 9.0, 9.5, 10.0],
        lows=[4.0] * 10,
        closes=[5.0] * 10,
        volumes=[100] * 10,
    )
    mask = new_high_of_day(bars).tolist()
    # Premarket group: first bar False (no prior), subsequent bars True/False
    # by their own rising highs within the premarket group.
    assert mask[:5] == [False, True, True, True, False]
    # Market-hours group: 9:30 is the first → False. 9:31-9:34 all exceed
    # their own preceding market-hours highs (rising 7 → 10). The $15
    # premarket peak is NOT in the market-hours comparison set.
    assert mask[5:] == [False, True, True, True, True]


def test_new_high_of_day_premarket_high_does_not_contaminate() -> None:
    """A premarket bar at $20 does not block market-hours HOD breaks below $20."""
    bars = _frame(
        times=[
            "2026-04-16 04:00",
            "2026-04-16 09:30",
            "2026-04-16 09:31",
            "2026-04-16 09:32",
        ],
        opens=[10, 8, 8, 8],
        highs=[20.0, 8.0, 9.0, 10.0],
        lows=[8, 7, 7, 8],
        closes=[15, 8, 9, 10],
        volumes=[100, 100, 100, 100],
    )
    mask = new_high_of_day(bars).tolist()
    # Premarket bar: first bar of its group → False.
    assert mask[0] is False
    # 9:30: first bar of market session → False.
    assert mask[1] is False
    # 9:31 high 9 > 9:30 high 8 → True. 9:32 high 10 > 9 → True.
    assert mask[2] is True
    assert mask[3] is True


def test_new_high_of_day_auud_scenario() -> None:
    """Replay AUUD Day 4 market-hours bars plus a $15 premarket wick.

    Pre-7.2 this scenario produced 37 consecutive ``not_new_hod`` rejections
    because the premarket print anchored the running max above $10. Post-fix
    bars 2-6 (each a visible new market-hours HOD) are flagged True.
    """
    bars = _frame(
        times=[
            "2026-04-23 04:00",  # premarket wick at 15
            "2026-04-23 09:30",  # bar 1 — first market bar, False
            "2026-04-23 09:31",  # bar 2 — 7.67 > 7.27, True
            "2026-04-23 09:32",  # bar 3 — 7.94 > 7.67, True
            "2026-04-23 09:35",  # bar 4 (post-halt) — 8.63 > 7.94, True
            "2026-04-23 09:36",  # bar 5 — 9.29 > 8.63, True
            "2026-04-23 09:40",  # bar 6 (post-halt) — 10.10 > 9.29, True
            "2026-04-23 09:41",  # bar 7 — 9.42 < 10.10, False
        ],
        opens=[10, 6.6, 7.0, 7.6, 8.0, 8.3, 9.1, 9.0],
        highs=[15.0, 7.27, 7.67, 7.94, 8.63, 9.29, 10.10, 9.42],
        lows=[9.0, 6.55, 6.91, 7.57, 7.97, 8.24, 9.09, 8.50],
        closes=[12, 7.12, 7.64, 7.97, 8.37, 9.29, 9.17, 8.50],
        volumes=[100] * 8,
    )
    mask = new_high_of_day(bars).tolist()
    # Premarket bar — first of its group, False.
    assert mask[0] is False
    # Bar 1 (9:30) — first market bar, False.
    assert mask[1] is False
    # Bars 2-6 — each a new market-hours HOD.
    assert mask[2:7] == [True, True, True, True, True]
    # Bar 7 (9:42) — below bar 6's 10.10, False.
    assert mask[7] is False


def test_market_hours_session_key_separates_premarket_and_market() -> None:
    """4:00 and 9:29 share 'premarket'; 9:30 and 15:59 share 'market' for the same date."""
    bars = _frame(
        times=[
            "2026-04-16 04:00",
            "2026-04-16 09:29",
            "2026-04-16 09:30",
            "2026-04-16 15:59",
        ],
        opens=[1, 1, 1, 1],
        highs=[1, 1, 1, 1],
        lows=[1, 1, 1, 1],
        closes=[1, 1, 1, 1],
        volumes=[1, 1, 1, 1],
    )
    keys = _market_hours_session_key(bars.index).tolist()
    # Premarket pair matches, market pair matches, and the two are distinct.
    assert keys[0] == keys[1]
    assert keys[2] == keys[3]
    assert keys[0] != keys[2]
    assert keys[0].endswith("-premarket")
    assert keys[2].endswith("-market")


def test_session_key_unchanged_includes_premarket() -> None:
    """Regression: old _session_key still groups premarket + market-hours by calendar date."""
    bars = _frame(
        times=[
            "2026-04-16 04:00",
            "2026-04-16 09:30",
            "2026-04-16 15:59",
            "2026-04-17 09:30",
        ],
        opens=[1, 1, 1, 1],
        highs=[1, 1, 1, 1],
        lows=[1, 1, 1, 1],
        closes=[1, 1, 1, 1],
        volumes=[1, 1, 1, 1],
    )
    keys = _session_key(bars.index).tolist()
    # All three bars on 04-16 share a single key; the 04-17 bar is distinct.
    assert keys[0] == keys[1] == keys[2]
    assert keys[3] != keys[0]


def test_vwap_still_includes_premarket() -> None:
    """Regression: premarket volume flows into VWAP (VWAP uses _session_key, not the 7.2 key)."""
    bars = _frame(
        times=[
            "2026-04-16 04:00",
            "2026-04-16 09:30",
        ],
        opens=[10, 20],
        highs=[10, 20],
        lows=[10, 20],
        closes=[10, 20],
        volumes=[100, 100],
    )
    # With _session_key (both bars same date), VWAP at 9:30 includes the 4:00
    # premarket bar → cumulative VWAP = (10*100 + 20*100) / 200 = 15.
    assert vwap(bars).iloc[-1] == pytest.approx(15.0)


def test_evaluate_hod_returns_context_for_latest_bar() -> None:
    """evaluate_hod exposes last_high, session_hod, bars_in_session + is_new_hod."""
    bars = _frame(
        times=[
            "2026-04-16 04:00",  # premarket — invisible to market-hours HOD
            "2026-04-16 09:30",  # first market bar
            "2026-04-16 09:31",  # breaks 9:30 high
        ],
        opens=[10, 7, 7],
        highs=[20.0, 8.0, 9.0],
        lows=[8, 7, 7],
        closes=[15, 8, 9],
        volumes=[100, 100, 100],
    )
    ctx = evaluate_hod(bars)
    assert ctx is not None
    assert ctx.is_new_hod is True
    assert ctx.last_high == pytest.approx(9.0)
    assert ctx.session_hod == pytest.approx(8.0)  # only the 9:30 market bar
    assert ctx.bars_in_session == 2


def test_evaluate_hod_first_market_bar_has_no_session_hod() -> None:
    """On the first market-hours bar of the day, session_hod is None and is_new_hod is False."""
    bars = _frame(
        times=["2026-04-16 04:00", "2026-04-16 09:30"],
        opens=[10, 7],
        highs=[20.0, 8.0],
        lows=[8, 7],
        closes=[15, 8],
        volumes=[100, 100],
    )
    ctx = evaluate_hod(bars)
    assert ctx is not None
    assert ctx.is_new_hod is False
    assert ctx.session_hod is None
    assert ctx.bars_in_session == 1


# ---------- Phase 9.1: by="close" rejects wick-and-retrace bars ---------- #


def test_new_high_of_day_by_high_default_unchanged() -> None:
    """Backward-compat regression: omitting `by` reproduces the prior high-based mask."""
    bars = _frame(
        times=[
            "2026-04-27 09:30",
            "2026-04-27 09:31",
            "2026-04-27 09:32",
        ],
        opens=[10.0, 10.05, 10.04],
        highs=[10.19, 10.19, 10.30],  # bar 3 wicks above prior high
        lows=[9.95, 10.00, 9.85],
        closes=[10.05, 10.04, 9.89],  # bar 3 closes red, BELOW prior high
        volumes=[100, 100, 100],
    )
    default_mask = new_high_of_day(bars).tolist()
    explicit_mask = new_high_of_day(bars, by="high").tolist()
    assert default_mask == explicit_mask
    # Bar 3 high $10.30 > prior max $10.19 → True under high-based semantics.
    assert default_mask == [False, False, True]


def test_new_high_of_day_by_close_rejects_wick_retrace() -> None:
    """Bar with high>HOD but close<HOD must be False under by='close'."""
    bars = _frame(
        times=[
            "2026-04-27 09:30",
            "2026-04-27 09:31",
            "2026-04-27 09:32",
        ],
        opens=[10.0, 10.05, 10.04],
        highs=[10.19, 10.19, 10.30],
        lows=[9.95, 10.00, 9.85],
        closes=[10.05, 10.04, 9.89],
        volumes=[100, 100, 100],
    )
    high_mask = new_high_of_day(bars, by="high").tolist()
    close_mask = new_high_of_day(bars, by="close").tolist()
    # by=high: bar 3's wick to $10.30 makes new HOD.
    assert high_mask == [False, False, True]
    # by=close: bar 3's $9.89 close fails to confirm — failed breakout.
    assert close_mask == [False, False, False]


def test_new_high_of_day_by_close_accepts_close_above_prior_high() -> None:
    """Bar that closes above prior session HOD confirms breakout under by='close'."""
    bars = _frame(
        times=[
            "2026-04-27 09:30",
            "2026-04-27 09:31",
            "2026-04-27 09:32",
        ],
        opens=[10.20, 10.25, 10.30],
        highs=[10.30, 10.30, 10.50],
        lows=[10.10, 10.20, 10.25],
        closes=[10.25, 10.30, 10.40],  # bar 3 closes $10.40 > prior high $10.30
        volumes=[100, 100, 100],
    )
    close_mask = new_high_of_day(bars, by="close").tolist()
    assert close_mask == [False, False, True]


def test_new_high_of_day_by_close_rmax_scenario() -> None:
    """Replay RMAX 2026-04-27: 9:31 confirms (close above HOD), 9:34 rejects (wick + retrace)."""
    bars = _frame(
        times=[
            "2026-04-27 09:30",
            "2026-04-27 09:31",
            "2026-04-27 09:32",
            "2026-04-27 09:33",
            "2026-04-27 09:34",
        ],
        opens=[9.80, 9.90, 10.05, 10.05, 10.04],
        highs=[9.90, 10.10, 10.19, 10.10, 10.30],
        lows=[9.75, 9.85, 9.95, 9.95, 9.85],
        closes=[9.90, 10.08, 10.10, 10.04, 9.89],
        volumes=[100, 100, 100, 100, 100],
    )
    close_mask = new_high_of_day(bars, by="close").tolist()
    # 9:30 first market bar → False.
    # 9:31 close $10.08 > prior high $9.90 → True (legitimate breakout).
    # 9:32 close $10.10 > prior max $10.10? Strict > so False.
    # 9:33 close $10.04 < prior max $10.19 → False.
    # 9:34 close $9.89 < prior max $10.19 → False (the bug being fixed).
    assert close_mask[0] is False
    assert close_mask[1] is True
    assert close_mask[4] is False


def test_new_high_of_day_running_max_unchanged_across_by_param() -> None:
    """Switching `by` does not affect which bars set the session HOD running max.

    A wick-only bar still raises the running max for subsequent bars to clear,
    even when `by="close"` rejects that bar itself. The HOD is always the
    high-based maximum.
    """
    bars = _frame(
        times=[
            "2026-04-27 09:30",
            "2026-04-27 09:31",  # high $10.30 (wick), close $9.89
            "2026-04-27 09:32",  # close $10.20 — below prior wick, above prior close
        ],
        opens=[10.00, 10.05, 9.95],
        highs=[10.10, 10.30, 10.25],
        lows=[9.95, 9.85, 9.95],
        closes=[10.05, 9.89, 10.20],
        volumes=[100, 100, 100],
    )
    close_mask = new_high_of_day(bars, by="close").tolist()
    # Bar 2 fails (wick-and-retrace). Bar 3's close $10.20 must still be
    # compared against the running max of HIGHS ($10.30), not the running
    # max of closes ($10.05). $10.20 < $10.30 → False.
    assert close_mask == [False, False, False]


def test_evaluate_hod_by_close_rejects_wick_retrace() -> None:
    """evaluate_hod with by='close' returns is_new_hod=False on a wick-and-retrace bar."""
    bars = _frame(
        times=[
            "2026-04-27 09:30",
            "2026-04-27 09:31",
            "2026-04-27 09:32",
        ],
        opens=[10.00, 10.05, 10.04],
        highs=[10.10, 10.19, 10.30],
        lows=[9.95, 10.00, 9.85],
        closes=[10.05, 10.04, 9.89],
        volumes=[100, 100, 100],
    )
    ctx_high = evaluate_hod(bars, by="high")
    ctx_close = evaluate_hod(bars, by="close")
    assert ctx_high is not None
    assert ctx_close is not None
    # Both report the same HIGH-based diagnostic fields.
    assert ctx_high.last_high == pytest.approx(10.30)
    assert ctx_close.last_high == pytest.approx(10.30)
    assert ctx_high.session_hod == pytest.approx(10.19)
    assert ctx_close.session_hod == pytest.approx(10.19)
    # Only the boolean changes.
    assert ctx_high.is_new_hod is True
    assert ctx_close.is_new_hod is False


def test_is_bull_flag_accepts_shallow_pullback() -> None:
    """Impulse to 105 followed by shallow pullback to 103 → bull flag (pullback ~1.9%)."""
    bars = _frame(
        times=[f"2026-04-16 09:{30 + i:02d}" for i in range(10)],
        opens=[100] * 10,
        highs=[102, 104, 105, 104, 104, 103.5, 103.5, 103.2, 103.1, 103.0],
        lows=[100, 102, 103, 103, 103, 102.8, 102.5, 102.8, 102.9, 102.9],
        closes=[101, 103, 105, 104, 103.8, 103.2, 103.0, 103.0, 103.0, 103.0],
        volumes=[100] * 10,
    )
    assert is_bull_flag(bars, max_pullback_pct=5.0, lookback=10) is True


def test_is_bull_flag_rejects_deep_pullback() -> None:
    """Pullback beyond the envelope (here >5%) must not register as a flag."""
    bars = _frame(
        times=[f"2026-04-16 09:{30 + i:02d}" for i in range(10)],
        opens=[100] * 10,
        highs=[110, 110, 110, 105, 103, 101, 99, 97, 96, 95],
        lows=[100] * 10,
        closes=[110, 110, 110, 104, 101, 98, 96, 95, 94, 94],
        volumes=[100] * 10,
    )
    assert is_bull_flag(bars, max_pullback_pct=5.0, lookback=10) is False


def test_is_extension_bar_atr_true_when_close_far_above_vwap() -> None:
    """Close sits many ATRs above VWAP → flagged as extension (skip chasing)."""
    bars = _frame(
        times=[f"2026-04-16 09:{30 + i:02d}" for i in range(16)],
        opens=[10] * 15 + [20],
        highs=[10] * 15 + [21],
        lows=[10] * 15 + [19],
        closes=[10] * 15 + [20],
        volumes=[100] * 16,
    )
    vwap_series = vwap(bars)
    assert is_extension_bar_atr(bars, vwap_series, atr_multiple=3.0) is True


def test_is_extension_bar_dollar_fires_on_clear_extension() -> None:
    """100 shares × $3 high-over-open = $300 ≥ $200 threshold → fires."""
    bar = pd.Series({"open": 10.0, "high": 13.0, "low": 9.9, "close": 12.5})
    assert is_extension_bar_dollar(bar, position_shares=100, dollar_threshold=200.0) is True


def test_is_extension_bar_dollar_silent_on_modest_up_bar() -> None:
    """100 shares × $1 = $100 < $200 threshold → does not fire."""
    bar = pd.Series({"open": 10.0, "high": 11.0, "low": 9.95, "close": 10.8})
    assert is_extension_bar_dollar(bar, position_shares=100, dollar_threshold=200.0) is False


def test_is_extension_bar_dollar_silent_on_red_bar() -> None:
    """Red bar (close < open) never fires regardless of high-over-open range."""
    bar = pd.Series({"open": 10.0, "high": 13.0, "low": 9.0, "close": 9.5})
    assert is_extension_bar_dollar(bar, position_shares=100, dollar_threshold=200.0) is False


# ---------- Phase 8.4: premarket_high helper ---------- #


def test_premarket_high_returns_max_of_premarket_bars() -> None:
    """Highest high across pre-09:30 bars on the latest session date."""
    from bot.indicators import premarket_high

    bars = _frame(
        times=[
            "2026-04-16 04:00",  # premarket
            "2026-04-16 06:30",  # premarket
            "2026-04-16 09:00",  # premarket
            "2026-04-16 09:30",  # market open
            "2026-04-16 10:30",  # market hours
        ],
        opens=[1.0, 1.0, 1.0, 1.0, 1.0],
        highs=[1.20, 1.55, 1.30, 1.45, 1.60],  # PMH = 1.55 at 06:30
        lows=[0.9, 0.9, 0.9, 0.9, 0.9],
        closes=[1.1, 1.5, 1.2, 1.4, 1.5],
        volumes=[100, 100, 100, 100, 100],
    )
    assert premarket_high(bars) == pytest.approx(1.55)


def test_premarket_high_returns_none_with_no_premarket_bars() -> None:
    """Frame with only market-hours bars → None."""
    from bot.indicators import premarket_high

    bars = _frame(
        times=["2026-04-16 09:30", "2026-04-16 10:00"],
        opens=[10.0, 10.0],
        highs=[10.5, 10.8],
        lows=[9.9, 10.0],
        closes=[10.4, 10.6],
        volumes=[100, 100],
    )
    assert premarket_high(bars) is None


def test_premarket_high_returns_none_for_empty_frame() -> None:
    """Empty frame → None (defensive)."""
    from bot.indicators import premarket_high

    bars = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    assert premarket_high(bars) is None


def test_premarket_high_only_considers_latest_session_date() -> None:
    """Premarket bars from a previous date must not contaminate today's PMH."""
    from bot.indicators import premarket_high

    bars = _frame(
        times=[
            "2026-04-15 04:00",  # PRIOR date premarket — must be excluded
            "2026-04-16 04:00",  # today's premarket
            "2026-04-16 09:30",  # today's market open
        ],
        opens=[1.0, 1.0, 1.0],
        highs=[5.00, 1.50, 1.40],  # 5.00 from prior date should NOT win
        lows=[0.9, 0.9, 0.9],
        closes=[4.5, 1.45, 1.35],
        volumes=[100, 100, 100],
    )
    assert premarket_high(bars) == pytest.approx(1.50)

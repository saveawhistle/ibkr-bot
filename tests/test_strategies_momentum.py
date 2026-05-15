"""Tests for ``bot.strategies.momentum.MomentumStrategy``."""

from __future__ import annotations

import pandas as pd
import pytest
from structlog.testing import capture_logs

from bot.strategies.momentum import MomentumStrategy


def _frame(
    times: list[str],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float] | None = None,
) -> pd.DataFrame:
    """Build a NY-tz 1-min bar DataFrame matching MarketData shape."""
    idx = pd.to_datetime(times).tz_localize("America/New_York")
    vols = volumes or [1_000.0] * len(times)
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vols,
            "vwap": closes,
        },
        index=idx,
    )


def _times(start_minute: int = 30, count: int = 10) -> list[str]:
    """Build timestamps starting at 10:30 + ``start_minute - 30`` minutes.

    Phase 12.6: momentum's default window now starts at 10:00 ET, so
    pre-12.6 fixtures stamped at 09:30 silently dropped under
    ``_within_window``. Fixture moves to 10:30 to land cleanly inside
    the 10:00-11:30 default window. The ``start_minute`` argument
    semantics are preserved (callers passing 30 still get a 30-bar
    offset from 10:00) for back-compat with the dozen call sites.
    """
    base_minute = 30  # baseline 10:30 anchor
    offset = start_minute - 30  # caller's offset relative to the legacy 30 default
    out: list[str] = []
    for i in range(count):
        total = base_minute + offset + i
        hour = 10 + total // 60
        minute = total % 60
        out.append(f"2026-04-16 {hour:02d}:{minute:02d}")
    return out


def test_emits_signal_on_bull_flag_hod_break() -> None:
    """Shallow pullback after an impulse followed by an HOD-breaking bar → momentum signal."""
    # Impulse to 10.50 in first 3 bars, shallow pullback to 10.30, then break to 10.60.
    bars = _frame(
        times=_times(30, 10),
        highs=[10.3, 10.5, 10.5, 10.4, 10.4, 10.35, 10.35, 10.32, 10.4, 10.6],
        lows=[10.0, 10.3, 10.3, 10.3, 10.25, 10.25, 10.3, 10.3, 10.3, 10.35],
        closes=[10.2, 10.45, 10.5, 10.4, 10.3, 10.3, 10.32, 10.32, 10.35, 10.6],
    )
    strategy = MomentumStrategy(flag_max_pullback_pct=5.0)
    signal = strategy.evaluate("MOVE", bars)
    assert signal is not None
    assert signal.strategy == "momentum"
    assert signal.entry == pytest.approx(10.6)
    assert signal.stop < signal.entry
    risk = signal.entry - signal.stop
    # Phase 4i: scale_out = entry + scale_out_multiple × initial_risk (default 2R).
    # runner_target_price is None — executor populates it only when enabled.
    assert signal.scale_out_price == pytest.approx(signal.entry + 2.0 * risk)
    assert signal.runner_target_price is None


def test_no_signal_outside_11_30_window() -> None:
    """A bar at 11:35 ET is past the momentum window — no signal."""
    bars = _frame(
        times=[f"2026-04-16 11:{31 + i:02d}" for i in range(10)],
        highs=[10.3, 10.5, 10.5, 10.4, 10.3, 10.35, 10.32, 10.32, 10.35, 10.6],
        lows=[10.0] * 10,
        closes=[10.2, 10.5, 10.5, 10.4, 10.3, 10.3, 10.32, 10.32, 10.35, 10.6],
    )
    strategy = MomentumStrategy()
    assert strategy.evaluate("MOVE", bars) is None


def test_no_signal_when_pullback_too_deep() -> None:
    """Pullback greater than ``flag_max_pullback_pct`` must suppress the signal."""
    bars = _frame(
        times=_times(30, 10),
        highs=[10.3, 10.5, 10.5, 10.0, 9.5, 9.5, 9.2, 9.0, 9.5, 10.6],
        lows=[10.0, 10.3, 10.3, 9.5, 9.0, 9.0, 8.8, 8.6, 9.0, 9.0],
        closes=[10.2, 10.5, 10.5, 9.8, 9.3, 9.1, 9.0, 9.0, 9.2, 10.6],
    )
    strategy = MomentumStrategy(flag_max_pullback_pct=1.0)  # too strict for the 13%+ pullback
    assert strategy.evaluate("MOVE", bars) is None


# ---------- Phase 5.5: momentum must NOT inherit gap_and_go's grace period ----------


def test_momentum_unaffected_by_grace_period() -> None:
    """A 9:39 extended-from-VWAP bar must still be rejected by momentum — no grace.

    Phase 5.5 added the VWAP-extension grace window to gap_and_go only.
    Momentum's extension check serves a different purpose (catching ongoing
    intraday overextension) and intentionally applies at every bar. Constructs
    a bull-flag frame with a final breakout bar that is >3× ATR above VWAP to
    force the extension branch and verify the rejection still fires.
    """
    # 10 bars at 9:30–9:39: tight impulse + shallow flag + spike-breakout last bar.
    # Breakout bar jumps to 12.0 while prior 9 bars hover around 10.2–10.5, so
    # (last_close − vwap) blows past 3× ATR.
    bars = _frame(
        times=_times(30, 10),
        highs=[10.3, 10.4, 10.5, 10.45, 10.4, 10.4, 10.4, 10.45, 10.5, 12.0],
        lows=[10.1, 10.25, 10.3, 10.3, 10.3, 10.3, 10.3, 10.3, 10.35, 11.7],
        closes=[10.25, 10.35, 10.45, 10.4, 10.35, 10.35, 10.35, 10.4, 10.45, 11.9],
    )
    strategy = MomentumStrategy()
    with capture_logs() as captured:
        signal = strategy.evaluate("MOVE", bars)
    assert signal is None, "momentum must reject extended-from-VWAP bars even near the open"
    rejected = [
        e
        for e in captured
        if e.get("event") == "signal.rejected" and e.get("reason") == "extended_from_vwap"
    ]
    assert len(rejected) == 1
    # And momentum must NOT emit the gap_and_go-specific bypass event.
    assert not any(e.get("event") == "gap_and_go.vwap_extension_bypassed" for e in captured)


# ---------- Phase 6.6: tunable extension threshold ---------- #


def test_momentum_extension_threshold_configurable() -> None:
    """Same ~4× ATR distance bar: rejects at 3.0×, passes at 5.0× (the new default).

    Mirrors ``test_gap_and_go_extension_threshold_configurable`` for the
    momentum strategy. Constructed so the breakout bar's distance from
    VWAP sits between 3× and 5× ATR — the rejection should toggle on the
    multiple alone.
    """
    # Tight bull-flag (~10.30, ATR ~0.05) then a moderate breakout to 10.55
    # — distance ≈ 0.20–0.25 above VWAP ≈ 4–5× ATR.
    bars = _frame(
        times=_times(30, 10),
        highs=[10.32, 10.34, 10.35, 10.34, 10.32, 10.32, 10.32, 10.33, 10.35, 10.56],
        lows=[10.28, 10.30, 10.31, 10.30, 10.28, 10.28, 10.28, 10.29, 10.31, 10.51],
        closes=[10.30, 10.32, 10.34, 10.32, 10.30, 10.30, 10.30, 10.31, 10.33, 10.55],
    )

    # multiple=3.0 → should reject extension.
    with capture_logs() as cap_3:
        s3 = MomentumStrategy(extended_from_vwap_atr_multiple=3.0).evaluate("MOVE", bars)
    assert s3 is None
    assert any(
        e.get("event") == "signal.rejected" and e.get("reason") == "extended_from_vwap"
        for e in cap_3
    )

    # multiple=5.0 → extension check passes; downstream filters may still
    # reject for unrelated reasons but ``extended_from_vwap`` must not fire.
    with capture_logs() as cap_5:
        MomentumStrategy(extended_from_vwap_atr_multiple=5.0).evaluate("MOVE", bars)
    assert not any(
        e.get("event") == "signal.rejected" and e.get("reason") == "extended_from_vwap"
        for e in cap_5
    ), "extended_from_vwap must NOT fire at 5.0× (the new default)"


def test_momentum_window_end_configurable() -> None:
    """Bars past the default 11:30 cutoff evaluate only when window_end is widened."""
    from datetime import time as time_cls

    # Same bull-flag setup as ``test_emits_signal_on_bull_flag_hod_break``
    # but stamped at 14:00 ET — way past the default 11:30 cutoff.
    bars = _frame(
        times=[f"2026-04-16 14:{i:02d}" for i in range(10)],
        highs=[10.3, 10.5, 10.5, 10.4, 10.4, 10.35, 10.35, 10.32, 10.4, 10.6],
        lows=[10.0, 10.3, 10.3, 10.3, 10.25, 10.25, 10.3, 10.3, 10.3, 10.35],
        closes=[10.2, 10.45, 10.5, 10.4, 10.3, 10.3, 10.32, 10.32, 10.35, 10.6],
    )

    # Default window_end=11:30 → silent None, outside window.
    assert MomentumStrategy().evaluate("MOVE", bars) is None

    # Widened window_end=16:00 → same 14:09 bar evaluates. The frame is
    # the known-good bull-flag setup, so we expect a signal.
    widened = MomentumStrategy(window_end=time_cls(16, 0))
    signal = widened.evaluate("MOVE", bars)
    assert signal is not None, "widening window_end to 16:00 must evaluate the 14:09 bar"
    assert signal.strategy == "momentum"


# ---------- Phase 7.1: market-hours filter guards the flag window ---------- #


def test_momentum_stop_ignores_premarket_bars() -> None:
    """A 4 AM premarket wick at $5.00 must not leak into the flag_low stop reference.

    Pre-7.1, ``bars.iloc[-_FLAG_LOOKBACK:]`` could reach into premarket when
    evaluating within the first ~10 market bars. After 7.1, the 10-bar slice
    is sourced from a market-hours-only session, so premarket wicks can never
    anchor the stop regardless of how early in the session we evaluate.
    """
    bars = _frame(
        times=["2026-04-16 04:00"] + _times(30, 10),
        highs=[9.0, 10.3, 10.5, 10.5, 10.4, 10.4, 10.35, 10.35, 10.32, 10.4, 10.6],
        lows=[5.0, 10.0, 10.3, 10.3, 10.3, 10.25, 10.25, 10.3, 10.3, 10.3, 10.35],
        closes=[9.0, 10.2, 10.45, 10.5, 10.4, 10.3, 10.3, 10.32, 10.32, 10.35, 10.6],
        volumes=[100] + [1000] * 10,
    )
    strategy = MomentumStrategy()
    signal = strategy.evaluate("MOVE", bars)
    assert signal is not None
    # Consolidation window (6 bars) excludes both the premarket bar AND the
    # impulse bars — stop is pinned to consolidation_low 10.25, never 5.0.
    assert signal.pullback_low == pytest.approx(10.25)
    assert signal.stop > 5.0
    assert signal.bars_available_for_lookback == 6


def test_momentum_signal_emitted_carries_observability_fields() -> None:
    """Momentum emits ``signal.emitted`` with the same diagnostic schema as gap_and_go."""
    bars = _frame(
        times=_times(30, 10),
        highs=[10.3, 10.5, 10.5, 10.4, 10.4, 10.35, 10.35, 10.32, 10.4, 10.6],
        lows=[10.0, 10.3, 10.3, 10.3, 10.25, 10.25, 10.3, 10.3, 10.3, 10.35],
        closes=[10.2, 10.45, 10.5, 10.4, 10.3, 10.3, 10.32, 10.32, 10.35, 10.6],
    )
    strategy = MomentumStrategy()
    with capture_logs() as captured:
        signal = strategy.evaluate("MOVE", bars)
    assert signal is not None
    emitted = [e for e in captured if e.get("event") == "signal.emitted"]
    assert len(emitted) == 1
    evt = emitted[0]
    assert evt["symbol"] == "MOVE"
    assert evt["strategy"] == "momentum"
    assert evt["pullback_low"] == pytest.approx(signal.pullback_low)
    assert evt["pullback_lookback_bars"] == 6  # Phase 14: scoped to consolidation window
    assert evt["bars_available_for_lookback"] == 6
    assert evt["vwap_at_entry"] == pytest.approx(signal.vwap_at_entry)


def test_not_new_hod_rejection_includes_hod_fields() -> None:
    """Phase 14: bar below session HOD that lacks a valid pattern rejects as no_momentum_pattern.

    Pre-Phase-14, any close below session HOD triggered a not_new_hod rejection.
    Phase 14 treats that case as a standing-order candidate and only rejects
    not_new_hod for the specific wick-and-retrace scenario (bar high pierces
    trigger but close fails). A deep pullback with no matching pattern resolves
    to no_momentum_pattern instead.
    """
    # 10 bars — session HOD is 10.8 (bar 2); last bar's close 10.35 is well below.
    # Consolidation pulls back >80% of the pole — no pattern matches.
    bars = _frame(
        times=_times(30, 10),
        highs=[10.3, 10.5, 10.8, 10.6, 10.5, 10.5, 10.4, 10.35, 10.3, 10.4],
        lows=[10.0, 10.2, 10.3, 10.3, 10.25, 10.25, 10.25, 10.25, 10.25, 10.3],
        closes=[10.2, 10.4, 10.7, 10.5, 10.4, 10.4, 10.35, 10.3, 10.3, 10.35],
    )
    strategy = MomentumStrategy()
    with capture_logs() as captured:
        signal = strategy.evaluate("AUUD", bars)
    assert signal is None
    rejections = [
        e
        for e in captured
        if e.get("event") == "signal.rejected" and e.get("reason") == "no_momentum_pattern"
    ]
    assert len(rejections) == 1
    evt = rejections[0]
    assert evt["is_standing_order"] is True
    assert evt["trigger_level"] == pytest.approx(10.8)


def test_extension_ratio_computed_correctly() -> None:
    """Math + boundary check on ``ExtensionCheck``.

    Two assertions:

    (a) **Internal consistency**: for any input where all fields are
        populated, ``extension_ratio == distance_from_vwap /
        threshold_distance``. The strategy-level rejection event uses
        these three fields independently; if the math drifts, operator
        calibration grep will read inconsistent ratios.

    (b) **Strict greater-than at the boundary**: the prior
        ``is_extension_bar_atr`` used ``distance > threshold`` (strict).
        Phase 6.6 must preserve that — equality is NOT extended.
    """
    from bot.indicators import evaluate_extension, vwap

    # Build a small frame and assert the math against the actual returned
    # values (not externally re-derived VWAP/ATR — those depend on the
    # close column, so rewriting close-then-recomputing is circular).
    bars = _frame(
        times=_times(30, 4),
        highs=[10.10, 10.10, 10.10, 10.40],
        lows=[9.90, 9.90, 9.90, 10.30],
        closes=[10.0, 10.0, 10.0, 10.35],
    )
    vwap_series = vwap(bars)
    check = evaluate_extension(bars, vwap_series, atr_multiple=5.0)
    assert check.last_atr_value is not None
    assert check.distance_from_vwap is not None
    assert check.threshold_distance is not None
    assert check.extension_ratio is not None
    # Internal-consistency invariant: ratio = distance / threshold.
    assert check.extension_ratio == pytest.approx(
        check.distance_from_vwap / check.threshold_distance, rel=1e-9
    )
    # Threshold = multiple × ATR (the documented relationship).
    assert check.threshold_distance == pytest.approx(5.0 * check.last_atr_value, rel=1e-9)
    # ``extended`` flag matches strict greater-than against the threshold.
    assert check.extended == (check.distance_from_vwap > check.threshold_distance)

    # Boundary at ratio == 1.0: setting close = VWAP + 5×ATR is fragile
    # (rewriting close shifts VWAP). Instead, walk the multiple to where
    # threshold equals the observed distance — same effect, single call.
    assert check.distance_from_vwap > 0
    equality_multiple = check.distance_from_vwap / check.last_atr_value
    boundary = evaluate_extension(bars, vwap_series, atr_multiple=equality_multiple)
    # Equality must NOT flag extended — strict greater-than.
    assert boundary.extended is False
    assert boundary.extension_ratio == pytest.approx(1.0, rel=1e-9)
    # One tick under (smaller threshold) — distance now exceeds threshold,
    # so the check rejects.
    over = evaluate_extension(bars, vwap_series, atr_multiple=equality_multiple - 1e-6)
    assert over.extended is True


# ---------- Phase 9.1: close-based HOD confirmation ---------- #


def test_rejects_wick_and_retrace_breakout() -> None:
    """Bar wicks above prior HOD then closes red — momentum must reject, not enter.

    Same wick-and-retrace pattern as the gap-and-go test but with the bull-flag
    consolidation pre-context that momentum requires. The trigger bar's high
    pierces prior HOD; close lands well below it — Phase 9.1's ``by="close"``
    rejects.
    """
    bars = _frame(
        times=_times(30, 10),
        # Impulse 10.30→10.50, shallow pullback to 10.32, trigger bar wicks to
        # 10.60 but closes at 10.20 (red, below prior HOD of 10.50).
        highs=[10.30, 10.50, 10.50, 10.40, 10.40, 10.35, 10.35, 10.32, 10.40, 10.60],
        lows=[10.00, 10.30, 10.30, 10.30, 10.25, 10.25, 10.30, 10.30, 10.30, 10.18],
        closes=[10.20, 10.45, 10.50, 10.40, 10.30, 10.30, 10.32, 10.32, 10.35, 10.20],
    )
    strategy = MomentumStrategy(flag_max_pullback_pct=5.0)
    with capture_logs() as captured:
        signal = strategy.evaluate("MOVE", bars)
    assert signal is None
    rejections = [
        e
        for e in captured
        if e.get("event") == "signal.rejected" and e.get("reason") == "not_new_hod"
    ]
    assert len(rejections) == 1
    evt = rejections[0]
    assert evt["last_high"] == pytest.approx(10.60)
    assert evt["last_close"] == pytest.approx(10.20)
    assert evt["session_hod"] == pytest.approx(10.50)


# ---------- Phase 14: new pattern + standing-order tests ---------- #


def test_bull_flag_scoped_stop() -> None:
    """Stop equals the consolidation low, not the impulse low.

    The impulse drops to 10.00 but the consolidation window floor is 10.25.
    Phase 14 scopes the stop to the consolidation bars only, so the impulse
    wick at 10.00 must not anchor the stop.
    """
    bars = _frame(
        times=_times(30, 10),
        highs=[10.3, 10.5, 10.5, 10.4, 10.4, 10.35, 10.35, 10.32, 10.4, 10.6],
        lows=[10.0, 10.3, 10.3, 10.3, 10.25, 10.25, 10.3, 10.3, 10.3, 10.35],
        closes=[10.2, 10.45, 10.5, 10.4, 10.3, 10.3, 10.32, 10.32, 10.35, 10.6],
    )
    strategy = MomentumStrategy()
    signal = strategy.evaluate("MOVE", bars)
    assert signal is not None
    # Consolidation bars (3-8) have min low = 10.25 — impulse low 10.0 excluded.
    assert signal.pullback_low == pytest.approx(10.25)
    assert signal.stop <= 10.25


def test_vwap_not_held_during_consolidation_rejected() -> None:
    """Consolidation bar closing below its VWAP → vwap_not_held_during_consolidation rejection."""
    from structlog.testing import capture_logs

    from bot.strategies.entry_quality import check_consolidation_vwap_hold

    bars = _frame(
        times=_times(30, 10),
        highs=[10.3, 10.5, 10.5, 10.4, 10.4, 10.35, 10.35, 10.32, 10.4, 10.6],
        lows=[10.0, 10.3, 10.3, 10.3, 10.25, 10.25, 10.3, 10.3, 10.3, 10.35],
        closes=[10.2, 10.45, 10.5, 10.4, 10.3, 10.3, 10.32, 10.32, 10.35, 10.6],
    )
    bar_time = bars.index[-1]
    result = check_consolidation_vwap_hold(
        vwap_hold=False,
        pattern_type="bull_flag",
        consolidation_low=10.25,
        symbol="TEST",
        strategy="momentum",
        bar_time=bar_time,
    )
    assert result == "vwap_not_held_during_consolidation"

    result_pass = check_consolidation_vwap_hold(
        vwap_hold=True,
        pattern_type="bull_flag",
        consolidation_low=10.25,
        symbol="TEST",
        strategy="momentum",
        bar_time=bar_time,
    )
    assert result_pass is None


def test_insufficient_breakout_volume_rejected() -> None:
    """Breakout bar volume below 1.5× consolidation avg → signal suppressed."""
    from bot.strategies.entry_quality import check_breakout_volume_ratio

    # Breakout vol = 1 000; consolidation avg = 1 000 → ratio = 1.0 < 1.5.
    bars = _frame(
        times=_times(30, 10),
        highs=[10.3, 10.5, 10.5, 10.4, 10.4, 10.35, 10.35, 10.32, 10.4, 10.6],
        lows=[10.0, 10.3, 10.3, 10.3, 10.25, 10.25, 10.3, 10.3, 10.3, 10.35],
        closes=[10.2, 10.45, 10.5, 10.4, 10.3, 10.3, 10.32, 10.32, 10.35, 10.6],
        volumes=[1_000.0] * 10,
    )
    result = check_breakout_volume_ratio(
        bars=bars,
        consolidation_window_bars=6,
        min_ratio=1.5,
        symbol="TEST",
        strategy="momentum",
        bar_time=bars.index[-1],
    )
    assert result == "insufficient_breakout_volume"


def test_micro_pullback_signal_emitted() -> None:
    """Tight consolidation (< 2% total range, > 0.5% high range) → 'micro_pullback' in reasons."""
    # Consolidation highs span 10.50-10.57 (0.66% range → clears flat_top threshold);
    # total range (10.57 - 10.48) / 10.57 = 0.85% ≤ 2% → micro_pullback.
    bars = _frame(
        times=_times(30, 10),
        highs=[10.2, 10.4, 10.52, 10.57, 10.50, 10.52, 10.52, 10.53, 10.54, 10.62],
        lows=[10.0, 10.2, 10.38, 10.51, 10.48, 10.50, 10.50, 10.50, 10.50, 10.58],
        closes=[10.1, 10.35, 10.50, 10.55, 10.49, 10.51, 10.51, 10.52, 10.53, 10.60],
    )
    strategy = MomentumStrategy()
    signal = strategy.evaluate("MOVE", bars)
    assert signal is not None
    assert "micro_pullback" in signal.reasons


def test_flat_top_signal_emitted() -> None:
    """Consolidation highs within 0.5% of each other → pattern_type 'flat_top' in signal.reasons."""
    # Impulse 10.0→10.50; 6 bars all pressing against the 10.50 ceiling.
    bars = _frame(
        times=_times(30, 10),
        highs=[10.20, 10.40, 10.502, 10.502, 10.501, 10.500, 10.501, 10.502, 10.502, 10.56],
        lows=[10.0, 10.20, 10.40, 10.48, 10.48, 10.48, 10.48, 10.48, 10.48, 10.53],
        closes=[10.1, 10.35, 10.50, 10.495, 10.492, 10.490, 10.492, 10.495, 10.498, 10.55],
    )
    strategy = MomentumStrategy()
    signal = strategy.evaluate("MOVE", bars)
    assert signal is not None
    assert "flat_top" in signal.reasons


def test_standing_stp_lmt_emitted_when_close_below_trigger() -> None:
    """Close below session HOD → STP_LMT signal at trigger level, not at close."""
    # Session HOD (by high) established in bars 0-8, last bar close is below it.
    bars = _frame(
        times=_times(30, 10),
        highs=[10.2, 10.4, 10.52, 10.505, 10.505, 10.50, 10.50, 10.505, 10.51, 10.48],
        lows=[10.0, 10.2, 10.40, 10.48, 10.48, 10.48, 10.48, 10.48, 10.49, 10.40],
        closes=[10.1, 10.35, 10.50, 10.49, 10.49, 10.49, 10.49, 10.49, 10.50, 10.45],
    )
    strategy = MomentumStrategy()
    signal = strategy.evaluate("MOVE", bars)
    assert signal is not None
    assert signal.preferred_order_type == "STP_LMT"
    # Entry must equal the trigger level (session HOD by high), not the last close.
    assert signal.entry > 10.45  # trigger > close


def test_standing_order_not_when_close_above_trigger() -> None:
    """Close exceeds session HOD → normal LMT, preferred_order_type is None."""
    bars = _frame(
        times=_times(30, 10),
        highs=[10.3, 10.5, 10.5, 10.4, 10.4, 10.35, 10.35, 10.32, 10.4, 10.6],
        lows=[10.0, 10.3, 10.3, 10.3, 10.25, 10.25, 10.3, 10.3, 10.3, 10.35],
        closes=[10.2, 10.45, 10.5, 10.4, 10.3, 10.3, 10.32, 10.32, 10.35, 10.6],
    )
    strategy = MomentumStrategy()
    signal = strategy.evaluate("MOVE", bars)
    assert signal is not None
    assert signal.preferred_order_type is None

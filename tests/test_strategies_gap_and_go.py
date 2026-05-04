"""Tests for ``bot.strategies.gap_and_go.GapAndGoStrategy``."""

from __future__ import annotations

import pandas as pd
import pytest
from structlog.testing import capture_logs

from bot.strategies.gap_and_go import GapAndGoStrategy


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


def test_emits_signal_when_new_hod_closes_above_vwap() -> None:
    """Rising bars that break HOD and close above VWAP should trigger a gap-and-go entry."""
    bars = _frame(
        times=[
            "2026-04-16 09:30",
            "2026-04-16 09:31",
            "2026-04-16 09:32",
            "2026-04-16 09:33",
        ],
        highs=[10.0, 10.2, 10.4, 10.8],
        lows=[9.8, 10.0, 10.1, 10.3],
        closes=[10.0, 10.1, 10.3, 10.7],
        volumes=[1000, 1000, 1000, 1000],
    )
    strategy = GapAndGoStrategy()
    signal = strategy.evaluate("TEST", bars)
    assert signal is not None
    assert signal.strategy == "gap_and_go"
    assert signal.entry == pytest.approx(10.7)
    assert signal.stop < signal.entry
    # Phase 4i: scale-out is entry + scale_out_multiple × initial_risk
    # (default 2.0 — the 2:1 R:R rule). ``runner_target_price`` is left None
    # by the strategy; the executor populates it only when
    # ``execution.runner_target_enabled`` is true.
    risk = signal.entry - signal.stop
    assert signal.scale_out_price == pytest.approx(signal.entry + 2.0 * risk)
    assert signal.runner_target_price is None
    # Phase 8.1: R:R floor removed — observability only. Signal's
    # risk_reward property is preserved and surfaced in logs.
    assert signal.risk_reward == pytest.approx(2.0)


def test_no_signal_outside_window() -> None:
    """A bar stamped at 10:30 ET is outside the 9:30-10:00 window — no signal."""
    bars = _frame(
        times=[
            "2026-04-16 10:28",
            "2026-04-16 10:29",
            "2026-04-16 10:30",
        ],
        highs=[10, 10.1, 10.5],
        lows=[9.9, 10.0, 10.2],
        closes=[10.0, 10.1, 10.4],
    )
    strategy = GapAndGoStrategy()
    assert strategy.evaluate("TEST", bars) is None


def test_no_signal_below_vwap() -> None:
    """A last-bar close below VWAP fails the hold-above-VWAP filter."""
    bars = _frame(
        times=[
            "2026-04-16 09:30",
            "2026-04-16 09:31",
            "2026-04-16 09:32",
        ],
        highs=[10.0, 10.1, 10.0],
        lows=[9.5, 9.6, 9.2],
        closes=[9.8, 9.9, 9.4],
    )
    strategy = GapAndGoStrategy()
    assert strategy.evaluate("TEST", bars) is None


# ---------- Phase 5.5: vwap_extension_grace_minutes ----------


def _extension_frame(last_minute: int) -> pd.DataFrame:
    """4-bar frame ending at ``09:<last_minute>`` with an extended-from-VWAP final bar.

    Three tight bars (~10.0, low ATR) followed by a spike to 11.4; this pushes
    the last-bar close well beyond 3× ATR above the volume-weighted mean, so
    ``is_extension_bar_atr`` fires. HOD and above-VWAP conditions are satisfied
    by construction, so the only gate blocking a signal is the extension check.
    """
    times = [f"2026-04-16 09:{last_minute - 3 + i:02d}" for i in range(4)]
    return _frame(
        times=times,
        highs=[10.05, 10.06, 10.08, 11.5],
        lows=[9.95, 9.98, 10.00, 11.3],
        closes=[10.00, 10.02, 10.04, 11.4],
        volumes=[1000, 1000, 1000, 1000],
    )


def test_gap_and_go_bypasses_vwap_extension_during_grace() -> None:
    """A 9:35 bar (within the 15-min grace) must bypass the extension check and emit a signal."""
    bars = _extension_frame(last_minute=35)
    strategy = GapAndGoStrategy()  # default 15-min grace
    with capture_logs() as captured:
        signal = strategy.evaluate("TEST", bars)
    assert signal is not None, "grace window should let an extended bar through"
    rejected = [e for e in captured if e.get("event") == "signal.rejected"]
    assert not any(e.get("reason") == "extended_from_vwap" for e in rejected)
    bypass = [e for e in captured if e.get("event") == "gap_and_go.vwap_extension_bypassed"]
    assert len(bypass) == 1
    evt = bypass[0]
    assert evt["symbol"] == "TEST"
    assert evt["minutes_since_open"] == 5
    assert evt["last_close"] == pytest.approx(11.4)
    assert evt["vwap"] > 10.0
    assert evt["atr"] is not None and evt["atr"] > 0.0


def test_gap_and_go_applies_vwap_extension_after_grace() -> None:
    """A 9:50 bar (past the 15-min grace) must still fire the extension rejection."""
    bars = _extension_frame(last_minute=50)
    strategy = GapAndGoStrategy()
    with capture_logs() as captured:
        signal = strategy.evaluate("TEST", bars)
    assert signal is None
    rejected = [
        e
        for e in captured
        if e.get("event") == "signal.rejected" and e.get("reason") == "extended_from_vwap"
    ]
    assert len(rejected) == 1
    assert not any(e.get("event") == "gap_and_go.vwap_extension_bypassed" for e in captured)


def test_gap_and_go_grace_period_configurable() -> None:
    """Grace window honours the configured minutes: wider grace bypasses later bars."""
    # grace=30: a 9:50 bar sits at minute 20, inside the wider grace → bypass.
    strategy_wide = GapAndGoStrategy(vwap_extension_grace_minutes=30)
    with capture_logs() as captured_wide:
        signal_wide = strategy_wide.evaluate("TEST", _extension_frame(last_minute=50))
    assert signal_wide is not None
    assert any(e.get("event") == "gap_and_go.vwap_extension_bypassed" for e in captured_wide)

    # grace=5: a 9:40 bar sits at minute 10, past the tighter grace → rejection fires.
    strategy_tight = GapAndGoStrategy(vwap_extension_grace_minutes=5)
    with capture_logs() as captured_tight:
        signal_tight = strategy_tight.evaluate("TEST", _extension_frame(last_minute=40))
    assert signal_tight is None
    assert any(
        e.get("event") == "signal.rejected" and e.get("reason") == "extended_from_vwap"
        for e in captured_tight
    )


def test_gap_and_go_grace_period_zero_disables() -> None:
    """``vwap_extension_grace_minutes=0`` restores pre-5.5 behaviour — no bypass at any time."""
    bars = _extension_frame(last_minute=35)  # 5 min past open, would be in default grace
    strategy = GapAndGoStrategy(vwap_extension_grace_minutes=0)
    with capture_logs() as captured:
        signal = strategy.evaluate("TEST", bars)
    assert signal is None
    rejected = [
        e
        for e in captured
        if e.get("event") == "signal.rejected" and e.get("reason") == "extended_from_vwap"
    ]
    assert len(rejected) == 1
    assert not any(e.get("event") == "gap_and_go.vwap_extension_bypassed" for e in captured)


# ---------- Phase 6.6: tunable extension threshold + enriched rejection ---------- #


def _calibration_frame(last_minute: int) -> pd.DataFrame:
    """4-bar frame whose final bar's distance from VWAP is between 3× and 5× ATR.

    Constructed for the threshold-tunability tests — at multiple=3.0 the
    final bar should reject; at multiple=5.0 it should pass; at 4.0 it
    should also pass. Bars are deliberately tight (range ~0.05) so ATR
    sits near 0.05, then a final bar pushes the close ~$0.20 above the
    cluster — distance / VWAP-distance lands in the 3-5× ATR band.
    """
    times = [f"2026-04-16 09:{last_minute - 3 + i:02d}" for i in range(4)]
    return _frame(
        times=times,
        highs=[10.05, 10.07, 10.08, 10.32],
        lows=[10.00, 10.02, 10.03, 10.27],
        closes=[10.02, 10.04, 10.05, 10.31],
        volumes=[1000, 1000, 1000, 1000],
    )


def test_gap_and_go_extension_threshold_configurable() -> None:
    """Same 4.5× distance bar: rejects at 3.0×, passes at 4.0× and 5.0×."""
    bars = _calibration_frame(last_minute=50)  # past 15-min default grace

    # multiple=3.0 → rejects.
    with capture_logs() as cap_3:
        s3 = GapAndGoStrategy(extended_from_vwap_atr_multiple=3.0).evaluate("TEST", bars)
    assert s3 is None
    rej3 = [
        e
        for e in cap_3
        if e.get("event") == "signal.rejected" and e.get("reason") == "extended_from_vwap"
    ]
    assert len(rej3) == 1, "extended_from_vwap rejection must fire at 3.0×"

    # multiple=4.0 → passes the extension check (signal may still emit or
    # be rejected for a different reason; the assertion is only that
    # extended_from_vwap does not fire).
    with capture_logs() as cap_4:
        GapAndGoStrategy(extended_from_vwap_atr_multiple=4.0).evaluate("TEST", bars)
    assert not any(
        e.get("event") == "signal.rejected" and e.get("reason") == "extended_from_vwap"
        for e in cap_4
    ), "extended_from_vwap must NOT fire at 4.0×"

    # multiple=5.0 → also passes.
    with capture_logs() as cap_5:
        GapAndGoStrategy(extended_from_vwap_atr_multiple=5.0).evaluate("TEST", bars)
    assert not any(
        e.get("event") == "signal.rejected" and e.get("reason") == "extended_from_vwap"
        for e in cap_5
    ), "extended_from_vwap must NOT fire at 5.0× (the new default)"


def test_rejection_event_includes_atr_fields() -> None:
    """Forced extension rejection carries last_atr_value, atr_multiple, distance, threshold, ratio.

    The Phase 6.6 enriched schema is documented in the prompt; downstream
    log consumers (calibration grep, dashboard tile) read these fields to
    reason about the threshold without re-deriving the math from raw
    close + vwap. Backward-compat: ``last_close`` / ``vwap`` are preserved.
    """
    bars = _extension_frame(last_minute=50)  # past grace, well over threshold
    strategy = GapAndGoStrategy(extended_from_vwap_atr_multiple=5.0)
    with capture_logs() as captured:
        signal = strategy.evaluate("TEST", bars)
    assert signal is None
    rejected = [
        e
        for e in captured
        if e.get("event") == "signal.rejected" and e.get("reason") == "extended_from_vwap"
    ]
    assert len(rejected) == 1
    evt = rejected[0]
    # Backward-compatible fields preserved.
    assert "last_close" in evt
    assert "vwap" in evt
    # New Phase 6.6 fields populated.
    assert evt["last_atr_value"] is not None and evt["last_atr_value"] > 0
    assert evt["atr_multiple"] == pytest.approx(5.0)
    assert evt["distance_from_vwap"] is not None and evt["distance_from_vwap"] > 0
    assert evt["threshold_distance"] == pytest.approx(evt["last_atr_value"] * evt["atr_multiple"])
    assert evt["extension_ratio"] is not None
    # By definition: a rejection means distance > threshold => ratio > 1.0.
    assert evt["extension_ratio"] > 1.0


def test_default_extended_from_vwap_atr_multiple_is_5() -> None:
    """Regression — the per-strategy default reads 5.0 (Day 3 calibration).

    Both the Pydantic config defaults and the strategy constructor
    defaults must agree on 5.0; a drift between the two would silently
    break runs whose YAML omits the field.
    """
    from bot.config import GapAndGoConfig, MomentumConfig
    from bot.strategies.momentum import MomentumStrategy

    assert GapAndGoConfig().extended_from_vwap_atr_multiple == pytest.approx(5.0)
    assert MomentumConfig().extended_from_vwap_atr_multiple == pytest.approx(5.0)
    assert GapAndGoStrategy().extended_from_vwap_atr_multiple == pytest.approx(5.0)
    assert MomentumStrategy().extended_from_vwap_atr_multiple == pytest.approx(5.0)


def test_gap_and_go_grace_period_unchanged_at_new_threshold() -> None:
    """Phase 5.5 grace bypass still suppresses extension check regardless of multiple.

    Ensures Phase 6.6's threshold tweak hasn't accidentally coupled the
    grace-window bypass to the multiple value: at any multiple, a bar
    inside the grace window should NEVER fire ``extended_from_vwap``.
    """
    bars = _extension_frame(last_minute=35)  # 5 min past open, inside default 15-min grace
    # Even a ridiculously tight 0.5× threshold must not produce a rejection
    # during the grace window — the bypass short-circuits before the check.
    strategy = GapAndGoStrategy(
        vwap_extension_grace_minutes=15, extended_from_vwap_atr_multiple=0.5
    )
    with capture_logs() as captured:
        signal = strategy.evaluate("TEST", bars)
    assert signal is not None, "grace must bypass extension check independent of multiple"
    assert not any(
        e.get("event") == "signal.rejected" and e.get("reason") == "extended_from_vwap"
        for e in captured
    )
    # Bypass log still emitted.
    assert any(e.get("event") == "gap_and_go.vwap_extension_bypassed" for e in captured)


# ---------- Phase 6.7: configurable window_end ---------- #


# ---------- Phase 7.1: market-hours filter + N-bar pullback low ---------- #


def test_stop_ignores_premarket_bars() -> None:
    """A 4 AM premarket wick at $5.00 must not pollute the intraday stop reference.

    Pre-7.1, ``session["low"].min()`` filtered by calendar date and pulled in
    backfilled premarket bars — a 1 AM wick permanently widened the stop for
    the rest of the session (TZOO Day 4, $7.87 ghost stop). The market-hours
    filter in 7.1 (>= 09:30 ET) removes that source of pollution.
    """
    bars = _frame(
        times=[
            "2026-04-16 04:00",  # premarket wick
            "2026-04-16 09:30",
            "2026-04-16 09:31",
            "2026-04-16 09:32",
            "2026-04-16 09:33",
            "2026-04-16 09:34",
            "2026-04-16 09:35",
            "2026-04-16 09:36",
            "2026-04-16 09:37",
            "2026-04-16 09:38",
        ],
        highs=[9.0, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8],
        lows=[5.0, 9.5, 9.6, 9.7, 9.75, 9.8, 9.85, 9.9, 9.92, 9.95],
        closes=[9.0, 9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.75],
        volumes=[100] + [1000] * 9,
    )
    strategy = GapAndGoStrategy()
    signal = strategy.evaluate("TEST", bars)
    assert signal is not None
    # Last 3 market-hours bars (09:36, 09:37, 09:38) have lows 9.9, 9.92, 9.95.
    assert signal.pullback_low == pytest.approx(9.9)
    # The $5.00 premarket wick must not leak through to the stop.
    assert signal.stop > 5.0
    assert signal.bars_available_for_lookback == 3


def test_pullback_low_uses_last_3_bars() -> None:
    """3-bar lookback ignores early-session wicks.

    Ten market-hours bars: a 9:30 wick to $7.50, then lows rising through $9.90.
    Breakout at 9:39 — the stop reference must be min($9.80, $9.85, $9.90) = $9.80,
    not the $7.50 that polluted the old session-min.
    """
    bars = _frame(
        times=[f"2026-04-16 09:{30 + i:02d}" for i in range(10)],
        highs=[10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 11.0],
        lows=[7.50, 9.00, 9.20, 9.40, 9.50, 9.60, 9.70, 9.80, 9.85, 9.90],
        closes=[9.90, 10.05, 10.15, 10.25, 10.35, 10.45, 10.55, 10.65, 10.75, 10.95],
    )
    # Widen grace to cover the 9:39 bar (minute 9) so the extension check does
    # not pre-empt — 7.1 is about the stop calc, not the extension gate.
    strategy = GapAndGoStrategy(vwap_extension_grace_minutes=30)
    signal = strategy.evaluate("TEST", bars)
    assert signal is not None, "happy-path frame should emit a signal"
    assert signal.pullback_low == pytest.approx(9.80)
    assert signal.bars_available_for_lookback == 3


def test_pullback_handles_fewer_than_3_bars_available() -> None:
    """At 9:31 with 2 market bars, pullback uses those 2 bars' min low — no crash."""
    # Premarket bars padded to clear _MIN_BARS. 9:30 is the first market-hours
    # bar so it can't itself be a HOD break (Phase 7.2 reset) — the last bar
    # must be 9:31 for the setup to fire with < 3 market-hours bars available.
    bars = _frame(
        times=[
            "2026-04-16 09:28",  # premarket
            "2026-04-16 09:29",  # premarket
            "2026-04-16 09:30",  # first market bar — no prior market HOD
            "2026-04-16 09:31",  # second market bar — breaks 9:30 high
        ],
        highs=[9.0, 9.5, 10.0, 10.5],
        lows=[8.0, 8.5, 9.6, 9.8],
        closes=[9.0, 9.4, 9.9, 10.4],
    )
    strategy = GapAndGoStrategy()
    signal = strategy.evaluate("TEST", bars)
    assert signal is not None, "2 market-hours bars must still compute a pullback_low"
    # recent_bars = last 3 market-hours bars clamped to available = [9:30, 9:31].
    assert signal.pullback_low == pytest.approx(9.6)
    assert signal.bars_available_for_lookback == 2


def test_stop_picks_min_of_vwap_and_pullback() -> None:
    """stop = min(vwap_at_entry, pullback_low) — invariant preserved by 7.1."""
    bars = _frame(
        times=[f"2026-04-16 09:{30 + i:02d}" for i in range(5)],
        highs=[10.0, 10.2, 10.4, 10.6, 10.8],
        lows=[9.8, 10.0, 10.1, 10.2, 10.3],
        closes=[10.0, 10.1, 10.3, 10.5, 10.7],
    )
    strategy = GapAndGoStrategy()
    signal = strategy.evaluate("TEST", bars)
    assert signal is not None
    assert signal.vwap_at_entry is not None
    assert signal.pullback_low is not None
    expected = round(min(signal.vwap_at_entry, signal.pullback_low), 4)
    assert signal.stop == pytest.approx(expected)


def test_signal_emitted_carries_observability_fields() -> None:
    """Strategy emits a ``signal.emitted`` log with pullback diagnostics for post-session review."""
    bars = _frame(
        times=[f"2026-04-16 09:{30 + i:02d}" for i in range(5)],
        highs=[10.0, 10.2, 10.4, 10.6, 10.8],
        lows=[9.8, 10.0, 10.1, 10.2, 10.3],
        closes=[10.0, 10.1, 10.3, 10.5, 10.7],
    )
    strategy = GapAndGoStrategy()
    with capture_logs() as captured:
        signal = strategy.evaluate("TEST", bars)
    assert signal is not None
    emitted = [e for e in captured if e.get("event") == "signal.emitted"]
    assert len(emitted) == 1
    evt = emitted[0]
    assert evt["symbol"] == "TEST"
    assert evt["strategy"] == "gap_and_go"
    assert evt["pullback_low"] == pytest.approx(signal.pullback_low)
    assert evt["pullback_lookback_bars"] == 3
    assert evt["bars_available_for_lookback"] == 3
    assert evt["vwap_at_entry"] == pytest.approx(signal.vwap_at_entry)


def test_gap_and_go_window_end_configurable() -> None:
    """Bars between the default 10:00 cutoff and a widened 16:00 evaluate only when extended."""
    from datetime import time as time_cls

    # Construct a bar at 10:30 ET that mirrors the happy-path frame:
    # rising highs + above VWAP + new HOD, so only the window check
    # would block the signal pre-6.7.
    bars = _frame(
        times=[
            "2026-04-16 10:27",
            "2026-04-16 10:28",
            "2026-04-16 10:29",
            "2026-04-16 10:30",
        ],
        highs=[10.0, 10.2, 10.4, 10.8],
        lows=[9.8, 10.0, 10.1, 10.3],
        closes=[10.0, 10.1, 10.3, 10.7],
        volumes=[1000, 1000, 1000, 1000],
    )

    # Default window_end=10:00 → 10:30 bar is outside the window → silent None.
    default_strategy = GapAndGoStrategy()
    assert default_strategy.evaluate("TEST", bars) is None

    # Widened window_end=16:00 → same 10:30 bar now evaluates. The 15-min
    # grace window has also elapsed (minutes_since_open = 60), so the
    # extension check runs; the frame is designed for the happy path so
    # we expect a signal.
    widened = GapAndGoStrategy(window_end=time_cls(16, 0))
    signal = widened.evaluate("TEST", bars)
    assert signal is not None, "widening window_end to 16:00 must evaluate the 10:30 bar"
    assert signal.strategy == "gap_and_go"


# ---------- Phase 8.4: premarket-high cap on scale_out ---------- #


def _frame_with_premarket_high(pmh: float) -> pd.DataFrame:
    """Helper — 4-bar happy-path frame with one premarket bar at the requested PMH."""
    return _frame(
        times=[
            "2026-04-16 04:00",  # premarket bar carrying the PMH
            "2026-04-16 09:30",
            "2026-04-16 09:31",
            "2026-04-16 09:32",
            "2026-04-16 09:33",
        ],
        highs=[pmh, 10.0, 10.2, 10.4, 10.8],
        lows=[pmh - 0.1, 9.8, 10.0, 10.1, 10.3],
        closes=[pmh - 0.05, 10.0, 10.1, 10.3, 10.7],
        volumes=[100, 1000, 1000, 1000, 1000],
    )


def test_pmh_cap_binds_when_pmh_between_entry_and_2r() -> None:
    """PMH at $11.00 sits between entry $10.70 and 2R target $11.50 → cap binds at $10.99."""
    bars = _frame_with_premarket_high(pmh=11.00)
    strategy = GapAndGoStrategy()
    signal = strategy.evaluate("TEST", bars)
    assert signal is not None
    # Entry = 10.70 (last close), R = entry − stop. With pullback_low logic the
    # stop falls out of last 3 bars' lows. PMH cap = 11.00 − $0.01 = $10.99,
    # which is below default 2R target → cap binds at $10.99.
    assert signal.scale_out_price == pytest.approx(10.99)


def test_pmh_cap_does_not_bind_when_entry_above_pmh() -> None:
    """Entry $10.70 > PMH $10.50 → gapper already broke PMH, cap doesn't bind."""
    bars = _frame_with_premarket_high(pmh=10.50)
    strategy = GapAndGoStrategy()
    signal = strategy.evaluate("TEST", bars)
    assert signal is not None
    # PMH below entry → no cap. Default 2R applies.
    risk = signal.entry - signal.stop
    assert signal.scale_out_price == pytest.approx(signal.entry + 2.0 * risk)


def test_pmh_cap_does_not_bind_when_pmh_above_2r() -> None:
    """PMH $20 way above entry $10.70 + 2R → 2R is binding, no cap."""
    bars = _frame_with_premarket_high(pmh=20.00)
    strategy = GapAndGoStrategy()
    signal = strategy.evaluate("TEST", bars)
    assert signal is not None
    risk = signal.entry - signal.stop
    assert signal.scale_out_price == pytest.approx(signal.entry + 2.0 * risk)


def test_pmh_cap_disabled_skips_cap_even_with_pmh_resistance() -> None:
    """``premarket_high_cap_enabled=False`` → strategy ignores PMH entirely."""
    bars = _frame_with_premarket_high(pmh=11.00)
    strategy = GapAndGoStrategy(premarket_high_cap_enabled=False)
    signal = strategy.evaluate("TEST", bars)
    assert signal is not None
    # Even with PMH at $11.00 (below default 2R), disabled cap → 2R target.
    risk = signal.entry - signal.stop
    assert signal.scale_out_price == pytest.approx(signal.entry + 2.0 * risk)


def test_pmh_cap_no_premarket_bars_falls_back_to_2r() -> None:
    """No premarket bars in frame → no PMH → cap can't bind, 2R stands."""
    # Same frame as the original happy path, no premarket bar prepended.
    bars = _frame(
        times=[
            "2026-04-16 09:30",
            "2026-04-16 09:31",
            "2026-04-16 09:32",
            "2026-04-16 09:33",
        ],
        highs=[10.0, 10.2, 10.4, 10.8],
        lows=[9.8, 10.0, 10.1, 10.3],
        closes=[10.0, 10.1, 10.3, 10.7],
        volumes=[1000, 1000, 1000, 1000],
    )
    strategy = GapAndGoStrategy()
    signal = strategy.evaluate("TEST", bars)
    assert signal is not None
    risk = signal.entry - signal.stop
    assert signal.scale_out_price == pytest.approx(signal.entry + 2.0 * risk)


def test_pmh_cap_logs_when_binding() -> None:
    """When the cap binds, ``strategy.scale_out_capped_premarket_high`` is logged."""
    bars = _frame_with_premarket_high(pmh=11.00)
    strategy = GapAndGoStrategy()
    with capture_logs() as captured:
        signal = strategy.evaluate("TEST", bars)
    assert signal is not None
    cap_events = [
        e for e in captured if e.get("event") == "strategy.scale_out_capped_premarket_high"
    ]
    assert len(cap_events) == 1
    evt = cap_events[0]
    assert evt["symbol"] == "TEST"
    assert evt["strategy"] == "gap_and_go"
    assert evt["premarket_high"] == pytest.approx(11.00)
    assert evt["capped_scale_out"] == pytest.approx(10.99)


# ---------- Phase 9.1: close-based HOD confirmation ---------- #


def test_rejects_wick_and_retrace_breakout() -> None:
    """Bar wicks above prior HOD then closes red — must be rejected, not entered.

    Same pattern as RMAX 2026-04-27 09:34: prior bars set HOD by close; the
    trigger bar wicks above prior HOD but closes back below it (still above
    VWAP — otherwise the VWAP filter would short-circuit before the HOD
    check). Phase 9.1's ``by="close"`` semantic correctly rejects this as
    a failed breakout.
    """
    bars = _frame(
        times=[
            "2026-04-27 09:30",
            "2026-04-27 09:31",
            "2026-04-27 09:32",
            "2026-04-27 09:33",
            "2026-04-27 09:34",
        ],
        # Prior HOD is bar 4's high $10.20. Bar 5 wicks to $10.30, closes
        # $10.05 — above VWAP (~9.88) but below prior HOD.
        highs=[9.55, 9.85, 10.10, 10.20, 10.30],
        lows=[9.45, 9.50, 9.85, 9.95, 9.95],
        closes=[9.50, 9.80, 10.00, 10.15, 10.05],
    )
    strategy = GapAndGoStrategy()
    with capture_logs() as captured:
        signal = strategy.evaluate("RMAX", bars)
    assert signal is None
    rejections = [
        e
        for e in captured
        if e.get("event") == "signal.rejected" and e.get("reason") == "not_new_hod"
    ]
    assert len(rejections) == 1
    evt = rejections[0]
    # Diagnostic fields show both the wick and the failed close.
    assert evt["last_high"] == pytest.approx(10.30)
    assert evt["last_close"] == pytest.approx(10.05)
    assert evt["session_hod"] == pytest.approx(10.20)


def test_accepts_close_confirmed_breakout() -> None:
    """Bar that closes above prior HOD must emit a signal under by='close'."""
    bars = _frame(
        times=[
            "2026-04-27 09:30",
            "2026-04-27 09:31",
            "2026-04-27 09:32",
            "2026-04-27 09:33",
        ],
        highs=[10.00, 10.10, 10.20, 10.50],
        lows=[9.80, 9.95, 10.05, 10.20],
        closes=[10.00, 10.08, 10.15, 10.45],  # final close $10.45 > prior high $10.20
    )
    strategy = GapAndGoStrategy()
    signal = strategy.evaluate("TEST", bars)
    assert signal is not None
    assert signal.entry == pytest.approx(10.45)

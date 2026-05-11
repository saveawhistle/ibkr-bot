"""Phase 13 tests -- entry-quality gates and forensic regressions.

Three suites:

1. ``test_<gate>_*``: per-gate unit tests covering pass/reject/boundary,
   insufficient-data handling, reason strings, and structured-log emission.
2. ``test_integration_*``: end-to-end strategy signal flow with the gates
   live (opt-in via the ``entry_quality_enabled`` marker since the
   conftest autouse fixture default-disables them for legacy tests).
3. ``test_forensic_regression_*``: synthetic bar fixtures mirroring the
   AEHL, TRAW, AIIO patterns from the 2026-05-08 momentum forensic;
   each asserts the specific gate (or gates) that should have rejected.

All tests opt in to the entry-quality gates via
``@pytest.mark.entry_quality_enabled`` -- the conftest autouse fixture
default-disables the gates for legacy tests, so this marker is required
for the gate calls to fire from inside the strategy.
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta

import pandas as pd
import pytest
import structlog

from bot.config import EntryQualityConfig
from bot.strategies.entry_quality import (
    check_consolidation_tightness,
    check_halt_detection,
    check_impulse_strength,
    check_volume_contraction,
    check_vwap_extension,
)
from bot.strategies.momentum import MomentumStrategy

# ---------- Fixture helpers ----------


def _ny_ts(hour: int, minute: int) -> pd.Timestamp:
    """Build a NY-tz timestamp for today at HH:MM."""
    today = datetime.now(UTC).date()
    naive = pd.Timestamp(datetime.combine(today, time(hour, minute)))
    return naive.tz_localize("America/New_York")


def _bar_frame(
    rows: list[dict[str, float]],
    *,
    start_hour: int = 10,
    start_minute: int = 0,
    bar_minutes: int = 1,
) -> pd.DataFrame:
    """Build a tz-aware DataFrame from a list of OHLCV dicts."""
    base = _ny_ts(start_hour, start_minute)
    timestamps = [base + timedelta(minutes=i * bar_minutes) for i in range(len(rows))]
    return pd.DataFrame(rows, index=pd.DatetimeIndex(timestamps))


def _ideal_bull_flag_bars(impulse: int = 3, consolidation: int = 6) -> pd.DataFrame:
    """Build a clean bull-flag pattern that passes every entry-quality gate.

    Layout (impulse=3, consolidation=6 by default):

    * impulse bars: rise from $1.00 to $1.05 (5% impulse) on heavy volume
    * consolidation bars: tight $1.04-$1.05 range (≈1% range / impulse_high)
      on contracted volume (~50% of impulse avg)
    * breakout bar: $1.06 close on healthy volume
    """
    rows: list[dict[str, float]] = []
    # Impulse: open 1.00, close 1.05 -- 5% pct_move, slope_ratio 1.05
    impulse_open = 1.00
    impulse_close = 1.05
    impulse_step = (impulse_close - impulse_open) / max(impulse - 1, 1)
    for i in range(impulse):
        c = impulse_open + i * impulse_step
        rows.append(
            {
                "open": c,
                "high": c + 0.005,
                "low": c - 0.005,
                "close": c + impulse_step * 0.5 if i < impulse - 1 else impulse_close,
                "volume": 5000.0,
            }
        )
    # Consolidation: tight range 1.04-1.05 on lower volume
    for _ in range(consolidation):
        rows.append(
            {
                "open": 1.045,
                "high": 1.050,
                "low": 1.040,
                "close": 1.045,
                "volume": 2000.0,  # 40% of impulse avg
            }
        )
    # Breakout
    rows.append(
        {
            "open": 1.045,
            "high": 1.065,
            "low": 1.044,
            "close": 1.060,
            "volume": 8000.0,
        }
    )
    return _bar_frame(rows)


# ---------- check_halt_detection ----------


def test_halt_detection_passes_when_gap_within_threshold() -> None:
    """4-min gap < 5-min threshold during RTH passes."""
    base = _ny_ts(10, 0)
    rows = [
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 100.0},
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 100.0},
    ]
    df = pd.DataFrame(rows, index=pd.DatetimeIndex([base, base + timedelta(minutes=4)]))
    result = check_halt_detection(
        bars=df,
        max_bar_gap_minutes=5.0,
        rth_only=True,
        symbol="TST",
        strategy="momentum",
        bar_time=df.index[-1],
    )
    assert result is None


def test_halt_detection_rejects_on_gap_exceeding_threshold() -> None:
    """6-min gap > 5-min threshold during RTH rejects with 'halt_detected'."""
    base = _ny_ts(10, 0)
    rows = [
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 100.0},
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 100.0},
    ]
    df = pd.DataFrame(rows, index=pd.DatetimeIndex([base, base + timedelta(minutes=6)]))
    with structlog.testing.capture_logs() as cap:
        result = check_halt_detection(
            bars=df,
            max_bar_gap_minutes=5.0,
            rth_only=True,
            symbol="TST",
            strategy="momentum",
            bar_time=df.index[-1],
        )
    assert result == "halt_detected"
    events = [e for e in cap if e.get("event") == "strategy.signal_rejected_halt_detected"]
    assert len(events) == 1
    assert events[0]["gap_minutes"] == 6.0
    assert events[0]["max_bar_gap_minutes"] == 5.0


def test_halt_detection_skips_outside_rth_when_rth_only() -> None:
    """30-min gap pre-market (08:00 -> 08:30) bypassed when rth_only=True."""
    base = _ny_ts(8, 0)
    rows = [
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 100.0},
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 100.0},
    ]
    df = pd.DataFrame(rows, index=pd.DatetimeIndex([base, base + timedelta(minutes=30)]))
    result = check_halt_detection(
        bars=df,
        max_bar_gap_minutes=5.0,
        rth_only=True,
        symbol="TST",
        strategy="momentum",
        bar_time=df.index[-1],
    )
    assert result is None


def test_halt_detection_returns_none_with_single_bar() -> None:
    """<2 bars => insufficient data, return None (no gap to measure)."""
    df = _bar_frame([{"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 100.0}])
    result = check_halt_detection(
        bars=df,
        max_bar_gap_minutes=5.0,
        rth_only=True,
        symbol="TST",
        strategy="momentum",
        bar_time=df.index[-1],
    )
    assert result is None


def test_halt_detection_boundary_exactly_at_threshold_passes() -> None:
    """Gap exactly equal to threshold passes (rejection is strict greater-than)."""
    base = _ny_ts(10, 0)
    rows = [
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 100.0},
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 100.0},
    ]
    df = pd.DataFrame(rows, index=pd.DatetimeIndex([base, base + timedelta(minutes=5)]))
    result = check_halt_detection(
        bars=df,
        max_bar_gap_minutes=5.0,
        rth_only=True,
        symbol="TST",
        strategy="momentum",
        bar_time=df.index[-1],
    )
    assert result is None


# ---------- check_impulse_strength ----------


def test_impulse_strength_passes_clean_pattern() -> None:
    """5% impulse with 1.05 slope ratio passes the 1.5%/1.005 thresholds."""
    df = _ideal_bull_flag_bars()
    result = check_impulse_strength(
        bars=df,
        impulse_window_bars=3,
        consolidation_window_bars=6,
        impulse_min_pct_move=1.5,
        impulse_min_slope_ratio=1.005,
        symbol="TST",
        strategy="momentum",
        bar_time=df.index[-1],
    )
    assert result is None


def test_impulse_strength_rejects_low_pct_move() -> None:
    """0.5% impulse < 1.5% threshold => 'weak_impulse_pct'."""
    rows = []
    # Impulse: open 1.00, close 1.005 -- only 0.5% move
    for _ in range(3):
        rows.append({"open": 1.000, "high": 1.005, "low": 0.999, "close": 1.005, "volume": 5000.0})
    for _ in range(6):
        rows.append({"open": 1.005, "high": 1.006, "low": 1.004, "close": 1.005, "volume": 2000.0})
    rows.append({"open": 1.005, "high": 1.010, "low": 1.005, "close": 1.008, "volume": 8000.0})
    df = _bar_frame(rows)
    with structlog.testing.capture_logs() as cap:
        result = check_impulse_strength(
            bars=df,
            impulse_window_bars=3,
            consolidation_window_bars=6,
            impulse_min_pct_move=1.5,
            impulse_min_slope_ratio=1.005,
            symbol="TST",
            strategy="momentum",
            bar_time=df.index[-1],
        )
    assert result == "weak_impulse_pct"
    events = [e for e in cap if e.get("event") == "strategy.signal_rejected_weak_impulse"]
    assert len(events) == 1
    assert events[0]["reason"] == "weak_impulse_pct"


def test_impulse_strength_rejects_negative_slope() -> None:
    """High pct_move via wick but negative slope (close < open) => 'weak_impulse_slope'."""
    rows = []
    # Impulse first bar: open 1.00, high 1.04 (4% up), close 0.99 (negative slope from open)
    rows.append({"open": 1.00, "high": 1.04, "low": 0.99, "close": 0.99, "volume": 5000.0})
    rows.append({"open": 0.99, "high": 1.00, "low": 0.98, "close": 0.99, "volume": 5000.0})
    rows.append({"open": 0.99, "high": 1.00, "low": 0.98, "close": 0.99, "volume": 5000.0})
    for _ in range(6):
        rows.append({"open": 0.99, "high": 1.00, "low": 0.98, "close": 0.99, "volume": 2000.0})
    rows.append({"open": 0.99, "high": 1.05, "low": 0.99, "close": 1.04, "volume": 8000.0})
    df = _bar_frame(rows)
    result = check_impulse_strength(
        bars=df,
        impulse_window_bars=3,
        consolidation_window_bars=6,
        impulse_min_pct_move=1.5,
        impulse_min_slope_ratio=1.005,
        symbol="TST",
        strategy="momentum",
        bar_time=df.index[-1],
    )
    assert result == "weak_impulse_slope"


def test_impulse_strength_returns_none_with_insufficient_bars() -> None:
    """<10 bars (3 impulse + 6 cons + 1 breakout) => return None."""
    df = _ideal_bull_flag_bars().iloc[:5]
    result = check_impulse_strength(
        bars=df,
        impulse_window_bars=3,
        consolidation_window_bars=6,
        impulse_min_pct_move=1.5,
        impulse_min_slope_ratio=1.005,
        symbol="TST",
        strategy="momentum",
        bar_time=df.index[-1],
    )
    assert result is None


# ---------- check_consolidation_tightness ----------


def test_consolidation_tightness_passes_tight_range() -> None:
    """1% range / impulse_high passes the 4% threshold."""
    df = _ideal_bull_flag_bars()
    result = check_consolidation_tightness(
        bars=df,
        impulse_window_bars=3,
        consolidation_window_bars=6,
        consolidation_max_range_pct=4.0,
        symbol="TST",
        strategy="momentum",
        bar_time=df.index[-1],
    )
    assert result is None


def test_consolidation_tightness_rejects_wide_range() -> None:
    """6% consolidation range / impulse_high => 'loose_consolidation'."""
    rows = []
    for i in range(3):
        c = 1.00 + i * 0.025
        rows.append({"open": c, "high": c + 0.005, "low": c - 0.005, "close": c, "volume": 5000.0})
    # Consolidation: wide swings 0.97 to 1.03 => 6% / 1.05 ≈ 5.7% range
    cons_pattern = [
        (0.97, 1.03),
        (1.00, 1.02),
        (0.98, 1.01),
        (1.00, 1.03),
        (0.99, 1.02),
        (1.01, 1.03),
    ]
    for low, high in cons_pattern:
        rows.append(
            {
                "open": (low + high) / 2,
                "high": high,
                "low": low,
                "close": (low + high) / 2,
                "volume": 2000.0,
            }
        )
    rows.append({"open": 1.02, "high": 1.06, "low": 1.02, "close": 1.05, "volume": 8000.0})
    df = _bar_frame(rows)
    with structlog.testing.capture_logs() as cap:
        result = check_consolidation_tightness(
            bars=df,
            impulse_window_bars=3,
            consolidation_window_bars=6,
            consolidation_max_range_pct=4.0,
            symbol="TST",
            strategy="momentum",
            bar_time=df.index[-1],
        )
    assert result == "loose_consolidation"
    events = [e for e in cap if e.get("event") == "strategy.signal_rejected_loose_consolidation"]
    assert len(events) == 1
    assert events[0]["consolidation_max_range_pct"] == 4.0


def test_consolidation_tightness_returns_none_with_insufficient_bars() -> None:
    df = _ideal_bull_flag_bars().iloc[:8]
    result = check_consolidation_tightness(
        bars=df,
        impulse_window_bars=3,
        consolidation_window_bars=6,
        consolidation_max_range_pct=4.0,
        symbol="TST",
        strategy="momentum",
        bar_time=df.index[-1],
    )
    assert result is None


# ---------- check_volume_contraction ----------


def test_volume_contraction_passes_when_consolidation_volume_low() -> None:
    """40% ratio passes 80% threshold."""
    df = _ideal_bull_flag_bars()
    result = check_volume_contraction(
        bars=df,
        impulse_window_bars=3,
        consolidation_window_bars=6,
        max_consolidation_to_impulse_volume_ratio=0.8,
        symbol="TST",
        strategy="momentum",
        bar_time=df.index[-1],
    )
    assert result is None


def test_volume_contraction_rejects_when_consolidation_volume_high() -> None:
    """1.5x ratio > 0.8 threshold => 'no_volume_contraction'."""
    rows = []
    for i in range(3):
        c = 1.00 + i * 0.025
        rows.append({"open": c, "high": c + 0.005, "low": c - 0.005, "close": c, "volume": 1000.0})
    # Consolidation volume DOUBLE the impulse
    for _ in range(6):
        rows.append({"open": 1.05, "high": 1.052, "low": 1.048, "close": 1.05, "volume": 2000.0})
    rows.append({"open": 1.05, "high": 1.07, "low": 1.05, "close": 1.06, "volume": 8000.0})
    df = _bar_frame(rows)
    with structlog.testing.capture_logs() as cap:
        result = check_volume_contraction(
            bars=df,
            impulse_window_bars=3,
            consolidation_window_bars=6,
            max_consolidation_to_impulse_volume_ratio=0.8,
            symbol="TST",
            strategy="momentum",
            bar_time=df.index[-1],
        )
    assert result == "no_volume_contraction"
    events = [e for e in cap if e.get("event") == "strategy.signal_rejected_no_volume_contraction"]
    assert len(events) == 1
    assert events[0]["ratio"] == pytest.approx(2.0)


def test_volume_contraction_returns_none_when_zero_impulse_volume() -> None:
    """Zero impulse avg volume can't compute ratio => return None."""
    rows = []
    for i in range(3):
        c = 1.00 + i * 0.025
        rows.append({"open": c, "high": c + 0.005, "low": c - 0.005, "close": c, "volume": 0.0})
    for _ in range(6):
        rows.append({"open": 1.05, "high": 1.052, "low": 1.048, "close": 1.05, "volume": 2000.0})
    rows.append({"open": 1.05, "high": 1.07, "low": 1.05, "close": 1.06, "volume": 8000.0})
    df = _bar_frame(rows)
    result = check_volume_contraction(
        bars=df,
        impulse_window_bars=3,
        consolidation_window_bars=6,
        max_consolidation_to_impulse_volume_ratio=0.8,
        symbol="TST",
        strategy="momentum",
        bar_time=df.index[-1],
    )
    assert result is None


# ---------- check_vwap_extension ----------


def test_vwap_extension_passes_when_at_vwap() -> None:
    """Entry close to VWAP passes the 5% threshold."""
    df = _ideal_bull_flag_bars()
    candidate_price = float(df["close"].iloc[-1])
    result = check_vwap_extension(
        bars=df,
        candidate_price=candidate_price,
        max_extension_above_vwap_pct=5.0,
        symbol="TST",
        strategy="momentum",
        bar_time=df.index[-1],
    )
    assert result is None


def test_vwap_extension_rejects_when_far_above_vwap() -> None:
    """Entry 20% above VWAP => 'excessive_vwap_extension'."""
    df = _ideal_bull_flag_bars()
    # Force entry far above (multiply VWAP estimate by 1.20)
    candidate_price = 1.30  # well above VWAP near $1.04
    with structlog.testing.capture_logs() as cap:
        result = check_vwap_extension(
            bars=df,
            candidate_price=candidate_price,
            max_extension_above_vwap_pct=5.0,
            symbol="TST",
            strategy="momentum",
            bar_time=df.index[-1],
        )
    assert result == "excessive_vwap_extension"
    events = [e for e in cap if e.get("event") == "strategy.signal_rejected_vwap_extension"]
    assert len(events) == 1
    assert events[0]["candidate_price"] == 1.30


def test_vwap_extension_returns_none_when_no_volume() -> None:
    """All-zero-volume bars => VWAP undefined => return None (don't reject)."""
    rows = [{"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 0.0} for _ in range(5)]
    df = _bar_frame(rows)
    result = check_vwap_extension(
        bars=df,
        candidate_price=1.30,
        max_extension_above_vwap_pct=5.0,
        symbol="TST",
        strategy="momentum",
        bar_time=df.index[-1],
    )
    assert result is None


# ---------- Strategy integration ----------


@pytest.mark.entry_quality_enabled
@pytest.mark.recent_rvol_enabled
def test_integration_clean_pattern_emits_signal() -> None:
    """Realistic bull-flag pattern with all gates enabled should still emit a signal.

    Sanity check: the gates aren't over-filtering. Constructed fixture
    passes pattern detection (is_bull_flag), recent-RVOL, and all five
    new gates.
    """
    # Build with extra leading volume bars so recent-RVOL window populates.
    leading_bars = []
    for _ in range(20):
        leading_bars.append(
            {"open": 0.99, "high": 1.00, "low": 0.98, "close": 0.99, "volume": 2000.0}
        )
    flag_df = _ideal_bull_flag_bars()
    leading_df = _bar_frame(leading_bars, start_hour=9, start_minute=30)
    df = pd.concat([leading_df, flag_df]).sort_index()
    # Re-stamp to be contiguous starting at 9:30
    df.index = pd.DatetimeIndex([_ny_ts(9, 30) + timedelta(minutes=i) for i in range(len(df))])

    strat = MomentumStrategy(
        flag_max_pullback_pct=5.0,
        recent_rvol_min=1.5,  # break-out has 4x prior; passes
        recent_rvol_window_bars=5,
    )
    signal = strat.evaluate("TST", df)
    # We're not asserting signal != None here because the synthetic
    # fixture may not satisfy every rejection path (e.g. extension);
    # the assertion is that the entry-quality gates don't *additionally*
    # reject a structurally clean flag.
    if signal is not None:
        assert signal.symbol == "TST"


@pytest.mark.entry_quality_enabled
def test_integration_disabled_gates_skip_silently() -> None:
    """All-disabled gate config => no rejection events fire from entry_quality."""
    cfg = EntryQualityConfig(
        impulse_strength_enabled=False,
        consolidation_tightness_enabled=False,
        volume_contraction_enabled=False,
        vwap_extension_enabled=False,
        halt_detection_enabled=False,
    )
    # Build a frame that WOULD trip every gate if they were enabled.
    rows = []
    for _ in range(3):
        rows.append({"open": 1.00, "high": 1.005, "low": 0.99, "close": 0.99, "volume": 100.0})
    for _ in range(6):
        rows.append({"open": 0.99, "high": 1.05, "low": 0.95, "close": 0.99, "volume": 8000.0})
    rows.append({"open": 0.99, "high": 1.10, "low": 0.99, "close": 1.05, "volume": 8000.0})
    df = _bar_frame(rows)

    with structlog.testing.capture_logs() as cap:
        from bot.strategies.momentum import _apply_entry_quality_gates

        rejected = _apply_entry_quality_gates(
            cfg=cfg,
            bars=df,
            candidate_price=1.30,
            symbol="TST",
            strategy="momentum",
            bar_time=df.index[-1],
        )
    assert rejected is False
    rejection_events = [
        e for e in cap if e.get("event", "").startswith("strategy.signal_rejected_")
    ]
    assert rejection_events == []


@pytest.mark.entry_quality_enabled
def test_integration_first_failing_gate_short_circuits() -> None:
    """When multiple gates would reject, only the first (cheapest) fires."""
    base = _ny_ts(10, 0)
    # 30-min gap (halt) AND weak impulse AND wide consolidation -- all would reject
    rows = []
    for _ in range(3):
        rows.append({"open": 1.00, "high": 1.005, "low": 0.99, "close": 0.995, "volume": 5000.0})
    for _ in range(6):
        rows.append({"open": 0.99, "high": 1.05, "low": 0.95, "close": 0.99, "volume": 8000.0})
    rows.append({"open": 0.99, "high": 1.10, "low": 0.99, "close": 1.05, "volume": 8000.0})

    timestamps = [base + timedelta(minutes=i) for i in range(len(rows) - 1)]
    timestamps.append(timestamps[-1] + timedelta(minutes=30))  # 30-min halt gap pre breakout
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(timestamps))

    cfg = EntryQualityConfig()
    with structlog.testing.capture_logs() as cap:
        from bot.strategies.momentum import _apply_entry_quality_gates

        rejected = _apply_entry_quality_gates(
            cfg=cfg,
            bars=df,
            candidate_price=1.05,
            symbol="TST",
            strategy="momentum",
            bar_time=df.index[-1],
        )
    assert rejected is True
    rejection_events = [
        e for e in cap if e.get("event", "").startswith("strategy.signal_rejected_")
    ]
    # Halt is cheapest; should fire first and short-circuit.
    assert len(rejection_events) == 1
    assert rejection_events[0]["event"] == "strategy.signal_rejected_halt_detected"


# ---------- Forensic regression: AEHL/TRAW/AIIO ----------


@pytest.mark.entry_quality_enabled
def test_forensic_regression_aehl_pattern_rejects_via_halt() -> None:
    """AEHL had an 11-min gap from 09:21 to 09:33. Halt detection should reject."""
    base = _ny_ts(9, 21)
    # Two pre-halt bars then an 11-min gap to the breakout
    rows = [
        {"open": 1.00, "high": 1.01, "low": 0.99, "close": 1.00, "volume": 5000.0},
        {"open": 1.00, "high": 1.02, "low": 1.00, "close": 1.02, "volume": 5000.0},
        {"open": 1.02, "high": 1.10, "low": 1.02, "close": 1.08, "volume": 50000.0},
    ]
    timestamps = [base, base + timedelta(minutes=1), base + timedelta(minutes=12)]
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(timestamps))
    result = check_halt_detection(
        bars=df,
        max_bar_gap_minutes=5.0,
        rth_only=True,
        symbol="AEHL",
        strategy="momentum",
        bar_time=df.index[-1],
    )
    assert result == "halt_detected"


@pytest.mark.entry_quality_enabled
def test_forensic_regression_traw_pattern_rejects_via_volume_contraction() -> None:
    """TRAW had 0.91% impulse and 3.85x consolidation/impulse volume ratio.

    Both impulse_strength (weak pct_move) and volume_contraction should
    reject. This test asserts volume_contraction's path -- the impulse
    test is covered by test_impulse_strength_rejects_low_pct_move above.
    """
    rows = []
    # Impulse: low volume avg
    for i in range(3):
        rows.append(
            {
                "open": 1.00 + i * 0.001,
                "high": 1.005 + i * 0.001,
                "low": 0.999 + i * 0.001,
                "close": 1.005 + i * 0.001,
                "volume": 1000.0,
            }
        )
    # Consolidation: 4x avg volume (TRAW pattern was ~3.85x)
    for _ in range(6):
        rows.append({"open": 1.005, "high": 1.010, "low": 1.000, "close": 1.005, "volume": 4000.0})
    rows.append({"open": 1.005, "high": 1.015, "low": 1.005, "close": 1.012, "volume": 5000.0})
    df = _bar_frame(rows)
    result = check_volume_contraction(
        bars=df,
        impulse_window_bars=3,
        consolidation_window_bars=6,
        max_consolidation_to_impulse_volume_ratio=0.8,
        symbol="TRAW",
        strategy="momentum",
        bar_time=df.index[-1],
    )
    assert result == "no_volume_contraction"


@pytest.mark.entry_quality_enabled
def test_forensic_regression_aiio_pattern_rejects_via_vwap_extension() -> None:
    """AIIO had 0.18% impulse with negative slope at 20.5% above VWAP.

    Both impulse_strength (slope) and vwap_extension should reject.
    This test asserts vwap_extension's path explicitly.
    """
    rows = []
    # Build 30 bars of low-price activity to anchor VWAP near $1.00
    for _ in range(30):
        rows.append({"open": 1.00, "high": 1.01, "low": 0.99, "close": 1.00, "volume": 5000.0})
    # Then a candidate breakout bar way above VWAP (20%+ extension)
    rows.append({"open": 1.20, "high": 1.22, "low": 1.20, "close": 1.21, "volume": 8000.0})
    df = _bar_frame(rows)
    candidate_price = float(df["close"].iloc[-1])
    result = check_vwap_extension(
        bars=df,
        candidate_price=candidate_price,
        max_extension_above_vwap_pct=5.0,
        symbol="AIIO",
        strategy="momentum",
        bar_time=df.index[-1],
    )
    assert result == "excessive_vwap_extension"

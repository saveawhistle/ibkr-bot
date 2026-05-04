"""Tests for the Phase 10.2 minimum stop-distance floor.

Two layers:

* Unit tests against the shared ``_apply_stop_distance_floor`` helper —
  every floor-bind branch (min_abs / min_pct / boundary / no-bind) plus
  the ZENA verbatim repro and the structured event payload.
* Integration tests through ``MomentumStrategy.evaluate`` and
  ``GapAndGoStrategy.evaluate`` confirming the floored value flows into
  the emitted ``Signal.stop`` end-to-end (so the risk module, executor
  bracket placement, and Phase 8.3 fill-anchored re-protection all see
  one consistent number).
"""

from __future__ import annotations

import pandas as pd
import pytest
from structlog.testing import capture_logs

from bot.strategies.base import _apply_stop_distance_floor  # noqa: PLC2701
from bot.strategies.gap_and_go import GapAndGoStrategy
from bot.strategies.momentum import MomentumStrategy

# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


def _floor(
    *,
    entry: float,
    structural: float,
    min_abs: float = 0.05,
    min_pct: float = 0.02,
    symbol: str = "TEST",
    strategy: str = "test",
    bar_time: pd.Timestamp | None = None,
) -> float:
    """Convenience wrapper — fixed bar_time so the event payload is deterministic."""
    ts = bar_time or pd.Timestamp("2026-04-30 09:31", tz="America/New_York")
    return _apply_stop_distance_floor(
        entry=entry,
        structural_stop=structural,
        floor_min_abs=min_abs,
        floor_min_pct=min_pct,
        symbol=symbol,
        strategy=strategy,
        bar_time=ts,
    )


def test_floor_does_not_bind_when_structural_already_wider() -> None:
    """Structural stop already further than the floor distance → returned unchanged, no event."""
    with capture_logs() as captured:
        out = _floor(entry=10.00, structural=9.50)
    assert out == pytest.approx(9.50)
    events = [e for e in captured if e.get("event") == "entry.stop_distance_floor_applied"]
    assert events == []


def test_min_abs_binds_on_low_priced_stock() -> None:
    """Sub-$2.50 stock where 5¢ > 2% of price — min_abs wins.

    At $1.00 entry: pct floor = 2¢, abs floor = 5¢ → floor distance = 5¢.
    Structural risk of 1¢ floors to 5¢, placed stop = $0.95.
    """
    with capture_logs() as captured:
        out = _floor(entry=1.00, structural=0.99)
    assert out == pytest.approx(0.95)
    event = next(e for e in captured if e.get("event") == "entry.stop_distance_floor_applied")
    assert event["which_floor_won"] == "min_abs"
    assert event["floor_distance"] == pytest.approx(0.05)
    assert event["floored_stop"] == pytest.approx(0.95)


def test_min_pct_binds_on_higher_priced_stock() -> None:
    """Above-$2.50 stock where 2% > 5¢ — min_pct wins.

    At $10.00 entry: abs floor = 5¢, pct floor = 20¢ → floor distance = 20¢.
    Structural risk of 10¢ floors to 20¢, placed stop = $9.80.
    """
    with capture_logs() as captured:
        out = _floor(entry=10.00, structural=9.90)
    assert out == pytest.approx(9.80)
    event = next(e for e in captured if e.get("event") == "entry.stop_distance_floor_applied")
    assert event["which_floor_won"] == "min_pct"
    assert event["floor_distance"] == pytest.approx(0.20)
    assert event["floored_stop"] == pytest.approx(9.80)


def test_floor_at_boundary_entry_2_50() -> None:
    """At $2.50 entry, 2% × $2.50 == 5¢ exactly — either branch is acceptable.

    Tests assert the floored value is correct ($2.45). The
    ``which_floor_won`` field is implementation-defined at the boundary
    (current impl reports ``min_abs`` because the tie-break uses ``>=``);
    we assert the value, not the branch label.
    """
    with capture_logs() as captured:
        out = _floor(entry=2.50, structural=2.49)
    assert out == pytest.approx(2.45)
    event = next(e for e in captured if e.get("event") == "entry.stop_distance_floor_applied")
    assert event["floored_stop"] == pytest.approx(2.45)
    # Either label is acceptable at the boundary; both branches give the same value.
    assert event["which_floor_won"] in {"min_abs", "min_pct"}


def test_zena_2026_04_30_reproduction() -> None:
    """ZENA verbatim: entry $2.18, structural $2.17 → floored stop $2.13.

    floor_distance = max(0.05, 2.18 × 0.02 = 0.0436) = 0.05 → min_abs wins.
    floored_stop = 2.18 − 0.05 = 2.13.
    """
    with capture_logs() as captured:
        out = _floor(entry=2.18, structural=2.17, symbol="ZENA", strategy="momentum")
    assert out == pytest.approx(2.13)
    event = next(e for e in captured if e.get("event") == "entry.stop_distance_floor_applied")
    assert event["symbol"] == "ZENA"
    assert event["strategy"] == "momentum"
    assert event["entry_price"] == pytest.approx(2.18)
    assert event["structural_stop"] == pytest.approx(2.17)
    assert event["floor_distance"] == pytest.approx(0.05)
    assert event["floored_stop"] == pytest.approx(2.13)
    assert event["which_floor_won"] == "min_abs"


def test_event_payload_keys() -> None:
    """The structured event must carry every key the operator needs for triage."""
    with capture_logs() as captured:
        _floor(entry=2.18, structural=2.17, symbol="ZENA", strategy="momentum")
    event = next(e for e in captured if e.get("event") == "entry.stop_distance_floor_applied")
    expected_keys = {
        "event",
        "symbol",
        "strategy",
        "bar_time",
        "entry_price",
        "structural_stop",
        "floor_distance",
        "floored_stop",
        "which_floor_won",
        "log_level",
    }
    # log_level is added by structlog; allow extras but require the expected set.
    assert expected_keys <= set(event.keys())


def test_no_event_when_floor_does_not_bind() -> None:
    """Quiet path — structural already wider than floor → no event at all."""
    with capture_logs() as captured:
        _floor(entry=10.00, structural=9.50)
    assert not any(
        e.get("event") == "entry.stop_distance_floor_applied" for e in captured
    )


def test_zero_min_abs_disables_abs_branch() -> None:
    """``min_abs=0.0`` is a legal config — pct branch becomes the sole floor.

    At $1.00 entry with min_abs=0 + min_pct=0.02: floor distance = 2¢.
    Structural risk of 1¢ → floored to 2¢, placed stop = $0.98.
    """
    with capture_logs() as captured:
        out = _floor(entry=1.00, structural=0.99, min_abs=0.0, min_pct=0.02)
    assert out == pytest.approx(0.98)
    event = next(e for e in captured if e.get("event") == "entry.stop_distance_floor_applied")
    assert event["which_floor_won"] == "min_pct"


def test_both_floors_zero_returns_structural_unchanged() -> None:
    """``min_abs=0`` + ``min_pct=0`` disables the floor entirely (escape hatch)."""
    with capture_logs() as captured:
        out = _floor(entry=2.18, structural=2.17, min_abs=0.0, min_pct=0.0)
    assert out == pytest.approx(2.17)
    assert not any(
        e.get("event") == "entry.stop_distance_floor_applied" for e in captured
    )


# ---------------------------------------------------------------------------
# Strategy integration tests — confirm the floored value flows into Signal.stop
# ---------------------------------------------------------------------------


def _momentum_frame_with_tight_stop() -> pd.DataFrame:
    """Construct a momentum-passing bar frame whose flag_low is 1¢ below the breakout close.

    Layout matches the existing momentum suite's setup:
      * 10 bars in the 09:30–09:39 window
      * Tight impulse, very shallow flag (lows hover within 1¢ of close)
      * Final bar breaks HOD by 1¢; flag_low across the 10-bar window is exactly 1¢
        below the breakout close, mirroring today's ZENA case.
    """
    times = [f"2026-04-30 09:{30 + i:02d}" for i in range(10)]
    # Closes 2.16 → 2.18 with a fresh HOD on the last bar.
    closes = [2.16, 2.165, 2.17, 2.17, 2.165, 2.165, 2.17, 2.17, 2.17, 2.18]
    # Lows tight: the global min across the 10-bar window is exactly 2.17.
    lows = [2.17, 2.17, 2.17, 2.17, 2.17, 2.17, 2.17, 2.17, 2.17, 2.175]
    # Highs: must not exceed the final bar's high so HOD breaks on the last close.
    highs = [2.165, 2.17, 2.17, 2.17, 2.17, 2.17, 2.17, 2.17, 2.175, 2.185]
    idx = pd.to_datetime(times).tz_localize("America/New_York")
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [10_000.0] * len(times),
            "vwap": closes,
        },
        index=idx,
    )


def test_momentum_zena_scenario_emits_floored_stop() -> None:
    """End-to-end: a momentum signal with structural risk = 1¢ emits ``Signal.stop`` at 5¢ floor.

    Confirms the floor lands inside ``MomentumStrategy.evaluate`` (the
    structural value is preserved in ``signal.emitted``'s ``pullback_low``
    field, the placed ``Signal.stop`` reflects the floored value).
    """
    bars = _momentum_frame_with_tight_stop()
    strategy = MomentumStrategy(
        flag_max_pullback_pct=5.0,
        # Loosen the extension check so the synthetic frame's tiny ATR
        # doesn't reject; the floor we're testing is independent of it.
        extended_from_vwap_atr_multiple=20.0,
    )
    with capture_logs() as captured:
        signal = strategy.evaluate("ZENA", bars)
    assert signal is not None, "expected a momentum signal on the synthetic frame"
    assert signal.entry == pytest.approx(2.18)
    # Floored stop = entry - max(0.05, 0.02 × 2.18) = 2.18 - 0.05 = 2.13
    assert signal.stop == pytest.approx(2.13)
    # The structural value is preserved in the emitted log for forensics.
    emitted = next(e for e in captured if e.get("event") == "signal.emitted")
    assert emitted["pullback_low"] == pytest.approx(2.17)
    assert emitted["stop"] == pytest.approx(2.13)
    # The floor-applied event fires with the correct branch tag.
    floor_evt = next(
        e for e in captured if e.get("event") == "entry.stop_distance_floor_applied"
    )
    assert floor_evt["symbol"] == "ZENA"
    assert floor_evt["strategy"] == "momentum"
    assert floor_evt["which_floor_won"] == "min_abs"


def test_momentum_no_floor_event_when_structural_already_wide_enough() -> None:
    """A momentum signal with comfortably-wide structural risk emits no floor event."""
    times = [f"2026-04-30 09:{30 + i:02d}" for i in range(10)]
    closes = [10.10, 10.20, 10.30, 10.30, 10.25, 10.25, 10.30, 10.30, 10.32, 10.50]
    # 10.00 is the lookback min, well below the 10.50 close (50¢ structural risk).
    lows = [10.00, 10.10, 10.20, 10.20, 10.20, 10.20, 10.25, 10.25, 10.25, 10.45]
    highs = [10.20, 10.30, 10.35, 10.35, 10.30, 10.30, 10.35, 10.35, 10.40, 10.55]
    idx = pd.to_datetime(times).tz_localize("America/New_York")
    bars = pd.DataFrame(
        {
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [10_000.0] * len(times),
            "vwap": closes,
        },
        index=idx,
    )
    strategy = MomentumStrategy(
        flag_max_pullback_pct=5.0,
        extended_from_vwap_atr_multiple=20.0,
    )
    with capture_logs() as captured:
        signal = strategy.evaluate("WIDE", bars)
    assert signal is not None
    assert signal.stop == pytest.approx(10.00)  # structural unchanged
    assert not any(
        e.get("event") == "entry.stop_distance_floor_applied" for e in captured
    )


def _gap_and_go_frame_with_tight_stop() -> pd.DataFrame:
    """Bar frame in the 09:30 gap-and-go window with a near-VWAP entry close.

    The structural stop is ``min(VWAP, 3-bar pullback low)``. We construct
    a frame where VWAP sits just below the breakout close (≤ 1¢ below)
    so the structural risk is sub-floor and the Phase 10.2 floor binds.
    """
    times = [f"2026-04-30 09:{30 + i:02d}" for i in range(5)]
    # Closes hover at 2.18; final bar fresh HOD by 1¢.
    closes = [2.17, 2.17, 2.17, 2.17, 2.18]
    highs = [2.175, 2.175, 2.175, 2.175, 2.185]
    lows = [2.17, 2.17, 2.17, 2.17, 2.175]
    idx = pd.to_datetime(times).tz_localize("America/New_York")
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [10_000.0] * len(times),
            "vwap": closes,
        },
        index=idx,
    )


def test_gap_and_go_floor_applies_when_structural_too_tight() -> None:
    """gap_and_go shares the floor — same pathology, same fix."""
    bars = _gap_and_go_frame_with_tight_stop()
    strategy = GapAndGoStrategy(
        vwap_extension_grace_minutes=15,
        extended_from_vwap_atr_multiple=20.0,
    )
    with capture_logs() as captured:
        signal = strategy.evaluate("ZENA", bars)
    assert signal is not None
    assert signal.entry == pytest.approx(2.18)
    # Structural is min(vwap≈2.174, pullback_low=2.17) = 2.17 (rounded).
    # Floored to 2.13 (5¢ floor wins).
    assert signal.stop == pytest.approx(2.13)
    assert any(
        e.get("event") == "entry.stop_distance_floor_applied"
        and e.get("strategy") == "gap_and_go"
        for e in captured
    )

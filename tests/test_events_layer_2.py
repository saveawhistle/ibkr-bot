"""Layer 2 detector tests.

Each event type has a positive case (event fires under the right
conditions) and a negative case (does not fire when conditions aren't
met). Once-per-session semantics get a dedicated test where applicable.
The tests run detectors directly against synthetic ``Bar`` sequences,
not through the full harness, to keep the assertions focused on
detection logic rather than harness wiring.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from bot.exit_advisor.core.events import (
    BarShapeDetected,
    ConsecutiveBars,
    Event,
    LevelDataUnavailable,
    LevelReclaimed,
    LevelTouched,
    MovingAverageCross,
    RVolDataUnavailable,
    RVolMilestone,
    VolumeDryUp,
    VolumeSpike,
    WickEvent,
)
from bot.exit_advisor.detectors.bar_shape import BarShapeDetector
from bot.exit_advisor.detectors.moving_averages import MovingAveragesDetector
from bot.exit_advisor.detectors.price_levels import PriceLevelsDetector
from bot.exit_advisor.detectors.volume import VolumeDetector
from bot.exit_advisor.replay.bar_history import BarHistory
from bot.exit_advisor.replay.replay_source import Bar


def _ts(minute: int) -> datetime:
    """13:30 UTC + minute. Aligns with 09:30 ET on a DST date."""
    return datetime(2026, 4, 30, 13, 30, tzinfo=UTC) + timedelta(minutes=minute)


def _bar(minute: int, o: float, h: float, low: float, c: float, v: int = 1000) -> Bar:
    return Bar(timestamp=_ts(minute), open=o, high=h, low=low, close=c, volume=v)


def _run(detector, bars: list[Bar]) -> list[Event]:  # type: ignore[no-untyped-def]
    history = BarHistory()
    out: list[Event] = []
    for b in bars:
        history.add_bar(b)
        out.extend(detector.on_bar(b, history))
    return out


# --- price_levels ---


def test_level_touched_hod_fires_once() -> None:
    """HOD touched fires on the bar that sets a new HOD. A subsequent bar
    that re-touches the same HOD without meaningful retreat does NOT
    re-fire."""
    det = PriceLevelsDetector(symbol="X")
    events = _run(
        det,
        [
            _bar(0, 10.0, 10.5, 9.9, 10.4),  # HOD = 10.5
            _bar(1, 10.4, 10.5, 10.3, 10.4),  # touches same HOD
        ],
    )
    touched = [e for e in events if isinstance(e, LevelTouched) and e.level_name == "hod"]
    assert len(touched) == 1


def test_level_touched_re_arms_after_meaningful_retreat() -> None:
    """A retreat of >= 0.5% on a 1-min close re-arms the latch; the next
    HOD touch fires another event."""
    det = PriceLevelsDetector(symbol="X")
    events = _run(
        det,
        [
            _bar(0, 10.0, 10.5, 10.0, 10.4),
            _bar(1, 10.4, 10.4, 9.5, 9.5),  # close 9.5 = 9.5% below HOD
            _bar(2, 9.5, 10.5, 9.5, 10.4),  # back to HOD
        ],
    )
    touched_below = [
        e for e in events if isinstance(e, LevelTouched)
        and e.level_name == "hod" and e.direction == "from_below"
    ]
    assert len(touched_below) == 2


def test_prior_day_close_fires_when_data_available() -> None:
    det = PriceLevelsDetector(symbol="X", prior_day_close=10.0)
    events = _run(det, [_bar(0, 9.5, 10.5, 9.5, 10.2)])
    assert any(
        isinstance(e, LevelTouched) and e.level_name == "prior_day_close" for e in events
    )


def test_prior_day_data_missing_emits_warning_once() -> None:
    det = PriceLevelsDetector(symbol="X")  # no prior-day data
    events = _run(det, [_bar(0, 10.0, 10.1, 9.9, 10.05), _bar(1, 10.05, 10.2, 10.0, 10.1)])
    warnings = [
        e for e in events
        if isinstance(e, LevelDataUnavailable) and e.level_name == "prior_day_high"
    ]
    assert len(warnings) == 1


def test_gap_fill_only_fires_when_gap_exceeds_threshold() -> None:
    """Today open 10.10 vs prior close 10.00 = 1% gap = at threshold;
    today open 10.05 vs 10.00 = 0.5% < threshold → no gap_fill events."""
    det_with_gap = PriceLevelsDetector(
        symbol="X",
        prior_day_close=10.0,
        today_open=10.20,  # 2% gap
        gap_threshold_pct=0.01,
    )
    events_with = _run(det_with_gap, [_bar(0, 10.20, 10.25, 9.95, 10.10)])
    assert any(isinstance(e, LevelTouched) and e.level_name == "gap_fill" for e in events_with)

    det_no_gap = PriceLevelsDetector(
        symbol="X",
        prior_day_close=10.0,
        today_open=10.05,  # 0.5% < 1% threshold
        gap_threshold_pct=0.01,
    )
    events_no = _run(det_no_gap, [_bar(0, 10.05, 10.20, 9.95, 10.10)])
    assert not any(
        isinstance(e, LevelTouched) and e.level_name == "gap_fill" for e in events_no
    )


def test_level_reclaimed_requires_prior_break() -> None:
    """Reclaimed only fires after price has been on the opposite side
    of the level. A bar that closes through a level for the first time
    sets the side but does not emit a reclaim event."""
    det = PriceLevelsDetector(symbol="X", prior_day_close=10.0)
    events = _run(
        det,
        [
            _bar(0, 9.5, 9.8, 9.4, 9.7),  # below 10.0
            _bar(1, 9.7, 10.5, 9.7, 10.3),  # touches & closes above; reclaim from below
            _bar(2, 10.3, 10.4, 9.6, 9.8),  # closes below; reclaim from above
        ],
    )
    reclaims = [e for e in events if isinstance(e, LevelReclaimed) and e.level_name == "prior_day_close"]
    assert len(reclaims) == 2
    assert reclaims[0].direction == "below_to_above"
    assert reclaims[1].direction == "above_to_below"


# --- moving_averages ---


def test_vwap_cross_fires_on_side_flip() -> None:
    det = MovingAveragesDetector(symbol="X", ema_9_enabled=False)
    bars = [
        _bar(0, 10.0, 10.0, 10.0, 10.0, v=100),  # VWAP = 10.0; side = above? equal — skipped
        _bar(1, 10.0, 10.5, 10.0, 10.4, v=100),  # close 10.4 > VWAP — set side above
        _bar(2, 10.4, 10.4, 9.5, 9.6, v=100),   # close 9.6 < VWAP — flip
    ]
    events = _run(det, bars)
    crosses = [e for e in events if isinstance(e, MovingAverageCross) and e.ma_name == "vwap"]
    assert len(crosses) == 1
    assert crosses[0].direction == "price_above_to_below"


def test_ema_9_warmup_no_events_until_9_bars() -> None:
    det = MovingAveragesDetector(symbol="X", vwap_enabled=False)
    bars = [_bar(i, 10.0, 10.5, 9.5, 10.0 + (i % 2) * 0.2) for i in range(8)]
    events = _run(det, bars)
    ema_events = [e for e in events if isinstance(e, MovingAverageCross) and e.ma_name == "ema_9"]
    assert ema_events == []


def test_ema_9_value_after_seed() -> None:
    det = MovingAveragesDetector(symbol="X", vwap_enabled=False)
    bars = [_bar(i, 10.0, 10.5, 9.5, 10.0) for i in range(9)]
    _run(det, bars)
    assert det.ema_9_value() is not None
    assert abs(det.ema_9_value() - 10.0) < 1e-9  # type: ignore[operator]


# --- volume ---


def test_volume_spike_fires_when_ratio_crosses_threshold() -> None:
    det = VolumeDetector(symbol="X", baseline_window_bars=5, spike_threshold_x_avg=2.0)
    bars = [_bar(i, 10, 10, 10, 10, v=100) for i in range(5)]
    bars.append(_bar(5, 10, 10, 10, 10, v=300))  # 3.0x avg
    events = _run(det, bars)
    spikes = [e for e in events if isinstance(e, VolumeSpike)]
    assert len(spikes) == 1
    assert spikes[0].ratio == 3.0


def test_volume_spike_does_not_fire_until_baseline_warm() -> None:
    """First N-1 bars can't fire a spike — baseline window isn't yet
    full. The Nth bar populates the window; the (N+1)th can fire."""
    det = VolumeDetector(symbol="X", baseline_window_bars=20, spike_threshold_x_avg=2.0)
    bars = [_bar(i, 10, 10, 10, 10, v=100) for i in range(5)]
    bars.append(_bar(5, 10, 10, 10, 10, v=10000))
    events = _run(det, bars)
    assert not any(isinstance(e, VolumeSpike) for e in events)


def test_volume_dryup_fires_when_ratio_below_threshold() -> None:
    det = VolumeDetector(symbol="X", baseline_window_bars=5, dryup_threshold_x_avg=0.4)
    bars = [_bar(i, 10, 10, 10, 10, v=1000) for i in range(5)]
    bars.append(_bar(5, 10, 10, 10, 10, v=200))  # 0.2x avg
    events = _run(det, bars)
    dryups = [e for e in events if isinstance(e, VolumeDryUp)]
    assert len(dryups) == 1


def test_rvol_data_unavailable_fires_once() -> None:
    det = VolumeDetector(symbol="X", baseline_window_bars=2)
    events = _run(det, [_bar(0, 10, 10, 10, 10, v=100), _bar(1, 10, 10, 10, 10, v=100)])
    warnings = [e for e in events if isinstance(e, RVolDataUnavailable)]
    assert len(warnings) == 1


def test_rvol_milestone_fires_when_data_available() -> None:
    det = VolumeDetector(
        symbol="X",
        baseline_window_bars=10,
        rvol_milestones=[1.0, 2.0],
        prior_day_cum_volume_by_minute={0: 100, 1: 200, 2: 300},
    )
    bars = [
        _bar(0, 10, 10, 10, 10, v=150),  # cum=150 vs prior 100 = rvol 1.5 → fires 1.0
        _bar(1, 10, 10, 10, 10, v=300),  # cum=450 vs prior 200 = rvol 2.25 → fires 2.0
    ]
    events = _run(det, bars)
    milestones = [e for e in events if isinstance(e, RVolMilestone)]
    fired = {e.milestone for e in milestones}
    assert 1.0 in fired
    assert 2.0 in fired


# --- bar_shape ---


def test_doji_detected_when_body_lt_10pct_of_range() -> None:
    det = BarShapeDetector(symbol="X")
    events = _run(det, [_bar(0, 10.00, 10.50, 9.50, 10.01)])  # body=0.01, range=1.0 → 1%
    shapes = {e.shape for e in events if isinstance(e, BarShapeDetected)}
    assert "doji" in shapes


def test_doji_not_detected_when_body_gte_10pct() -> None:
    det = BarShapeDetector(symbol="X")
    events = _run(det, [_bar(0, 10.00, 10.50, 9.50, 10.20)])  # body=0.20 = 20%
    shapes = {e.shape for e in events if isinstance(e, BarShapeDetected)}
    assert "doji" not in shapes


def test_hammer_detected() -> None:
    """Body 9.95-10.0 (small green body in upper portion); long lower wick to 9.0;
    barely any upper wick. Hammer."""
    det = BarShapeDetector(symbol="X")
    events = _run(det, [_bar(0, 9.95, 10.00, 9.00, 10.00)])
    shapes = {e.shape for e in events if isinstance(e, BarShapeDetected)}
    assert "hammer" in shapes


def test_shooting_star_detected() -> None:
    det = BarShapeDetector(symbol="X")
    events = _run(det, [_bar(0, 10.00, 11.00, 9.95, 9.95)])  # body 0.05, upper wick 1.0
    shapes = {e.shape for e in events if isinstance(e, BarShapeDetected)}
    assert "shooting_star" in shapes


def test_engulfing_requires_prior_bar() -> None:
    det = BarShapeDetector(symbol="X")
    bars = [
        _bar(0, 10.0, 10.2, 9.9, 9.95),    # red small
        _bar(1, 9.90, 10.50, 9.85, 10.40), # green large engulfs prior body
    ]
    events = _run(det, bars)
    shapes = [e for e in events if isinstance(e, BarShapeDetected) and e.shape == "engulfing"]
    assert len(shapes) == 1


def test_inside_bar_detected() -> None:
    det = BarShapeDetector(symbol="X")
    bars = [
        _bar(0, 10.0, 11.0, 9.0, 10.5),
        _bar(1, 10.5, 10.8, 9.5, 10.2),  # h<11 and l>9 → inside
    ]
    events = _run(det, bars)
    shapes = [e for e in events if isinstance(e, BarShapeDetected) and e.shape == "inside_bar"]
    assert len(shapes) == 1


def test_outside_bar_detected() -> None:
    det = BarShapeDetector(symbol="X")
    bars = [
        _bar(0, 10.0, 10.5, 9.5, 10.2),
        _bar(1, 10.2, 11.0, 9.0, 10.8),  # h>10.5 and l<9.5 → outside
    ]
    events = _run(det, bars)
    shapes = [e for e in events if isinstance(e, BarShapeDetected) and e.shape == "outside_bar"]
    assert len(shapes) == 1


def test_wick_event_upper_and_lower_can_both_fire() -> None:
    det = BarShapeDetector(symbol="X", wick_threshold_pct=0.3)
    # range=2, body=0.6 (9.7 → 10.3), upper_wick=0.7, lower_wick=0.7 — both > 30%
    events = _run(det, [_bar(0, 9.7, 11.0, 9.0, 10.3)])
    sides = {e.wick_side for e in events if isinstance(e, WickEvent)}
    assert "upper" in sides
    assert "lower" in sides


def test_consecutive_bars_fires_at_and_past_threshold() -> None:
    det = BarShapeDetector(symbol="X", consecutive_bars_threshold=3)
    bars = [_bar(i, 10.0 + i * 0.1, 10.0 + i * 0.1 + 0.1, 9.95, 10.0 + i * 0.1 + 0.05) for i in range(5)]
    events = _run(det, bars)
    consecutive = [e for e in events if isinstance(e, ConsecutiveBars)]
    counts = [e.count for e in consecutive]
    assert counts == [3, 4, 5]


def test_consecutive_bars_resets_on_direction_flip() -> None:
    det = BarShapeDetector(symbol="X", consecutive_bars_threshold=3)
    bars = [
        _bar(0, 10.0, 10.1, 9.9, 10.05),  # green
        _bar(1, 10.05, 10.15, 9.95, 10.10),  # green
        _bar(2, 10.10, 10.10, 9.95, 9.95),  # red — streak resets
        _bar(3, 9.95, 9.95, 9.85, 9.90),  # red
    ]
    events = _run(det, bars)
    assert not any(isinstance(e, ConsecutiveBars) for e in events)

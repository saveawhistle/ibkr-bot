"""Unit tests for EventBuffer significance triggering."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from bot.exit_advisor.advisor.buffer import BufferDecision, EventBuffer
from bot.exit_advisor.core.events import (
    BarFinalizedEvent,
    DrawdownFromPeak,
    LargePrint,
    MovingAverageCross,
    OrderRejectionEvent,
    PartialFillEvent,
    PositionProtected,
    RMultipleReached,
    VolumeSpike,
)

_T0 = datetime(2026, 5, 4, 14, 30, 0, tzinfo=UTC)


def _bar(t: datetime) -> BarFinalizedEvent:
    return BarFinalizedEvent(
        timestamp=t, symbol="ABC", open=1.0, high=1.05, low=0.99, close=1.04, volume=1000
    )


def _l2_print(t: datetime) -> LargePrint:
    return LargePrint(
        timestamp=t,
        symbol="ABC",
        price=1.04,
        size=5000,
        rolling_average_size=500.0,
        ratio=10.0,
        aggressor_side="buy",
    )


def test_constructor_validates_floors() -> None:
    with pytest.raises(ValueError):
        EventBuffer(time_floor_seconds=-1.0)
    with pytest.raises(ValueError):
        EventBuffer(hard_floor_seconds=-1.0)


def test_always_trigger_event_fires_on_arrival() -> None:
    buf = EventBuffer()
    event = PositionProtected(
        timestamp=_T0,
        symbol="ABC",
        entry_price=1.0,
        initial_stop=0.95,
        initial_scale_out=1.10,
        position_size=100,
    )
    decision = buf.consume(event, _T0)
    assert decision.trigger
    assert decision.triggering_event is event
    assert event in decision.buffered_events


def test_buffer_only_event_never_triggers() -> None:
    buf = EventBuffer()
    decision = buf.consume(_l2_print(_T0), _T0)
    assert not decision.trigger
    assert decision.skip_reason == "non_significant"
    assert buf.pending_count() == 1


def test_time_floor_event_blocked_until_elapsed() -> None:
    buf = EventBuffer(time_floor_seconds=30.0, hard_floor_seconds=10.0)
    # First trigger: bar event needs no prior trigger to fire.
    first = buf.consume(_bar(_T0), _T0)
    assert first.trigger

    # 5 seconds later: another bar — within hard floor.
    too_soon = buf.consume(_bar(_T0 + timedelta(seconds=5)), _T0 + timedelta(seconds=5))
    assert not too_soon.trigger
    assert too_soon.skip_reason == "hard_floor_active"

    # 15 seconds later: past hard floor, still within time floor.
    still_too_soon = buf.consume(_bar(_T0 + timedelta(seconds=15)), _T0 + timedelta(seconds=15))
    assert not still_too_soon.trigger
    assert still_too_soon.skip_reason == "time_floor_active"

    # 35 seconds later: past time floor.
    ok = buf.consume(_bar(_T0 + timedelta(seconds=35)), _T0 + timedelta(seconds=35))
    assert ok.trigger


def test_hard_floor_caps_always_trigger_events() -> None:
    buf = EventBuffer(hard_floor_seconds=10.0)
    first_event = PartialFillEvent(
        timestamp=_T0,
        symbol="ABC",
        order_id=1,
        filled_quantity=50,
        remaining_quantity=50,
        fill_price=1.0,
        side="buy",
    )
    first = buf.consume(first_event, _T0)
    assert first.trigger

    second_event = PartialFillEvent(
        timestamp=_T0 + timedelta(seconds=3),
        symbol="ABC",
        order_id=1,
        filled_quantity=50,
        remaining_quantity=0,
        fill_price=1.01,
        side="buy",
    )
    second = buf.consume(second_event, _T0 + timedelta(seconds=3))
    assert not second.trigger
    assert second.skip_reason == "hard_floor_active"


def test_drains_on_trigger_and_accumulates_between_triggers() -> None:
    buf = EventBuffer()
    # First a buffer-only event accumulates.
    buf.consume(_l2_print(_T0), _T0)
    buf.consume(
        VolumeSpike(
            timestamp=_T0,
            symbol="ABC",
            bar_volume=10000,
            rolling_average=1000.0,
            ratio=10.0,
            threshold=2.0,
        ),
        _T0,
    )
    assert buf.pending_count() == 2

    # Now an always-trigger event fires; drained buffer should include all 3.
    drawdown = DrawdownFromPeak(
        timestamp=_T0,
        symbol="ABC",
        drawdown_pct=0.5,
        peak_r_multiple=1.5,
        current_r_multiple=0.75,
    )
    decision = buf.consume(drawdown, _T0)
    assert decision.trigger
    assert len(decision.buffered_events) == 3
    assert buf.pending_count() == 0

    # Subsequent buffer-only event should accumulate into a fresh slot.
    buf.consume(_l2_print(_T0 + timedelta(seconds=1)), _T0 + timedelta(seconds=1))
    assert buf.pending_count() == 1


def test_all_documented_always_trigger_classes_actually_trigger() -> None:
    events: list[BufferDecision] = []
    candidates = [
        PositionProtected(
            timestamp=_T0,
            symbol="ABC",
            entry_price=1.0,
            initial_stop=0.95,
            initial_scale_out=1.1,
            position_size=100,
        ),
        PartialFillEvent(
            timestamp=_T0,
            symbol="ABC",
            order_id=1,
            filled_quantity=10,
            remaining_quantity=0,
            fill_price=1.0,
            side="buy",
        ),
        OrderRejectionEvent(
            timestamp=_T0, symbol="ABC", order_id=2, error_code=201, reason="margin"
        ),
        RMultipleReached(timestamp=_T0, symbol="ABC", r_multiple=1.0, direction="up"),
        DrawdownFromPeak(
            timestamp=_T0,
            symbol="ABC",
            drawdown_pct=0.5,
            peak_r_multiple=2.0,
            current_r_multiple=1.0,
        ),
    ]
    for ev in candidates:
        # Reset the buffer's last-trigger so each candidate is independent.
        events.append(EventBuffer(hard_floor_seconds=0.0).consume(ev, _T0))
    assert all(d.trigger for d in events)


def test_time_floor_class_triggers_when_no_prior_call() -> None:
    buf = EventBuffer()
    cross = MovingAverageCross(
        timestamp=_T0,
        symbol="ABC",
        ma_name="vwap",
        ma_value=1.02,
        direction="price_above_to_below",
        bar_close=1.01,
    )
    decision = buf.consume(cross, _T0)
    assert decision.trigger, "time-floor event should fire when no prior trigger exists"

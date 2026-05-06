"""Per-event-class tests using a synthetic replay-data fixture.

Each layer-1 event type has its own positive test (fires under the right
conditions) plus a config-flag test (does not fire when its class is
disabled).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from bot.config import ExitEventsConfig
from bot.exit_advisor.core.events import (
    DrawdownFromPeak,
    MaxFavorableExcursionUpdate,
    PartialFillEvent,
    PositionProtected,
    RMultipleReached,
    TimeInTradeMilestone,
    TimeOfDayMilestone,
)
from bot.exit_advisor.replay.harness import TradeReplayHarness
from bot.exit_advisor.replay.replay_source import Bar, TradeReplayData


def _ts(h: int, m: int, s: int = 0, ms: int = 0) -> datetime:
    """UTC timestamp on 2026-04-30 at HH:MM:SS.MS — keeps the synthetic
    fixture's date aligned with ``logs/session_2026-04-30.jsonl``'s ZENA."""
    return datetime(2026, 4, 30, h, m, s, ms * 1000, tzinfo=UTC)


def _build_replay(
    bars: list[Bar], exit_price: float = 1.95, exit_minutes: int = 10
) -> TradeReplayData:
    """Synthetic long trade: entry 2.00, stop 1.90 ($0.10 risk), 100 shares."""
    entry_ts = _ts(13, 30, 0)  # 09:30 ET in DST
    exit_ts = entry_ts + timedelta(minutes=exit_minutes)

    bracket_event = {
        "symbol": "TST",
        "parent_order_id": 1,
        "shares": 100,
        "entry_price": 2.00,
        "limit_price": 2.10,
        "stop_price": 1.90,
        "event": "executor.lmt_bracket_placed",
        "timestamp": entry_ts.isoformat().replace("+00:00", "Z"),
    }
    entry_event = {
        "symbol": "TST",
        "parent_order_id": 1,
        "shares": 100,
        "stop": 1.90,
        "scale_out": 2.20,
        "event": "position.opened",
        "timestamp": entry_ts.isoformat().replace("+00:00", "Z"),
    }
    fill_event = {
        "symbol": "TST",
        "fill_price": 2.00,
        "filled_shares": 100,
        "event": "position.filled",
        "timestamp": entry_ts.isoformat().replace("+00:00", "Z"),
    }
    protection_event = {
        "symbol": "TST",
        "new_stop_price": 1.90,
        "scale_lmt_price": 2.20,
        "event": "executor.protection_fill_anchored",
        "timestamp": entry_ts.isoformat().replace("+00:00", "Z"),
    }
    exit_pnl = (exit_price - 2.00) * 100
    exit_event = {
        "symbol": "TST",
        "exit_price": exit_price,
        "pnl": exit_pnl,
        "reason": "stop",
        "event": "position.closed",
        "timestamp": exit_ts.isoformat().replace("+00:00", "Z"),
    }
    return TradeReplayData(
        symbol="TST",
        trade_date=date(2026, 4, 30),
        bars=bars,
        entry_event=entry_event,
        bracket_event=bracket_event,
        order_events=[],
        exit_event=exit_event,
        recorded_pnl=exit_pnl,
        recorded_exit_price=exit_price,
        recorded_exit_timestamp=exit_ts,
        fill_event=fill_event,
        protection_anchored_event=protection_event,
    )


class _NullPolicy:
    def on_event(self, trade_state, event):  # type: ignore[no-untyped-def]
        return None


def test_position_protected_fires_once() -> None:
    """PositionProtected emits exactly once at the protection timestamp."""
    rd = _build_replay(bars=[])
    harness = TradeReplayHarness(rd, _NullPolicy(), ExitEventsConfig())
    result = harness.run()
    protected = [e for e in result.events_emitted if isinstance(e, PositionProtected)]
    assert len(protected) == 1
    assert protected[0].entry_price == 2.00
    assert protected[0].initial_stop == 1.90
    assert protected[0].position_size == 100


def test_r_multiple_up_fires_on_favorable_bar() -> None:
    """Bar high 2.10 = +1R; should fire R=1.0 up (not down)."""
    bars = [Bar(_ts(13, 30), 2.00, 2.10, 1.95, 2.05, 1000)]
    rd = _build_replay(bars=bars)
    harness = TradeReplayHarness(rd, _NullPolicy(), ExitEventsConfig())
    result = harness.run()

    r_events = [e for e in result.events_emitted if isinstance(e, RMultipleReached)]
    ups = [e for e in r_events if e.direction == "up"]
    assert any(e.r_multiple == 1.0 for e in ups)
    assert any(e.r_multiple == 0.5 for e in ups)


def test_r_multiple_down_fires_on_adverse_bar() -> None:
    bars = [Bar(_ts(13, 30), 2.00, 2.00, 1.85, 1.95, 1000)]
    rd = _build_replay(bars=bars)
    harness = TradeReplayHarness(rd, _NullPolicy(), ExitEventsConfig())
    result = harness.run()
    r_events = [e for e in result.events_emitted if isinstance(e, RMultipleReached)]
    downs = [e for e in r_events if e.direction == "down"]
    assert any(e.r_multiple == 1.0 for e in downs)


def test_drawdown_from_peak_fires() -> None:
    """Bar 1: high 2.20 = +2R peak. Bar 2: close 2.10 = +1R, drawdown 50%
    from peak triggers the 0.5 threshold (and the 0.25 threshold)."""
    bars = [
        Bar(_ts(13, 30), 2.00, 2.20, 2.00, 2.18, 1000),
        Bar(_ts(13, 31), 2.18, 2.18, 2.05, 2.10, 1000),
    ]
    rd = _build_replay(bars=bars, exit_minutes=15)
    harness = TradeReplayHarness(rd, _NullPolicy(), ExitEventsConfig())
    result = harness.run()
    dd = [e for e in result.events_emitted if isinstance(e, DrawdownFromPeak)]
    thresholds = {round(e.drawdown_pct, 4) for e in dd}
    assert 0.25 in thresholds
    assert 0.5 in thresholds


def test_mfe_update_fires_on_new_peak() -> None:
    bars = [
        Bar(_ts(13, 30), 2.00, 2.10, 1.95, 2.05, 1000),
        Bar(_ts(13, 31), 2.05, 2.20, 2.05, 2.15, 1000),
    ]
    rd = _build_replay(bars=bars, exit_minutes=15)
    harness = TradeReplayHarness(rd, _NullPolicy(), ExitEventsConfig())
    result = harness.run()
    mfe = [e for e in result.events_emitted if isinstance(e, MaxFavorableExcursionUpdate)]
    assert len(mfe) >= 2  # bar 1 (peak rises to 2.10), bar 2 (peak rises to 2.20)
    assert mfe[-1].new_peak_r_multiple > mfe[-1].previous_peak_r_multiple


def test_time_of_day_milestone_fires() -> None:
    """Bar at 09:35 ET (13:35 UTC during DST) = 5 min after open."""
    bars = [Bar(_ts(13, 34), 2.00, 2.05, 1.95, 2.02, 1000)]
    rd = _build_replay(bars=bars)
    harness = TradeReplayHarness(rd, _NullPolicy(), ExitEventsConfig())
    result = harness.run()
    tod = [e for e in result.events_emitted if isinstance(e, TimeOfDayMilestone)]
    assert 5 in {e.minutes_after_open for e in tod}


def test_time_in_trade_milestone_fires() -> None:
    bars = [
        Bar(_ts(13, 30), 2.00, 2.05, 1.95, 2.02, 1000),
        Bar(_ts(13, 32), 2.02, 2.04, 2.00, 2.01, 1000),
    ]
    rd = _build_replay(bars=bars, exit_minutes=10)
    harness = TradeReplayHarness(rd, _NullPolicy(), ExitEventsConfig())
    result = harness.run()
    in_trade = [e for e in result.events_emitted if isinstance(e, TimeInTradeMilestone)]
    assert 2 in {e.minutes_in_trade for e in in_trade}


def test_pnl_disabled_does_not_emit_pnl_events() -> None:
    """Setting exit_events.pnl.enabled=False silences R-multiple, MFE,
    drawdown — even when bar data would naturally trigger them."""
    bars = [Bar(_ts(13, 30), 2.00, 2.20, 1.85, 2.05, 1000)]
    rd = _build_replay(bars=bars)
    cfg = ExitEventsConfig()
    cfg.pnl.enabled = False
    harness = TradeReplayHarness(rd, _NullPolicy(), cfg)
    result = harness.run()

    assert not any(isinstance(e, RMultipleReached) for e in result.events_emitted)
    assert not any(isinstance(e, MaxFavorableExcursionUpdate) for e in result.events_emitted)
    assert not any(isinstance(e, DrawdownFromPeak) for e in result.events_emitted)


def test_time_disabled_does_not_emit_time_events() -> None:
    bars = [Bar(_ts(13, 34), 2.00, 2.05, 1.95, 2.02, 1000)]
    rd = _build_replay(bars=bars)
    cfg = ExitEventsConfig()
    cfg.time.enabled = False
    harness = TradeReplayHarness(rd, _NullPolicy(), cfg)
    result = harness.run()

    assert not any(isinstance(e, TimeOfDayMilestone) for e in result.events_emitted)
    assert not any(isinstance(e, TimeInTradeMilestone) for e in result.events_emitted)


def test_partial_fill_event_construction() -> None:
    """PartialFillEvent shape — layer 1 leaves emission as a synthetic
    construction site since session logs do not yet record partial fills
    structurally. Verifying the event class itself is wired up so layer
    2 can plug into it."""
    e = PartialFillEvent(
        timestamp=_ts(13, 30),
        symbol="TST",
        order_id=42,
        filled_quantity=50,
        remaining_quantity=50,
        fill_price=2.00,
        side="buy",
    )
    assert e.symbol == "TST"
    assert e.filled_quantity == 50
    assert e.side == "buy"


def test_layer_3_plus_classes_still_gated() -> None:
    """Layer 2 unblocked price_levels / moving_averages / volume / bar_shape;
    layer L2-A unblocked l2; only market_context / news / halts stay gated.
    Config validation must reject any YAML that flips these on so the
    failure mode stays loud rather than silent."""
    for class_name in ("market_context", "news", "halts"):
        cfg_dict = {class_name: {"enabled": True}}
        with pytest.raises(ValueError, match="not implemented in this layer"):
            ExitEventsConfig.model_validate(cfg_dict)


def test_layer_2_classes_now_permitted() -> None:
    """Sanity: layer 2 classes that were rejected in layer 1's gate now
    load without error."""
    cfg = ExitEventsConfig.model_validate(
        {
            "price_levels": {"enabled": True},
            "moving_averages": {"enabled": True},
            "volume": {"enabled": True},
            "bar_shape": {"enabled": True},
        }
    )
    assert cfg.price_levels.enabled
    assert cfg.moving_averages.enabled
    assert cfg.volume.enabled
    assert cfg.bar_shape.enabled

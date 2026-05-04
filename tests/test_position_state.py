"""Tests for ``bot.execution.position_state`` — status machine + multi-symbol isolation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from bot.execution.position_state import (
    InvalidPositionTransitionError,
    PositionStore,
    SymbolHistory,
    UnknownPositionError,
)

_OPEN_TS = datetime(2026, 4, 16, 9, 31, tzinfo=UTC)
_CLOSE_TS = datetime(2026, 4, 16, 9, 45, tzinfo=UTC)


def _store_with_one(symbol: str = "AAA") -> PositionStore:
    """Build a store containing a single pending-entry position for ``symbol``."""
    store = PositionStore()
    store.open_position(
        symbol=symbol,
        strategy="gap_and_go",
        shares=20,
        stop_price=9.0,
        scale_out_price=13.0,
        runner_target_price=13.0,
        parent_order_id=1,
        stop_order_id=2,
        target_order_id=3,
        opened_at=_OPEN_TS,
    )
    return store


def test_happy_path_open_filled_closed() -> None:
    """The canonical life cycle: pending_entry → open → closed with a winning PnL."""
    store = _store_with_one()

    filled = store.mark_filled("AAA", fill_price=10.05, filled_shares=20)
    assert filled.status == "open"
    assert filled.avg_price == 10.05
    assert filled.shares == 20

    closed = store.mark_closed("AAA", exit_price=13.0, pnl=(13.0 - 10.05) * 20, closed_at=_CLOSE_TS)
    assert closed.status == "closed"
    assert closed.exit_price == 13.0
    assert closed.realized_pnl == pytest.approx(59.0)
    assert closed.closed_at == _CLOSE_TS


def test_has_active_across_states() -> None:
    """``has_active`` is True for pending/open/closing, False for closed or unknown."""
    store = _store_with_one()
    assert store.has_active("AAA")
    store.mark_filled("AAA", fill_price=10.0, filled_shares=20)
    assert store.has_active("AAA")
    store.mark_closing("AAA", reason="stop_hit")
    assert store.has_active("AAA")
    store.mark_closed("AAA", exit_price=9.0, pnl=-20.0, closed_at=_CLOSE_TS)
    assert not store.has_active("AAA")
    assert not store.has_active("UNKNOWN")


def test_two_symbols_are_isolated() -> None:
    """Opening a second symbol does not affect the first's lifecycle."""
    store = _store_with_one("AAA")
    store.open_position(
        symbol="BBB",
        strategy="momentum",
        shares=10,
        stop_price=4.0,
        scale_out_price=7.0,
        runner_target_price=7.0,
        parent_order_id=11,
        stop_order_id=12,
        target_order_id=13,
        opened_at=_OPEN_TS,
    )
    assert store.has_active("AAA")
    assert store.has_active("BBB")
    store.mark_closed("AAA", exit_price=9.0, pnl=-10.0, closed_at=_CLOSE_TS)
    assert not store.has_active("AAA")
    assert store.has_active("BBB")


def test_list_active_filters_closed() -> None:
    """Closed positions drop out of ``list_active`` but remain retrievable via ``get``."""
    store = _store_with_one("AAA")
    store.open_position(
        symbol="BBB",
        strategy="momentum",
        shares=10,
        stop_price=4.0,
        scale_out_price=7.0,
        runner_target_price=7.0,
        parent_order_id=11,
        stop_order_id=12,
        target_order_id=13,
        opened_at=_OPEN_TS,
    )
    store.mark_closed("AAA", exit_price=9.0, pnl=-20.0, closed_at=_CLOSE_TS)
    active = store.list_active()
    assert [p.symbol for p in active] == ["BBB"]
    closed = store.get("AAA")
    assert closed is not None
    assert closed.status == "closed"


def test_mark_filled_unknown_symbol_raises() -> None:
    """Mutating a symbol the store never saw surfaces a clean ``UnknownPositionError``."""
    store = PositionStore()
    with pytest.raises(UnknownPositionError):
        store.mark_filled("GHOST", fill_price=10.0, filled_shares=5)


def test_symbol_history_first_touch_creates_blank() -> None:
    """``symbol_history`` is get-or-create; first call returns a zero-count blank."""
    store = PositionStore()
    history = store.symbol_history("AAA")
    assert isinstance(history, SymbolHistory)
    assert history.symbol == "AAA"
    assert history.entries_count == 0
    assert history.last_exit_type is None
    # Second call returns the same instance (not a fresh one).
    assert store.symbol_history("AAA") is history


def test_symbol_history_records_entry_and_exit() -> None:
    """``record_entry`` bumps the counter; ``record_exit`` sets the last-exit trio."""
    store = PositionStore()
    history = store.symbol_history("AAA")
    history.record_entry()
    history.record_entry()
    assert history.entries_count == 2
    exit_ts = _OPEN_TS + timedelta(minutes=5)
    history.record_exit(exit_time=exit_ts, pnl=42.0, exit_type="target_hit")
    assert history.last_exit_time == exit_ts
    assert history.last_exit_pnl == 42.0
    assert history.last_exit_type == "target_hit"
    # Subsequent exit overwrites, entries_count unaffected.
    history.record_exit(exit_time=exit_ts + timedelta(minutes=1), pnl=-10.0, exit_type="stop_hit")
    assert history.entries_count == 2
    assert history.last_exit_type == "stop_hit"


def test_reset_symbol_histories_clears_all() -> None:
    """Session-start reset drops every per-symbol history."""
    store = PositionStore()
    store.symbol_history("AAA").record_entry()
    store.symbol_history("BBB").record_entry()
    store.reset_symbol_histories()
    # Fresh access rebuilds a blank — entries_count starts at 0 again.
    assert store.symbol_history("AAA").entries_count == 0
    # list_symbol_histories reflects only the freshly-touched entry.
    assert [h.symbol for h in store.list_symbol_histories()] == ["AAA"]


def test_closed_is_terminal() -> None:
    """No transition escapes ``closed`` — it's a one-way machine."""
    store = _store_with_one()
    store.mark_filled("AAA", fill_price=10.0, filled_shares=20)
    store.mark_closed("AAA", exit_price=13.0, pnl=60.0, closed_at=_CLOSE_TS)
    with pytest.raises(InvalidPositionTransitionError):
        store.mark_filled("AAA", fill_price=10.0, filled_shares=20)
    with pytest.raises(InvalidPositionTransitionError):
        store.mark_closing("AAA", reason="late_signal")
    with pytest.raises(InvalidPositionTransitionError):
        store.mark_closed("AAA", exit_price=14.0, pnl=80.0, closed_at=_CLOSE_TS)


def test_mark_scaled_persists_post_scaleout_fields() -> None:
    """Phase 4h — ``mark_scaled`` writes the stop-type + adjustment-trigger onto the Position."""
    store = _store_with_one()
    store.mark_filled("AAA", fill_price=10.0, filled_shares=20)
    updated = store.mark_scaled(
        "AAA",
        remaining_shares=10,
        scale_partial_pnl=10.0,
        new_stop_price=10.0,
        new_stop_order_id=99,
        post_scaleout_stop_type="adjustable_to_trail",
        post_scaleout_adjustment_trigger_price=12.0,
    )
    assert updated.scaled_out is True
    assert updated.post_scaleout_stop_type == "adjustable_to_trail"
    assert updated.post_scaleout_adjustment_trigger_price == pytest.approx(12.0)


def test_mark_scaled_defaults_leave_post_scaleout_fields_unset() -> None:
    """Callers that omit the Phase 4h kwargs still get a valid Position with NULL fields."""
    store = _store_with_one()
    store.mark_filled("AAA", fill_price=10.0, filled_shares=20)
    updated = store.mark_scaled(
        "AAA",
        remaining_shares=10,
        scale_partial_pnl=10.0,
        new_stop_price=10.0,
        new_stop_order_id=99,
    )
    assert updated.scaled_out is True
    assert updated.post_scaleout_stop_type is None
    assert updated.post_scaleout_adjustment_trigger_price is None


def test_position_default_red_candle_exit_suppressed_is_false() -> None:
    """Phase 4i: a freshly-opened Position does not yet suppress red-candle exits."""
    store = _store_with_one()
    position = store.get_active("AAA")
    assert position is not None
    assert position.red_candle_exit_suppressed is False


def test_mark_scaled_sets_red_candle_exit_suppressed_by_default() -> None:
    """Phase 4i: scale-out flips the suppression flag unless explicitly opted out."""
    store = _store_with_one()
    store.mark_filled("AAA", fill_price=10.0, filled_shares=20)
    updated = store.mark_scaled(
        "AAA",
        remaining_shares=10,
        scale_partial_pnl=40.0,
        new_stop_price=10.0,
        new_stop_order_id=99,
    )
    assert updated.red_candle_exit_suppressed is True


def test_mark_scaled_allows_explicit_false_for_red_candle_suppression() -> None:
    """Phase 4i: pre-4h static-breakeven callers can opt out of the suppression."""
    store = _store_with_one()
    store.mark_filled("AAA", fill_price=10.0, filled_shares=20)
    updated = store.mark_scaled(
        "AAA",
        remaining_shares=10,
        scale_partial_pnl=40.0,
        new_stop_price=10.0,
        new_stop_order_id=99,
        red_candle_exit_suppressed=False,
    )
    assert updated.red_candle_exit_suppressed is False


def test_position_runner_target_price_is_optional() -> None:
    """Phase 4i: positions may open with no runner ceiling (runner_target disabled)."""
    store = PositionStore()
    store.open_position(
        symbol="NONE",
        strategy="gap_and_go",
        shares=20,
        stop_price=9.0,
        scale_out_price=12.0,
        runner_target_price=None,
        parent_order_id=42,
        stop_order_id=43,
        target_order_id=0,
        opened_at=_OPEN_TS,
    )
    position = store.get_active("NONE")
    assert position is not None
    assert position.runner_target_price is None


# ---------- Phase 4j pending_entry_trigger state + entry_order_type ---------- #


def test_open_position_pending_entry_trigger_status() -> None:
    """Phase 4j — STP-LMT resting parent opens in the pre-trigger state."""
    store = PositionStore()
    position = store.open_position(
        symbol="STPL",
        strategy="gap_and_go",
        shares=50,
        stop_price=9.0,
        scale_out_price=13.0,
        runner_target_price=None,
        parent_order_id=100,
        stop_order_id=0,
        target_order_id=0,
        opened_at=_OPEN_TS,
        status="pending_entry_trigger",
        entry_order_type="STP_LMT",
        entry_trigger_price=10.0,
    )
    assert position.status == "pending_entry_trigger"
    assert position.entry_order_type == "STP_LMT"
    assert position.entry_trigger_price == pytest.approx(10.0)
    assert store.has_active("STPL")


def test_mark_filled_transitions_from_pending_entry_trigger() -> None:
    """STP-LMT parent triggers + fills → ``pending_entry_trigger`` → ``open`` directly."""
    store = PositionStore()
    store.open_position(
        symbol="STPL",
        strategy="gap_and_go",
        shares=50,
        stop_price=9.0,
        scale_out_price=13.0,
        runner_target_price=None,
        parent_order_id=100,
        stop_order_id=0,
        target_order_id=0,
        opened_at=_OPEN_TS,
        status="pending_entry_trigger",
        entry_order_type="STP_LMT",
        entry_trigger_price=10.0,
    )
    filled = store.mark_filled("STPL", fill_price=10.07, filled_shares=50)
    assert filled.status == "open"
    assert filled.avg_price == pytest.approx(10.07)
    assert filled.entry_order_type == "STP_LMT"


def test_mark_entry_never_triggered_closes_with_reason() -> None:
    """Auto-flatten cancels a resting parent → closed with ``entry_never_triggered``."""
    store = PositionStore()
    store.open_position(
        symbol="STPL",
        strategy="gap_and_go",
        shares=50,
        stop_price=9.0,
        scale_out_price=13.0,
        runner_target_price=None,
        parent_order_id=100,
        stop_order_id=0,
        target_order_id=0,
        opened_at=_OPEN_TS,
        status="pending_entry_trigger",
        entry_order_type="STP_LMT",
        entry_trigger_price=10.0,
    )
    closed = store.mark_entry_never_triggered("STPL", closed_at=_CLOSE_TS)
    assert closed.status == "closed"
    assert closed.closing_reason == "entry_never_triggered"
    assert closed.realized_pnl == 0.0
    assert closed.exit_price == 0.0
    assert not store.has_active("STPL")


def test_mark_entry_never_triggered_accepts_pending_entry_lmt_path() -> None:
    """Phase 6.5 widens the contract — LMT pending_entry can also auto-expire.

    The pre-6.5 behavior required ``pending_entry_trigger`` (Phase 4j STP-LMT
    path). The orchestrator's auto-expire on the next-bar boundary applies
    equally to both entry order types, so the close transition is symmetric.
    """
    store = _store_with_one()  # opens in pending_entry (LMT path)
    closed = store.mark_entry_never_triggered("AAA", closed_at=_CLOSE_TS)
    assert closed.status == "closed"
    assert closed.closing_reason == "entry_never_triggered"


def test_mark_entry_never_triggered_rejects_open_state() -> None:
    """A filled position can't be auto-expired — must close via the normal exit path."""
    store = _store_with_one()
    store.mark_filled("AAA", fill_price=10.0, filled_shares=20)
    with pytest.raises(InvalidPositionTransitionError):
        store.mark_entry_never_triggered("AAA", closed_at=_CLOSE_TS)


def test_attach_protection_children_updates_ids() -> None:
    """After parent fills, executor plants children + writes their IDs onto position."""
    store = PositionStore()
    store.open_position(
        symbol="STPL",
        strategy="gap_and_go",
        shares=50,
        stop_price=9.0,
        scale_out_price=13.0,
        runner_target_price=None,
        parent_order_id=100,
        stop_order_id=0,
        target_order_id=0,
        opened_at=_OPEN_TS,
        status="pending_entry_trigger",
        entry_order_type="STP_LMT",
        entry_trigger_price=10.0,
    )
    store.mark_filled("STPL", fill_price=10.07, filled_shares=50)
    updated = store.attach_protection_children("STPL", stop_order_id=201, target_order_id=202)
    assert updated.stop_order_id == 201
    assert updated.target_order_id == 202


def test_position_entry_order_type_default_is_lmt() -> None:
    """Existing callers that don't pass ``entry_order_type`` default to LMT."""
    store = _store_with_one()
    position = store.get_active("AAA")
    assert position is not None
    assert position.entry_order_type == "LMT"


# ---------- Phase 4k commission accumulators ---------- #


def test_commission_fields_default_to_zero() -> None:
    """A freshly-opened Position starts with zero commission on every leg."""
    store = _store_with_one()
    position = store.get_active("AAA")
    assert position is not None
    assert position.entry_commission == 0.0
    assert position.scale_commission == 0.0
    assert position.exit_commission == 0.0


def test_add_entry_commission_accumulates() -> None:
    """Two reports on the parent leg sum onto ``entry_commission``."""
    store = _store_with_one()
    store.add_entry_commission("AAA", 0.75)
    store.add_entry_commission("AAA", 0.25)
    assert store.get_active("AAA").entry_commission == pytest.approx(1.00)  # type: ignore[union-attr]


def test_add_scale_and_exit_commissions_land_on_separate_buckets() -> None:
    """Scale and exit commissions accumulate independently of each other."""
    store = _store_with_one()
    store.add_scale_commission("AAA", 0.50)
    store.add_exit_commission("AAA", 0.80)
    position = store.get_active("AAA")
    assert position is not None
    assert position.scale_commission == pytest.approx(0.50)
    assert position.exit_commission == pytest.approx(0.80)
    assert position.entry_commission == 0.0


def test_add_commission_rejects_negative_amounts() -> None:
    """Commissions are non-negative costs; a negative report is a bug, not data."""
    store = _store_with_one()
    with pytest.raises(ValueError, match="non-negative"):
        store.add_entry_commission("AAA", -0.10)


def test_add_commission_unknown_symbol_raises() -> None:
    """No position for symbol → UnknownPositionError (same as other mutations)."""
    store = _store_with_one()
    with pytest.raises(UnknownPositionError):
        store.add_entry_commission("ZZZ", 0.50)

"""BookStateTracker: insert/update/delete sequences + print accumulation."""

from __future__ import annotations

from datetime import UTC, datetime

from bot.exit_advisor.market.book_state import BookStateTracker
from bot.exit_advisor.market.l2_events import L2BookUpdate, L2Print


def _ts(s: int) -> datetime:
    return datetime(2026, 5, 5, 13, 30, s, tzinfo=UTC)


def _bid(op: str, price: float, size: int, position: int = 0) -> L2BookUpdate:
    return L2BookUpdate(
        timestamp=_ts(0),
        symbol="X",
        side="bid",
        operation=op,  # type: ignore[arg-type]
        position=position,
        price=price,
        size=size,
    )


def _ask(op: str, price: float, size: int, position: int = 0) -> L2BookUpdate:
    return L2BookUpdate(
        timestamp=_ts(0),
        symbol="X",
        side="ask",
        operation=op,  # type: ignore[arg-type]
        position=position,
        price=price,
        size=size,
    )


def test_insert_creates_level() -> None:
    t = BookStateTracker()
    t.consume(_bid("insert", 10.00, 100))
    state = t.get_state()
    assert len(state.bids) == 1
    assert state.bids[0].price == 10.00
    assert state.bids[0].size == 100


def test_update_replaces_size() -> None:
    t = BookStateTracker()
    t.consume(_bid("insert", 10.00, 100))
    t.consume(_bid("update", 10.00, 250))
    state = t.get_state()
    assert state.bids[0].size == 250


def test_delete_removes_level() -> None:
    t = BookStateTracker()
    t.consume(_bid("insert", 10.00, 100))
    t.consume(_bid("delete", 10.00, 0))
    state = t.get_state()
    assert state.bids == []


def test_size_zero_treated_as_delete() -> None:
    """IBKR sometimes sends update with size=0 as a soft delete; the
    tracker should treat it as removing the level."""
    t = BookStateTracker()
    t.consume(_bid("insert", 10.00, 100))
    t.consume(_bid("update", 10.00, 0))
    assert t.get_state().bids == []


def test_bids_sorted_descending_asks_ascending() -> None:
    t = BookStateTracker()
    t.consume(_bid("insert", 10.00, 100))
    t.consume(_bid("insert", 9.95, 100))
    t.consume(_bid("insert", 10.05, 100))  # better bid arrives later
    t.consume(_ask("insert", 10.10, 100))
    t.consume(_ask("insert", 10.20, 100))
    t.consume(_ask("insert", 10.07, 100))  # better ask arrives later
    state = t.get_state()
    assert [lv.price for lv in state.bids] == [10.05, 10.00, 9.95]
    assert [lv.price for lv in state.asks] == [10.07, 10.10, 10.20]


def test_spread_recomputed_on_query() -> None:
    t = BookStateTracker()
    t.consume(_bid("insert", 10.00, 100))
    t.consume(_ask("insert", 10.05, 100))
    state = t.get_state()
    assert state.spread is not None
    assert abs(state.spread - 0.05) < 1e-9


def test_spread_none_when_one_sided() -> None:
    t = BookStateTracker()
    t.consume(_bid("insert", 10.00, 100))
    state = t.get_state()
    assert state.spread is None


def test_delete_on_nonexistent_level_logs_no_crash(caplog) -> None:  # type: ignore[no-untyped-def]
    """A delete for a price the tracker doesn't have shouldn't crash —
    out-of-order updates are common on busy books."""
    import logging

    t = BookStateTracker()
    with caplog.at_level(logging.WARNING):
        t.consume(_bid("delete", 10.00, 0))
    assert any("non-existent" in m.getMessage() for m in caplog.records)


def test_print_accumulation_credits_correct_side() -> None:
    """Buy aggressor → liquidity taken from ask side; sell → bid side."""
    t = BookStateTracker()
    t.consume(_bid("insert", 10.00, 100))
    t.consume(_ask("insert", 10.05, 100))
    t.consume(L2Print(_ts(1), "X", 10.05, 50, "buy"))
    t.consume(L2Print(_ts(2), "X", 10.00, 30, "sell"))
    state = t.get_state()
    assert state.cumulative_volume_at_level[("ask", 10.05)] == 50
    assert state.cumulative_volume_at_level[("bid", 10.00)] == 30


def test_print_with_unknown_aggressor_does_not_credit() -> None:
    t = BookStateTracker()
    t.consume(_bid("insert", 10.00, 100))
    t.consume(_ask("insert", 10.05, 100))
    t.consume(L2Print(_ts(1), "X", 10.02, 50, "unknown"))
    state = t.get_state()
    assert state.cumulative_volume_at_level == {}


def test_recent_prints_capped_at_max_history() -> None:
    t = BookStateTracker(max_print_history=5)
    for i in range(10):
        t.consume(L2Print(_ts(i), "X", 10.00, 10, "buy"))
    state = t.get_state()
    assert len(state.recent_prints) == 5


def test_delete_resets_cumulative_at_level() -> None:
    """When a level disappears entirely, cumulative tracking resets so
    a future re-insertion of the same price doesn't carry over old volume."""
    t = BookStateTracker()
    t.consume(_bid("insert", 10.00, 100))
    t.consume(L2Print(_ts(1), "X", 10.00, 50, "sell"))
    assert t.get_state().cumulative_volume_at_level[("bid", 10.00)] == 50
    t.consume(_bid("delete", 10.00, 0))
    assert t.get_state().cumulative_volume_at_level == {}

"""L2 canonical event types + aggressor-side derivation rules."""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime

from bot.exit_advisor.market.book_state import BookLevel, BookState
from bot.exit_advisor.market.l2_events import L2BookUpdate, L2Print, derive_aggressor_side


def _ts(s: int) -> datetime:
    return datetime(2026, 5, 5, 13, 30, s, tzinfo=UTC)


def _state(bid: float | None, ask: float | None) -> BookState:
    bids = (
        [BookLevel(price=bid, size=100, last_operation="insert", last_update_timestamp=_ts(0))]
        if bid is not None
        else []
    )
    asks = (
        [BookLevel(price=ask, size=100, last_operation="insert", last_update_timestamp=_ts(0))]
        if ask is not None
        else []
    )
    return BookState(
        bids=bids,
        asks=asks,
        recent_prints=deque(),
        cumulative_volume_at_level={},
        spread=(ask - bid) if (bid is not None and ask is not None) else None,
    )


def test_aggressor_buy_when_print_at_or_above_ask() -> None:
    state = _state(bid=10.00, ask=10.05)
    assert derive_aggressor_side(10.05, state) == "buy"
    assert derive_aggressor_side(10.07, state) == "buy"


def test_aggressor_sell_when_print_at_or_below_bid() -> None:
    state = _state(bid=10.00, ask=10.05)
    assert derive_aggressor_side(10.00, state) == "sell"
    assert derive_aggressor_side(9.99, state) == "sell"


def test_aggressor_unknown_when_print_mid_spread() -> None:
    state = _state(bid=10.00, ask=10.05)
    assert derive_aggressor_side(10.02, state) == "unknown"


def test_aggressor_unknown_when_book_empty() -> None:
    state = _state(bid=None, ask=None)
    assert derive_aggressor_side(10.00, state) == "unknown"


def test_aggressor_unknown_when_book_one_sided() -> None:
    """Pre-market and after-hours often have one-sided books — must
    return unknown rather than guessing."""
    bid_only = _state(bid=10.00, ask=None)
    ask_only = _state(bid=None, ask=10.05)
    assert derive_aggressor_side(10.00, bid_only) == "unknown"
    assert derive_aggressor_side(10.05, ask_only) == "unknown"


def test_aggressor_unknown_when_state_is_none() -> None:
    assert derive_aggressor_side(10.00, None) == "unknown"


def test_l2_book_update_is_immutable() -> None:
    """Frozen dataclass — detector code can't accidentally mutate inputs."""
    from dataclasses import FrozenInstanceError

    import pytest

    evt = L2BookUpdate(
        timestamp=_ts(0),
        symbol="X",
        side="bid",
        operation="insert",
        position=0,
        price=10.00,
        size=100,
    )
    with pytest.raises(FrozenInstanceError):
        evt.size = 200  # type: ignore[misc]


def test_l2_print_carries_aggressor_side() -> None:
    p = L2Print(
        timestamp=_ts(0),
        symbol="X",
        price=10.05,
        size=100,
        aggressor_side="buy",
    )
    assert p.aggressor_side == "buy"

"""Canonical L2 input event types + aggressor-side derivation.

Detectors consume these — not raw ib_async objects. The two-stream
adapter in :mod:`l2_adapter` translates raw API events into these.
The separation keeps detector logic stable across IBKR/ib_async API
shape changes; the adapter is the single integration boundary.

Two event types feed the detectors:
- :class:`L2BookUpdate`: one insert/update/delete on one price level
- :class:`L2Print`: one trade print, with derived ``aggressor_side``

Aggressor-side derivation is documented inline. It is approximate by
nature — IBKR's tick-by-tick stream doesn't carry the maker/taker flag
directly, so detectors must tolerate ``"unknown"`` results.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .book_state import BookState


@dataclass(frozen=True)
class L2BookUpdate:
    """A single update to one side of the order book at one price level.

    ``operation`` semantics mirror the IBKR ``updateMktDepth`` callback:
    ``insert`` adds a level (shifting later positions), ``update`` replaces
    the size at an existing position, ``delete`` removes the level. A
    canonical sequence on a busy book is alternating ``update`` calls on
    position 0 plus the occasional ``insert``/``delete`` at the edges.

    ``market_maker`` is populated when the feed is from ``reqMktDepthExchanges``
    (NASDAQ TotalView, NYSE OpenBook, etc.) which carry per-MM detail. SMART
    aggregated depth leaves it ``None``.
    """

    timestamp: datetime
    symbol: str
    side: Literal["bid", "ask"]
    operation: Literal["insert", "update", "delete"]
    position: int
    price: float
    size: int
    market_maker: str | None = None


@dataclass(frozen=True)
class L2Print:
    """One trade print (time-and-sales).

    ``aggressor_side`` is derived (not transmitted by IBKR). Use
    :func:`derive_aggressor_side` to compute it against the book state
    at print time.
    """

    timestamp: datetime
    symbol: str
    price: float
    size: int
    aggressor_side: Literal["buy", "sell", "unknown"]


def derive_aggressor_side(
    print_price: float, book_state: BookState | None
) -> Literal["buy", "sell", "unknown"]:
    """Standard quote-rule derivation of trade aggressor side.

    - Print at-or-above the current best ask: aggressor was a buyer
      (someone hit the offer).
    - Print at-or-below the current best bid: aggressor was a seller
      (someone hit the bid).
    - Print between bid and ask, or book state unknown / one-sided:
      ``"unknown"``. Detectors must tolerate this and typically treat
      unknown prints as neutral.

    Approximations involved:
    - Clock drift between depth feed and print feed can flip the
      classification on prints near the spread.
    - Mid-spread prints (e.g. RegNMS price improvement) are
      legitimately ambiguous — the derivation says "unknown" rather
      than guessing.
    - Pre-market / post-market with thin liquidity often produces
      one-sided book states; aggressor is then "unknown" too.

    The derivation is intentionally simple. A more sophisticated
    classifier (Lee-Ready algorithm + tick test fallback) is a possible
    later upgrade if the simple rule produces too many "unknown" results
    in practice.
    """
    if book_state is None:
        return "unknown"
    bids = book_state.bids
    asks = book_state.asks
    if not bids or not asks:
        return "unknown"
    best_bid = bids[0].price
    best_ask = asks[0].price
    if print_price >= best_ask:
        return "buy"
    if print_price <= best_bid:
        return "sell"
    return "unknown"

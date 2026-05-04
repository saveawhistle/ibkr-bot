"""Two-stream IBKR adapter — translates ``reqMktDepth`` and
``reqTickByTickData('AllLast')`` into the canonical L2 event stream
(:class:`L2BookUpdate` + :class:`L2Print`) that detectors consume.

The adapter is the single integration boundary. If ib_async's API
shape changes (or Monday's probe surfaces a quirk the docs missed),
the adjustment lives here, NOT in detector logic. That separation is
what lets us write detectors against synthetic fixtures with
confidence today and only revisit the boundary later.

Aggressor-side derivation runs inside the adapter. Each incoming print
is classified against the adapter's own minimal book mirror at the
print's arrival time. The adapter's mirror is kept separate from
:class:`book_state.BookStateTracker` deliberately — the tracker is for
detector consumption (richer state, configurable history), and the
mirror is a single-purpose helper for the buy/sell classification.
Conflating them would couple the integration boundary to detector
internals.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from .l2_events import L2BookUpdate, L2Print, derive_aggressor_side

if TYPE_CHECKING:
    from .book_state import BookState, BookStateTracker

log = logging.getLogger(__name__)

L2Event = L2BookUpdate | L2Print
L2Consumer = Callable[[L2Event], None]


# ----------------------------------------------------------------------
# Raw → canonical translation
# ----------------------------------------------------------------------


# IBKR's depth ``operation`` field is documented as int 0/1/2 = insert/update/delete
# and ``side`` as 0=ask, 1=bid. ib_async typically forwards these as-is. Map both
# at the boundary so detectors only ever see the named literals.
_OPERATION_MAP: dict[int, Literal["insert", "update", "delete"]] = {
    0: "insert",
    1: "update",
    2: "delete",
}
_SIDE_MAP: dict[int, Literal["bid", "ask"]] = {
    0: "ask",
    1: "bid",
}


def translate_depth_event(raw: Any, symbol: str) -> L2BookUpdate | None:
    """Convert one raw ib_async depth row into a canonical
    :class:`L2BookUpdate`. Returns ``None`` if the row's shape is too
    far off to interpret — the adapter logs and continues rather than
    crashing on a single malformed row.

    Probe-pending: if Monday's probe shows ib_async exposes the side
    or operation as named strings or different field names, only this
    function changes.
    """
    try:
        operation_raw = raw.operation
        side_raw = raw.side
        position = int(raw.position)
        price = float(raw.price)
        size = int(raw.size)
    except (AttributeError, TypeError, ValueError) as exc:
        log.warning("could not translate depth event %r: %s", raw, exc)
        return None

    operation: Literal["insert", "update", "delete"]
    if isinstance(operation_raw, int):
        if operation_raw not in _OPERATION_MAP:
            log.warning("unknown depth operation int %s; ignoring", operation_raw)
            return None
        operation = _OPERATION_MAP[operation_raw]
    elif isinstance(operation_raw, str) and operation_raw in {"insert", "update", "delete"}:
        operation = operation_raw  # type: ignore[assignment]
    else:
        log.warning("unknown depth operation %r; ignoring", operation_raw)
        return None

    side: Literal["bid", "ask"]
    if isinstance(side_raw, int):
        if side_raw not in _SIDE_MAP:
            log.warning("unknown depth side int %s; ignoring", side_raw)
            return None
        side = _SIDE_MAP[side_raw]
    elif isinstance(side_raw, str) and side_raw in {"bid", "ask"}:
        side = side_raw  # type: ignore[assignment]
    else:
        log.warning("unknown depth side %r; ignoring", side_raw)
        return None

    timestamp_raw = getattr(raw, "time", None) or getattr(raw, "timestamp", None)
    timestamp = _coerce_timestamp(timestamp_raw)
    market_maker = getattr(raw, "marketMaker", None) or getattr(raw, "market_maker", None)

    return L2BookUpdate(
        timestamp=timestamp,
        symbol=symbol,
        side=side,
        operation=operation,
        position=position,
        price=price,
        size=size,
        market_maker=market_maker,
    )


def translate_print_event(
    raw: Any, symbol: str, book_state: BookState | None
) -> L2Print | None:
    """Convert one raw ``TickByTickAllLast`` (or equivalent) event into
    a canonical :class:`L2Print`. ``aggressor_side`` is derived against
    the supplied ``book_state``; if state is ``None`` or one-sided,
    aggressor is ``"unknown"``.
    """
    try:
        price = float(raw.price)
        size = int(raw.size)
    except (AttributeError, TypeError, ValueError) as exc:
        log.warning("could not translate print event %r: %s", raw, exc)
        return None
    if price <= 0 or size <= 0:
        return None
    timestamp = _coerce_timestamp(
        getattr(raw, "time", None) or getattr(raw, "timestamp", None)
    )
    aggressor = derive_aggressor_side(price, book_state)
    return L2Print(
        timestamp=timestamp,
        symbol=symbol,
        price=price,
        size=size,
        aggressor_side=aggressor,
    )


def _coerce_timestamp(raw: Any) -> datetime:
    """ib_async may provide either a ``datetime`` or a Unix epoch int
    in the time field; coerce to a tz-aware UTC ``datetime``. Falls
    back to ``datetime.now(UTC)`` if nothing usable was supplied — that
    keeps the canonical event still usable for detectors that don't
    care about the exact arrival time."""
    if isinstance(raw, datetime):
        if raw.tzinfo is None:
            return raw.replace(tzinfo=UTC)
        return raw.astimezone(UTC)
    if isinstance(raw, int | float):
        return datetime.fromtimestamp(raw, tz=UTC)
    return datetime.now(UTC)


# ----------------------------------------------------------------------
# Two-stream adapter
# ----------------------------------------------------------------------


@dataclass
class L2StreamAdapter:
    """Subscribes to ``reqMktDepth`` + ``reqTickByTickData('AllLast')``
    for one symbol; merges and translates events; emits canonical
    :class:`L2BookUpdate` / :class:`L2Print` events to ``consumer``.

    The adapter is structured so tests pass mocked ib_async tickers
    directly — :meth:`feed_depth_update` and :meth:`feed_print_update`
    are the test entry points; :meth:`start` / :meth:`stop` are the
    runtime entry points.
    """

    ibkr_client: Any
    """Project's :class:`bot.brokerage.ibkr_client.IBKRClient`. Typed as ``Any``
    here so the adapter is testable without importing the production
    client; the runtime path passes the real one."""

    symbol: str
    consumer: L2Consumer
    num_depth_rows: int = 10

    _adapter_book: BookStateTracker | None = field(default=None, init=False)
    _depth_ticker: Any = field(default=None, init=False)
    _prints_ticker: Any = field(default=None, init=False)
    _started: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        # Late import to avoid a top-level circular import (book_state
        # imports l2_events, which is fine; this module imports
        # book_state which imports l2_events — order matters).
        from .book_state import BookStateTracker

        self._adapter_book = BookStateTracker(max_print_history=20)

    async def start(self) -> None:
        """Subscribe to both streams. Must be called from an async
        context with a connected IBKR client."""
        if self._started:
            return
        contract = await self.ibkr_client.qualify_stock(self.symbol)
        self._depth_ticker = self.ibkr_client.ib.reqMktDepth(
            contract, numRows=self.num_depth_rows
        )
        self._prints_ticker = self.ibkr_client.ib.reqTickByTickData(
            contract, "AllLast", numberOfTicks=0, ignoreSize=False
        )
        self._depth_ticker.updateEvent += self._on_depth_update
        self._prints_ticker.updateEvent += self._on_prints_update
        self._started = True
        log.info("L2StreamAdapter started for %s", self.symbol)

    async def stop(self) -> None:
        if not self._started:
            return
        contract = await self.ibkr_client.qualify_stock(self.symbol)
        try:
            self.ibkr_client.ib.cancelMktDepth(contract)
        except Exception as exc:  # noqa: BLE001
            log.warning("cancelMktDepth raised: %s", exc)
        try:
            self.ibkr_client.ib.cancelTickByTickData(contract, "AllLast")
        except Exception as exc:  # noqa: BLE001
            log.warning("cancelTickByTickData raised: %s", exc)
        self._started = False
        log.info("L2StreamAdapter stopped for %s", self.symbol)

    # --- runtime ib_async event handlers ---

    def _on_depth_update(self, ticker: Any) -> None:
        for raw in getattr(ticker, "domTicks", []):
            self.feed_depth_update(raw)

    def _on_prints_update(self, ticker: Any) -> None:
        for raw in getattr(ticker, "tickByTicks", []):
            self.feed_print_update(raw)

    # --- test-friendly entry points ---

    def feed_depth_update(self, raw: Any) -> None:
        """Translate one raw depth event and emit. Public so tests can
        drive the adapter without spinning up ib_async."""
        evt = translate_depth_event(raw, self.symbol)
        if evt is None:
            return
        assert self._adapter_book is not None
        self._adapter_book.consume(evt)
        self.consumer(evt)

    def feed_print_update(self, raw: Any) -> None:
        """Translate one raw print event (deriving aggressor side
        against the adapter's own book mirror at this moment) and emit."""
        assert self._adapter_book is not None
        state = self._adapter_book.get_state()
        evt = translate_print_event(raw, self.symbol, state)
        if evt is None:
            return
        # Update mirror AFTER classifying — the print didn't move the
        # book, but our aggregate cumulative-volume tracking should
        # see it. (This isn't strictly necessary for the adapter's
        # buy/sell classifier, but the mirror is harmless to maintain.)
        self._adapter_book.consume(evt)
        self.consumer(evt)

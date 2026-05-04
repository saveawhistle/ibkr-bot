"""L2 stream ingestion + book-state tracking.

Owns the canonical L2 event types, the ib_async → canonical adapter,
and the book-state machine that L2 detectors read from. Anything that
needs "what does the order book look like right now?" goes through
:class:`BookStateTracker`.
"""

from bot.exit_advisor.market.book_state import BookLevel, BookState, BookStateTracker
from bot.exit_advisor.market.l2_adapter import (
    L2StreamAdapter,
    translate_depth_event,
    translate_print_event,
)
from bot.exit_advisor.market.l2_events import (
    L2BookUpdate,
    L2Print,
    derive_aggressor_side,
)

__all__ = [
    "BookLevel",
    "BookState",
    "BookStateTracker",
    "L2BookUpdate",
    "L2Print",
    "L2StreamAdapter",
    "derive_aggressor_side",
    "translate_depth_event",
    "translate_print_event",
]

"""Core domain types: hook contract, event taxonomy, time helpers.

Anything that the rest of the exit-advisor subpackages depend on
without depending on each other lives here. Imports go *into* core,
never out — keeping it as a leaf module makes circular-import risk
zero.
"""

from bot.exit_advisor.core.events import (
    AbsorptionDetected,
    BarFinalizedEvent,
    BarShapeDetected,
    BidPulled,
    ConsecutiveBars,
    DrawdownFromPeak,
    Event,
    GateChainResult,
    GateRejection,
    ImbalanceEvent,
    LargePrint,
    LevelDataUnavailable,
    LevelReclaimed,
    LevelTouched,
    MaxFavorableExcursionUpdate,
    MovingAverageCross,
    OfferPulled,
    OrderRejectionEvent,
    PartialFillEvent,
    PositionProtected,
    PrintCluster,
    ReplayTerminalTick,
    RMultipleReached,
    RVolDataUnavailable,
    RVolMilestone,
    SpreadEvent,
    TimeInTradeMilestone,
    TimeOfDayMilestone,
    VolumeDryUp,
    VolumeSpike,
    WickEvent,
)
from bot.exit_advisor.core.timeutil import (
    NY,
    RTH_CLOSE,
    RTH_OPEN,
    rth_close_utc,
    rth_open_for,
    rth_open_utc,
)
from bot.exit_advisor.core.types import (
    AdvisorResponse,
    ExitAction,
    ExitAdvisorHook,
    ExitRecommendation,
    PositionLike,
)

__all__ = [
    # types
    "AdvisorResponse",
    "ExitAction",
    "ExitAdvisorHook",
    "ExitRecommendation",
    "PositionLike",
    # event base + production hook events
    "Event",
    "BarFinalizedEvent",
    # event taxonomy — time + pnl + order_state
    "TimeOfDayMilestone",
    "TimeInTradeMilestone",
    "RMultipleReached",
    "DrawdownFromPeak",
    "MaxFavorableExcursionUpdate",
    "PositionProtected",
    "PartialFillEvent",
    "OrderRejectionEvent",
    # event taxonomy — price levels + moving averages
    "LevelTouched",
    "LevelReclaimed",
    "LevelDataUnavailable",
    "MovingAverageCross",
    # event taxonomy — volume
    "VolumeSpike",
    "VolumeDryUp",
    "RVolMilestone",
    "RVolDataUnavailable",
    # event taxonomy — bar shape
    "BarShapeDetected",
    "WickEvent",
    "ConsecutiveBars",
    # event taxonomy — L2
    "BidPulled",
    "OfferPulled",
    "AbsorptionDetected",
    "SpreadEvent",
    "ImbalanceEvent",
    "PrintCluster",
    "LargePrint",
    # event taxonomy — gates + harness
    "GateRejection",
    "GateChainResult",
    "ReplayTerminalTick",
    # timeutil
    "NY",
    "RTH_CLOSE",
    "RTH_OPEN",
    "rth_close_utc",
    "rth_open_for",
    "rth_open_utc",
]

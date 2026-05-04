"""Exit advisor package: hook surface, detectors, decision framework,
replay harness, and analysis tooling for advisory exit recommendations.

Layout (by concern):

* :mod:`bot.exit_advisor.core` — shared types + event taxonomy + time helpers
* :mod:`bot.exit_advisor.hook` — production hook surface (registry + applier)
* :mod:`bot.exit_advisor.detectors` — bar / volume / level / L2 detectors
* :mod:`bot.exit_advisor.market` — L2 stream ingestion + book state
* :mod:`bot.exit_advisor.decision` — gate framework + exit policies
* :mod:`bot.exit_advisor.replay` — offline harness + historical-bar cache
* :mod:`bot.exit_advisor.analysis` — multi-trade aggregation + classification

The most-commonly-imported public symbols are re-exported below so
external callers can use ergonomic ``from bot.exit_advisor import X``
syntax. Internal code (anything inside this package) imports from the
explicit subpackage paths to avoid circular-import risk.
"""

from __future__ import annotations

# Event taxonomy lives in core.events; re-export the production-hook
# event surface alongside the bare ``Event`` base. Detector-specific
# event classes (RVolMilestone, AbsorptionDetected, etc.) are still
# importable via ``bot.exit_advisor.core``.
from bot.exit_advisor.core.events import BarFinalizedEvent, Event
from bot.exit_advisor.core.types import (
    AdvisorResponse,
    ExitAdvisorHook,
    ExitRecommendation,
    PositionLike,
)
from bot.exit_advisor.hook.apply import RecommendationApplier
from bot.exit_advisor.hook.registry import (
    notify_event,
    notify_position_closed,
    notify_position_protected,
    register_exit_advisor,
    registered_advisor,
    unregister_exit_advisor,
)

__all__ = [
    "AdvisorResponse",
    "BarFinalizedEvent",
    "Event",
    "ExitAdvisorHook",
    "ExitRecommendation",
    "PositionLike",
    "RecommendationApplier",
    "notify_event",
    "notify_position_closed",
    "notify_position_protected",
    "register_exit_advisor",
    "registered_advisor",
    "unregister_exit_advisor",
]

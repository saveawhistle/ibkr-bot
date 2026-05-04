"""Production hook surface — registry + invocation wrappers + applier.

External callers shouldn't need to know whether a function lives in
``registry.py`` or ``apply.py``. They import from this subpackage and
the appropriate module is found via the re-exports below.

The Phase 11 hook contract is defined in
:mod:`bot.exit_advisor.core.types`; this subpackage just *uses* it.
"""

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
    "RecommendationApplier",
    "notify_event",
    "notify_position_closed",
    "notify_position_protected",
    "register_exit_advisor",
    "registered_advisor",
    "unregister_exit_advisor",
]

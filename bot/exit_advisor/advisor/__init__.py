"""Live LLM-powered exit advisor.

Implements :class:`bot.exit_advisor.core.types.ExitAdvisorHook` against the
Phase 11 hook surface. Runs an Anthropic-Claude advisor in production
alongside three mechanical baseline policies (:mod:`shadow_baselines`)
whose recommendations are logged but never acted on.

The advisor is constructed and registered by
:func:`bot.exit_advisor.advisor.bootstrap.bootstrap_advisor`, which the
CLI calls once per session at startup. The bootstrap function is the
sub-package's only public entry point — every other class is internal
plumbing exposed for tests.
"""

from __future__ import annotations

from bot.exit_advisor.advisor.agent import ExitAdvisor, _LiveTradeState
from bot.exit_advisor.advisor.bootstrap import bootstrap_advisor
from bot.exit_advisor.advisor.buffer import BufferDecision, EventBuffer
from bot.exit_advisor.advisor.cost_tracker import CostTracker
from bot.exit_advisor.advisor.llm_client import (
    SONNET_INPUT_COST_PER_TOKEN,
    SONNET_OUTPUT_COST_PER_TOKEN,
    AnthropicLLMClient,
    LLMCallResult,
)
from bot.exit_advisor.advisor.prompts import (
    EXIT_ADVISOR_SYSTEM_PROMPT,
    EXIT_RECOMMENDATION_TOOL_SCHEMA,
)
from bot.exit_advisor.advisor.shadow_baselines import ShadowBaselines

__all__ = [
    "EXIT_ADVISOR_SYSTEM_PROMPT",
    "EXIT_RECOMMENDATION_TOOL_SCHEMA",
    "SONNET_INPUT_COST_PER_TOKEN",
    "SONNET_OUTPUT_COST_PER_TOKEN",
    "AnthropicLLMClient",
    "BufferDecision",
    "CostTracker",
    "EventBuffer",
    "ExitAdvisor",
    "LLMCallResult",
    "ShadowBaselines",
    "_LiveTradeState",
    "bootstrap_advisor",
]

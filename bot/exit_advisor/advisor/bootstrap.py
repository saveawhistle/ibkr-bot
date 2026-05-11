"""Single entry point that constructs + registers the live exit advisor.

Called by the bot's CLI startup path (one line in
:mod:`bot.cli`). The function is safe to call regardless of config
state: returns ``None`` when the advisor is disabled, when
``ANTHROPIC_API_KEY`` is not set, or when construction raises. The
bot continues normally in any of those branches.

Construction failures are logged at ERROR with full traceback so the
operator can see why the advisor didn't start. The bot is never
crashed by an advisor bootstrap bug.
"""

from __future__ import annotations

import os

import structlog
from dotenv import load_dotenv

from bot.config import Settings
from bot.exit_advisor.advisor.agent import ExitAdvisor
from bot.exit_advisor.advisor.buffer import EventBuffer
from bot.exit_advisor.advisor.cost_tracker import CostTracker
from bot.exit_advisor.advisor.llm_client import AnthropicLLMClient
from bot.exit_advisor.advisor.shadow_baselines import ShadowBaselines
from bot.exit_advisor.hook.registry import register_exit_advisor

_log = structlog.get_logger("bot.exit_advisor.advisor.bootstrap")

ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"


def bootstrap_advisor(config: Settings) -> ExitAdvisor | None:
    """Construct the live LLM advisor and register it against the Phase 11 hook.

    Returns the advisor instance on success; returns ``None`` when:

    * ``config.exit_advisor.enabled`` is False (the production-main default),
    * the ``ANTHROPIC_API_KEY`` environment variable is not set, or
    * construction raises (e.g. the Anthropic SDK is misconfigured).

    The bot tolerates ``None`` and runs without the advisor in any of
    those cases. ``hook_acts`` is honoured at the hook layer (the
    registry checks the flag before forwarding actionable
    recommendations to TradeManager); we only need to register the
    advisor here.
    """
    cfg = config.exit_advisor
    if not cfg.enabled:
        _log.info("advisor.bootstrap_skipped", reason="exit_advisor.enabled=false")
        return None

    # Promote .env keys into os.environ so operators can keep ANTHROPIC_API_KEY
    # alongside the rest of their secrets. ``override=False`` means a real OS
    # env var still wins over the .env file, matching pydantic-settings'
    # precedence elsewhere in the bot.
    load_dotenv(override=False)
    api_key = os.environ.get(ANTHROPIC_API_KEY_ENV, "").strip()
    if not api_key:
        _log.error(
            "advisor.bootstrap_skipped",
            reason="missing_api_key",
            env_var=ANTHROPIC_API_KEY_ENV,
            note="exit_advisor.enabled=true but ANTHROPIC_API_KEY is not set",
        )
        return None

    try:
        llm_client = AnthropicLLMClient(
            api_key=api_key,
            model=cfg.llm_model,
            max_tokens=cfg.llm_max_tokens,
            timeout_seconds=cfg.llm_timeout_seconds,
        )
        cost_tracker = CostTracker(
            soft_cap_usd=cfg.cost_soft_cap_usd,
            hard_cap_usd=cfg.cost_hard_cap_usd,
        )
        shadow_baselines = ShadowBaselines()

        def _make_buffer() -> EventBuffer:
            return EventBuffer(
                time_floor_seconds=cfg.event_buffer_time_floor_seconds,
                hard_floor_seconds=cfg.event_buffer_hard_floor_seconds,
            )

        advisor = ExitAdvisor(
            llm_client=llm_client,
            cost_tracker=cost_tracker,
            event_buffer_factory=_make_buffer,
            shadow_baselines=shadow_baselines,
            hook_acts=cfg.hook_acts,
            self_disable_failure_rate=cfg.self_disable_failure_rate,
            self_disable_min_calls=cfg.self_disable_min_calls,
            min_hold_minutes_for_full_exit=cfg.min_hold_minutes_for_full_exit,
            min_r_for_full_exit=cfg.min_r_for_full_exit,
        )
    except Exception:
        _log.exception("advisor.bootstrap_failed", model=cfg.llm_model)
        return None

    register_exit_advisor(advisor)
    _log.info(
        "advisor.bootstrap_succeeded",
        model=cfg.llm_model,
        hook_acts=cfg.hook_acts,
        cost_soft_cap_usd=cfg.cost_soft_cap_usd,
        cost_hard_cap_usd=cfg.cost_hard_cap_usd,
    )
    return advisor

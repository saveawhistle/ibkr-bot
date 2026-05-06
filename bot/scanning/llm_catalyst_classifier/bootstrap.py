"""Single entry point that constructs the Phase 12 LLM catalyst classifier.

Called by the bot's startup path when
``catalyst_classifier.llm.enabled`` is True. The function returns
``None`` when:

* ``catalyst_classifier.llm.enabled`` is False (the keyword path stays in charge),
* ``ANTHROPIC_API_KEY`` is not set in the environment,
* construction raises (e.g. the Anthropic SDK is misconfigured).

The scanner tolerates ``None`` and falls back to the keyword classifier
with a warning log; the bot never crashes on a bootstrap bug.
"""

from __future__ import annotations

import os

import structlog
from dotenv import load_dotenv

from bot.config import Settings
from bot.scanning.llm_catalyst_classifier.cache import ClassificationCache
from bot.scanning.llm_catalyst_classifier.classifier import LLMCatalystClassifier
from bot.scanning.llm_catalyst_classifier.cost_tracker import CatalystCostTracker
from bot.scanning.llm_catalyst_classifier.llm_client import AnthropicCatalystClient

_log = structlog.get_logger("bot.scanning.llm_catalyst_classifier.bootstrap")

ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"


def bootstrap_catalyst_classifier(config: Settings) -> LLMCatalystClassifier | None:
    """Construct the Phase 12 LLM catalyst classifier.

    Returns the classifier on success; returns ``None`` on the documented
    skip / failure paths. The scanner short-circuits to the keyword
    classifier when ``None`` is returned.
    """
    cfg = config.catalyst_classifier
    if not cfg.llm.enabled:
        _log.info("catalyst_classifier.bootstrap_skipped", reason="llm.enabled=false")
        return None

    # Same .env loading semantics as the exit advisor — operators keep
    # secrets in .env, bootstrap promotes them into os.environ here.
    load_dotenv(override=False)
    api_key = os.environ.get(ANTHROPIC_API_KEY_ENV, "").strip()
    if not api_key:
        _log.error(
            "catalyst_classifier.bootstrap_skipped",
            reason="missing_api_key",
            env_var=ANTHROPIC_API_KEY_ENV,
        )
        return None

    try:
        llm_client = AnthropicCatalystClient(
            api_key=api_key,
            model=cfg.llm.model,
            max_tokens=AnthropicCatalystClient.DEFAULT_MAX_TOKENS,
            timeout_seconds=cfg.llm.timeout_seconds,
        )
        cache = ClassificationCache(
            ttl_seconds=cfg.llm.cache_ttl_seconds,
            capacity=cfg.llm.cache_capacity,
        )
        cost_tracker = CatalystCostTracker(
            soft_cap_usd=cfg.llm.cost_soft_cap_usd_per_day,
            hard_cap_usd=cfg.llm.cost_hard_cap_usd_per_day,
        )
        classifier = LLMCatalystClassifier(
            llm_client=llm_client,
            cache=cache,
            cost_tracker=cost_tracker,
            max_input_chars=cfg.llm.max_input_tokens * 4,  # ~4 chars/token
            self_disable_failure_rate=cfg.llm.self_disable_failure_rate,
            self_disable_min_calls=cfg.llm.self_disable_min_calls,
        )
    except Exception:
        _log.exception(
            "catalyst_classifier.bootstrap_failed",
            model=cfg.llm.model,
        )
        return None

    _log.info(
        "catalyst_classifier.bootstrap_succeeded",
        model=cfg.llm.model,
        timeout_seconds=cfg.llm.timeout_seconds,
        cache_ttl_seconds=cfg.llm.cache_ttl_seconds,
        cache_capacity=cfg.llm.cache_capacity,
        cost_soft_cap_usd_per_day=cfg.llm.cost_soft_cap_usd_per_day,
        cost_hard_cap_usd_per_day=cfg.llm.cost_hard_cap_usd_per_day,
    )
    return classifier


__all__ = [
    "ANTHROPIC_API_KEY_ENV",
    "bootstrap_catalyst_classifier",
]

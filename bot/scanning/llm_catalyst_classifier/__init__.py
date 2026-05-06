"""Phase 12 LLM-driven catalyst classifier.

Replaces the keyword classifier in :mod:`bot.scanning.catalyst` when
``catalyst_classifier.llm.enabled`` is True. The keyword classifier
remains in the repo, gated behind the same config block, so a rollback
is a config flip and not a code revert.

Public API exposed at the package boundary:

* :class:`LLMCatalystClassifier` — the main classifier (one instance
  per session).
* :class:`ClassificationResult` — the per-ticker disposition the scanner
  attaches to its ``ScanHit`` rows.
* :func:`bootstrap_catalyst_classifier` — constructs the classifier
  from ``Settings``; returns ``None`` when the LLM path is disabled
  or the bootstrap fails.
* :class:`CatalystClassification` — the structured tool-use result
  inside a successful classification.
"""

from __future__ import annotations

from bot.scanning.llm_catalyst_classifier.bootstrap import (
    ANTHROPIC_API_KEY_ENV,
    bootstrap_catalyst_classifier,
)
from bot.scanning.llm_catalyst_classifier.cache import (
    ClassificationCache,
    hash_headlines,
)
from bot.scanning.llm_catalyst_classifier.classifier import (
    ClassificationResult,
    LLMCatalystClassifier,
)
from bot.scanning.llm_catalyst_classifier.cost_tracker import CatalystCostTracker
from bot.scanning.llm_catalyst_classifier.llm_client import (
    AnthropicCatalystClient,
    CatalystClassification,
    CategoryLiteral,
    LLMCallResult,
)
from bot.scanning.llm_catalyst_classifier.prompts import (
    CATALYST_CLASSIFIER_SYSTEM_PROMPT,
    CLASSIFY_CATALYST_TOOL,
    CLASSIFY_CATALYST_TOOL_NAME,
    render_user_message,
)

__all__ = [
    "ANTHROPIC_API_KEY_ENV",
    "CATALYST_CLASSIFIER_SYSTEM_PROMPT",
    "CLASSIFY_CATALYST_TOOL",
    "CLASSIFY_CATALYST_TOOL_NAME",
    "AnthropicCatalystClient",
    "CatalystClassification",
    "CatalystCostTracker",
    "CategoryLiteral",
    "ClassificationCache",
    "ClassificationResult",
    "LLMCallResult",
    "LLMCatalystClassifier",
    "bootstrap_catalyst_classifier",
    "hash_headlines",
    "render_user_message",
]

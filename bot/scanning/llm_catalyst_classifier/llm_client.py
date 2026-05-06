"""Anthropic API client wrapper for the Phase 12 catalyst classifier.

Mirrors :class:`bot.exit_advisor.advisor.llm_client.AnthropicLLMClient` —
synchronous request/response per call, no SDK auto-retry (max_retries=0
per the 2026-05-05 exit-advisor finding), structured failure result
rather than exception propagation. Asynchronous wrapper at the
classifier level dispatches multiple per-ticker calls concurrently
via ``asyncio.gather``; this client is the single-call primitive.

Pricing constants are per Sonnet 4.6 ($3 input / $15 output per 1M
tokens) — same as the exit advisor.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import anthropic

# Sonnet 4.6 pricing — match bot.exit_advisor.advisor.llm_client. Update both
# constants together if Anthropic changes pricing or the model swaps.
SONNET_INPUT_COST_PER_TOKEN = 3.0 / 1_000_000
SONNET_OUTPUT_COST_PER_TOKEN = 15.0 / 1_000_000


CategoryLiteral = Literal[
    "earnings_beat",
    "clinical_data",
    "fda_approval",
    "m_a_definitive",
    "contract_win",
    "regulatory_milestone",
    "fundamental_inflection",
    "sympathy_only",
    "stale_news",
    "announcement_only",
    "routine_filings",
    "pump_indicators",
    "non_qualifying_other",
]

_VALID_CATEGORIES = frozenset(
    {
        "earnings_beat",
        "clinical_data",
        "fda_approval",
        "m_a_definitive",
        "contract_win",
        "regulatory_milestone",
        "fundamental_inflection",
        "sympathy_only",
        "stale_news",
        "announcement_only",
        "routine_filings",
        "pump_indicators",
        "non_qualifying_other",
    }
)
_VALID_CONCERNS = frozenset(
    {
        "dilutive_financing",
        "chronic_dilution_pattern",
        "non_binding_agreement",
        "post_close_news",
    }
)


@dataclass(frozen=True)
class CatalystClassification:
    """Structured tool-use result. Frozen so the cache can hash it safely."""

    qualifies: bool
    category: CategoryLiteral
    confidence: float
    reasoning: str
    concerns: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class LLMCallResult:
    """One classification call's outcome.

    Either ``success=True`` with a populated ``classification``, or
    ``success=False`` with a populated ``failure_reason``. ``cost_usd``
    is always populated; failures cost zero (no usage reported).

    ``transient=True`` marks failures the operator's normal cadence will
    naturally recover from -- the canonical case is Anthropic returning
    HTTP 529 ``OverloadedError`` (a capacity blip on their side, not a
    bug on ours). The classifier excludes transient failures from the
    self-disable failure-rate counter so a 5-minute Anthropic hiccup
    can't take the catalyst pillar offline for the rest of the session
    when the next 5-minute rescan would have recovered cleanly.
    """

    success: bool
    classification: CatalystClassification | None
    cost_usd: float
    duration_seconds: float
    failure_reason: str | None = None
    raw_response_for_forensics: dict[str, Any] | None = field(default=None)
    transient: bool = False


class AnthropicCatalystClient:
    """Synchronous Anthropic API client for catalyst classification.

    Synchronous because the classifier dispatches concurrently at a
    higher level via ``asyncio.gather`` over per-ticker tasks; each
    task wraps a single call to this client. Adding asyncio inside
    this class would duplicate concurrency without simplifying.

    The client holds zero per-call state. Construct once, share across
    all ticker classifications.
    """

    DEFAULT_MODEL = "claude-sonnet-4-6"
    DEFAULT_MAX_TOKENS = 1024
    DEFAULT_TIMEOUT_SECONDS = 12.0
    # max_retries=0: same rationale as bot.exit_advisor.advisor.llm_client.
    # SDK default of 2 silently triples the configured timeout in worst case;
    # the 2026-05-05 ENVB session demonstrated the cost. Application-level
    # retry happens via the next scanner pass (rescan interval ~5 min by
    # default), which is the natural retry cadence for catalyst evaluation.
    DEFAULT_MAX_RETRIES = 0

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        client: anthropic.Anthropic | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        if not api_key:
            raise ValueError("AnthropicCatalystClient: api_key must be a non-empty string")
        if max_tokens <= 0:
            raise ValueError(f"AnthropicCatalystClient: max_tokens must be > 0 (got {max_tokens})")
        if timeout_seconds <= 0.0:
            raise ValueError(
                f"AnthropicCatalystClient: timeout_seconds must be > 0 (got {timeout_seconds})"
            )
        if max_retries < 0:
            raise ValueError(
                f"AnthropicCatalystClient: max_retries must be >= 0 (got {max_retries})"
            )
        self._model = model
        self._max_tokens = max_tokens
        self._timeout_seconds = timeout_seconds
        self._client = (
            client
            if client is not None
            else anthropic.Anthropic(api_key=api_key, max_retries=max_retries)
        )

    @property
    def model(self) -> str:
        return self._model

    def call(
        self,
        system_prompt: str,
        user_message: str,
        tool_schema: dict[str, Any],
    ) -> LLMCallResult:
        """Make one classification call. Always returns ``LLMCallResult``; never raises."""
        started = time.monotonic()
        try:
            # The SDK's typed-dict signature is restrictive; we hand it raw
            # dicts since the schema and tool_choice shape are validated at
            # the API boundary, not by mypy. Same pattern as the exit
            # advisor's ``AnthropicLLMClient``.
            response = self._client.messages.create(  # type: ignore[call-overload]
                model=self._model,
                max_tokens=self._max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                tools=[tool_schema],
                tool_choice={"type": "tool", "name": tool_schema["name"]},
                timeout=self._timeout_seconds,
            )
        except anthropic.APITimeoutError:
            return LLMCallResult(
                success=False,
                classification=None,
                cost_usd=0.0,
                duration_seconds=time.monotonic() - started,
                failure_reason="llm_timeout",
            )
        except anthropic.APIStatusError as exc:
            # Anthropic capacity blip (HTTP 529 OverloadedError). The next
            # 5-minute scanner rescan typically recovers cleanly, so flag
            # it transient -- the classifier excludes transient failures
            # from the self-disable rate so a short outage on Anthropic's
            # side doesn't take the catalyst pillar offline for the rest
            # of the session. ``OverloadedError`` isn't exported at the
            # SDK top level (anthropic 0.97), so we match by status code.
            transient = getattr(exc, "status_code", None) == 529
            reason_prefix = "overloaded" if transient else "api_error"
            return LLMCallResult(
                success=False,
                classification=None,
                cost_usd=0.0,
                duration_seconds=time.monotonic() - started,
                failure_reason=f"{reason_prefix}: {type(exc).__name__}: {exc}",
                transient=transient,
            )
        except anthropic.APIError as exc:
            return LLMCallResult(
                success=False,
                classification=None,
                cost_usd=0.0,
                duration_seconds=time.monotonic() - started,
                failure_reason=f"api_error: {type(exc).__name__}: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 - never let an unexpected exc escape
            return LLMCallResult(
                success=False,
                classification=None,
                cost_usd=0.0,
                duration_seconds=time.monotonic() - started,
                failure_reason=f"unexpected_error: {type(exc).__name__}: {exc}",
            )

        duration = time.monotonic() - started
        cost = _compute_cost(response)
        raw = _response_to_dict(response)

        tool_block = _find_tool_use_block(response, tool_schema["name"])
        if tool_block is None:
            return LLMCallResult(
                success=False,
                classification=None,
                cost_usd=cost,
                duration_seconds=duration,
                failure_reason="no_tool_use_block",
                raw_response_for_forensics=raw,
            )

        try:
            classification = _build_classification(tool_block.input)
        except ValueError as exc:
            return LLMCallResult(
                success=False,
                classification=None,
                cost_usd=cost,
                duration_seconds=duration,
                failure_reason=f"response_validation_failed: {exc}",
                raw_response_for_forensics=raw,
            )

        return LLMCallResult(
            success=True,
            classification=classification,
            cost_usd=cost,
            duration_seconds=duration,
            failure_reason=None,
            raw_response_for_forensics=raw,
        )


def _compute_cost(response: Any) -> float:
    """Multiply reported input/output tokens by the pricing constants."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0.0
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    return input_tokens * SONNET_INPUT_COST_PER_TOKEN + output_tokens * SONNET_OUTPUT_COST_PER_TOKEN


def _find_tool_use_block(response: Any, expected_name: str) -> Any:
    """Return the first ``tool_use`` content block matching ``expected_name``, or None."""
    content = getattr(response, "content", None) or []
    for block in content:
        if (
            getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == expected_name
        ):
            return block
    return None


def _response_to_dict(response: Any) -> dict[str, Any]:
    """Best-effort serialisation of the API response for forensic logging."""
    dump = getattr(response, "model_dump", None)
    if callable(dump):
        try:
            result = dump()
            if isinstance(result, dict):
                return result
        except Exception:  # noqa: BLE001 - forensics is best-effort
            pass
    return {"repr": repr(response)}


def _build_classification(args: Any) -> CatalystClassification:
    """Validate the LLM's tool input and construct ``CatalystClassification``."""
    if not isinstance(args, dict):
        raise ValueError(f"tool input must be an object; got {type(args).__name__}")

    qualifies_raw = args.get("qualifies")
    if not isinstance(qualifies_raw, bool):
        raise ValueError(f"qualifies must be a boolean; got {type(qualifies_raw).__name__}")

    category_raw = args.get("category")
    if category_raw not in _VALID_CATEGORIES:
        raise ValueError(
            f"category must be one of the documented enum values; got {category_raw!r}"
        )

    confidence_raw = args.get("confidence")
    if not isinstance(confidence_raw, int | float):
        raise ValueError(f"confidence must be a number; got {type(confidence_raw).__name__}")
    confidence = float(confidence_raw)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be in [0.0, 1.0]; got {confidence}")

    reasoning_raw = args.get("reasoning", "")
    if not isinstance(reasoning_raw, str):
        raise ValueError(f"reasoning must be a string; got {type(reasoning_raw).__name__}")

    concerns_raw = args.get("concerns", [])
    # Sonnet 4.6 occasionally returns ``concerns`` as a bare string (or
    # comma/semicolon-delimited string) despite the input_schema declaring
    # ``type: array``. Coerce these shapes rather than dropping the whole
    # classification — see the 2026-05-06 ELPW/BIYA failures.
    if concerns_raw is None:
        concerns_raw = []
    elif isinstance(concerns_raw, str):
        concerns_raw = [s.strip() for s in re.split(r"[,;]", concerns_raw) if s.strip()]
    elif not isinstance(concerns_raw, list):
        raise ValueError(f"concerns must be a list; got {type(concerns_raw).__name__}")
    concerns: list[str] = []
    for entry in concerns_raw:
        if not isinstance(entry, str):
            raise ValueError(f"concerns entries must be strings; got {type(entry).__name__}")
        if entry not in _VALID_CONCERNS:
            # Unknown concern strings are dropped silently rather than rejecting
            # the whole classification — the LLM may suggest a label we haven't
            # promoted to the enum yet, and the operator's downstream tooling
            # only matches the documented set anyway.
            continue
        concerns.append(entry)

    return CatalystClassification(
        qualifies=qualifies_raw,
        category=category_raw,
        confidence=confidence,
        reasoning=reasoning_raw[:500],
        concerns=tuple(concerns),
    )


__all__ = [
    "SONNET_INPUT_COST_PER_TOKEN",
    "SONNET_OUTPUT_COST_PER_TOKEN",
    "AnthropicCatalystClient",
    "CatalystClassification",
    "CategoryLiteral",
    "LLMCallResult",
]

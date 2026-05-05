"""Anthropic API client wrapper for the live exit advisor.

Wraps a single tool-use call to Claude Sonnet 4.6: hand it a system
prompt + user message + tool schema, expect a ``tool_use`` content
block back, parse the arguments into an :class:`ExitRecommendation`.

Failures (timeouts, malformed responses, validation errors, API errors)
NEVER raise to the caller — they're returned as
``LLMCallResult(success=False, failure_reason=...)`` so the agent can
log them, count them toward the self-disable threshold, and keep the
bot running. The hook registry's outer try/except is a safety net,
not the primary defense.

Pricing constants (``SONNET_INPUT_COST_PER_TOKEN`` /
``SONNET_OUTPUT_COST_PER_TOKEN``) are module-level so a price change
or a model swap only touches this file. Cost is computed from the
API's own ``usage`` block — never estimated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import anthropic

from bot.exit_advisor.core.types import ExitAction, ExitRecommendation

# --- Pricing (USD per token) -----------------------------------------------
# Claude Sonnet 4.6: $3 / 1M input, $15 / 1M output. Update both constants
# together if Anthropic changes pricing or the agent moves to a different
# model. The cost is multiplied by reported usage tokens, not estimated.
SONNET_INPUT_COST_PER_TOKEN = 3.0 / 1_000_000
SONNET_OUTPUT_COST_PER_TOKEN = 15.0 / 1_000_000


@dataclass(frozen=True)
class LLMCallResult:
    """Outcome of one LLM advisor call.

    On success, ``recommendation`` is populated and ``failure_reason`` is None.
    On failure, ``recommendation`` is None and ``failure_reason`` carries a
    short machine-readable tag (``llm_timeout``, ``api_error: <msg>``,
    ``no_tool_use_block``, ``response_validation_failed: <detail>``, etc.)
    plus optional ``raw_response_for_forensics`` for post-mortem grep.
    ``cost_usd`` is always populated; failures cost zero (no usage reported).
    """

    success: bool
    recommendation: ExitRecommendation | None
    cost_usd: float
    duration_seconds: float
    failure_reason: str | None = None
    raw_response_for_forensics: dict[str, Any] | None = field(default=None)


class AnthropicLLMClient:
    """Synchronous Anthropic API client wired for the advisor's tool-use protocol.

    Synchronous because the Phase 11 hook wrapper already runs each
    advisor call in a worker thread with its own timeout. Adding asyncio
    here would force the whole advisor stack async for no benefit; the
    hook's thread pool is the concurrency model.

    The client holds zero per-trade state — it's pure transport.
    """

    DEFAULT_MODEL = "claude-sonnet-4-6"
    DEFAULT_MAX_TOKENS = 1024
    DEFAULT_TIMEOUT_SECONDS = 12.0

    # max_retries=0 disables the SDK's default 2-retry behavior. With the
    # default max_retries=2, a configured timeout becomes effectively 3x as
    # long in worst case (initial attempt + 2 retries, each respecting the
    # per-attempt timeout). This made the configured timeout misleading —
    # 8s configured, ~25s actual on 3 consecutive failures during the
    # 2026-05-05 CLRB session, which tipped the advisor's self-disable
    # threshold and killed it for the rest of the session.
    #
    # We bound total wait to ``timeout_seconds`` and rely on the
    # application-level retry path (the agent's event-driven buffer/trigger
    # loop in ``agent.py``) for resilience to transient failures. The next
    # event-driven advisor call provides a fresh attempt within typically
    # <60 seconds anyway, so SDK-level retries don't add meaningful value
    # and only obscure the timeout contract.
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
            raise ValueError("AnthropicLLMClient: api_key must be a non-empty string")
        if max_tokens <= 0:
            raise ValueError(f"AnthropicLLMClient: max_tokens must be > 0 (got {max_tokens})")
        if timeout_seconds <= 0.0:
            raise ValueError(
                f"AnthropicLLMClient: timeout_seconds must be > 0 (got {timeout_seconds})"
            )
        if max_retries < 0:
            raise ValueError(f"AnthropicLLMClient: max_retries must be >= 0 (got {max_retries})")
        self._model = model
        self._max_tokens = max_tokens
        self._timeout_seconds = timeout_seconds
        # Allow injection of a fake client for tests; default to the real SDK.
        # Pass ``max_retries=0`` (see DEFAULT_MAX_RETRIES comment above) so the
        # configured ``timeout_seconds`` is the truthful upper bound on wait
        # time, not multiplied by the SDK's auto-retry budget.
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
        """Make one advisor call. Always returns an ``LLMCallResult``; never raises."""
        started = time.monotonic()
        try:
            # The SDK's typed-dict signature is intentionally restrictive; we hand it
            # raw dicts since the schema and tool_choice shape are validated at the
            # API boundary, not by mypy.
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
                recommendation=None,
                cost_usd=0.0,
                duration_seconds=time.monotonic() - started,
                failure_reason="llm_timeout",
            )
        except anthropic.APIError as exc:
            return LLMCallResult(
                success=False,
                recommendation=None,
                cost_usd=0.0,
                duration_seconds=time.monotonic() - started,
                failure_reason=f"api_error: {type(exc).__name__}: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 - never let an unexpected exc escape
            return LLMCallResult(
                success=False,
                recommendation=None,
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
                recommendation=None,
                cost_usd=cost,
                duration_seconds=duration,
                failure_reason="no_tool_use_block",
                raw_response_for_forensics=raw,
            )

        try:
            recommendation = _build_recommendation(tool_block.input)
        except ValueError as exc:
            return LLMCallResult(
                success=False,
                recommendation=None,
                cost_usd=cost,
                duration_seconds=duration,
                failure_reason=f"response_validation_failed: {exc}",
                raw_response_for_forensics=raw,
            )

        return LLMCallResult(
            success=True,
            recommendation=recommendation,
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
    # Fallback: stringify what we can. Never raise from forensic capture.
    return {"repr": repr(response)}


def _build_recommendation(args: Any) -> ExitRecommendation:
    """Validate the LLM's tool input and construct an ``ExitRecommendation``.

    Mirrors the dataclass's own ``__post_init__`` validation so we surface
    a short, machine-readable failure_reason from the client rather than
    letting the dataclass raise from inside the agent.
    """
    if not isinstance(args, dict):
        raise ValueError(f"tool input must be an object; got {type(args).__name__}")

    raw_action = args.get("action")
    if raw_action not in ("hold", "exit_full", "exit_partial", "tighten_stop"):
        raise ValueError(f"action must be one of the four ExitAction values; got {raw_action!r}")
    action: ExitAction = raw_action

    confidence_raw = args.get("confidence")
    if not isinstance(confidence_raw, int | float):
        raise ValueError(f"confidence must be a number; got {type(confidence_raw).__name__}")
    confidence = float(confidence_raw)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be in [0.0, 1.0]; got {confidence}")

    reasoning_raw = args.get("reasoning", "")
    if not isinstance(reasoning_raw, str):
        raise ValueError(f"reasoning must be a string; got {type(reasoning_raw).__name__}")
    reasoning = reasoning_raw

    partial_pct = 0.0
    new_stop_price: float | None = None

    if action == "exit_partial":
        partial_raw = args.get("partial_pct")
        if not isinstance(partial_raw, int | float):
            raise ValueError(
                "partial_pct required and must be a number for action='exit_partial'; "
                f"got {type(partial_raw).__name__}"
            )
        partial_pct = float(partial_raw)
        if not 0.0 < partial_pct <= 0.95:
            raise ValueError(
                f"partial_pct must be in (0.0, 0.95] for action='exit_partial'; got {partial_pct}"
            )
    elif action == "tighten_stop":
        stop_raw = args.get("new_stop_price")
        if not isinstance(stop_raw, int | float):
            raise ValueError(
                "new_stop_price required and must be a number for action='tighten_stop'; "
                f"got {type(stop_raw).__name__}"
            )
        new_stop_price = float(stop_raw)
        if new_stop_price <= 0.0:
            raise ValueError(
                f"new_stop_price must be > 0.0 for action='tighten_stop'; got {new_stop_price}"
            )

    return ExitRecommendation(
        action=action,
        partial_pct=partial_pct,
        new_stop_price=new_stop_price,
        confidence=confidence,
        reason=reasoning[:200],
        source="live_llm_advisor",
    )

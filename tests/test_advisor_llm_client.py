"""Unit tests for the Anthropic LLM client wrapper.

All tests use a fake `anthropic.Anthropic` client to keep the API
unmocked at the network level. The wrapper's contract — return
``LLMCallResult`` and never raise — is exercised across the success
path, validation failures, malformed responses, timeouts, and
arbitrary unexpected exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import anthropic
import pytest

from bot.exit_advisor.advisor.llm_client import (
    SONNET_INPUT_COST_PER_TOKEN,
    SONNET_OUTPUT_COST_PER_TOKEN,
    AnthropicLLMClient,
    _build_recommendation,
    _compute_cost,
)
from bot.exit_advisor.advisor.prompts import (
    EXIT_ADVISOR_SYSTEM_PROMPT,
    EXIT_RECOMMENDATION_TOOL_NAME,
    EXIT_RECOMMENDATION_TOOL_SCHEMA,
)


@dataclass
class _FakeUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class _FakeBlock:
    type: str
    name: str | None = None
    input: Any = None

    def model_dump(self) -> dict[str, Any]:
        return {"type": self.type, "name": self.name, "input": self.input}


@dataclass
class _FakeResponse:
    content: list[_FakeBlock]
    usage: _FakeUsage

    def model_dump(self) -> dict[str, Any]:
        return {
            "content": [b.model_dump() for b in self.content],
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
            },
        }


class _FakeMessages:
    def __init__(self, response: Any | None = None, raise_exc: BaseException | None = None) -> None:
        self._response = response
        self._raise = raise_exc
        self.last_kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
        if self._raise is not None:
            raise self._raise
        return self._response


class _FakeClient:
    def __init__(self, messages: _FakeMessages) -> None:
        self.messages = messages


def _client_with(
    response: Any | None = None, raise_exc: BaseException | None = None
) -> tuple[AnthropicLLMClient, _FakeMessages]:
    fake_messages = _FakeMessages(response=response, raise_exc=raise_exc)
    fake_client = _FakeClient(fake_messages)
    client = AnthropicLLMClient(api_key="sk-test", client=fake_client)  # type: ignore[arg-type]
    return client, fake_messages


def _ok_tool_response(args: dict[str, Any], usage: _FakeUsage | None = None) -> _FakeResponse:
    return _FakeResponse(
        content=[_FakeBlock(type="tool_use", name=EXIT_RECOMMENDATION_TOOL_NAME, input=args)],
        usage=usage or _FakeUsage(input_tokens=1000, output_tokens=200),
    )


def test_successful_tool_use_response_is_parsed() -> None:
    response = _ok_tool_response(
        {"action": "exit_full", "confidence": 0.85, "reasoning": "exhaustion"}
    )
    client, fake = _client_with(response=response)

    result = client.call(EXIT_ADVISOR_SYSTEM_PROMPT, "user msg", EXIT_RECOMMENDATION_TOOL_SCHEMA)

    assert result.success
    assert result.recommendation is not None
    assert result.recommendation.action == "exit_full"
    assert result.recommendation.confidence == pytest.approx(0.85)
    assert result.recommendation.source == "live_llm_advisor"
    assert result.failure_reason is None
    assert fake.last_kwargs["model"] == AnthropicLLMClient.DEFAULT_MODEL
    assert fake.last_kwargs["tools"][0] is EXIT_RECOMMENDATION_TOOL_SCHEMA


def test_cost_calculation_matches_pricing_constants() -> None:
    response = _ok_tool_response(
        {"action": "hold", "confidence": 0.5, "reasoning": "no signal"},
        usage=_FakeUsage(input_tokens=2000, output_tokens=300),
    )
    client, _ = _client_with(response=response)
    result = client.call(EXIT_ADVISOR_SYSTEM_PROMPT, "user msg", EXIT_RECOMMENDATION_TOOL_SCHEMA)

    expected = 2000 * SONNET_INPUT_COST_PER_TOKEN + 300 * SONNET_OUTPUT_COST_PER_TOKEN
    assert result.cost_usd == pytest.approx(expected)


def test_no_tool_use_block_returns_failure() -> None:
    response = _FakeResponse(
        content=[_FakeBlock(type="text")],
        usage=_FakeUsage(input_tokens=500, output_tokens=50),
    )
    client, _ = _client_with(response=response)

    result = client.call(EXIT_ADVISOR_SYSTEM_PROMPT, "user msg", EXIT_RECOMMENDATION_TOOL_SCHEMA)

    assert not result.success
    assert result.recommendation is None
    assert result.failure_reason == "no_tool_use_block"
    assert result.cost_usd > 0.0  # cost is still recorded
    assert result.raw_response_for_forensics is not None


def test_validation_failure_partial_pct_out_of_range() -> None:
    response = _ok_tool_response(
        {"action": "exit_partial", "confidence": 0.7, "reasoning": "scale out", "partial_pct": 0.99}
    )
    client, _ = _client_with(response=response)
    result = client.call(EXIT_ADVISOR_SYSTEM_PROMPT, "user msg", EXIT_RECOMMENDATION_TOOL_SCHEMA)
    assert not result.success
    assert result.failure_reason is not None
    assert "response_validation_failed" in result.failure_reason
    assert "partial_pct" in result.failure_reason


def test_validation_failure_confidence_out_of_range() -> None:
    response = _ok_tool_response({"action": "hold", "confidence": 1.5, "reasoning": "buggy"})
    client, _ = _client_with(response=response)
    result = client.call(EXIT_ADVISOR_SYSTEM_PROMPT, "user msg", EXIT_RECOMMENDATION_TOOL_SCHEMA)
    assert not result.success
    assert result.failure_reason is not None
    assert "confidence" in result.failure_reason


def test_validation_failure_unknown_action() -> None:
    response = _ok_tool_response({"action": "bogus", "confidence": 0.5, "reasoning": "x"})
    client, _ = _client_with(response=response)
    result = client.call(EXIT_ADVISOR_SYSTEM_PROMPT, "user msg", EXIT_RECOMMENDATION_TOOL_SCHEMA)
    assert not result.success
    assert result.failure_reason is not None
    assert "action" in result.failure_reason


def test_validation_failure_tighten_stop_missing_stop_price() -> None:
    response = _ok_tool_response(
        {"action": "tighten_stop", "confidence": 0.7, "reasoning": "trail tighter"}
    )
    client, _ = _client_with(response=response)
    result = client.call(EXIT_ADVISOR_SYSTEM_PROMPT, "user msg", EXIT_RECOMMENDATION_TOOL_SCHEMA)
    assert not result.success
    assert result.failure_reason is not None
    assert "new_stop_price" in result.failure_reason


def test_timeout_returns_failure_with_llm_timeout_reason() -> None:
    timeout_exc = anthropic.APITimeoutError(request=None)  # type: ignore[arg-type]
    client, _ = _client_with(raise_exc=timeout_exc)
    result = client.call(EXIT_ADVISOR_SYSTEM_PROMPT, "user msg", EXIT_RECOMMENDATION_TOOL_SCHEMA)
    assert not result.success
    assert result.failure_reason == "llm_timeout"
    assert result.cost_usd == 0.0
    assert result.recommendation is None


def test_unexpected_exception_caught_and_reported() -> None:
    client, _ = _client_with(raise_exc=RuntimeError("network gone"))
    result = client.call(EXIT_ADVISOR_SYSTEM_PROMPT, "user msg", EXIT_RECOMMENDATION_TOOL_SCHEMA)
    assert not result.success
    assert result.failure_reason is not None
    assert "unexpected_error" in result.failure_reason
    assert "network gone" in result.failure_reason


def test_compute_cost_handles_missing_usage() -> None:
    class _NoUsage:
        pass

    assert _compute_cost(_NoUsage()) == 0.0


def test_build_recommendation_rejects_non_dict() -> None:
    with pytest.raises(ValueError, match="object"):
        _build_recommendation("not a dict")


def test_constructor_rejects_empty_api_key() -> None:
    with pytest.raises(ValueError, match="api_key"):
        AnthropicLLMClient(api_key="")

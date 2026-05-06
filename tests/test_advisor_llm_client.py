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
from unittest.mock import MagicMock

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


def test_constructor_rejects_negative_max_retries() -> None:
    """``max_retries`` must be >= 0 — negative would be meaningless and the SDK rejects it."""
    with pytest.raises(ValueError, match="max_retries"):
        AnthropicLLMClient(api_key="sk-test", max_retries=-1)


# ---- SDK constructor kwargs (regression for 2026-05-05 CLRB self-disable) ----
#
# Existing tests in this file inject a fake ``client`` via the ``client=``
# kwarg, which short-circuits the ``anthropic.Anthropic(...)`` constructor
# call entirely. As a result, the kwargs passed to the SDK constructor were
# previously untested — the wrapper could ship any ``max_retries`` value (or
# none at all) without a single test failing. The two tests below close that
# coverage gap.


def test_constructor_passes_max_retries_zero_to_anthropic_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrapper MUST pass ``max_retries=0`` to ``anthropic.Anthropic``.

    With the SDK default ``max_retries=2``, the configured ``timeout_seconds``
    becomes effectively 3x as long in worst case. That misled the
    self-disable threshold during the 2026-05-05 CLRB session: 8s configured,
    ~25s actual on three consecutive failures. ``max_retries=0`` keeps the
    timeout truthful.
    """
    captured_kwargs: dict[str, Any] = {}

    def _fake_anthropic(**kwargs: Any) -> Any:
        captured_kwargs.update(kwargs)
        return MagicMock(name="FakeAnthropic")

    import bot.exit_advisor.advisor.llm_client as llm_client_module

    monkeypatch.setattr(llm_client_module.anthropic, "Anthropic", _fake_anthropic)

    AnthropicLLMClient(api_key="sk-test")

    assert captured_kwargs.get("api_key") == "sk-test"
    assert captured_kwargs.get("max_retries") == 0, (
        "Wrapper must pass max_retries=0 so the configured timeout is the truthful "
        "upper bound on wait time. SDK default of 2 multiplies the timeout by up "
        "to 3x — see DEFAULT_MAX_RETRIES comment in llm_client.py."
    )


def test_constructor_max_retries_is_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit non-default ``max_retries`` flows through to the SDK constructor.

    Belt-and-suspenders pin so a future caller that wants retry behavior (e.g. a
    long-running backfill harness, not the live advisor) can still opt in
    without the wrapper silently clamping to zero.
    """
    captured_kwargs: dict[str, Any] = {}

    def _fake_anthropic(**kwargs: Any) -> Any:
        captured_kwargs.update(kwargs)
        return MagicMock(name="FakeAnthropic")

    import bot.exit_advisor.advisor.llm_client as llm_client_module

    monkeypatch.setattr(llm_client_module.anthropic, "Anthropic", _fake_anthropic)

    AnthropicLLMClient(api_key="sk-test", max_retries=5)

    assert captured_kwargs.get("max_retries") == 5


def test_timeout_not_multiplied_by_sdk_retries() -> None:
    """A single timeout from the SDK returns failure once — no SDK-level retry loop.

    Companion to ``test_constructor_passes_max_retries_zero_to_anthropic_sdk``:
    that one pins the kwarg, this one confirms the runtime contract — when the
    underlying call raises ``APITimeoutError``, the wrapper returns
    ``LLMCallResult(success=False, failure_reason='llm_timeout')`` immediately
    without invoking ``messages.create`` a second time. Together they establish
    that 12s configured = 12s actual, not 3x.
    """
    timeout_exc = anthropic.APITimeoutError(request=None)  # type: ignore[arg-type]
    client, fake = _client_with(raise_exc=timeout_exc)

    # Tracking-side counter on the existing _FakeMessages.create — the fake's
    # implementation always raises if ``_raise`` is set, so a retry loop would
    # call .create twice. We assert it was called exactly once.
    call_count = 0
    original_create = fake.create

    def _counting_create(**kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        return original_create(**kwargs)

    fake.create = _counting_create  # type: ignore[method-assign]

    result = client.call(EXIT_ADVISOR_SYSTEM_PROMPT, "user msg", EXIT_RECOMMENDATION_TOOL_SCHEMA)

    assert not result.success
    assert result.failure_reason == "llm_timeout"
    assert call_count == 1, (
        f"Wrapper must not retry on APITimeoutError (got {call_count} calls). "
        "If this fails, max_retries=0 is no longer being honored — "
        "see the DEFAULT_MAX_RETRIES comment in llm_client.py for context."
    )


def test_default_timeout_constant_matches_today_normal_case_latency() -> None:
    """The default timeout pins at 12.0s.

    Bumped from 8.0 → 12.0 after the 2026-05-05 CLRB session showed normal-case
    Sonnet 4.6 tool-use latencies of 6.4s and 7.8s — too tight against the old
    ceiling. 12.0 leaves ~50% headroom over observed normal latency. If you
    have evidence to bump again, update this constant AND this test together.
    """
    assert AnthropicLLMClient.DEFAULT_TIMEOUT_SECONDS == 12.0
    assert AnthropicLLMClient.DEFAULT_MAX_RETRIES == 0


def test_settings_load_pins_llm_timeout_at_or_above_twelve_seconds() -> None:
    """The shipped ``config.yaml`` must keep ``exit_advisor.llm_timeout_seconds`` >= 12.0.

    Pin the on-disk config value so a future YAML edit doesn't silently revert
    BELOW the 2026-05-05 CLRB session bump (originally 8.0 → 12.0). Operators
    are free to raise it above 12.0 (the 2026-05-05 ENVB session showed the
    operator legitimately tuning it to 30.0); only a regression to <12.0
    breaks the assertion.
    """
    from bot.config import get_settings

    settings = get_settings()
    assert settings.exit_advisor.llm_timeout_seconds >= 12.0, (
        f"llm_timeout_seconds must not regress below 12.0; got "
        f"{settings.exit_advisor.llm_timeout_seconds}. See 2026-05-05 CLRB session."
    )

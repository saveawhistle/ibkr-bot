"""Anthropic client wrapper tests for the Phase 12 catalyst classifier.

Mirrors ``tests/test_advisor_llm_client.py`` — fakes the SDK at the
``anthropic.Anthropic`` boundary, never makes real API calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import anthropic
import pytest

from bot.scanning.llm_catalyst_classifier.llm_client import (
    SONNET_INPUT_COST_PER_TOKEN,
    SONNET_OUTPUT_COST_PER_TOKEN,
    AnthropicCatalystClient,
    _build_classification,
    _compute_cost,
)
from bot.scanning.llm_catalyst_classifier.prompts import (
    CATALYST_CLASSIFIER_SYSTEM_PROMPT,
    CLASSIFY_CATALYST_TOOL,
    CLASSIFY_CATALYST_TOOL_NAME,
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
        self.call_count = 0

    def create(self, **kwargs: Any) -> Any:
        self.call_count += 1
        self.last_kwargs = kwargs
        if self._raise is not None:
            raise self._raise
        return self._response


class _FakeClient:
    def __init__(self, messages: _FakeMessages) -> None:
        self.messages = messages


def _client_with(
    response: Any | None = None,
    raise_exc: BaseException | None = None,
) -> tuple[AnthropicCatalystClient, _FakeMessages]:
    fake = _FakeMessages(response=response, raise_exc=raise_exc)
    client = AnthropicCatalystClient(api_key="sk-test", client=_FakeClient(fake))  # type: ignore[arg-type]
    return client, fake


def _ok_tool_response(args: dict[str, Any], usage: _FakeUsage | None = None) -> _FakeResponse:
    return _FakeResponse(
        content=[_FakeBlock(type="tool_use", name=CLASSIFY_CATALYST_TOOL_NAME, input=args)],
        usage=usage or _FakeUsage(input_tokens=1500, output_tokens=120),
    )


# ---------------- constructor ---------------- #


def test_constructor_rejects_empty_api_key() -> None:
    with pytest.raises(ValueError, match="api_key"):
        AnthropicCatalystClient(api_key="")


def test_constructor_rejects_negative_max_retries() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        AnthropicCatalystClient(api_key="sk-test", max_retries=-1)


def test_constructor_rejects_zero_timeout() -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        AnthropicCatalystClient(api_key="sk-test", timeout_seconds=0.0)


def test_constructor_passes_max_retries_zero_to_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 12 must pass max_retries=0 to anthropic.Anthropic by default.

    Closes the same coverage gap as the exit advisor's
    ``test_constructor_passes_max_retries_zero_to_anthropic_sdk`` —
    existing tests inject a fake client and never exercise the real
    constructor; this test monkeypatches the SDK and asserts the kwarg.
    """
    captured: dict[str, Any] = {}

    def _fake_anthropic(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return MagicMock(name="FakeAnthropic")

    import bot.scanning.llm_catalyst_classifier.llm_client as llm_client_module

    monkeypatch.setattr(llm_client_module.anthropic, "Anthropic", _fake_anthropic)

    AnthropicCatalystClient(api_key="sk-test")
    assert captured.get("api_key") == "sk-test"
    assert captured.get("max_retries") == 0


# ---------------- success path ---------------- #


def test_successful_tool_use_response_parsed() -> None:
    response = _ok_tool_response(
        {
            "qualifies": True,
            "category": "earnings_beat",
            "confidence": 0.85,
            "reasoning": "Beat with raised guidance",
            "concerns": [],
        }
    )
    client, fake = _client_with(response=response)
    result = client.call(CATALYST_CLASSIFIER_SYSTEM_PROMPT, "user msg", CLASSIFY_CATALYST_TOOL)
    assert result.success
    assert result.classification is not None
    assert result.classification.qualifies is True
    assert result.classification.category == "earnings_beat"
    assert result.classification.confidence == pytest.approx(0.85)
    assert result.failure_reason is None
    assert fake.last_kwargs["model"] == AnthropicCatalystClient.DEFAULT_MODEL
    assert fake.last_kwargs["tools"][0] is CLASSIFY_CATALYST_TOOL


def test_cost_calculation_matches_pricing_constants() -> None:
    response = _ok_tool_response(
        {
            "qualifies": False,
            "category": "stale_news",
            "confidence": 0.6,
            "reasoning": "older than 72h",
        },
        usage=_FakeUsage(input_tokens=2500, output_tokens=200),
    )
    client, _ = _client_with(response=response)
    result = client.call(CATALYST_CLASSIFIER_SYSTEM_PROMPT, "user msg", CLASSIFY_CATALYST_TOOL)
    expected = 2500 * SONNET_INPUT_COST_PER_TOKEN + 200 * SONNET_OUTPUT_COST_PER_TOKEN
    assert result.cost_usd == pytest.approx(expected)


def test_classification_strips_unknown_concern_tags() -> None:
    """LLM may suggest a concern label outside the documented enum; drop silently."""
    response = _ok_tool_response(
        {
            "qualifies": True,
            "category": "earnings_beat",
            "confidence": 0.7,
            "reasoning": "ok",
            "concerns": ["dilutive_financing", "speculative_label", "non_binding_agreement"],
        }
    )
    client, _ = _client_with(response=response)
    result = client.call(CATALYST_CLASSIFIER_SYSTEM_PROMPT, "user msg", CLASSIFY_CATALYST_TOOL)
    assert result.success
    assert result.classification is not None
    assert set(result.classification.concerns) == {
        "dilutive_financing",
        "non_binding_agreement",
    }


# ---------------- failure paths ---------------- #


def test_no_tool_use_block_returns_failure() -> None:
    response = _FakeResponse(
        content=[_FakeBlock(type="text")],
        usage=_FakeUsage(input_tokens=500, output_tokens=50),
    )
    client, _ = _client_with(response=response)
    result = client.call(CATALYST_CLASSIFIER_SYSTEM_PROMPT, "user msg", CLASSIFY_CATALYST_TOOL)
    assert not result.success
    assert result.failure_reason == "no_tool_use_block"


def test_validation_failure_invalid_category() -> None:
    response = _ok_tool_response(
        {"qualifies": True, "category": "bogus_category", "confidence": 0.5, "reasoning": "x"}
    )
    client, _ = _client_with(response=response)
    result = client.call(CATALYST_CLASSIFIER_SYSTEM_PROMPT, "user msg", CLASSIFY_CATALYST_TOOL)
    assert not result.success
    assert result.failure_reason is not None
    assert "category" in result.failure_reason


def test_validation_failure_invalid_confidence() -> None:
    response = _ok_tool_response(
        {"qualifies": True, "category": "earnings_beat", "confidence": 1.5, "reasoning": "x"}
    )
    client, _ = _client_with(response=response)
    result = client.call(CATALYST_CLASSIFIER_SYSTEM_PROMPT, "user msg", CLASSIFY_CATALYST_TOOL)
    assert not result.success
    assert result.failure_reason is not None
    assert "confidence" in result.failure_reason


def test_validation_failure_qualifies_not_bool() -> None:
    response = _ok_tool_response(
        {
            "qualifies": "true",  # string instead of bool — invalid
            "category": "earnings_beat",
            "confidence": 0.5,
            "reasoning": "x",
        }
    )
    client, _ = _client_with(response=response)
    result = client.call(CATALYST_CLASSIFIER_SYSTEM_PROMPT, "user msg", CLASSIFY_CATALYST_TOOL)
    assert not result.success
    assert result.failure_reason is not None
    assert "qualifies" in result.failure_reason


def test_timeout_returns_failure_with_llm_timeout_reason() -> None:
    timeout_exc = anthropic.APITimeoutError(request=None)  # type: ignore[arg-type]
    client, fake = _client_with(raise_exc=timeout_exc)
    result = client.call(CATALYST_CLASSIFIER_SYSTEM_PROMPT, "user msg", CLASSIFY_CATALYST_TOOL)
    assert not result.success
    assert result.failure_reason == "llm_timeout"
    assert result.cost_usd == 0.0
    # max_retries=0 contract: only ONE call attempt despite the timeout.
    assert fake.call_count == 1


def test_unexpected_exception_caught() -> None:
    client, _ = _client_with(raise_exc=RuntimeError("network gone"))
    result = client.call(CATALYST_CLASSIFIER_SYSTEM_PROMPT, "user msg", CLASSIFY_CATALYST_TOOL)
    assert not result.success
    assert result.failure_reason is not None
    assert "unexpected_error" in result.failure_reason


# ---------------- helpers ---------------- #


def test_compute_cost_handles_missing_usage() -> None:
    class _NoUsage:
        pass

    assert _compute_cost(_NoUsage()) == 0.0


def test_build_classification_rejects_non_dict() -> None:
    with pytest.raises(ValueError, match="object"):
        _build_classification("not a dict")


def _valid_args(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "qualifies": False,
        "category": "non_qualifying_other",
        "confidence": 0.5,
        "reasoning": "test",
    }
    base.update(overrides)
    return base


def test_build_classification_coerces_concerns_string_to_single_item() -> None:
    result = _build_classification(_valid_args(concerns="dilutive_financing"))
    assert result.concerns == ("dilutive_financing",)


def test_build_classification_coerces_concerns_comma_separated_string() -> None:
    result = _build_classification(
        _valid_args(concerns="dilutive_financing, post_close_news")
    )
    assert result.concerns == ("dilutive_financing", "post_close_news")


def test_build_classification_coerces_concerns_semicolon_separated_string() -> None:
    result = _build_classification(
        _valid_args(concerns="dilutive_financing; post_close_news")
    )
    assert result.concerns == ("dilutive_financing", "post_close_news")


def test_build_classification_treats_concerns_none_as_empty() -> None:
    result = _build_classification(_valid_args(concerns=None))
    assert result.concerns == ()


def test_build_classification_drops_unknown_concern_strings_after_split() -> None:
    result = _build_classification(
        _valid_args(concerns="dilutive_financing, made_up_label")
    )
    assert result.concerns == ("dilutive_financing",)


def test_build_classification_rejects_concerns_dict() -> None:
    with pytest.raises(ValueError, match="concerns must be a list"):
        _build_classification(_valid_args(concerns={"bad": "shape"}))

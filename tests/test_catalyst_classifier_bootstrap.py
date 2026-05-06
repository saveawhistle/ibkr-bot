"""Tests for ``bot.scanning.llm_catalyst_classifier.bootstrap``."""

from __future__ import annotations

import pytest

from bot.config import Settings
from bot.scanning.llm_catalyst_classifier.bootstrap import bootstrap_catalyst_classifier


@pytest.fixture(autouse=True)
def _isolate_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub out load_dotenv so the operator's real .env doesn't bleed into tests.

    Same shape as the exit advisor's bootstrap fixture — without this,
    ``test_returns_none_when_api_key_missing`` fails when the operator
    has a real ANTHROPIC_API_KEY in .env (load_dotenv promotes it back
    into os.environ even after the test deletes it).
    """
    monkeypatch.setattr(
        "bot.scanning.llm_catalyst_classifier.bootstrap.load_dotenv",
        lambda **_: False,
    )


def _settings_with_llm(*, enabled: bool = True) -> Settings:
    base = Settings()
    return base.model_copy(
        update={
            "catalyst_classifier": base.catalyst_classifier.model_copy(
                update={
                    "llm": base.catalyst_classifier.llm.model_copy(update={"enabled": enabled}),
                }
            )
        }
    )


def test_returns_none_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert bootstrap_catalyst_classifier(_settings_with_llm(enabled=False)) is None


def test_returns_none_when_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert bootstrap_catalyst_classifier(_settings_with_llm(enabled=True)) is None


def test_returns_none_when_api_key_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    assert bootstrap_catalyst_classifier(_settings_with_llm(enabled=True)) is None


def test_returns_none_on_construction_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """If AnthropicCatalystClient construction raises, bootstrap returns None."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("construction failed")

    monkeypatch.setattr(
        "bot.scanning.llm_catalyst_classifier.bootstrap.AnthropicCatalystClient",
        _explode,
    )
    assert bootstrap_catalyst_classifier(_settings_with_llm(enabled=True)) is None


def test_successful_bootstrap_returns_classifier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    classifier = bootstrap_catalyst_classifier(_settings_with_llm(enabled=True))
    assert classifier is not None
    # Session set starts empty.
    assert classifier.qualified_this_session() == set()
    assert not classifier.is_self_disabled()

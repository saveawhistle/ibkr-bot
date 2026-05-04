"""Unit tests for bootstrap_advisor()."""

from __future__ import annotations

import pytest

from bot.config import Settings
from bot.exit_advisor.advisor.bootstrap import bootstrap_advisor
from bot.exit_advisor.hook.registry import registered_advisor, unregister_exit_advisor


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    """Make sure each test starts and ends with no registered advisor."""
    unregister_exit_advisor()
    yield
    unregister_exit_advisor()


@pytest.fixture(autouse=True)
def _isolate_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub out load_dotenv so the operator's real .env doesn't bleed into tests.

    Without this, ``test_returns_none_when_api_key_missing`` fails as soon as the
    operator adds a real ANTHROPIC_API_KEY to .env: bootstrap calls load_dotenv()
    with override=False, which then promotes the real key into os.environ even
    though the test has explicitly delenv'd it.
    """
    monkeypatch.setattr("bot.exit_advisor.advisor.bootstrap.load_dotenv", lambda **_: False)


def _settings_with_advisor(*, enabled: bool = True, hook_acts: bool = False) -> Settings:
    """Build a Settings whose exit_advisor block matches the test scenario."""
    raw: dict[str, dict[str, object]] = {
        "exit_advisor": {
            "enabled": enabled,
            "hook_acts": hook_acts,
        }
    }
    base = Settings()
    return base.model_copy(
        update={"exit_advisor": base.exit_advisor.model_copy(update=raw["exit_advisor"])}
    )


def test_returns_none_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    settings = _settings_with_advisor(enabled=False)
    assert bootstrap_advisor(settings) is None
    assert registered_advisor() is None


def test_returns_none_when_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = _settings_with_advisor(enabled=True)
    assert bootstrap_advisor(settings) is None
    assert registered_advisor() is None


def test_returns_none_when_api_key_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    settings = _settings_with_advisor(enabled=True)
    assert bootstrap_advisor(settings) is None
    assert registered_advisor() is None


def test_returns_none_on_construction_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """If AnthropicLLMClient.__init__ raises, bootstrap returns None and logs."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("constructor blew up")

    monkeypatch.setattr(
        "bot.exit_advisor.advisor.bootstrap.AnthropicLLMClient",
        _explode,
    )
    settings = _settings_with_advisor(enabled=True)
    assert bootstrap_advisor(settings) is None
    assert registered_advisor() is None


def test_successful_bootstrap_returns_advisor_and_registers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    settings = _settings_with_advisor(enabled=True, hook_acts=False)
    advisor = bootstrap_advisor(settings)
    assert advisor is not None
    assert registered_advisor() is advisor

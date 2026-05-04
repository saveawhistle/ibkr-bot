"""Phase 11 — ExitAdvisorConfig validation + production-default invariants."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bot.config import ExitAdvisorConfig, get_settings


def test_defaults_match_production_main_invariants() -> None:
    """The shipping defaults are: hook OFF, hook_acts OFF, 10s timeout, log skipped ON.

    These are the **production-main** invariants the spec calls out:
    Phase 11 ships disabled; the spike branch flips both ``enabled`` and
    ``hook_acts`` to true in its own config.
    """
    cfg = ExitAdvisorConfig()
    assert cfg.enabled is False
    assert cfg.hook_acts is False
    assert cfg.timeout_seconds == 10.0
    assert cfg.log_skipped_events is True


def test_settings_defaults_load_with_phase_11_disabled() -> None:
    """Loading the repo config.yaml ⇒ exit_advisor block disabled (production-main)."""
    settings = get_settings()
    assert settings.exit_advisor.enabled is False
    assert settings.exit_advisor.hook_acts is False


def test_timeout_must_be_positive() -> None:
    """Zero/negative timeouts would abandon every call → reject at startup."""
    with pytest.raises(ValidationError):
        ExitAdvisorConfig(timeout_seconds=0.0)
    with pytest.raises(ValidationError):
        ExitAdvisorConfig(timeout_seconds=-1.0)


def test_hook_acts_requires_enabled() -> None:
    """``hook_acts=true`` without ``enabled=true`` is the obvious YAML typo."""
    with pytest.raises(ValidationError, match="hook_acts=true requires"):
        ExitAdvisorConfig(enabled=False, hook_acts=True)


def test_enabled_true_hook_acts_false_is_valid() -> None:
    """The diagnostic 'log-only' mode is supported (and useful)."""
    cfg = ExitAdvisorConfig(enabled=True, hook_acts=False)
    assert cfg.enabled is True
    assert cfg.hook_acts is False


def test_full_enable_combination_is_valid() -> None:
    """Both ``enabled`` and ``hook_acts`` true together (the spike-branch config)."""
    cfg = ExitAdvisorConfig(enabled=True, hook_acts=True, timeout_seconds=5.0)
    assert cfg.enabled is True
    assert cfg.hook_acts is True
    assert cfg.timeout_seconds == 5.0


def test_log_skipped_events_can_be_disabled() -> None:
    """Operators in busy sessions can flip ``log_skipped_events`` off."""
    cfg = ExitAdvisorConfig(enabled=True, log_skipped_events=False)
    assert cfg.log_skipped_events is False

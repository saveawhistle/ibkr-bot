"""Phase 11 — exit-advisor hook registry, invocation, timeout, exception handling.

Covers the public surface of :mod:`bot.exit_advisor.hook.registry`: registration
lifecycle, the disabled-by-default no-op contract, exception
isolation, timeout enforcement, and the structured logging that
distinguishes skipped / held / actionable outcomes.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import pytest
from structlog.testing import capture_logs

from bot.config import (
    AccountConfig,
    CatalystConfig,
    DataSourcesSettings,
    ExecutionConfig,
    ExitAdvisorConfig,
    IBKRConfig,
    LoggingSettings,
    RiskConfig,
    SessionConfig,
    Settings,
    StrategiesConfig,
    UniverseConfig,
    WatchdogConfig,
)
from bot.config import (
    TestingConfig as _TestingConfigModel,  # aliased: avoids pytest "Test*" collection warning
)
from bot.exit_advisor import (
    AdvisorResponse,
    BarFinalizedEvent,
    ExitRecommendation,
    notify_event,
    notify_position_closed,
    notify_position_protected,
    register_exit_advisor,
    registered_advisor,
    unregister_exit_advisor,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_settings(
    *,
    enabled: bool = True,
    hook_acts: bool = False,
    timeout_seconds: float = 1.0,
    log_skipped_events: bool = True,
) -> Settings:
    """Build a Settings instance with a custom exit_advisor block.

    Side-steps YAML loading so tests are deterministic regardless of
    repo-root ``config.yaml`` contents.
    """
    return Settings(
        account=AccountConfig(),
        risk=RiskConfig(),
        execution=ExecutionConfig(),
        session=SessionConfig(),
        universe=UniverseConfig(),
        strategies=StrategiesConfig(),
        ibkr=IBKRConfig(),
        data_sources=DataSourcesSettings(),
        logging=LoggingSettings(),
        testing=_TestingConfigModel(),
        watchdog=WatchdogConfig(),
        catalyst=CatalystConfig(),
        exit_advisor=ExitAdvisorConfig(
            enabled=enabled,
            hook_acts=hook_acts,
            timeout_seconds=timeout_seconds,
            log_skipped_events=log_skipped_events,
        ),
    )


class _StubPosition:
    """Minimal duck-typed position satisfying the PositionLike protocol."""

    def __init__(self, symbol: str = "TEST") -> None:
        self.symbol = symbol
        self.strategy = "test"
        self.shares = 100
        self.avg_price = 2.20
        self.stop_price = 2.15
        self.scale_out_price = 2.30
        self.status = "open"
        self.scaled_out = False


def _make_bar_event(symbol: str = "TEST") -> BarFinalizedEvent:
    """Construct a deterministic BarFinalizedEvent for tests."""
    return BarFinalizedEvent(
        timestamp=datetime(2026, 4, 30, 14, 31, tzinfo=UTC),
        symbol=symbol,
        open=2.20,
        high=2.31,
        low=2.18,
        close=2.27,
        volume=10_000.0,
    )


@pytest.fixture(autouse=True)
def _isolated_registry() -> Any:
    """Ensure each test starts/ends with no advisor registered."""
    unregister_exit_advisor()
    yield
    unregister_exit_advisor()


# ---------------------------------------------------------------------------
# Registry semantics
# ---------------------------------------------------------------------------


class _RecordingAdvisor:
    """Captures calls so tests can assert routing without mocks."""

    def __init__(self, response: AdvisorResponse | None = None) -> None:
        self.response = response or AdvisorResponse()
        self.protected_calls: list[Any] = []
        self.event_calls: list[tuple[Any, Any]] = []
        self.closed_calls: list[tuple[Any, float]] = []

    def on_position_protected(self, position: Any) -> None:
        self.protected_calls.append(position)

    def on_event(self, position: Any, event: Any) -> AdvisorResponse:
        self.event_calls.append((position, event))
        return self.response

    def on_position_closed(self, position: Any, final_pnl: float) -> None:
        self.closed_calls.append((position, final_pnl))


def test_register_and_unregister() -> None:
    """register/unregister symmetric; registered_advisor reflects state."""
    assert registered_advisor() is None
    advisor = _RecordingAdvisor()
    register_exit_advisor(advisor)
    assert registered_advisor() is advisor
    unregister_exit_advisor()
    assert registered_advisor() is None


def test_re_register_logs_warning() -> None:
    """Replacing an existing advisor logs ``advisor_replaced`` for visibility."""
    register_exit_advisor(_RecordingAdvisor())
    with capture_logs() as captured:
        register_exit_advisor(_RecordingAdvisor())
    events = [e["event"] for e in captured]
    assert "exit_advisor.advisor_replaced" in events


# ---------------------------------------------------------------------------
# Disabled-by-default contract
# ---------------------------------------------------------------------------


def test_notify_with_disabled_config_short_circuits_even_when_advisor_registered() -> None:
    """``enabled=false`` ⇒ no advisor methods called even when one is registered.

    This is the production-main contract: the bot's behaviour is
    identical to pre-Phase-11 with the default config.
    """
    advisor = _RecordingAdvisor()
    register_exit_advisor(advisor)
    settings = _make_settings(enabled=False)
    pos = _StubPosition()
    ev = _make_bar_event()

    notify_position_protected(pos, settings=settings)
    response = notify_event(pos, ev, settings=settings)
    notify_position_closed(pos, 12.5, settings=settings)

    assert advisor.protected_calls == []
    assert advisor.event_calls == []
    assert advisor.closed_calls == []
    assert response.is_skipped


def test_notify_with_no_advisor_short_circuits() -> None:
    """``enabled=true`` but no advisor registered ⇒ silent no-op."""
    settings = _make_settings(enabled=True)
    pos = _StubPosition()
    ev = _make_bar_event()
    response = notify_event(pos, ev, settings=settings)
    assert response.is_skipped
    notify_position_protected(pos, settings=settings)
    notify_position_closed(pos, 0.0, settings=settings)


# ---------------------------------------------------------------------------
# Hook lifecycle is invoked when enabled + registered
# ---------------------------------------------------------------------------


def test_lifecycle_methods_called_when_enabled() -> None:
    """All three lifecycle methods route to the registered advisor."""
    advisor = _RecordingAdvisor()
    register_exit_advisor(advisor)
    settings = _make_settings(enabled=True)
    pos = _StubPosition("ZENA")
    ev = _make_bar_event("ZENA")

    notify_position_protected(pos, settings=settings)
    notify_event(pos, ev, settings=settings)
    notify_position_closed(pos, 22.50, settings=settings)

    assert len(advisor.protected_calls) == 1
    assert advisor.protected_calls[0] is pos
    assert len(advisor.event_calls) == 1
    assert advisor.event_calls[0][1] is ev
    assert advisor.closed_calls == [(pos, 22.50)]


def test_actionable_response_returned_to_caller() -> None:
    """An actionable response from the advisor flows back to notify_event's caller."""
    rec = ExitRecommendation(action="exit_full", reason="break", source="test")
    advisor = _RecordingAdvisor(
        AdvisorResponse(recommendation=rec, evaluation_performed=True, reasoning="...")
    )
    register_exit_advisor(advisor)
    settings = _make_settings(enabled=True)
    response = notify_event(_StubPosition(), _make_bar_event(), settings=settings)
    assert response.is_actionable
    assert response.recommendation is rec


# ---------------------------------------------------------------------------
# Exception isolation — hook bugs MUST NOT crash the bot
# ---------------------------------------------------------------------------


class _RaisingAdvisor:
    """Every method raises — used to verify exception swallowing + logging."""

    def on_position_protected(self, position: Any) -> None:
        raise RuntimeError("boom-protected")

    def on_event(self, position: Any, event: Any) -> AdvisorResponse:
        raise RuntimeError("boom-event")

    def on_position_closed(self, position: Any, final_pnl: float) -> None:
        raise RuntimeError("boom-closed")


def test_exception_in_on_position_protected_is_caught_and_logged() -> None:
    """Hook exception ⇒ logged at ERROR with traceback, no propagation."""
    register_exit_advisor(_RaisingAdvisor())
    settings = _make_settings(enabled=True)
    with capture_logs() as captured:
        notify_position_protected(_StubPosition(), settings=settings)  # must not raise
    failures = [e for e in captured if e["event"] == "exit_advisor.position_protected_failed"]
    assert len(failures) == 1
    assert "boom-protected" in failures[0]["error"]
    assert "Traceback" in failures[0]["traceback"] or "RuntimeError" in failures[0]["traceback"]


def test_exception_in_on_event_returns_skipped_and_logs_failure() -> None:
    """on_event exception ⇒ AdvisorResponse(skipped) returned + event_failed log."""
    register_exit_advisor(_RaisingAdvisor())
    settings = _make_settings(enabled=True)
    with capture_logs() as captured:
        response = notify_event(_StubPosition(), _make_bar_event(), settings=settings)
    assert response.is_skipped
    failures = [e for e in captured if e["event"] == "exit_advisor.event_failed"]
    assert len(failures) == 1
    assert failures[0]["cause"] == "exception"


def test_exception_in_on_position_closed_is_caught() -> None:
    """on_position_closed exception ⇒ swallowed; the close path must not crash."""
    register_exit_advisor(_RaisingAdvisor())
    settings = _make_settings(enabled=True)
    with capture_logs() as captured:
        notify_position_closed(_StubPosition(), 0.0, settings=settings)  # must not raise
    failures = [e for e in captured if e["event"] == "exit_advisor.position_closed_failed"]
    assert len(failures) == 1


# ---------------------------------------------------------------------------
# Timeout enforcement
# ---------------------------------------------------------------------------


class _SlowAdvisor:
    """on_event sleeps longer than the configured timeout."""

    def __init__(self, sleep_seconds: float) -> None:
        self.sleep_seconds = sleep_seconds

    def on_position_protected(self, position: Any) -> None:
        time.sleep(self.sleep_seconds)

    def on_event(self, position: Any, event: Any) -> AdvisorResponse:
        time.sleep(self.sleep_seconds)
        return AdvisorResponse()

    def on_position_closed(self, position: Any, final_pnl: float) -> None:
        time.sleep(self.sleep_seconds)


def test_on_event_timeout_returns_skipped_and_logs_failure() -> None:
    """A hook call exceeding ``timeout_seconds`` → skipped + ``cause=timeout``."""
    register_exit_advisor(_SlowAdvisor(sleep_seconds=0.5))
    settings = _make_settings(enabled=True, timeout_seconds=0.1)
    with capture_logs() as captured:
        response = notify_event(_StubPosition(), _make_bar_event(), settings=settings)
    assert response.is_skipped
    failures = [e for e in captured if e["event"] == "exit_advisor.event_failed"]
    assert len(failures) == 1
    assert failures[0]["cause"] == "timeout"


def test_on_position_protected_timeout_logs_and_returns() -> None:
    """Protected timeout produces a ``position_protected_timeout`` log; no crash."""
    register_exit_advisor(_SlowAdvisor(sleep_seconds=0.5))
    settings = _make_settings(enabled=True, timeout_seconds=0.1)
    with capture_logs() as captured:
        notify_position_protected(_StubPosition(), settings=settings)
    timeouts = [e for e in captured if e["event"] == "exit_advisor.position_protected_timeout"]
    assert len(timeouts) == 1


# ---------------------------------------------------------------------------
# Three-state response logging
# ---------------------------------------------------------------------------


def test_skipped_event_logged_when_log_skipped_events_true() -> None:
    """Skipped (advisor returned default ``AdvisorResponse()``) → event_skipped log."""
    register_exit_advisor(_RecordingAdvisor(AdvisorResponse()))
    settings = _make_settings(enabled=True, log_skipped_events=True)
    with capture_logs() as captured:
        notify_event(_StubPosition(), _make_bar_event(), settings=settings)
    events = [e["event"] for e in captured]
    assert "exit_advisor.event_skipped" in events


def test_skipped_event_suppressed_when_log_skipped_events_false() -> None:
    """``log_skipped_events=false`` suppresses the high-volume skipped log."""
    register_exit_advisor(_RecordingAdvisor(AdvisorResponse()))
    settings = _make_settings(enabled=True, log_skipped_events=False)
    with capture_logs() as captured:
        notify_event(_StubPosition(), _make_bar_event(), settings=settings)
    events = [e["event"] for e in captured]
    assert "exit_advisor.event_skipped" not in events


def test_held_event_logged_distinctly_from_skipped() -> None:
    """held (evaluation_performed=True, no recommendation) → event_held log."""
    register_exit_advisor(
        _RecordingAdvisor(AdvisorResponse(evaluation_performed=True, reasoning="hold runner"))
    )
    settings = _make_settings(enabled=True, log_skipped_events=False)
    with capture_logs() as captured:
        notify_event(_StubPosition(), _make_bar_event(), settings=settings)
    events = [e["event"] for e in captured]
    assert "exit_advisor.event_held" in events
    assert "exit_advisor.event_skipped" not in events


def test_actionable_event_logged_with_recommendation_fields() -> None:
    """actionable → event_actionable log with action + reasoning + source."""
    rec = ExitRecommendation(
        action="exit_full",
        reason="9ema break",
        source="advisor_v1",
        confidence=0.8,
    )
    register_exit_advisor(
        _RecordingAdvisor(
            AdvisorResponse(recommendation=rec, evaluation_performed=True, reasoning="...")
        )
    )
    settings = _make_settings(enabled=True)
    with capture_logs() as captured:
        notify_event(_StubPosition(), _make_bar_event(), settings=settings)
    actionable = [e for e in captured if e["event"] == "exit_advisor.event_actionable"]
    assert len(actionable) == 1
    assert actionable[0]["action"] == "exit_full"
    assert actionable[0]["confidence"] == 0.8
    assert actionable[0]["source"] == "advisor_v1"


def test_advisor_returning_bare_none_normalises_to_skipped() -> None:
    """Legacy interface returning ``None`` is normalised to a skipped response."""

    class _BareNoneAdvisor:
        def on_position_protected(self, position: Any) -> None:
            return None

        def on_event(self, position: Any, event: Any) -> Any:
            return None  # legacy / minimal advisor

        def on_position_closed(self, position: Any, final_pnl: float) -> None:
            return None

    register_exit_advisor(_BareNoneAdvisor())
    settings = _make_settings(enabled=True)
    response = notify_event(_StubPosition(), _make_bar_event(), settings=settings)
    assert response.is_skipped


# ---------------------------------------------------------------------------
# Settings smoke (via replace, to avoid YAML coupling)
# ---------------------------------------------------------------------------


def test_settings_can_construct_with_alternative_exit_advisor_block() -> None:
    """Sanity: pydantic model_copy used to flip per-test config actually works."""
    base = _make_settings(enabled=False)
    flipped_block = base.exit_advisor.model_copy(update={"enabled": True})
    flipped = base.model_copy(update={"exit_advisor": flipped_block})
    assert flipped.exit_advisor.enabled is True

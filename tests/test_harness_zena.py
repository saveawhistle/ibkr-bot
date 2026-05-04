"""End-to-end ZENA replay test — the layer 1 correctness oracle."""

from __future__ import annotations

from datetime import UTC, date

import pytest

from bot.config import ExitEventsConfig, Settings
from bot.exit_advisor.core.events import (
    DrawdownFromPeak,
    Event,
    MaxFavorableExcursionUpdate,
    PositionProtected,
    ReplayTerminalTick,
    RMultipleReached,
    TimeOfDayMilestone,
)
from bot.exit_advisor.decision.policy import ActualPolicy, ExitDecision, TradeState
from bot.exit_advisor.replay.harness import TradeReplayHarness
from bot.exit_advisor.replay.replay_source import TradeReplayData, load_trade_replay_data

ZENA_DATE = date(2026, 4, 30)


@pytest.fixture(scope="module")
def zena_replay() -> TradeReplayData:
    return load_trade_replay_data("ZENA", ZENA_DATE)


@pytest.fixture(scope="module")
def exit_events_config() -> ExitEventsConfig:
    return Settings().exit_events


def test_zena_actual_policy_replay_matches_recorded_pnl(
    zena_replay: TradeReplayData, exit_events_config: ExitEventsConfig
) -> None:
    """The harness, run with ActualPolicy, must reproduce the recorded P&L
    within $0.01. This is the layer 1 correctness oracle."""
    harness = TradeReplayHarness(zena_replay, ActualPolicy(zena_replay), exit_events_config)
    result = harness.run()

    assert result.exit_price == zena_replay.recorded_exit_price
    assert abs(result.final_pnl - zena_replay.recorded_pnl) < 0.01, (
        f"replay PnL {result.final_pnl} drifted from recorded {zena_replay.recorded_pnl}"
    )


def test_zena_position_protected_fires_after_bracket(
    zena_replay: TradeReplayData, exit_events_config: ExitEventsConfig
) -> None:
    """PositionProtected fires exactly once and not before the bracket
    placement timestamp."""
    harness = TradeReplayHarness(zena_replay, ActualPolicy(zena_replay), exit_events_config)
    result = harness.run()

    protected = [e for e in result.events_emitted if isinstance(e, PositionProtected)]
    assert len(protected) == 1
    bracket_ts_str = zena_replay.bracket_event["timestamp"].rstrip("Z")
    from datetime import datetime

    bracket_ts = datetime.fromisoformat(bracket_ts_str + "+00:00").astimezone(UTC)
    assert protected[0].timestamp >= bracket_ts


class _AlwaysExitPolicy:
    """Returns exit_full unconditionally on every event. Used to verify that
    the harness suppresses policy invocation before PositionProtected fires."""

    def __init__(self) -> None:
        self.events_seen: list[Event] = []

    def on_event(
        self, trade_state: TradeState, event: Event
    ) -> ExitDecision | None:
        self.events_seen.append(event)
        return ExitDecision(action="exit_full", reason="AlwaysExit")


def test_zena_advisor_suppressed_before_protection(
    zena_replay: TradeReplayData, exit_events_config: ExitEventsConfig
) -> None:
    """No event the policy receives may have a timestamp earlier than the
    PositionProtected event — the harness gates ``on_event`` on
    ``state.is_protected``, which only flips True at protection time."""
    policy = _AlwaysExitPolicy()
    harness = TradeReplayHarness(zena_replay, policy, exit_events_config)
    result = harness.run()

    protected = [e for e in result.events_emitted if isinstance(e, PositionProtected)]
    assert len(protected) == 1
    protection_ts = protected[0].timestamp

    # The very first event the policy sees must have ts >= protection_ts.
    # Any pre-protection event (e.g. an OrderRejection earlier in the
    # day) is logged but never reaches on_event.
    for ev in policy.events_seen:
        assert ev.timestamp >= protection_ts, (
            f"policy was invoked on pre-protection event {type(ev).__name__} at {ev.timestamp}"
        )


def test_zena_pnl_events_fire(
    zena_replay: TradeReplayData, exit_events_config: ExitEventsConfig
) -> None:
    """ZENA's risk = $0.02 (entry 2.18, stop 2.16). Bar low 2.17 = -0.5R,
    which must trigger R=0.5 down. Time milestones 5 + 30 minutes after
    09:30 ET fire on the entry bar (which closes at 11:03 ET). MFE never
    fires because the bar high never exceeds entry. Drawdown never fires
    because peak_R never goes positive."""
    harness = TradeReplayHarness(zena_replay, ActualPolicy(zena_replay), exit_events_config)
    result = harness.run()

    r_events = [e for e in result.events_emitted if isinstance(e, RMultipleReached)]
    assert any(e.r_multiple == 0.5 and e.direction == "down" for e in r_events)

    tod_milestones = {
        e.minutes_after_open
        for e in result.events_emitted
        if isinstance(e, TimeOfDayMilestone)
    }
    assert 5 in tod_milestones
    assert 30 in tod_milestones
    assert 120 not in tod_milestones

    mfe = [e for e in result.events_emitted if isinstance(e, MaxFavorableExcursionUpdate)]
    assert mfe == []

    drawdown = [e for e in result.events_emitted if isinstance(e, DrawdownFromPeak)]
    assert drawdown == []

    # Terminal tick must be present so ActualPolicy could fire.
    terminal = [e for e in result.events_emitted if isinstance(e, ReplayTerminalTick)]
    assert len(terminal) == 1

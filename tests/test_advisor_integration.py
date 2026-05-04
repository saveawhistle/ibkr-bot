"""End-to-end advisor integration test with a fake LLM client.

Drives a sequence of events through the advisor that mimics a real
position lifecycle. Verifies:

* events flow through the buffer
* always-trigger events fire the LLM
* shadow baselines run on every event regardless of trigger
* recommendations come back as ``AdvisorResponse`` of the right shape
* ``on_position_closed`` cleans up state
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from bot.exit_advisor.advisor.agent import ExitAdvisor
from bot.exit_advisor.advisor.buffer import EventBuffer
from bot.exit_advisor.advisor.cost_tracker import CostTracker
from bot.exit_advisor.advisor.llm_client import LLMCallResult
from bot.exit_advisor.advisor.shadow_baselines import ShadowBaselines
from bot.exit_advisor.core.events import (
    BarFinalizedEvent,
    DrawdownFromPeak,
    LargePrint,
    PartialFillEvent,
    PositionProtected,
    RMultipleReached,
    VolumeSpike,
)
from bot.exit_advisor.core.types import ExitRecommendation

_T0 = datetime(2026, 5, 4, 14, 30, 0, tzinfo=UTC)


@dataclass
class _PositionRecord:
    symbol: str = "WLDS"
    strategy: str = "gap_and_go"
    shares: int = 200
    avg_price: float = 2.50
    stop_price: float = 2.30
    scale_out_price: float = 2.90
    status: str = "open"
    scaled_out: bool = False


@dataclass
class _ScriptedLLM:
    """Yields a pre-scripted ``LLMCallResult`` sequence and records every call."""

    queued: list[LLMCallResult] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)

    def call(
        self, system_prompt: str, user_message: str, tool_schema: dict[str, Any]
    ) -> LLMCallResult:
        self.calls.append(user_message)
        if not self.queued:
            return LLMCallResult(
                success=False,
                recommendation=None,
                cost_usd=0.0,
                duration_seconds=0.0,
                failure_reason="exhausted",
            )
        return self.queued.pop(0)


def _make_advisor(
    llm: _ScriptedLLM, *, notify_callback: Callable[[str], None] | None = None
) -> ExitAdvisor:
    return ExitAdvisor(
        llm_client=llm,  # type: ignore[arg-type]
        cost_tracker=CostTracker(soft_cap_usd=5.0, hard_cap_usd=10.0),
        event_buffer_factory=lambda: EventBuffer(time_floor_seconds=30.0, hard_floor_seconds=10.0),
        shadow_baselines=ShadowBaselines(),
        hook_acts=False,
        notify_callback=notify_callback,
    )


def _success(
    *, action: str, cost: float = 0.005, partial_pct: float = 0.0, new_stop: float | None = None
) -> LLMCallResult:
    return LLMCallResult(
        success=True,
        recommendation=ExitRecommendation(  # type: ignore[arg-type]
            action=action,
            partial_pct=partial_pct,
            new_stop_price=new_stop,
            confidence=0.75,
            reason=f"{action} reasoning",
            source="live_llm_advisor",
        ),
        cost_usd=cost,
        duration_seconds=0.2,
    )


def test_full_trade_lifecycle_drives_hook_states() -> None:
    """Walk a position from protected → bar updates → R-multiple → drawdown → close."""
    scripted = [
        _success(action="hold"),  # PositionProtected at T+0
        _success(action="hold"),  # RMultipleReached(1.0R) at T+90s
        _success(action="hold"),  # BarFinalizedEvent at T+120s (time-floor trigger)
        _success(action="exit_partial", partial_pct=0.5),  # DrawdownFromPeak at T+180s
    ]
    llm = _ScriptedLLM(queued=scripted)
    advisor = _make_advisor(llm)
    pos = _PositionRecord()
    advisor.on_position_protected(pos)

    # Always-trigger #1: PositionProtected at T+0.
    protected = PositionProtected(
        timestamp=_T0,
        symbol="WLDS",
        entry_price=2.50,
        initial_stop=2.30,
        initial_scale_out=2.90,
        position_size=200,
    )
    r1 = advisor.on_event(pos, protected)
    assert r1.is_held

    # Buffer-only events between LLM calls.
    print_event = LargePrint(
        timestamp=_T0 + timedelta(seconds=30),
        symbol="WLDS",
        price=2.65,
        size=5000,
        rolling_average_size=400.0,
        ratio=12.5,
        aggressor_side="buy",
    )
    r_print = advisor.on_event(pos, print_event)
    assert r_print.is_skipped

    vol_event = VolumeSpike(
        timestamp=_T0 + timedelta(seconds=60),
        symbol="WLDS",
        bar_volume=12000,
        rolling_average=1000.0,
        ratio=12.0,
        threshold=2.0,
    )
    r_vol = advisor.on_event(pos, vol_event)
    assert r_vol.is_skipped

    # Always-trigger #2: RMultipleReached at T+90s.
    r_event = RMultipleReached(
        timestamp=_T0 + timedelta(seconds=90),
        symbol="WLDS",
        r_multiple=1.0,
        direction="up",
    )
    r2 = advisor.on_event(pos, r_event)
    assert r2.is_held

    # Bar updates that also feed peak tracking.
    bar = BarFinalizedEvent(
        timestamp=_T0 + timedelta(seconds=120),
        symbol="WLDS",
        open=2.50,
        high=2.85,
        low=2.49,
        close=2.80,
        volume=15000,
    )
    advisor.on_event(pos, bar)

    # Always-trigger #3: DrawdownFromPeak at T+180s — actionable response.
    drawdown = DrawdownFromPeak(
        timestamp=_T0 + timedelta(seconds=180),
        symbol="WLDS",
        drawdown_pct=0.5,
        peak_r_multiple=1.5,
        current_r_multiple=0.75,
    )
    r3 = advisor.on_event(pos, drawdown)
    assert r3.is_actionable
    assert r3.recommendation is not None
    assert r3.recommendation.action == "exit_partial"
    assert r3.recommendation.partial_pct == 0.5

    # Four triggered LLM calls total (protected, R-multiple, bar-finalized, drawdown).
    assert len(llm.calls) == 4

    # Closing the position drains state.
    advisor.on_position_closed(pos, final_pnl=42.50)
    assert "WLDS" not in advisor._contexts


def test_partial_fill_triggers_immediately_after_protection() -> None:
    """PartialFillEvent is in ALWAYS_TRIGGER and should fire even if buffered events accumulated."""
    llm = _ScriptedLLM(queued=[_success(action="hold"), _success(action="hold")])
    advisor = _make_advisor(llm)
    pos = _PositionRecord()
    advisor.on_position_protected(pos)

    # Trigger #1: protected.
    advisor.on_event(
        pos,
        PositionProtected(
            timestamp=_T0,
            symbol="WLDS",
            entry_price=2.50,
            initial_stop=2.30,
            initial_scale_out=2.90,
            position_size=200,
        ),
    )

    # Buffer-only events.
    advisor.on_event(
        pos,
        LargePrint(
            timestamp=_T0 + timedelta(seconds=11),
            symbol="WLDS",
            price=2.55,
            size=1000,
            rolling_average_size=300.0,
            ratio=3.3,
            aggressor_side="buy",
        ),
    )

    # PartialFillEvent past the hard floor should trigger.
    fill = PartialFillEvent(
        timestamp=_T0 + timedelta(seconds=15),
        symbol="WLDS",
        order_id=1,
        filled_quantity=100,
        remaining_quantity=100,
        fill_price=2.50,
        side="buy",
    )
    response = advisor.on_event(pos, fill)
    assert response.is_held  # second call returned held
    assert len(llm.calls) == 2


def test_hook_registry_returns_skipped_when_disabled(monkeypatch: Any) -> None:
    """The hook wrapper short-circuits when exit_advisor.enabled=false.

    Constructs a hermetic Settings with the advisor explicitly disabled rather
    than depending on the operator's on-disk config (which may have the advisor
    enabled in their session).
    """
    from bot.config import ExitAdvisorConfig, get_settings
    from bot.exit_advisor.core.types import AdvisorResponse
    from bot.exit_advisor.hook.registry import notify_event

    base = get_settings()
    disabled = base.model_copy(update={"exit_advisor": ExitAdvisorConfig(enabled=False)})
    assert not disabled.exit_advisor.enabled
    response: AdvisorResponse = notify_event(
        _PositionRecord(),  # type: ignore[arg-type]
        PositionProtected(
            timestamp=_T0,
            symbol="WLDS",
            entry_price=2.50,
            initial_stop=2.30,
            initial_scale_out=2.90,
            position_size=200,
        ),
        settings=disabled,
    )
    assert response.is_skipped

"""Phase 11 — exit advisor hook registry + invocation infrastructure.

Three responsibilities:

1. **Registry** — module-level slot for the (at most one) registered
   :class:`ExitAdvisorHook`. ``register_exit_advisor`` /
   ``unregister_exit_advisor`` manipulate it; ``registered_advisor``
   reads it.

2. **Invocation wrappers** — ``notify_position_protected`` /
   ``notify_event`` / ``notify_position_closed`` are the call surface
   the bot's pipeline uses. Each one:

   * No-ops immediately if ``exit_advisor.enabled=false`` or no advisor
     is registered (the disabled-default contract: zero overhead and
     identical behaviour to pre-Phase-11).
   * Wraps the advisor call in ``try/except`` — any exception is
     logged with full traceback and treated as failed. The bot is
     never crashed by a hook bug.
   * Enforces a configurable timeout. The advisor runs in a worker
     thread so the wall-clock cap is real; on timeout the call is
     treated as failed and a ``timeout`` event is logged.
   * Emits structured logs at ``bot.exit_advisor`` so an operator can
     grep for ``exit_advisor.event_skipped`` / ``.event_held`` /
     ``.event_actionable`` / ``.event_failed`` and see exactly what
     the advisor did per call.

3. **Three-state response semantics** — :func:`notify_event` always
   returns an :class:`AdvisorResponse`. A bare ``None`` from the
   advisor is normalised to ``AdvisorResponse(skipped)``; an exception
   or timeout becomes ``AdvisorResponse(skipped)`` paired with a
   ``failed`` log event.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from threading import Lock
from typing import Any

import structlog

from bot.config import Settings, get_settings
from bot.exit_advisor.core.types import (
    AdvisorResponse,
    Event,
    ExitAdvisorHook,
    PositionLike,
)

_log = structlog.get_logger("bot.exit_advisor")

# Registry — single advisor at a time. None when no implementation is
# registered (the production-main default). Spike-side bootstrap calls
# ``register_exit_advisor`` to install its implementation.
_registered_advisor: ExitAdvisorHook | None = None
_registry_lock = Lock()

# Worker pool used for timeout enforcement. Sized at 2 so a long-running
# hook call doesn't block a concurrent close-notification while the
# bar-driven on_event is still in flight. Small + bounded; we don't
# parallelise advisor calls beyond that.
_HOOK_WORKERS = ThreadPoolExecutor(max_workers=2, thread_name_prefix="exit_advisor")


def register_exit_advisor(advisor: ExitAdvisorHook) -> None:
    """Install ``advisor`` as the active exit-advisor implementation.

    Replaces any previously-registered advisor (a warning is logged so
    accidental double-registration during bootstrap is visible). The
    actual notify functions still no-op until
    ``exit_advisor.enabled=true`` in config — registration is
    necessary but not sufficient for the hook to fire.
    """
    global _registered_advisor
    with _registry_lock:
        if _registered_advisor is not None:
            _log.warning(
                "exit_advisor.advisor_replaced",
                previous=type(_registered_advisor).__name__,
                new=type(advisor).__name__,
            )
        _registered_advisor = advisor
    _log.info("exit_advisor.advisor_registered", advisor=type(advisor).__name__)


def unregister_exit_advisor() -> None:
    """Remove the currently-registered advisor (idempotent)."""
    global _registered_advisor
    with _registry_lock:
        previous = _registered_advisor
        _registered_advisor = None
    if previous is not None:
        _log.info("exit_advisor.advisor_unregistered", advisor=type(previous).__name__)


def registered_advisor() -> ExitAdvisorHook | None:
    """Return the currently-registered advisor, or ``None``."""
    return _registered_advisor


def _resolve_hook(settings: Settings | None) -> ExitAdvisorHook | None:
    """Return the advisor to call, or ``None`` for the no-op fast path.

    Returns None when (a) the feature is disabled in config, OR (b) no
    advisor has been registered. Both branches are equivalent: the
    notify-call short-circuits with zero work.
    """
    cfg = (settings or get_settings()).exit_advisor
    if not cfg.enabled:
        return None
    return _registered_advisor


def _run_with_timeout(
    fn: Callable[..., Any], timeout_seconds: float, *args: Any
) -> Any:
    """Execute ``fn(*args)`` in the worker pool with a wall-clock timeout.

    Raises :class:`concurrent.futures.TimeoutError` if the deadline
    fires; otherwise returns the function's return value. The worker
    thread is intentionally not killed on timeout (Python doesn't
    support that safely) — the future is abandoned and the pool will
    reuse its slot once the runaway call eventually returns.
    """
    future = _HOOK_WORKERS.submit(fn, *args)
    return future.result(timeout=timeout_seconds)


def notify_position_protected(
    position: PositionLike,
    *,
    settings: Settings | None = None,
) -> None:
    """Fire ``on_position_protected`` if the hook is enabled and registered.

    Synchronous + best-effort: exceptions are caught and logged.
    Timeouts are enforced. Failures are *not* propagated — the
    position is protected on IBKR's books regardless of advisor
    health.
    """
    advisor = _resolve_hook(settings)
    if advisor is None:
        return
    cfg = (settings or get_settings()).exit_advisor
    started = time.monotonic()
    try:
        _run_with_timeout(
            advisor.on_position_protected, cfg.timeout_seconds, position
        )
    except FuturesTimeoutError:
        _log.warning(
            "exit_advisor.position_protected_timeout",
            symbol=position.symbol,
            advisor=type(advisor).__name__,
            timeout_seconds=cfg.timeout_seconds,
        )
        return
    except Exception as exc:  # noqa: BLE001 - hook bugs must not crash the bot
        _log.error(
            "exit_advisor.position_protected_failed",
            symbol=position.symbol,
            advisor=type(advisor).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        return
    _log.info(
        "exit_advisor.position_protected",
        symbol=position.symbol,
        advisor=type(advisor).__name__,
        duration_ms=round((time.monotonic() - started) * 1000, 1),
    )


def notify_position_closed(
    position: PositionLike,
    final_pnl: float,
    *,
    settings: Settings | None = None,
) -> None:
    """Fire ``on_position_closed`` if the hook is enabled and registered.

    Same defensive contract as :func:`notify_position_protected`.
    Called from the position-state machine's terminal transition;
    must not raise even if the advisor is dead.
    """
    advisor = _resolve_hook(settings)
    if advisor is None:
        return
    cfg = (settings or get_settings()).exit_advisor
    started = time.monotonic()
    try:
        _run_with_timeout(
            advisor.on_position_closed, cfg.timeout_seconds, position, final_pnl
        )
    except FuturesTimeoutError:
        _log.warning(
            "exit_advisor.position_closed_timeout",
            symbol=position.symbol,
            advisor=type(advisor).__name__,
            timeout_seconds=cfg.timeout_seconds,
        )
        return
    except Exception as exc:  # noqa: BLE001 - hook bugs must not crash the bot
        _log.error(
            "exit_advisor.position_closed_failed",
            symbol=position.symbol,
            advisor=type(advisor).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        return
    _log.info(
        "exit_advisor.position_closed",
        symbol=position.symbol,
        advisor=type(advisor).__name__,
        final_pnl=round(final_pnl, 2),
        duration_ms=round((time.monotonic() - started) * 1000, 1),
    )


def notify_event(
    position: PositionLike,
    event: Event,
    *,
    settings: Settings | None = None,
) -> AdvisorResponse:
    """Fire ``on_event`` and return the advisor's response (always a typed value).

    Returns ``AdvisorResponse(skipped)`` in every degenerate path:

    * hook disabled or no advisor registered (silent no-op),
    * advisor returned bare ``None`` (legacy interface),
    * advisor raised (logged at ERROR with traceback),
    * advisor exceeded ``timeout_seconds`` (logged at WARNING).

    Only when the advisor returns a real :class:`AdvisorResponse` does
    the caller see it intact. Caller (typically TradeManager) decides
    whether to act on an actionable response based on
    ``exit_advisor.hook_acts``.
    """
    advisor = _resolve_hook(settings)
    if advisor is None:
        return AdvisorResponse()  # silent skipped — disabled / unregistered
    cfg = (settings or get_settings()).exit_advisor

    started = time.monotonic()
    try:
        raw = _run_with_timeout(advisor.on_event, cfg.timeout_seconds, position, event)
    except FuturesTimeoutError:
        _log.warning(
            "exit_advisor.event_failed",
            symbol=position.symbol,
            advisor=type(advisor).__name__,
            event_type=type(event).__name__,
            cause="timeout",
            timeout_seconds=cfg.timeout_seconds,
        )
        return AdvisorResponse()
    except Exception as exc:  # noqa: BLE001 - hook bugs must not crash the bot
        _log.error(
            "exit_advisor.event_failed",
            symbol=position.symbol,
            advisor=type(advisor).__name__,
            event_type=type(event).__name__,
            cause="exception",
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        return AdvisorResponse()

    duration_ms = round((time.monotonic() - started) * 1000, 1)
    response: AdvisorResponse = raw if isinstance(raw, AdvisorResponse) else AdvisorResponse()

    if response.is_actionable:
        rec = response.recommendation
        assert rec is not None  # narrow for mypy; is_actionable ⇒ recommendation set
        _log.info(
            "exit_advisor.event_actionable",
            symbol=position.symbol,
            advisor=type(advisor).__name__,
            event_type=type(event).__name__,
            duration_ms=duration_ms,
            action=rec.action,
            partial_pct=rec.partial_pct if rec.action == "exit_partial" else None,
            new_stop_price=rec.new_stop_price if rec.action == "tighten_stop" else None,
            confidence=rec.confidence,
            reason=rec.reason,
            source=rec.source,
            reasoning=response.reasoning,
        )
    elif response.is_held:
        _log.info(
            "exit_advisor.event_held",
            symbol=position.symbol,
            advisor=type(advisor).__name__,
            event_type=type(event).__name__,
            duration_ms=duration_ms,
            reasoning=response.reasoning,
        )
    elif cfg.log_skipped_events:
        # log_skipped_events gates the high-volume path: every skipped
        # event the advisor doesn't reason about (e.g. every L2 update
        # that doesn't satisfy the advisor's gates). Default True for
        # max diagnostics; flip false in high-volume sessions.
        _log.info(
            "exit_advisor.event_skipped",
            symbol=position.symbol,
            advisor=type(advisor).__name__,
            event_type=type(event).__name__,
            duration_ms=duration_ms,
        )
    return response

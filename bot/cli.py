"""Typer CLI: Phase 1–4b commands — ``ping``, ``scan``, ``watch``, ``trade``,
``backtest``, ``flatten``, ``positions``, ``status``, ``reset-halt``."""

from __future__ import annotations

import asyncio
import contextlib
import json
import signal
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from datetime import date as date_cls
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import structlog
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from bot.backtest import (
    STRATEGY_BOTH,
    STRATEGY_GAP_AND_GO,
    STRATEGY_MOMENTUM,
    Replayer,
    ReplayError,
)
from bot.brokerage.ibkr_client import IBKRClient
from bot.brokerage.market_data import MarketData
from bot.config import (
    AccountConfig,
    ConfigurationError,
    ExecutionConfig,
    Settings,
    get_settings,
    warn_on_missing_data_source_credentials,
)
from bot.execution.executor import Executor
from bot.execution.position_state import Position, PositionStore
from bot.execution.trade_manager import TradeManager
from bot.execution.watchdog import Watchdog
from bot.exit_advisor.advisor import bootstrap_advisor
from bot.logging_setup import configure_logging, resolve_session_log_path
from bot.notify import Notifier
from bot.orchestrator import Orchestrator, run_strategy_loop
from bot.persistence.journal import Journal
from bot.risk import RiskEngine, delete_halt_flag, read_halt_flag
from bot.risk.rehab import (
    RehabEngine,
    RehabTier,
    aggregate_daily_pnl,
    classify_tier,
    read_rehab_flag,
)
from bot.scanning.catalyst import VALID_CATEGORIES
from bot.scanning.catalyst_overrides import (
    DEFAULT_STORE_PATH as _OVERRIDES_DEFAULT_PATH,
)
from bot.scanning.catalyst_overrides import (
    CatalystOverride,
    upsert_override,
)
from bot.scanning.finnhub_client import FinnhubClient
from bot.scanning.float_source import (
    SOURCE_FINNHUB_FALLBACK,
    SOURCE_YFINANCE,
    FloatSource,
)
from bot.scanning.scanner import IBKRScanner, ScanHit
from bot.signal_bus import SignalBus
from bot.strategies.base import RejectedCandidate, Signal
from bot.strategies.gap_and_go import GapAndGoStrategy
from bot.strategies.momentum import MomentumStrategy

app = typer.Typer(
    name="ibkr-bot",
    help="Automated day-trading bot.",
    no_args_is_help=True,
)
_console = Console()
_log = structlog.get_logger("bot.cli")

_PING_TAGS = ("NetLiquidation", "TotalCashValue", "BuyingPower", "DayTradesRemaining")


def _install_shutdown_handler(
    event: asyncio.Event,
) -> Callable[[], None]:
    """Install a SIGINT handler that sets ``event`` and returns the uninstaller.

    TWS paper-trading day-1 showed a Ctrl-C during the strategy loop killed
    the process before ``disconnect()`` could sweep subscriptions, leaving
    market-data lines allocated on the TWS side across four restarts. The
    shutdown event lets the strategy loop exit cleanly through its normal
    ``finally`` path, which in turn calls ``IBKRClient.disconnect`` →
    ``cancel_all_subscriptions`` before the socket closes.

    Cross-platform: prefer ``loop.add_signal_handler`` on POSIX (safe w.r.t.
    the asyncio loop state), fall back to ``signal.signal`` on Windows
    where ``add_signal_handler`` raises NotImplementedError. The signal
    handler bounces back into the loop via ``call_soon_threadsafe`` so we
    don't touch asyncio state from the signal context.
    """
    loop = asyncio.get_running_loop()

    def _on_signal() -> None:
        if not event.is_set():
            _log.warning("cli.shutdown_signal_received", signal="SIGINT")
            event.set()

    previous: Any = None
    using_asyncio_handler = False
    try:
        loop.add_signal_handler(signal.SIGINT, _on_signal)
        using_asyncio_handler = True
    except (NotImplementedError, RuntimeError):

        def _threadsafe_handler(_signum: int, _frame: Any) -> None:
            loop.call_soon_threadsafe(_on_signal)

        previous = signal.signal(signal.SIGINT, _threadsafe_handler)

    def uninstall() -> None:
        if using_asyncio_handler:
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.remove_signal_handler(signal.SIGINT)
        else:
            with contextlib.suppress(Exception):
                signal.signal(signal.SIGINT, previous or signal.SIG_DFL)

    return uninstall


@app.callback()
def _main() -> None:
    """Hold the multi-command shape open so more commands can land in later phases."""


@app.command()
def ping() -> None:
    """Connect to the configured IBKR paper account, print an account summary, and disconnect."""
    configure_logging(get_settings())
    _run_with_connection_handling(_ping)


@app.command()
def scan(
    limit: int = typer.Option(10, "--limit", "-n", help="Max number of hits to display/notify."),
    no_notify: bool = typer.Option(False, "--no-notify", help="Skip the Telegram push (dev use)."),
) -> None:
    """Run the morning 5-Pillar scan and print a ranked watchlist (optionally Telegram-push it)."""
    configure_logging(get_settings())
    warn_on_missing_data_source_credentials()
    _run_with_connection_handling(lambda: _scan(limit=limit, notify=not no_notify))


@app.command()
def backtest(
    symbol: str = typer.Argument(..., help="Ticker to replay (e.g. AMC)."),
    date: str = typer.Argument(..., help="Trading date in YYYY-MM-DD format."),
    catalyst: bool = typer.Option(
        False,
        "--catalyst/--no-catalyst",
        help="Force a ``manual_override`` catalyst so Gap-and-Go participates (replay only).",
    ),
    strategy: str = typer.Option(
        STRATEGY_BOTH,
        "--strategy",
        help="Which strategies to replay: gap_and_go | momentum | both.",
    ),
    max_rejections: int = typer.Option(
        25,
        "--max-rejections",
        help="Max rejection rows to render in the table; full list still goes to JSONL.",
    ),
) -> None:
    """Replay one ticker against one historical date and print every signal the strategies fire."""
    configure_logging(get_settings())
    try:
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError as exc:
        typer.echo(f"Invalid date {date!r}: expected YYYY-MM-DD.", err=True)
        raise typer.Exit(code=1) from exc
    if strategy not in (STRATEGY_GAP_AND_GO, STRATEGY_MOMENTUM, STRATEGY_BOTH):
        typer.echo(
            f"Invalid --strategy {strategy!r}: must be gap_and_go | momentum | both.",
            err=True,
        )
        raise typer.Exit(code=1)
    if max_rejections < 0:
        typer.echo("--max-rejections must be non-negative.", err=True)
        raise typer.Exit(code=1)
    _run_with_connection_handling(
        lambda: _backtest(
            symbol=symbol.upper(),
            target_date=target_date,
            force_catalyst=catalyst,
            strategy_selection=strategy,
            max_rejections=max_rejections,
        )
    )


@app.command()
def watch(
    duration: float | None = typer.Option(
        None,
        "--duration",
        "-d",
        help=(
            "Minutes to keep the strategy loop alive. If omitted, runs until the "
            "configured session.flatten_all time (minus a 60s safety buffer)."
        ),
    ),
    limit: int = typer.Option(
        10, "--limit", "-n", help="Max watchlist symbols fed into the strategy loop."
    ),
    poll_interval: float = typer.Option(
        5.0, "--poll-interval", help="Seconds between evaluation passes."
    ),
    no_notify: bool = typer.Option(False, "--no-notify", help="Skip Telegram push for signals."),
) -> None:
    """Scan once, then subscribe to the watchlist's live 1-min bars and print detected signals."""
    configure_logging(get_settings())
    warn_on_missing_data_source_credentials()
    _run_with_connection_handling(
        lambda: _watch(
            duration=duration,
            limit=limit,
            poll_interval=poll_interval,
            notify=not no_notify,
        )
    )


@app.command()
def trade(
    duration: float | None = typer.Option(
        None,
        "--duration",
        "-d",
        help=(
            "Minutes to keep the trading loop alive. If omitted, runs until the "
            "configured session.flatten_all time (minus a 60s safety buffer)."
        ),
    ),
    limit: int = typer.Option(
        10, "--limit", "-n", help="Max watchlist symbols fed into the trading loop."
    ),
    poll_interval: float = typer.Option(
        5.0, "--poll-interval", help="Seconds between evaluation passes."
    ),
    no_notify: bool = typer.Option(
        False, "--no-notify", help="Skip Telegram push for signals and fills."
    ),
    dry_run_signal: str = typer.Option(
        "",
        "--dry-run-signal",
        help=(
            "Fabricate one signal ``SYMBOL:entry:stop:scale_out:strategy`` and run it "
            "through the executor without the scanner/strategy loop. "
            "The 4th field is the strategy's +1R scale-out; the bracket's "
            "runner-target LMT is derived from execution.runner_target_multiple. "
            "Useful for smoke-testing bracket placement end-to-end."
        ),
    ),
    live: bool = typer.Option(
        False,
        "--live",
        help=(
            "Enable live trading against the REAL MONEY account. "
            "Requires typing the literal word CONFIRM at the prompt."
        ),
    ),
    simulate_loss_usd: float = typer.Option(
        0.0,
        "--simulate-loss-usd",
        help=(
            "Paper-only. Preload the RiskEngine with this much realized loss "
            "(positive number → treated as negative PnL). Demonstrates halt behavior."
        ),
    ),
    simulate_time: str = typer.Option(
        "",
        "--simulate-time",
        help=(
            "Paper-only. ``HH:MM`` NY-local. Fire the 15:55 auto-flatten job "
            "one minute after this time. Demonstrates flatten behavior."
        ),
    ),
    simulate_reentry: int = typer.Option(
        0,
        "--simulate-reentry",
        min=0,
        max=10,
        help=(
            "Paper-only. Seed the --dry-run-signal symbol with N prior profitable "
            "entries so the next signal demonstrates the N+1th re-entry's size "
            "multiplier. Requires --dry-run-signal."
        ),
    ),
    simulate_config_override: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--simulate-config-override",
        help=(
            "Paper-only. ``execution.<field>=<value>`` — temporarily override one "
            "execution-config field for this run, e.g. "
            "``runner_target_multiple=5.0``. Repeatable. Useful for showcasing "
            "the Phase 4e runner-target knob without editing config.yaml."
        ),
    ),
    simulate_red_days: int = typer.Option(
        0,
        "--simulate-red-days",
        min=0,
        max=30,
        help=(
            "Paper-only. Seed the Phase 4g RehabEngine with N synthetic "
            "back-to-back losing days (each = -max_daily_loss_usd) so the "
            "session starts in REHAB/DEEP_REHAB. Nothing persists — the "
            "override lives in memory and is dropped on CLI exit."
        ),
    ),
) -> None:
    """Scan + subscribe + place bracket orders with Phase 4b risk + trade management.

    Defaults to paper. ``--live`` gates on an interactive CONFIRM prompt inside a
    red rich.Panel so it's visually unmistakable. ``--simulate-*`` flags are
    paper-only demo hooks; they exit with a clear error on a live account to
    avoid accidentally putting a demo into production.
    """
    configure_logging(get_settings())
    warn_on_missing_data_source_credentials()
    settings = get_settings()
    if live:
        settings = _confirm_live_or_exit(settings)
    if (
        simulate_loss_usd
        or simulate_time
        or simulate_reentry
        or simulate_config_override
        or simulate_red_days
    ) and settings.account.mode == "live":
        typer.echo(
            "--simulate-* flags are paper-only. Refusing to run against a live account.",
            err=True,
        )
        raise typer.Exit(code=1)
    if simulate_reentry and not dry_run_signal:
        typer.echo(
            "--simulate-reentry requires --dry-run-signal (the seeded symbol is the dry-run symbol).",
            err=True,
        )
        raise typer.Exit(code=1)
    if simulate_config_override:
        settings = _apply_simulate_config_override(settings, simulate_config_override)
    # Phase 11 advisor: register the live LLM exit advisor against the hook.
    # No-op when exit_advisor.enabled=false (production-main default), so this
    # call is safe regardless of config state.
    bootstrap_advisor(settings)
    _run_with_connection_handling(
        lambda: _trade(
            duration=duration,
            limit=limit,
            poll_interval=poll_interval,
            notify=not no_notify,
            settings=settings,
            dry_run_signal=dry_run_signal,
            simulate_loss_usd=simulate_loss_usd,
            simulate_time=simulate_time,
            simulate_reentry=simulate_reentry,
            simulate_red_days=simulate_red_days,
        )
    )


@app.command()
def status() -> None:
    """Print the current risk + halt state and any active positions.

    Read-only: reconciles with IBKR so the position table reflects the
    broker's authoritative view, then renders the risk-state row and the
    halt-flag contents (if any). Useful as a morning pre-flight check.
    """
    configure_logging(get_settings())
    _run_with_connection_handling(_status)


@app.command("reset-halt")
def reset_halt(
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the confirmation prompt (automation use)."
    ),
) -> None:
    """Delete ``logs/halt.flag`` so new entries are allowed again. Idempotent.

    Deliberate friction — halts are durable across restarts. The flag must
    be cleared *explicitly* (never automatically on a new session) so the
    operator consciously acknowledges the halt before resuming.
    """
    configure_logging(get_settings())
    settings = get_settings()
    path = _halt_flag_path()
    record = read_halt_flag(path)
    if record is None:
        _console.print("[green]No halt flag present.[/green]")
        return
    _console.print(
        f"[yellow]Halt flag found:[/yellow] "
        f"date={record.date.isoformat()} reason={record.reason} "
        f"pnl_at_halt=${record.pnl_at_halt:.2f}"
    )
    if not yes and not typer.confirm("Delete halt flag and allow new entries?"):
        raise typer.Exit(code=1)
    delete_halt_flag(path)
    _ = settings  # reserved for future per-account flag paths
    _console.print("[green]Halt flag cleared.[/green]")


@app.command("rehab-status")
def rehab_status() -> None:
    """Print the active Phase 4g rehab tier + effective caps + recovery target.

    Read-only. Shows the persisted flag (tier, trigger, entered-at,
    drawdown-at-entry), today's computed tier (from the current journal),
    the base vs. tier-adjusted caps, and — when in REHAB/DEEP_REHAB —
    the recovery threshold the operator must earn back to return to
    NORMAL. Journal I/O happens here so a stale flag is visible even if
    the live engine hasn't yet re-evaluated.
    """
    configure_logging(get_settings())
    _run_with_connection_handling(_rehab_status)


@app.command("suggest-caps")
def suggest_caps(
    lookback_days: int = typer.Option(
        30, "--lookback-days", help="Journal window to analyze (default 30 days)."
    ),
    compare: bool = typer.Option(
        False,
        "--compare",
        help="Print suggested caps alongside current config values with deltas.",
    ),
) -> None:
    """Analyze the journal + print suggested risk caps. Advisory only; never writes config.

    Runs read-only against ``logs/trades.db``. Prints a Rich table of
    the observed stats (win rate, avg win/loss, worst day, trades/day)
    and a second table of suggested ``max_loss_per_trade_usd`` /
    ``max_daily_loss_usd`` / ``daily_profit_goal_usd`` /
    ``max_trades_per_day``. With ``--compare``, shows the current
    configured value and the suggested delta side-by-side.

    **Never auto-applies.** "pick your own numbers and
    stick to them." This command is an input to that judgment, not a
    replacement for it.
    """
    configure_logging(get_settings())
    if lookback_days < 1:
        typer.echo("--lookback-days must be >= 1.", err=True)
        raise typer.Exit(code=1)
    asyncio.run(_suggest_caps(lookback_days=lookback_days, compare=compare))


@app.command()
def commissions(
    lookback_days: int = typer.Option(
        14,
        "--lookback-days",
        help="Journal window (default 14 days — matches 2-week paper review).",
    ),
) -> None:
    """Phase 4k — print a commission cost summary across the recent trading window.

    Read-only against ``logs/trades.db``. Rolls up per-leg commissions
    (entry / scale / exit), gross PnL, net PnL, and the ratio of
    commissions to gross profit — the ``% of gross`` column is the
    load-bearing number for live sizing decisions. If the bot is running
    in paper mode the absolute dollars are simulated by IBKR; the
    structural ratios still apply if your paper tier matches the live
    tier you'll trade.
    """
    configure_logging(get_settings())
    if lookback_days < 1:
        typer.echo("--lookback-days must be >= 1.", err=True)
        raise typer.Exit(code=1)
    asyncio.run(_commissions(lookback_days=lookback_days))


@app.command()
def flatten() -> None:
    """Reconcile with IBKR, then flatten every active position via ``executor.flatten_symbol``.

    Last-resort manual kill-switch. Cancels all bracket legs + sends a market
    close for any live shares. Run only when the ``trade`` command is not
    already running (double-cancels on the same order IDs are harmless but
    the command is meant to be used when the bot is offline).
    """
    configure_logging(get_settings())
    _run_with_connection_handling(_flatten)


@app.command()
def positions() -> None:
    """Reconcile with IBKR and print the current active-position table. Read-only."""
    configure_logging(get_settings())
    _run_with_connection_handling(_positions)


@app.command("inject-catalyst")
def inject_catalyst(
    symbol: str = typer.Argument(..., help="Ticker symbol to inject (e.g. AGPU)."),
    category: str = typer.Option(
        ...,
        "--category",
        help=(
            "Catalyst category to inject. Must be one of the classifier's green "
            "buckets: " + ", ".join(sorted(VALID_CATEGORIES)) + "."
        ),
    ),
    duration_hours: float = typer.Option(
        4.0,
        "--duration-hours",
        help="How long the injection stays active. Default 4h. Mutually exclusive with --expires-at.",
    ),
    expires_at: str | None = typer.Option(
        None,
        "--expires-at",
        help="Explicit ISO-8601 expiration timestamp (e.g. 2026-04-22T20:38:00-04:00).",
    ),
    note: str | None = typer.Option(
        None,
        "--note",
        help="Free-form description for forensic clarity (e.g. 'Trump psychedelic EO').",
    ),
) -> None:
    """PAPER-TRADING ONLY: inject a catalyst for a symbol so the scanner skips Finnhub.

    Day 1 paper trading missed ENVB's 92% gap on the Trump psychedelic
    executive order because the classifier's keyword matchers can't see
    sector-wide policy catalysts. This command lets the operator tell
    the bot "yes, this symbol has a catalyst, proceed with normal
    evaluation" — the scanner then applies the injection instead of
    calling Finnhub for that symbol on the next scan.

    SAFETY: gated by ``testing.allow_catalyst_overrides`` in config.yaml.
    The flag MUST stay ``false`` in any live-trading config. If the flag
    is off, this command exits non-zero and writes nothing.

    Overrides auto-expire at ``expires_at`` (default: now + 4 hours) so
    a forgotten injection can't leak into future sessions. Re-injecting
    the same symbol replaces (not appends) the prior entry.
    """
    settings = get_settings()
    configure_logging(settings)

    if not settings.testing.allow_catalyst_overrides:
        typer.echo(
            "ERROR: Manual catalyst injection is disabled.\n"
            "testing.allow_catalyst_overrides is false in config.yaml\n"
            "Set to true in PAPER TRADING configs only. NEVER enable in live.",
            err=True,
        )
        raise typer.Exit(code=1)

    if category not in VALID_CATEGORIES:
        typer.echo(
            f"ERROR: unknown category {category!r}. "
            f"Must be one of: {', '.join(sorted(VALID_CATEGORIES))}.",
            err=True,
        )
        raise typer.Exit(code=1)

    now_utc = datetime.now(UTC)
    ny_tz = ZoneInfo(settings.session.timezone)
    if expires_at is not None:
        try:
            expires_dt = datetime.fromisoformat(expires_at)
        except ValueError as exc:
            typer.echo(
                f"ERROR: --expires-at must be ISO-8601, got {expires_at!r}: {exc}",
                err=True,
            )
            raise typer.Exit(code=1) from exc
        if expires_dt.tzinfo is None:
            expires_dt = expires_dt.replace(tzinfo=ny_tz)
    else:
        if duration_hours <= 0:
            typer.echo(f"ERROR: --duration-hours must be > 0, got {duration_hours}.", err=True)
            raise typer.Exit(code=1)
        expires_dt = now_utc + timedelta(hours=duration_hours)

    if expires_dt <= now_utc:
        typer.echo(
            f"ERROR: expiration {expires_dt.isoformat()} is in the past; "
            "an already-expired injection would never be applied.",
            err=True,
        )
        raise typer.Exit(code=1)

    override = CatalystOverride(
        symbol=symbol.upper(),
        category=category,
        expires_at=expires_dt,
        note=note,
        injected_at=now_utc,
        injected_by="cli",
    )
    upsert_override(override, _OVERRIDES_DEFAULT_PATH)

    _log.info(
        "catalyst.manual_override_injected",
        symbol=override.symbol,
        category=override.category,
        expires_at=override.expires_at.isoformat(),
        injected_at=override.injected_at.isoformat(),
        note=override.note,
        injected_by=override.injected_by,
    )

    expires_local = expires_dt.astimezone(ny_tz)
    _console.print(
        f"[green]Catalyst injected:[/green] [bold]{override.symbol}[/bold] "
        f"([cyan]{override.category}[/cyan])"
    )
    if note:
        _console.print(f"[dim]Note:[/dim] {note}")
    _console.print(
        f"[dim]Expires:[/dim] {expires_local.strftime('%Y-%m-%d %H:%M %Z')} "
        f"({_format_duration(expires_dt - now_utc)} from now)"
    )
    _console.print(f"[dim]Store:[/dim] {_OVERRIDES_DEFAULT_PATH}")


def _format_duration(delta: timedelta) -> str:
    """Render a ``timedelta`` as a compact human string (``Xh Ym``) for injection confirm."""
    total_minutes = max(int(delta.total_seconds() // 60), 0)
    hours, minutes = divmod(total_minutes, 60)
    if hours and minutes:
        return f"{hours}h {minutes}m"
    if hours:
        return f"{hours}h"
    return f"{minutes}m"


@app.command("force-entry")
def force_entry(
    symbol: str = typer.Argument(..., help="Ticker symbol (e.g. AGPU). Case-insensitive."),
    entry: float = typer.Option(
        ..., "--entry", help="Signal entry price (strategy-emit equivalent)."
    ),
    stop: float = typer.Option(
        ..., "--stop", help="Structural stop price (strategy-chosen, below entry)."
    ),
    scale_out: float | None = typer.Option(
        None,
        "--scale-out",
        help="First-target scale-out price. Defaults to entry + 2R (the 2:1 R:R rule).",
    ),
    strategy_name: str = typer.Option(
        "manual_test",
        "--strategy",
        help="Strategy tag for the synthesized Signal (labels the position + journal).",
    ),
) -> None:
    """PAPER-TRADING ONLY: synthesize a Signal and place it through the executor.

    Bypasses scanner + strategy evaluation — builds a Signal with the
    supplied prices and hands it to ``Executor.handle_signal``. Exercises
    the full placement path: risk engine gates (halt, PDT, sizing,
    max_stop_width), order construction (LMT / STP_LMT / MKT per config),
    parent fill event, protection-children planting, journal write.

    Useful for end-to-end validation of Phase 6.12 MKT entries,
    Phase 6.9 tick rounding, and post-fill protection without waiting
    for a live breakout. Not useful for exit testing — TradeManager
    isn't wired here, so the scale-out + trailing stop won't fire from
    this command. Start ``bot trade`` afterward to pick up the open
    position via reconcile.

    SAFETY: double-gated. Requires BOTH
    ``testing.allow_force_entry: true`` in config.yaml AND
    ``account.mode: paper``. Either false/live kills the command.
    """
    settings = get_settings()
    configure_logging(settings)

    if not settings.testing.allow_force_entry:
        typer.echo(
            "ERROR: force-entry is disabled.\n"
            "testing.allow_force_entry is false in config.yaml\n"
            "Set to true in PAPER TRADING configs only. NEVER enable in live.",
            err=True,
        )
        raise typer.Exit(code=1)

    if settings.account.mode != "paper":
        typer.echo(
            "ERROR: force-entry is forbidden in live mode.\n"
            f"account.mode is {settings.account.mode!r}. This command only runs with 'paper'.",
            err=True,
        )
        raise typer.Exit(code=1)

    if stop >= entry:
        typer.echo(
            f"ERROR: --stop ({stop}) must be below --entry ({entry}) for a long signal.",
            err=True,
        )
        raise typer.Exit(code=1)

    resolved_scale_out = (
        scale_out if scale_out is not None else round(entry + 2.0 * (entry - stop), 4)
    )
    if resolved_scale_out <= entry:
        typer.echo(
            f"ERROR: --scale-out ({resolved_scale_out}) must be above --entry ({entry}).",
            err=True,
        )
        raise typer.Exit(code=1)

    _run_with_connection_handling(
        lambda: _force_entry(
            symbol=symbol.upper(),
            entry=entry,
            stop=stop,
            scale_out=resolved_scale_out,
            strategy_name=strategy_name,
        )
    )


async def _force_entry(
    *,
    symbol: str,
    entry: float,
    stop: float,
    scale_out: float,
    strategy_name: str,
) -> None:
    """Async body of ``force-entry`` — build executor, synthesize Signal, place."""
    settings = get_settings()
    ibkr = IBKRClient(settings=settings)
    await ibkr.connect()
    journal = Journal()
    store = PositionStore()
    risk_engine = RiskEngine(settings=settings)
    executor = Executor(
        ibkr=ibkr,
        position_store=store,
        journal=journal,
        risk_engine=risk_engine,
        settings=settings,
    )
    try:
        # Reconcile so a pre-existing IBKR position (from a prior session,
        # or another bot instance) is visible to the risk engine's
        # single-active-position-per-symbol guardrail.
        await executor.reconcile()

        signal = Signal(
            symbol=symbol,
            strategy=strategy_name,
            entry=entry,
            stop=stop,
            scale_out_price=scale_out,
            runner_target_price=None,
            timestamp=datetime.now(UTC),
            reasons=["force_entry"],
            # A plausibly-large bar volume so Phase 4c's 2%-of-bar-volume
            # liquidity cap doesn't reject a manual test. Other risk gates
            # (max_loss, max_stop_width, PDT) still apply normally.
            recent_bar_volume=1_000_000,
        )

        _console.print(
            f"[cyan]force-entry[/cyan] {symbol} entry={entry} stop={stop} "
            f"scale_out={scale_out} strategy={strategy_name}"
        )

        await executor.handle_signal(signal)

        position = store.get(symbol)
        if position is None:
            _console.print(
                "[red]No position recorded[/red] — the risk engine likely rejected the signal. "
                "Check stdout / session jsonl for ``signal.rejected`` events."
            )
            return

        # Phase 6.13 fix: ``drain_pending_fills`` only waits for tasks
        # already scheduled. On MKT/LMT paths the parent fill event can
        # arrive from IBKR's websocket *after* we returned from
        # ``handle_signal`` but before any task is scheduled — so drain
        # sees an empty set and we'd disconnect before
        # ``_handle_parent_fill`` ran, leaving the position naked (no
        # stop, no target). Poll the parent trade until it fills or
        # times out so the event system gets CPU time to deliver the
        # execution. STP_LMT rests until triggered and is expected to
        # stay un-filled here; we skip the wait for that path.
        entry_order_type = settings.execution.entry_order_type
        bracket = executor.active_trades.get(symbol)
        if (
            entry_order_type in {"MKT", "LMT"}
            and bracket is not None
            and bracket.parent is not None
        ):
            parent_trade = bracket.parent
            deadline = asyncio.get_event_loop().time() + 10.0
            while asyncio.get_event_loop().time() < deadline:
                if parent_trade.fills or parent_trade.isDone():
                    break
                await asyncio.sleep(0.1)
            if not parent_trade.fills and not parent_trade.isDone():
                _console.print(
                    f"[yellow]Parent trade did not fill within 10s.[/yellow] "
                    f"Order type: {entry_order_type}. Check TWS for status."
                )

        # Drain any fill-handler tasks scheduled by the event above so
        # the protection children land (and the journal row commits)
        # before we disconnect.
        await executor.drain_pending_fills()

        # Re-fetch in case the fill moved status from pending_entry → open
        # and attached children IDs via _place_entry_protection_children.
        position = store.get(symbol) or position
        bracket = executor.active_trades.get(symbol)
        _print_force_entry_result(position, settings.execution.entry_order_type, bracket)
    finally:
        await journal.close()
        await ibkr.disconnect()


def _print_force_entry_result(
    position: Position,
    entry_order_type: str,
    bracket: Any = None,
) -> None:
    """Render a summary table for ``force-entry``'s outcome.

    Phase 6.14 — also shows the atomic bracket's per-leg quantity + price
    straight from the Trade objects on IBKR's side. This is what actually
    got sent to IBKR, useful for diagnosing "TWS shows different size"
    reports (the on-wire truth vs. the GUI interpretation).
    """
    table = Table(title=f"force-entry result — {position.symbol}")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Entry order type", entry_order_type)
    table.add_row("Position status", position.status)
    table.add_row("Shares (position)", str(position.shares))
    table.add_row("Avg fill price", f"${position.avg_price:.4f}")
    table.add_row("Planned entry", f"${position.entry_trigger_price:.4f}")
    table.add_row("Stop price", f"${position.stop_price:.4f}")
    table.add_row("Scale-out price", f"${position.scale_out_price:.4f}")
    table.add_row("Parent order id", str(position.parent_order_id))
    table.add_row("Stop order id", str(position.stop_order_id))
    table.add_row("Target order id", str(position.target_order_id))
    # Per-leg truth from the bracket — shows the actual Order quantities
    # we submitted to IBKR, regardless of what TWS's UI renders.
    if bracket is not None:
        if bracket.parent is not None:
            table.add_row(
                "  Parent (on wire)",
                f"{bracket.parent.order.orderType} × {bracket.parent.order.totalQuantity}",
            )
        if bracket.stop is not None:
            aux = getattr(bracket.stop.order, "auxPrice", 0.0)
            table.add_row(
                "  Stop (on wire)",
                f"{bracket.stop.order.orderType} × {bracket.stop.order.totalQuantity} @ ${aux:.2f}",
            )
        if bracket.scale_lmt is not None:
            lmt = getattr(bracket.scale_lmt.order, "lmtPrice", 0.0)
            table.add_row(
                "  Scale LMT (on wire)",
                f"{bracket.scale_lmt.order.orderType} × "
                f"{bracket.scale_lmt.order.totalQuantity} @ ${lmt:.2f}",
            )
        if bracket.target is not None:
            lmt = getattr(bracket.target.order, "lmtPrice", 0.0)
            table.add_row(
                "  Target (on wire)",
                f"{bracket.target.order.orderType} × "
                f"{bracket.target.order.totalQuantity} @ ${lmt:.2f}",
            )
    _console.print(table)


async def _ping() -> None:
    """Async body of the ``ping`` command."""
    client = IBKRClient()
    await client.connect()
    try:
        # Phase 10.3 — ``ping`` is interactive; the operator expects current
        # values, not a possibly-cached snapshot. Force a fresh fetch.
        summary = await client.account_summary(refresh=True)
    finally:
        await client.disconnect()
    _print_summary(summary)


async def _scan(limit: int, notify: bool) -> None:
    """Async body of the ``scan`` command."""
    settings: Settings = get_settings()
    # Fail fast on missing Finnhub key before we open the IBKR socket.
    async with FinnhubClient(settings=settings) as finnhub:
        float_source = FloatSource(finnhub=finnhub)
        ibkr = IBKRClient(settings=settings)
        await ibkr.connect()
        try:
            scanner = IBKRScanner(
                ibkr=ibkr, finnhub=finnhub, settings=settings, float_source=float_source
            )
            hits = await scanner.scan_top_gappers()
        finally:
            await ibkr.disconnect()

    hits = hits[:limit]
    for hit in hits:
        _log.info(
            "scan.hit",
            symbol=hit.symbol,
            float_shares=hit.float_shares,
            float_source=hit.float_source,
            catalyst=hit.catalyst,
            reasons=hit.reasons,
            news_count=len(hit.news_items),
        )
    _print_scan_table(hits)

    if notify:
        notifier = Notifier(settings=settings)
        await notifier.send_watchlist(hits)


async def _watch(duration: float | None, limit: int, poll_interval: float, notify: bool) -> None:
    """Async body of the ``watch`` command — scan once, then drive the strategy loop."""
    settings: Settings = get_settings()
    shutdown_event = asyncio.Event()
    uninstall_signal = _install_shutdown_handler(shutdown_event)
    async with FinnhubClient(settings=settings) as finnhub:
        float_source = FloatSource(finnhub=finnhub)
        ibkr = IBKRClient(settings=settings)
        await ibkr.connect()
        try:
            scanner = IBKRScanner(
                ibkr=ibkr, finnhub=finnhub, settings=settings, float_source=float_source
            )
            hits = await scanner.scan_top_gappers()
            hits = hits[:limit]
            _print_scan_table(hits)
            if not hits:
                return

            market_data = MarketData(ibkr=ibkr)
            signal_bus = SignalBus()
            notifier = Notifier(settings=settings) if notify else None
            consumer = asyncio.create_task(
                _consume_signals(signal_bus=signal_bus, notifier=notifier)
            )
            try:
                await run_strategy_loop(
                    watchlist=hits,
                    market_data=market_data,
                    signal_bus=signal_bus,
                    duration_minutes=duration,
                    poll_interval=poll_interval,
                    settings=settings,
                    shutdown_event=shutdown_event,
                    scanner=scanner,
                )
            finally:
                consumer.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await consumer
                await market_data.close()
        finally:
            await ibkr.disconnect()
            uninstall_signal()


async def _trade(
    duration: float | None,
    limit: int,
    poll_interval: float,
    notify: bool,
    settings: Settings,
    dry_run_signal: str,
    simulate_loss_usd: float = 0.0,
    simulate_time: str = "",
    simulate_reentry: int = 0,
    simulate_red_days: int = 0,
) -> None:
    """Async body of the ``trade`` command — full Phase 4b wiring."""
    shutdown_event = asyncio.Event()
    uninstall_signal = _install_shutdown_handler(shutdown_event)
    ibkr = IBKRClient(settings=settings)
    await ibkr.connect()
    journal = Journal()
    store = PositionStore()
    notifier = Notifier(settings=settings) if notify else None
    rehab_engine = RehabEngine(settings=settings, journal=journal)
    if simulate_red_days > 0:
        _apply_simulate_red_days(rehab_engine=rehab_engine, n=simulate_red_days, settings=settings)
    risk_engine = RiskEngine(settings=settings, rehab_engine=rehab_engine)
    executor = Executor(
        ibkr=ibkr,
        position_store=store,
        journal=journal,
        risk_engine=risk_engine,
        notifier=notifier,
        settings=settings,
    )
    market_data = MarketData(ibkr=ibkr)
    trade_manager = TradeManager(
        ibkr=ibkr,
        store=store,
        market_data=market_data,
        executor=executor,
        journal=journal,
        settings=settings,
    )
    # Phase 10.1 — naked-position watchdog. Detection-only; ships in shadow
    # mode (telegram alerts suppressed; events still emitted) until the
    # operator flips ``watchdog.shadow_mode`` to False after one clean
    # session of shadow logs. See README "Phase 10.1 watchdog" note.
    watchdog = (
        Watchdog(ibkr=ibkr, position_store=store, notifier=notifier, settings=settings)
        if settings.watchdog.enabled
        else None
    )
    orchestrator = Orchestrator(
        executor=executor,
        store=store,
        trade_manager=trade_manager,
        settings=settings,
        rehab_engine=rehab_engine,
        notifier=notifier,
    )
    try:
        startup = await orchestrator.startup()
        if startup.get("rehab_tier") and startup["rehab_tier"] != "NORMAL":
            _console.print(
                f"[bold yellow]Rehab tier active: {startup['rehab_tier']}.[/bold yellow] "
                "Risk caps are scaled down — run `python -m bot rehab-status` for detail."
            )
        if startup["halted"]:
            record = startup["halt_record"]
            _console.print(
                f"[bold red]Session halted: {record.reason} at {record.triggered_at}. "
                f"Existing brackets run to their own stops; no new entries will be placed. "
                f"Run `python -m bot reset-halt` to clear.[/bold red]"
            )
        await _apply_simulate_hooks(
            simulate_loss_usd=simulate_loss_usd,
            simulate_time=simulate_time,
            risk_engine=risk_engine,
            orchestrator=orchestrator,
            store=store,
            notifier=notifier,
        )

        if dry_run_signal:
            signal = _parse_dry_run_signal(dry_run_signal)
            if simulate_reentry:
                _apply_simulate_reentry(
                    store=store, symbol=signal.symbol, n=simulate_reentry, settings=settings
                )
            runner_display = (
                f"{signal.runner_target_price}"
                if signal.runner_target_price is not None
                else "none"
            )
            _console.print(
                f"[yellow]--dry-run-signal[/yellow] handing fabricated signal to executor: "
                f"{signal.symbol} entry={signal.entry} stop={signal.stop} "
                f"scale={signal.scale_out_price} runner={runner_display}"
            )
            await executor.handle_signal(signal)
            _console.print(
                "[green]Bracket placement submitted.[/green] Watch TWS order monitor; "
                "cancel manually when you're done."
            )
            _print_positions_table(store.list_active())
            await asyncio.sleep(2.0)
            return

        async with FinnhubClient(settings=settings) as finnhub:
            float_source = FloatSource(finnhub=finnhub)
            scanner = IBKRScanner(
                ibkr=ibkr, finnhub=finnhub, settings=settings, float_source=float_source
            )
            hits = await scanner.scan_top_gappers()
            hits = hits[:limit]
            _print_scan_table(hits)
            if not hits:
                # Phase 9.3: Day 7 (2026-04-28) start-up exited here on empty
                # initial scan, missing intraday catalyst news that breaks
                # after the open (e.g., ATLX qualified at 10:32 ET on a
                # rescan, not the first pass). Continue into the loop and
                # let the Phase 6.2 rescan tick populate the watchlist as
                # candidates emerge.
                _console.print(
                    "[yellow]Initial scan returned no candidates; entering main loop "
                    "to await periodic rescans.[/yellow]"
                )

            signal_bus = SignalBus()
            try:
                await run_strategy_loop(
                    watchlist=hits,
                    market_data=market_data,
                    signal_bus=signal_bus,
                    executor=executor,
                    trade_manager=trade_manager,
                    rehab_engine=rehab_engine,
                    notifier=notifier,
                    watchdog=watchdog,
                    duration_minutes=duration,
                    poll_interval=poll_interval,
                    settings=settings,
                    shutdown_event=shutdown_event,
                    scanner=scanner,
                    position_store=store,
                )
            finally:
                await market_data.close()
                _print_risk_state(risk_engine)
                _print_positions_table(store.list_active())
                _console.print(
                    "[yellow]Positions remain open on IBKR. Use 'python -m bot flatten' "
                    "to close, or manage in TWS.[/yellow]"
                )
                if notifier is not None:
                    s = risk_engine.state
                    with contextlib.suppress(Exception):
                        await notifier.send_daily_summary(
                            trades=s.trades_today,
                            realized_pnl=s.realized_pnl_usd,
                            peak_pnl=s.max_pnl_today_usd,
                            reached_goal=s.realized_pnl_usd >= settings.risk.daily_profit_goal_usd,
                        )
    finally:
        await orchestrator.shutdown()
        await journal.close()
        await ibkr.disconnect()
        uninstall_signal()


async def _apply_simulate_hooks(
    *,
    simulate_loss_usd: float,
    simulate_time: str,
    risk_engine: RiskEngine,
    orchestrator: Orchestrator,
    store: PositionStore,
    notifier: Notifier | None,
) -> None:
    """Inject a synthetic loss and/or schedule an immediate auto-flatten for demos.

    Paper-only (guarded at the CLI). The synthetic loss is fed through the
    real ``on_fill_closed`` path so halt persistence + notifier wiring are
    exercised end-to-end. The ``--simulate-time`` flag currently just
    triggers ``flatten_all_active`` immediately — we don't try to monkey
    the apscheduler clock; the point is to prove the callback works
    against whatever positions exist.
    """
    if simulate_loss_usd > 0:
        fake_position = Position(
            symbol="__SIMULATED__",
            strategy="simulate",
            shares=0,
            avg_price=0.0,
            stop_price=0.0,
            scale_out_price=0.0,
            runner_target_price=None,
            parent_order_id=0,
            stop_order_id=0,
            target_order_id=0,
            opened_at=datetime.now(UTC),
            status="closed",
        )
        await risk_engine.on_fill_closed(fake_position, -abs(simulate_loss_usd))
        _print_risk_state(risk_engine)
        if risk_engine.is_halted() and notifier is not None:
            with contextlib.suppress(Exception):
                await notifier.send_halt(
                    risk_engine.state.halt_reason or "unknown",
                    risk_engine.state.realized_pnl_usd,
                )
    if simulate_time:
        _console.print(
            f"[yellow]--simulate-time[/yellow] {simulate_time}: firing "
            f"auto_flatten immediately against the current store."
        )
        _ = store  # store is already the one the scheduler uses
        flattened = await orchestrator.auto_flatten.flatten_all_active()
        _console.print(f"[yellow]Flattened {flattened} position(s).[/yellow]")


async def _status() -> None:
    """Async body of the ``status`` command — reconcile + print risk + positions + re-entries."""
    settings = get_settings()
    ibkr = IBKRClient(settings=settings)
    await ibkr.connect()
    journal = Journal()
    store = PositionStore()
    rehab_engine = RehabEngine(settings=settings, journal=journal)
    risk_engine = RiskEngine(settings=settings, rehab_engine=rehab_engine)
    executor = Executor(
        ibkr=ibkr,
        position_store=store,
        journal=journal,
        risk_engine=risk_engine,
        settings=settings,
    )
    try:
        await executor.reconcile()
        await risk_engine.apply_halt_flag_if_current()
        rehab_engine.load_state()
        # Phase 4d: rebuild today's per-symbol history from the journal so the
        # status surface reflects "re-entries today" even across restarts.
        tz = ZoneInfo(settings.session.timezone)
        today = datetime.now(tz).date()
        trades = await journal.trades_for_session(today, settings.session.timezone)
        store.rebuild_symbol_histories_from_journal(trades)
        _print_risk_state(risk_engine)
        _print_halt_flag(risk_engine)
        _print_positions_table(store.list_active())
        _print_reentries_table(store, settings)
        _print_execution_post_scaleout(settings)
        _print_logging_destination(settings)
        await _print_status_rehab_summary(settings=settings, rehab_engine=rehab_engine)
    finally:
        await journal.close()
        await ibkr.disconnect()


def _print_logging_destination(settings: Settings) -> None:
    """Phase 5.1 — show where the next session's JSONL log file will be written.

    Prints the NY-dated filename computed from ``settings.logging.path`` so
    operators can confirm file logging is actually wired, and surfaces
    ``no file log`` when ``logging.path`` is unset (stdout-only mode).
    """
    path = resolve_session_log_path(settings)
    if path is None:
        _console.print("[dim]Logging: stdout only (set logging.path to enable file logs).[/dim]")
        return
    _console.print(f"[dim]Logging: stdout + {path.as_posix()} (next session).[/dim]")


def _print_execution_post_scaleout(settings: Settings) -> None:
    """One-line summary of the Phase 4h/4i execution configuration.

    Rendered under the positions table so an operator can verify at a
    glance (a) whether the next scale-out will install an adjustable stop
    (IBKR server-side STP → TRAIL conversion) or a static breakeven, and
    (b) the Phase 4i exit discipline — scale-out multiple +
    whether the bracket plants a runner-target ceiling at entry. Trail
    activation trigger is rendered relative to the scale-out anchor to
    match the Phase 4i formula.
    """
    exec_cfg = settings.execution
    entry_line = f"[dim]Entry type: {exec_cfg.entry_order_type}"
    if exec_cfg.entry_order_type == "STP_LMT":
        entry_line += f" (LMT buffer ${exec_cfg.entry_limit_buffer_usd:.2f})"
    entry_line += ".[/dim]"
    _console.print(entry_line)
    _console.print(
        f"[dim]Exits: scale-out at +{exec_cfg.scale_out_multiple:g}R | "
        f"runner target {'on' if exec_cfg.runner_target_enabled else 'off'} "
        f"(mult {exec_cfg.runner_target_multiple:g}R).[/dim]"
    )
    mode = exec_cfg.post_scaleout_stop_mode
    if mode == "static_breakeven":
        _console.print("[dim]Post-scale-out: static breakeven stop (Phase 4e fallback).[/dim]")
    elif mode == "adjustable_to_trail":
        _console.print(
            f"[dim]Post-scale-out: adjustable stop → TRAIL at "
            f"scale_out + {exec_cfg.trail_activation_r_multiple:g}R "
            f"(trail distance = {exec_cfg.trail_amount_r_multiple:g}R).[/dim]"
        )
    else:  # immediate_trail
        _console.print(
            f"[dim]Post-scale-out: immediate TRAIL at "
            f"scale_out − {exec_cfg.trail_amount_r_multiple:g}R "
            f"(trails the runner, exits on {exec_cfg.trail_amount_r_multiple:g}R reversal).[/dim]"
        )


async def _print_status_rehab_summary(*, settings: Settings, rehab_engine: RehabEngine) -> None:
    """One-line rehab tier summary appended to the ``status`` command output."""
    if not rehab_engine.enabled:
        _console.print("[dim]Rehab: disabled.[/dim]")
        return
    stats = await rehab_engine.compute_stats()
    caps = rehab_engine.apply_to_caps(settings.risk)
    if caps.tier is RehabTier.NORMAL:
        _console.print(
            f"[green]Rehab tier: NORMAL[/green] "
            f"(reds={stats.consecutive_red_days}, "
            f"drawdown=${stats.cumulative_drawdown_usd:+.0f})"
        )
        return
    _console.print(
        f"[bold yellow]Rehab tier: {caps.tier.value}[/bold yellow] "
        f"(trigger={caps.trigger_reason}, "
        f"per-trade=${caps.max_loss_per_trade_usd:.0f}/${caps.base_max_loss_per_trade_usd:.0f}, "
        f"daily=${caps.max_daily_loss_usd:.0f}/${caps.base_max_daily_loss_usd:.0f}, "
        f"trades={caps.max_trades_per_day}/{caps.base_max_trades_per_day})"
    )


async def _rehab_status() -> None:
    """Async body of the ``rehab-status`` command — read flag + compute current tier."""
    settings: Settings = get_settings()
    journal = Journal()
    rehab_engine = RehabEngine(settings=settings, journal=journal)
    try:
        rehab_engine.load_state()
        stats = await rehab_engine.compute_stats()
        computed_tier, computed_reason = classify_tier(
            stats, settings.risk.rehab, settings.risk.max_daily_loss_usd
        )
        _print_rehab_status(
            settings=settings,
            rehab_engine=rehab_engine,
            computed_tier=computed_tier,
            computed_reason=computed_reason,
            drawdown_usd=stats.cumulative_drawdown_usd,
            consecutive_red_days=stats.consecutive_red_days,
            lookback_days=stats.lookback_days,
        )
    finally:
        await journal.close()


async def _commissions(lookback_days: int) -> None:
    """Async body of the ``commissions`` command — journal read + aggregate + render."""
    from bot.reports import commission_summary  # noqa: PLC0415 - late import keeps startup lean

    settings: Settings = get_settings()
    journal = Journal()
    try:
        trades = await journal.recent_trades(n=1_000)
    finally:
        await journal.close()

    tz = settings.session.timezone
    today_ny = datetime.now(ZoneInfo(tz)).date()
    window_start = today_ny - timedelta(days=lookback_days)
    closed = [
        row
        for row in trades
        if row.closed_at is not None
        and window_start <= row.closed_at.astimezone(ZoneInfo(tz)).date() <= today_ny
    ]
    if not closed:
        _console.print(f"[yellow]No closed trades in the last {lookback_days} days.[/yellow]")
        return

    summary = commission_summary(closed)

    paper_suffix = (
        " [PAPER — absolute $ simulated, ratios are structural]"
        if settings.account.mode == "paper"
        else ""
    )
    _console.print(
        Panel(
            f"[bold]Commission summary — last {lookback_days} days[/bold]{paper_suffix}",
            expand=False,
        )
    )

    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Trades counted", str(summary.trades_counted))
    table.add_row("Trades with commission data", str(summary.trades_with_commission_data))
    table.add_row("Gross PnL", f"${summary.total_gross_pnl:,.2f}")
    table.add_row("Total commission", f"${summary.total_commission:,.2f}")
    table.add_row("  Entry legs", f"${summary.total_entry_commission:,.2f}")
    table.add_row("  Scale legs", f"${summary.total_scale_commission:,.2f}")
    table.add_row("  Exit legs", f"${summary.total_exit_commission:,.2f}")
    table.add_row("Net PnL", f"${summary.net_pnl:,.2f}")
    table.add_row("Avg commission / trade", f"${summary.avg_commission_per_trade:,.4f}")
    if summary.commission_pct_of_gross is not None:
        table.add_row("Commission % of gross", f"{summary.commission_pct_of_gross * 100:.2f}%")
    else:
        table.add_row("Commission % of gross", "— (gross PnL ≤ 0)")
    if summary.scale_out_commission_share is not None:
        table.add_row(
            "Scale-out share of commission",
            f"{summary.scale_out_commission_share * 100:.2f}%",
        )
    else:
        table.add_row("Scale-out share of commission", "— (no commission data)")
    _console.print(table)


async def _suggest_caps(lookback_days: int, compare: bool) -> None:
    """Async body of the ``suggest-caps`` command — journal analysis + advisory table."""
    settings: Settings = get_settings()
    journal = Journal()
    try:
        trades = await journal.recent_trades(n=1_000)
    finally:
        await journal.close()

    tz = settings.session.timezone
    today_ny = datetime.now(ZoneInfo(tz)).date()
    window_start = today_ny - timedelta(days=lookback_days)
    closed = [
        row
        for row in trades
        if row.pnl is not None
        and row.closed_at is not None
        and window_start <= row.closed_at.astimezone(ZoneInfo(tz)).date() <= today_ny
    ]
    if not closed:
        _console.print(
            f"[yellow]No closed trades in the last {lookback_days} days — nothing to suggest.[/yellow]"
        )
        return

    stats = _compute_suggest_caps_stats(closed, timezone=tz)
    suggestions = _suggested_caps(stats)
    _print_suggest_caps_stats(stats, lookback_days=lookback_days)
    _print_suggested_caps(suggestions, settings=settings, compare=compare)
    _console.print(
        "[dim]Advisory only — suggest-caps never modifies config.yaml. "
        "the rule: pick your numbers, stick to them.[/dim]"
    )


async def _flatten() -> None:
    """Async body of the ``flatten`` command — reconcile + flatten every active symbol."""
    settings = get_settings()
    ibkr = IBKRClient(settings=settings)
    await ibkr.connect()
    journal = Journal()
    store = PositionStore()
    risk_engine = RiskEngine(settings=settings)
    executor = Executor(
        ibkr=ibkr,
        position_store=store,
        journal=journal,
        risk_engine=risk_engine,
        settings=settings,
    )
    try:
        await executor.reconcile()
        active = store.list_active()
        if not active:
            _console.print("[green]No active positions.[/green]")
            return
        for position in active:
            _console.print(
                f"[red]Flattening[/red] {position.symbol} (strategy={position.strategy}, "
                f"shares={position.shares})"
            )
            await executor.flatten_symbol(position.symbol, reason="manual_flatten")
    finally:
        await journal.close()
        await ibkr.disconnect()


async def _positions() -> None:
    """Async body of the ``positions`` command — read-only reconcile + print."""
    settings = get_settings()
    ibkr = IBKRClient(settings=settings)
    await ibkr.connect()
    journal = Journal()
    store = PositionStore()
    risk_engine = RiskEngine(settings=settings)
    executor = Executor(
        ibkr=ibkr,
        position_store=store,
        journal=journal,
        risk_engine=risk_engine,
        settings=settings,
    )
    try:
        await executor.reconcile()
        _print_positions_table(store.list_active())
    finally:
        await journal.close()
        await ibkr.disconnect()


async def _backtest(
    symbol: str,
    target_date: date_cls,
    force_catalyst: bool,
    strategy_selection: str,
    max_rejections: int,
) -> None:
    """Async body of the ``backtest`` command — qualify + replay + print + JSONL log."""
    settings: Settings = get_settings()
    ibkr = IBKRClient(settings=settings)
    await ibkr.connect()
    try:
        market_data = MarketData(ibkr=ibkr)
        replayer = Replayer(
            ibkr=ibkr,
            market_data=market_data,
            gap_and_go=GapAndGoStrategy(),
            momentum=MomentumStrategy(
                flag_max_pullback_pct=settings.strategies.momentum.flag_max_pullback_pct,
            ),
        )
        try:
            result = await replayer.replay(
                symbol=symbol,
                target_date=target_date,
                strategy_selection=strategy_selection,
                force_catalyst=force_catalyst,
            )
        except ReplayError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from None
    finally:
        await ibkr.disconnect()

    _print_backtest_table(symbol, target_date, result.signals)
    _write_backtest_jsonl(symbol, target_date, result.signals)
    _print_rejections_table(symbol, target_date, result.rejections, max_rows=max_rejections)
    _write_rejections_jsonl(symbol, target_date, result.rejections)
    _print_backtest_summary(symbol, target_date, result.signals, result.rejections)


def _print_backtest_table(symbol: str, target_date: date_cls, signals: list[Signal]) -> None:
    """Render backtest signals as a rich table; skip rendering entirely when empty."""
    if not signals:
        return
    table = Table(title=f"Backtest {symbol} — {target_date.isoformat()}", show_lines=False)
    table.add_column("Bar Time", no_wrap=True)
    table.add_column("Strategy", no_wrap=True)
    table.add_column("Entry", justify="right", no_wrap=True)
    table.add_column("Stop", justify="right", no_wrap=True)
    table.add_column("Scale", justify="right", no_wrap=True)
    table.add_column("Runner", justify="right", no_wrap=True)
    table.add_column("R:R", justify="right", no_wrap=True)
    table.add_column("First Reason", no_wrap=True)
    for sig in signals:
        table.add_row(
            sig.timestamp.strftime("%Y-%m-%d %H:%M"),
            sig.strategy,
            f"{sig.entry:.2f}",
            f"{sig.stop:.2f}",
            f"{sig.scale_out_price:.2f}",
            f"{sig.runner_target_price:.2f}" if sig.runner_target_price is not None else "-",
            f"{sig.risk_reward:.2f}",
            sig.reasons[0] if sig.reasons else "-",
        )
    _console.print(table)


def _write_backtest_jsonl(symbol: str, target_date: date_cls, signals: list[Signal]) -> None:
    """Persist signals as JSONL at ``logs/backtest_<symbol>_<date>.jsonl`` for later inspection."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    path = logs_dir / f"backtest_{symbol}_{target_date.isoformat()}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for sig in signals:
            record = {
                "symbol": sig.symbol,
                "strategy": sig.strategy,
                "timestamp": sig.timestamp.isoformat(),
                "entry": sig.entry,
                "stop": sig.stop,
                "scale_out_price": sig.scale_out_price,
                "runner_target_price": sig.runner_target_price,
                "risk_per_share": sig.risk_per_share,
                "reward_per_share": sig.reward_per_share,
                "risk_reward": sig.risk_reward,
                "reasons": sig.reasons,
            }
            f.write(json.dumps(record) + "\n")
    _log.info("backtest.jsonl_written", path=str(path), signal_count=len(signals))


def _print_rejections_table(
    symbol: str,
    target_date: date_cls,
    rejections: list[RejectedCandidate],
    *,
    max_rows: int,
) -> None:
    """Render rejected candidates as a second rich table; noop when empty."""
    if not rejections:
        return
    title = f"Rejected candidates — {symbol} {target_date.isoformat()}"
    if len(rejections) > max_rows:
        title += f" (showing {max_rows} of {len(rejections)})"
    table = Table(title=title, show_lines=False)
    table.add_column("Bar Time", no_wrap=True)
    table.add_column("Strategy", no_wrap=True)
    table.add_column("Stage", no_wrap=True)
    table.add_column("Reason", no_wrap=True)
    table.add_column("Key Context", overflow="fold")
    for rejection in rejections[:max_rows]:
        table.add_row(
            rejection.bar_time.strftime("%Y-%m-%d %H:%M"),
            rejection.strategy,
            rejection.stage,
            rejection.reason,
            _format_rejection_context(rejection.context),
        )
    _console.print(table)


def _write_rejections_jsonl(
    symbol: str, target_date: date_cls, rejections: list[RejectedCandidate]
) -> None:
    """Persist rejections as JSONL at ``logs/backtest_rejections_<symbol>_<date>.jsonl``."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    path = logs_dir / f"backtest_rejections_{symbol}_{target_date.isoformat()}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for rejection in rejections:
            record = {
                "symbol": rejection.symbol,
                "strategy": rejection.strategy,
                "bar_time": rejection.bar_time.isoformat(),
                "stage": rejection.stage,
                "reason": rejection.reason,
                "context": rejection.context,
            }
            f.write(json.dumps(record, default=str) + "\n")
    _log.info("backtest.rejections_jsonl_written", path=str(path), count=len(rejections))


def _print_backtest_summary(
    symbol: str,
    target_date: date_cls,
    signals: list[Signal],
    rejections: list[RejectedCandidate],
) -> None:
    """Print the final one-line summary — the diagnostic hook for zero-signal days."""
    if not signals and not rejections:
        typer.echo(
            f"No signals and no rejected candidates on {symbol} for {target_date.isoformat()}. "
            "Detectors did not trigger on any bar. This may indicate the day lacked the "
            "setup patterns, or a detector calibration issue."
        )
        return
    stage_breakdown = _rejection_stage_breakdown(rejections)
    breakdown_str = f" ({stage_breakdown})" if stage_breakdown else ""
    typer.echo(f"{len(signals)} signals, {len(rejections)} rejections{breakdown_str}")


def _rejection_stage_breakdown(rejections: list[RejectedCandidate]) -> str:
    """Render ``setup=12, entry_trigger=30, stop_calculation=5`` for the summary line."""
    counts: dict[str, int] = {}
    for rejection in rejections:
        counts[rejection.stage] = counts.get(rejection.stage, 0) + 1
    return ", ".join(f"{stage}={counts[stage]}" for stage in sorted(counts))


def _format_rejection_context(context: dict[str, Any]) -> str:
    """Collapse the context dict into a ``k=v, k=v`` preview (top 3 entries)."""
    if not context:
        return "-"
    pieces: list[str] = []
    for key, value in list(context.items())[:3]:
        if isinstance(value, float):
            pieces.append(f"{key}={value:.2f}")
        else:
            pieces.append(f"{key}={value}")
    return ", ".join(pieces)


async def _consume_signals(signal_bus: SignalBus, notifier: Notifier | None) -> None:
    """Print each signal to the CLI and optionally push to Telegram."""
    async for sig in signal_bus.stream():
        _print_signal(sig)
        if notifier is not None:
            await notifier.send_signal(sig)


def _print_signal(signal: Signal) -> None:
    """Render one Signal as a rich panel-friendly single-line row."""
    ts_local = signal.timestamp.strftime("%H:%M:%S")
    runner_part = (
        f"runner=[green]{signal.runner_target_price:.2f}[/green] "
        if signal.runner_target_price is not None
        else "runner=[dim]-[/dim] "
    )
    _console.print(
        f"[bold green]⚡ {signal.strategy}[/bold green] "
        f"${signal.symbol} @ {ts_local}  "
        f"entry=[cyan]{signal.entry:.2f}[/cyan] "
        f"stop=[red]{signal.stop:.2f}[/red] "
        f"scale=[green]{signal.scale_out_price:.2f}[/green] "
        f"{runner_part}"
        f"R:R=[yellow]{signal.risk_reward:.2f}[/yellow]  "
        f"{', '.join(signal.reasons)}"
    )


def _confirm_live_or_exit(settings: Settings) -> Settings:
    """Show the red-border CONFIRM panel; require the literal word ``CONFIRM``.

    Returns the Settings object with ``account.mode`` flipped to ``live``
    so the rest of the trade path operates on a consistent view. Anything
    other than a literal ``CONFIRM`` — including empty, lower-case, or
    ``yes`` — exits non-zero. The panel lists every guard that's still in
    effect so the operator can double-check before committing.
    """
    panel = Panel(
        "\n".join(
            [
                "You are about to enable LIVE trading against the REAL MONEY account.",
                "",
                "Active guards (will still apply):",
                f"  - Max loss per trade:     ${settings.risk.max_loss_per_trade_usd:.0f}",
                f"  - Max position value:     ${settings.risk.max_position_value_usd:,.0f}",
                f"  - Max daily loss:         ${settings.risk.max_daily_loss_usd:.0f}",
                f"  - Daily profit goal:      ${settings.risk.daily_profit_goal_usd:.0f}",
                (
                    f"  - Give-back trigger:      ${settings.risk.giveback_trigger_usd:.0f} "
                    f"/ {settings.risk.giveback_pct:.0f}%"
                ),
                f"  - Max concurrent positions: {settings.risk.max_concurrent_positions}",
                f"  - Max trades per day:     {settings.risk.max_trades_per_day}",
                f"  - Max stop width:         ${settings.risk.max_stop_width_usd:.2f}",
                f"  - Max % of bar volume:    {settings.risk.max_pct_of_bar_volume:.1f}%",
                (
                    f"  - Extension bar trigger:  $max_loss x "
                    f"{settings.risk.extension_bar_trigger_multiple:.1f} = $"
                    f"{settings.risk.max_loss_per_trade_usd * settings.risk.extension_bar_trigger_multiple:.0f}"
                ),
                f"  - Auto-flatten at:        {settings.session.flatten_all} "
                f"({settings.session.timezone})",
                "",
                "Type the literal word [bold]CONFIRM[/bold] (uppercase) to proceed.",
            ]
        ),
        title="[bold red]⚠  LIVE TRADING[/bold red]",
        border_style="bold red",
    )
    _console.print(panel)
    response = typer.prompt("Enter CONFIRM to continue", default="", show_default=False)
    if response != "CONFIRM":
        _console.print("[yellow]Live trading not confirmed — exiting.[/yellow]")
        raise typer.Exit(code=1)
    return settings.model_copy(update={"account": AccountConfig(mode="live")})


def _halt_flag_path() -> Path:
    """Resolve the halt-flag path. One place to override for tests in future."""
    return Path("logs") / "halt.flag"


def _print_risk_state(risk_engine: RiskEngine) -> None:
    """Render a one-row risk-state summary: halt, trades, PnL, peak."""
    s = risk_engine.state
    table = Table(title="Risk state", show_lines=False)
    table.add_column("Halted", no_wrap=True)
    table.add_column("Reason", no_wrap=True)
    table.add_column("Trades", justify="right", no_wrap=True)
    table.add_column("PnL", justify="right", no_wrap=True)
    table.add_column("Peak PnL", justify="right", no_wrap=True)
    halted_cell = "[bold red]YES[/bold red]" if s.halted else "[green]no[/green]"
    table.add_row(
        halted_cell,
        s.halt_reason or "-",
        str(s.trades_today),
        f"${s.realized_pnl_usd:+.2f}",
        f"${s.max_pnl_today_usd:+.2f}",
    )
    _console.print(table)
    _print_risk_caps(get_settings())


def _print_risk_caps(settings: Settings) -> None:
    """Render Phase 4c rule-based caps — compact, two-line layout under risk state."""
    risk = settings.risk
    extension_threshold = risk.max_loss_per_trade_usd * risk.extension_bar_trigger_multiple
    _console.print(
        f"[dim]Risk caps:[/dim] "
        f"max stop width ${risk.max_stop_width_usd:.2f} | "
        f"max % of bar vol {risk.max_pct_of_bar_volume:.1f}% | "
        f"extension bar trigger $max_loss x {risk.extension_bar_trigger_multiple:.1f} = "
        f"${extension_threshold:.0f} | "
        f"max position value ${risk.max_position_value_usd:,.0f}"
    )


def _print_halt_flag(risk_engine: RiskEngine) -> None:
    """Print halt-flag file contents if any; friendly note when absent."""
    record = read_halt_flag(risk_engine.halt_flag_path)
    if record is None:
        _console.print("[green]No halt flag on disk.[/green]")
        return
    _console.print(
        f"[yellow]Halt flag:[/yellow] date={record.date.isoformat()} "
        f"reason={record.reason} pnl_at_halt=${record.pnl_at_halt:.2f} "
        f"triggered_at={record.triggered_at.isoformat()}"
    )


def _parse_dry_run_signal(spec: str) -> Signal:
    """Parse a ``--dry-run-signal SYMBOL:entry:stop:scale_out[:strategy]`` smoke-test spec.

    The fourth field is the strategy's scale-out price (Phase 4i: the
    2:1 anchor). Phase 4i: strategies emit ``runner_target_price=None``
    and the executor populates the bracket's runner LMT only when
    ``execution.runner_target_enabled`` is true — this CLI mirrors that
    by leaving ``runner_target_price`` as None. To showcase a 3-leg
    bracket, toggle ``execution.runner_target_enabled=true`` (config
    or env).
    """
    parts = spec.split(":")
    if len(parts) < 4:
        raise typer.BadParameter(
            "Expected SYMBOL:entry:stop:scale_out[:strategy], e.g. AMC:10.5:9.8:12.0:gap_and_go"
        )
    symbol, entry_s, stop_s, scale_s = parts[:4]
    strategy = parts[4] if len(parts) >= 5 else "gap_and_go"
    try:
        entry = float(entry_s)
        stop = float(stop_s)
        scale_out = float(scale_s)
    except ValueError as exc:
        raise typer.BadParameter(f"Non-numeric price in --dry-run-signal: {spec!r}") from exc
    return Signal(
        symbol=symbol.upper(),
        strategy=strategy,
        entry=entry,
        stop=stop,
        scale_out_price=scale_out,
        runner_target_price=None,
        timestamp=datetime.now(),
        reasons=["dry_run_signal"],
    )


def _print_positions_table(positions: list[Position]) -> None:
    """Render active positions as a rich table; empty case prints a friendly note."""
    if not positions:
        _console.print("[green]No active positions.[/green]")
        return
    table = Table(title="Active positions", show_lines=False)
    table.add_column("Symbol", no_wrap=True)
    table.add_column("Adopted", no_wrap=True)
    table.add_column("Strategy", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Shares", justify="right", no_wrap=True)
    table.add_column("Entry", justify="right", no_wrap=True)
    table.add_column("Stop", justify="right", no_wrap=True)
    table.add_column("Scale", justify="right", no_wrap=True)
    table.add_column("Runner", justify="right", no_wrap=True)
    for position in positions:
        adopted_cell = "[yellow]ADOPTED[/yellow]" if position.adopted_from_reconcile else "-"
        # Phase 4h: append the trail-conversion trigger when this is an
        # adjustable stop, so operators can see at a glance that the stop will
        # auto-convert to a TRAIL when price tags the given price.
        # Phase 4i: mark the tail with a [post-scaleout] tag once the first
        # half has been banked so red-candle suppression is visible in status.
        if position.stop_price:
            stop_cell = f"{position.stop_price:.2f}"
            if (
                position.post_scaleout_stop_type == "adjustable_to_trail"
                and position.post_scaleout_adjustment_trigger_price is not None
            ):
                stop_cell += f" (→ TRAIL @ {position.post_scaleout_adjustment_trigger_price:.2f})"
        else:
            stop_cell = "-"
        symbol_cell = position.symbol
        if position.scaled_out:
            symbol_cell += " [magenta][post-scaleout][/magenta]"
        # Phase 4j: a ``pending_entry_trigger`` bracket's parent is resting
        # on IBKR's servers waiting for the breakout tick; surface that
        # state distinctly so the operator doesn't read avg/scale/runner
        # columns as "already in the trade". The trigger price lives on
        # ``entry_trigger_price`` so the actual stop column still reads
        # the real stop for fresh placements.
        if position.status == "pending_entry_trigger":
            symbol_cell += f" [cyan][PENDING_ENTRY @ ${position.entry_trigger_price:.2f}][/cyan]"
            entry_cell = "pending"
        else:
            entry_cell = f"{position.avg_price:.2f}" if position.avg_price else "-"
        table.add_row(
            symbol_cell,
            adopted_cell,
            position.strategy,
            position.status,
            str(position.shares),
            entry_cell,
            stop_cell,
            f"{position.scale_out_price:.2f}" if position.scale_out_price else "-",
            f"{position.runner_target_price:.2f}" if position.runner_target_price else "-",
        )
    _console.print(table)


def _print_reentries_table(store: PositionStore, settings: Settings) -> None:
    """Render the Phase 4d per-symbol re-entry history — one row per symbol seen today.

    Columns: Symbol, Entries (n/max), Last exit type, Last PnL, Cooldown left.
    When no symbols have recorded any entries the table is suppressed so
    ``status`` stays terse on cold mornings.
    """
    histories = store.list_symbol_histories()
    if not histories:
        return
    cfg = settings.risk.re_entry
    table = Table(title="Re-entries today", show_lines=False)
    table.add_column("Symbol", no_wrap=True)
    table.add_column("Entries", justify="right", no_wrap=True)
    table.add_column("Last exit", no_wrap=True)
    table.add_column("Last PnL", justify="right", no_wrap=True)
    table.add_column("Cooldown left", justify="right", no_wrap=True)
    now = datetime.now(UTC)
    for history in histories:
        entries_cell = f"{history.entries_count}/{cfg.max_entries_per_symbol}"
        last_type = history.last_exit_type or "-"
        if history.last_exit_pnl is None:
            pnl_cell = "-"
        else:
            sign = "+" if history.last_exit_pnl >= 0 else "-"
            pnl_cell = f"{sign}${abs(history.last_exit_pnl):.2f}"
        if history.last_exit_time is None:
            cooldown_cell = "-"
        else:
            elapsed = (now - history.last_exit_time).total_seconds()
            remaining = max(0, int(cfg.cooldown_seconds - elapsed))
            cooldown_cell = f"{remaining}s" if remaining > 0 else "0s"
        table.add_row(history.symbol, entries_cell, last_type, pnl_cell, cooldown_cell)
    _console.print(table)


def _apply_simulate_reentry(
    *,
    store: PositionStore,
    symbol: str,
    n: int,
    settings: Settings,
) -> None:
    """Seed ``SymbolHistory`` with N prior profitable entries for the demo.

    Mutates the store's history for ``symbol`` so the next ``--dry-run-signal``
    exercises the N+1th entry's size multiplier. ``last_exit_time`` is pushed
    past the cooldown so the gate does not reject on ``reentry_cooldown_active``.
    ``last_exit_pnl`` is +$1.00 (profitable) so ``require_profitable_prior_exit``
    passes. Paper-only — the CLI guard refuses this on a live account.
    """
    cfg = settings.risk.re_entry
    history = store.symbol_history(symbol)
    history.entries_count = n
    if n > 0:
        history.last_exit_time = datetime.now(UTC) - timedelta(seconds=cfg.cooldown_seconds + 1)
        history.last_exit_pnl = 1.0
        history.last_exit_type = "target_hit"
    _console.print(
        f"[yellow]--simulate-reentry[/yellow] seeded {symbol} with {n} prior "
        f"entries (last exit: target_hit +$1.00, cooldown satisfied). "
        f"Next entry will use multiplier {cfg.size_multipliers[min(n, len(cfg.size_multipliers) - 1)]:.2f}."
    )
    _ = history  # silence unused-variable warnings in future refactors


@dataclass
class _SuggestCapsStats:
    """Raw aggregates produced from journal rows — fed into the advisory heuristic."""

    total_trades: int
    winners: int
    losers: int
    avg_win_usd: float
    avg_loss_usd: float
    worst_loss_usd: float
    best_day_usd: float
    worst_day_usd: float
    avg_trades_per_day: float
    sessions: int


@dataclass
class _SuggestedCaps:
    """Advisory numbers; never written to config.yaml."""

    max_loss_per_trade_usd: float
    max_daily_loss_usd: float
    daily_profit_goal_usd: float
    max_trades_per_day: int


def _compute_suggest_caps_stats(
    trades: list[Any],
    *,
    timezone: str,
) -> _SuggestCapsStats:
    """Walk the closed-trade list once and produce the aggregates.

    ``trades`` are ``TradeRecord`` rows (already filtered to the window).
    Daily sums use the NY-local session date so a late-in-the-day close
    lands on the correct session. An empty winners or losers list
    produces 0.0 averages — the caller already guards against "no
    trades" before invoking this helper.
    """
    daily_totals = aggregate_daily_pnl(trades, timezone)
    winners = [float(row.pnl) for row in trades if row.pnl is not None and row.pnl > 0]
    losers = [float(row.pnl) for row in trades if row.pnl is not None and row.pnl < 0]
    sessions = len(daily_totals)
    best_day = max(daily_totals.values()) if daily_totals else 0.0
    worst_day = min(daily_totals.values()) if daily_totals else 0.0
    return _SuggestCapsStats(
        total_trades=len(trades),
        winners=len(winners),
        losers=len(losers),
        avg_win_usd=sum(winners) / len(winners) if winners else 0.0,
        avg_loss_usd=sum(losers) / len(losers) if losers else 0.0,
        worst_loss_usd=min(losers) if losers else 0.0,
        best_day_usd=best_day,
        worst_day_usd=worst_day,
        avg_trades_per_day=len(trades) / sessions if sessions else 0.0,
        sessions=sessions,
    )


def _suggested_caps(stats: _SuggestCapsStats) -> _SuggestedCaps:
    """Heuristic: scale observed loss/day stats into advisory caps.

    Rules (rounded to the nearest $25 so the numbers read like the round-number caps):
      * max_loss_per_trade_usd ≈ |avg loser| × 1.25 (20-ish % buffer).
      * max_daily_loss_usd ≈ |worst observed day| × 1.2, floored at
        3× suggested per-trade so a lucky-small drawdown doesn't
        under-size the daily cap.
      * daily_profit_goal_usd ≈ max(best observed day, 2× suggested
        daily loss) — the 2:1 R:R on the session.
      * max_trades_per_day = round(avg trades/day × 1.2), clamped
        to [3, 10] so a thin dataset doesn't suggest 1-trade days.
    """
    per_trade_raw = abs(stats.avg_loss_usd) * 1.25 if stats.losers else 50.0
    per_trade = max(25.0, _round_to_25(per_trade_raw))

    daily_loss_raw = max(abs(stats.worst_day_usd) * 1.2, per_trade * 3.0)
    daily_loss = max(50.0, _round_to_25(daily_loss_raw))

    profit_goal_raw = max(stats.best_day_usd, daily_loss * 2.0)
    profit_goal = max(100.0, _round_to_25(profit_goal_raw))

    trades_raw = int(round(stats.avg_trades_per_day * 1.2))
    trades_cap = max(3, min(10, trades_raw)) if stats.sessions else 5

    return _SuggestedCaps(
        max_loss_per_trade_usd=per_trade,
        max_daily_loss_usd=daily_loss,
        daily_profit_goal_usd=profit_goal,
        max_trades_per_day=trades_cap,
    )


def _round_to_25(value: float) -> float:
    """Round up to the nearest $25 so suggestions look like round-number caps."""
    return float(((int(value) + 24) // 25) * 25)


def _print_suggest_caps_stats(stats: _SuggestCapsStats, *, lookback_days: int) -> None:
    """Render the observed-stats table (left panel of the advisory output)."""
    table = Table(title=f"Observed stats (last {lookback_days} days)", show_lines=False)
    table.add_column("Metric", no_wrap=True)
    table.add_column("Value", justify="right", no_wrap=True)
    win_rate = stats.winners / stats.total_trades if stats.total_trades else 0.0
    table.add_row("Sessions", str(stats.sessions))
    table.add_row("Total trades", str(stats.total_trades))
    table.add_row("Win rate", f"{win_rate * 100:.1f}%")
    table.add_row("Avg win", f"${stats.avg_win_usd:+.2f}")
    table.add_row("Avg loss", f"${stats.avg_loss_usd:+.2f}")
    table.add_row("Worst single trade", f"${stats.worst_loss_usd:+.2f}")
    table.add_row("Best day", f"${stats.best_day_usd:+.2f}")
    table.add_row("Worst day", f"${stats.worst_day_usd:+.2f}")
    table.add_row("Avg trades/day", f"{stats.avg_trades_per_day:.2f}")
    _console.print(table)


def _print_suggested_caps(
    suggestions: _SuggestedCaps,
    *,
    settings: Settings,
    compare: bool,
) -> None:
    """Render the suggested-caps table; with ``compare`` adds current + delta columns."""
    title = "Suggested caps" + (" (vs current)" if compare else "")
    table = Table(title=title, show_lines=False)
    table.add_column("Field", no_wrap=True)
    table.add_column("Suggested", justify="right", no_wrap=True)
    if compare:
        table.add_column("Current", justify="right", no_wrap=True)
        table.add_column("Delta", justify="right", no_wrap=True)
    rows: list[tuple[str, float, float]] = [
        (
            "max_loss_per_trade_usd",
            suggestions.max_loss_per_trade_usd,
            settings.risk.max_loss_per_trade_usd,
        ),
        ("max_daily_loss_usd", suggestions.max_daily_loss_usd, settings.risk.max_daily_loss_usd),
        (
            "daily_profit_goal_usd",
            suggestions.daily_profit_goal_usd,
            settings.risk.daily_profit_goal_usd,
        ),
        (
            "max_trades_per_day",
            float(suggestions.max_trades_per_day),
            float(settings.risk.max_trades_per_day),
        ),
    ]
    for name, suggested, current in rows:
        if compare:
            delta = suggested - current
            sign = "+" if delta >= 0 else ""
            table.add_row(name, f"{suggested:g}", f"{current:g}", f"{sign}{delta:g}")
        else:
            table.add_row(name, f"{suggested:g}")
    _console.print(table)


def _print_rehab_status(
    *,
    settings: Settings,
    rehab_engine: RehabEngine,
    computed_tier: RehabTier,
    computed_reason: str,
    drawdown_usd: float,
    consecutive_red_days: int,
    lookback_days: int,
) -> None:
    """Render the rehab-status Rich tables: flag, caps, recovery target."""
    flag_record = read_rehab_flag(rehab_engine.flag_path)
    caps = rehab_engine.apply_to_caps(settings.risk)
    _console.print(
        Panel(
            "\n".join(
                [
                    f"Active tier: [bold]{caps.tier.value}[/bold]"
                    + (
                        f"  (trigger: {caps.trigger_reason})"
                        if caps.trigger_reason
                        else "  (rehab disabled)"
                        if not rehab_engine.enabled
                        else ""
                    ),
                    f"Today's computed tier: {computed_tier.value}  (reason: {computed_reason})",
                    f"Lookback: {lookback_days} days  |  "
                    f"Consecutive red days: {consecutive_red_days}  |  "
                    f"Cumulative drawdown: ${drawdown_usd:+.2f}",
                ]
            ),
            title="Rehab tier",
            border_style="yellow" if caps.tier is not RehabTier.NORMAL else "green",
        )
    )

    caps_table = Table(title="Effective caps", show_lines=False)
    caps_table.add_column("Field", no_wrap=True)
    caps_table.add_column("Base", justify="right", no_wrap=True)
    caps_table.add_column("Effective", justify="right", no_wrap=True)
    caps_table.add_row(
        "max_loss_per_trade_usd",
        f"${caps.base_max_loss_per_trade_usd:.0f}",
        f"${caps.max_loss_per_trade_usd:.0f}",
    )
    caps_table.add_row(
        "max_daily_loss_usd",
        f"${caps.base_max_daily_loss_usd:.0f}",
        f"${caps.max_daily_loss_usd:.0f}",
    )
    caps_table.add_row(
        "max_trades_per_day",
        str(caps.base_max_trades_per_day),
        str(caps.max_trades_per_day),
    )
    _console.print(caps_table)

    if flag_record is not None and flag_record.tier is not RehabTier.NORMAL:
        recovery_fraction = settings.risk.rehab.recovery_drawdown_recovered_fraction
        recovery_target_usd = abs(flag_record.drawdown_at_entry_usd) * recovery_fraction
        recovered_so_far = abs(flag_record.drawdown_at_entry_usd) - abs(drawdown_usd)
        remaining = max(0.0, recovery_target_usd - recovered_so_far)
        _console.print(
            f"[dim]Recovery target: ${recovery_target_usd:.2f} "
            f"({recovery_fraction * 100:.0f}% of entry drawdown "
            f"${flag_record.drawdown_at_entry_usd:.2f}). "
            f"Recovered so far: ${recovered_so_far:.2f}. "
            f"Remaining: ${remaining:.2f}.[/dim]"
        )
    elif flag_record is None:
        _console.print("[dim]No rehab flag on disk.[/dim]")


def _apply_simulate_red_days(
    *,
    rehab_engine: RehabEngine,
    n: int,
    settings: Settings,
) -> None:
    """Register N synthetic back-to-back losing days on the ``RehabEngine``.

    Each synthetic day is ``-max_daily_loss_usd`` — the worst-case red day
    under the active config — so the engine naturally trips REHAB or
    DEEP_REHAB depending on how ``n`` stacks against
    ``rehab_consecutive_red_days`` + ``deep_rehab_consecutive_red_days``.
    Days are dated ``today - i`` (oldest first) so the tier computation
    sees the same ordering a real journal would produce.
    """
    today_ny = datetime.now(ZoneInfo(settings.session.timezone)).date()
    loss_per_day = -abs(settings.risk.max_daily_loss_usd)
    synthetic: list[tuple[date_cls, float]] = [
        (today_ny - timedelta(days=offset), loss_per_day) for offset in range(n, 0, -1)
    ]
    rehab_engine.set_simulation_override(synthetic)
    _console.print(
        f"[yellow]--simulate-red-days[/yellow] seeded {n} synthetic "
        f"{loss_per_day:+.0f}$ losing days (oldest→newest)."
    )


def _apply_simulate_config_override(settings: Settings, overrides: list[str]) -> Settings:
    """Re-instantiate ``settings.execution`` with ``key=value`` overrides applied.

    Each override must be ``execution.<field>=<value>`` or just
    ``<field>=<value>`` (the ``execution.`` prefix is accepted for clarity).
    Values are passed as strings to pydantic which handles coercion +
    re-runs the ExecutionConfig validators (so a <1R override still errors).
    Other config sections aren't touched; Phase 4e's knob is execution-only.
    Paper-only — the CLI guard refuses this on a live account.
    """
    overrides_kv: dict[str, str] = {}
    for spec in overrides:
        if "=" not in spec:
            raise typer.BadParameter(f"--simulate-config-override expects key=value, got {spec!r}.")
        key, value = spec.split("=", 1)
        key = key.strip()
        if key.startswith("execution."):
            key = key[len("execution.") :]
        overrides_kv[key] = value.strip()
    merged: dict[str, Any] = settings.execution.model_dump()
    for k, v in overrides_kv.items():
        merged[k] = _coerce_execution_field(settings.execution, k, v)
    # Fresh instantiation re-runs the validators so <1R still errors out.
    new_execution = ExecutionConfig(**merged)
    summary = ", ".join(f"{k}={v}" for k, v in overrides_kv.items())
    _console.print(f"[yellow]--simulate-config-override[/yellow] execution: {summary}")
    return settings.model_copy(update={"execution": new_execution})


def _coerce_execution_field(current: ExecutionConfig, key: str, raw: str) -> object:
    """Cast ``raw`` to the type of ``current.<key>`` (float/int/bool/str).

    Unknown fields raise a typer BadParameter so a typo doesn't silently
    add a stray attribute to the config object.
    """
    if key not in ExecutionConfig.model_fields:
        raise typer.BadParameter(
            f"Unknown execution field {key!r}; "
            f"valid fields: {', '.join(sorted(ExecutionConfig.model_fields))}."
        )
    current_value = getattr(current, key)
    if isinstance(current_value, bool):
        return raw.lower() in {"1", "true", "yes", "on"}
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(raw)
    if isinstance(current_value, float):
        return float(raw)
    return raw


def _run_with_connection_handling(
    coro_factory: Callable[[], Coroutine[Any, Any, None]],
) -> None:
    """Run an async entrypoint, surfacing a friendly message for common TWS / config errors."""
    try:
        asyncio.run(coro_factory())
    except ConnectionRefusedError:
        settings = get_settings()
        typer.echo(
            f"Could not connect to TWS on {settings.ibkr.host}:{settings.ibkr.port}. "
            "Is TWS running with the API enabled?",
            err=True,
        )
        raise typer.Exit(code=1) from None
    except ConfigurationError as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(code=1) from None


def _print_summary(summary: dict[str, str]) -> None:
    """Render the Phase 1 account-summary fields as a fixed-width table."""
    width = max(len(tag) for tag in _PING_TAGS)
    typer.echo("")
    typer.echo("Account summary")
    typer.echo("-" * (width + 24))
    for tag in _PING_TAGS:
        value = summary.get(tag, "(not reported)")
        typer.echo(f"{tag:<{width}}  {value}")
    typer.echo("")


def _print_scan_table(hits: list[ScanHit]) -> None:
    """Render the scan hits to the terminal via ``rich.table.Table``."""
    if not hits:
        _console.print("[yellow]No hits passed the 5-Pillar filters.[/yellow]")
        return
    table = Table(title="Morning Watchlist", show_lines=False)
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("Symbol", no_wrap=True)
    table.add_column("Change %", justify="right", no_wrap=True)
    table.add_column("Price", justify="right", no_wrap=True)
    table.add_column("Float", justify="right", no_wrap=True)
    table.add_column("Catalyst", no_wrap=True)
    table.add_column("Reasons", overflow="fold")
    for index, hit in enumerate(hits, start=1):
        table.add_row(
            str(index),
            f"${hit.symbol}",
            f"{hit.change_pct:+.1f}%" if hit.change_pct is not None else "-",
            f"${hit.price:.2f}" if hit.price is not None else "-",
            _format_float_cell(hit.float_shares, hit.float_source),
            hit.catalyst or "-",
            ", ".join(hit.reasons) if hit.reasons else "-",
        )
    _console.print(table)


def _format_float_cell(value: int | None, source: str | None) -> str:
    """Render the Float cell with provenance suffix: ``3.2M (yf)`` or ``11M (fh_out*)`` or ``?``."""
    if value is None:
        return "?"
    return f"{_format_shares(value)} ({_float_source_tag(source)})"


def _float_source_tag(source: str | None) -> str:
    """Short provenance tag; asterisk marks the Finnhub fallback for at-a-glance review."""
    if source == SOURCE_YFINANCE:
        return "yf"
    if source == SOURCE_FINNHUB_FALLBACK:
        return "fh_out*"
    return "?"


def _format_shares(value: int | None) -> str:
    """Compact share-count rendering for the CLI table."""
    if value is None:
        return "?"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return str(value)


if __name__ == "__main__":
    app()

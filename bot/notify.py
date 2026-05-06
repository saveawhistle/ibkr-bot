"""Telegram notifier for the morning watchlist.

Uses python-telegram-bot's ``Bot`` class with MarkdownV2 formatting. Credentials
are optional — if ``telegram_bot_token`` or ``telegram_chat_id`` is missing, the
notifier logs a warning and returns silently. Telegram-side errors (network,
bad chat id) are logged but never propagated to the scanner.

Phase 10.1 adds an ack flow used by the naked-position watchdog:

* :meth:`Notifier.send_alert_with_ack` posts a message with a single inline-
  keyboard "Ack" button whose ``callback_data`` is the caller-supplied
  ``ack_id``.
* :meth:`Notifier.start_ack_listener` spawns a background task that polls
  Telegram for ``callback_query`` updates and dispatches matching ack ids
  into an in-memory registry.
* :meth:`Notifier.is_alert_acked` / :meth:`Notifier.clear_alert_ack` /
  :meth:`Notifier.mark_alert_acked` expose the registry so the watchdog
  (or tests) can read / re-arm without going through Telegram.

The ack registry is in-memory only by design — bot restart re-evaluates
everything from scratch, matching the watchdog's documented re-arm rules.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import structlog
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.error import TelegramError
from telegram.helpers import escape_markdown

from bot.config import Settings, get_settings

if TYPE_CHECKING:
    from bot.execution.position_state import Position
    from bot.scanning.scanner import ScanHit
    from bot.strategies.base import Signal

_log = structlog.get_logger("bot.notify")

# Phase 10.1 — prefix on watchdog inline-keyboard callback_data values.
# The ack listener filters callback queries by this prefix so a future
# unrelated keyboard handler (different feature) doesn't clobber the
# watchdog registry.
_WATCHDOG_ACK_CALLBACK_PREFIX = "watchdog:"

# Phase 10.1 — ack-listener long-poll timeout (seconds). 30 s is the
# python-telegram-bot recommended value for low-latency callback delivery
# without burning request quota; the listener wakes on any update or on
# stop_ack_listener cancellation.
_ACK_LISTENER_LONG_POLL_SECONDS = 30


def _format_float_shares(float_shares: int | None) -> str:
    """Render a share count as a short ``3.2M``/``450K``/``?`` string for display."""
    if float_shares is None:
        return "?"
    if float_shares >= 1_000_000:
        return f"{float_shares / 1_000_000:.1f}M"
    if float_shares >= 1_000:
        return f"{float_shares / 1_000:.0f}K"
    return str(float_shares)


class Notifier:
    """Wraps python-telegram-bot's ``Bot`` to push the watchlist to a configured chat."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        bot: Bot | None = None,
    ) -> None:
        """Resolve credentials from settings; optionally accept an injected ``Bot`` for tests."""
        resolved = settings or get_settings()
        self._settings = resolved
        self._token = resolved.data_sources.telegram_bot_token
        self._chat_id = resolved.data_sources.telegram_chat_id
        self._injected_bot = bot
        # Phase 10.1 — ack registry + listener task. Both are populated only
        # when start_ack_listener is invoked (or tests call mark_alert_acked
        # directly); a notifier without ack support behaves identically to
        # the pre-10.1 fire-and-forget path.
        self._acked_ids: set[str] = set()
        self._ack_timestamps: dict[str, datetime] = {}
        self._listener_task: asyncio.Task[None] | None = None
        self._listener_stop = asyncio.Event()
        self._listener_offset = 0

    async def send_watchlist(self, hits: list[ScanHit]) -> None:
        """Send the ranked watchlist to Telegram; silently skip when creds are missing."""
        await self._send(self._format(hits))

    async def send_signal(self, signal: Signal) -> None:
        """Push a single strategy signal (entry/stop/target/R:R) to Telegram."""
        await self._send(self._format_signal(signal))

    async def send_fill(
        self,
        position: Position,
        fill_type: str,
        *,
        entry_number: int | None = None,
    ) -> None:
        """Notify on an executor fill event (``entry``, ``stop``, or ``target``).

        Exit fills (``stop`` / ``target``) include realized PnL; entry fills
        show the executed price + shares. ``entry_number`` (Phase 4d) is
        1-indexed and only rendered on ``entry`` fills — passed by the
        executor from ``SymbolHistory.entries_count``. Missing credentials
        silently no-op, matching the rest of the notifier's
        graceful-degradation policy.
        """
        await self._send(self._format_fill(position, fill_type, entry_number=entry_number))

    async def send_rehab_tier_change(
        self,
        *,
        old: str,
        new: str,
        reason: str,
    ) -> None:
        """Push a Phase 4g rehab-tier change (NORMAL ↔ REHAB ↔ DEEP_REHAB).

        Fires on both upgrades (cold-streak or drawdown trip) and
        downgrades (recovery threshold met). ``old`` / ``new`` are the
        ``RehabTier`` values, ``reason`` is the kebab trigger tag
        (``consecutive_red_days``, ``cumulative_drawdown``, ``recovery``).
        Missing credentials silently no-op, same as every other notifier.
        """
        await self._send(self._format_rehab_tier_change(old=old, new=new, reason=reason))

    async def send_halt(self, reason: str, pnl: float) -> None:
        """Push a halt notification — daily loss, profit goal, or give-back.

        Reason is the RiskEngine's kebab-case label (``daily_loss_limit``,
        ``daily_profit_goal``, ``giveback_limit``); we humanise it for the
        message but keep the raw string for log-side correlation.
        """
        await self._send(self._format_halt(reason, pnl))

    async def send_daily_summary(
        self,
        *,
        trades: int,
        realized_pnl: float,
        peak_pnl: float,
        reached_goal: bool,
    ) -> None:
        """End-of-session recap: trade count, realized PnL, peak, goal-hit flag."""
        await self._send(
            self._format_daily_summary(
                trades=trades,
                realized_pnl=realized_pnl,
                peak_pnl=peak_pnl,
                reached_goal=reached_goal,
            )
        )

    async def send_alert_with_ack(self, *, text: str, ack_id: str) -> None:
        """Phase 10.1 — send a Telegram alert with a single inline-keyboard "Ack" button.

        ``ack_id`` is the value the listener will record into the ack
        registry when the operator taps the button. The watchdog reads it
        back via :meth:`is_alert_acked`. The button label is fixed (single
        button per alert by spec); callers who need richer interactions
        should extend rather than reuse this surface.

        The text is sent as plain (non-MarkdownV2) text on this path so
        the watchdog's structured payload (which contains literal ``$``,
        ``.``, parentheses, etc. from share counts and prices) doesn't
        require defensive escaping. MarkdownV2 stays the default for the
        watchlist + signal + fill paths.
        """
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("Ack", callback_data=ack_id)]])
        await self._send(text, reply_markup=keyboard, parse_mode=None)

    def is_alert_acked(self, ack_id: str) -> bool:
        """Phase 10.1 — True iff the operator has tapped Ack on the alert with this id."""
        return ack_id in self._acked_ids

    def mark_alert_acked(self, ack_id: str) -> None:
        """Phase 10.1 — record an ack. Called by the listener loop and by tests."""
        self._acked_ids.add(ack_id)
        self._ack_timestamps[ack_id] = datetime.now(UTC)

    def clear_alert_ack(self, ack_id: str) -> None:
        """Phase 10.1 — drop an ack record. Watchdog calls this on re-arm."""
        self._acked_ids.discard(ack_id)
        self._ack_timestamps.pop(ack_id, None)

    def ack_timestamp(self, ack_id: str) -> datetime | None:
        """Phase 10.1 — return when an ack was recorded, or None. Forensics aid."""
        return self._ack_timestamps.get(ack_id)

    async def start_ack_listener(self) -> None:
        """Phase 10.1 — spawn the background callback-query polling task.

        No-ops when credentials are missing, an injected (test) bot is in
        use, or a listener is already running. Idempotent — safe to call
        multiple times. Cancellable via :meth:`stop_ack_listener`.
        """
        if self._listener_task is not None and not self._listener_task.done():
            return
        if self._injected_bot is not None:
            # Tests drive the registry directly via ``mark_alert_acked``;
            # there's no real Telegram socket to long-poll.
            return
        if not self._token or not self._chat_id:
            _log.warning(
                "notify.ack_listener_skipped",
                reason="missing_credentials",
            )
            return
        self._listener_stop.clear()
        self._listener_task = asyncio.create_task(self._ack_listener_loop())
        _log.info("notify.ack_listener_started")

    async def stop_ack_listener(self) -> None:
        """Phase 10.1 — request the listener to exit and await its termination.

        Sets the stop event (the polling loop checks between long-poll
        windows) and cancels the task as a fallback so a wedged
        ``get_updates`` doesn't block shutdown.
        """
        self._listener_stop.set()
        task = self._listener_task
        if task is None:
            return
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(task, timeout=5.0)
        self._listener_task = None
        _log.info("notify.ack_listener_stopped")

    async def _ack_listener_loop(self) -> None:
        """Long-poll Telegram for callback queries and dispatch matching ack ids.

        Filters on ``allowed_updates=["callback_query"]`` so other chat
        traffic (which we don't expect on a private bot, but defensive)
        doesn't waste cycles. The listener answers each callback so the
        Telegram client clears the loading spinner promptly. ``offset``
        is per-instance state — a restart re-fetches the unprocessed
        backlog (Telegram retains updates for ~24 h).

        Caller (``start_ack_listener``) guarantees ``self._token`` is a
        real string; the assertion narrows the Optional for type checkers.
        """
        assert self._token is not None  # noqa: S101 - narrowing; checked by caller
        bot = Bot(token=self._token)
        try:
            await bot.initialize()
            while not self._listener_stop.is_set():
                try:
                    updates = await bot.get_updates(
                        offset=self._listener_offset,
                        timeout=_ACK_LISTENER_LONG_POLL_SECONDS,
                        allowed_updates=["callback_query"],
                    )
                except (TimeoutError, TelegramError) as exc:
                    # Network blips / transient errors must not crash the
                    # listener — back off briefly and retry.
                    _log.warning("notify.ack_listener_poll_failed", error=str(exc))
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(self._listener_stop.wait(), timeout=5.0)
                    continue
                except Exception as exc:  # noqa: BLE001 - never crash the loop
                    _log.error("notify.ack_listener_unexpected_error", error=str(exc))
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(self._listener_stop.wait(), timeout=5.0)
                    continue
                for update in updates:
                    self._listener_offset = update.update_id + 1
                    cq = update.callback_query
                    if cq is None:
                        continue
                    data = cq.data or ""
                    if not data.startswith(_WATCHDOG_ACK_CALLBACK_PREFIX):
                        continue
                    self.mark_alert_acked(data)
                    _log.info("notify.ack_received", ack_id=data)
                    with contextlib.suppress(TelegramError, Exception):
                        await cq.answer("Acked")
        finally:
            with contextlib.suppress(Exception):
                await bot.shutdown()

    async def _send(
        self,
        text: str,
        *,
        reply_markup: InlineKeyboardMarkup | None = None,
        parse_mode: str | None = ParseMode.MARKDOWN_V2,
    ) -> None:
        """Shared dispatch: missing creds → warn-and-return, errors → log.

        Phase 10.1 adds optional ``reply_markup`` (used by the watchdog ack
        path) and a ``parse_mode`` override (the watchdog sends plain text
        because its payload contains literal ``$``/``.``/parentheses that
        would otherwise need MarkdownV2 escaping).
        """
        if not self._token or not self._chat_id:
            _log.warning(
                "notify.telegram_credentials_missing",
                hint="Set BOT_DATA_SOURCES__TELEGRAM_BOT_TOKEN and "
                "BOT_DATA_SOURCES__TELEGRAM_CHAT_ID to enable Telegram pushes.",
            )
            return
        try:
            if self._injected_bot is not None:
                await self._injected_bot.send_message(
                    chat_id=self._chat_id,
                    text=text,
                    parse_mode=parse_mode,
                    reply_markup=reply_markup,
                )
            else:
                async with Bot(token=self._token) as bot:
                    await bot.send_message(
                        chat_id=self._chat_id,
                        text=text,
                        parse_mode=parse_mode,
                        reply_markup=reply_markup,
                    )
        except TelegramError as exc:
            _log.error("notify.send_failed", error=str(exc))
        except Exception as exc:  # noqa: BLE001 - network etc. must not crash the scanner
            _log.error("notify.unexpected_error", error=str(exc))

    def _format(self, hits: list[ScanHit]) -> str:
        """Render hits as a MarkdownV2-escaped Telegram message."""
        tz = ZoneInfo(self._settings.session.timezone)
        now_local = datetime.now(tz)
        header = f"📋 Morning Watchlist — {now_local.strftime('%Y-%m-%d %H:%M')} ET"
        bar = "━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        lines: list[str] = [header, bar]
        if not hits:
            lines.append("(no hits passed the 5-Pillar filters)")
        for index, hit in enumerate(hits, start=1):
            parts = [f"{index}. ${hit.symbol}"]
            if hit.change_pct is not None:
                parts.append(f"{hit.change_pct:+.1f}%")
            if hit.price is not None:
                parts.append(f"${hit.price:.2f}")
            parts.append(f"Float {_format_float_shares(hit.float_shares)}")
            tag = self._tag_for(hit)
            if tag:
                parts.append(tag)
            lines.append("  ".join(parts))
        raw = "\n".join(lines)
        return escape_markdown(raw, version=2)

    def _format_signal(self, signal: Signal) -> str:
        """Render one Signal as a MarkdownV2-escaped Telegram message.

        Phase 4i: ``runner_target_price`` is optional (strategies always
        leave it ``None``; the executor fills it in only when
        ``execution.runner_target_enabled`` is true). Render it only when
        present so the signal line stays readable for the the methodology-default
        no-runner-ceiling flow.
        """
        ts_local = signal.timestamp.strftime("%H:%M")
        price_parts = [
            f"Entry {signal.entry:.2f}",
            f"Stop {signal.stop:.2f}",
            f"Scale {signal.scale_out_price:.2f}",
        ]
        if signal.runner_target_price is not None:
            price_parts.append(f"Runner {signal.runner_target_price:.2f}")
        lines = [
            f"⚡ {signal.strategy.upper()} — ${signal.symbol} @ {ts_local} ET",
            "  ".join(price_parts),
            f"R:R {signal.risk_reward:.2f}  Risk ${signal.risk_per_share:.2f}/sh",
        ]
        if signal.reasons:
            lines.append("Why: " + ", ".join(signal.reasons))
        return escape_markdown("\n".join(lines), version=2)

    def _format_fill(
        self,
        position: Position,
        fill_type: str,
        *,
        entry_number: int | None = None,
    ) -> str:
        """Render a bracket fill event — entry prints price+shares, exits include PnL.

        Phase 4h: ``scale_out`` is a new fill-type rendered mid-lifetime (not
        a close) that shows the partial-fill dollars banked + which post-
        scale-out stop was installed. Adjustable stops append the
        server-side TRAIL conversion trigger so operators can read the
        next decision point straight off the Telegram message.
        """
        kind = fill_type.lower()
        emoji = {
            "entry": "🟢",
            "target": "🎯",
            "stop": "🔴",
            "scale_out": "💰",
        }.get(kind, "ℹ️")
        entry_tag = f" — Entry #{entry_number}" if kind == "entry" and entry_number else ""
        header = f"{emoji} {kind.upper()} FILL{entry_tag} — ${position.symbol}"
        lines = [header]
        if kind == "entry":
            entry_parts = [
                f"Filled {position.shares} @ {position.avg_price:.2f}",
                f"stop {position.stop_price:.2f}",
                f"scale {position.scale_out_price:.2f}",
            ]
            if position.runner_target_price is not None:
                entry_parts.append(f"runner {position.runner_target_price:.2f}")
            else:
                entry_parts.append("runner none (trails post-scale)")
            lines.append("  ".join(entry_parts))
        elif kind == "scale_out":
            pnl = position.scale_partial_pnl
            pnl_sign = "+" if pnl >= 0 else "-"
            lines.append(
                f"Banked {pnl_sign}${abs(pnl):.2f}  "
                f"Remaining {position.shares}  "
                f"New stop {position.stop_price:.2f}"
            )
            if (
                position.post_scaleout_stop_type == "adjustable_to_trail"
                and position.post_scaleout_adjustment_trigger_price is not None
            ):
                lines.append(
                    f"Post-scale-out protection: breakeven stop + auto-TRAIL @ "
                    f"${position.post_scaleout_adjustment_trigger_price:.2f}"
                )
            elif position.post_scaleout_stop_type == "static_breakeven":
                lines.append(
                    f"Post-scale-out protection: static breakeven stop @ ${position.stop_price:.2f}"
                )
            # Phase 4i: tail behaviour — advertise the
            # red-candle suppression so operators see *why* the bot isn't
            # panicking on a single red bar.
            if position.red_candle_exit_suppressed:
                lines.append(
                    "Tail: red-candle exit suppressed (the methodology: hold through "
                    "reds while breakeven stop holds)"
                )
        else:
            exit_price = position.exit_price or position.avg_price
            pnl = position.realized_pnl
            pnl_sign = "+" if pnl >= 0 else "-"
            lines.append(
                f"Exit {exit_price:.2f}  Shares {position.shares}  PnL {pnl_sign}${abs(pnl):.2f}"
            )
            lines.append(f"Strategy {position.strategy}")
        return escape_markdown("\n".join(lines), version=2)

    def _format_rehab_tier_change(self, *, old: str, new: str, reason: str) -> str:
        """Render a rehab-tier transition with a per-direction emoji + the rule."""
        if new == "DEEP_REHAB":
            emoji = "🎯"
            tagline = "the methodology: trade one setup a day, earn your way back."
        elif new == "REHAB":
            emoji = "🛟"
            tagline = "the methodology: when you're in a drawdown, trade smaller."
        else:  # recovery back to NORMAL
            emoji = "✅"
            tagline = "Recovery threshold met — base caps restored."
        pretty_reason = {
            "consecutive_red_days": "cold streak",
            "cumulative_drawdown": "drawdown threshold",
            "recovery": "drawdown recovered",
        }.get(reason, reason)
        lines = [
            f"{emoji} Rehab tier: {old} → {new}",
            f"Trigger: {pretty_reason}",
            tagline,
        ]
        return escape_markdown("\n".join(lines), version=2)

    def _format_halt(self, reason: str, pnl: float) -> str:
        """Render a halt alert with the RiskEngine reason + current realized PnL."""
        pretty = {
            "daily_loss_limit": "🛑 Daily loss limit hit",
            "daily_profit_goal": "🎯 Daily profit goal reached",
            "giveback_limit": "↩️ Give-back guard tripped",
        }.get(reason, f"⚠️ Halt: {reason}")
        sign = "+" if pnl >= 0 else "-"
        lines = [
            pretty,
            f"Realized PnL: {sign}${abs(pnl):.2f}",
            "New entries blocked; existing brackets still run to their stops.",
            "Run `python -m bot reset-halt` after session close to clear.",
        ]
        return escape_markdown("\n".join(lines), version=2)

    def _format_daily_summary(
        self,
        *,
        trades: int,
        realized_pnl: float,
        peak_pnl: float,
        reached_goal: bool,
    ) -> str:
        """Render the end-of-day recap — trades, PnL, peak PnL, goal flag."""
        sign = "+" if realized_pnl >= 0 else "-"
        peak_sign = "+" if peak_pnl >= 0 else "-"
        goal = "✅ Goal reached" if reached_goal else "—"
        lines = [
            "📊 Daily Summary",
            f"Trades: {trades}",
            f"Realized PnL: {sign}${abs(realized_pnl):.2f}",
            f"Peak PnL: {peak_sign}${abs(peak_pnl):.2f}",
            f"Profit goal: {goal}",
        ]
        return escape_markdown("\n".join(lines), version=2)

    @staticmethod
    def _tag_for(hit: ScanHit) -> str:
        """Pick the display tag for a hit: catalyst category with fire emoji, else first reason."""
        if hit.catalyst:
            emoji = "🔥 " if hit.catalyst == "earnings_beat" else ""
            return f"{emoji}{hit.catalyst}"
        if hit.reasons:
            return hit.reasons[0]
        return ""

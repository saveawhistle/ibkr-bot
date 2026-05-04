"""Tests for the Phase 10.1 ack-flow extension to ``bot.notify.Notifier``.

Covers:
* ``send_alert_with_ack`` builds a single-button inline keyboard with the
  caller-supplied ack id as ``callback_data``, and routes through ``_send``
  with parse_mode disabled (alerts contain literal ``$``/``.``/parens).
* ``mark_alert_acked`` / ``is_alert_acked`` / ``clear_alert_ack`` form a
  consistent in-memory registry with ack timestamps.
* The listener lifecycle (``start_ack_listener`` / ``stop_ack_listener``)
  no-ops when an injected bot is in use (tests drive the registry directly).

The deeper end-to-end test of the listener loop polling Telegram for
callback queries is intentionally out of scope here — that requires a
mock Telegram server. The watchdog tests exercise the consumer-side
behaviour through the public ``mark_alert_acked`` surface.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.config import DataSourcesSettings, Settings
from bot.notify import Notifier


def _settings_with_creds() -> Settings:
    """Settings with populated Telegram credentials (matches test_notify fixture)."""
    return Settings(
        data_sources=DataSourcesSettings(
            telegram_bot_token="fake-token",
            telegram_chat_id="42",
        )
    )


def _settings_without_creds() -> Settings:
    return Settings(data_sources=DataSourcesSettings())


@pytest.mark.asyncio
async def test_send_alert_with_ack_builds_inline_keyboard() -> None:
    """``send_alert_with_ack`` posts a message with a single-button keyboard.

    The button label is fixed ("Ack") per spec; the callback_data is the
    caller-supplied ack_id verbatim so the listener can dispatch it back.
    """
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()
    notifier = Notifier(settings=_settings_with_creds(), bot=mock_bot)
    await notifier.send_alert_with_ack(
        text="🟥 NAKED — $BIYA\nPosition: 41 shares",
        ack_id="watchdog:BIYA:NAKED",
    )
    mock_bot.send_message.assert_awaited_once()
    kwargs = mock_bot.send_message.await_args.kwargs
    assert kwargs["chat_id"] == "42"
    # Watchdog alerts are sent as plain text — the structured payload
    # contains literal ``$``, ``.``, and parens that would otherwise need
    # MarkdownV2 escaping.
    assert kwargs["parse_mode"] is None
    keyboard = kwargs["reply_markup"]
    assert keyboard is not None
    rows = keyboard.inline_keyboard
    assert len(rows) == 1
    assert len(rows[0]) == 1
    button = rows[0][0]
    assert button.text == "Ack"
    assert button.callback_data == "watchdog:BIYA:NAKED"


@pytest.mark.asyncio
async def test_send_alert_with_ack_skips_when_credentials_missing() -> None:
    """Missing credentials → no Telegram call, matches existing graceful-degradation policy."""
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()
    notifier = Notifier(settings=_settings_without_creds())
    notifier._injected_bot = mock_bot  # type: ignore[assignment]
    await notifier.send_alert_with_ack(text="hi", ack_id="watchdog:ABC:NAKED")
    mock_bot.send_message.assert_not_awaited()


def test_mark_and_is_alert_acked_round_trip() -> None:
    """Marking and reading an ack id flows through ``mark_alert_acked`` + ``is_alert_acked``."""
    notifier = Notifier(settings=_settings_with_creds(), bot=MagicMock())
    ack_id = "watchdog:BIYA:NAKED"
    assert notifier.is_alert_acked(ack_id) is False
    notifier.mark_alert_acked(ack_id)
    assert notifier.is_alert_acked(ack_id) is True


def test_clear_alert_ack_drops_registration() -> None:
    """``clear_alert_ack`` removes the id from the registry and the timestamp map."""
    notifier = Notifier(settings=_settings_with_creds(), bot=MagicMock())
    ack_id = "watchdog:ABC:UNDERPROTECTED"
    notifier.mark_alert_acked(ack_id)
    assert notifier.is_alert_acked(ack_id)
    notifier.clear_alert_ack(ack_id)
    assert notifier.is_alert_acked(ack_id) is False
    assert notifier.ack_timestamp(ack_id) is None


def test_ack_timestamp_returned_after_mark() -> None:
    """``ack_timestamp`` returns when the ack was recorded; None for unknown ids."""
    notifier = Notifier(settings=_settings_with_creds(), bot=MagicMock())
    before = datetime.now(UTC)
    notifier.mark_alert_acked("watchdog:X:NAKED")
    after = datetime.now(UTC)
    ts = notifier.ack_timestamp("watchdog:X:NAKED")
    assert ts is not None
    assert before <= ts <= after
    assert notifier.ack_timestamp("watchdog:Y:NAKED") is None


def test_clear_alert_ack_on_unknown_id_is_a_noop() -> None:
    """Clearing an unknown ack id must not raise — used in re-arm paths defensively."""
    notifier = Notifier(settings=_settings_with_creds(), bot=MagicMock())
    notifier.clear_alert_ack("watchdog:NEVER_SEEN:NAKED")  # no-op


@pytest.mark.asyncio
async def test_start_ack_listener_noops_with_injected_bot() -> None:
    """An injected (test) bot disables the listener — tests use ``mark_alert_acked`` directly."""
    notifier = Notifier(settings=_settings_with_creds(), bot=MagicMock())
    await notifier.start_ack_listener()
    assert notifier._listener_task is None


@pytest.mark.asyncio
async def test_start_ack_listener_noops_without_credentials() -> None:
    """Without Telegram credentials there's nothing to long-poll — listener does not start."""
    notifier = Notifier(settings=_settings_without_creds())
    await notifier.start_ack_listener()
    assert notifier._listener_task is None


@pytest.mark.asyncio
async def test_stop_ack_listener_when_never_started_is_safe() -> None:
    """``stop_ack_listener`` is idempotent; safe to call when nothing was started."""
    notifier = Notifier(settings=_settings_with_creds(), bot=MagicMock())
    await notifier.stop_ack_listener()  # no-op


@pytest.mark.asyncio
async def test_send_alert_with_ack_uses_persistent_path_without_injected_bot() -> None:
    """Without an injected bot the notifier opens a fresh ``Bot`` context per send.

    Verified indirectly: with creds + no injected_bot the call shouldn't
    raise (the actual Telegram API call is mocked at the module level
    elsewhere; here we just assert the dispatch path doesn't hit the
    "missing credentials" early-return).
    """
    notifier = Notifier(settings=_settings_with_creds())
    # Patch the Bot factory so we don't try to talk to Telegram.
    bot_mock = MagicMock()
    bot_mock.send_message = AsyncMock()
    bot_mock.__aenter__ = AsyncMock(return_value=bot_mock)
    bot_mock.__aexit__ = AsyncMock(return_value=None)
    import bot.notify as notify_mod  # noqa: PLC0415

    original_bot = notify_mod.Bot
    notify_mod.Bot = MagicMock(return_value=bot_mock)  # type: ignore[misc]
    try:
        await notifier.send_alert_with_ack(text="hi", ack_id="watchdog:Z:NAKED")
        bot_mock.send_message.assert_awaited_once()
        kwargs = bot_mock.send_message.await_args.kwargs
        assert kwargs["reply_markup"] is not None
        assert kwargs["parse_mode"] is None
    finally:
        notify_mod.Bot = original_bot  # type: ignore[misc]


def test_existing_send_paths_preserve_markdown_v2_default() -> None:
    """Phase 10.1 must not regress the watchlist/signal/fill paths to plain text.

    Verified by inspecting the ``_send`` default: parse_mode defaults to
    MARKDOWN_V2 unless the caller (the ack path) overrides it.
    """
    import inspect

    sig = inspect.signature(Notifier._send)
    parse_mode_param = sig.parameters["parse_mode"]
    # MARKDOWN_V2 enum string repr
    assert parse_mode_param.default is not None
    assert getattr(parse_mode_param.default, "name", "") == "MARKDOWN_V2"

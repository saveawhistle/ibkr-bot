"""Tests for ``bot.notify.Notifier``: formatting, sending, and graceful degradation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.error import NetworkError

from bot.config import DataSourcesSettings, Settings
from bot.notify import Notifier
from bot.scanning.scanner import ScanHit


def _settings_with_creds() -> Settings:
    """Build a Settings instance with populated Telegram credentials."""
    return Settings(
        data_sources=DataSourcesSettings(
            telegram_bot_token="fake-token",
            telegram_chat_id="42",
        )
    )


def _hits() -> list[ScanHit]:
    """Produce a pair of representative watchlist hits for format assertions."""
    return [
        ScanHit(
            symbol="ABCD",
            price=4.20,
            change_pct=18.5,
            volume=1_200_000,
            float_shares=3_200_000,
            catalyst="earnings_beat",
            news_items=[],
            reasons=[],
        ),
        ScanHit(
            symbol="WXYZ",
            price=None,
            change_pct=None,
            volume=None,
            float_shares=None,
            catalyst=None,
            news_items=[],
            reasons=["float_unknown", "no_catalyst"],
        ),
    ]


@pytest.mark.asyncio
async def test_missing_credentials_skips_send(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing token or chat id → warn + return silently, never raise."""
    monkeypatch.delenv("BOT_DATA_SOURCES__TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("BOT_DATA_SOURCES__TELEGRAM_CHAT_ID", raising=False)
    empty = Settings(data_sources=DataSourcesSettings())
    notifier = Notifier(settings=empty)
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()
    notifier._injected_bot = mock_bot  # type: ignore[assignment]
    await notifier.send_watchlist(_hits())
    mock_bot.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_watchlist_uses_injected_bot() -> None:
    """With creds present the notifier must call send_message with MarkdownV2 and chat_id."""
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()
    notifier = Notifier(settings=_settings_with_creds(), bot=mock_bot)
    await notifier.send_watchlist(_hits())
    mock_bot.send_message.assert_awaited_once()
    kwargs = mock_bot.send_message.await_args.kwargs
    assert kwargs["chat_id"] == "42"
    assert kwargs["parse_mode"].name == "MARKDOWN_V2"
    text = kwargs["text"]
    # Symbol and a MarkdownV2-escaped percent figure should be present.
    assert "ABCD" in text
    assert "WXYZ" in text
    # MarkdownV2 escapes underscores — earnings_beat → earnings\_beat.
    assert "earnings\\_beat" in text
    # Float rendering: 3.2M for known, ? for unknown — periods escaped by MarkdownV2.
    assert "3\\.2M" in text
    assert "Float ?" in text


@pytest.mark.asyncio
async def test_send_watchlist_swallows_telegram_error() -> None:
    """A telegram network error must be logged, not re-raised — the scanner keeps running."""
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock(side_effect=NetworkError("boom"))
    notifier = Notifier(settings=_settings_with_creds(), bot=mock_bot)
    # Must not raise.
    await notifier.send_watchlist(_hits())
    mock_bot.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_format_renders_header_and_numbered_lines() -> None:
    """Direct format check: header line, separator, and one line per hit with an index."""
    notifier = Notifier(settings=_settings_with_creds())
    text = notifier._format(_hits())
    assert "Morning Watchlist" in text
    # Two hits → numbered 1. and 2. (escaped periods in MarkdownV2).
    assert "1\\." in text
    assert "2\\." in text


@pytest.mark.asyncio
async def test_format_empty_watchlist_shows_placeholder() -> None:
    """An empty hits list still produces a valid message with a placeholder line."""
    notifier = Notifier(settings=_settings_with_creds())
    text = notifier._format([])
    assert "no hits" in text.lower()

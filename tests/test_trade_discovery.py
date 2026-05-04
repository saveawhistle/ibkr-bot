"""Trade-discovery tests with synthetic session logs."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from bot.exit_advisor.replay.trade_discovery import (
    discover_closed_trades,
    read_manifest,
    write_manifest,
)


def _write_session_log(path: Path, lines: list[dict]) -> None:  # type: ignore[type-arg]
    path.write_text("\n".join(json.dumps(ln) for ln in lines) + "\n", encoding="utf-8")


def _opened(symbol: str, ts: str, strategy: str = "momentum") -> dict:  # type: ignore[type-arg]
    return {
        "event": "position.opened",
        "symbol": symbol,
        "strategy": strategy,
        "timestamp": ts,
    }


def _closed(symbol: str, ts: str) -> dict:  # type: ignore[type-arg]
    return {"event": "position.closed", "symbol": symbol, "timestamp": ts}


def test_discovers_basic_closed_trade(tmp_path: Path) -> None:
    log = tmp_path / "session_2026-04-30.jsonl"
    _write_session_log(
        log,
        [
            _opened("AAPL", "2026-04-30T15:00:00Z"),
            _closed("AAPL", "2026-04-30T15:30:00Z"),
        ],
    )
    refs = discover_closed_trades(tmp_path)
    assert len(refs) == 1
    assert refs[0].symbol == "AAPL"
    assert refs[0].trade_date == date(2026, 4, 30)
    assert refs[0].trade_id == "2026-04-30T15:00:00+00:00"


def test_excludes_test_strategies(tmp_path: Path) -> None:
    """Synthetic test trades (strategy=test or manual_test) are excluded."""
    log = tmp_path / "session_2026-04-30.jsonl"
    _write_session_log(
        log,
        [
            _opened("ZZZ", "2026-04-30T15:00:00Z", strategy="test"),
            _closed("ZZZ", "2026-04-30T15:30:00Z"),
            _opened("YYY", "2026-04-30T15:05:00Z", strategy="manual_test"),
            _closed("YYY", "2026-04-30T15:35:00Z"),
            _opened("AAPL", "2026-04-30T15:10:00Z", strategy="momentum"),
            _closed("AAPL", "2026-04-30T15:40:00Z"),
        ],
    )
    refs = discover_closed_trades(tmp_path)
    symbols = {r.symbol for r in refs}
    assert symbols == {"AAPL"}


def test_excludes_test_symbols(tmp_path: Path) -> None:
    """TEST/AAA/BBB are operator-injected test symbols regardless of strategy."""
    log = tmp_path / "session_2026-04-30.jsonl"
    _write_session_log(
        log,
        [
            _opened("TEST", "2026-04-30T15:00:00Z"),
            _closed("TEST", "2026-04-30T15:01:00Z"),
            _opened("AAA", "2026-04-30T15:02:00Z"),
            _closed("AAA", "2026-04-30T15:03:00Z"),
            _opened("BBB", "2026-04-30T15:04:00Z"),
            _closed("BBB", "2026-04-30T15:05:00Z"),
        ],
    )
    refs = discover_closed_trades(tmp_path)
    assert refs == []


def test_excludes_unclosed_trades_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A trade that opened but never closed is excluded and logs a WARNING."""
    import logging

    log_path = tmp_path / "session_2026-04-30.jsonl"
    _write_session_log(
        log_path,
        [
            _opened("AAPL", "2026-04-30T15:00:00Z"),
            # No matching close.
        ],
    )
    with caplog.at_level(logging.WARNING):
        refs = discover_closed_trades(tmp_path)
    assert refs == []
    assert any("opened but never closed" in m.getMessage() for m in caplog.records)


def test_multiple_trades_same_symbol_each_distinct(tmp_path: Path) -> None:
    """Two opens + two closes on the same symbol → two ClosedTradeRefs,
    matched in FIFO order, distinguished by entry_timestamp."""
    log = tmp_path / "session_2026-04-30.jsonl"
    _write_session_log(
        log,
        [
            _opened("AAPL", "2026-04-30T15:00:00Z"),
            _closed("AAPL", "2026-04-30T15:10:00Z"),
            _opened("AAPL", "2026-04-30T15:20:00Z"),
            _closed("AAPL", "2026-04-30T15:30:00Z"),
        ],
    )
    refs = discover_closed_trades(tmp_path)
    assert len(refs) == 2
    assert refs[0].entry_timestamp.minute == 0
    assert refs[1].entry_timestamp.minute == 20
    assert refs[0].trade_id != refs[1].trade_id


def test_chronological_ordering_across_logs(tmp_path: Path) -> None:
    _write_session_log(
        tmp_path / "session_2026-04-30.jsonl",
        [
            _opened("AAPL", "2026-04-30T16:00:00Z"),
            _closed("AAPL", "2026-04-30T16:30:00Z"),
        ],
    )
    _write_session_log(
        tmp_path / "session_2026-04-29.jsonl",
        [
            _opened("MSFT", "2026-04-29T15:00:00Z"),
            _closed("MSFT", "2026-04-29T15:30:00Z"),
        ],
    )
    refs = discover_discoverable(tmp_path)
    assert refs[0].symbol == "MSFT"  # earlier date first
    assert refs[1].symbol == "AAPL"


def discover_discoverable(p: Path):  # type: ignore[no-untyped-def]
    return discover_closed_trades(p)


def test_truncated_log_handled_gracefully(tmp_path: Path) -> None:
    """A line that fails JSON parse is skipped without crashing."""
    log = tmp_path / "session_2026-04-30.jsonl"
    log.write_text(
        json.dumps(_opened("AAPL", "2026-04-30T15:00:00Z")) + "\n"
        + "MALFORMED LINE\n"
        + json.dumps(_closed("AAPL", "2026-04-30T15:30:00Z")) + "\n",
        encoding="utf-8",
    )
    refs = discover_closed_trades(tmp_path)
    assert len(refs) == 1


def test_manifest_round_trip(tmp_path: Path) -> None:
    log = tmp_path / "session_2026-04-30.jsonl"
    _write_session_log(
        log,
        [
            _opened("AAPL", "2026-04-30T15:00:00Z"),
            _closed("AAPL", "2026-04-30T15:30:00Z"),
        ],
    )
    refs = discover_closed_trades(tmp_path)
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(refs, manifest_path)
    round_tripped = read_manifest(manifest_path)
    assert round_tripped == refs

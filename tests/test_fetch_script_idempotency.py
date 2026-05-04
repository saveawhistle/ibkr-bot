"""fetch_historical_bars idempotency tests with a mocked IBKR client.

Real IBKR calls are NEVER made — every test patches in a fake client
that records its calls and returns canned data. The tests assert:
  - Already-fetched (symbol, date) pairs are skipped
  - Empty IBKR response writes a ``.unavailable`` placeholder
  - Connection failures leave the cache untouched
  - Running twice produces identical state
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from scripts import fetch_historical_bars as fhb


def _fake_bar(minute_offset: int) -> SimpleNamespace:
    """Mimic ib_async's BarData minimal surface — has ``date``, ``open``,
    ``high``, ``low``, ``close``, ``volume`` attributes."""
    return SimpleNamespace(
        date=datetime(2026, 4, 30, 13, 30, tzinfo=UTC) + timedelta(minutes=minute_offset),
        open=1.00 + minute_offset * 0.001,
        high=1.05,
        low=0.95,
        close=1.00 + minute_offset * 0.001,
        volume=100,
    )


@pytest.fixture
def fake_client() -> AsyncMock:
    """A mock IBKRClient with the methods _fetch_one calls."""
    client = AsyncMock()
    client.qualify_stock = AsyncMock(return_value=SimpleNamespace(symbol="ZENA"))
    client._ib = SimpleNamespace(reqHistoricalDataAsync=AsyncMock(return_value=[_fake_bar(i) for i in range(390)]))
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    return client


def test_fetch_writes_cache_file(tmp_path: Path, fake_client: AsyncMock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fhb, "CACHE_DIR", tmp_path)
    asyncio.run(
        fhb._fetch_one(fake_client, "ZENA", date(2026, 4, 30), logging.getLogger("t"))
    )
    cache_file = tmp_path / "ZENA_2026-04-30.jsonl"
    assert cache_file.exists()
    lines = [ln for ln in cache_file.read_text().splitlines() if ln]
    assert len(lines) == 390


def test_skip_when_cache_exists(tmp_path: Path, fake_client: AsyncMock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fhb, "CACHE_DIR", tmp_path)
    (tmp_path / "ZENA_2026-04-30.jsonl").write_text("{}\n")  # pre-existing cache

    status, _ = asyncio.run(
        fhb._fetch_one(fake_client, "ZENA", date(2026, 4, 30), logging.getLogger("t"))
    )
    assert status == "hit"
    fake_client.qualify_stock.assert_not_called()


def test_skip_when_unavailable_marker_exists(
    tmp_path: Path, fake_client: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fhb, "CACHE_DIR", tmp_path)
    (tmp_path / "ZENA_2026-04-30.unavailable").write_text("{}")

    status, _ = asyncio.run(
        fhb._fetch_one(fake_client, "ZENA", date(2026, 4, 30), logging.getLogger("t"))
    )
    assert status == "hit"
    fake_client.qualify_stock.assert_not_called()


def test_empty_response_writes_unavailable_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fhb, "CACHE_DIR", tmp_path)
    client = AsyncMock()
    client.qualify_stock = AsyncMock(return_value=SimpleNamespace(symbol="DEAD"))
    client._ib = SimpleNamespace(reqHistoricalDataAsync=AsyncMock(return_value=[]))

    status, _ = asyncio.run(
        fhb._fetch_one(client, "DEAD", date(2026, 4, 30), logging.getLogger("t"))
    )
    assert status == "unavailable"
    assert (tmp_path / "DEAD_2026-04-30.unavailable").exists()
    assert not (tmp_path / "DEAD_2026-04-30.jsonl").exists()


def test_qualify_failure_writes_unavailable_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A symbol that won't qualify (delisted, ticker changed) must write
    the ``.unavailable`` marker and let the batch continue. Previously
    the qualify_stock call sat outside the exception handler and the
    whole batch crashed — surfaced when SBLX failed during layer 4.5's
    pipeline run on 2026-05-02."""
    monkeypatch.setattr(fhb, "CACHE_DIR", tmp_path)
    client = AsyncMock()
    client.qualify_stock = AsyncMock(
        side_effect=ValueError("Could not qualify stock symbol: 'SBLX'")
    )

    status, _ = asyncio.run(
        fhb._fetch_one(client, "SBLX", date(2026, 4, 28), logging.getLogger("t"))
    )
    assert status == "unavailable"
    marker = tmp_path / "SBLX_2026-04-28.unavailable"
    assert marker.exists()
    payload = marker.read_text(encoding="utf-8")
    assert "qualify_stock failed" in payload
    assert "SBLX" in payload
    # No JSONL written; only the marker.
    assert not (tmp_path / "SBLX_2026-04-28.jsonl").exists()


def test_fetch_failure_leaves_cache_untouched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fhb, "CACHE_DIR", tmp_path)
    client = AsyncMock()
    client.qualify_stock = AsyncMock(return_value=SimpleNamespace(symbol="ZENA"))
    client._ib = SimpleNamespace(
        reqHistoricalDataAsync=AsyncMock(side_effect=ConnectionError("socket gone"))
    )

    status, _ = asyncio.run(
        fhb._fetch_one(client, "ZENA", date(2026, 4, 30), logging.getLogger("t"))
    )
    assert status == "error"
    assert not (tmp_path / "ZENA_2026-04-30.jsonl").exists()
    assert not (tmp_path / "ZENA_2026-04-30.unavailable").exists()


def test_running_twice_produces_identical_state(
    tmp_path: Path, fake_client: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Idempotency: a second run must NOT re-fetch (cache file exists),
    and the cache file's contents must be byte-identical to the first run."""
    monkeypatch.setattr(fhb, "CACHE_DIR", tmp_path)
    asyncio.run(
        fhb._fetch_one(fake_client, "ZENA", date(2026, 4, 30), logging.getLogger("t"))
    )
    contents_after_first = (tmp_path / "ZENA_2026-04-30.jsonl").read_bytes()
    fake_client.qualify_stock.reset_mock()

    asyncio.run(
        fhb._fetch_one(fake_client, "ZENA", date(2026, 4, 30), logging.getLogger("t"))
    )
    contents_after_second = (tmp_path / "ZENA_2026-04-30.jsonl").read_bytes()

    assert contents_after_first == contents_after_second
    fake_client.qualify_stock.assert_not_called()


def test_atomic_write_no_tmp_left_behind(
    tmp_path: Path, fake_client: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fhb, "CACHE_DIR", tmp_path)
    asyncio.run(
        fhb._fetch_one(fake_client, "ZENA", date(2026, 4, 30), logging.getLogger("t"))
    )
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == []


def test_prior_trading_day_skips_weekends() -> None:
    # 2026-04-27 is a Monday — prior trading day should be Friday 04-24
    assert fhb.prior_trading_day(date(2026, 4, 27)) == date(2026, 4, 24)


def test_prior_trading_day_skips_holiday() -> None:
    # 2026-01-20 is the day after MLK Day (Jan 19, 2026, Monday holiday)
    assert fhb.prior_trading_day(date(2026, 1, 20)) == date(2026, 1, 16)


def test_enumerate_closed_trades_finds_zena(tmp_path: Path) -> None:
    """Smoke test against the real logs/ dir — ZENA on 2026-04-30 should
    be discoverable via the closed-trade enumerator."""
    targets = fhb._enumerate_closed_trades(Path("logs"))
    symbols = {t.symbol for t in targets}
    dates = {t.trading_date for t in targets}
    assert "ZENA" in symbols
    assert date(2026, 4, 30) in dates


@patch("argparse._sys.argv", ["fetch_historical_bars"])
def test_main_requires_args() -> None:
    with patch("sys.argv", ["fetch_historical_bars"]):
        rc = fhb.main([])
    assert rc == 2  # neither --all-trades nor (--symbol + --date)

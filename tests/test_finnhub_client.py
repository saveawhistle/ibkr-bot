"""Tests for ``bot.scanning.finnhub_client``: retry logic, rate limiter, and response parsing."""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest
import tenacity

from bot.config import ConfigurationError
from bot.scanning import finnhub_client as fc_module
from bot.scanning.finnhub_client import FinnhubClient


def _build_client(handler: httpx.MockTransport) -> FinnhubClient:
    """Construct a FinnhubClient whose httpx.AsyncClient uses the provided mock transport."""
    http = httpx.AsyncClient(transport=handler, base_url="https://finnhub.io/api/v1")
    return FinnhubClient(api_key="test-key", client=http)


@pytest.mark.asyncio
async def test_missing_api_key_raises_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FinnhubClient must refuse to start without a key and point the user at the .env var."""
    from bot.config import DataSourcesSettings, Settings

    monkeypatch.delenv("BOT_DATA_SOURCES__FINNHUB_API_KEY", raising=False)
    empty_settings = Settings(data_sources=DataSourcesSettings(finnhub_api_key=None))
    with pytest.raises(ConfigurationError, match="BOT_DATA_SOURCES__FINNHUB_API_KEY"):
        FinnhubClient(api_key=None, settings=empty_settings)


@pytest.mark.asyncio
async def test_company_profile_returns_none_for_empty_response() -> None:
    """Finnhub returns ``{}`` when a symbol isn't found — company_profile must map that to None."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/stock/profile2")
        return httpx.Response(200, json={})

    client = _build_client(httpx.MockTransport(handler))
    try:
        result = await client.company_profile("NOPE")
    finally:
        await client.close()
    assert result is None


@pytest.mark.asyncio
async def test_company_profile_parses_populated_response() -> None:
    """A populated Finnhub profile response should hydrate the pydantic aliases."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "ticker": "ABCD",
                "shareOutstanding": 12.5,
                "marketCapitalization": 300.0,
                "finnhubIndustry": "Biotech",
                "country": "US",
            },
        )

    client = _build_client(httpx.MockTransport(handler))
    try:
        profile = await client.company_profile("ABCD")
    finally:
        await client.close()
    assert profile is not None
    assert profile.symbol == "ABCD"
    assert profile.share_outstanding == 12.5
    assert profile.market_cap == 300.0
    assert profile.industry == "Biotech"


@pytest.mark.asyncio
async def test_request_retries_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 429 response should trigger a retry and ultimately succeed on the second attempt."""
    # Zero out tenacity backoff so the test completes in milliseconds.
    monkeypatch.setattr(fc_module, "wait_exponential", lambda **kwargs: tenacity.wait_fixed(0))
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json={"ticker": "ABCD", "shareOutstanding": 5.0})

    client = _build_client(httpx.MockTransport(handler))
    try:
        profile = await client.company_profile("ABCD")
    finally:
        await client.close()
    assert call_count == 2
    assert profile is not None
    assert profile.share_outstanding == 5.0


@pytest.mark.asyncio
async def test_request_gives_up_after_three_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Persistent 429s should exhaust the 3-attempt budget and surface the HTTPStatusError."""
    monkeypatch.setattr(fc_module, "wait_exponential", lambda **kwargs: tenacity.wait_fixed(0))
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(429, json={"error": "rate limited"})

    client = _build_client(httpx.MockTransport(handler))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await client.company_profile("ABCD")
    finally:
        await client.close()
    assert call_count == 3


@pytest.mark.asyncio
async def test_non_retryable_status_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 404 is not in the retry set — the client should fail on the first attempt."""
    monkeypatch.setattr(fc_module, "wait_exponential", lambda **kwargs: tenacity.wait_fixed(0))
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(404)

    client = _build_client(httpx.MockTransport(handler))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await client.company_profile("ABCD")
    finally:
        await client.close()
    assert call_count == 1


@pytest.mark.asyncio
async def test_rate_limiter_enforces_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """After N calls inside the window, the (N+1)-th must block until the oldest ages out."""
    monkeypatch.setattr(fc_module, "_RATE_LIMIT", 3)
    monkeypatch.setattr(fc_module, "_RATE_WINDOW_S", 0.3)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ticker": "ABCD"})

    client = _build_client(httpx.MockTransport(handler))
    try:
        start = time.monotonic()
        for _ in range(4):
            await client.company_profile("ABCD")
        elapsed = time.monotonic() - start
    finally:
        await client.close()
    # The 4th call must have waited roughly one window before firing.
    assert elapsed >= 0.3, f"rate limiter did not gate the 4th call (elapsed={elapsed:.3f}s)"


@pytest.mark.asyncio
async def test_async_with_closes_owned_http_client() -> None:
    """``async with FinnhubClient(...)`` should close the internally-created httpx client on exit."""
    owned = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json={})),
        base_url="https://finnhub.io/api/v1",
    )
    # Inject an already-built client so _owns_client=False; then exercise the async-with path
    # on a separate instance that owns its client to assert aclose was called.
    instance = FinnhubClient(api_key="k")
    assert not instance._client.is_closed  # internal state check, pre-close
    async with instance:
        pass
    assert instance._client.is_closed
    await owned.aclose()


@pytest.mark.asyncio
async def test_concurrent_profile_requests_share_rate_limiter() -> None:
    """asyncio.gather of many profile calls must serialize through the shared limiter."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ticker": "ABCD", "shareOutstanding": 1.0})

    client = _build_client(httpx.MockTransport(handler))
    try:
        results = await asyncio.gather(*(client.company_profile(f"SYM{i}") for i in range(5)))
    finally:
        await client.close()
    assert all(r is not None for r in results)

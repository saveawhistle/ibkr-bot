"""Finnhub free-tier HTTP client: company profiles (float proxy) + company news.

Rate-limited to 60 requests per rolling 60 seconds; retries 429 and 5xx responses
with 1s→2s→4s exponential backoff via ``tenacity``. Validates responses with
Pydantic — Finnhub's payloads are treated as untrusted external data.
"""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import Any

import httpx
import structlog
from pydantic import BaseModel, ConfigDict, Field, field_validator
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from bot.config import ConfigurationError, Settings, get_settings

_BASE_URL = "https://finnhub.io/api/v1"
_RATE_LIMIT = 60
_RATE_WINDOW_S = 60.0
_RETRY_ATTEMPTS = 3

_log = structlog.get_logger("bot.scanning.finnhub_client")


class CompanyProfile(BaseModel):
    """Subset of Finnhub's ``/stock/profile2`` response used by the scanner."""

    model_config = ConfigDict(populate_by_name=True)

    symbol: str
    share_outstanding: float | None = Field(default=None, alias="shareOutstanding")
    float_shares: int | None = None
    market_cap: float | None = Field(default=None, alias="marketCapitalization")
    industry: str | None = Field(default=None, alias="finnhubIndustry")
    country: str | None = None


class NewsItem(BaseModel):
    """Subset of Finnhub's ``/company-news`` response used for catalyst classification."""

    model_config = ConfigDict(populate_by_name=True)

    headline: str
    source: str
    url: str
    datetime: datetime
    summary: str = ""
    category: str = ""

    @field_validator("datetime", mode="before")
    @classmethod
    def _coerce_unix_timestamp(cls, value: Any) -> Any:
        """Convert Finnhub's unix-seconds integer into a tz-aware UTC datetime."""
        if isinstance(value, int | float):
            return datetime.fromtimestamp(float(value), tz=UTC)
        return value


def _is_retryable_http_error(exc: BaseException) -> bool:
    """Return True if ``exc`` is a 429 or 5xx HTTP error worth retrying."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or 500 <= status < 600
    return False


def _log_retry(retry_state: RetryCallState) -> None:
    """Structlog ``before_sleep`` hook for tenacity — emitted once per retry attempt."""
    outcome = retry_state.outcome
    exc = outcome.exception() if outcome is not None else None
    next_action = retry_state.next_action
    _log.warning(
        "finnhub.retry",
        attempt=retry_state.attempt_number,
        next_wait_s=round(next_action.sleep, 2) if next_action is not None else None,
        error=str(exc) if exc is not None else None,
    )


class FinnhubClient:
    """Async Finnhub client with a shared token-bucket rate limiter and retrying ``_request``."""

    def __init__(
        self,
        api_key: str | None = None,
        settings: Settings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 10.0,
    ) -> None:
        """Build a client from an explicit ``api_key`` or the process settings singleton."""
        resolved_settings = settings or get_settings()
        key = api_key or resolved_settings.data_sources.finnhub_api_key
        if not key:
            raise ConfigurationError(
                "Finnhub API key missing. Set BOT_DATA_SOURCES__FINNHUB_API_KEY in your .env "
                "(sign up at https://finnhub.io)."
            )
        self._api_key: str = key
        self._client: httpx.AsyncClient = client or httpx.AsyncClient(
            base_url=_BASE_URL, timeout=timeout
        )
        self._owns_client: bool = client is None
        self._request_times: deque[float] = deque()
        self._throttle_lock: asyncio.Lock = asyncio.Lock()

    async def __aenter__(self) -> FinnhubClient:
        """Enter an ``async with`` block — returns ``self``."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Exit the ``async with`` block by closing the owned httpx client."""
        await self.close()

    async def close(self) -> None:
        """Close the underlying ``httpx.AsyncClient`` if this instance owns it."""
        if self._owns_client:
            await self._client.aclose()

    async def company_profile(self, symbol: str) -> CompanyProfile | None:
        """Fetch and validate a company profile; return ``None`` if Finnhub reports no such symbol."""
        data = await self._request("/stock/profile2", {"symbol": symbol})
        if not data:
            return None
        payload: dict[str, Any] = {"symbol": symbol, **data}
        return CompanyProfile.model_validate(payload)

    async def company_news(self, symbol: str, hours_back: int = 24) -> list[NewsItem]:
        """Fetch recent company news items from the last ``hours_back`` hours."""
        now = datetime.now(UTC)
        from_dt = now - timedelta(hours=hours_back)
        params = {
            "symbol": symbol,
            "from": from_dt.strftime("%Y-%m-%d"),
            "to": now.strftime("%Y-%m-%d"),
        }
        data = await self._request("/company-news", params)
        if not data:
            return []
        items: list[NewsItem] = []
        for row in data:
            try:
                items.append(NewsItem.model_validate(row))
            except Exception as exc:  # noqa: BLE001 - per-item skip on validation failure
                _log.warning("finnhub.news_item_invalid", symbol=symbol, error=str(exc))
        return items

    async def _request(self, path: str, params: dict[str, Any]) -> Any:
        """Throttled, retrying GET against the Finnhub API; returns parsed JSON."""
        retryer = AsyncRetrying(
            retry=retry_if_exception(_is_retryable_http_error),
            stop=stop_after_attempt(_RETRY_ATTEMPTS),
            wait=wait_exponential(multiplier=1, min=1, max=4),
            before_sleep=_log_retry,
            reraise=True,
        )
        return await retryer(self._do_request, path, params)

    async def _do_request(self, path: str, params: dict[str, Any]) -> Any:
        """Single-shot GET with rate-limit gating — called by the tenacity retryer."""
        await self._throttle()
        query = {**params, "token": self._api_key}
        resp = await self._client.get(path, params=query)
        resp.raise_for_status()
        return resp.json()

    async def _throttle(self) -> None:
        """Block until the rolling 60-request / 60-second budget allows another call."""
        async with self._throttle_lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            self._evict_expired(now)
            if len(self._request_times) >= _RATE_LIMIT:
                wait_s = _RATE_WINDOW_S - (now - self._request_times[0])
                if wait_s > 0:
                    _log.info("finnhub.rate_limit_wait", wait_s=round(wait_s, 3))
                    await asyncio.sleep(wait_s)
                now = loop.time()
                self._evict_expired(now)
            self._request_times.append(now)

    def _evict_expired(self, now: float) -> None:
        """Drop timestamps older than the rolling window from the deque."""
        while self._request_times and now - self._request_times[0] >= _RATE_WINDOW_S:
            self._request_times.popleft()

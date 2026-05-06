"""In-memory LRU + TTL cache for catalyst classifications.

Keys: ``(ticker, sha256_of_sorted_headlines)``. Hashing the sorted-headline
strings (NOT summaries — small text variations in summaries shouldn't
bust the cache) means re-running the classifier on the same news body
is O(1). Cache survives only the bot process; cleared on restart.

Caches both ``qualifies=True`` and ``qualifies=False`` results — same
news within the TTL is identical news, regardless of disposition.
Failures are NOT cached — a transient API blip shouldn't condemn a
ticker for 30 minutes.

Pure data structure; no time module imports beyond ``time.monotonic`` for
TTL. Tests can construct with a fixed clock if needed.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from bot.scanning.llm_catalyst_classifier.llm_client import CatalystClassification

DEFAULT_TTL_SECONDS = 1800  # 30 minutes — matches config.yaml default
DEFAULT_CAPACITY = 200


def hash_headlines(headlines: list[str]) -> str:
    """Stable SHA-256 of sorted unique headline strings.

    Sorted to make order-independence. Unique to prevent duplicate
    Finnhub responses (same headline returned twice in a single
    company-news call) from busting the cache. Strips and lowercases
    each headline before hashing so trivial whitespace / casing
    differences also coalesce.
    """
    canonical = sorted({h.strip().lower() for h in headlines if h and h.strip()})
    digest = hashlib.sha256()
    for h in canonical:
        digest.update(h.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class CacheEntry:
    """One cached classification + the wall-clock at which it was inserted.

    ``inserted_at_monotonic`` uses ``time.monotonic`` so wall-clock
    adjustments mid-session don't surprise-evict everything. The cache
    only ever compares deltas, never absolute times.
    """

    classification: CatalystClassification
    inserted_at_monotonic: float


class ClassificationCache:
    """LRU cache with TTL eviction.

    Capacity-bounded by ``capacity`` (LRU evicts oldest on insert).
    TTL-bounded by ``ttl_seconds`` (entries past the TTL are skipped on
    lookup AND removed; subsequent identical lookups go through to
    the LLM as expected).

    The clock function is injectable so tests can drive deterministic
    TTL expiry. Defaults to ``time.monotonic``.
    """

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        capacity: int = DEFAULT_CAPACITY,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be > 0 (got {ttl_seconds})")
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0 (got {capacity})")
        self._ttl = ttl_seconds
        self._capacity = capacity
        self._clock = clock
        self._store: OrderedDict[tuple[str, str], CacheEntry] = OrderedDict()

    def get(self, ticker: str, headlines_hash: str) -> CatalystClassification | None:
        """Return the cached classification or ``None`` on miss / expired."""
        key = (ticker, headlines_hash)
        entry = self._store.get(key)
        if entry is None:
            return None
        age = self._clock() - entry.inserted_at_monotonic
        if age >= self._ttl:
            # Expired — drop and miss.
            del self._store[key]
            return None
        # LRU touch: re-insert at the end of the OrderedDict.
        self._store.move_to_end(key)
        return entry.classification

    def put(
        self,
        ticker: str,
        headlines_hash: str,
        classification: CatalystClassification,
    ) -> None:
        """Insert or refresh an entry; evict oldest if at capacity."""
        key = (ticker, headlines_hash)
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = CacheEntry(
            classification=classification,
            inserted_at_monotonic=self._clock(),
        )
        while len(self._store) > self._capacity:
            self._store.popitem(last=False)

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        """Empty the cache. Exposed for tests + the scanner's session-reset path."""
        self._store.clear()


__all__ = [
    "DEFAULT_CAPACITY",
    "DEFAULT_TTL_SECONDS",
    "CacheEntry",
    "ClassificationCache",
    "hash_headlines",
]

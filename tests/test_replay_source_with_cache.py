"""Replay-source merge behavior with the historical bar cache.

Uses the synthetic ZENA fixtures in ``tests/fixtures/`` rather than the
real cache directory, so tests are deterministic and don't depend on
operator-populated cache state.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from bot.exit_advisor.replay.replay_source import load_trade_replay_data

ZENA_DATE = date(2026, 4, 30)
FIXTURES = Path(__file__).parent / "fixtures" / "exit_advisor_zena"


def test_pre_trade_bars_merge_session_log_and_cache() -> None:
    """With the cache populated, ZENA's pre-trade backfill should be
    much larger than the session-log-only count from layer 2 — covering
    full session open through subscription start."""
    rd_no_cache = load_trade_replay_data("ZENA", ZENA_DATE, cache_dir="/nonexistent")
    rd_with_cache = load_trade_replay_data("ZENA", ZENA_DATE, cache_dir=FIXTURES)
    assert len(rd_with_cache.pre_trade_bars) > len(rd_no_cache.pre_trade_bars)
    # Fixture covers 09:30 → 10:52 ET (82 bars). After merge with the 10
    # session-log bars (which the bot received live), the union sized
    # against the bracket minute should be about 82 — session-log bars
    # win on overlap, but the time window is identical.
    assert 70 <= len(rd_with_cache.pre_trade_bars) <= 100


def test_pre_trade_bar_sources_attribution() -> None:
    """Every pre-trade bar must have a source attribution — either
    ``session_log`` or ``historical_cache``. Forensic tooling reads this
    map to disambiguate live-observed from retrospectively-fetched."""
    rd = load_trade_replay_data("ZENA", ZENA_DATE, cache_dir=FIXTURES)
    assert len(rd.pre_trade_bar_sources) == len(rd.pre_trade_bars)
    sources = set(rd.pre_trade_bar_sources.values())
    assert sources <= {"session_log", "historical_cache"}
    # We expect both types to be present: cache fills the early window,
    # session log covers the late minutes once the bot subscribed.
    assert "historical_cache" in sources
    assert "session_log" in sources


def test_session_log_wins_over_cache_at_overlap() -> None:
    """When both the session log and the cache have a bar at the same
    timestamp, the session log's bar (what the bot actually received)
    is the source of truth."""
    rd = load_trade_replay_data("ZENA", ZENA_DATE, cache_dir=FIXTURES)
    # Find timestamps that exist in both the session-log-only result and
    # the cache-merged result.
    rd_no_cache = load_trade_replay_data("ZENA", ZENA_DATE, cache_dir="/nonexistent")
    session_log_timestamps = {b.timestamp for b in rd_no_cache.pre_trade_bars}
    for ts in session_log_timestamps:
        assert rd.pre_trade_bar_sources[ts] == "session_log"


def test_prior_day_bars_loaded_from_cache() -> None:
    rd = load_trade_replay_data("ZENA", ZENA_DATE, cache_dir=FIXTURES)
    assert rd.prior_day_cache_state == "hit"
    assert len(rd.prior_day_bars) > 0
    assert rd.prior_day_session_high is not None
    assert rd.prior_day_session_low is not None
    assert rd.prior_day_session_close is not None
    assert rd.prior_day_session_high >= rd.prior_day_session_low


def test_prior_day_cache_state_not_populated() -> None:
    rd = load_trade_replay_data("ZENA", ZENA_DATE, cache_dir="/nonexistent")
    assert rd.prior_day_cache_state == "not_populated"
    assert rd.prior_day_bars == []
    assert rd.prior_day_session_high is None


def test_prior_day_cache_state_marked_unavailable(tmp_path: Path) -> None:
    """A ``.unavailable`` marker means data was attempted and confirmed
    missing — distinct from never-attempted."""
    (tmp_path / "ZENA_2026-04-29.unavailable").write_text("{}", encoding="utf-8")
    rd = load_trade_replay_data("ZENA", ZENA_DATE, cache_dir=tmp_path)
    assert rd.prior_day_cache_state == "marked_unavailable"
    assert rd.prior_day_bars == []


def test_trade_window_bars_unchanged_by_cache() -> None:
    """The trade-window ``bars`` field remains session-log-only — those
    are the bars the bot actually saw and responded to. Cache only fills
    pre-trade and prior-day gaps."""
    rd_no_cache = load_trade_replay_data("ZENA", ZENA_DATE, cache_dir="/nonexistent")
    rd_with_cache = load_trade_replay_data("ZENA", ZENA_DATE, cache_dir=FIXTURES)
    assert [b.timestamp for b in rd_no_cache.bars] == [b.timestamp for b in rd_with_cache.bars]

"""Tests for ``bot.risk.rehab`` — config validation, pure helpers, engine, flag I/O."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from datetime import date as date_cls
from pathlib import Path

import pytest
from pydantic import ValidationError

from bot.config import RehabConfig, RiskConfig, Settings
from bot.risk.rehab import (
    RehabEngine,
    RehabRecord,
    RehabTier,
    classify_tier,
    compute_stats_from_journal_entries,
    consecutive_red_days,
    cumulative_drawdown_usd,
    read_rehab_flag,
    write_rehab_flag,
)

# ---------- RehabConfig validation ---------- #


def test_rehab_config_default_matches_ross_rule() -> None:
    """Defaults ship with the 0.5× REHAB + 0.25× DEEP_REHAB multipliers."""
    cfg = RehabConfig()
    assert cfg.rehab_max_loss_multiplier == pytest.approx(0.5)
    assert cfg.deep_rehab_max_loss_multiplier == pytest.approx(0.25)
    assert cfg.deep_rehab_max_trades_per_day == 1  # the methodology: one setup a day in deep slump
    assert cfg.recovery_drawdown_recovered_fraction == pytest.approx(0.5)


def test_rehab_config_multiplier_above_one_rejects() -> None:
    """A multiplier > 1.0 would inflate caps — must fail loudly."""
    with pytest.raises(ValidationError) as exc_info:
        RehabConfig(rehab_max_loss_multiplier=1.5)
    assert "rehab_max_loss_multiplier" in str(exc_info.value)


def test_rehab_config_zero_multiplier_rejects() -> None:
    """A multiplier of 0 would silently disable trading — use halt.flag instead."""
    with pytest.raises(ValidationError):
        RehabConfig(deep_rehab_max_daily_loss_multiplier=0.0)


def test_rehab_config_zero_trades_per_day_rejects() -> None:
    """A 0-trade tier should use halt.flag, not rehab."""
    with pytest.raises(ValidationError):
        RehabConfig(deep_rehab_max_trades_per_day=0)


def test_rehab_config_recovery_fraction_above_one_rejects() -> None:
    """Recovery fraction > 1.0 would require *more* than the entry drawdown back."""
    with pytest.raises(ValidationError):
        RehabConfig(recovery_drawdown_recovered_fraction=1.5)


# ---------- Pure helpers ---------- #


def test_consecutive_red_days_counts_trailing_reds() -> None:
    """Trailing red streak; stops at the first non-negative day."""
    days = [
        (date_cls(2026, 4, 1), 100.0),
        (date_cls(2026, 4, 2), -50.0),
        (date_cls(2026, 4, 3), -75.0),
        (date_cls(2026, 4, 4), -200.0),
    ]
    assert consecutive_red_days(days) == 3


def test_consecutive_red_days_zero_breaks_streak() -> None:
    """A zero-PnL day is not red and ends the streak."""
    days = [
        (date_cls(2026, 4, 1), -10.0),
        (date_cls(2026, 4, 2), 0.0),
        (date_cls(2026, 4, 3), -50.0),
    ]
    assert consecutive_red_days(days) == 1


def test_cumulative_drawdown_finds_worst_peak_to_trough() -> None:
    """Walks the equity curve and returns peak-to-trough as a negative dollar amount."""
    days = [
        (date_cls(2026, 4, 1), 200.0),  # equity +200, peak 200
        (date_cls(2026, 4, 2), -300.0),  # equity -100, drawdown -300
        (date_cls(2026, 4, 3), -100.0),  # equity -200, drawdown -400
        (date_cls(2026, 4, 4), 50.0),  # equity -150, drawdown -350 (worst still -400)
    ]
    assert cumulative_drawdown_usd(days) == pytest.approx(-400.0)


def test_cumulative_drawdown_empty_is_zero() -> None:
    """No trades → no drawdown."""
    assert cumulative_drawdown_usd([]) == 0.0


def test_compute_stats_windows_on_lookback() -> None:
    """Only days in the (today - lookback, today) window contribute."""
    daily = {
        date_cls(2026, 4, 1): -100.0,  # inside window
        date_cls(2026, 3, 1): -1_000.0,  # before window; should be ignored
        date_cls(2026, 4, 10): -50.0,  # inside window
    }
    stats = compute_stats_from_journal_entries(daily, date_cls(2026, 4, 15), lookback_days=14)
    # Window is [2026-04-01, 2026-04-15) — both in; 3/1 out.
    assert stats.consecutive_red_days == 2
    assert stats.cumulative_drawdown_usd == pytest.approx(-150.0)


# ---------- classify_tier ---------- #


def _rehab_config() -> RehabConfig:
    """Lightweight test config — base defaults plus explicit thresholds."""
    return RehabConfig()


def test_classify_tier_normal_on_clean_stats() -> None:
    """Clean window → NORMAL / baseline reason."""
    stats = compute_stats_from_journal_entries({}, date_cls(2026, 4, 15), 10)
    tier, reason = classify_tier(stats, _rehab_config(), max_daily_loss_usd=300.0)
    assert tier is RehabTier.NORMAL
    assert reason == "baseline"


def test_classify_tier_rehab_on_two_reds() -> None:
    """Default config: 2 consecutive reds triggers REHAB."""
    daily = {date_cls(2026, 4, 13): -50.0, date_cls(2026, 4, 14): -30.0}
    stats = compute_stats_from_journal_entries(daily, date_cls(2026, 4, 15), 10)
    tier, reason = classify_tier(stats, _rehab_config(), 300.0)
    assert tier is RehabTier.REHAB
    assert reason == "consecutive_red_days"


def test_classify_tier_deep_rehab_on_four_reds() -> None:
    """Default config: 4 consecutive reds triggers DEEP_REHAB, stricter wins."""
    daily = {
        date_cls(2026, 4, 11): -10.0,
        date_cls(2026, 4, 12): -10.0,
        date_cls(2026, 4, 13): -10.0,
        date_cls(2026, 4, 14): -10.0,
    }
    stats = compute_stats_from_journal_entries(daily, date_cls(2026, 4, 15), 10)
    tier, _reason = classify_tier(stats, _rehab_config(), 300.0)
    assert tier is RehabTier.DEEP_REHAB


def test_classify_tier_rehab_on_cumulative_drawdown() -> None:
    """Drawdown ≥ 3× max_daily_loss triggers REHAB even with one red day."""
    daily = {
        date_cls(2026, 4, 13): 100.0,  # peak climbs to +100
        date_cls(2026, 4, 14): -1_000.0,  # drawdown = -1100
    }
    stats = compute_stats_from_journal_entries(daily, date_cls(2026, 4, 15), 10)
    tier, reason = classify_tier(stats, _rehab_config(), 300.0)
    # threshold = 3 * 300 = 900; drawdown = -1100 so |dd| > 900 → REHAB
    assert tier is RehabTier.REHAB
    assert reason == "cumulative_drawdown"


# ---------- Flag file I/O ---------- #


def test_rehab_flag_round_trip(tmp_path: Path) -> None:
    """write → read returns an equal record."""
    record = RehabRecord(
        tier=RehabTier.REHAB,
        trigger_reason="consecutive_red_days",
        entered_at=datetime(2026, 4, 15, 13, 30, tzinfo=UTC),
        drawdown_at_entry_usd=-450.0,
        consecutive_red_days_at_entry=2,
    )
    path = tmp_path / "rehab.flag"
    write_rehab_flag(path, record)
    loaded = read_rehab_flag(path)
    assert loaded == record


def test_rehab_flag_corrupt_returns_none(tmp_path: Path) -> None:
    """Corrupt flag → None (engine falls back to NORMAL)."""
    path = tmp_path / "rehab.flag"
    path.write_text("not json", encoding="utf-8")
    assert read_rehab_flag(path) is None


# ---------- RehabEngine ---------- #


def _settings(
    *,
    enabled: bool = True,
    max_daily_loss_usd: float = 300.0,
    max_loss_per_trade_usd: float = 100.0,
    max_trades_per_day: int = 5,
) -> Settings:
    """Build a Settings with overrides for rehab + risk."""
    base = Settings()
    return base.model_copy(
        update={
            "risk": RiskConfig(
                max_loss_per_trade_usd=max_loss_per_trade_usd,
                max_position_value_usd=25_000.0,
                max_daily_loss_usd=max_daily_loss_usd,
                daily_profit_goal_usd=500.0,
                giveback_trigger_usd=400.0,
                giveback_pct=50.0,
                max_concurrent_positions=1,
                max_trades_per_day=max_trades_per_day,
                max_stop_width_usd=100.0,
                max_pct_of_bar_volume=2.0,
                extension_bar_trigger_multiple=2.0,
                rehab=RehabConfig(enabled=enabled),
            ),
        }
    )


def test_apply_to_caps_disabled_returns_base(tmp_path: Path) -> None:
    """``rehab.enabled: false`` returns the base caps unchanged."""
    engine = RehabEngine(settings=_settings(enabled=False), flag_path=tmp_path / "rehab.flag")
    caps = engine.apply_to_caps()
    assert caps.tier is RehabTier.NORMAL
    assert caps.trigger_reason is None
    assert caps.max_loss_per_trade_usd == pytest.approx(100.0)
    assert caps.max_daily_loss_usd == pytest.approx(300.0)
    assert caps.max_trades_per_day == 5


def test_apply_to_caps_normal_returns_base(tmp_path: Path) -> None:
    """Fresh engine sits at NORMAL; caps match base config."""
    engine = RehabEngine(settings=_settings(), flag_path=tmp_path / "rehab.flag")
    caps = engine.apply_to_caps()
    assert caps.tier is RehabTier.NORMAL
    assert caps.max_loss_per_trade_usd == pytest.approx(100.0)
    assert caps.max_trades_per_day == 5


def test_apply_to_caps_rehab_scales_down(tmp_path: Path) -> None:
    """In REHAB, the three caps are halved (default multiplier 0.5)."""
    engine = RehabEngine(settings=_settings(), flag_path=tmp_path / "rehab.flag")
    engine.save_state(
        engine._build_entry_state(  # noqa: SLF001 — test-only reach-in
            RehabTier.REHAB,
            "consecutive_red_days",
            stats=compute_stats_from_journal_entries(
                {date_cls(2026, 4, 14): -50.0}, date_cls(2026, 4, 15), 10
            ),
        )
    )
    caps = engine.apply_to_caps()
    assert caps.tier is RehabTier.REHAB
    assert caps.max_loss_per_trade_usd == pytest.approx(50.0)
    assert caps.max_daily_loss_usd == pytest.approx(150.0)
    assert caps.max_trades_per_day == 3


def test_apply_to_caps_deep_rehab_quartering(tmp_path: Path) -> None:
    """DEEP_REHAB quarters the per-trade + daily cap and forces 1 trade/day."""
    engine = RehabEngine(settings=_settings(), flag_path=tmp_path / "rehab.flag")
    engine.save_state(
        engine._build_entry_state(  # noqa: SLF001 — test-only reach-in
            RehabTier.DEEP_REHAB,
            "consecutive_red_days",
            stats=compute_stats_from_journal_entries(
                {date_cls(2026, 4, 14): -50.0}, date_cls(2026, 4, 15), 10
            ),
        )
    )
    caps = engine.apply_to_caps()
    assert caps.tier is RehabTier.DEEP_REHAB
    assert caps.max_loss_per_trade_usd == pytest.approx(25.0)
    assert caps.max_daily_loss_usd == pytest.approx(75.0)
    assert caps.max_trades_per_day == 1  # the "one setup a day" rule


@pytest.mark.asyncio
async def test_check_transitions_upgrades_immediately(tmp_path: Path) -> None:
    """NORMAL → REHAB fires without hysteresis (worsening can't be gated)."""
    engine = RehabEngine(settings=_settings(), flag_path=tmp_path / "rehab.flag")
    today = date_cls(2026, 4, 15)
    engine.set_simulation_override(
        [(today - timedelta(days=2), -50.0), (today - timedelta(days=1), -50.0)]
    )
    transition = await engine.check_transitions(today)
    assert transition is not None
    assert transition.old_tier is RehabTier.NORMAL
    assert transition.new_tier is RehabTier.REHAB


@pytest.mark.asyncio
async def test_check_transitions_upgrade_rehab_to_deep(tmp_path: Path) -> None:
    """Stricter tier always supersedes a saved looser one."""
    engine = RehabEngine(settings=_settings(), flag_path=tmp_path / "rehab.flag")
    today = date_cls(2026, 4, 15)
    # First 2 reds → REHAB
    engine.set_simulation_override(
        [(today - timedelta(days=2), -50.0), (today - timedelta(days=1), -50.0)]
    )
    await engine.check_transitions(today)
    # Extend to 4 reds → DEEP_REHAB
    engine.set_simulation_override(
        [(today - timedelta(days=offset), -50.0) for offset in range(4, 0, -1)]
    )
    transition = await engine.check_transitions(today)
    assert transition is not None
    assert transition.new_tier is RehabTier.DEEP_REHAB


@pytest.mark.asyncio
async def test_check_transitions_downgrade_gated_by_recovery(tmp_path: Path) -> None:
    """REHAB → NORMAL only once the operator recovers ≥ 50% of entry drawdown."""
    engine = RehabEngine(settings=_settings(), flag_path=tmp_path / "rehab.flag")
    today = date_cls(2026, 4, 15)
    # Enter REHAB with cumulative drawdown = -1000 (peak +200, trough -800)
    engine.set_simulation_override(
        [
            (today - timedelta(days=3), 200.0),
            (today - timedelta(days=2), -400.0),
            (today - timedelta(days=1), -600.0),
        ]
    )
    await engine.check_transitions(today)
    assert engine.state.tier is RehabTier.REHAB
    assert engine.state.drawdown_at_entry_usd == pytest.approx(-1000.0)

    # Recovery: only 400 of 1000 recovered (< 50% threshold) — stay in REHAB.
    # Current dd = -600 (peak +200, equity -400).
    engine.set_simulation_override(
        [
            (today - timedelta(days=3), 200.0),
            (today - timedelta(days=2), -400.0),
            (today - timedelta(days=1), -200.0),
        ]
    )
    held = await engine.check_transitions(today)
    assert held is None
    assert engine.state.tier is RehabTier.REHAB


@pytest.mark.asyncio
async def test_check_transitions_downgrade_fires_above_recovery_threshold(
    tmp_path: Path,
) -> None:
    """Once ≥ 50% of the entry drawdown is recovered + streak broken, tier drops.

    Downgrades need two things: (1) classify_tier to compute a weaker
    tier — so the streak must break with a green day, and cumulative
    drawdown must fall below the 3× daily-loss trigger; and (2) the
    ``_recovery_met`` check on the saved entry drawdown.
    """
    engine = RehabEngine(settings=_settings(), flag_path=tmp_path / "rehab.flag")
    today = date_cls(2026, 4, 15)
    engine.set_simulation_override(
        [
            (today - timedelta(days=3), 200.0),
            (today - timedelta(days=2), -400.0),
            (today - timedelta(days=1), -600.0),
        ]
    )
    await engine.check_transitions(today)
    assert engine.state.tier is RehabTier.REHAB
    # A green day breaks the red streak (consecutive=0, classify → NORMAL)
    # and drags equity back up. Entry drawdown was -1000; new drawdown is
    # -200 (peak +200, trough -200), so 800 recovered ≥ 500 threshold.
    engine.set_simulation_override(
        [
            (today - timedelta(days=3), 200.0),
            (today - timedelta(days=2), -400.0),
            (today - timedelta(days=1), 100.0),
        ]
    )
    transition = await engine.check_transitions(today)
    assert transition is not None
    assert transition.new_tier is RehabTier.NORMAL
    assert transition.reason == "recovery"


@pytest.mark.asyncio
async def test_check_transitions_disabled_is_noop(tmp_path: Path) -> None:
    """``rehab.enabled: false`` short-circuits check_transitions to None."""
    engine = RehabEngine(settings=_settings(enabled=False), flag_path=tmp_path / "rehab.flag")
    today = date_cls(2026, 4, 15)
    engine.set_simulation_override([(today - timedelta(days=4 - i), -50.0) for i in range(4)])
    assert await engine.check_transitions(today) is None
    assert engine.state.tier is RehabTier.NORMAL


def test_load_state_stale_flag_cleaned(tmp_path: Path) -> None:
    """A flag older than 30 days is silently deleted; returns NORMAL."""
    stale_record = RehabRecord(
        tier=RehabTier.REHAB,
        trigger_reason="consecutive_red_days",
        entered_at=datetime.now(UTC) - timedelta(days=45),
        drawdown_at_entry_usd=-450.0,
        consecutive_red_days_at_entry=2,
    )
    path = tmp_path / "rehab.flag"
    write_rehab_flag(path, stale_record)
    engine = RehabEngine(settings=_settings(), flag_path=path)
    state = engine.load_state()
    assert state.tier is RehabTier.NORMAL
    assert not path.exists()


def test_load_state_fresh_flag_adopted(tmp_path: Path) -> None:
    """A recent flag is adopted into engine state verbatim."""
    record = RehabRecord(
        tier=RehabTier.DEEP_REHAB,
        trigger_reason="cumulative_drawdown",
        entered_at=datetime.now(UTC) - timedelta(hours=3),
        drawdown_at_entry_usd=-2_000.0,
        consecutive_red_days_at_entry=5,
    )
    path = tmp_path / "rehab.flag"
    write_rehab_flag(path, record)
    engine = RehabEngine(settings=_settings(), flag_path=path)
    state = engine.load_state()
    assert state.tier is RehabTier.DEEP_REHAB
    assert state.trigger_reason == "cumulative_drawdown"
    assert state.consecutive_red_days_at_entry == 5


@pytest.mark.asyncio
async def test_simulation_override_drives_tier_without_journal(tmp_path: Path) -> None:
    """``set_simulation_override`` bypasses the journal entirely."""
    engine = RehabEngine(settings=_settings(), journal=None, flag_path=tmp_path / "rehab.flag")
    today = date_cls(2026, 4, 15)
    engine.set_simulation_override([(today - timedelta(days=4 - i), -50.0) for i in range(4)])
    stats = await engine.compute_stats(today)
    assert stats.consecutive_red_days == 4


# ---------- rule-based traceability ---------- #


def test_deep_rehab_one_setup_a_day(tmp_path: Path) -> None:
    """The methodology rule: *"one setup a day during a slump"* → DEEP_REHAB forces 1 trade/day.

    Guardrail test: any change that raises
    ``deep_rehab_max_trades_per_day`` above 1 must justify violating
    the explicit rule.
    """
    engine = RehabEngine(settings=_settings(), flag_path=tmp_path / "rehab.flag")
    engine.save_state(
        engine._build_entry_state(  # noqa: SLF001
            RehabTier.DEEP_REHAB,
            "consecutive_red_days",
            stats=compute_stats_from_journal_entries(
                {date_cls(2026, 4, 14): -50.0}, date_cls(2026, 4, 15), 10
            ),
        )
    )
    caps = engine.apply_to_caps()
    assert caps.max_trades_per_day == 1, (
        "the rule: one setup a day in a deep slump. deep_rehab_max_trades_per_day must stay at 1."
    )

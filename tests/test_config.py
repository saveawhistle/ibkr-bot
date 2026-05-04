"""Tests for ``bot.config`` — Phase 4i exit discipline validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from structlog.testing import capture_logs

from bot.config import DataSourcesSettings, ExecutionConfig, LoggingSettings, Settings


def test_runner_target_multiple_default_is_three() -> None:
    """Default ExecutionConfig ships with the documented 3R runner ceiling."""
    config = ExecutionConfig()
    assert config.runner_target_multiple == pytest.approx(3.0)


def test_runner_target_enabled_default_is_false() -> None:
    """Phase 4i default: the methodology does not plant hard profit ceilings on the runner."""
    config = ExecutionConfig()
    assert config.runner_target_enabled is False


def test_runner_target_multiple_below_scale_out_raises_when_enabled() -> None:
    """Runner ceiling must exceed scale-out when enabled — otherwise LMT fires first."""
    with pytest.raises(ValidationError) as exc_info:
        ExecutionConfig(
            runner_target_enabled=True,
            scale_out_multiple=2.0,
            runner_target_multiple=1.5,
        )
    assert "runner_target_multiple" in str(exc_info.value)


def test_runner_target_multiple_below_scale_out_ignored_when_disabled() -> None:
    """Disabled runner: the multiple is unused, so a sub-scale-out value is fine."""
    # No raise: validator short-circuits when the runner is off.
    ExecutionConfig(
        runner_target_enabled=False,
        scale_out_multiple=2.0,
        runner_target_multiple=0.5,
    )


def test_runner_target_multiple_equal_scale_out_raises_when_enabled() -> None:
    """Runner at exactly scale-out would fire the ceiling before the scale-out."""
    with pytest.raises(ValidationError) as exc_info:
        ExecutionConfig(
            runner_target_enabled=True,
            scale_out_multiple=2.0,
            runner_target_multiple=2.0,
        )
    assert "runner_target_multiple" in str(exc_info.value)


# ---------- Phase 4i scale-out multiple validation ---------- #


def test_scale_out_multiple_default_is_two() -> None:
    """the 2:1 R:R rule shows up as the default anchor."""
    assert ExecutionConfig().scale_out_multiple == pytest.approx(2.0)


def test_scale_out_multiple_below_one_raises() -> None:
    """Below 1R, the scale-out books the first half at a loss vs. initial risk."""
    with pytest.raises(ValidationError) as exc_info:
        ExecutionConfig(scale_out_multiple=0.8)
    assert "scale_out_multiple" in str(exc_info.value)


def test_scale_out_multiple_above_three_warns() -> None:
    """Values above 3R depart from the published rule — emit a warning."""
    with capture_logs() as captured:
        ExecutionConfig(scale_out_multiple=4.0)
    events = [e.get("event") for e in captured]
    assert "config.scale_out_multiple_high" in events


# ---------- Phase 4h: post-scale-out adjustable-trail validation ---------- #


def test_post_scaleout_trail_defaults() -> None:
    """Phase 6.14 defaults: immediate_trail mode, activation 1R, trail 1R.

    Replaces the Phase 4i ``post_scaleout_trail_enabled: bool`` with the
    ``post_scaleout_stop_mode`` enum. Default flipped from
    ``adjustable_to_trail`` to ``immediate_trail`` to lock in runner
    profit from the scale-out moment (the methodology-aligned).
    """
    config = ExecutionConfig()
    assert config.post_scaleout_stop_mode == "immediate_trail"
    assert config.trail_activation_r_multiple == pytest.approx(1.0)
    assert config.trail_amount_r_multiple == pytest.approx(1.0)


def test_trail_activation_r_multiple_negative_raises() -> None:
    """Negative activation is nonsensical — must fail loudly."""
    with pytest.raises(ValidationError) as exc_info:
        ExecutionConfig(trail_activation_r_multiple=-0.5)
    assert "trail_activation_r_multiple" in str(exc_info.value)


def test_trail_amount_r_multiple_zero_raises() -> None:
    """Zero-distance trail would exit on the first down-tick — must fail."""
    with pytest.raises(ValidationError) as exc_info:
        ExecutionConfig(trail_amount_r_multiple=0.0)
    assert "trail_amount_r_multiple" in str(exc_info.value)


def test_trail_amount_r_multiple_below_half_warns() -> None:
    """Values in (0.0, 0.5) are legal but emit a whip-out warning."""
    with capture_logs() as captured:
        ExecutionConfig(trail_amount_r_multiple=0.3)
    events = [e.get("event") for e in captured]
    assert "config.trail_amount_r_multiple_tight" in events


def test_trail_amount_r_multiple_at_half_does_not_warn() -> None:
    """Exactly 0.5 is the threshold — no warning at or above it."""
    with capture_logs() as captured:
        ExecutionConfig(trail_amount_r_multiple=0.5)
    events = [e.get("event") for e in captured]
    assert "config.trail_amount_r_multiple_tight" not in events


# ---------- Phase 4j entry-order-type + LMT-buffer validation ---------- #


def test_entry_order_type_default_is_stp_lmt() -> None:
    """Phase 4j default — server-side tick-level breakout trigger."""
    assert ExecutionConfig().entry_order_type == "STP_LMT"


def test_entry_order_type_lmt_accepted() -> None:
    """Legacy Phase 4i LMT behaviour remains available via explicit opt-in."""
    assert ExecutionConfig(entry_order_type="LMT").entry_order_type == "LMT"


def test_entry_order_type_mkt_accepted() -> None:
    """Phase 6.12 — the immediate-market-buy hotkey flow is a valid enum."""
    assert ExecutionConfig(entry_order_type="MKT").entry_order_type == "MKT"


def test_entry_order_type_invalid_value_fails() -> None:
    """Pydantic ``Literal`` rejects any value outside the three-member enum."""
    with pytest.raises(ValidationError) as exc_info:
        ExecutionConfig(entry_order_type="TRAILING_STP")  # type: ignore[arg-type]
    assert "entry_order_type" in str(exc_info.value)


def test_entry_limit_buffer_negative_fails() -> None:
    """Negative buffer would invert the STP/LMT relationship; IBKR would reject."""
    with pytest.raises(ValidationError) as exc_info:
        ExecutionConfig(entry_limit_buffer_usd=-0.05)
    assert "entry_limit_buffer_usd" in str(exc_info.value)


def test_entry_limit_buffer_large_warns() -> None:
    """Buffers above $0.50 allow meaningful slippage — warn but don't reject."""
    with capture_logs() as captured:
        ExecutionConfig(entry_limit_buffer_usd=0.75)
    events = [e.get("event") for e in captured]
    assert "config.entry_limit_buffer_usd_loose" in events


# ---------- Phase 5.1: DataSourcesSettings news window validation ---------- #


def test_data_sources_news_window_defaults() -> None:
    """Defaults cover Friday late → Monday scan (96h) with 72h classify cap."""
    cfg = DataSourcesSettings()
    assert cfg.news_lookback_hours == 96
    assert cfg.news_max_age_hours_for_classify == 72


def test_data_sources_news_lookback_hours_zero_raises() -> None:
    """Zero-hour fetch window means never see news — refuse."""
    with pytest.raises(ValidationError) as exc_info:
        DataSourcesSettings(news_lookback_hours=0)
    assert "news_lookback_hours" in str(exc_info.value)


def test_data_sources_news_max_age_zero_raises() -> None:
    """Zero-hour classify window means reject all news as stale — refuse."""
    with pytest.raises(ValidationError) as exc_info:
        DataSourcesSettings(news_max_age_hours_for_classify=0)
    assert "news_max_age_hours_for_classify" in str(exc_info.value)


def test_data_sources_classify_exceeds_fetch_raises() -> None:
    """Classifier window cannot exceed the fetch window — items beyond fetch don't exist."""
    with pytest.raises(ValidationError) as exc_info:
        DataSourcesSettings(
            news_lookback_hours=48,
            news_max_age_hours_for_classify=72,
        )
    assert "news_max_age_hours_for_classify" in str(exc_info.value)


# ---------- Phase 5.1: LoggingSettings ---------- #


def test_logging_settings_defaults() -> None:
    """Default: INFO level, JSON renderer, no file path (stdout only)."""
    cfg = LoggingSettings()
    assert cfg.level == "INFO"
    assert cfg.json_renderer is True
    assert cfg.path is None


def test_logging_settings_normalises_level_case() -> None:
    """``level`` is case-insensitive; stored upper-case."""
    cfg = LoggingSettings(level="debug")
    assert cfg.level == "DEBUG"


def test_logging_settings_invalid_level_raises() -> None:
    """Unknown level names fail loudly."""
    with pytest.raises(ValidationError) as exc_info:
        LoggingSettings(level="VERBOSE")
    assert "level" in str(exc_info.value)


def test_logging_settings_path_accepts_str(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """``path`` accepts a string and coerces to ``Path``."""
    from pathlib import Path

    cfg = LoggingSettings(path=str(tmp_path))
    assert isinstance(cfg.path, Path)
    assert cfg.path == tmp_path


# ---------- Phase 5.1: extra="forbid" regression ---------- #


def test_settings_rejects_unknown_top_level_keys() -> None:
    """Unknown top-level keys now fail validation (regression guard).

    The pre-5.1 ``extra="ignore"`` silently dropped a ``logging:`` block
    the operator added to config.yaml, masking the file-logging bug for
    weeks. ``extra="forbid"`` must surface drift the moment it appears.
    """
    with pytest.raises(ValidationError) as exc_info:
        Settings(bogus_top_level_key={"foo": "bar"})  # type: ignore[call-arg]
    assert "bogus_top_level_key" in str(exc_info.value)


# ---------- Phase 10.2: stop-distance floor validation ---------- #


def test_stop_floor_defaults() -> None:
    """Default StopFloorConfig matches the published 5¢ / 2% values."""
    from bot.config import StopFloorConfig

    cfg = StopFloorConfig()
    assert cfg.min_abs == pytest.approx(0.05)
    assert cfg.min_pct == pytest.approx(0.02)


def test_stop_floor_negative_min_abs_raises() -> None:
    """Negative absolute floor is nonsense — validator must reject."""
    from bot.config import StopFloorConfig

    with pytest.raises(ValidationError) as exc_info:
        StopFloorConfig(min_abs=-0.01)
    assert "min_abs" in str(exc_info.value)


def test_stop_floor_negative_min_pct_raises() -> None:
    """Negative percentage floor is nonsense — validator must reject."""
    from bot.config import StopFloorConfig

    with pytest.raises(ValidationError) as exc_info:
        StopFloorConfig(min_pct=-0.005)
    assert "min_pct" in str(exc_info.value)


def test_stop_floor_zero_values_accepted() -> None:
    """Both branches at 0.0 is a legal escape hatch (effectively disables the floor)."""
    from bot.config import StopFloorConfig

    cfg = StopFloorConfig(min_abs=0.0, min_pct=0.0)
    assert cfg.min_abs == 0.0
    assert cfg.min_pct == 0.0


def test_stop_floor_non_numeric_raises() -> None:
    """Non-numeric values rejected by Pydantic's float coercion."""
    from bot.config import StopFloorConfig

    with pytest.raises(ValidationError):
        StopFloorConfig(min_abs="ten cents")  # type: ignore[arg-type]


def test_stop_floor_attached_to_strategies_config() -> None:
    """``Settings.strategies.stop_floor`` exposes the nested config with defaults."""
    settings = Settings()
    assert settings.strategies.stop_floor.min_abs == pytest.approx(0.05)
    assert settings.strategies.stop_floor.min_pct == pytest.approx(0.02)

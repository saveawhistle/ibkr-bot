"""RVolMilestone activation tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from bot.exit_advisor.core.events import RVolDataUnavailable, RVolMilestone
from bot.exit_advisor.detectors.volume import VolumeDetector
from bot.exit_advisor.replay.bar_history import BarHistory
from bot.exit_advisor.replay.replay_source import Bar


def _ts(minute: int) -> datetime:
    return datetime(2026, 5, 5, 13, 30, tzinfo=UTC) + timedelta(minutes=minute)


def _bar(minute: int, volume: int) -> Bar:
    return Bar(_ts(minute), 10, 10.1, 9.9, 10.0, volume)


def test_milestone_fires_when_rvol_crosses_threshold() -> None:
    """Today's volume = 2x prior-day average → RVOL=2.0 fires."""
    detector = VolumeDetector(
        symbol="X",
        baseline_window_bars=2,
        rvol_milestones=[1.0, 2.0, 5.0],
        prior_day_cum_volume_by_minute={0: 100.0, 1: 200.0, 2: 300.0},
        rvol_prior_days_used=10,
        rvol_prior_days_configured=10,
    )
    history = BarHistory()
    out = []
    bars = [_bar(0, 200), _bar(1, 200), _bar(2, 200)]  # cum 200, 400, 600
    for b in bars:
        history.add_bar(b)
        out.extend(detector.on_bar(b, history))
    fires = [e for e in out if isinstance(e, RVolMilestone)]
    fired_milestones = {f.milestone for f in fires}
    assert 1.0 in fired_milestones
    assert 2.0 in fired_milestones


def test_milestone_carries_payload() -> None:
    detector = VolumeDetector(
        symbol="X",
        baseline_window_bars=2,
        rvol_milestones=[1.0],
        prior_day_cum_volume_by_minute={0: 100.0},
        rvol_prior_days_used=10,
        rvol_prior_days_configured=10,
    )
    history = BarHistory()
    out = []
    for b in [_bar(0, 200)]:
        history.add_bar(b)
        out.extend(detector.on_bar(b, history))
    fires = [e for e in out if isinstance(e, RVolMilestone)]
    assert fires[0].cumulative_volume_today == 200
    assert fires[0].prior_n_day_average_at_time == 100.0
    assert fires[0].prior_days_used == 10
    assert fires[0].rvol == 2.0


def test_once_per_milestone() -> None:
    """Each milestone fires at most once per session."""
    detector = VolumeDetector(
        symbol="X",
        baseline_window_bars=2,
        rvol_milestones=[1.0],
        prior_day_cum_volume_by_minute={0: 100.0, 1: 200.0, 2: 300.0},
        rvol_prior_days_used=10,
        rvol_prior_days_configured=10,
    )
    history = BarHistory()
    out = []
    bars = [_bar(0, 200), _bar(1, 200), _bar(2, 200)]
    for b in bars:
        history.add_bar(b)
        out.extend(detector.on_bar(b, history))
    fires = [e for e in out if isinstance(e, RVolMilestone) and e.milestone == 1.0]
    assert len(fires) == 1


def test_partial_data_emits_warning_and_milestones() -> None:
    """When prior_days_used < prior_days_configured, the warning fires
    once AND milestones still compute on the partial data."""
    detector = VolumeDetector(
        symbol="X",
        baseline_window_bars=2,
        rvol_milestones=[1.0],
        prior_day_cum_volume_by_minute={0: 100.0},
        rvol_prior_days_used=2,
        rvol_prior_days_configured=10,
    )
    history = BarHistory()
    out = []
    for b in [_bar(0, 200)]:
        history.add_bar(b)
        out.extend(detector.on_bar(b, history))
    warnings = [e for e in out if isinstance(e, RVolDataUnavailable)]
    assert len(warnings) == 1
    assert "2 of 10" in warnings[0].reason
    fires = [e for e in out if isinstance(e, RVolMilestone)]
    assert len(fires) == 1


def test_no_data_emits_warning_no_milestones() -> None:
    detector = VolumeDetector(
        symbol="X",
        baseline_window_bars=2,
        rvol_milestones=[1.0],
        prior_day_cum_volume_by_minute=None,
        rvol_prior_days_used=0,
        rvol_prior_days_configured=10,
    )
    history = BarHistory()
    out = []
    for b in [_bar(0, 200)]:
        history.add_bar(b)
        out.extend(detector.on_bar(b, history))
    warnings = [e for e in out if isinstance(e, RVolDataUnavailable)]
    fires = [e for e in out if isinstance(e, RVolMilestone)]
    assert len(warnings) == 1
    assert fires == []

"""Continued gating: only the truly-deferred event classes (market_context,
news, halts) still raise on enabled=True. Layer L2-A activated ``l2``."""

from __future__ import annotations

import pytest

from bot.config import ExitEventsConfig


def test_market_context_still_gated() -> None:
    with pytest.raises(ValueError, match="not implemented in this layer"):
        ExitEventsConfig.model_validate({"market_context": {"enabled": True}})


def test_news_still_gated() -> None:
    with pytest.raises(ValueError, match="not implemented in this layer"):
        ExitEventsConfig.model_validate({"news": {"enabled": True}})


def test_halts_still_gated() -> None:
    with pytest.raises(ValueError, match="not implemented in this layer"):
        ExitEventsConfig.model_validate({"halts": {"enabled": True}})


def test_l2_now_activatable() -> None:
    """Layer L2-A: l2 is no longer in the deferred-class list."""
    cfg = ExitEventsConfig.model_validate({"l2": {"enabled": True}})
    assert cfg.l2.enabled is True

"""Tests for DeepSeek integration in ModelRouter."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import nexus.core.model_router as model_router_mod
from nexus.core.model_router import (
    TIER_PRIMARY_DEEPSEEK,
    ModelRouter,
)


# ---------------------------------------------------------------------------
# Tier definition tests
# ---------------------------------------------------------------------------

def test_tier_primary_deepseek_exists() -> None:
    """TIER_PRIMARY_DEEPSEEK must be defined and contain deepseek-chat."""
    assert "deepseek/deepseek-chat" in TIER_PRIMARY_DEEPSEEK


def test_deepseek_cost_entry_exists() -> None:
    """deepseek/deepseek-chat must have a cost entry in MODEL_COSTS."""
    from nexus.core.model_router import MODEL_COSTS
    assert "deepseek/deepseek-chat" in MODEL_COSTS


# ---------------------------------------------------------------------------
# Key loading / routing priority tests
# ---------------------------------------------------------------------------

def test_deepseek_first_in_candidates_when_key_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When DEEPSEEK_API_KEY is set, deepseek/deepseek-chat is first candidate."""
    # Key must be in env so _get_deepseek_key() (instance method) picks it up.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-testkey")
    monkeypatch.setattr(model_router_mod, "_openrouter_key", None)
    monkeypatch.setattr(model_router_mod, "_gemini_key", None)
    monkeypatch.setattr(model_router_mod, "_groq_key", None)

    bl = tmp_path / "blacklist.json"
    bl.write_text(json.dumps({"version": 2, "entries": {}}), encoding="utf-8")
    cost = tmp_path / "cost.json"
    cost.write_text(json.dumps({"date": "2000-01-01", "total_cost": 0.0}), encoding="utf-8")
    monkeypatch.setattr(model_router_mod, "_BLACKLIST_FILE", bl)
    monkeypatch.setattr(model_router_mod, "_COST_FILE", cost)

    captured: list[str] = []

    def fake_try_model(self, model, messages, max_tokens, house, label):  # noqa: N802
        captured.append(model)
        return None  # force all models to fail so we capture the full list

    monkeypatch.setattr(ModelRouter, "_try_model", fake_try_model)

    router = ModelRouter()
    with pytest.raises(ValueError):
        router.complete(house="house_b", system="s", user="u")

    assert len(captured) > 0, "No models were tried"
    assert captured[0] == "deepseek/deepseek-chat", (
        f"Expected deepseek/deepseek-chat first, got: {captured[0]}"
    )


def test_deepseek_skipped_when_key_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When DEEPSEEK_API_KEY is absent, no deepseek/ model is attempted."""
    monkeypatch.setattr(model_router_mod, "_deepseek_key", None)
    monkeypatch.setattr(model_router_mod, "_openrouter_key", None)
    monkeypatch.setattr(model_router_mod, "_gemini_key", None)
    monkeypatch.setattr(model_router_mod, "_groq_key", None)

    bl = tmp_path / "blacklist.json"
    bl.write_text(json.dumps({"version": 2, "entries": {}}), encoding="utf-8")
    cost = tmp_path / "cost.json"
    cost.write_text(json.dumps({"date": "2000-01-01", "total_cost": 0.0}), encoding="utf-8")
    monkeypatch.setattr(model_router_mod, "_BLACKLIST_FILE", bl)
    monkeypatch.setattr(model_router_mod, "_COST_FILE", cost)

    captured: list[str] = []

    def fake_try_model(self, model, messages, max_tokens, house, label):  # noqa: N802
        captured.append(model)
        return None

    monkeypatch.setattr(ModelRouter, "_try_model", fake_try_model)

    router = ModelRouter()
    with pytest.raises(ValueError):
        router.complete(house="house_b", system="s", user="u")

    deepseek_attempts = [m for m in captured if m.startswith("deepseek/")]
    assert deepseek_attempts == [], (
        f"DeepSeek models should be skipped but were tried: {deepseek_attempts}"
    )


# ---------------------------------------------------------------------------
# _try_model: api_base and api_key injection
# ---------------------------------------------------------------------------

def test_try_model_sets_deepseek_api_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_try_model must set api_base=https://api.deepseek.com for deepseek/ models."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-testkey123")

    captured_kwargs: dict = {}

    def fake_completion(**kw):
        captured_kwargs.update(kw)
        resp = MagicMock()
        resp.choices[0].message.content = "hello from deepseek"
        return resp

    monkeypatch.setattr(model_router_mod.litellm, "completion", fake_completion)

    router = ModelRouter()
    result = router._try_model(
        "deepseek/deepseek-chat",
        [{"role": "user", "content": "hi"}],
        100,
        "house_b",
        "test",
    )

    assert result == "hello from deepseek"
    assert captured_kwargs.get("api_base") == "https://api.deepseek.com"
    assert captured_kwargs.get("api_key") == "sk-testkey123"


def test_try_model_returns_none_when_deepseek_key_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_try_model returns None immediately if DEEPSEEK_API_KEY is unset."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    completion_called = []

    def fake_completion(**kw):
        completion_called.append(kw)
        return MagicMock()

    monkeypatch.setattr(model_router_mod.litellm, "completion", fake_completion)

    router = ModelRouter()
    result = router._try_model(
        "deepseek/deepseek-chat",
        [{"role": "user", "content": "hi"}],
        100,
        "house_b",
        "test",
    )

    assert result is None
    assert completion_called == [], "litellm.completion should not be called without a key"


# ---------------------------------------------------------------------------
# Cost estimate
# ---------------------------------------------------------------------------

def test_deepseek_cost_estimate_is_nonzero() -> None:
    """deepseek/deepseek-chat should have a positive cost estimate (it's not free)."""
    router = ModelRouter()
    cost = router._estimate_cost("deepseek/deepseek-chat")
    assert cost > 0.0

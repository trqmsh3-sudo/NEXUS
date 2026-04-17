"""Tests for Gemini and Groq integration in ModelRouter (TDD — RED phase).

What's being added:
  1. _get_gemini_key() / _get_groq_key() — vault-aware key accessors
     (mirror _get_deepseek_key so keys can come from GuardianVault).
  2. _try_model explicit api_key injection for gemini/ and groq/ models
     (currently only deepseek/ and openrouter/ inject api_key explicitly).
  3. Groq rate limiting — _can_use_groq() with per-model RPM tracking
     persisted to data/groq_rate_limits.json (mirrors Gemini RL).
  4. Gemini/Groq removed from candidates when key absent.
  5. Gemini selected as first fallback when DeepSeek missing + Gemini key set.
  6. Groq selected when DeepSeek + Gemini missing + Groq key set.

Coverage:
  === Tier definitions ===
   1.  Gemini models present in TIER_0_GEMINI_FREE
   2.  Groq models present in TIER_1_FREE
   3.  Gemini cost entries exist in MODEL_COSTS
   4.  Groq cost entries exist in MODEL_COSTS

  === Key accessors ===
   5.  _get_gemini_key() returns env value when vault absent
   6.  _get_gemini_key() returns vault value when vault has key
   7.  _get_gemini_key() returns None when key absent everywhere
   8.  _get_groq_key() returns env value when vault absent
   9.  _get_groq_key() returns vault value when vault has key
  10.  _get_groq_key() returns None when key absent everywhere

  === Routing with keys present/absent ===
  11.  Gemini models excluded when GEMINI_API_KEY missing
  12.  Groq models excluded when GROQ_API_KEY missing
  13.  Gemini models included when GEMINI_API_KEY set
  14.  Groq models included when GROQ_API_KEY set
  15.  Gemini tried first after DeepSeek when no OpenRouter key
  16.  Groq tried after Gemini when no OpenRouter key

  === _try_model key injection ===
  17.  gemini/ model call includes api_key from _get_gemini_key()
  18.  groq/ model call includes api_key from _get_groq_key()
  19.  gemini/ skipped (returns None) when key absent
  20.  groq/ skipped (returns None) when key absent

  === Groq rate limiting ===
  21.  _can_use_groq() returns False when no key
  22.  _can_use_groq() returns True when key set and within RPM limit
  23.  _can_use_groq() returns False when at RPM limit
  24.  _can_use_groq() resets counter after 60s window expires
  25.  Groq RL state persisted to _GROQ_RL_FILE
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import nexus.core.model_router as model_router_mod
from nexus.core.model_router import (
    MODEL_COSTS,
    TIER_0_GEMINI_FREE,
    TIER_1_FREE,
    ModelRouter,
)


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _router_with_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModelRouter:
    bl = tmp_path / "blacklist.json"
    bl.write_text(json.dumps({"version": 2, "entries": {}}), encoding="utf-8")
    cost = tmp_path / "cost.json"
    cost.write_text(json.dumps({"date": "2000-01-01", "total_cost": 0.0}), encoding="utf-8")
    monkeypatch.setattr(model_router_mod, "_BLACKLIST_FILE", bl)
    monkeypatch.setattr(model_router_mod, "_COST_FILE", cost)
    return ModelRouter()


def _capture_all_models(
    router: ModelRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    """Run complete() with all models forced to fail; return list of tried models."""
    captured: list[str] = []

    def fake_try_model(self, model, messages, max_tokens, house, label):
        captured.append(model)
        return None

    monkeypatch.setattr(ModelRouter, "_try_model", fake_try_model)
    with pytest.raises(ValueError):
        router.complete(house="house_b", system="sys", user="usr")
    return captured


# ═══════════════════════════════════════════════════════════════════════════
# 1–4  Tier / cost definitions
# ═══════════════════════════════════════════════════════════════════════════

class TestTierDefinitions:

    def test_gemini_flash_in_tier0(self):
        assert "gemini/gemini-2.0-flash" in TIER_0_GEMINI_FREE

    def test_gemini_15_flash_in_tier0(self):
        assert "gemini/gemini-1.5-flash" in TIER_0_GEMINI_FREE

    def test_groq_70b_in_tier1(self):
        assert "groq/llama-3.3-70b-versatile" in TIER_1_FREE

    def test_groq_8b_in_tier1(self):
        assert "groq/llama-3.1-8b-instant" in TIER_1_FREE

    def test_gemini_cost_entries_exist(self):
        assert "gemini/gemini-2.0-flash" in MODEL_COSTS
        assert "gemini/gemini-1.5-flash" in MODEL_COSTS

    def test_groq_cost_entries_exist(self):
        assert "groq/llama-3.3-70b-versatile" in MODEL_COSTS
        assert "groq/llama-3.1-8b-instant" in MODEL_COSTS


# ═══════════════════════════════════════════════════════════════════════════
# 5–10  Key accessors
# ═══════════════════════════════════════════════════════════════════════════

class TestKeyAccessors:

    def test_get_gemini_key_from_env(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-gemini")
        router = ModelRouter()
        assert router._get_gemini_key() == "AIza-test-gemini"

    def test_get_gemini_key_from_vault(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        vault = MagicMock()
        vault.has.return_value = True
        vault.get.return_value = "AIza-vault-gemini"
        router = ModelRouter(vault=vault)
        assert router._get_gemini_key() == "AIza-vault-gemini"
        vault.get.assert_called_once_with("GEMINI_API_KEY")

    def test_get_gemini_key_returns_none_when_absent(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        router = ModelRouter()
        assert router._get_gemini_key() is None

    def test_get_groq_key_from_env(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test_groq")
        router = ModelRouter()
        assert router._get_groq_key() == "gsk_test_groq"

    def test_get_groq_key_from_vault(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        vault = MagicMock()
        vault.has.return_value = True
        vault.get.return_value = "gsk_vault_groq"
        router = ModelRouter(vault=vault)
        assert router._get_groq_key() == "gsk_vault_groq"
        vault.get.assert_called_once_with("GROQ_API_KEY")

    def test_get_groq_key_returns_none_when_absent(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        router = ModelRouter()
        assert router._get_groq_key() is None


# ═══════════════════════════════════════════════════════════════════════════
# 11–16  Routing with keys present / absent
# ═══════════════════════════════════════════════════════════════════════════

class TestRoutingWithKeys:

    def test_gemini_excluded_when_key_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_router_mod, "_gemini_key", None)
        monkeypatch.setattr(model_router_mod, "_groq_key", None)
        monkeypatch.setattr(model_router_mod, "_openrouter_key", None)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        router = _router_with_files(tmp_path, monkeypatch)
        tried = _capture_all_models(router, monkeypatch)
        gemini_tried = [m for m in tried if m.startswith("gemini/")]
        assert gemini_tried == [], f"Gemini tried without key: {gemini_tried}"

    def test_groq_excluded_when_key_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_router_mod, "_gemini_key", None)
        monkeypatch.setattr(model_router_mod, "_groq_key", None)
        monkeypatch.setattr(model_router_mod, "_openrouter_key", None)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        router = _router_with_files(tmp_path, monkeypatch)
        tried = _capture_all_models(router, monkeypatch)
        groq_tried = [m for m in tried if m.startswith("groq/")]
        assert groq_tried == [], f"Groq tried without key: {groq_tried}"

    def test_gemini_included_when_key_set(self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_router_mod, "_gemini_key", "AIza-test")
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
        monkeypatch.setattr(model_router_mod, "_groq_key", None)
        monkeypatch.setattr(model_router_mod, "_openrouter_key", None)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        router = _router_with_files(tmp_path, monkeypatch)
        tried = _capture_all_models(router, monkeypatch)
        gemini_tried = [m for m in tried if m.startswith("gemini/")]
        assert len(gemini_tried) > 0, "Gemini models should be in candidates"

    def test_groq_included_when_key_set(self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_router_mod, "_gemini_key", None)
        monkeypatch.setattr(model_router_mod, "_groq_key", "gsk_test")
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setattr(model_router_mod, "_openrouter_key", None)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        router = _router_with_files(tmp_path, monkeypatch)
        tried = _capture_all_models(router, monkeypatch)
        groq_tried = [m for m in tried if m.startswith("groq/")]
        assert len(groq_tried) > 0, "Groq models should be in candidates"

    def test_gemini_tried_before_groq_in_fallback(self, tmp_path, monkeypatch):
        """When DeepSeek absent, Gemini (tier 0) appears before Groq (tier 1)."""
        monkeypatch.setattr(model_router_mod, "_gemini_key", "AIza-test")
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
        monkeypatch.setattr(model_router_mod, "_groq_key", "gsk_test")
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setattr(model_router_mod, "_openrouter_key", None)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        router = _router_with_files(tmp_path, monkeypatch)
        tried = _capture_all_models(router, monkeypatch)
        first_gemini = next((i for i, m in enumerate(tried) if m.startswith("gemini/")), None)
        first_groq = next((i for i, m in enumerate(tried) if m.startswith("groq/")), None)
        assert first_gemini is not None, "No Gemini model tried"
        assert first_groq is not None, "No Groq model tried"
        assert first_gemini < first_groq, (
            f"Gemini (pos {first_gemini}) should come before Groq (pos {first_groq})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 17–20  _try_model key injection
# ═══════════════════════════════════════════════════════════════════════════

class TestTryModelKeyInjection:

    def test_gemini_call_includes_api_key(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-gemini-key")
        captured_kw: dict = {}

        def fake_completion(**kw):
            captured_kw.update(kw)
            resp = MagicMock()
            resp.choices[0].message.content = "gemini response"
            return resp

        monkeypatch.setattr(model_router_mod.litellm, "completion", fake_completion)
        router = ModelRouter()
        result = router._try_model(
            "gemini/gemini-2.0-flash",
            [{"role": "user", "content": "hi"}],
            100, "house_b", "test",
        )
        assert result == "gemini response"
        assert captured_kw.get("api_key") == "AIza-gemini-key"

    def test_groq_call_includes_api_key(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-groq-key")
        captured_kw: dict = {}

        def fake_completion(**kw):
            captured_kw.update(kw)
            resp = MagicMock()
            resp.choices[0].message.content = "groq response"
            return resp

        monkeypatch.setattr(model_router_mod.litellm, "completion", fake_completion)
        router = ModelRouter()
        result = router._try_model(
            "groq/llama-3.3-70b-versatile",
            [{"role": "user", "content": "hi"}],
            100, "house_b", "test",
        )
        assert result == "groq response"
        assert captured_kw.get("api_key") == "gsk-groq-key"

    def test_gemini_returns_none_when_key_absent(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        completion_called = []

        def fake_completion(**kw):
            completion_called.append(kw)
            return MagicMock()

        monkeypatch.setattr(model_router_mod.litellm, "completion", fake_completion)
        router = ModelRouter()
        result = router._try_model(
            "gemini/gemini-2.0-flash",
            [{"role": "user", "content": "hi"}],
            100, "house_b", "test",
        )
        assert result is None
        assert completion_called == [], "Should not call litellm without Gemini key"

    def test_groq_returns_none_when_key_absent(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        completion_called = []

        def fake_completion(**kw):
            completion_called.append(kw)
            return MagicMock()

        monkeypatch.setattr(model_router_mod.litellm, "completion", fake_completion)
        router = ModelRouter()
        result = router._try_model(
            "groq/llama-3.3-70b-versatile",
            [{"role": "user", "content": "hi"}],
            100, "house_b", "test",
        )
        assert result is None
        assert completion_called == [], "Should not call litellm without Groq key"


# ═══════════════════════════════════════════════════════════════════════════
# 21–25  Groq rate limiting
# ═══════════════════════════════════════════════════════════════════════════

class TestGroqRateLimiting:

    def test_can_use_groq_false_when_no_key(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        router = ModelRouter()
        assert router._can_use_groq("groq/llama-3.3-70b-versatile") is False

    def test_can_use_groq_true_within_limit(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        rl_file = tmp_path / "groq_rl.json"
        monkeypatch.setattr(model_router_mod, "_GROQ_RL_FILE", rl_file)
        router = ModelRouter()
        assert router._can_use_groq("groq/llama-3.3-70b-versatile") is True

    def test_can_use_groq_false_at_limit(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        rl_file = tmp_path / "groq_rl.json"
        monkeypatch.setattr(model_router_mod, "_GROQ_RL_FILE", rl_file)
        # Simulate window at max RPM (30 for 70b-versatile)
        now = datetime.now(timezone.utc)
        rl_file.write_text(json.dumps({
            "window_start": now.isoformat(),
            "counters": {"groq/llama-3.3-70b-versatile": 30},
        }))
        router = ModelRouter()
        assert router._can_use_groq("groq/llama-3.3-70b-versatile") is False

    def test_can_use_groq_resets_after_window(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        rl_file = tmp_path / "groq_rl.json"
        monkeypatch.setattr(model_router_mod, "_GROQ_RL_FILE", rl_file)
        # Window started 61 seconds ago — should be reset
        old_start = (datetime.now(timezone.utc) - timedelta(seconds=61)).isoformat()
        rl_file.write_text(json.dumps({
            "window_start": old_start,
            "counters": {"groq/llama-3.3-70b-versatile": 30},
        }))
        router = ModelRouter()
        assert router._can_use_groq("groq/llama-3.3-70b-versatile") is True

    def test_groq_rl_persisted_after_call(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        rl_file = tmp_path / "groq_rl.json"
        monkeypatch.setattr(model_router_mod, "_GROQ_RL_FILE", rl_file)
        router = ModelRouter()
        router._can_use_groq("groq/llama-3.3-70b-versatile")
        assert rl_file.exists()
        data = json.loads(rl_file.read_text())
        assert data["counters"].get("groq/llama-3.3-70b-versatile", 0) == 1

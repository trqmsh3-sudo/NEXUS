"""Tests for model blacklist TTL and legacy migration."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import nexus.core.model_router as model_router_mod
from nexus.core.model_router import ModelRouter


def test_legacy_blacklist_migrates_to_empty_ttl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bl = tmp_path / "model_blacklist.json"
    bl.write_text(
        json.dumps({"date": "2026-03-19", "models": ["gemini/gemini-2.0-flash"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(model_router_mod, "_BLACKLIST_FILE", bl)
    r = ModelRouter()
    entries = r._read_and_prune_blacklist_entries()
    assert entries == {}
    data = json.loads(bl.read_text(encoding="utf-8"))
    assert data.get("version") == 2
    assert data.get("entries") == {}


def test_ttl_prunes_expired_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bl = tmp_path / "model_blacklist.json"
    now = datetime.now(timezone.utc)
    past = (now - timedelta(hours=3)).isoformat()
    future = (now + timedelta(hours=2)).isoformat()
    bl.write_text(
        json.dumps(
            {
                "version": 2,
                "entries": {
                    "expired/model": past,
                    "active/model": future,
                },
            },
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(model_router_mod, "_BLACKLIST_FILE", bl)
    r = ModelRouter()
    entries = r._read_and_prune_blacklist_entries()
    assert set(entries.keys()) == {"active/model"}
    data = json.loads(bl.read_text(encoding="utf-8"))
    assert "expired/model" not in data.get("entries", {})
    assert "active/model" in data.get("entries", {})


def test_blacklist_model_sets_future_expiry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bl = tmp_path / "model_blacklist.json"
    bl.write_text(json.dumps({"version": 2, "entries": {}}), encoding="utf-8")
    monkeypatch.setattr(model_router_mod, "_BLACKLIST_FILE", bl)
    monkeypatch.setattr(model_router_mod, "_BLACKLIST_TTL_SECONDS", 3600)
    r = ModelRouter()
    r._blacklist_model("groq/test-model")
    data = json.loads(bl.read_text(encoding="utf-8"))
    exp = datetime.fromisoformat(data["entries"]["groq/test-model"])
    assert exp > datetime.now(timezone.utc)

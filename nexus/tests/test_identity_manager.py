"""Tests for nexus/core/identity_manager.py — TDD RED phase.

Coverage:
  === Identity loading ===
  1.  load_identities returns [] when data/identities.json missing
  2.  load_identities returns list of Identity objects from file
  3.  get_active_identity returns the identity with active=True
  4.  get_active_identity returns None when no identity is active
  5.  get_active_identity returns None when file missing

  === Saving ===
  6.  save_identity writes identity to identities.json
  7.  save_identity creates the data directory if missing
  8.  save_identity overwrites existing entry with same id

  === Vault/env resolution ===
  9.  resolved_email returns vault value when vault has GMAIL_ADDRESS
  10. resolved_email falls back to env var GMAIL_ADDRESS
  11. resolved_email returns raw value when not a vault sentinel
  12. resolved_paypal returns vault value when vault has PAYPAL_EMAIL
  13. resolved_paypal falls back to env var PAYPAL_EMAIL

  === Identity fields ===
  14. Identity has all required fields: id, name, business, niche, email, bio, paypal
  15. niche is stored and returned as a list
  16. active flag defaults to False

  === add_identity ===
  17. add_identity sets active=True when it is the first identity
  18. add_identity sets active=False when another identity already active
  19. add_identity persists to file and is loadable via load_identities
  20. get_active_identity returns resolved email (not vault sentinel)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from nexus.core.identity_manager import Identity, IdentityManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_identity(**kwargs) -> dict:
    defaults = {
        "id": "alex",
        "name": "Alex",
        "business": "Alex Digital Services",
        "niche": ["Virtual Assistant", "Data Entry", "Content Writing", "AI tasks"],
        "email": "vault:GMAIL_ADDRESS",
        "bio": "Experienced digital professional.",
        "paypal": "vault:PAYPAL_EMAIL",
        "active": True,
    }
    defaults.update(kwargs)
    return defaults


class _FakeVault:
    """Minimal vault stub for tests."""

    def __init__(self, data: dict[str, str] | None = None):
        self._data = data or {}

    def has(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str) -> str:
        return self._data[key]


# ---------------------------------------------------------------------------
# 1–5  load_identities / get_active_identity
# ---------------------------------------------------------------------------

class TestLoadIdentities:

    def test_returns_empty_list_when_file_missing(self, tmp_path):
        mgr = IdentityManager(data_dir=str(tmp_path))
        assert mgr.load_identities() == []

    def test_returns_identity_objects_from_file(self, tmp_path):
        ids_file = tmp_path / "identities.json"
        ids_file.write_text(json.dumps([_make_identity()]))
        mgr = IdentityManager(data_dir=str(tmp_path))
        result = mgr.load_identities()
        assert len(result) == 1
        assert isinstance(result[0], Identity)
        assert result[0].name == "Alex"

    def test_get_active_returns_active_identity(self, tmp_path):
        ids_file = tmp_path / "identities.json"
        ids_file.write_text(json.dumps([
            _make_identity(id="bob", name="Bob", active=False),
            _make_identity(id="alex", name="Alex", active=True),
        ]))
        mgr = IdentityManager(data_dir=str(tmp_path))
        active = mgr.get_active_identity()
        assert active is not None
        assert active.name == "Alex"

    def test_get_active_returns_none_when_none_active(self, tmp_path):
        ids_file = tmp_path / "identities.json"
        ids_file.write_text(json.dumps([_make_identity(active=False)]))
        mgr = IdentityManager(data_dir=str(tmp_path))
        assert mgr.get_active_identity() is None

    def test_get_active_returns_none_when_file_missing(self, tmp_path):
        mgr = IdentityManager(data_dir=str(tmp_path))
        assert mgr.get_active_identity() is None


# ---------------------------------------------------------------------------
# 6–8  save_identity
# ---------------------------------------------------------------------------

class TestSaveIdentity:

    def test_writes_identity_to_json_file(self, tmp_path):
        mgr = IdentityManager(data_dir=str(tmp_path))
        identity = Identity(**_make_identity())
        mgr.save_identity(identity)
        ids_file = tmp_path / "identities.json"
        assert ids_file.exists()
        data = json.loads(ids_file.read_text())
        assert len(data) == 1
        assert data[0]["id"] == "alex"

    def test_creates_data_dir_if_missing(self, tmp_path):
        new_dir = tmp_path / "subdir" / "data"
        mgr = IdentityManager(data_dir=str(new_dir))
        identity = Identity(**_make_identity())
        mgr.save_identity(identity)
        assert (new_dir / "identities.json").exists()

    def test_overwrites_existing_entry_with_same_id(self, tmp_path):
        ids_file = tmp_path / "identities.json"
        ids_file.write_text(json.dumps([_make_identity(bio="Old bio")]))
        mgr = IdentityManager(data_dir=str(tmp_path))
        updated = Identity(**_make_identity(bio="New bio"))
        mgr.save_identity(updated)
        data = json.loads(ids_file.read_text())
        assert len(data) == 1
        assert data[0]["bio"] == "New bio"


# ---------------------------------------------------------------------------
# 9–13  Vault / env resolution
# ---------------------------------------------------------------------------

class TestResolution:

    def test_resolved_email_from_vault(self, tmp_path):
        vault = _FakeVault({"GMAIL_ADDRESS": "vault@example.com"})
        mgr = IdentityManager(data_dir=str(tmp_path), vault=vault)
        identity = Identity(**_make_identity(email="vault:GMAIL_ADDRESS"))
        assert mgr.resolve_field(identity.email) == "vault@example.com"

    def test_resolved_email_falls_back_to_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GMAIL_ADDRESS", "env@example.com")
        vault = _FakeVault()  # empty vault
        mgr = IdentityManager(data_dir=str(tmp_path), vault=vault)
        identity = Identity(**_make_identity(email="vault:GMAIL_ADDRESS"))
        assert mgr.resolve_field(identity.email) == "env@example.com"

    def test_resolved_email_returns_raw_when_not_sentinel(self, tmp_path):
        mgr = IdentityManager(data_dir=str(tmp_path))
        identity = Identity(**_make_identity(email="direct@example.com"))
        assert mgr.resolve_field(identity.email) == "direct@example.com"

    def test_resolved_paypal_from_vault(self, tmp_path):
        vault = _FakeVault({"PAYPAL_EMAIL": "paypal@example.com"})
        mgr = IdentityManager(data_dir=str(tmp_path), vault=vault)
        identity = Identity(**_make_identity(paypal="vault:PAYPAL_EMAIL"))
        assert mgr.resolve_field(identity.paypal) == "paypal@example.com"

    def test_resolved_paypal_falls_back_to_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PAYPAL_EMAIL", "paypalenv@example.com")
        vault = _FakeVault()
        mgr = IdentityManager(data_dir=str(tmp_path), vault=vault)
        identity = Identity(**_make_identity(paypal="vault:PAYPAL_EMAIL"))
        assert mgr.resolve_field(identity.paypal) == "paypalenv@example.com"


# ---------------------------------------------------------------------------
# 14–16  Identity fields
# ---------------------------------------------------------------------------

class TestIdentityFields:

    def test_identity_has_required_fields(self):
        i = Identity(**_make_identity())
        assert i.id == "alex"
        assert i.name == "Alex"
        assert i.business == "Alex Digital Services"
        assert isinstance(i.niche, list)
        assert i.email == "vault:GMAIL_ADDRESS"
        assert i.bio == "Experienced digital professional."
        assert i.paypal == "vault:PAYPAL_EMAIL"

    def test_niche_stored_as_list(self):
        i = Identity(**_make_identity(niche=["VA", "Data Entry"]))
        assert i.niche == ["VA", "Data Entry"]

    def test_active_defaults_to_false(self):
        data = _make_identity()
        del data["active"]
        i = Identity(**data)
        assert i.active is False


# ---------------------------------------------------------------------------
# 17–20  add_identity
# ---------------------------------------------------------------------------

class TestAddIdentity:

    def test_first_identity_is_set_active(self, tmp_path):
        mgr = IdentityManager(data_dir=str(tmp_path))
        mgr.add_identity(Identity(**_make_identity(active=False)))
        active = mgr.get_active_identity()
        assert active is not None
        assert active.id == "alex"

    def test_second_identity_is_not_set_active(self, tmp_path):
        mgr = IdentityManager(data_dir=str(tmp_path))
        mgr.add_identity(Identity(**_make_identity(id="alex", name="Alex", active=True)))
        mgr.add_identity(Identity(**_make_identity(id="bob", name="Bob", active=False)))
        identities = mgr.load_identities()
        active = [i for i in identities if i.active]
        assert len(active) == 1
        assert active[0].id == "alex"

    def test_add_identity_persists_to_file(self, tmp_path):
        mgr = IdentityManager(data_dir=str(tmp_path))
        mgr.add_identity(Identity(**_make_identity()))
        ids_file = tmp_path / "identities.json"
        assert ids_file.exists()
        data = json.loads(ids_file.read_text())
        assert data[0]["name"] == "Alex"

    def test_get_active_returns_resolved_email(self, tmp_path):
        vault = _FakeVault({"GMAIL_ADDRESS": "alex@gmail.com", "PAYPAL_EMAIL": "alex@paypal.com"})
        mgr = IdentityManager(data_dir=str(tmp_path), vault=vault)
        mgr.add_identity(Identity(**_make_identity()))
        active = mgr.get_active_identity()
        assert active is not None
        resolved = mgr.resolve_identity(active)
        assert resolved["email"] == "alex@gmail.com"
        assert resolved["paypal"] == "alex@paypal.com"

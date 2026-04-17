"""Tests for OpenClaw Google session management (TDD — RED phase).

Problems being solved:
  1. OpenClaw doesn't detect an existing valid Google session before
     attempting automated login — wastes time, risks blocks.
  2. No session backup — if the profile is lost, cookies are gone.
  3. No expiry warning — session expires silently, causing failures.

Architecture:
  GoogleSessionManager reads the Chromium Cookies SQLite DB directly
  (no Playwright required).  It is decoupled from the browser so it
  can be tested without launching Chromium.

Coverage:
  1.  is_valid() False when profile dir does not exist
  2.  is_valid() False when Cookies DB does not exist
  3.  is_valid() True when google.com has non-expired auth cookies
  4.  is_valid() False when only expired Google cookies present
  5.  is_valid() False when no google.com cookies at all
  6.  hours_remaining() returns positive float for valid session
  7.  hours_remaining() returns 0 when session expired
  8.  hours_remaining() returns 0 when no profile
  9.  export_to_json() writes readable JSON with cookie data
  10. export_to_json() raises if Cookies DB missing
  11. import_from_json() creates a Playwright-compatible storage state
  12. import_from_json() round-trips: export then import preserves values
  13. session_summary() returns dict with is_valid, hours_remaining, count
  14. GOOGLE_AUTH_COOKIES contains __Host-GAPS and __Secure-ENID
  15. is_valid() uses GOOGLE_AUTH_COOKIES to detect real session
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from nexus.core.openclaw_session import GoogleSessionManager, GOOGLE_AUTH_COOKIES

# Chrome epoch offset: microseconds since 1601-01-01
_CHROME_EPOCH_US = 11_644_473_600 * 1_000_000


def _chrome_time(unix_ts: float) -> int:
    """Convert unix timestamp to Chrome cookie expiry microseconds."""
    return int((unix_ts + 11_644_473_600) * 1_000_000)


def _make_profile(tmp_path: Path, cookies: list[dict] | None = None) -> Path:
    """Create a minimal Chromium profile with a Cookies SQLite DB."""
    profile = tmp_path / "browser_profile"
    default = profile / "Default"
    default.mkdir(parents=True)

    conn = sqlite3.connect(str(default / "Cookies"))
    conn.execute("""
        CREATE TABLE cookies (
            host_key TEXT,
            name TEXT,
            value TEXT,
            path TEXT,
            expires_utc INTEGER,
            is_secure INTEGER,
            is_httponly INTEGER,
            last_access_utc INTEGER,
            has_expires INTEGER,
            is_persistent INTEGER,
            priority INTEGER DEFAULT 1,
            samesite INTEGER DEFAULT -1,
            source_scheme INTEGER DEFAULT 0,
            source_port INTEGER DEFAULT -1,
            last_update_utc INTEGER DEFAULT 0
        )
    """)

    if cookies:
        for c in cookies:
            conn.execute(
                "INSERT INTO cookies (host_key, name, value, path, expires_utc, "
                "is_secure, is_httponly, has_expires, is_persistent) VALUES "
                "(?,?,?,?,?,?,?,?,?)",
                (
                    c.get("host_key", ".google.com"),
                    c.get("name", "test"),
                    c.get("value", "val"),
                    c.get("path", "/"),
                    c.get("expires_utc", _chrome_time(time.time() + 3600)),
                    c.get("is_secure", 1),
                    c.get("is_httponly", 1),
                    1,
                    1,
                ),
            )

    conn.commit()
    conn.close()
    return profile


# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

class TestConstants:

    def test_google_auth_cookies_contains_host_gaps(self):
        assert "__Host-GAPS" in GOOGLE_AUTH_COOKIES

    def test_google_auth_cookies_contains_secure_enid(self):
        assert "__Secure-ENID" in GOOGLE_AUTH_COOKIES


# ═══════════════════════════════════════════════════════════════
#  1–5. is_valid()
# ═══════════════════════════════════════════════════════════════

class TestIsValid:

    def test_false_when_profile_dir_missing(self, tmp_path):
        mgr = GoogleSessionManager(str(tmp_path / "nonexistent"))
        assert mgr.is_valid() is False

    def test_false_when_cookies_db_missing(self, tmp_path):
        profile = tmp_path / "browser_profile"
        (profile / "Default").mkdir(parents=True)
        mgr = GoogleSessionManager(str(profile))
        assert mgr.is_valid() is False

    def test_true_when_google_auth_cookie_present_and_valid(self, tmp_path):
        future = time.time() + 86400 * 30  # 30 days from now
        profile = _make_profile(tmp_path, cookies=[
            {
                "host_key": "accounts.google.com",
                "name": "__Host-GAPS",
                "value": "1:abc123",
                "expires_utc": _chrome_time(future),
            }
        ])
        mgr = GoogleSessionManager(str(profile))
        assert mgr.is_valid() is True

    def test_false_when_only_expired_google_cookies(self, tmp_path):
        past = time.time() - 3600  # 1 hour ago
        profile = _make_profile(tmp_path, cookies=[
            {
                "host_key": "accounts.google.com",
                "name": "__Host-GAPS",
                "value": "1:expired",
                "expires_utc": _chrome_time(past),
            }
        ])
        mgr = GoogleSessionManager(str(profile))
        assert mgr.is_valid() is False

    def test_false_when_no_google_cookies_at_all(self, tmp_path):
        profile = _make_profile(tmp_path, cookies=[
            {
                "host_key": ".freelancer.com",
                "name": "_ga",
                "value": "GA1.2.xyz",
                "expires_utc": _chrome_time(time.time() + 86400),
            }
        ])
        mgr = GoogleSessionManager(str(profile))
        assert mgr.is_valid() is False

    def test_false_when_google_cookie_not_auth_type(self, tmp_path):
        """Non-auth Google cookies (e.g. _ga tracking) must not count as session."""
        future = time.time() + 86400 * 30
        profile = _make_profile(tmp_path, cookies=[
            {
                "host_key": ".google.com",
                "name": "_ga",  # tracking, not auth
                "value": "GA1.2.xyz",
                "expires_utc": _chrome_time(future),
            }
        ])
        mgr = GoogleSessionManager(str(profile))
        assert mgr.is_valid() is False


# ═══════════════════════════════════════════════════════════════
#  6–8. hours_remaining()
# ═══════════════════════════════════════════════════════════════

class TestHoursRemaining:

    def test_positive_for_valid_session(self, tmp_path):
        future = time.time() + 86400 * 10  # 240 hours
        profile = _make_profile(tmp_path, cookies=[
            {
                "host_key": "accounts.google.com",
                "name": "__Host-GAPS",
                "expires_utc": _chrome_time(future),
            }
        ])
        mgr = GoogleSessionManager(str(profile))
        hrs = mgr.hours_remaining()
        assert hrs > 200  # ~240h, allow some tolerance

    def test_zero_when_expired(self, tmp_path):
        past = time.time() - 3600
        profile = _make_profile(tmp_path, cookies=[
            {
                "host_key": "accounts.google.com",
                "name": "__Host-GAPS",
                "expires_utc": _chrome_time(past),
            }
        ])
        mgr = GoogleSessionManager(str(profile))
        assert mgr.hours_remaining() == 0

    def test_zero_when_no_profile(self, tmp_path):
        mgr = GoogleSessionManager(str(tmp_path / "missing"))
        assert mgr.hours_remaining() == 0


# ═══════════════════════════════════════════════════════════════
#  9–12. export_to_json() / import_from_json()
# ═══════════════════════════════════════════════════════════════

class TestExportImport:

    def test_export_writes_json(self, tmp_path):
        future = time.time() + 86400
        profile = _make_profile(tmp_path, cookies=[
            {
                "host_key": "accounts.google.com",
                "name": "__Host-GAPS",
                "value": "1:secret",
                "expires_utc": _chrome_time(future),
            }
        ])
        mgr = GoogleSessionManager(str(profile))
        out = tmp_path / "backup.json"
        mgr.export_to_json(str(out))
        assert out.exists()
        data = json.loads(out.read_text())
        assert "cookies" in data
        assert len(data["cookies"]) >= 1

    def test_export_raises_when_no_db(self, tmp_path):
        profile = tmp_path / "empty_profile"
        (profile / "Default").mkdir(parents=True)
        mgr = GoogleSessionManager(str(profile))
        with pytest.raises((FileNotFoundError, RuntimeError)):
            mgr.export_to_json(str(tmp_path / "out.json"))

    def test_import_creates_storage_state_file(self, tmp_path):
        """import_from_json() writes a Playwright-compatible storage state JSON."""
        cookie_data = {
            "cookies": [
                {
                    "name": "__Host-GAPS",
                    "value": "1:test",
                    "domain": "accounts.google.com",
                    "path": "/",
                    "expires": time.time() + 86400,
                    "httpOnly": True,
                    "secure": True,
                    "sameSite": "None",
                }
            ]
        }
        src = tmp_path / "backup.json"
        src.write_text(json.dumps(cookie_data))

        profile = _make_profile(tmp_path)
        mgr = GoogleSessionManager(str(profile))
        state_file = tmp_path / "storage_state.json"
        mgr.import_from_json(str(src), str(state_file))

        assert state_file.exists()
        state = json.loads(state_file.read_text())
        assert "cookies" in state

    def test_round_trip_preserves_cookie_name_and_domain(self, tmp_path):
        future = time.time() + 86400 * 30
        profile = _make_profile(tmp_path, cookies=[
            {
                "host_key": "accounts.google.com",
                "name": "__Host-GAPS",
                "value": "1:roundtrip",
                "expires_utc": _chrome_time(future),
            }
        ])
        mgr = GoogleSessionManager(str(profile))
        backup = tmp_path / "backup.json"
        mgr.export_to_json(str(backup))

        data = json.loads(backup.read_text())
        names = [c["name"] for c in data["cookies"]]
        domains = [c["domain"] for c in data["cookies"]]
        assert "__Host-GAPS" in names
        assert any("google" in d for d in domains)


# ═══════════════════════════════════════════════════════════════
#  13. session_summary()
# ═══════════════════════════════════════════════════════════════

class TestSessionSummary:

    def test_summary_contains_required_keys(self, tmp_path):
        profile = _make_profile(tmp_path)
        mgr = GoogleSessionManager(str(profile))
        s = mgr.session_summary()
        assert "is_valid" in s
        assert "hours_remaining" in s
        assert "cookie_count" in s
        assert "profile_exists" in s

    def test_summary_is_valid_false_for_empty_profile(self, tmp_path):
        profile = _make_profile(tmp_path)
        mgr = GoogleSessionManager(str(profile))
        s = mgr.session_summary()
        assert s["is_valid"] is False  # no google auth cookies

    def test_summary_valid_true_for_good_session(self, tmp_path):
        future = time.time() + 86400 * 30
        profile = _make_profile(tmp_path, cookies=[
            {
                "host_key": "accounts.google.com",
                "name": "__Host-GAPS",
                "expires_utc": _chrome_time(future),
            }
        ])
        mgr = GoogleSessionManager(str(profile))
        s = mgr.session_summary()
        assert s["is_valid"] is True
        assert s["hours_remaining"] > 0

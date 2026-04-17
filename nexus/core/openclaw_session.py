"""GoogleSessionManager — detect and manage OpenClaw's Google session.

Reads the Chromium Cookies SQLite database directly (no Playwright required)
to check whether a valid Google authenticated session exists in the persistent
browser profile.  Used by:

  - OpenClaw gateway.py to skip redundant login attempts
  - Monitoring scripts to warn before session expiry
  - Backup/restore tools to preserve session across profile resets
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Chrome stores cookie expiry as microseconds since 1601-01-01.
# Add this to a Chrome timestamp to get a Unix timestamp.
_CHROME_EPOCH_OFFSET_US = 11_644_473_600 * 1_000_000

# Cookies that indicate a real Google authenticated session.
# Tracking cookies (_ga, NID, etc.) do NOT count.
GOOGLE_AUTH_COOKIES: frozenset[str] = frozenset([
    "__Host-GAPS",   # Google Account Page Session
    "__Secure-ENID", # Enhanced session identifier
    "SID",           # Session ID (older format)
    "HSID",          # Secure session
    "SSID",          # SSL session
    "APISID",        # API session
    "SAPISID",       # Secure API session
])

_GOOGLE_HOSTS = frozenset([
    "accounts.google.com",
    ".google.com",
    "google.com",
])


def _chrome_to_unix(chrome_us: int) -> float:
    """Convert Chrome cookie expiry (microseconds since 1601) to Unix timestamp."""
    return (chrome_us - _CHROME_EPOCH_OFFSET_US) / 1_000_000


def _unix_to_chrome(unix_ts: float) -> int:
    """Convert Unix timestamp to Chrome cookie expiry microseconds."""
    return int((unix_ts * 1_000_000) + _CHROME_EPOCH_OFFSET_US)


class GoogleSessionManager:
    """Detect, export, and import Google session cookies for OpenClaw.

    Works by reading the Chromium Cookies SQLite database directly —
    no browser launch required.

    Args:
        profile_dir: Path to the Chromium persistent profile directory
                     (e.g. /opt/openclaw/browser_profile).
    """

    def __init__(self, profile_dir: str) -> None:
        self._profile = Path(profile_dir)
        self._cookies_db = self._profile / "Default" / "Cookies"

    # ──────────────────────────────────────────────────────────
    #  Session detection
    # ──────────────────────────────────────────────────────────

    def is_valid(self) -> bool:
        """Return True if a non-expired Google auth session exists in the profile."""
        return self.hours_remaining() > 0

    def hours_remaining(self) -> float:
        """Return hours until the Google session expires, or 0 if expired/missing."""
        if not self._cookies_db.exists():
            return 0.0

        now_chrome = _unix_to_chrome(time.time())
        try:
            conn = sqlite3.connect(str(self._cookies_db))
            rows = conn.execute(
                "SELECT name, expires_utc FROM cookies "
                "WHERE host_key IN ({hosts}) "
                "AND expires_utc > ? "
                "AND name IN ({names})".format(
                    hosts=",".join("?" * len(_GOOGLE_HOSTS)),
                    names=",".join("?" * len(GOOGLE_AUTH_COOKIES)),
                ),
                (*_GOOGLE_HOSTS, now_chrome, *GOOGLE_AUTH_COOKIES),
            ).fetchall()
            conn.close()
        except Exception as exc:
            logger.warning("GoogleSessionManager: cookie read failed: %s", exc)
            return 0.0

        if not rows:
            return 0.0

        # Return hours until the longest-lived auth cookie expires
        max_expiry_chrome = max(exp for _, exp in rows)
        remaining_secs = _chrome_to_unix(max_expiry_chrome) - time.time()
        return max(0.0, remaining_secs / 3600)

    def session_summary(self) -> dict[str, Any]:
        """Return a snapshot of the current session state."""
        profile_exists = self._profile.exists()
        if not profile_exists or not self._cookies_db.exists():
            return {
                "is_valid": False,
                "hours_remaining": 0,
                "cookie_count": 0,
                "profile_exists": profile_exists,
            }

        try:
            conn = sqlite3.connect(str(self._cookies_db))
            total = conn.execute("SELECT count(*) FROM cookies").fetchone()[0]
            conn.close()
        except Exception:
            total = 0

        hrs = self.hours_remaining()
        return {
            "is_valid": hrs > 0,
            "hours_remaining": round(hrs, 1),
            "cookie_count": total,
            "profile_exists": True,
        }

    # ──────────────────────────────────────────────────────────
    #  Export / Import
    # ──────────────────────────────────────────────────────────

    def export_to_json(self, output_path: str) -> int:
        """Export all cookies from the profile to a JSON backup file.

        Args:
            output_path: Path to write the JSON backup.

        Returns:
            Number of cookies exported.

        Raises:
            FileNotFoundError: If the Cookies database does not exist.
        """
        if not self._cookies_db.exists():
            raise FileNotFoundError(
                f"Cookies database not found at {self._cookies_db}"
            )

        conn = sqlite3.connect(str(self._cookies_db))
        rows = conn.execute(
            "SELECT host_key, name, value, path, expires_utc, "
            "is_secure, is_httponly FROM cookies"
        ).fetchall()
        conn.close()

        cookies: list[dict[str, Any]] = []
        for host, name, value, path, expires_chrome, secure, httponly in rows:
            unix_exp = _chrome_to_unix(expires_chrome) if expires_chrome else None
            cookies.append({
                "domain": host,
                "name": name,
                "value": value,
                "path": path,
                "expires": unix_exp,
                "secure": bool(secure),
                "httpOnly": bool(httponly),
                "sameSite": "None",
            })

        data = {
            "exported_at": time.time(),
            "profile": str(self._profile),
            "cookies": cookies,
        }
        Path(output_path).write_text(json.dumps(data, indent=2))
        logger.info(
            "GoogleSessionManager: exported %d cookies to %s", len(cookies), output_path
        )
        return len(cookies)

    def import_from_json(self, input_path: str, storage_state_path: str) -> int:
        """Convert a JSON backup into a Playwright storage_state file.

        The storage_state file can be passed to `context.storage_state(path=...)`
        or loaded via `browser.new_context(storage_state=...)`.

        Args:
            input_path:         Path to the JSON backup written by export_to_json().
            storage_state_path: Path to write the Playwright storage_state JSON.

        Returns:
            Number of cookies written.
        """
        data = json.loads(Path(input_path).read_text())
        raw_cookies = data.get("cookies", [])

        # Playwright storage state format
        state: dict[str, Any] = {
            "cookies": raw_cookies,
            "origins": [],
        }
        Path(storage_state_path).write_text(json.dumps(state, indent=2))
        logger.info(
            "GoogleSessionManager: imported %d cookies → %s",
            len(raw_cookies),
            storage_state_path,
        )
        return len(raw_cookies)

    # ──────────────────────────────────────────────────────────
    #  Factory
    # ──────────────────────────────────────────────────────────

    @classmethod
    def default(cls) -> "GoogleSessionManager":
        """Return a manager pointing at the standard OpenClaw profile directory."""
        return cls("/opt/openclaw/browser_profile")

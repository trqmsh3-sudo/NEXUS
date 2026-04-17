"""OpenClaw session_manager.py — deployed to /opt/openclaw/session_manager.py

Reads Google session status from the Chromium profile and provides
backup/restore utilities.  Imported by gateway.py.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CHROME_EPOCH_OFFSET_US = 11_644_473_600 * 1_000_000

GOOGLE_AUTH_COOKIES: frozenset[str] = frozenset([
    "__Host-GAPS",
    "__Secure-ENID",
    "SID",
    "HSID",
    "SSID",
    "APISID",
    "SAPISID",
])

_GOOGLE_HOSTS = frozenset([
    "accounts.google.com",
    ".google.com",
    "google.com",
])

PROFILE_DIR = Path("/opt/openclaw/browser_profile")
BACKUP_PATH = Path("/opt/openclaw/session_backup.json")


def _chrome_to_unix(chrome_us: int) -> float:
    return (chrome_us - _CHROME_EPOCH_OFFSET_US) / 1_000_000


def _unix_to_chrome(unix_ts: float) -> int:
    return int((unix_ts * 1_000_000) + _CHROME_EPOCH_OFFSET_US)


def is_google_session_valid(profile_dir: Path = PROFILE_DIR) -> bool:
    """Return True if a non-expired Google auth session exists."""
    return google_session_hours_remaining(profile_dir) > 0


def google_session_hours_remaining(profile_dir: Path = PROFILE_DIR) -> float:
    """Return hours until the Google session expires, 0 if missing/expired."""
    cookies_db = profile_dir / "Default" / "Cookies"
    if not cookies_db.exists():
        return 0.0

    now_chrome = _unix_to_chrome(time.time())
    try:
        conn = sqlite3.connect(str(cookies_db))
        hosts_ph = ",".join("?" * len(_GOOGLE_HOSTS))
        names_ph = ",".join("?" * len(GOOGLE_AUTH_COOKIES))
        rows = conn.execute(
            f"SELECT name, expires_utc FROM cookies "
            f"WHERE host_key IN ({hosts_ph}) "
            f"AND expires_utc > ? "
            f"AND name IN ({names_ph})",
            (*_GOOGLE_HOSTS, now_chrome, *GOOGLE_AUTH_COOKIES),
        ).fetchall()
        conn.close()
    except Exception as exc:
        logger.warning("session_manager: cookie read error: %s", exc)
        return 0.0

    if not rows:
        return 0.0

    max_expiry = max(exp for _, exp in rows)
    remaining = _chrome_to_unix(max_expiry) - time.time()
    return max(0.0, remaining / 3600)


def session_summary(profile_dir: Path = PROFILE_DIR) -> dict[str, Any]:
    """Return a dict with session status for logging/Telegram reporting."""
    hours = google_session_hours_remaining(profile_dir)
    cookies_db = profile_dir / "Default" / "Cookies"
    total = 0
    if cookies_db.exists():
        try:
            conn = sqlite3.connect(str(cookies_db))
            total = conn.execute("SELECT count(*) FROM cookies").fetchone()[0]
            conn.close()
        except Exception:
            pass
    return {
        "is_valid": hours > 0,
        "hours_remaining": round(hours, 1),
        "cookie_count": total,
        "profile_exists": profile_dir.exists(),
    }


def backup_session(
    profile_dir: Path = PROFILE_DIR,
    backup_path: Path = BACKUP_PATH,
) -> int:
    """Export all cookies to JSON backup. Returns cookie count."""
    cookies_db = profile_dir / "Default" / "Cookies"
    if not cookies_db.exists():
        raise FileNotFoundError(f"No Cookies DB at {cookies_db}")

    conn = sqlite3.connect(str(cookies_db))
    rows = conn.execute(
        "SELECT host_key, name, value, path, expires_utc, is_secure, is_httponly "
        "FROM cookies"
    ).fetchall()
    conn.close()

    cookies = []
    for host, name, value, path, exp_chrome, secure, httponly in rows:
        cookies.append({
            "domain": host,
            "name": name,
            "value": value,
            "path": path,
            "expires": _chrome_to_unix(exp_chrome) if exp_chrome else None,
            "secure": bool(secure),
            "httpOnly": bool(httponly),
            "sameSite": "None",
        })

    data = {"exported_at": time.time(), "cookies": cookies}
    backup_path.write_text(json.dumps(data, indent=2))
    logger.info("session_manager: backed up %d cookies → %s", len(cookies), backup_path)
    return len(cookies)


if __name__ == "__main__":
    summary = session_summary()
    print(f"Google session valid: {summary['is_valid']}")
    print(f"Hours remaining:      {summary['hours_remaining']:.0f}h")
    print(f"Total cookies:        {summary['cookie_count']}")
    print(f"Profile exists:       {summary['profile_exists']}")

    if summary["is_valid"]:
        n = backup_session()
        print(f"Backup saved:         {n} cookies → {BACKUP_PATH}")
    else:
        print("WARNING: No valid Google session — run manual_login.py to re-authenticate")

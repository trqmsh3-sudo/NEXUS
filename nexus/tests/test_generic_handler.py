"""Tests for generic_handler.py fixes (TDD — RED phase).

Bugs being fixed:
  1. Cookie injection — load Google session from SQLite into browser context
     before navigating so gemini.google.com (and other Google sites) receive
     the stored auth cookies instead of an anonymous request.

  2. Nav link filtering — _extract_listings must skip anchor text that
     starts with action verbs ("Hire", "Post", "Find", ...) which are
     site navigation, not actual job listings.  Links must also be
     prioritised when they fall under the target URL's path.

Coverage:
  === Cookie loading (_load_session_cookies) ===
  1.  Returns empty list when profile dir does not exist
  2.  Returns empty list when Cookies DB does not exist
  3.  Returns non-expired cookies from the DB
  4.  Excludes cookies whose expires_utc is in the past
  5.  Returned dicts have required Playwright add_cookies() keys
  6.  Includes is_secure and is_httponly flags correctly

  === Nav link detection (_is_nav_link) ===
  7.  "Hire Freelancers" → True
  8.  "Post a Project" → True
  9.  "Find a Developer" → True
 10.  "Browse Jobs" → True
 11.  "Search Gigs" → True
 12.  "Senior Python Developer — $75/hr" → False
 13.  "Python backend for API integration project" → False
 14.  Case-insensitive: "HIRE engineers" → True

  === _extract_listings with nav filtering ===
 15.  "Hire Freelancers" nav link excluded from results
 16.  "Post a Job" nav link excluded from results
 17.  Actual listing with keyword match IS included
 18.  Links under target URL path get higher priority score
 19.  Links NOT under target path still included but ranked lower
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import helpers — scripts/openclaw_generic_handler.py is not a package,
# so we add the scripts/ directory to sys.path.
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from openclaw_generic_handler import (  # noqa: E402
    _load_session_cookies,
    _is_nav_link,
    _extract_listings,
)

# Chrome epoch offset (microseconds since 1601-01-01)
_CHROME_EPOCH_US = 11_644_473_600 * 1_000_000


def _chrome_time(unix_ts: float) -> int:
    return int(unix_ts * 1_000_000 + _CHROME_EPOCH_US)


def _make_cookies_db(path: Path, cookies: list[dict]) -> None:
    """Create a minimal Chromium Cookies SQLite DB for tests."""
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE cookies (
            host_key TEXT, name TEXT, value TEXT, path TEXT,
            expires_utc INTEGER, is_secure INTEGER, is_httponly INTEGER
        )
    """)
    for c in cookies:
        conn.execute(
            "INSERT INTO cookies VALUES (?,?,?,?,?,?,?)",
            (
                c.get("host_key", ".google.com"),
                c.get("name", "test"),
                c.get("value", "val"),
                c.get("path", "/"),
                c.get("expires_utc", _chrome_time(time.time() + 3600)),
                c.get("is_secure", 1),
                c.get("is_httponly", 1),
            ),
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Mock Playwright objects for _extract_listings
# ---------------------------------------------------------------------------

class _MockAnchor:
    def __init__(self, href: str, text: str):
        self._href = href
        self._text = text

    async def get_attribute(self, attr: str):
        return self._href if attr == "href" else None

    async def inner_text(self):
        return self._text


class _MockPage:
    def __init__(self, anchors: list[_MockAnchor], url: str = "https://www.freelancer.com/projects"):
        self._anchors = anchors
        self.url = url

    async def query_selector_all(self, _selector: str):
        return self._anchors


# ═══════════════════════════════════════════════════════════════════════════
# 1–6  _load_session_cookies
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadSessionCookies:

    def test_empty_when_profile_dir_missing(self, tmp_path):
        result = _load_session_cookies(str(tmp_path / "nonexistent"))
        assert result == []

    def test_empty_when_cookies_db_missing(self, tmp_path):
        profile = tmp_path / "browser_profile"
        (profile / "Default").mkdir(parents=True)
        result = _load_session_cookies(str(profile))
        assert result == []

    def test_returns_non_expired_cookies(self, tmp_path):
        profile = tmp_path / "profile"
        db_dir = profile / "Default"
        db_dir.mkdir(parents=True)
        future = time.time() + 3600
        _make_cookies_db(
            db_dir / "Cookies",
            [{"host_key": ".google.com", "name": "SID", "value": "abc",
              "expires_utc": _chrome_time(future)}],
        )
        result = _load_session_cookies(str(profile))
        assert len(result) == 1
        assert result[0]["name"] == "SID"

    def test_excludes_expired_cookies(self, tmp_path):
        profile = tmp_path / "profile"
        db_dir = profile / "Default"
        db_dir.mkdir(parents=True)
        past = time.time() - 3600
        _make_cookies_db(
            db_dir / "Cookies",
            [{"host_key": ".google.com", "name": "SID", "value": "old",
              "expires_utc": _chrome_time(past)}],
        )
        result = _load_session_cookies(str(profile))
        assert result == []

    def test_returned_dicts_have_playwright_required_keys(self, tmp_path):
        profile = tmp_path / "profile"
        db_dir = profile / "Default"
        db_dir.mkdir(parents=True)
        _make_cookies_db(db_dir / "Cookies", [
            {"host_key": ".google.com", "name": "SSID", "value": "xyz"}
        ])
        result = _load_session_cookies(str(profile))
        assert len(result) == 1
        c = result[0]
        assert "name" in c
        assert "value" in c
        assert "domain" in c
        assert "path" in c
        assert "expires" in c
        assert "sameSite" in c

    def test_includes_secure_and_httponly_flags(self, tmp_path):
        profile = tmp_path / "profile"
        db_dir = profile / "Default"
        db_dir.mkdir(parents=True)
        _make_cookies_db(db_dir / "Cookies", [
            {"host_key": ".google.com", "name": "HSID", "value": "h",
             "is_secure": 1, "is_httponly": 1}
        ])
        result = _load_session_cookies(str(profile))
        assert result[0]["secure"] is True
        assert result[0]["httpOnly"] is True


# ═══════════════════════════════════════════════════════════════════════════
# 7–14  _is_nav_link
# ═══════════════════════════════════════════════════════════════════════════

class TestIsNavLink:

    def test_hire_freelancers_is_nav(self):
        assert _is_nav_link("Hire Freelancers") is True

    def test_post_a_project_is_nav(self):
        assert _is_nav_link("Post a Project") is True

    def test_find_a_developer_is_nav(self):
        assert _is_nav_link("Find a Developer") is True

    def test_browse_jobs_is_nav(self):
        assert _is_nav_link("Browse Jobs") is True

    def test_search_gigs_is_nav(self):
        assert _is_nav_link("Search Gigs") is True

    def test_actual_listing_is_not_nav(self):
        assert _is_nav_link("Senior Python Developer — $75/hr") is False

    def test_job_description_is_not_nav(self):
        assert _is_nav_link("Python backend for API integration project") is False

    def test_case_insensitive(self):
        assert _is_nav_link("HIRE engineers") is True


# ═══════════════════════════════════════════════════════════════════════════
# 15–19  _extract_listings with nav link filtering
# ═══════════════════════════════════════════════════════════════════════════

class TestExtractListingsNavFilter:

    @pytest.mark.asyncio
    async def test_hire_nav_link_excluded(self):
        anchors = [
            _MockAnchor("https://www.freelancer.com/freelancers", "Hire Freelancers"),
            _MockAnchor("https://www.freelancer.com/projects/123", "Python API developer needed"),
        ]
        page = _MockPage(anchors)
        results = await _extract_listings(page, "https://www.freelancer.com/projects", ["python"])
        urls = [r["url"] for r in results]
        assert "https://www.freelancer.com/freelancers" not in urls

    @pytest.mark.asyncio
    async def test_post_a_job_nav_link_excluded(self):
        anchors = [
            _MockAnchor("https://www.freelancer.com/post-project", "Post a Job"),
            _MockAnchor("https://www.freelancer.com/projects/456", "Django backend engineer wanted — remote"),
        ]
        page = _MockPage(anchors)
        results = await _extract_listings(page, "https://www.freelancer.com/projects", ["django"])
        urls = [r["url"] for r in results]
        assert "https://www.freelancer.com/post-project" not in urls

    @pytest.mark.asyncio
    async def test_actual_listing_included(self):
        anchors = [
            _MockAnchor("https://www.freelancer.com/projects/789", "Python developer needed for data pipeline"),
        ]
        page = _MockPage(anchors)
        results = await _extract_listings(page, "https://www.freelancer.com/projects", ["python", "data"])
        assert len(results) == 1
        assert results[0]["url"] == "https://www.freelancer.com/projects/789"

    @pytest.mark.asyncio
    async def test_links_under_target_path_ranked_higher(self):
        """Links whose URL starts with the target path get a score bonus."""
        target = "https://www.freelancer.com/projects"
        anchors = [
            # Off-path: matches keyword but not under /projects
            _MockAnchor("https://www.freelancer.com/freelancers/python-experts",
                        "Top Python freelancers for hire"),
            # On-path: under /projects — should rank first
            _MockAnchor("https://www.freelancer.com/projects/999",
                        "Python API developer needed urgently"),
        ]
        page = _MockPage(anchors, url=target)
        results = await _extract_listings(page, target, ["python"])
        assert len(results) >= 1
        # The on-path link must appear first
        assert results[0]["url"] == "https://www.freelancer.com/projects/999"

    @pytest.mark.asyncio
    async def test_off_path_links_still_included_but_ranked_lower(self):
        """Links not under target path are still returned if they match keywords."""
        target = "https://www.freelancer.com/projects"
        anchors = [
            _MockAnchor("https://www.freelancer.com/projects/888",
                        "Senior Python engineer for API work"),
            _MockAnchor("https://toptal.com/python",
                        "Python remote contract developer"),
        ]
        page = _MockPage(anchors, url=target)
        results = await _extract_listings(page, target, ["python"])
        urls = [r["url"] for r in results]
        assert "https://www.freelancer.com/projects/888" in urls
        assert "https://toptal.com/python" in urls
        # On-path link ranked first
        assert results[0]["url"] == "https://www.freelancer.com/projects/888"

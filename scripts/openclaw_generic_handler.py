"""OpenClaw generic browser task handler (v4).

Deployed to: /opt/openclaw/generic_handler.py

Fixes in v4:
  1. Cookie injection — loads Google session cookies from the persistent
     Chrome profile's SQLite DB and injects them into the browser context
     via ctx.add_cookies() before navigating.  Fixes Gemini and other
     Google-authenticated targets always redirecting to google.com.

  2. Nav link filtering — _extract_listings now skips anchors whose text
     starts with action verbs ("Hire", "Post", "Find", ...) that indicate
     site navigation rather than actual job/gig listings.

  3. Path-priority scoring — links whose URL falls under the task's target
     path prefix (e.g. /projects for freelancer.com/projects) get a bonus
     score so they rank above off-site links that happen to match keywords.
"""
from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
import time
from pathlib import Path
from urllib.parse import urlparse

log = logging.getLogger("openclaw")

# ---------------------------------------------------------------------------
# Profile directory — where the persistent Chrome session is stored.
# Override by setting OPENCLAW_PROFILE_DIR environment variable.
# ---------------------------------------------------------------------------
import os as _os
_DEFAULT_PROFILE_DIR = _os.getenv(
    "OPENCLAW_PROFILE_DIR", "/opt/openclaw/browser_profile"
)

# Chrome stores cookie expiry as microseconds since 1601-01-01.
_CHROME_EPOCH_OFFSET_US = 11_644_473_600 * 1_000_000

# ---------------------------------------------------------------------------
# Keywords / patterns
# ---------------------------------------------------------------------------

_GIG_KEYWORDS = (
    "python", "javascript", "typescript", "ai", "ml", "data", "devops",
    "web", "django", "flask", "react", "node", "api", "freelance",
    "remote", "contract", "gig", "hire", "hiring", "developer",
    "engineer", "designer", "consultant", "writer",
)

_SKIP_TEXT = (
    "log in", "sign in", "sign up", "register", "privacy", "terms",
    "cookie", "about us", "contact", "careers", "help", "support",
    "blog", "subscribe", "newsletter", "footer", "menu", "navigation",
)

# Nav-verb prefixes — link text starting with these is site navigation,
# not a real listing.  Case-insensitive match against .lower().
_NAV_VERB_PREFIXES: tuple[str, ...] = (
    "hire ",
    "post ",
    "find ",
    "browse ",
    "search ",
    "join ",
    "view ",
    "get ",
    "start ",
    "create ",
    "explore ",
    "discover ",
    "learn ",
    "try ",
    "see all",
    "show all",
    "load more",
    "read more",
)

_PRICE_RE = re.compile(
    r"([\$€£]\s?\d{2,5}(?:[,.]\d{3})*(?:\s?[-–]\s?[\$€£]?\s?\d{2,5})?"
    r"(?:\s?(?:/hr|/hour|/day|/month|/mo|/project|per hour|fixed))?)",
    re.IGNORECASE,
)

_URL_RE = re.compile(r"https?://[^\s)\"'<>\]]+")

_DOMAIN_PATH_RE = re.compile(
    r"(?i)\b((?:[a-z0-9][a-z0-9-]*\.)+(?:com|org|net|io|co|dev|app|gov|edu)"
    r"(?:/[^\s)\"'<>\]]*)?)"
)


# ---------------------------------------------------------------------------
# Bug 1 fix: session cookie loading
# ---------------------------------------------------------------------------

def _load_session_cookies(profile_dir: str) -> list[dict]:
    """Read all non-expired cookies from the Chrome profile SQLite DB.

    Returns a list of cookie dicts compatible with Playwright's
    BrowserContext.add_cookies() — keys: name, value, domain, path,
    expires, secure, httpOnly, sameSite.

    Returns empty list if the profile or Cookies DB does not exist, or
    if the DB cannot be read.
    """
    db = Path(profile_dir) / "Default" / "Cookies"
    if not db.exists():
        return []

    now_chrome = int(time.time() * 1_000_000 + _CHROME_EPOCH_OFFSET_US)
    try:
        conn = sqlite3.connect(str(db))
        rows = conn.execute(
            "SELECT host_key, name, value, path, expires_utc, "
            "is_secure, is_httponly FROM cookies WHERE expires_utc > ?",
            (now_chrome,),
        ).fetchall()
        conn.close()
    except Exception as exc:
        log.warning("session cookie load failed: %s", exc)
        return []

    cookies: list[dict] = []
    for host, name, value, path, exp_chrome, secure, httponly in rows:
        unix_exp = (exp_chrome - _CHROME_EPOCH_OFFSET_US) / 1_000_000
        cookies.append({
            "name": name,
            "value": value,
            "domain": host,
            "path": path or "/",
            "expires": unix_exp,
            "secure": bool(secure),
            "httpOnly": bool(httponly),
            "sameSite": "None",
        })
    return cookies


# ---------------------------------------------------------------------------
# Bug 2 fix: nav link detection
# ---------------------------------------------------------------------------

def _is_nav_link(text: str) -> bool:
    """Return True if anchor text looks like site navigation, not a listing.

    Detects action-verb phrases like "Hire Freelancers", "Post a Job",
    "Find a Developer" that are navigation buttons, not job listings.
    """
    low = text.lower().strip()
    return any(low.startswith(prefix) for prefix in _NAV_VERB_PREFIXES)


# ---------------------------------------------------------------------------
# URL / keyword extraction
# ---------------------------------------------------------------------------

def _extract_target_url(task: str) -> str | None:
    """Find an explicit URL or domain+path mention in the task string."""
    urls = _URL_RE.findall(task)
    if urls:
        return urls[0]
    for m in _DOMAIN_PATH_RE.finditer(task):
        d = m.group(1).lower()
        host = d.split("/")[0]
        if host in ("example.com", "localhost", "openclaw.com"):
            continue
        # Only prepend www. for bare second-level domains (e.g. google.com).
        # Subdomains (gemini.google.com, remote.co) must not get www. added.
        host_parts = host.split(".")
        if len(host_parts) == 2 and not host.startswith("www."):
            prefix = "https://www."
        else:
            prefix = "https://"
        return prefix + d
    return None


def _extract_keywords(task: str) -> list[str]:
    """Pull gig-related keywords from the task string."""
    low = task.lower()
    seen: set[str] = set()
    out: list[str] = []
    for kw in _GIG_KEYWORDS:
        if kw in low and kw not in seen:
            seen.add(kw)
            out.append(kw)
    return out


# ---------------------------------------------------------------------------
# Listing extraction (async — uses Playwright page)
# ---------------------------------------------------------------------------

async def _extract_listings(page, target_url: str, keywords: list[str]) -> list[dict]:
    """Scan anchors on the current page for gig-like listings.

    Changes vs v3:
    - Skips nav links (_is_nav_link check).
    - Gives +1 score bonus to links under the target URL path prefix.
    """
    listings: list[dict] = []
    try:
        anchors = await page.query_selector_all("a[href]")
    except Exception as exc:
        log.warning("anchor scan failed: %s", exc)
        return listings

    seen_hrefs: set[str] = set()
    parsed_target = urlparse(target_url or page.url)
    target_path = parsed_target.path.rstrip("/")

    for a in anchors[:200]:
        try:
            href = await a.get_attribute("href") or ""
            text = (await a.inner_text() or "").strip()
        except Exception:
            continue

        if not href or not text or len(text) < 15:
            continue

        low_text = text.lower()

        # Skip known boilerplate text
        if any(skip in low_text for skip in _SKIP_TEXT):
            continue

        # Bug 2 fix: skip navigation verb phrases
        if _is_nav_link(text):
            continue

        # Resolve relative URLs
        if href.startswith("/"):
            href = f"{parsed_target.scheme}://{parsed_target.netloc}{href}"
        elif href.startswith("#") or not href.startswith("http"):
            continue

        if href in seen_hrefs:
            continue

        # Score: keyword match
        if keywords and any(k in low_text for k in keywords):
            match_score = 2
        elif any(
            term in low_text
            for term in ("developer", "engineer", "freelance", "contract", "remote", "project")
        ):
            match_score = 1
        else:
            continue

        # Bug 2 fix: path-priority bonus
        parsed_href = urlparse(href)
        if (
            target_path
            and parsed_href.netloc == parsed_target.netloc
            and parsed_href.path.startswith(target_path + "/")
        ):
            match_score += 1

        seen_hrefs.add(href)
        listings.append({"title": text[:150], "url": href, "score": match_score})
        if len(listings) >= 10:
            break

    listings.sort(key=lambda x: -x["score"])
    return listings[:5]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_generic_browser_task(
    ctx,
    task: str,
    profile_dir: str = _DEFAULT_PROFILE_DIR,
) -> str:
    """Execute a generic browser task: inject cookies, navigate, extract.

    Args:
        ctx:         Playwright BrowserContext (persistent or ephemeral).
        task:        Natural-language task string from House C.
        profile_dir: Path to the Chrome persistent profile used for cookie
                     injection.  Defaults to /opt/openclaw/browser_profile.
    """
    # Bug 1 fix: inject Google session cookies before any navigation
    cookies = _load_session_cookies(profile_dir)
    if cookies:
        try:
            await ctx.add_cookies(cookies)
            log.info("Injected %d session cookies from profile", len(cookies))
        except Exception as exc:
            log.warning("Cookie injection failed: %s", exc)
    else:
        log.warning("No session cookies found at %s", profile_dir)

    page = await ctx.new_page()
    try:
        target_url = _extract_target_url(task)
        keywords = _extract_keywords(task)
        log.info(
            "Generic task: target=%s keywords=%s",
            target_url or "<search>",
            keywords[:5],
        )

        if target_url:
            try:
                await page.goto(target_url, timeout=30000, wait_until="domcontentloaded")
                await asyncio.sleep(3)
            except Exception as exc:
                log.warning("goto failed: %s — falling back to search", exc)
                try:
                    await page.close()
                except Exception:
                    pass
                page = await ctx.new_page()
                target_url = None

        if not target_url:
            query = "+".join(keywords[:5]) or "freelance+gigs+today"
            search_url = f"https://www.google.com/search?q={query}"
            try:
                await page.goto(search_url, timeout=25000, wait_until="domcontentloaded")
                await asyncio.sleep(3)
                target_url = search_url
            except Exception as exc:
                try:
                    await page.close()
                except Exception:
                    pass
                return f"NO_DATA: Browser could not reach search engine — {exc}"

        title = ""
        current_url = ""
        body_text = ""
        try:
            title = await page.title()
            current_url = page.url
            body_text = await page.locator("body").inner_text()
        except Exception as exc:
            log.warning("page read failed: %s", exc)

        low_body = body_text.lower()[:3000]

        # Detect login walls
        if any(
            sig in low_body
            for sig in (
                "please log in", "sign in to continue", "login required",
                "create an account to", "you must be logged in",
            )
        ):
            await page.close()
            return (
                f"NO_DATA: {current_url} requires login to view content "
                f"(title: {title[:80]})"
            )

        # Detect CAPTCHA
        if any(sig in low_body for sig in ("prove your humanity", "verify you are human", "captcha")):
            await page.close()
            return f"NO_DATA: {current_url} shows CAPTCHA challenge — cannot proceed"

        # Extract prices from whole page
        prices = _PRICE_RE.findall(body_text)
        top_price = prices[0] if prices else ""

        listings = await _extract_listings(page, target_url, keywords)

        await page.close()

        if not listings:
            snippet = body_text.strip().replace("\n", " ")[:200]
            return (
                f"NO_DATA: Reached {current_url} (title: {title[:80]}) — "
                f"no listings matching keywords={keywords[:3] or 'any'}. "
                f"Page snippet: {snippet}"
            )

        findings: list[str] = []
        for lst in listings[:3]:
            price_for = ""
            for p in prices[:20]:
                if p:
                    price_for = p
                    break
            if not price_for:
                price_for = top_price or "rate not listed"
            findings.append(f"FINDING: {lst['title']} | {lst['url']} | {price_for}")

        return "\n".join(findings)

    except Exception as exc:
        log.error("generic task error: %s", exc)
        try:
            await page.close()
        except Exception:
            pass
        return f"NO_DATA: Browser error — {exc}"

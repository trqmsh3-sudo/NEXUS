"""DirectJobFetcher — fetch remote job listings via direct HTTP APIs.

Bypasses browser automation entirely. Uses remoteok.com's public JSON API
which requires no authentication and is not blocked by Cloudflare.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import List

logger = logging.getLogger(__name__)

_REMOTEOK_API = "https://remoteok.com/api"
_MAX_RESULTS = 5
_REQUEST_TIMEOUT = 15


def _extract_keywords(task: str) -> list[str]:
    """Extract meaningful keywords from the task string."""
    stopwords = {
        "find", "search", "get", "fetch", "one", "a", "an", "the", "on", "in",
        "for", "from", "at", "of", "and", "or", "remote", "job", "gig", "listing",
        "today", "posted", "freelance",
    }
    words = task.lower().split()
    return [w for w in words if w not in stopwords and len(w) > 2]


def _job_matches(job: dict, keywords: list[str]) -> bool:
    """Return True if any keyword appears in the job's searchable text."""
    if not keywords:
        return True
    position = (job.get("position") or "").lower()
    company = (job.get("company") or "").lower()
    tags = " ".join(job.get("tags") or []).lower()
    description = (job.get("description") or "")[:200].lower()
    text = f"{position} {company} {tags} {description}"
    return any(kw in text for kw in keywords)


def _format_finding(job: dict) -> str:
    """Format a job dict as a FINDING line."""
    position = job.get("position") or "Unknown Position"
    company = job.get("company") or "Unknown Company"
    url = job.get("url") or job.get("apply_url") or ""
    salary_min = job.get("salary_min") or 0
    salary_max = job.get("salary_max") or 0

    salary_str = ""
    if salary_min and salary_max:
        salary_str = f" | ${salary_min}-${salary_max}/yr"
    elif salary_min:
        salary_str = f" | ${salary_min}+/yr"

    tags = job.get("tags") or []
    tags_str = f" | {', '.join(tags[:4])}" if tags else ""

    return f"FINDING: {position} at {company} | {url}{salary_str}{tags_str}"


class DirectJobFetcher:
    """Fetch remote job listings via direct HTTP APIs, no browser required."""

    def fetch_remoteok(self, task: str) -> str:
        """Query remoteok.com JSON API and return FINDING lines matching task keywords."""
        keywords = _extract_keywords(task)
        logger.info("DirectJobFetcher: fetching remoteok.com keywords=%s", keywords)

        try:
            req = urllib.request.Request(
                _REMOTEOK_API,
                headers={"User-Agent": "Mozilla/5.0 (compatible; NEXUS/1.0)"},
            )
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
                raw = resp.read()
        except Exception as exc:
            logger.warning("DirectJobFetcher: remoteok fetch failed — %s", exc)
            return ""

        try:
            data = json.loads(raw)
        except Exception as exc:
            logger.warning("DirectJobFetcher: remoteok JSON parse failed — %s", exc)
            return ""

        # First element is metadata, rest are job listings
        jobs = [item for item in data if isinstance(item, dict) and "position" in item]
        matched = [j for j in jobs if _job_matches(j, keywords)]

        if not matched:
            logger.info("DirectJobFetcher: no matches for keywords=%s in %d jobs", keywords, len(jobs))
            return ""

        findings = [_format_finding(j) for j in matched[:_MAX_RESULTS]]
        logger.info("DirectJobFetcher: found %d/%d matching jobs", len(findings), len(jobs))
        return "\n".join(findings)

    def fetch(self, task: str) -> str:
        """Try all job sources and return the first non-empty result."""
        result = self.fetch_remoteok(task)
        if result:
            return result
        logger.warning("DirectJobFetcher: all sources returned empty for task=%r", task[:80])
        return ""

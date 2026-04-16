"""External signal provider for NEXUS — IRON LAW: House A receives fresh data every N cycles.

Fetches real-world business intelligence from free, public sources (no API key required).
Converts external data into BeliefCertificates for injection into House A.

Sources:
  - Reddit (r/entrepreneur, r/forhire) — trending business discussions and hiring demand
  - Google Trends RSS                  — what markets are searching for right now
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

from nexus.core.belief_certificate import BeliefCertificate
from nexus.core.text_utils import clean_text

logger: logging.Logger = logging.getLogger(__name__)

# Reddit JSON API — no key required, needs a descriptive User-Agent
_REDDIT_ENTREPRENEUR_URL: str = "https://www.reddit.com/r/entrepreneur/hot.json?limit=5"
_REDDIT_FORHIRE_URL: str = "https://www.reddit.com/r/forhire/hot.json?limit=5"

# Google Trends daily trending searches RSS (US market)
_GOOGLE_TRENDS_URL: str = (
    "https://trends.google.com/trends/trendingsearches/daily/rss?geo=US"
)

_REDDIT_HEADERS: dict[str, str] = {
    "User-Agent": "NEXUS/2.8 business-intelligence (+https://github.com/trqmsh3-sudo/NEXUS)",
    "Accept": "application/json",
}


class ExternalSignalProvider:
    """Fetches external business intelligence and converts to BeliefCertificates.

    All sources are free and require no API key. If a source fails, it is
    logged and skipped; the provider never raises.
    """

    def fetch_all(self) -> list[BeliefCertificate]:
        """Fetch from all sources and return combined list of BeliefCertificates.

        On per-source failure: log warning, skip that source, continue.

        Returns:
            Combined list of beliefs from Reddit and Google Trends.
        """
        beliefs: list[BeliefCertificate] = []
        beliefs.extend(self.fetch_reddit())
        beliefs.extend(self.fetch_trends())
        return beliefs

    def fetch_reddit(self) -> list[BeliefCertificate]:
        """Fetch hot posts from r/entrepreneur and r/forhire.

        Returns BeliefCertificates representing trending business discussions
        and active hiring demand. Confidence is scaled from post score.

        Returns:
            List of BeliefCertificates with domain "Business Opportunity".
        """
        beliefs: list[BeliefCertificate] = []
        sources = [
            ("entrepreneur", _REDDIT_ENTREPRENEUR_URL),
            ("forhire", _REDDIT_FORHIRE_URL),
        ]
        for subreddit, url in sources:
            try:
                req = urllib.request.Request(url, headers=_REDDIT_HEADERS)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
                logger.warning("ExternalSignal Reddit r/%s fetch failed: %s", subreddit, exc)
                continue
            except Exception as exc:
                logger.warning("ExternalSignal Reddit r/%s unexpected error: %s", subreddit, exc)
                continue

            children = data.get("data", {}).get("children", [])
            for child in children:
                post = child.get("data", {})
                if not isinstance(post, dict):
                    continue
                title = clean_text(post.get("title") or "")
                score = int(post.get("score") or 0)
                if not title:
                    continue

                confidence = min(0.92, max(0.55, score / 5000.0))
                claim = clean_text(f"Trending on r/{subreddit}: {title}")
                if len(claim) > 500:
                    claim = claim[:497] + "..."
                proof = clean_text(f"# Reddit r/{subreddit} upvotes: {score}")
                bc = BeliefCertificate(
                    claim=claim,
                    source=f"Reddit r/{subreddit}",
                    confidence=confidence,
                    domain="Business Opportunity",
                    executable_proof=proof,
                    decay_rate=0.5,
                )
                beliefs.append(bc)
        return beliefs

    def fetch_trends(self) -> list[BeliefCertificate]:
        """Fetch Google Trends daily trending searches for the US market.

        Returns BeliefCertificates representing what consumers and businesses
        are actively searching for right now.

        Returns:
            List of BeliefCertificates with domain "Market Trends".
        """
        beliefs: list[BeliefCertificate] = []
        try:
            req = urllib.request.Request(_GOOGLE_TRENDS_URL)
            with urllib.request.urlopen(req, timeout=15) as resp:
                xml_bytes = resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            logger.warning("ExternalSignal Google Trends fetch failed: %s", exc)
            return beliefs
        except Exception as exc:
            logger.warning("ExternalSignal Google Trends unexpected error: %s", exc)
            return beliefs

        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as exc:
            logger.warning("ExternalSignal Google Trends XML parse failed: %s", exc)
            return beliefs

        # Items are direct children of <channel>; skip the channel <title> element
        channel = root.find("channel")
        if channel is None:
            return beliefs

        for item in channel.findall("item"):
            title_elem = item.find("title")
            term = clean_text(
                (title_elem.text or "").strip() if title_elem is not None else ""
            )
            if not term:
                continue
            claim = clean_text(f"Currently trending in search: {term}")
            if len(claim) > 500:
                claim = claim[:497] + "..."
            proof = clean_text(f"# Google Trends daily: {term}")
            bc = BeliefCertificate(
                claim=claim,
                source="Google Trends RSS",
                confidence=0.80,
                domain="Market Trends",
                executable_proof=proof,
                decay_rate=0.7,
            )
            beliefs.append(bc)
        return beliefs

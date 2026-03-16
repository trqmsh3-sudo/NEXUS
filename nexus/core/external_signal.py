"""External signal provider for NEXUS — IRON LAW: House A receives fresh data every N cycles.

Fetches real-world knowledge from free, public APIs (no API key required).
Converts external data into BeliefCertificates for injection into House A.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

from nexus.core.belief_certificate import BeliefCertificate

logger: logging.Logger = logging.getLogger(__name__)

_GITHUB_URL: str = (
    "https://api.github.com/search/repositories"
    "?q=stars:>1000&sort=stars&order=desc&per_page=5"
)
_ARXIV_URL: str = (
    "http://export.arxiv.org/api/query"
    "?search_query=cat:cs.AI&sortBy=submittedDate"
    "&sortOrder=descending&max_results=3"
)

# ArXiv Atom namespace
ATOM_NS: str = "http://www.w3.org/2005/Atom"


class ExternalSignalProvider:
    """Fetches external knowledge from GitHub and ArXiv and converts to BeliefCertificates.

    All sources are free and require no API key. If a source fails, it is
    logged and skipped; the provider never raises.
    """

    def fetch_all(self) -> list[BeliefCertificate]:
        """Fetch from all sources and return combined list of BeliefCertificates.

        On per-source failure: log warning, skip that source, continue.

        Returns:
            Combined list of beliefs from GitHub and ArXiv.
        """
        beliefs: list[BeliefCertificate] = []
        beliefs.extend(self.fetch_github())
        beliefs.extend(self.fetch_arxiv())
        return beliefs

    def fetch_github(self) -> list[BeliefCertificate]:
        """Fetch top trending GitHub repos and convert to BeliefCertificates.

        Returns:
            List of BeliefCertificates for highly-starred repositories.
        """
        beliefs: list[BeliefCertificate] = []
        try:
            req = urllib.request.Request(
                _GITHUB_URL,
                headers={"Accept": "application/vnd.github.v3+json"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
            logger.warning("ExternalSignal GitHub fetch failed: %s", exc)
            return beliefs

        items = data.get("items", [])
        for item in items:
            if not isinstance(item, dict):
                continue
            full_name = item.get("full_name") or item.get("name") or "unknown"
            description = item.get("description") or ""
            language = item.get("language") or "general"
            stars = int(item.get("stargazers_count", 0))
            confidence = min(0.95, stars / 100000.0)
            if confidence < 0.5:
                confidence = 0.5
            claim = f"{full_name} is a highly-starred {language} project: {description}"
            if len(claim) > 500:
                claim = claim[:497] + "..."
            proof = f"# GitHub stars: {stars}"
            bc = BeliefCertificate(
                claim=claim,
                source="GitHub Trending",
                confidence=confidence,
                domain="Software Engineering",
                executable_proof=proof,
                decay_rate=0.3,
            )
            beliefs.append(bc)
        return beliefs

    def fetch_arxiv(self) -> list[BeliefCertificate]:
        """Fetch latest ArXiv AI papers and convert to BeliefCertificates.

        Returns:
            List of BeliefCertificates for recent AI research.
        """
        beliefs: list[BeliefCertificate] = []
        try:
            req = urllib.request.Request(_ARXIV_URL)
            with urllib.request.urlopen(req, timeout=15) as resp:
                xml_bytes = resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            logger.warning("ExternalSignal ArXiv fetch failed: %s", exc)
            return beliefs

        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as exc:
            logger.warning("ExternalSignal ArXiv parse failed: %s", exc)
            return beliefs

        ns = {"atom": ATOM_NS}
        entries = root.findall("atom:entry", ns)
        for entry in entries:
            title_elem = entry.find("atom:title", ns)
            title = (title_elem.text or "").strip() if title_elem is not None else ""
            if not title:
                continue
            claim = f"Recent AI research: {title}"
            if len(claim) > 500:
                claim = claim[:497] + "..."
            proof = f"# ArXiv paper: {title}"
            bc = BeliefCertificate(
                claim=claim,
                source="ArXiv API",
                confidence=0.75,
                domain="AI Research",
                executable_proof=proof,
                decay_rate=0.4,
            )
            beliefs.append(bc)
        return beliefs

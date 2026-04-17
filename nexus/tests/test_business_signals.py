"""Tests for business-oriented ExternalSignalProvider.

All tests fail against the current GitHub+ArXiv implementation
and pass after the Reddit+Trends replacement.
"""
from __future__ import annotations

import io
import json
import urllib.error
import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch

import pytest

from nexus.core.external_signal import ExternalSignalProvider
from nexus.core.belief_certificate import BeliefCertificate


# ---------------------------------------------------------------------------
# Method surface — old methods gone, new ones present
# ---------------------------------------------------------------------------

def test_provider_has_fetch_reddit() -> None:
    assert hasattr(ExternalSignalProvider, "fetch_reddit"), (
        "ExternalSignalProvider must have fetch_reddit()"
    )


def test_provider_has_fetch_trends() -> None:
    assert hasattr(ExternalSignalProvider, "fetch_trends"), (
        "ExternalSignalProvider must have fetch_trends()"
    )


def test_provider_no_fetch_github() -> None:
    assert not hasattr(ExternalSignalProvider, "fetch_github"), (
        "fetch_github() must be removed — replace with fetch_reddit()"
    )


def test_provider_no_fetch_arxiv() -> None:
    assert not hasattr(ExternalSignalProvider, "fetch_arxiv"), (
        "fetch_arxiv() must be removed — replace with fetch_trends()"
    )


# ---------------------------------------------------------------------------
# fetch_all() delegates correctly
# ---------------------------------------------------------------------------

def test_fetch_all_calls_reddit_and_trends() -> None:
    provider = ExternalSignalProvider()
    reddit_called = []
    trends_called = []

    def fake_reddit(self=None):
        reddit_called.append(1)
        return []

    def fake_trends(self=None):
        trends_called.append(1)
        return []

    provider.fetch_reddit = fake_reddit
    provider.fetch_trends = fake_trends
    provider.fetch_all()

    assert reddit_called, "fetch_all() did not call fetch_reddit()"
    assert trends_called, "fetch_all() did not call fetch_trends()"


# ---------------------------------------------------------------------------
# fetch_reddit() — happy path with mocked HTTP
# ---------------------------------------------------------------------------

_REDDIT_PAYLOAD = json.dumps({
    "data": {
        "children": [
            {"data": {"title": "How I made $2k with a simple newsletter", "score": 1200}},
            {"data": {"title": "Finding underserved niches in 2024", "score": 800}},
        ]
    }
}).encode()


def _mock_urlopen_reddit(url_or_req, **kw):
    resp = MagicMock()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    resp.read.return_value = _REDDIT_PAYLOAD
    return resp


def test_fetch_reddit_returns_belief_certificates() -> None:
    provider = ExternalSignalProvider()
    with patch("urllib.request.urlopen", _mock_urlopen_reddit):
        beliefs = provider.fetch_reddit()
    assert len(beliefs) > 0, "fetch_reddit() returned no beliefs"
    for b in beliefs:
        assert isinstance(b, BeliefCertificate)


def test_fetch_reddit_beliefs_domain() -> None:
    provider = ExternalSignalProvider()
    with patch("urllib.request.urlopen", _mock_urlopen_reddit):
        beliefs = provider.fetch_reddit()
    for b in beliefs:
        assert b.domain == "Business Opportunity", (
            f"Expected domain 'Business Opportunity', got {b.domain!r}"
        )


def test_fetch_reddit_beliefs_source_mentions_reddit() -> None:
    provider = ExternalSignalProvider()
    with patch("urllib.request.urlopen", _mock_urlopen_reddit):
        beliefs = provider.fetch_reddit()
    for b in beliefs:
        assert "reddit" in b.source.lower() or "Reddit" in b.source, (
            f"Source should mention Reddit: {b.source!r}"
        )


def test_fetch_reddit_beliefs_are_valid() -> None:
    provider = ExternalSignalProvider()
    with patch("urllib.request.urlopen", _mock_urlopen_reddit):
        beliefs = provider.fetch_reddit()
    for b in beliefs:
        assert b.is_valid(), (
            f"Belief not valid (conf={b.confidence}, proof={b.executable_proof!r}): {b.claim!r}"
        )


def test_fetch_reddit_claims_mention_content() -> None:
    provider = ExternalSignalProvider()
    with patch("urllib.request.urlopen", _mock_urlopen_reddit):
        beliefs = provider.fetch_reddit()
    claims = " ".join(b.claim for b in beliefs).lower()
    assert "newsletter" in claims or "niche" in claims or "entrepreneur" in claims, (
        "Reddit claims should reflect the post titles"
    )


# ---------------------------------------------------------------------------
# fetch_reddit() — error handling
# ---------------------------------------------------------------------------

def test_fetch_reddit_returns_empty_on_url_error() -> None:
    provider = ExternalSignalProvider()
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timeout")):
        beliefs = provider.fetch_reddit()
    assert beliefs == [], "fetch_reddit() should return [] on network error"


def test_fetch_reddit_returns_empty_on_bad_json() -> None:
    provider = ExternalSignalProvider()

    def bad_json(url_or_req, **kw):
        resp = MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        resp.read.return_value = b"not json at all {{{{"
        return resp

    with patch("urllib.request.urlopen", bad_json):
        beliefs = provider.fetch_reddit()
    assert beliefs == [], "fetch_reddit() should return [] on bad JSON"


def test_fetch_reddit_never_raises() -> None:
    provider = ExternalSignalProvider()
    with patch("urllib.request.urlopen", side_effect=Exception("unexpected")):
        try:
            beliefs = provider.fetch_reddit()
            assert isinstance(beliefs, list)
        except Exception as exc:
            pytest.fail(f"fetch_reddit() raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# fetch_trends() — happy path with mocked HTTP
# ---------------------------------------------------------------------------

_TRENDS_RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:ht="https://trends.google.com/trends/trendingsearches/daily">
  <channel>
    <title>Google Trends</title>
    <item>
      <title>AI productivity tools</title>
      <ht:approx_traffic>100000+</ht:approx_traffic>
    </item>
    <item>
      <title>dropshipping suppliers 2024</title>
      <ht:approx_traffic>50000+</ht:approx_traffic>
    </item>
  </channel>
</rss>"""


def _mock_urlopen_trends(url_or_req, **kw):
    resp = MagicMock()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    resp.read.return_value = _TRENDS_RSS
    return resp


def test_fetch_trends_returns_belief_certificates() -> None:
    provider = ExternalSignalProvider()
    with patch("urllib.request.urlopen", _mock_urlopen_trends):
        beliefs = provider.fetch_trends()
    assert len(beliefs) > 0, "fetch_trends() returned no beliefs"
    for b in beliefs:
        assert isinstance(b, BeliefCertificate)


def test_fetch_trends_beliefs_domain() -> None:
    provider = ExternalSignalProvider()
    with patch("urllib.request.urlopen", _mock_urlopen_trends):
        beliefs = provider.fetch_trends()
    for b in beliefs:
        assert b.domain == "Market Trends", (
            f"Expected domain 'Market Trends', got {b.domain!r}"
        )


def test_fetch_trends_beliefs_source_mentions_trends() -> None:
    provider = ExternalSignalProvider()
    with patch("urllib.request.urlopen", _mock_urlopen_trends):
        beliefs = provider.fetch_trends()
    for b in beliefs:
        assert "trend" in b.source.lower(), (
            f"Source should mention trends: {b.source!r}"
        )


def test_fetch_trends_beliefs_are_valid() -> None:
    provider = ExternalSignalProvider()
    with patch("urllib.request.urlopen", _mock_urlopen_trends):
        beliefs = provider.fetch_trends()
    for b in beliefs:
        assert b.is_valid(), (
            f"Belief not valid (conf={b.confidence}, proof={b.executable_proof!r}): {b.claim!r}"
        )


def test_fetch_trends_claims_reflect_trending_terms() -> None:
    provider = ExternalSignalProvider()
    with patch("urllib.request.urlopen", _mock_urlopen_trends):
        beliefs = provider.fetch_trends()
    claims = " ".join(b.claim for b in beliefs).lower()
    assert "productivity" in claims or "dropshipping" in claims or "ai" in claims, (
        "Trends claims should reflect the trending terms"
    )


# ---------------------------------------------------------------------------
# fetch_trends() — error handling
# ---------------------------------------------------------------------------

def test_fetch_trends_returns_empty_on_url_error() -> None:
    provider = ExternalSignalProvider()
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timeout")):
        beliefs = provider.fetch_trends()
    assert beliefs == [], "fetch_trends() should return [] on network error"


def test_fetch_trends_returns_empty_on_bad_xml() -> None:
    provider = ExternalSignalProvider()

    def bad_xml(url_or_req, **kw):
        resp = MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        resp.read.return_value = b"<broken xml <<<<"
        return resp

    with patch("urllib.request.urlopen", bad_xml):
        beliefs = provider.fetch_trends()
    assert beliefs == [], "fetch_trends() should return [] on bad XML"


def test_fetch_trends_never_raises() -> None:
    provider = ExternalSignalProvider()
    with patch("urllib.request.urlopen", side_effect=Exception("unexpected")):
        try:
            beliefs = provider.fetch_trends()
            assert isinstance(beliefs, list)
        except Exception as exc:
            pytest.fail(f"fetch_trends() raised unexpectedly: {exc}")

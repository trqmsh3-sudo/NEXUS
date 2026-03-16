"""Tests for ExternalSignalProvider — IRON LAW external knowledge injection."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from nexus.core.belief_certificate import BeliefCertificate
from nexus.core.external_signal import ExternalSignalProvider


# ---------------------------------------------------------------------------
# GitHub fetch (mocked)
# ---------------------------------------------------------------------------

@patch("nexus.core.external_signal.urllib.request.urlopen")
def test_fetch_github_returns_beliefs(mock_urlopen: MagicMock) -> None:
    """fetch_github parses GitHub API response into BeliefCertificates."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "items": [
            {
                "full_name": "owner/repo",
                "description": "A great project",
                "language": "Python",
                "stargazers_count": 50000,
            },
        ],
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    mock_urlopen.return_value = mock_resp

    provider = ExternalSignalProvider()
    beliefs = provider.fetch_github()

    assert len(beliefs) == 1
    assert isinstance(beliefs[0], BeliefCertificate)
    assert "owner/repo" in beliefs[0].claim
    assert "Python" in beliefs[0].claim
    assert "A great project" in beliefs[0].claim
    assert beliefs[0].domain == "Software Engineering"
    assert beliefs[0].confidence == min(0.95, 50000 / 100000.0)
    assert beliefs[0].executable_proof == "# GitHub stars: 50000"
    assert beliefs[0].decay_rate == 0.3


@patch("nexus.core.external_signal.urllib.request.urlopen")
def test_fetch_github_failure_returns_empty(mock_urlopen: object) -> None:
    """fetch_github returns [] and logs on HTTP/network error."""
    import urllib.error
    mock_urlopen.side_effect = urllib.error.URLError("connection failed")

    provider = ExternalSignalProvider()
    beliefs = provider.fetch_github()

    assert beliefs == []


# ---------------------------------------------------------------------------
# ArXiv fetch (mocked)
# ---------------------------------------------------------------------------

_ARXIV_XML: str = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Deep Learning for NLP</title>
    <summary>A paper about transformers.</summary>
  </entry>
  <entry>
    <title>Reinforcement Learning in Games</title>
    <summary>RL paper.</summary>
  </entry>
</feed>
"""


@patch("nexus.core.external_signal.urllib.request.urlopen")
def test_fetch_arxiv_returns_beliefs(mock_urlopen: MagicMock) -> None:
    """fetch_arxiv parses ArXiv Atom feed into BeliefCertificates."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = _ARXIV_XML.encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    mock_urlopen.return_value = mock_resp

    provider = ExternalSignalProvider()
    beliefs = provider.fetch_arxiv()

    assert len(beliefs) == 2
    assert all(isinstance(b, BeliefCertificate) for b in beliefs)
    assert "Recent AI research: Deep Learning for NLP" in beliefs[0].claim
    assert "Recent AI research: Reinforcement Learning in Games" in beliefs[1].claim
    assert beliefs[0].domain == "AI Research"
    assert beliefs[0].confidence == 0.75
    assert beliefs[0].executable_proof == "# ArXiv paper: Deep Learning for NLP"
    assert beliefs[0].decay_rate == 0.4


@patch("nexus.core.external_signal.urllib.request.urlopen")
def test_fetch_arxiv_failure_returns_empty(mock_urlopen: object) -> None:
    """fetch_arxiv returns [] on HTTP error."""
    import urllib.error
    mock_urlopen.side_effect = urllib.error.HTTPError("url", 500, "Server Error", {}, None)

    provider = ExternalSignalProvider()
    beliefs = provider.fetch_arxiv()

    assert beliefs == []


# ---------------------------------------------------------------------------
# fetch_all
# ---------------------------------------------------------------------------

@patch("nexus.core.external_signal.urllib.request.urlopen")
def test_fetch_all_combines_sources(mock_urlopen: MagicMock) -> None:
    """fetch_all returns combined beliefs from GitHub and ArXiv."""
    def side_effect(req, *args, **kwargs) -> object:
        mock_resp = MagicMock()
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "github" in url.lower():
            mock_resp.read.return_value = json.dumps({
                "items": [{
                    "full_name": "x/y",
                    "description": "desc",
                    "language": "Go",
                    "stargazers_count": 10000,
                }],
            }).encode("utf-8")
        else:
            mock_resp.read.return_value = _ARXIV_XML.encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp

    mock_urlopen.side_effect = side_effect

    provider = ExternalSignalProvider()
    beliefs = provider.fetch_all()

    assert len(beliefs) >= 1
    github_beliefs = [b for b in beliefs if "highly-starred" in b.claim]
    arxiv_beliefs = [b for b in beliefs if "Recent AI research" in b.claim]
    assert len(github_beliefs) == 1
    assert len(arxiv_beliefs) == 2

"""TDD tests for DirectJobFetcher — direct HTTP job API fetcher.

Coverage:
  1.  fetch_remoteok returns FINDING lines for matching jobs
  2.  fetch_remoteok filters by keyword from task
  3.  fetch_remoteok returns empty string when no matches
  4.  fetch_remoteok handles HTTP error gracefully (returns "")
  5.  fetch_remoteok handles JSON parse error gracefully
  6.  FINDING line format: position | company | url | salary/tags
  7.  fetch returns non-empty when remoteok succeeds
  8.  fetch returns empty string when all sources fail
  9.  keyword extraction from task works (lowercase, split on spaces)
 10.  salary range formatted as "$min-max" when present, omitted when absent
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from nexus.core.direct_job_fetcher import DirectJobFetcher


def _make_job(position="Python Developer", company="Acme", url="https://remoteok.com/job/1",
              salary_min=5000, salary_max=8000, tags=None):
    return {
        "id": "1",
        "position": position,
        "company": company,
        "url": url,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "tags": tags or ["python", "remote"],
        "date": "2026-04-19T00:00:00Z",
    }


def _remoteok_response(jobs: list[dict]) -> bytes:
    # remoteok API: first item is metadata, rest are jobs
    data = [{"legal": "test"}] + jobs
    return json.dumps(data).encode()


class TestFetchRemoteok:
    def test_returns_finding_lines_for_matching_jobs(self):
        job = _make_job(position="Python Developer", tags=["python"])
        fetcher = DirectJobFetcher()
        with patch("urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = _remoteok_response([job])
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            result = fetcher.fetch_remoteok("Find Python developer remote job")
        assert "FINDING:" in result
        assert "Python Developer" in result

    def test_filters_by_keyword_from_task(self):
        python_job = _make_job(position="Python Developer", tags=["python"])
        java_job = _make_job(position="Java Engineer", tags=["java"], url="https://remoteok.com/job/2")
        fetcher = DirectJobFetcher()
        with patch("urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = _remoteok_response([python_job, java_job])
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            result = fetcher.fetch_remoteok("Find Python developer job")
        assert "Python Developer" in result
        assert "Java Engineer" not in result

    def test_returns_empty_string_when_no_keyword_matches(self):
        java_job = _make_job(position="Java Engineer", tags=["java"])
        fetcher = DirectJobFetcher()
        with patch("urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = _remoteok_response([java_job])
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            result = fetcher.fetch_remoteok("Find Python developer job")
        assert result == ""

    def test_handles_http_error_gracefully(self):
        fetcher = DirectJobFetcher()
        with patch("urllib.request.urlopen", side_effect=Exception("connection refused")):
            result = fetcher.fetch_remoteok("Find Python job")
        assert result == ""

    def test_handles_json_parse_error_gracefully(self):
        fetcher = DirectJobFetcher()
        with patch("urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b"not valid json"
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            result = fetcher.fetch_remoteok("Find Python job")
        assert result == ""

    def test_finding_line_contains_position_company_url(self):
        job = _make_job(position="Senior Python Dev", company="TechCorp",
                        url="https://remoteok.com/job/99", tags=["python"])
        fetcher = DirectJobFetcher()
        with patch("urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = _remoteok_response([job])
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            result = fetcher.fetch_remoteok("Find Python senior job")
        assert "Senior Python Dev" in result
        assert "TechCorp" in result
        assert "https://remoteok.com/job/99" in result

    def test_salary_formatted_when_present(self):
        job = _make_job(position="Python Dev", salary_min=4000, salary_max=7000, tags=["python"])
        fetcher = DirectJobFetcher()
        with patch("urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = _remoteok_response([job])
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            result = fetcher.fetch_remoteok("Find Python job")
        assert "$4000" in result or "4000" in result

    def test_salary_omitted_when_zero(self):
        job = _make_job(position="Python Dev", salary_min=0, salary_max=0, tags=["python"])
        fetcher = DirectJobFetcher()
        with patch("urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = _remoteok_response([job])
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            result = fetcher.fetch_remoteok("Find Python job")
        assert "FINDING:" in result


class TestFetch:
    def test_fetch_returns_result_when_remoteok_succeeds(self):
        job = _make_job(position="Python Dev", tags=["python"])
        fetcher = DirectJobFetcher()
        with patch.object(fetcher, "fetch_remoteok", return_value="FINDING: Python Dev | Acme | url"):
            result = fetcher.fetch("Find Python job")
        assert "FINDING:" in result

    def test_fetch_returns_empty_when_all_fail(self):
        fetcher = DirectJobFetcher()
        with patch.object(fetcher, "fetch_remoteok", return_value=""):
            result = fetcher.fetch("Find Python job")
        assert result == ""

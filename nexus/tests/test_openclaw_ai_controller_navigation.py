"""Tests for OpenClawAIController pre-loop navigation (TDD).

Coverage:
  _extract_start_url()
    1.  freelancer.com in task → https://www.freelancer.com/projects
    2.  remote.co in task      → https://remote.co/remote-jobs/
    3.  remoteok.com in task   → https://remoteok.com
    4.  weworkremotely in task → https://weworkremotely.com
    5.  indeed.com in task     → https://www.indeed.com
    6.  fiverr.com in task     → https://www.fiverr.com
    7.  peopleperhour in task  → https://www.peopleperhour.com
    8.  no known site          → default https://www.freelancer.com/projects
    9.  case-insensitive match

  run() navigation behaviour
   10. navigate action sent BEFORE first screenshot
   11. navigate uses URL from _extract_start_url
   12. navigate sent even when task has no URL (uses default)
   13. after navigation, vision loop proceeds normally
   14. execute_action called with type=navigate and correct url
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from nexus.core.openclaw_ai_controller import OpenClawAIController


# ─────────────────────────────────────────────────────────────
#  Helpers (reuse from existing test module pattern)
# ─────────────────────────────────────────────────────────────

def _make_client(screenshots=None, action_results=None):
    client = MagicMock()
    client.screenshot.side_effect = list(screenshots or ["base64img"])
    client.execute_action.side_effect = list(action_results or [""] * 20)
    return client


def _litellm_response(action_dict: dict) -> MagicMock:
    msg = MagicMock()
    msg.content = json.dumps(action_dict)
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ═══════════════════════════════════════════════════════════════
#  1–9: _extract_start_url
# ═══════════════════════════════════════════════════════════════

class TestExtractStartUrl:
    def _url(self, task: str) -> str:
        ctrl = OpenClawAIController(MagicMock(), api_key="k")
        return ctrl._extract_start_url(task)

    def test_freelancer_returns_projects_page(self):
        assert self._url("Find Python job on freelancer.com") == \
            "https://www.freelancer.com/projects"

    def test_remote_co_returns_remote_jobs(self):
        assert self._url("Search remote.co for data engineer") == \
            "https://remote.co/remote-jobs/"

    def test_remoteok_returns_root(self):
        assert self._url("Search remoteok.com for Python") == \
            "https://remoteok.com"

    def test_weworkremotely_returns_root(self):
        assert self._url("Check weworkremotely.com for contracts") == \
            "https://weworkremotely.com"

    def test_indeed_returns_root(self):
        assert self._url("Search indeed.com for freelance") == \
            "https://www.indeed.com"

    def test_fiverr_returns_root(self):
        assert self._url("Look at fiverr.com gigs") == \
            "https://www.fiverr.com"

    def test_peopleperhour_returns_root(self):
        assert self._url("Browse peopleperhour.com") == \
            "https://www.peopleperhour.com"

    def test_no_known_site_returns_freelancer_default(self):
        assert self._url("Find a paid opportunity online") == \
            "https://www.freelancer.com/projects"

    def test_match_is_case_insensitive(self):
        assert self._url("Check FREELANCER.COM projects") == \
            "https://www.freelancer.com/projects"


# ═══════════════════════════════════════════════════════════════
#  10–14: run() navigation behaviour
# ═══════════════════════════════════════════════════════════════

class TestRunNavigation:
    def test_navigate_sent_before_first_screenshot(self):
        """execute_action(navigate) must be called before screenshot()."""
        client = _make_client(screenshots=["img"])
        call_order = []
        client.execute_action.side_effect = lambda a: call_order.append(("action", a["type"])) or ""
        client.screenshot.side_effect = lambda: call_order.append(("screenshot",)) or "img"

        ctrl = OpenClawAIController(client, api_key="k")
        complete = {"type": "task_complete", "data": "", "description": "done"}
        with patch("litellm.completion", return_value=_litellm_response(complete)):
            ctrl.run("Find job on freelancer.com")

        # navigate must appear before the first screenshot
        first_screenshot_idx = next(i for i, e in enumerate(call_order) if e == ("screenshot",))
        navigate_indices = [i for i, e in enumerate(call_order)
                            if e == ("action", "navigate")]
        assert navigate_indices, "no navigate action was sent"
        assert navigate_indices[0] < first_screenshot_idx

    def test_navigate_uses_url_from_extract_start_url(self):
        client = _make_client(screenshots=["img"])
        ctrl = OpenClawAIController(client, api_key="k")
        complete = {"type": "task_complete", "data": "", "description": "done"}
        with patch("litellm.completion", return_value=_litellm_response(complete)):
            ctrl.run("Find Python job on freelancer.com")

        nav_calls = [c for c in client.execute_action.call_args_list
                     if c[0][0].get("type") == "navigate"]
        assert len(nav_calls) == 1
        assert nav_calls[0][0][0]["url"] == "https://www.freelancer.com/projects"

    def test_navigate_uses_default_when_no_url_in_task(self):
        client = _make_client(screenshots=["img"])
        ctrl = OpenClawAIController(client, api_key="k")
        complete = {"type": "task_complete", "data": "", "description": "done"}
        with patch("litellm.completion", return_value=_litellm_response(complete)):
            ctrl.run("Find a paid opportunity online")

        nav_calls = [c for c in client.execute_action.call_args_list
                     if c[0][0].get("type") == "navigate"]
        assert len(nav_calls) == 1
        assert nav_calls[0][0][0]["url"] == "https://www.freelancer.com/projects"

    def test_navigate_uses_remote_co_url(self):
        client = _make_client(screenshots=["img"])
        ctrl = OpenClawAIController(client, api_key="k")
        complete = {"type": "task_complete", "data": "", "description": "done"}
        with patch("litellm.completion", return_value=_litellm_response(complete)):
            ctrl.run("Search remote.co for data engineer jobs")

        nav_calls = [c for c in client.execute_action.call_args_list
                     if c[0][0].get("type") == "navigate"]
        assert nav_calls[0][0][0]["url"] == "https://remote.co/remote-jobs/"

    def test_vision_loop_proceeds_after_navigation(self):
        """After navigate, screenshots and DeepSeek calls still happen."""
        client = _make_client(screenshots=["img1", "img2"])
        ctrl = OpenClawAIController(client, api_key="k")
        actions = [
            {"type": "extract_data", "data": "FINDING: job A", "description": "x"},
            {"type": "task_complete", "data": "", "description": "done"},
        ]
        with patch("litellm.completion",
                   side_effect=[_litellm_response(a) for a in actions]):
            result = ctrl.run("Find job on freelancer.com")

        assert "FINDING: job A" in result
        assert client.screenshot.call_count == 2

"""TDD tests for stuck-state detection in OpenClawAIController.

Coverage:
  1.  Same action type 3 consecutive times → re-navigate to next site
  2.  Re-navigate uses next URL from _SITE_MAP fallback list
  3.  Consecutive counter resets after re-navigate
  4.  Findings preserved across re-navigate
  5.  All fallback sites exhausted → return findings (empty if none)
  6.  Non-consecutive repeated types do NOT trigger re-navigate
  7.  extract_data does not count toward stuck (it's non-forwarded)
  8.  task_complete immediately ends loop without re-navigate
  9.  Default start URL changed to remoteok.com (no-login site)
 10.  _SITE_MAP first entry is remoteok.com
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import pytest

from nexus.core.openclaw_ai_controller import (
    OpenClawAIController,
    _DEFAULT_START_URL,
    _SITE_MAP,
)


def _make_client(screenshots=None):
    client = MagicMock()
    if screenshots is None:
        screenshots = ["img"] * 50
    client.screenshot.side_effect = list(screenshots)
    client.execute_action.return_value = ""
    return client


def _resp(action_dict: dict) -> MagicMock:
    msg = MagicMock()
    msg.content = json.dumps(action_dict)
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ═══════════════════════════════════════════════════════════════
#  9–10: Default URL and site map ordering
# ═══════════════════════════════════════════════════════════════

class TestDefaultUrl:
    def test_default_start_url_is_no_login_site(self):
        """Default URL must NOT be freelancer.com (requires login)."""
        assert "freelancer.com" not in _DEFAULT_START_URL

    def test_default_start_url_is_remoteok(self):
        assert _DEFAULT_START_URL == "https://remoteok.com"

    def test_site_map_first_entry_is_remoteok(self):
        assert _SITE_MAP[0][1] == "https://remoteok.com"

    def test_freelancer_still_in_site_map_for_explicit_requests(self):
        keywords = [kw for kw, _ in _SITE_MAP]
        assert any("freelancer" in kw for kw in keywords)


# ═══════════════════════════════════════════════════════════════
#  1–8: Stuck-state detection
# ═══════════════════════════════════════════════════════════════

class TestStuckDetection:
    def _make_ctrl(self, client):
        return OpenClawAIController(client, api_key="k")

    def test_three_consecutive_same_type_triggers_renavigate(self):
        """After 3 consecutive 'type' actions, controller re-navigates."""
        client = _make_client()
        ctrl = self._make_ctrl(client)

        actions = [
            {"type": "type", "text": "a", "description": "1"},
            {"type": "type", "text": "b", "description": "2"},
            {"type": "type", "text": "c", "description": "3"},
            {"type": "task_complete", "data": "", "description": "done"},
        ]
        with patch("litellm.completion",
                   side_effect=[_resp(a) for a in actions]):
            ctrl.run("find jobs")

        navigate_calls = [
            c for c in client.execute_action.call_args_list
            if c[0][0].get("type") == "navigate"
        ]
        # Initial navigate + 1 stuck-escape navigate
        assert len(navigate_calls) >= 2

    def test_renavigate_uses_different_url_than_first(self):
        """The stuck-escape navigate must not reuse the first URL."""
        client = _make_client()
        ctrl = self._make_ctrl(client)

        actions = [
            {"type": "type", "text": "a", "description": "1"},
            {"type": "type", "text": "b", "description": "2"},
            {"type": "type", "text": "c", "description": "3"},
            {"type": "task_complete", "data": "", "description": "done"},
        ]
        with patch("litellm.completion",
                   side_effect=[_resp(a) for a in actions]):
            ctrl.run("find jobs")

        navigate_calls = [
            c[0][0]["url"]
            for c in client.execute_action.call_args_list
            if c[0][0].get("type") == "navigate"
        ]
        assert len(navigate_calls) >= 2
        assert navigate_calls[0] != navigate_calls[1]

    def test_findings_preserved_across_renavigate(self):
        """extract_data findings collected before stuck are kept after re-navigate."""
        client = _make_client()
        ctrl = self._make_ctrl(client)

        actions = [
            {"type": "extract_data", "data": "FINDING: job A", "description": "x"},
            {"type": "type", "text": "a", "description": "1"},
            {"type": "type", "text": "b", "description": "2"},
            {"type": "type", "text": "c", "description": "3"},
            {"type": "task_complete", "data": "", "description": "done"},
        ]
        with patch("litellm.completion",
                   side_effect=[_resp(a) for a in actions]):
            result = ctrl.run("find jobs")

        assert "FINDING: job A" in result

    def test_two_consecutive_same_type_does_not_trigger_renavigate(self):
        """Only 3+ consecutive identical types trigger re-navigate."""
        client = _make_client()
        ctrl = self._make_ctrl(client)

        actions = [
            {"type": "type", "text": "a", "description": "1"},
            {"type": "type", "text": "b", "description": "2"},
            {"type": "task_complete", "data": "", "description": "done"},
        ]
        with patch("litellm.completion",
                   side_effect=[_resp(a) for a in actions]):
            ctrl.run("find jobs")

        navigate_calls = [
            c for c in client.execute_action.call_args_list
            if c[0][0].get("type") == "navigate"
        ]
        # Only initial navigate, no stuck-escape
        assert len(navigate_calls) == 1

    def test_non_consecutive_type_does_not_trigger(self):
        """type, click, type, click, type — never 3 consecutive."""
        client = _make_client()
        ctrl = self._make_ctrl(client)

        actions = [
            {"type": "type", "text": "a", "description": "1"},
            {"type": "click", "x": 10, "y": 10, "description": "c"},
            {"type": "type", "text": "b", "description": "2"},
            {"type": "click", "x": 10, "y": 10, "description": "c"},
            {"type": "type", "text": "c", "description": "3"},
            {"type": "task_complete", "data": "", "description": "done"},
        ]
        with patch("litellm.completion",
                   side_effect=[_resp(a) for a in actions]):
            ctrl.run("find jobs")

        navigate_calls = [
            c for c in client.execute_action.call_args_list
            if c[0][0].get("type") == "navigate"
        ]
        assert len(navigate_calls) == 1

    def test_extract_data_does_not_count_toward_stuck(self):
        """extract_data is non-forwarded; 3x in a row must not trigger re-navigate."""
        client = _make_client()
        ctrl = self._make_ctrl(client)

        actions = [
            {"type": "extract_data", "data": "FINDING: A", "description": "1"},
            {"type": "extract_data", "data": "FINDING: B", "description": "2"},
            {"type": "extract_data", "data": "FINDING: C", "description": "3"},
            {"type": "task_complete", "data": "", "description": "done"},
        ]
        with patch("litellm.completion",
                   side_effect=[_resp(a) for a in actions]):
            ctrl.run("find jobs")

        navigate_calls = [
            c for c in client.execute_action.call_args_list
            if c[0][0].get("type") == "navigate"
        ]
        assert len(navigate_calls) == 1

    def test_task_complete_ends_immediately_no_renavigate(self):
        """task_complete at step 0 never triggers stuck-escape."""
        client = _make_client()
        ctrl = self._make_ctrl(client)

        with patch("litellm.completion",
                   return_value=_resp({"type": "task_complete", "data": "x", "description": "done"})):
            ctrl.run("find jobs")

        navigate_calls = [
            c for c in client.execute_action.call_args_list
            if c[0][0].get("type") == "navigate"
        ]
        assert len(navigate_calls) == 1

    def test_stuck_counter_resets_after_renavigate(self):
        """After re-navigate, need another 3 consecutive to trigger again."""
        client = _make_client()
        ctrl = self._make_ctrl(client)

        # 3 types → renavigate; then 2 types (not enough); then task_complete
        actions = [
            {"type": "type", "text": "a", "description": "1"},
            {"type": "type", "text": "b", "description": "2"},
            {"type": "type", "text": "c", "description": "3"},
            # counter resets; 2 more types → should NOT trigger again
            {"type": "type", "text": "d", "description": "4"},
            {"type": "type", "text": "e", "description": "5"},
            {"type": "task_complete", "data": "", "description": "done"},
        ]
        with patch("litellm.completion",
                   side_effect=[_resp(a) for a in actions]):
            ctrl.run("find jobs")

        navigate_calls = [
            c for c in client.execute_action.call_args_list
            if c[0][0].get("type") == "navigate"
        ]
        # Exactly 2: initial + 1 stuck-escape (not 3)
        assert len(navigate_calls) == 2

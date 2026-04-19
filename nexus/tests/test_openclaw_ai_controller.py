"""Tests for OpenClawAIController — vision-based browser control loop.

Written before implementation (TDD).

Coverage:
  1. Loop exits cleanly when screenshot fails
  2. Loop completes immediately on task_complete action
  3. extract_data findings accumulate across steps
  4. click / type / scroll actions forwarded to OpenClaw
  5. MAX_STEPS hard cap respected
  6. DeepSeek API failure stops loop gracefully
  7. Malformed JSON from DeepSeek stops loop gracefully
  8. Markdown code fences stripped from DeepSeek response
  9. Screenshot sent as base64 image_url to DeepSeek; api_key forwarded
 10. Findings-so-far context passed to DeepSeek on subsequent steps
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import pytest

from nexus.core.openclaw_ai_controller import OpenClawAIController


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def _make_client(screenshots=None, action_results=None):
    """Mock OpenClawClient with pre-canned screenshot / action_results."""
    client = MagicMock()
    client.screenshot.side_effect = list(screenshots or ["base64imgdata"])
    client.execute_action.side_effect = list(action_results or [""] * 20)
    return client


def _litellm_response(action_dict: dict) -> MagicMock:
    """Build a mock litellm completion response containing the given action JSON."""
    msg = MagicMock()
    msg.content = json.dumps(action_dict)
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ═══════════════════════════════════════════════════════════════
#  1. Screenshot failure
# ═══════════════════════════════════════════════════════════════

class TestScreenshotFailure:
    def test_empty_screenshot_stops_loop_immediately(self):
        client = _make_client(screenshots=[""])
        ctrl = OpenClawAIController(client, api_key="test-key")
        result = ctrl.run("find jobs")
        assert result == ""

    def test_no_deepseek_call_when_screenshot_fails(self):
        client = _make_client(screenshots=[""])
        ctrl = OpenClawAIController(client, api_key="test-key")
        with patch("litellm.completion") as mock_llm:
            ctrl.run("find jobs")
        mock_llm.assert_not_called()

    def test_no_execute_action_when_screenshot_fails(self):
        client = _make_client(screenshots=[""])
        ctrl = OpenClawAIController(client, api_key="test-key")
        ctrl.run("find jobs")
        client.execute_action.assert_not_called()


# ═══════════════════════════════════════════════════════════════
#  2. task_complete action
# ═══════════════════════════════════════════════════════════════

class TestTaskComplete:
    def test_returns_data_field_on_task_complete(self):
        client = _make_client(screenshots=["img"])
        ctrl = OpenClawAIController(client, api_key="test-key")
        action = {"type": "task_complete", "data": "summary of results", "description": "done"}
        with patch("litellm.completion", return_value=_litellm_response(action)):
            result = ctrl.run("find jobs")
        assert result == "summary of results"

    def test_no_execute_action_on_task_complete(self):
        client = _make_client(screenshots=["img"])
        ctrl = OpenClawAIController(client, api_key="test-key")
        action = {"type": "task_complete", "data": "done", "description": "finished"}
        with patch("litellm.completion", return_value=_litellm_response(action)):
            ctrl.run("find jobs")
        client.execute_action.assert_not_called()

    def test_empty_data_field_returns_empty_string(self):
        client = _make_client(screenshots=["img"])
        ctrl = OpenClawAIController(client, api_key="test-key")
        action = {"type": "task_complete", "data": "", "description": "done"}
        with patch("litellm.completion", return_value=_litellm_response(action)):
            result = ctrl.run("find jobs")
        assert result == ""


# ═══════════════════════════════════════════════════════════════
#  3. extract_data accumulation
# ═══════════════════════════════════════════════════════════════

class TestExtractData:
    def test_single_extraction_returned(self):
        client = _make_client(screenshots=["img1", "img2"])
        ctrl = OpenClawAIController(client, api_key="test-key")
        actions = [
            {"type": "extract_data", "data": "FINDING: Job A | a.com | $100", "description": "extract"},
            {"type": "task_complete", "data": "", "description": "done"},
        ]
        with patch("litellm.completion", side_effect=[_litellm_response(a) for a in actions]):
            result = ctrl.run("find jobs")
        assert "FINDING: Job A | a.com | $100" in result

    def test_multiple_extractions_joined_by_newline(self):
        client = _make_client(screenshots=["img1", "img2", "img3"])
        ctrl = OpenClawAIController(client, api_key="test-key")
        actions = [
            {"type": "extract_data", "data": "FINDING: Job A | a.com | $100", "description": "extract"},
            {"type": "extract_data", "data": "FINDING: Job B | b.com | $200", "description": "extract"},
            {"type": "task_complete", "data": "", "description": "done"},
        ]
        with patch("litellm.completion", side_effect=[_litellm_response(a) for a in actions]):
            result = ctrl.run("find jobs")
        lines = result.splitlines()
        assert len(lines) == 2
        assert "Job A" in lines[0]
        assert "Job B" in lines[1]

    def test_extract_data_action_not_forwarded_to_execute_action(self):
        client = _make_client(screenshots=["img1", "img2"])
        ctrl = OpenClawAIController(client, api_key="test-key")
        actions = [
            {"type": "extract_data", "data": "FINDING: something", "description": "extract"},
            {"type": "task_complete", "data": "", "description": "done"},
        ]
        with patch("litellm.completion", side_effect=[_litellm_response(a) for a in actions]):
            ctrl.run("find jobs")
        client.execute_action.assert_not_called()


# ═══════════════════════════════════════════════════════════════
#  4. click / type / scroll actions forwarded to OpenClaw
# ═══════════════════════════════════════════════════════════════

class TestActionExecution:
    def test_click_action_forwarded_with_coordinates(self):
        client = _make_client(screenshots=["img1", "img2"])
        ctrl = OpenClawAIController(client, api_key="test-key")
        actions = [
            {"type": "click", "x": 150, "y": 300, "description": "click submit"},
            {"type": "task_complete", "data": "", "description": "done"},
        ]
        with patch("litellm.completion", side_effect=[_litellm_response(a) for a in actions]):
            ctrl.run("submit form")
        client.execute_action.assert_called_once_with(
            {"type": "click", "x": 150, "y": 300, "description": "click submit"}
        )

    def test_type_action_forwarded_with_text(self):
        client = _make_client(screenshots=["img1", "img2"])
        ctrl = OpenClawAIController(client, api_key="test-key")
        actions = [
            {"type": "type", "text": "python developer", "description": "type search"},
            {"type": "task_complete", "data": "", "description": "done"},
        ]
        with patch("litellm.completion", side_effect=[_litellm_response(a) for a in actions]):
            ctrl.run("search for jobs")
        dispatched = client.execute_action.call_args[0][0]
        assert dispatched["type"] == "type"
        assert dispatched["text"] == "python developer"

    def test_scroll_action_forwarded(self):
        client = _make_client(screenshots=["img1", "img2"])
        ctrl = OpenClawAIController(client, api_key="test-key")
        actions = [
            {"type": "scroll", "direction": "down", "amount": 500, "description": "scroll"},
            {"type": "task_complete", "data": "", "description": "done"},
        ]
        with patch("litellm.completion", side_effect=[_litellm_response(a) for a in actions]):
            ctrl.run("find jobs")
        dispatched = client.execute_action.call_args[0][0]
        assert dispatched["type"] == "scroll"
        assert dispatched["direction"] == "down"

    def test_multiple_actions_executed_in_order(self):
        client = _make_client(screenshots=["img1", "img2", "img3"])
        ctrl = OpenClawAIController(client, api_key="test-key")
        actions = [
            {"type": "click", "x": 10, "y": 20, "description": "click search"},
            {"type": "type", "text": "python", "description": "type query"},
            {"type": "task_complete", "data": "", "description": "done"},
        ]
        with patch("litellm.completion", side_effect=[_litellm_response(a) for a in actions]):
            ctrl.run("search")
        assert client.execute_action.call_count == 2
        first_call = client.execute_action.call_args_list[0][0][0]
        second_call = client.execute_action.call_args_list[1][0][0]
        assert first_call["type"] == "click"
        assert second_call["type"] == "type"


# ═══════════════════════════════════════════════════════════════
#  5. MAX_STEPS cap
# ═══════════════════════════════════════════════════════════════

class TestMaxSteps:
    def test_loop_stops_at_max_steps(self):
        client = _make_client(screenshots=["img"] * 20, action_results=[""] * 20)
        ctrl = OpenClawAIController(client, api_key="test-key")
        ctrl.MAX_STEPS = 4
        scroll = {"type": "scroll", "direction": "down", "amount": 300, "description": "scroll"}
        with patch("litellm.completion", return_value=_litellm_response(scroll)):
            result = ctrl.run("find jobs")
        assert client.screenshot.call_count == 4
        assert result == ""

    def test_default_max_steps_is_15(self):
        client = _make_client()
        ctrl = OpenClawAIController(client, api_key="test-key")
        assert ctrl.MAX_STEPS == 15


# ═══════════════════════════════════════════════════════════════
#  6. DeepSeek failure
# ═══════════════════════════════════════════════════════════════

class TestDeepSeekFailure:
    def test_api_exception_stops_loop(self):
        client = _make_client(screenshots=["img"])
        ctrl = OpenClawAIController(client, api_key="test-key")
        with patch("litellm.completion", side_effect=Exception("API error")):
            result = ctrl.run("find jobs")
        assert result == ""
        client.execute_action.assert_not_called()

    def test_api_exception_does_not_propagate(self):
        client = _make_client(screenshots=["img"])
        ctrl = OpenClawAIController(client, api_key="test-key")
        with patch("litellm.completion", side_effect=RuntimeError("boom")):
            # must not raise
            ctrl.run("find jobs")


# ═══════════════════════════════════════════════════════════════
#  7. Malformed JSON from DeepSeek
# ═══════════════════════════════════════════════════════════════

class TestMalformedJson:
    def _bad_response(self, text: str) -> MagicMock:
        msg = MagicMock()
        msg.content = text
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def test_non_json_text_stops_loop(self):
        client = _make_client(screenshots=["img"])
        ctrl = OpenClawAIController(client, api_key="test-key")
        with patch("litellm.completion", return_value=self._bad_response("sorry, I cannot help")):
            result = ctrl.run("find jobs")
        assert result == ""

    def test_json_without_type_field_stops_loop(self):
        client = _make_client(screenshots=["img"])
        ctrl = OpenClawAIController(client, api_key="test-key")
        with patch("litellm.completion", return_value=self._bad_response('{"action": "click"}')):
            result = ctrl.run("find jobs")
        assert result == ""

    def test_json_array_instead_of_object_stops_loop(self):
        client = _make_client(screenshots=["img"])
        ctrl = OpenClawAIController(client, api_key="test-key")
        with patch("litellm.completion", return_value=self._bad_response('[1, 2, 3]')):
            result = ctrl.run("find jobs")
        assert result == ""


# ═══════════════════════════════════════════════════════════════
#  8. Markdown fence stripping
# ═══════════════════════════════════════════════════════════════

class TestMarkdownFenceStripping:
    def _fenced_response(self, action_dict: dict, lang: str = "json") -> MagicMock:
        msg = MagicMock()
        msg.content = f"```{lang}\n{json.dumps(action_dict)}\n```"
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def test_json_fence_stripped_and_parsed(self):
        client = _make_client(screenshots=["img"])
        ctrl = OpenClawAIController(client, api_key="test-key")
        action = {"type": "task_complete", "data": "done", "description": "finished"}
        with patch("litellm.completion", return_value=self._fenced_response(action, "json")):
            result = ctrl.run("find jobs")
        assert result == "done"

    def test_plain_fence_stripped_and_parsed(self):
        client = _make_client(screenshots=["img"])
        ctrl = OpenClawAIController(client, api_key="test-key")
        action = {"type": "task_complete", "data": "done", "description": "finished"}
        with patch("litellm.completion", return_value=self._fenced_response(action, "")):
            result = ctrl.run("find jobs")
        assert result == "done"


# ═══════════════════════════════════════════════════════════════
#  9. DeepSeek API call format
# ═══════════════════════════════════════════════════════════════

class TestDeepSeekCallFormat:
    def test_screenshot_passed_as_base64_image_url(self):
        client = _make_client(screenshots=["mybase64screenshot"])
        ctrl = OpenClawAIController(client, api_key="sk-test")
        action = {"type": "task_complete", "data": "", "description": "done"}
        with patch("litellm.completion", return_value=_litellm_response(action)) as mock_llm:
            ctrl.run("find jobs")
        messages = mock_llm.call_args[1]["messages"]
        user_msg = messages[-1]
        image_parts = [p for p in user_msg["content"] if p.get("type") == "image_url"]
        assert len(image_parts) == 1
        assert "mybase64screenshot" in image_parts[0]["image_url"]["url"]

    def test_api_key_forwarded_to_litellm(self):
        client = _make_client(screenshots=["img"])
        ctrl = OpenClawAIController(client, api_key="sk-my-deepseek-key")
        action = {"type": "task_complete", "data": "", "description": "done"}
        with patch("litellm.completion", return_value=_litellm_response(action)) as mock_llm:
            ctrl.run("find jobs")
        assert mock_llm.call_args[1]["api_key"] == "sk-my-deepseek-key"

    def test_task_text_included_in_user_message(self):
        client = _make_client(screenshots=["img"])
        ctrl = OpenClawAIController(client, api_key="test-key")
        action = {"type": "task_complete", "data": "", "description": "done"}
        with patch("litellm.completion", return_value=_litellm_response(action)) as mock_llm:
            ctrl.run("find Python freelance gigs")
        messages = mock_llm.call_args[1]["messages"]
        user_msg = messages[-1]
        text_parts = [p for p in user_msg["content"] if p.get("type") == "text"]
        full_text = " ".join(p["text"] for p in text_parts)
        assert "find Python freelance gigs" in full_text

    def test_default_model_is_deepseek_chat(self):
        client = _make_client(screenshots=["img"])
        ctrl = OpenClawAIController(client, api_key="test-key")
        action = {"type": "task_complete", "data": "", "description": "done"}
        with patch("litellm.completion", return_value=_litellm_response(action)) as mock_llm:
            ctrl.run("task")
        assert mock_llm.call_args[1]["model"] == "deepseek/deepseek-chat"

    def test_custom_model_forwarded(self):
        client = _make_client(screenshots=["img"])
        ctrl = OpenClawAIController(client, api_key="test-key", model="gemini/gemini-2.0-flash")
        action = {"type": "task_complete", "data": "", "description": "done"}
        with patch("litellm.completion", return_value=_litellm_response(action)) as mock_llm:
            ctrl.run("task")
        assert mock_llm.call_args[1]["model"] == "gemini/gemini-2.0-flash"


# ═══════════════════════════════════════════════════════════════
#  10. Findings context passed on subsequent steps
# ═══════════════════════════════════════════════════════════════

class TestFindingsContext:
    def test_prior_findings_included_in_second_step(self):
        client = _make_client(screenshots=["img1", "img2"])
        ctrl = OpenClawAIController(client, api_key="test-key")
        actions = [
            {"type": "extract_data", "data": "FINDING: Job A | a.com | $100", "description": "extract"},
            {"type": "task_complete", "data": "", "description": "done"},
        ]
        captured_messages: list = []
        original_responses = [_litellm_response(a) for a in actions]

        def capture_and_return(*args, **kwargs):
            captured_messages.append(kwargs.get("messages", []))
            return original_responses.pop(0)

        with patch("litellm.completion", side_effect=capture_and_return):
            ctrl.run("find jobs")

        # Second call should include the first finding in context
        assert len(captured_messages) == 2
        second_call_text = str(captured_messages[1])
        assert "Job A" in second_call_text

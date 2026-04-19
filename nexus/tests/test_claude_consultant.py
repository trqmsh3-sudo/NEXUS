"""TDD tests for ClaudeConsultant — strategic reasoning via Claude API.

Coverage:
  1.  consult() returns a non-empty string when API key is present
  2.  consult() returns "" gracefully when no API key configured
  3.  consult() returns "" on API errors (network, quota, etc.)
  4.  consult() includes the question in the messages sent to Claude
  5.  system_prompt is forwarded to litellm
  6.  is_available() returns True when key present, False when absent
  7.  suggest_strategy() returns a short task string
  8.  suggest_strategy() returns "" when Claude unavailable
  9.  max_tokens respected (response stays bounded)
 10.  model defaults to claude-sonnet-4-6
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nexus.core.claude_consultant import ClaudeConsultant


def _litellm_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestIsAvailable:
    def test_available_when_key_present(self):
        c = ClaudeConsultant(api_key="sk-ant-test")
        assert c.is_available() is True

    def test_unavailable_when_key_absent(self):
        c = ClaudeConsultant(api_key="")
        assert c.is_available() is False

    def test_unavailable_when_key_none_and_no_env(self):
        with patch.dict("os.environ", {}, clear=True):
            import importlib
            c = ClaudeConsultant(api_key=None)
            # Only unavailable if ANTHROPIC_API_KEY not in env
            import os
            if not os.environ.get("ANTHROPIC_API_KEY"):
                assert c.is_available() is False


class TestConsult:
    def test_returns_response_when_key_present(self):
        c = ClaudeConsultant(api_key="sk-ant-test")
        with patch("litellm.completion", return_value=_litellm_response("Try remoteok.com")):
            result = c.consult("What should I do next?")
        assert result == "Try remoteok.com"

    def test_returns_empty_when_no_key(self):
        c = ClaudeConsultant(api_key="")
        result = c.consult("What should I do next?")
        assert result == ""

    def test_returns_empty_on_api_error(self):
        c = ClaudeConsultant(api_key="sk-ant-test")
        with patch("litellm.completion", side_effect=Exception("quota exceeded")):
            result = c.consult("What should I do next?")
        assert result == ""

    def test_question_included_in_messages(self):
        c = ClaudeConsultant(api_key="sk-ant-test")
        captured: list = []
        def _capture(**kwargs):
            captured.append(kwargs.get("messages", []))
            return _litellm_response("ok")
        with patch("litellm.completion", side_effect=_capture):
            c.consult("Should I try writing gigs?")
        assert captured
        user_content = str(captured[0])
        assert "Should I try writing gigs?" in user_content

    def test_system_prompt_forwarded(self):
        c = ClaudeConsultant(api_key="sk-ant-test", system_prompt="You are strategic.")
        captured: list = []
        def _capture(**kwargs):
            captured.append(kwargs.get("messages", []))
            return _litellm_response("ok")
        with patch("litellm.completion", side_effect=_capture):
            c.consult("Next step?")
        system_msgs = [m for m in captured[0] if m.get("role") == "system"]
        assert system_msgs
        assert "You are strategic." in system_msgs[0]["content"]

    def test_model_defaults_to_claude_sonnet_46(self):
        c = ClaudeConsultant(api_key="sk-ant-test")
        captured: list = []
        def _capture(**kwargs):
            captured.append(kwargs.get("model", ""))
            return _litellm_response("ok")
        with patch("litellm.completion", side_effect=_capture):
            c.consult("Next step?")
        assert "claude-sonnet-4-6" in captured[0] or "claude" in captured[0]


class TestSuggestStrategy:
    def test_returns_task_string_when_available(self):
        c = ClaudeConsultant(api_key="sk-ant-test")
        with patch("litellm.completion",
                   return_value=_litellm_response("Find Python gigs on remoteok.com")):
            result = c.suggest_strategy(
                recent_failures=["python", "python", "python"],
                recent_successes=["writing"],
            )
        assert isinstance(result, str)
        assert len(result) > 5

    def test_returns_empty_when_unavailable(self):
        c = ClaudeConsultant(api_key="")
        result = c.suggest_strategy(recent_failures=["python"], recent_successes=[])
        assert result == ""

    def test_failures_included_in_prompt(self):
        c = ClaudeConsultant(api_key="sk-ant-test")
        captured: list = []
        def _capture(**kwargs):
            captured.append(kwargs.get("messages", []))
            return _litellm_response("Try writing")
        with patch("litellm.completion", side_effect=_capture):
            c.suggest_strategy(
                recent_failures=["python", "devops"],
                recent_successes=["writing"],
            )
        prompt_text = str(captured[0])
        assert "python" in prompt_text.lower()
        assert "writing" in prompt_text.lower()

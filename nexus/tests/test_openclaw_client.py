"""Tests for OpenClawClient — thin urllib wrapper for OpenClaw gateway.

Written before implementation (TDD).

Coverage:
  1. Construction & defaults
  2. is_available() — socket probe, graceful on all failures
  3. send() — POST to /v1/chat/completions, vault token, graceful failures
  4. GUARDIAN scanner detects OPENCLAW_TOKEN patterns
"""

from __future__ import annotations

import json
import socket
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch, call

import pytest

from nexus.core.openclaw_client import OpenClawClient


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def _mock_response(body: dict | str, status: int = 200) -> MagicMock:
    """Return a mock that behaves like urllib's http.client.HTTPResponse."""
    if isinstance(body, dict):
        raw = json.dumps(body).encode("utf-8")
    else:
        raw = body.encode("utf-8") if isinstance(body, str) else body
    mock = MagicMock()
    mock.read.return_value = raw
    mock.status = status
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    return mock


def _chat_response(content: str) -> dict:
    return {
        "choices": [{"message": {"content": content}}]
    }


# ═══════════════════════════════════════════════════════════════
#  1. Construction & Defaults
# ═══════════════════════════════════════════════════════════════

class TestOpenClawClientConstruction:
    def test_can_import(self):
        assert OpenClawClient is not None

    def test_default_base_url(self):
        client = OpenClawClient()
        assert "127.0.0.1" in client.base_url
        assert "18789" in client.base_url

    def test_custom_base_url(self):
        client = OpenClawClient(base_url="http://10.0.0.1:9999")
        assert client.base_url == "http://10.0.0.1:9999"

    def test_vault_is_none_by_default(self):
        client = OpenClawClient()
        assert client.vault is None

    def test_accepts_vault(self, tmp_path):
        from nexus.core.guardian import GuardianVault
        v = GuardianVault(str(tmp_path / "v.enc"), master_key="test-master")
        client = OpenClawClient(vault=v)
        assert client.vault is v

    def test_trailing_slash_stripped_from_base_url(self):
        client = OpenClawClient(base_url="http://127.0.0.1:18789/")
        assert not client.base_url.endswith("/")


# ═══════════════════════════════════════════════════════════════
#  2. is_available()
# ═══════════════════════════════════════════════════════════════

class TestIsAvailable:
    def test_returns_true_when_socket_connects(self):
        client = OpenClawClient()
        mock_conn = MagicMock()
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)
        with patch("socket.create_connection", return_value=mock_conn) as mock_sock:
            result = client.is_available()
        assert result is True
        mock_sock.assert_called_once()

    def test_returns_false_on_connection_refused(self):
        client = OpenClawClient()
        with patch("socket.create_connection", side_effect=OSError("Connection refused")):
            result = client.is_available()
        assert result is False

    def test_returns_false_on_timeout(self):
        client = OpenClawClient()
        with patch("socket.create_connection", side_effect=socket.timeout("timed out")):
            result = client.is_available()
        assert result is False

    def test_never_raises(self):
        client = OpenClawClient()
        with patch("socket.create_connection", side_effect=Exception("unexpected")):
            result = client.is_available()
        assert result is False

    def test_uses_correct_host_and_port(self):
        client = OpenClawClient(base_url="http://192.168.1.5:9000")
        mock_conn = MagicMock()
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)
        with patch("socket.create_connection", return_value=mock_conn) as mock_sock:
            client.is_available(timeout=5)
        args, kwargs = mock_sock.call_args
        assert args[0] == ("192.168.1.5", 9000)
        assert args[1] == 5 or kwargs.get("timeout") == 5


# ═══════════════════════════════════════════════════════════════
#  3. send()
# ═══════════════════════════════════════════════════════════════

class TestSend:
    def test_returns_content_on_success(self):
        client = OpenClawClient()
        resp = _mock_response(_chat_response("Found: $120/hr gigs on Upwork"))
        with patch("urllib.request.urlopen", return_value=resp):
            result = client.send("find freelance rates")
        assert result == "Found: $120/hr gigs on Upwork"

    def test_strips_whitespace_from_content(self):
        client = OpenClawClient()
        resp = _mock_response(_chat_response("  result with spaces  \n"))
        with patch("urllib.request.urlopen", return_value=resp):
            result = client.send("task")
        assert result == "result with spaces"

    def test_returns_empty_on_connection_refused(self):
        client = OpenClawClient()
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            result = client.send("task")
        assert result == ""

    def test_returns_empty_on_timeout(self):
        client = OpenClawClient()
        with patch(
            "urllib.request.urlopen",
            side_effect=TimeoutError("timed out"),
        ):
            result = client.send("task")
        assert result == ""

    def test_returns_empty_on_bad_json(self):
        client = OpenClawClient()
        resp = _mock_response("not valid json{{{")
        with patch("urllib.request.urlopen", return_value=resp):
            result = client.send("task")
        assert result == ""

    def test_returns_empty_on_missing_choices_key(self):
        client = OpenClawClient()
        resp = _mock_response({"error": "model not found"})
        with patch("urllib.request.urlopen", return_value=resp):
            result = client.send("task")
        assert result == ""

    def test_never_raises_on_any_exception(self):
        client = OpenClawClient()
        with patch("urllib.request.urlopen", side_effect=RuntimeError("boom")):
            result = client.send("task")
        assert result == ""

    def test_uses_vault_token_in_authorization_header(self, tmp_path):
        from nexus.core.guardian import GuardianVault
        v = GuardianVault(str(tmp_path / "v.enc"), master_key="test-master")
        v.set("OPENCLAW_TOKEN", "my-secret-gateway-token")
        client = OpenClawClient(vault=v)

        captured_headers: list[dict] = []

        def capture_request(req, timeout=None):
            captured_headers.append(dict(req.headers))
            return _mock_response(_chat_response("ok"))

        with patch("urllib.request.urlopen", side_effect=capture_request):
            client.send("task")

        assert any(
            "my-secret-gateway-token" in str(v)
            for v in captured_headers[0].values()
        )

    def test_sends_without_auth_when_no_vault(self):
        """Sending without a vault must not raise — auth header simply absent."""
        client = OpenClawClient()
        captured: list = []

        def capture(req, timeout=None):
            captured.append(req)
            return _mock_response(_chat_response("ok"))

        with patch("urllib.request.urlopen", side_effect=capture):
            result = client.send("task")

        assert result == "ok"
        assert "Authorization" not in captured[0].headers

    def test_sends_without_auth_when_token_not_in_vault(self, tmp_path):
        from nexus.core.guardian import GuardianVault
        v = GuardianVault(str(tmp_path / "v.enc"), master_key="test-master")
        # vault present but OPENCLAW_TOKEN not stored
        client = OpenClawClient(vault=v)
        captured: list = []

        def capture(req, timeout=None):
            captured.append(req)
            return _mock_response(_chat_response("ok"))

        with patch("urllib.request.urlopen", side_effect=capture):
            result = client.send("task")

        assert result == "ok"
        assert "Authorization" not in captured[0].headers

    def test_posts_to_chat_completions_endpoint(self):
        client = OpenClawClient(base_url="http://127.0.0.1:18789")
        captured: list = []

        def capture(req, timeout=None):
            captured.append(req)
            return _mock_response(_chat_response("ok"))

        with patch("urllib.request.urlopen", side_effect=capture):
            client.send("task")

        assert captured[0].full_url == "http://127.0.0.1:18789/v1/chat/completions"
        assert captured[0].method == "POST"

    def test_request_body_uses_openclaw_model(self):
        client = OpenClawClient()
        captured: list = []

        def capture(req, timeout=None):
            captured.append(req)
            return _mock_response(_chat_response("ok"))

        with patch("urllib.request.urlopen", side_effect=capture):
            client.send("my task text")

        body = json.loads(captured[0].data.decode("utf-8"))
        assert body["model"] == "openclaw/default"
        assert body["messages"][0]["role"] == "user"
        assert body["messages"][0]["content"] == "my task text"


# ═══════════════════════════════════════════════════════════════
#  4. GUARDIAN scanner detects openclaw token patterns
# ═══════════════════════════════════════════════════════════════

class TestGuardianOpenclaw:
    def test_scanner_detects_openclaw_bearer_token(self):
        from nexus.core.guardian import SecretScanner
        scanner = SecretScanner()
        findings = scanner.scan_string(
            'OPENCLAW_TOKEN = "oc-tok-abcdef1234567890abcdef1234567890"'
        )
        assert len(findings) > 0

    def test_scanner_pattern_name_is_openclaw_token(self):
        from nexus.core.guardian import SecretScanner
        scanner = SecretScanner()
        findings = scanner.scan_string(
            'config.openclaw_token = "oc-tok-abcdef1234567890abcdef1234567890"'
        )
        pattern_names = {f.pattern_name for f in findings}
        assert "openclaw_token" in pattern_names

    def test_openclaw_token_severity_is_critical(self):
        from nexus.core.guardian import SecretScanner
        scanner = SecretScanner()
        findings = scanner.scan_string(
            'OPENCLAW_TOKEN="oc-tok-abcdef1234567890abcdef1234567890"'
        )
        critical = [f for f in findings if f.pattern_name == "openclaw_token"]
        assert all(f.severity == "CRITICAL" for f in critical)


# ═══════════════════════════════════════════════════════════════
#  5. screenshot()
# ═══════════════════════════════════════════════════════════════

class TestScreenshot:
    def test_returns_base64_image_from_gateway(self):
        client = OpenClawClient()
        resp = _mock_response({"image": "abc123base64=="})
        with patch("urllib.request.urlopen", return_value=resp):
            result = client.screenshot()
        assert result == "abc123base64=="

    def test_returns_empty_string_on_network_error(self):
        client = OpenClawClient()
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            result = client.screenshot()
        assert result == ""

    def test_returns_empty_string_on_bad_json(self):
        client = OpenClawClient()
        resp = _mock_response("not json")
        with patch("urllib.request.urlopen", return_value=resp):
            result = client.screenshot()
        assert result == ""

    def test_returns_empty_when_image_key_missing(self):
        client = OpenClawClient()
        resp = _mock_response({"status": "ok"})
        with patch("urllib.request.urlopen", return_value=resp):
            result = client.screenshot()
        assert result == ""

    def test_requests_correct_endpoint(self):
        client = OpenClawClient(base_url="http://127.0.0.1:18789")
        captured = []
        def capture(req, timeout=None):
            captured.append(req)
            return _mock_response({"image": "data"})
        with patch("urllib.request.urlopen", side_effect=capture):
            client.screenshot()
        assert captured[0].full_url == "http://127.0.0.1:18789/screenshot"

    def test_never_raises(self):
        client = OpenClawClient()
        with patch("urllib.request.urlopen", side_effect=RuntimeError("boom")):
            result = client.screenshot()
        assert result == ""


# ═══════════════════════════════════════════════════════════════
#  6. execute_action()
# ═══════════════════════════════════════════════════════════════

class TestExecuteAction:
    def test_returns_result_from_gateway(self):
        client = OpenClawClient()
        resp = _mock_response({"result": "clicked successfully"})
        with patch("urllib.request.urlopen", return_value=resp):
            result = client.execute_action({"type": "click", "x": 50, "y": 100})
        assert result == "clicked successfully"

    def test_sends_action_as_json_body(self):
        client = OpenClawClient()
        captured = []
        def capture(req, timeout=None):
            captured.append(req)
            return _mock_response({"result": "ok"})
        with patch("urllib.request.urlopen", side_effect=capture):
            client.execute_action({"type": "type", "text": "hello"})
        body = json.loads(captured[0].data.decode("utf-8"))
        assert body["type"] == "type"
        assert body["text"] == "hello"

    def test_posts_to_action_endpoint(self):
        client = OpenClawClient(base_url="http://127.0.0.1:18789")
        captured = []
        def capture(req, timeout=None):
            captured.append(req)
            return _mock_response({"result": "ok"})
        with patch("urllib.request.urlopen", side_effect=capture):
            client.execute_action({"type": "scroll", "direction": "down"})
        assert captured[0].full_url == "http://127.0.0.1:18789/action"
        assert captured[0].method == "POST"

    def test_returns_empty_string_on_network_error(self):
        client = OpenClawClient()
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            result = client.execute_action({"type": "click", "x": 0, "y": 0})
        assert result == ""

    def test_returns_empty_string_when_result_key_missing(self):
        client = OpenClawClient()
        resp = _mock_response({"status": "ok"})
        with patch("urllib.request.urlopen", return_value=resp):
            result = client.execute_action({"type": "click", "x": 0, "y": 0})
        assert result == ""

    def test_never_raises(self):
        client = OpenClawClient()
        with patch("urllib.request.urlopen", side_effect=RuntimeError("boom")):
            result = client.execute_action({"type": "click", "x": 0, "y": 0})
        assert result == ""

    def test_vault_token_included_in_auth_header(self, tmp_path):
        from nexus.core.guardian import GuardianVault
        v = GuardianVault(str(tmp_path / "v.enc"), master_key="test-master")
        v.set("OPENCLAW_TOKEN", "gw-token-xyz")
        client = OpenClawClient(vault=v)
        captured = []
        def capture(req, timeout=None):
            captured.append(req)
            return _mock_response({"result": "ok"})
        with patch("urllib.request.urlopen", side_effect=capture):
            client.execute_action({"type": "click", "x": 0, "y": 0})
        assert any("gw-token-xyz" in str(v) for v in captured[0].headers.values())

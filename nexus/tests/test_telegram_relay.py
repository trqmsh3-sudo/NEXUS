"""Tests for TelegramRelay — human-in-the-loop SMS code forwarding.

Architecture:
  1. Browser automation (OpenClaw) hits an SMS verification screen.
  2. It outputs: WAITING_FOR_SMS: <site>
  3. HouseC detects that signal and calls TelegramRelay.request_sms_code(site).
  4. TelegramRelay sends: "SMS code needed for <site>. Reply with the code."
  5. User receives the real SMS, reads it, replies in Telegram.
  6. TelegramRelay returns the code string.
  7. HouseC sends the code back to OpenClaw to complete verification.

Coverage:
  Module and class exist
  Constructor stores token, chat_id, timeout
  send_message() posts to Telegram sendMessage API
  request_sms_code() sends the formatted message
  request_sms_code() polls getUpdates for a reply
  request_sms_code() returns the code on success
  request_sms_code() returns None on timeout (no reply)
  request_sms_code() ignores updates from before the request
  Timeout defaults to 300 seconds (5 minutes)
  Message format: "SMS code needed for [site]. Reply with the code."
  poll_for_reply() uses offset to skip stale updates
  from_env() class method builds instance from env vars
  from_env() returns None when credentials are missing
  Vault credentials list includes OWNER_PHONE
  Vault credentials list includes PROXY_GOOGLE_EMAIL / PROXY_GOOGLE_PASS
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, call, patch

import pytest

import scripts.vault_store_credentials as vault_script
from nexus.core.telegram_relay import TelegramRelay


# ── helpers ────────────────────────────────────────────────────────────────────

TOKEN = "123:ABC"
CHAT  = "9999"


def _relay(timeout: int = 300) -> TelegramRelay:
    return TelegramRelay(bot_token=TOKEN, chat_id=CHAT, timeout=timeout)


def _updates(*texts: str, base_id: int = 10) -> bytes:
    """Fake getUpdates response with one message per text."""
    results = [
        {
            "update_id": base_id + i,
            "message": {
                "message_id": 100 + i,
                "chat": {"id": int(CHAT)},
                "text": t,
                "date": int(time.time()),
            },
        }
        for i, t in enumerate(texts)
    ]
    return json.dumps({"ok": True, "result": results}).encode()


def _empty_updates() -> bytes:
    return json.dumps({"ok": True, "result": []}).encode()


# ══════════════════════════════════════════════════════════════════
#  Module and class
# ══════════════════════════════════════════════════════════════════

class TestTelegramRelayExists:
    def test_module_importable(self):
        import nexus.core.telegram_relay  # noqa: F401

    def test_class_exists(self):
        assert TelegramRelay is not None

    def test_constructor_stores_token(self):
        r = _relay()
        assert r.bot_token == TOKEN

    def test_constructor_stores_chat_id(self):
        r = _relay()
        assert r.chat_id == CHAT

    def test_default_timeout_is_300(self):
        r = TelegramRelay(bot_token=TOKEN, chat_id=CHAT)
        assert r.timeout == 300

    def test_custom_timeout_stored(self):
        r = TelegramRelay(bot_token=TOKEN, chat_id=CHAT, timeout=60)
        assert r.timeout == 60


# ══════════════════════════════════════════════════════════════════
#  send_message()
# ══════════════════════════════════════════════════════════════════

class TestSendMessage:
    def test_send_message_calls_telegram_api(self):
        r = _relay()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps({"ok": True, "result": {}}).encode()

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            r.send_message("hello")

        assert mock_open.called
        req = mock_open.call_args[0][0]
        assert f"bot{TOKEN}/sendMessage" in req.full_url

    def test_send_message_includes_chat_id(self):
        r = _relay()
        sent_data = {}
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps({"ok": True, "result": {}}).encode()

        def capture(req, **kw):
            sent_data["body"] = json.loads(req.data.decode())
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=capture):
            r.send_message("test message")

        assert sent_data["body"]["chat_id"] == CHAT

    def test_send_message_includes_text(self):
        r = _relay()
        sent_data = {}
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps({"ok": True, "result": {}}).encode()

        def capture(req, **kw):
            sent_data["body"] = json.loads(req.data.decode())
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=capture):
            r.send_message("my test text")

        assert sent_data["body"]["text"] == "my test text"


# ══════════════════════════════════════════════════════════════════
#  request_sms_code() — message format
# ══════════════════════════════════════════════════════════════════

class TestRequestSmsCodeMessage:
    def _run(self, site: str) -> tuple[list[str], str | None]:
        """Call request_sms_code and capture sent messages."""
        r = _relay(timeout=1)
        sent = []

        def fake_open(req, **kw):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            mock_resp = MagicMock()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            if "sendMessage" in url:
                body = json.loads(req.data.decode())
                sent.append(body["text"])
                mock_resp.read.return_value = json.dumps({"ok": True, "result": {}}).encode()
            else:
                mock_resp.read.return_value = _empty_updates()
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_open):
            result = r.request_sms_code(site)

        return sent, result

    def test_message_mentions_site(self):
        sent, _ = self._run("reddit.com")
        assert sent, "No message was sent"
        assert "reddit.com" in sent[0].lower()

    def test_message_says_sms_code_needed(self):
        sent, _ = self._run("reddit.com")
        assert "sms code" in sent[0].lower() or "verification" in sent[0].lower()

    def test_message_asks_to_reply(self):
        sent, _ = self._run("reddit.com")
        assert "reply" in sent[0].lower()

    def test_message_exact_format(self):
        sent, _ = self._run("reddit.com")
        assert "SMS code needed for reddit.com" in sent[0] or \
               "sms code needed for reddit.com" in sent[0].lower()


# ══════════════════════════════════════════════════════════════════
#  request_sms_code() — polling and return value
# ══════════════════════════════════════════════════════════════════

class TestRequestSmsCodePolling:
    def test_returns_code_when_reply_received(self):
        r = _relay(timeout=10)
        call_count = 0

        def fake_open(req, **kw):
            nonlocal call_count
            mock_resp = MagicMock()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "sendMessage" in url:
                mock_resp.read.return_value = json.dumps({"ok": True, "result": {}}).encode()
            else:
                call_count += 1
                if call_count == 1:
                    mock_resp.read.return_value = _empty_updates()
                else:
                    mock_resp.read.return_value = _updates("123456")
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_open):
            result = r.request_sms_code("reddit.com")

        assert result == "123456"

    def test_returns_none_on_timeout(self):
        r = TelegramRelay(bot_token=TOKEN, chat_id=CHAT, timeout=0)

        def fake_open(req, **kw):
            mock_resp = MagicMock()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "sendMessage" in url:
                mock_resp.read.return_value = json.dumps({"ok": True, "result": {}}).encode()
            else:
                mock_resp.read.return_value = _empty_updates()
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_open):
            result = r.request_sms_code("reddit.com")

        assert result is None

    def test_uses_offset_to_skip_old_updates(self):
        """After seeing update_id N, next poll must use offset=N+1."""
        r = _relay(timeout=10)
        polled_offsets = []

        def fake_open(req, **kw):
            mock_resp = MagicMock()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "sendMessage" in url:
                mock_resp.read.return_value = json.dumps({"ok": True, "result": {}}).encode()
            elif "getUpdates" in url:
                # Parse offset from URL or body
                import urllib.parse
                parsed = urllib.parse.urlparse(url)
                params = urllib.parse.parse_qs(parsed.query)
                offset = int(params.get("offset", [0])[0])
                polled_offsets.append(offset)
                if len(polled_offsets) == 1:
                    # First poll: return update_id=42
                    mock_resp.read.return_value = _updates("wait", base_id=42)
                else:
                    mock_resp.read.return_value = _updates("654321", base_id=43)
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_open):
            r.request_sms_code("test.com")

        assert len(polled_offsets) >= 2
        assert polled_offsets[1] == 43, (
            f"After update_id=42, offset must be 43, got {polled_offsets[1]}"
        )

    def test_trims_whitespace_from_code(self):
        r = _relay(timeout=10)

        def fake_open(req, **kw):
            mock_resp = MagicMock()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "sendMessage" in url:
                mock_resp.read.return_value = json.dumps({"ok": True, "result": {}}).encode()
            else:
                mock_resp.read.return_value = _updates("  789012  ")
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_open):
            result = r.request_sms_code("site.com")

        assert result == "789012"


# ══════════════════════════════════════════════════════════════════
#  from_env() factory
# ══════════════════════════════════════════════════════════════════

class TestFromEnv:
    def test_from_env_returns_instance_when_vars_set(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok123")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
        r = TelegramRelay.from_env()
        assert r is not None
        assert r.bot_token == "tok123"
        assert r.chat_id == "42"

    def test_from_env_returns_none_when_token_missing(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
        assert TelegramRelay.from_env() is None

    def test_from_env_returns_none_when_chat_id_missing(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok123")
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        assert TelegramRelay.from_env() is None


# ══════════════════════════════════════════════════════════════════
#  Vault credentials
# ══════════════════════════════════════════════════════════════════

class TestVaultCredentials:
    def _keys(self) -> list[str]:
        return [k for k, _ in vault_script.CREDENTIALS]

    def test_owner_phone_in_credentials(self):
        assert "OWNER_PHONE" in self._keys()

    def test_proxy_google_email_in_credentials(self):
        assert "PROXY_GOOGLE_EMAIL" in self._keys()

    def test_proxy_google_pass_in_credentials(self):
        assert "PROXY_GOOGLE_PASS" in self._keys()

    def test_telegram_bot_token_in_credentials(self):
        assert "TELEGRAM_BOT_TOKEN" in self._keys()

    def test_telegram_chat_id_in_credentials(self):
        assert "TELEGRAM_CHAT_ID" in self._keys()

    def test_all_new_credentials_have_labels(self):
        labels = {k: v for k, v in vault_script.CREDENTIALS}
        for key in ("OWNER_PHONE", "PROXY_GOOGLE_EMAIL", "PROXY_GOOGLE_PASS"):
            assert labels.get(key), f"{key} must have a non-empty label"

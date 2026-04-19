"""OpenClawClient — thin urllib wrapper for the OpenClaw Gateway.

Sends browser tasks to a locally-running OpenClaw Gateway via its
OpenAI-compatible HTTP endpoint.  Uses only the Python standard library.

The client never raises — all failures are logged and an empty string
is returned from send().  is_available() probes the gateway port via a
raw socket connection so it works regardless of which HTTP endpoints the
gateway has enabled.
"""

from __future__ import annotations

import json
import logging
import socket
import urllib.error
import urllib.parse
import urllib.request
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL  = "http://127.0.0.1:18789"
_COMPLETIONS_PATH  = "/v1/chat/completions"
_SCREENSHOT_PATH   = "/screenshot"
_ACTION_PATH       = "/action"
_DEFAULT_MODEL     = "openclaw/default"


class OpenClawClient:
    """Sends natural-language browser tasks to an OpenClaw Gateway.

    Usage::

        client = OpenClawClient(vault=guardian.vault)
        if client.is_available():
            findings = client.send("navigate to Upwork, list top Python gigs")

    Args:
        vault:    Optional :class:`~nexus.core.guardian.GuardianVault`.
                  If present and contains ``OPENCLAW_TOKEN``, that value
                  is sent as the ``Authorization: Bearer`` header.
        base_url: Gateway base URL.  Defaults to ``http://127.0.0.1:18789``.
    """

    def __init__(
        self,
        vault=None,                      # GuardianVault | None
        base_url: str = _DEFAULT_BASE_URL,
        token: str | None = None,        # direct token — takes precedence over vault
    ) -> None:
        self.vault    = vault
        self.base_url = base_url.rstrip("/")
        self._token   = token            # explicit token overrides vault

    # ──────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────

    def is_available(self, timeout: int = 3) -> bool:
        """Return True if the gateway is reachable on its port.

        Uses a raw socket probe — does not depend on any specific HTTP
        endpoint being enabled in the gateway config.
        """
        parsed = urlparse(self.base_url)
        host   = parsed.hostname or "127.0.0.1"
        port   = parsed.port or 18789
        try:
            with socket.create_connection((host, port), timeout):
                return True
        except Exception as exc:
            logger.debug("OpenClawClient: gateway not reachable — %s", exc)
            return False

    def send(self, task: str, timeout: int = 30) -> str:
        """Send *task* to the gateway as a chat-completion request.

        Returns the assistant's response text, stripped of whitespace.
        Returns an empty string on any failure — never raises.

        Args:
            task:    Natural-language instruction for the browser agent.
            timeout: HTTP request timeout in seconds.
        """
        payload = json.dumps({
            "model": _DEFAULT_MODEL,
            "messages": [{"role": "user", "content": task}],
        }).encode("utf-8")

        headers: dict[str, str] = {"Content-Type": "application/json"}
        token = self._get_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        url = self.base_url + _COMPLETIONS_PATH
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
            data    = json.loads(body)
            content = data["choices"][0]["message"]["content"]
            return content.strip()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning("OpenClawClient: request failed — %s", exc)
            return ""
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            logger.warning("OpenClawClient: unexpected response format — %s", exc)
            return ""
        except Exception as exc:
            logger.warning("OpenClawClient: unexpected error — %s", exc)
            return ""

    def screenshot(self, timeout: int = 10) -> str:
        """Request a base64-encoded PNG screenshot from the gateway.

        Returns the base64 string, or an empty string on any failure.
        """
        headers: dict[str, str] = {}
        token = self._get_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        url = self.base_url + _SCREENSHOT_PATH
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
            data = json.loads(body)
            return data.get("image", "")
        except Exception as exc:
            logger.warning("OpenClawClient.screenshot: failed — %s", exc)
            return ""

    def execute_action(self, action: dict, timeout: int = 15) -> str:
        """Send a structured action to the gateway and return any result text.

        Args:
            action:  Dict with at minimum a ``type`` key
                     (e.g. ``{"type": "click", "x": 100, "y": 200}``).
            timeout: HTTP request timeout in seconds.

        Returns an empty string on any failure — never raises.
        """
        payload = json.dumps(action).encode("utf-8")
        headers: dict[str, str] = {"Content-Type": "application/json"}
        token = self._get_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        url = self.base_url + _ACTION_PATH
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
            data = json.loads(body)
            return data.get("result", "")
        except Exception as exc:
            logger.warning("OpenClawClient.execute_action: failed — %s", exc)
            return ""

    # ──────────────────────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────────────────────

    def _get_token(self) -> str | None:
        """Return the gateway token: explicit > vault > None."""
        if self._token:
            return self._token
        if self.vault is not None and self.vault.has("OPENCLAW_TOKEN"):
            return self.vault.get("OPENCLAW_TOKEN")
        return None

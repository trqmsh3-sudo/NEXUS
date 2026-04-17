"""TelegramRelay — human-in-the-loop SMS code forwarding via Telegram Bot API.

When browser automation hits an SMS verification screen it cannot complete
on its own, PROXY calls TelegramRelay.request_sms_code(site).  The relay:

  1. Sends a Telegram message to the owner:
       "SMS code needed for <site>. Reply with the code."
  2. Polls getUpdates for up to `timeout` seconds (default 300 = 5 minutes).
  3. Returns the first reply text (stripped) as the verification code.
  4. Returns None on timeout so the caller can decide how to handle it.

The human receives the real SMS on their phone, reads it, and replies in
Telegram — PROXY never intercepts SMS directly.  All control stays with
the owner.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.parse
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org"


class TelegramRelay:
    """Send Telegram notifications and wait for human replies.

    Attributes:
        bot_token: Telegram Bot API token.
        chat_id:   The owner's Telegram chat ID to message.
        timeout:   Maximum seconds to wait for a reply (default 300).
    """

    def __init__(self, bot_token: str, chat_id: str, timeout: int = 300) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_message(self, text: str) -> bool:
        """Send a text message to the owner's chat.

        Args:
            text: Message body.

        Returns:
            True on success, False on any error.
        """
        url = f"{_TELEGRAM_API}/bot{self.bot_token}/sendMessage"
        payload = json.dumps({"chat_id": self.chat_id, "text": text}).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                ok = data.get("ok", False)
                if ok:
                    logger.info("TELEGRAM sent message ok")
                else:
                    logger.warning("TELEGRAM send failed: %s", data)
                return ok
        except Exception as exc:
            logger.warning("TELEGRAM send error: %s", exc)
            return False

    def request_sms_code(self, site: str) -> Optional[str]:
        """Notify owner that SMS code is needed and wait for their reply.

        Sends: "SMS code needed for <site>. Reply with the code."
        Then polls getUpdates until a reply arrives or timeout expires.

        Args:
            site: The site/service that is requesting verification
                  (e.g. "reddit.com").

        Returns:
            The code string (stripped) if the owner replied in time,
            or None on timeout.
        """
        msg = f"SMS code needed for {site}. Reply with the code."
        self.send_message(msg)
        logger.info("TELEGRAM waiting for SMS code for %s (timeout=%ds)", site, self.timeout)

        # Establish a starting offset so we only see replies sent AFTER
        # our message, not old messages already in the bot's queue.
        offset = self._get_next_offset()

        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            updates, offset = self._poll(offset)
            for text in updates:
                code = text.strip()
                if code:
                    logger.info("TELEGRAM code received for %s", site)
                    return code
            # Brief pause to avoid hammering the API when timeout is short
            if time.monotonic() < deadline:
                time.sleep(min(1, max(0, deadline - time.monotonic())))

        logger.warning("TELEGRAM timeout waiting for SMS code for %s", site)
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_next_offset(self) -> int:
        """Return offset = max_seen_update_id + 1 so we skip stale updates."""
        _, next_offset = self._poll(offset=0, timeout=0)
        return next_offset

    def _poll(self, offset: int, timeout: int = 5) -> tuple[list[str], int]:
        """Call getUpdates and return (reply_texts, new_offset).

        Only messages from the configured chat_id are returned.
        """
        url = (
            f"{_TELEGRAM_API}/bot{self.bot_token}/getUpdates"
            f"?timeout={timeout}&offset={offset}"
        )
        req = urllib.request.Request(url)
        try:
            with urllib.request.urlopen(req, timeout=timeout + 5) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:
            logger.debug("TELEGRAM poll error: %s", exc)
            return [], offset

        results = data.get("result", [])
        texts: list[str] = []
        new_offset = offset
        for update in results:
            uid = update.get("update_id", offset)
            new_offset = max(new_offset, uid + 1)
            msg = update.get("message", {})
            chat_id = str(msg.get("chat", {}).get("id", ""))
            text = msg.get("text", "")
            if chat_id == self.chat_id and text:
                texts.append(text)

        return texts, new_offset

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> Optional["TelegramRelay"]:
        """Build a TelegramRelay from environment variables.

        Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (both populated
        from the GUARDIAN vault via .env).

        Returns:
            A TelegramRelay instance, or None if either variable is absent.
        """
        token   = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        if not token or not chat_id:
            logger.debug("TelegramRelay.from_env: missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
            return None
        return cls(bot_token=token, chat_id=chat_id)

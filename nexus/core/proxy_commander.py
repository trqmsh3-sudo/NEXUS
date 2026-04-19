"""ProxyCommander — Telegram command interface for PROXY.

Connects PROXY to its owner via Telegram Bot API:
  1. Polls for incoming messages from the owner chat.
  2. Executes each message as a HouseOmega task.
  3. Replies with the result.
  4. Sends a daily morning report at 08:00 Israel time.

Credentials are read from GuardianVault first, env vars as fallback.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from nexus.core.telegram_relay import TelegramRelay

if TYPE_CHECKING:
    from nexus.core.house_omega import HouseOmega

logger = logging.getLogger(__name__)

_ISRAEL = ZoneInfo("Asia/Jerusalem")
_DAILY_REPORT_HOUR = 8        # 8 AM Israel time
_DAILY_REPORT_WINDOW_MIN = 10  # send window: 08:00–08:09
_POLL_INTERVAL_SECONDS = 5.0


class ProxyCommander:
    """Telegram interface for PROXY owner control.

    Attributes:
        relay:  TelegramRelay for sending/receiving messages.
        omega:  HouseOmega instance to run tasks against.
    """

    def __init__(self, relay: TelegramRelay, omega: "HouseOmega") -> None:
        self.relay = relay
        self.omega = omega
        self._last_report_date: date | None = None

    # ──────────────────────────────────────────────────────────
    #  Factory methods
    # ──────────────────────────────────────────────────────────

    @classmethod
    def from_vault(
        cls,
        vault_path: str,
        master_key: str,
        omega: "HouseOmega",
    ) -> "ProxyCommander":
        """Build ProxyCommander reading credentials from GuardianVault.

        Raises:
            KeyError: if TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not in vault.
            ValueError: if vault cannot be decrypted (wrong master key).
        """
        from nexus.core.guardian import GuardianVault
        vault = GuardianVault(vault_path, master_key=master_key)
        token = vault.get("TELEGRAM_BOT_TOKEN")
        chat_id = vault.get("TELEGRAM_CHAT_ID")
        relay = TelegramRelay(bot_token=token, chat_id=chat_id)
        return cls(relay=relay, omega=omega)

    @classmethod
    def from_env(cls, omega: "HouseOmega") -> "ProxyCommander | None":
        """Build ProxyCommander from environment variables.

        Returns:
            ProxyCommander instance, or None if either variable is absent.
        """
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        if not token or not chat_id:
            logger.debug("ProxyCommander.from_env: missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
            return None
        relay = TelegramRelay(bot_token=token, chat_id=chat_id)
        return cls(relay=relay, omega=omega)

    # ──────────────────────────────────────────────────────────
    #  Message handling
    # ──────────────────────────────────────────────────────────

    def handle_message(self, text: str) -> str:
        """Execute an owner command and return a reply string.

        Special commands (/start, /help) are handled directly.
        Everything else is run through HouseOmega.
        """
        stripped = text.strip()

        if stripped == "/start":
            return (
                "PROXY is online.\n\n"
                "Send me any task and I will execute it:\n"
                "  • Find paid gigs on freelancer.com\n"
                "  • Search remote.co for Python contracts\n\n"
                "Commands: /help /status /google_login"
            )

        if stripped == "/help":
            return (
                "PROXY command interface\n\n"
                "Send any natural-language task to run a full cycle.\n"
                "Examples:\n"
                "  • Find one paid Python contract on freelancer.com\n"
                "  • Search remote.co for data engineer jobs\n\n"
                "Commands:\n"
                "  /status       — recent cycle summary\n"
                "  /google_login — open accounts.google.com and wait for manual login\n\n"
                "I will report results when the cycle completes."
            )

        if stripped == "/status":
            history = self.omega.get_cycle_history(last_n=10)
            total = len(history)
            successes = sum(1 for c in history if c.success)
            return f"PROXY status: {successes}/{total} recent cycles succeeded."

        if stripped == "/google_login":
            return self._handle_google_login()

        logger.info("ProxyCommander: running task %r", stripped[:80])
        result = self.omega.run(stripped)

        if result.success:
            problem = ""
            if result.sso:
                problem = result.sso.redefined_problem[:100]
            belief_note = " Belief stored." if result.belief_added else ""
            return (
                f"✓ Cycle complete ({result.cycle_time_seconds:.0f}s).\n"
                f"Task: {problem}\n"
                f"Belief added: {'yes' if result.belief_added else 'no'}.{belief_note}"
            )
        else:
            reason = result.failure_reason or "unknown error"
            return f"✗ Cycle failed: {reason}"

    # ──────────────────────────────────────────────────────────
    #  Google login via OpenClaw
    # ──────────────────────────────────────────────────────────

    def _handle_google_login(self) -> str:
        """Open accounts.google.com via OpenClaw and wait for manual login.

        Sends an immediate Telegram prompt asking the user to log in, then
        blocks (up to 5 minutes) until OpenClaw reports a successful login.
        """
        from nexus.core.openclaw_client import OpenClawClient

        client = OpenClawClient()
        if not client.is_available():
            return (
                "OpenClaw gateway is not reachable. "
                "Start the browser agent and try again."
            )

        self.relay.send_message(
            "Opening accounts.google.com in browser.\n"
            "Please log in to Google account 2 manually.\n"
            "I will notify you here when login is detected."
        )
        logger.info("ProxyCommander: waiting for Google login via OpenClaw")

        result = client.send(
            "Navigate to https://accounts.google.com. "
            "Do not fill in any credentials — wait for the user to log in manually. "
            "Once the browser shows a Google account dashboard or myaccount.google.com "
            "(indicating a successful login), respond with exactly: LOGIN_DETECTED. "
            "Otherwise respond with: WAITING.",
            timeout=300,
        )

        if "LOGIN_DETECTED" in result.upper():
            logger.info("ProxyCommander: Google login detected")
            return "Google account 2 login detected. Browser is now authenticated."
        elif result:
            return f"OpenClaw response: {result}"
        else:
            return "No response from OpenClaw (timed out or gateway error)."

    # ──────────────────────────────────────────────────────────
    #  Daily report
    # ──────────────────────────────────────────────────────────

    def _daily_report_text(self) -> str:
        """Build the daily 8am morning report string."""
        cycles = self.omega.get_cycle_history(last_n=200)
        total_cycles = len(cycles)
        opportunities = sum(1 for c in cycles if c.belief_added)
        actions_taken = sum(
            1 for c in cycles
            if c.build_result is not None or (c.success and c.belief_added)
        )
        now_israel = datetime.now(_ISRAEL)
        date_str = now_israel.strftime("%Y-%m-%d")

        return (
            f"PROXY Daily Report — {date_str}\n"
            f"{'─' * 32}\n"
            f"Cycles run:           {total_cycles}\n"
            f"Opportunities found:  {opportunities}\n"
            f"Actions taken:        {actions_taken}\n"
            f"{'─' * 32}\n"
            f"Next report: tomorrow 08:00 IL"
        )

    def _should_send_daily_report(self, now_utc: datetime | None = None) -> bool:
        """Return True if it is time to send the daily report and it hasn't been sent today."""
        if now_utc is None:
            now_utc = datetime.now(timezone.utc)
        now_israel = now_utc.astimezone(_ISRAEL)
        today = now_israel.date()

        if self._last_report_date == today:
            return False

        hour_ok = now_israel.hour == _DAILY_REPORT_HOUR
        minute_ok = now_israel.minute < _DAILY_REPORT_WINDOW_MIN
        return hour_ok and minute_ok

    def send_daily_report(self) -> bool:
        """Compose and send the daily report. Marks today as reported."""
        text = self._daily_report_text()
        ok = self.relay.send_message(text)
        self._last_report_date = datetime.now(_ISRAEL).date()
        logger.info("ProxyCommander: daily report sent ok=%s", ok)
        return ok

    # ──────────────────────────────────────────────────────────
    #  Async polling loop
    # ──────────────────────────────────────────────────────────

    async def run_polling_loop(self) -> None:
        """Poll Telegram for owner messages, dispatch tasks, send daily reports.

        Runs indefinitely; cancel via asyncio.CancelledError.
        """
        logger.info("ProxyCommander: polling loop started chat_id=%s", self.relay.chat_id)
        offset = self.relay._get_next_offset()

        while True:
            # Check daily report first (non-blocking)
            if self._should_send_daily_report():
                await asyncio.to_thread(self.send_daily_report)

            # Poll for messages (blocking network call, run in thread)
            try:
                texts, offset = await asyncio.to_thread(
                    self.relay._poll, offset, 5
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("ProxyCommander: poll error: %s", exc)
                await asyncio.sleep(_POLL_INTERVAL_SECONDS)
                continue

            for text in texts:
                if not text or not text.strip():
                    continue
                logger.info("ProxyCommander: message received %r", text[:60])
                try:
                    reply = await asyncio.to_thread(self.handle_message, text)
                    await asyncio.to_thread(self.relay.send_message, reply)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.error("ProxyCommander: handle error: %s", exc)
                    try:
                        await asyncio.to_thread(
                            self.relay.send_message,
                            f"Error processing your message: {exc}",
                        )
                    except Exception:
                        pass

            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

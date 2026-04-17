"""Tests for ProxyCommander — TDD RED phase.

ProxyCommander wires TelegramRelay to HouseOmega so PROXY can:
  1. Receive task commands from the owner via Telegram
  2. Execute them through HouseOmega.run()
  3. Reply with the result
  4. Send a daily morning report at 8am Israel time

Coverage:
  1.  from_vault() reads TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID from vault
  2.  from_vault() raises if vault missing required keys
  3.  from_env() builds commander from environment variables
  4.  from_env() returns None when env vars absent
  5.  handle_message() calls omega.run() with the message text
  6.  handle_message() returns success summary including redefined_problem
  7.  handle_message() returns failure message when omega fails
  8.  handle_message() strips /start and /help commands gracefully
  9.  _daily_report_text() contains required sections
  10. _daily_report_text() correctly counts belief_added cycles as opportunities
  11. _daily_report_text() shows total cycles and actions taken
  12. _should_send_daily_report() True at exactly 08:00 Israel time, not yet sent
  13. _should_send_daily_report() False at 09:00 Israel time
  14. _should_send_daily_report() False at 07:59 Israel time
  15. _should_send_daily_report() False if already sent today
  16. send_daily_report() calls relay.send_message() with formatted text
  17. send_daily_report() marks today as reported so it won't re-send
  18. run_polling_loop() dispatches incoming text and sends reply
  19. run_polling_loop() sends daily report when _should_send_daily_report is True
  20. run_polling_loop() does not reply to empty/None updates
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, call
from zoneinfo import ZoneInfo

import pytest

from nexus.core.proxy_commander import ProxyCommander
from nexus.core.telegram_relay import TelegramRelay

_ISRAEL = ZoneInfo("Asia/Jerusalem")

# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def _make_relay(bot_token: str = "tok", chat_id: str = "42") -> TelegramRelay:
    relay = MagicMock(spec=TelegramRelay)
    relay.bot_token = bot_token
    relay.chat_id = chat_id
    relay.send_message.return_value = True
    return relay


def _make_omega(success: bool = True, belief_added: bool = True, problem: str = "find a gig") -> MagicMock:
    omega = MagicMock()
    result = MagicMock()
    result.success = success
    result.belief_added = belief_added
    result.failure_reason = None if success else "House C failed"
    result.cycle_time_seconds = 42.0
    sso = MagicMock()
    sso.redefined_problem = problem
    result.sso = sso if success else None
    omega.run.return_value = result

    # get_cycle_history returns 3 cycles: 2 successful, 1 not
    c1 = MagicMock(); c1.success = True; c1.belief_added = True; c1.build_result = MagicMock()
    c2 = MagicMock(); c2.success = True; c2.belief_added = False; c2.build_result = None
    c3 = MagicMock(); c3.success = False; c3.belief_added = False; c3.build_result = None
    omega.get_cycle_history.return_value = [c1, c2, c3]
    return omega


def _at_israel_hour(hour: int, minute: int = 0) -> datetime:
    """Return a UTC datetime that corresponds to hour:minute in Israel time."""
    israel_dt = datetime(2026, 4, 17, hour, minute, 0, tzinfo=_ISRAEL)
    return israel_dt.astimezone(timezone.utc)


# ═══════════════════════════════════════════════════════════════
#  1–4. Construction / credentials
# ═══════════════════════════════════════════════════════════════

class TestConstruction:

    def test_from_vault_reads_token_and_chat_id(self, tmp_path):
        """from_vault() must read TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from vault."""
        from nexus.core.guardian import GuardianVault
        vault = GuardianVault(str(tmp_path / "vault.enc"), master_key="testkey")
        vault.set("TELEGRAM_BOT_TOKEN", "bot123:abc")
        vault.set("TELEGRAM_CHAT_ID", "99999")

        omega = _make_omega()
        cmd = ProxyCommander.from_vault(str(tmp_path / "vault.enc"), "testkey", omega)
        assert cmd is not None
        assert cmd.relay.bot_token == "bot123:abc"
        assert cmd.relay.chat_id == "99999"

    def test_from_vault_raises_if_token_missing(self, tmp_path):
        """from_vault() must raise KeyError if TELEGRAM_BOT_TOKEN not in vault."""
        from nexus.core.guardian import GuardianVault
        vault = GuardianVault(str(tmp_path / "vault.enc"), master_key="testkey")
        vault.set("TELEGRAM_CHAT_ID", "99999")
        # No TELEGRAM_BOT_TOKEN

        with pytest.raises(KeyError):
            ProxyCommander.from_vault(str(tmp_path / "vault.enc"), "testkey", _make_omega())

    def test_from_env_builds_from_env_vars(self, monkeypatch):
        """from_env() builds a ProxyCommander when env vars are set."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "envtok:xyz")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "77777")
        cmd = ProxyCommander.from_env(_make_omega())
        assert cmd is not None
        assert cmd.relay.bot_token == "envtok:xyz"
        assert cmd.relay.chat_id == "77777"

    def test_from_env_returns_none_when_missing(self, monkeypatch):
        """from_env() returns None when env vars are absent."""
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        cmd = ProxyCommander.from_env(_make_omega())
        assert cmd is None


# ═══════════════════════════════════════════════════════════════
#  5–8. handle_message()
# ═══════════════════════════════════════════════════════════════

class TestHandleMessage:

    def test_handle_message_calls_omega_run(self):
        """handle_message() must call omega.run() with the message text."""
        relay = _make_relay()
        omega = _make_omega()
        cmd = ProxyCommander(relay=relay, omega=omega)
        cmd.handle_message("find a Python gig on freelancer.com")
        omega.run.assert_called_once_with("find a Python gig on freelancer.com")

    def test_handle_message_returns_success_summary(self):
        """handle_message() returns a non-empty success summary."""
        relay = _make_relay()
        omega = _make_omega(success=True, problem="Find Python gig on freelancer.com")
        cmd = ProxyCommander(relay=relay, omega=omega)
        reply = cmd.handle_message("find a gig")
        assert reply  # non-empty
        assert "success" in reply.lower() or "find python gig" in reply.lower() or "freelancer" in reply.lower()

    def test_handle_message_includes_redefined_problem_on_success(self):
        """Success reply must include the SSO's redefined_problem."""
        relay = _make_relay()
        omega = _make_omega(success=True, problem="Search freelancer.com for Python gigs")
        cmd = ProxyCommander(relay=relay, omega=omega)
        reply = cmd.handle_message("find a gig")
        assert "freelancer.com" in reply.lower() or "python" in reply.lower()

    def test_handle_message_returns_failure_on_omega_fail(self):
        """handle_message() returns a failure message when omega fails."""
        relay = _make_relay()
        omega = _make_omega(success=False)
        cmd = ProxyCommander(relay=relay, omega=omega)
        reply = cmd.handle_message("find a gig")
        assert "fail" in reply.lower() or "error" in reply.lower() or "house c" in reply.lower()

    def test_handle_start_command(self):
        """/start command returns a welcome message without running omega."""
        relay = _make_relay()
        omega = _make_omega()
        cmd = ProxyCommander(relay=relay, omega=omega)
        reply = cmd.handle_message("/start")
        omega.run.assert_not_called()
        assert reply  # non-empty welcome

    def test_handle_help_command(self):
        """/help command returns help text without running omega."""
        relay = _make_relay()
        omega = _make_omega()
        cmd = ProxyCommander(relay=relay, omega=omega)
        reply = cmd.handle_message("/help")
        omega.run.assert_not_called()
        assert reply


# ═══════════════════════════════════════════════════════════════
#  9–11. _daily_report_text()
# ═══════════════════════════════════════════════════════════════

class TestDailyReportText:

    def test_report_contains_cycles_run_section(self):
        """Report text must mention total cycles run."""
        relay = _make_relay()
        omega = _make_omega()
        cmd = ProxyCommander(relay=relay, omega=omega)
        text = cmd._daily_report_text()
        assert "cycle" in text.lower()

    def test_report_counts_opportunities_as_belief_added(self):
        """Opportunities found = cycles where belief_added=True."""
        relay = _make_relay()
        omega = _make_omega()
        cmd = ProxyCommander(relay=relay, omega=omega)
        text = cmd._daily_report_text()
        # 1 of 3 mock cycles has belief_added=True
        assert "1" in text

    def test_report_shows_total_cycle_count(self):
        """Report must show the total number of cycles (3 in mock)."""
        relay = _make_relay()
        omega = _make_omega()
        cmd = ProxyCommander(relay=relay, omega=omega)
        text = cmd._daily_report_text()
        assert "3" in text

    def test_report_contains_opportunities_section(self):
        """Report must contain 'opportunit' (opportun-ities/y)."""
        relay = _make_relay()
        omega = _make_omega()
        cmd = ProxyCommander(relay=relay, omega=omega)
        text = cmd._daily_report_text()
        assert "opportunit" in text.lower()

    def test_report_contains_actions_section(self):
        """Report must mention actions taken."""
        relay = _make_relay()
        omega = _make_omega()
        cmd = ProxyCommander(relay=relay, omega=omega)
        text = cmd._daily_report_text()
        assert "action" in text.lower() or "build" in text.lower() or "browser" in text.lower()


# ═══════════════════════════════════════════════════════════════
#  12–15. _should_send_daily_report()
# ═══════════════════════════════════════════════════════════════

class TestShouldSendDailyReport:

    def test_true_at_8am_israel_not_yet_sent(self):
        """Returns True when Israel time is 08:00 and report not yet sent today."""
        relay = _make_relay()
        omega = _make_omega()
        cmd = ProxyCommander(relay=relay, omega=omega)
        now_utc = _at_israel_hour(8, 0)
        assert cmd._should_send_daily_report(now_utc) is True

    def test_true_at_8am_israel_within_window(self):
        """Returns True at 08:05 Israel time (within 10-minute send window)."""
        relay = _make_relay()
        omega = _make_omega()
        cmd = ProxyCommander(relay=relay, omega=omega)
        now_utc = _at_israel_hour(8, 5)
        assert cmd._should_send_daily_report(now_utc) is True

    def test_false_at_9am_israel(self):
        """Returns False when Israel time is 09:00."""
        relay = _make_relay()
        omega = _make_omega()
        cmd = ProxyCommander(relay=relay, omega=omega)
        now_utc = _at_israel_hour(9, 0)
        assert cmd._should_send_daily_report(now_utc) is False

    def test_false_at_7_59_israel(self):
        """Returns False at 07:59 Israel time (too early)."""
        relay = _make_relay()
        omega = _make_omega()
        cmd = ProxyCommander(relay=relay, omega=omega)
        now_utc = _at_israel_hour(7, 59)
        assert cmd._should_send_daily_report(now_utc) is False

    def test_false_if_already_sent_today(self):
        """Returns False if report already sent today."""
        relay = _make_relay()
        omega = _make_omega()
        cmd = ProxyCommander(relay=relay, omega=omega)
        now_utc = _at_israel_hour(8, 0)
        # Mark as sent
        cmd._last_report_date = datetime.now(_ISRAEL).date()
        assert cmd._should_send_daily_report(now_utc) is False


# ═══════════════════════════════════════════════════════════════
#  16–17. send_daily_report()
# ═══════════════════════════════════════════════════════════════

class TestSendDailyReport:

    def test_send_daily_report_calls_relay(self):
        """send_daily_report() must call relay.send_message() with report text."""
        relay = _make_relay()
        omega = _make_omega()
        cmd = ProxyCommander(relay=relay, omega=omega)
        cmd.send_daily_report()
        relay.send_message.assert_called_once()
        text = relay.send_message.call_args[0][0]
        assert "cycle" in text.lower() or "opportunit" in text.lower()

    def test_send_daily_report_marks_sent(self):
        """After send_daily_report(), _should_send_daily_report returns False for today."""
        relay = _make_relay()
        omega = _make_omega()
        cmd = ProxyCommander(relay=relay, omega=omega)
        now_utc = _at_israel_hour(8, 0)
        cmd.send_daily_report()
        assert cmd._should_send_daily_report(now_utc) is False


# ═══════════════════════════════════════════════════════════════
#  18–20. run_polling_loop()
# ═══════════════════════════════════════════════════════════════

class TestRunPollingLoop:
    """Tests for the async polling loop."""

    @pytest.mark.asyncio
    async def test_dispatches_message_and_replies(self):
        """When relay._poll() returns a message, omega.run() is called and relay.send_message() replies."""
        relay = _make_relay()
        omega = _make_omega(success=True, problem="Find Python gig")

        # _poll: first call returns one message, subsequent calls return empty (stop via CancelledError)
        call_count = 0
        def mock_poll(offset, timeout=5):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (["find a Python gig on remote.co"], offset + 1)
            raise asyncio.CancelledError()

        import asyncio
        relay._poll = mock_poll
        relay._get_next_offset = MagicMock(return_value=0)

        cmd = ProxyCommander(relay=relay, omega=omega)

        with pytest.raises(asyncio.CancelledError):
            await cmd.run_polling_loop()

        omega.run.assert_called_once_with("find a Python gig on remote.co")
        # relay.send_message called for the reply
        assert relay.send_message.call_count >= 1

    @pytest.mark.asyncio
    async def test_sends_daily_report_when_due(self):
        """run_polling_loop() sends daily report at 8am Israel time."""
        import asyncio
        relay = _make_relay()
        omega = _make_omega()

        # No incoming messages; just trigger the daily report check
        call_count = 0
        def mock_poll(offset, timeout=5):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                raise asyncio.CancelledError()
            return ([], offset)

        relay._poll = mock_poll
        relay._get_next_offset = MagicMock(return_value=0)

        cmd = ProxyCommander(relay=relay, omega=omega)

        # Patch _should_send_daily_report to return True
        cmd._should_send_daily_report = MagicMock(return_value=True)
        # Patch send_daily_report so we can verify it's called
        cmd.send_daily_report = MagicMock()

        with pytest.raises(asyncio.CancelledError):
            await cmd.run_polling_loop()

        cmd.send_daily_report.assert_called_once()

    @pytest.mark.asyncio
    async def test_ignores_empty_updates(self):
        """Empty poll results must not trigger omega.run()."""
        import asyncio
        relay = _make_relay()
        omega = _make_omega()

        call_count = 0
        def mock_poll(offset, timeout=5):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError()
            return ([], offset)

        relay._poll = mock_poll
        relay._get_next_offset = MagicMock(return_value=0)

        cmd = ProxyCommander(relay=relay, omega=omega)
        cmd._should_send_daily_report = MagicMock(return_value=False)

        with pytest.raises(asyncio.CancelledError):
            await cmd.run_polling_loop()

        omega.run.assert_not_called()

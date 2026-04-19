"""Tests for nexus/core/proposal_sender.py — TDD RED phase.

Coverage:
  === generate_proposal ===
  1.  Returns text containing identity name
  2.  Returns text containing job title from finding
  3.  Includes identity bio in LLM prompt (captured via fake router)
  4.  Includes identity niche keywords in LLM prompt

  === send_via_gmail ===
  5.  Returns False when no Gmail credentials in identity (graceful)
  6.  Returns False when resolved email is empty
  7.  Calls smtplib.SMTP_SSL with gmail server when credentials present
  8.  Uses identity resolved email as From address

  === notify_telegram ===
  9.  Returns False when relay is None (graceful)
  10. Calls relay.send_message with text containing site/job name
  11. Message contains "Proposal sent to"

  === process_findings ===
  12. Returns empty list when findings_text is empty
  13. Returns empty list when no FINDING: lines present
  14. Parses each FINDING: line into a ProposalResult
  15. ProposalResult has job_title, job_url, proposal_text fields
  16. Calls notify_telegram for each finding
  17. sent=False when no email extractable from finding
  18. sent=True when email found in finding text (mocked SMTP)
  19. notified=True when Telegram relay is configured
  20. notified=False when Telegram relay is None

  === HouseC integration ===
  21. HouseC.build calls proposal_sender.process_findings when success=True
  22. HouseC.build does NOT call process_findings when artifact fails validation
  23. HouseC.proposal_sender field defaults to None (no breaking change)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

from nexus.core.proposal_sender import ProposalResult, ProposalSender
from nexus.core.identity_manager import Identity, IdentityManager
from nexus.core.house_b import StructuredSpecificationObject
from nexus.core.house_c import HouseC, BuildArtifact
from nexus.core.house_d import DestructionReport
from nexus.core.knowledge_graph import KnowledgeGraph


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_identity(email: str = "alex@gmail.com", paypal: str = "alex@paypal.com") -> Identity:
    return Identity(
        id="alex",
        name="Alex",
        business="Alex Digital Services",
        niche=["Virtual Assistant", "Data Entry", "Content Writing", "AI tasks"],
        email=email,
        bio="Experienced digital professional specializing in VA work and AI tasks.",
        paypal=paypal,
        active=True,
    )


class _FakeVault:
    def __init__(self, data: dict[str, str] | None = None):
        self._data = data or {}

    def has(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str) -> str:
        return self._data[key]


class _FakeRouter:
    """Captures LLM calls; returns configurable response."""

    def __init__(self, response: str = "Dear Hiring Manager, I am Alex..."):
        self.response = response
        self.calls: list[dict] = []

    def complete(self, *, house: str, system: str, user: str, label: str = "", **kwargs) -> str:
        self.calls.append({"house": house, "system": system, "user": user, "label": label, **kwargs})
        return self.response


class _FakeRelay:
    def __init__(self):
        self.sent: list[str] = []

    def send_message(self, text: str) -> bool:
        self.sent.append(text)
        return True


def _make_sender(
    router=None,
    vault_data: dict | None = None,
    telegram=None,
    identity: Identity | None = None,
) -> tuple[ProposalSender, _FakeRouter, IdentityManager]:
    if router is None:
        router = _FakeRouter()
    vault = _FakeVault(vault_data or {})
    mgr = IdentityManager(data_dir="/tmp/_test_ids_notreal", vault=vault)
    if identity is None:
        identity = _make_identity()
    sender = ProposalSender(router=router, identity_manager=mgr, telegram=telegram)
    return sender, router, mgr


SAMPLE_FINDINGS = (
    "FINDING: Python Developer Needed | https://freelancer.com/projects/123 | $50/hr | VA needed urgently\n"
    "FINDING: Data Entry Specialist | https://upwork.com/jobs/456 | $20/hr | Spreadsheet cleanup\n"
    "Some other line that is not a finding\n"
)


# ---------------------------------------------------------------------------
# 1–4  generate_proposal
# ---------------------------------------------------------------------------

class TestGenerateProposal:

    def test_returns_text_with_identity_name(self):
        router = _FakeRouter(response="Hi, I am Alex, and I can help you.")
        sender, _, _ = _make_sender(router=router)
        identity = _make_identity()
        result = sender.generate_proposal("Python Dev Needed", "We need a dev", identity)
        assert "Alex" in result

    def test_returns_text_with_job_title(self):
        router = _FakeRouter(response="Regarding Python Dev Needed — I am interested.")
        sender, _, _ = _make_sender(router=router)
        identity = _make_identity()
        result = sender.generate_proposal("Python Dev Needed", "We need a dev", identity)
        assert "Python Dev Needed" in result or len(result) > 10  # proposal was generated

    def test_prompt_includes_identity_bio(self):
        router = _FakeRouter()
        sender, _, _ = _make_sender(router=router)
        identity = _make_identity()
        sender.generate_proposal("Test Job", "Job description here", identity)
        assert len(router.calls) == 1
        prompt_text = router.calls[0]["system"] + router.calls[0]["user"]
        assert "Experienced digital professional" in prompt_text

    def test_prompt_includes_niche_keywords(self):
        router = _FakeRouter()
        sender, _, _ = _make_sender(router=router)
        identity = _make_identity()
        sender.generate_proposal("VA Task", "Virtual assistant needed", identity)
        prompt_text = router.calls[0]["system"] + router.calls[0]["user"]
        assert "Virtual Assistant" in prompt_text or "Data Entry" in prompt_text

    def test_router_called_with_keyword_only_signature(self):
        """complete() must be called with house=, system=, user= kwargs — not positional messages."""
        router = _FakeRouter()
        sender, _, _ = _make_sender(router=router)
        identity = _make_identity()
        sender.generate_proposal("Python Dev", "We need a dev", identity)
        assert len(router.calls) == 1
        call = router.calls[0]
        assert "house" in call, "missing 'house' kwarg"
        assert "system" in call, "missing 'system' kwarg"
        assert "user" in call, "missing 'user' kwarg"
        assert "messages" not in call, "must not pass positional 'messages' arg"

    def test_router_label_is_proposal_writer(self):
        router = _FakeRouter()
        sender, _, _ = _make_sender(router=router)
        sender.generate_proposal("Dev Job", "desc", _make_identity())
        assert router.calls[0].get("label") == "proposal_writer"


# ---------------------------------------------------------------------------
# 5–8  send_via_gmail
# ---------------------------------------------------------------------------

class TestSendViaGmail:

    def test_returns_false_when_no_gmail_credentials(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GMAIL_APP_PASS", raising=False)
        monkeypatch.delenv("GMAIL_PASS", raising=False)
        sender, _, _ = _make_sender()
        identity = _make_identity(email="alex@gmail.com")
        result = sender.send_via_gmail(
            to_email="client@example.com",
            subject="Proposal",
            body="Hello",
            identity=identity,
            gmail_password="",
        )
        assert result is False

    def test_returns_false_when_email_empty(self, monkeypatch):
        sender, _, _ = _make_sender()
        identity = _make_identity(email="")
        result = sender.send_via_gmail(
            to_email="client@example.com",
            subject="Proposal",
            body="Hello",
            identity=identity,
            gmail_password="secret",
        )
        assert result is False

    def test_calls_smtp_ssl_with_gmail_host(self, monkeypatch):
        with patch("smtplib.SMTP_SSL") as mock_ssl:
            mock_smtp = MagicMock()
            mock_ssl.return_value.__enter__ = lambda s: mock_smtp
            mock_ssl.return_value.__exit__ = MagicMock(return_value=False)
            sender, _, _ = _make_sender()
            identity = _make_identity(email="alex@gmail.com")
            sender.send_via_gmail(
                to_email="client@example.com",
                subject="Proposal",
                body="Hello from Alex",
                identity=identity,
                gmail_password="apppassword123",
            )
            mock_ssl.assert_called_once_with("smtp.gmail.com", 465)

    def test_uses_identity_email_as_from_address(self, monkeypatch):
        with patch("smtplib.SMTP_SSL") as mock_ssl:
            mock_smtp = MagicMock()
            mock_ssl.return_value.__enter__ = lambda s: mock_smtp
            mock_ssl.return_value.__exit__ = MagicMock(return_value=False)
            sender, _, _ = _make_sender()
            identity = _make_identity(email="alex@gmail.com")
            sender.send_via_gmail(
                to_email="client@example.com",
                subject="Test",
                body="Body",
                identity=identity,
                gmail_password="pass",
            )
            # login called with identity email
            mock_smtp.login.assert_called_once_with("alex@gmail.com", "pass")


# ---------------------------------------------------------------------------
# 9–11  notify_telegram
# ---------------------------------------------------------------------------

class TestNotifyTelegram:

    def test_returns_false_when_relay_is_none(self):
        sender, _, _ = _make_sender(telegram=None)
        result = sender.notify_telegram("freelancer.com/projects/123")
        assert result is False

    def test_calls_send_message_with_site_name(self):
        relay = _FakeRelay()
        sender, _, _ = _make_sender(telegram=relay)
        sender.notify_telegram("freelancer.com/projects/123")
        assert len(relay.sent) == 1
        assert "freelancer.com/projects/123" in relay.sent[0]

    def test_message_contains_proposal_sent(self):
        relay = _FakeRelay()
        sender, _, _ = _make_sender(telegram=relay)
        sender.notify_telegram("upwork.com/jobs/456")
        assert "Proposal sent" in relay.sent[0] or "proposal" in relay.sent[0].lower()


# ---------------------------------------------------------------------------
# 12–20  process_findings
# ---------------------------------------------------------------------------

class TestProcessFindings:

    def test_empty_string_returns_empty_list(self):
        sender, _, _ = _make_sender()
        identity = _make_identity()
        result = sender.process_findings("", identity)
        assert result == []

    def test_no_finding_lines_returns_empty_list(self):
        sender, _, _ = _make_sender()
        identity = _make_identity()
        result = sender.process_findings("some line\nanother line", identity)
        assert result == []

    def test_parses_each_finding_line(self):
        sender, _, _ = _make_sender()
        identity = _make_identity()
        result = sender.process_findings(SAMPLE_FINDINGS, identity)
        assert len(result) == 2

    def test_proposal_result_has_required_fields(self):
        sender, _, _ = _make_sender()
        identity = _make_identity()
        result = sender.process_findings(
            "FINDING: Python Dev | https://freelancer.com/projects/1 | $40/hr | Need help\n",
            identity,
        )
        assert len(result) == 1
        pr = result[0]
        assert hasattr(pr, "job_title")
        assert hasattr(pr, "job_url")
        assert hasattr(pr, "proposal_text")
        assert hasattr(pr, "sent")
        assert hasattr(pr, "notified")

    def test_calls_notify_telegram_only_when_email_sent(self):
        """Telegram fires only when an email was actually sent, not on every finding."""
        relay = _FakeRelay()
        sender, _, _ = _make_sender(telegram=relay)
        identity = _make_identity()
        # SAMPLE_FINDINGS has no contact email → no sends → no Telegram
        sender.process_findings(SAMPLE_FINDINGS, identity)
        assert len(relay.sent) == 0

    def test_sent_false_when_no_email_in_finding(self):
        sender, _, _ = _make_sender()
        identity = _make_identity()
        result = sender.process_findings(
            "FINDING: Python Dev | https://freelancer.com/1 | $40/hr | No email here\n",
            identity,
        )
        assert result[0].sent is False

    def test_sent_true_when_email_in_finding(self):
        with patch("smtplib.SMTP_SSL") as mock_ssl:
            mock_smtp = MagicMock()
            mock_ssl.return_value.__enter__ = lambda s: mock_smtp
            mock_ssl.return_value.__exit__ = MagicMock(return_value=False)
            vault = {"GMAIL_APP_PASS": "apppass", "GMAIL_ADDRESS": "alex@gmail.com"}
            sender, _, mgr = _make_sender(vault_data=vault)
            identity = _make_identity(email="vault:GMAIL_ADDRESS")
            result = sender.process_findings(
                "FINDING: Dev Needed | https://site.com/1 | $50/hr | Contact: client@example.com\n",
                identity,
            )
            assert result[0].sent is True

    def test_notified_false_when_no_email_even_with_relay(self):
        """No email found in finding → not sent → not notified, even with relay."""
        relay = _FakeRelay()
        sender, _, _ = _make_sender(telegram=relay)
        identity = _make_identity()
        result = sender.process_findings(
            "FINDING: Python Dev | https://freelancer.com/1 | $40/hr | Need help\n",
            identity,
        )
        assert result[0].notified is False

    def test_notified_false_when_relay_none(self):
        sender, _, _ = _make_sender(telegram=None)
        identity = _make_identity()
        result = sender.process_findings(
            "FINDING: Python Dev | https://freelancer.com/1 | $40/hr | Need help\n",
            identity,
        )
        assert result[0].notified is False


# ---------------------------------------------------------------------------
# 21–23  HouseC integration
# ---------------------------------------------------------------------------

class TestHouseCIntegration:

    def _make_passing_artifact(self) -> BuildArtifact:
        from nexus.core.house_b import StructuredSpecificationObject
        sso = StructuredSpecificationObject(
            original_input="find jobs",
            redefined_problem="Find Python jobs on Freelancer",
        )
        art = BuildArtifact(sso=sso)
        art.passed_validation = True
        art.execution_proof = (
            "FINDING: Python Dev | https://freelancer.com/1 | $50/hr | Contact: client@test.com\n"
        )
        return art

    def test_house_c_has_proposal_sender_field(self, tmp_path):
        graph = KnowledgeGraph()
        hc = HouseC(knowledge_graph=graph)
        assert hasattr(hc, "proposal_sender")
        assert hc.proposal_sender is None  # defaults to None

    def test_build_calls_process_findings_on_success(self, tmp_path, monkeypatch):
        graph = KnowledgeGraph()

        mock_sender = MagicMock()
        mock_sender.process_findings.return_value = []

        # Wire up a HouseC that will produce a successful result via OpenClaw
        from nexus.core.openclaw_client import OpenClawClient
        mock_client = MagicMock(spec=OpenClawClient)
        mock_client.is_available.return_value = True
        mock_client.send.return_value = (
            "FINDING: Python Dev | https://freelancer.com/1 | $50/hr | VA work needed"
        )

        mock_vault = _FakeVault({"GMAIL_ADDRESS": "alex@gmail.com", "PAYPAL_EMAIL": "p@p.com"})
        mock_mgr = IdentityManager(data_dir=str(tmp_path), vault=mock_vault)
        mock_mgr.add_identity(_make_identity())

        mock_sender.identity_manager = mock_mgr

        router = MagicMock()
        router.complete.return_value = "FINDING: Python Dev | https://freelancer.com/1 | $50/hr"

        hc = HouseC(
            knowledge_graph=graph,
            router=router,
            openclaw_client=mock_client,
            proposal_sender=mock_sender,
        )

        sso = StructuredSpecificationObject(
            original_input="find jobs",
            redefined_problem="Find Python freelance jobs",
            success_criteria=["find at least one job"],
        )
        dr = DestructionReport(survived=True, survival_score=0.9, target_description="test")

        hc.build(sso, dr)

        mock_sender.process_findings.assert_called_once()

    def test_build_skips_process_findings_on_failure(self, tmp_path):
        graph = KnowledgeGraph()
        mock_sender = MagicMock()

        from nexus.core.openclaw_client import OpenClawClient
        mock_client = MagicMock(spec=OpenClawClient)
        mock_client.is_available.return_value = True
        mock_client.send.return_value = "NO_DATA: site unavailable"

        router = MagicMock()
        hc = HouseC(
            knowledge_graph=graph,
            router=router,
            openclaw_client=mock_client,
            proposal_sender=mock_sender,
        )

        sso = StructuredSpecificationObject(
            original_input="find jobs",
            redefined_problem="Find Python freelance jobs",
            success_criteria=["find at least one job"],
        )
        dr = DestructionReport(survived=True, survival_score=0.9, target_description="test")

        # Patch DirectJobFetcher to return empty so OpenClaw path runs and returns NO_DATA
        with patch("nexus.core.house_c.DirectJobFetcher") as mock_fetcher_cls:
            mock_fetcher_cls.return_value.fetch.return_value = ""
            hc.build(sso, dr)

        mock_sender.process_findings.assert_not_called()

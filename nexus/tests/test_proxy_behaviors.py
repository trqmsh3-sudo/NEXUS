"""Tests for two PROXY behaviors.

Behavior 1 — Market-rate research before quoting
    COMPETITIVE_PERCENTILE_BAND constant exists (tuple of two floats in (0,1))
    RATE_REGEX constant exists and matches common rate patterns
    _research_market_rates() method exists on HouseC
    Returns dict with required keys: market_low, market_high, competitive_quote,
        sample_size, currency, context
    competitive_quote is between market_low and market_high
    competitive_quote is within the COMPETITIVE_PERCENTILE_BAND of the spread
    Falls back to sensible defaults when Reddit is unreachable
    Falls back to sensible defaults when no rates found in posts
    _generate_action_script prompt includes market-rate context
    Prompt instructs LLM to quote competitively (not cheapest, not highest)

Behavior 2 — Follow-up email after completed job
    FOLLOWUP_EMAIL_TEMPLATE constant exists
    Template contains: thank / referral / 10% / discount
    Template has a non-empty subject line (FOLLOWUP_EMAIL_SUBJECT)
    _send_followup_email() method exists on HouseC
    Uses GMAIL_USER / GMAIL_APP_PASS (or GMAIL_ADDRESS / GMAIL_APP_PASS)
    Returns True on success, False on failure without raising
    SMTP_SSL is called with Gmail host on port 465
    Email body contains the 10% referral discount offer
    build() calls _send_followup_email when success=True and proof contains email
    build() does NOT call _send_followup_email when success=False
    build() silently skips when no client email is extractable from proof
"""

from __future__ import annotations

import re
import smtplib
from unittest.mock import MagicMock, call, patch

import pytest

import nexus.core.house_c as house_c_mod
from nexus.core.house_c import (
    COMPETITIVE_PERCENTILE_BAND,
    FOLLOWUP_EMAIL_SUBJECT,
    FOLLOWUP_EMAIL_TEMPLATE,
    RATE_REGEX,
    HouseC,
    BuildArtifact,
)
from nexus.core.house_b import StructuredSpecificationObject
from nexus.core.house_d import DestructionReport
from nexus.core.knowledge_graph import KnowledgeGraph


# ── helpers ────────────────────────────────────────────────────────────────────

def _sso(problem: str = "find freelance work", domain: str = "freelance") -> StructuredSpecificationObject:
    return StructuredSpecificationObject(
        original_input=problem,
        redefined_problem=problem,
        domain=domain,
        constraints=[],
        success_criteria=["land at least one client"],
    )


def _dr(survived: bool = True) -> DestructionReport:
    return DestructionReport(
        target_description="test",
        survived=survived,
        survival_score=0.8,
        cycles_survived=1,
        recommendation="PROMOTE",
        attacks=[],
    )


def _graph() -> KnowledgeGraph:
    return MagicMock(spec=KnowledgeGraph)


def _hc(tmp_path) -> HouseC:
    return HouseC(knowledge_graph=_graph(), workspace_dir=str(tmp_path))


# ══════════════════════════════════════════════════════════════════
#  Behavior 1 — Constants
# ══════════════════════════════════════════════════════════════════

class TestMarketRateConstants:
    def test_competitive_percentile_band_exists(self):
        assert hasattr(house_c_mod, "COMPETITIVE_PERCENTILE_BAND")

    def test_competitive_percentile_band_is_two_floats(self):
        assert isinstance(COMPETITIVE_PERCENTILE_BAND, tuple)
        assert len(COMPETITIVE_PERCENTILE_BAND) == 2
        low, high = COMPETITIVE_PERCENTILE_BAND
        assert 0.0 < low < high < 1.0

    def test_competitive_band_excludes_extremes(self):
        """Band must exclude cheapest and most expensive."""
        low, high = COMPETITIVE_PERCENTILE_BAND
        assert low >= 0.25, "Band must not include the cheapest quartile"
        assert high <= 0.80, "Band must not include the most expensive quintile"

    def test_rate_regex_exists(self):
        assert hasattr(house_c_mod, "RATE_REGEX")

    def test_rate_regex_matches_dollar_hourly(self):
        assert re.search(RATE_REGEX, "$50/hr")
        assert re.search(RATE_REGEX, "$75/hour")
        assert re.search(RATE_REGEX, "$120 per hour")

    def test_rate_regex_matches_dollar_project(self):
        assert re.search(RATE_REGEX, "$500/project")
        assert re.search(RATE_REGEX, "$1500 per project")
        assert re.search(RATE_REGEX, "$200 fixed")

    def test_rate_regex_matches_range(self):
        assert re.search(RATE_REGEX, "$30-$60/hr")
        assert re.search(RATE_REGEX, "$50-100/hour")

    def test_rate_regex_does_not_match_plain_text(self):
        assert not re.search(RATE_REGEX, "I am a developer")
        assert not re.search(RATE_REGEX, "great opportunity awaits")


# ══════════════════════════════════════════════════════════════════
#  Behavior 1 — _research_market_rates()
# ══════════════════════════════════════════════════════════════════

class TestResearchMarketRates:
    def test_method_exists(self, tmp_path):
        assert hasattr(_hc(tmp_path), "_research_market_rates")

    def test_returns_dict(self, tmp_path):
        hc = _hc(tmp_path)
        with patch("urllib.request.urlopen", side_effect=OSError("offline")):
            result = hc._research_market_rates()
        assert isinstance(result, dict)

    def test_required_keys_present_on_success(self, tmp_path):
        hc = _hc(tmp_path)
        required = {"market_low", "market_high", "competitive_quote", "sample_size", "currency", "context"}
        with patch("urllib.request.urlopen", side_effect=OSError("offline")):
            result = hc._research_market_rates()
        assert required.issubset(result.keys()), f"Missing keys: {required - result.keys()}"

    def test_fallback_on_network_error(self, tmp_path):
        hc = _hc(tmp_path)
        with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
            result = hc._research_market_rates()
        assert result["sample_size"] == 0
        assert result["competitive_quote"] > 0, "Must provide a non-zero default quote"

    def test_fallback_on_no_rates_found(self, tmp_path):
        import json, io
        hc = _hc(tmp_path)
        # Reddit returns posts but none mention rates
        payload = {"data": {"children": [
            {"data": {"title": "Looking for a Python developer", "selftext": "great project", "author": "user1"}},
        ]}}
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps(payload).encode()
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = hc._research_market_rates()
        assert result["sample_size"] == 0
        assert result["competitive_quote"] > 0

    def test_parses_rates_from_posts(self, tmp_path):
        import json, io
        hc = _hc(tmp_path)
        payload = {"data": {"children": [
            {"data": {"title": "Need dev $40/hr budget", "selftext": "", "author": "a"}},
            {"data": {"title": "Web dev needed", "selftext": "$60/hour", "author": "b"}},
            {"data": {"title": "Quick project $80/hr available", "selftext": "", "author": "c"}},
            {"data": {"title": "Looking for help $100/hour", "selftext": "", "author": "d"}},
        ]}}
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps(payload).encode()
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = hc._research_market_rates()
        assert result["sample_size"] >= 2
        assert result["market_low"] <= result["competitive_quote"] <= result["market_high"]

    def test_competitive_quote_not_cheapest(self, tmp_path):
        import json
        hc = _hc(tmp_path)
        payload = {"data": {"children": [
            {"data": {"title": f"Dev work ${r}/hr", "selftext": "", "author": f"u{i}"}}
            for i, r in enumerate([20, 40, 60, 80, 100])
        ]}}
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps(payload).encode()
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = hc._research_market_rates()
        assert result["competitive_quote"] > result["market_low"], (
            "competitive_quote must be above the cheapest rate"
        )

    def test_competitive_quote_not_most_expensive(self, tmp_path):
        import json
        hc = _hc(tmp_path)
        payload = {"data": {"children": [
            {"data": {"title": f"Dev work ${r}/hr", "selftext": "", "author": f"u{i}"}}
            for i, r in enumerate([20, 40, 60, 80, 100])
        ]}}
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps(payload).encode()
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = hc._research_market_rates()
        assert result["competitive_quote"] < result["market_high"], (
            "competitive_quote must be below the highest rate"
        )

    def test_context_string_mentions_competitive(self, tmp_path):
        hc = _hc(tmp_path)
        with patch("urllib.request.urlopen", side_effect=OSError):
            result = hc._research_market_rates()
        assert "competitive" in result["context"].lower() or "market" in result["context"].lower()


# ══════════════════════════════════════════════════════════════════
#  Behavior 1 — Prompt injection
# ══════════════════════════════════════════════════════════════════

class TestMarketRatePromptInjection:
    def test_generate_script_prompt_includes_market_rate_context(self, tmp_path):
        hc = _hc(tmp_path)
        captured = {}

        def fake_llm(self_inner, system, user, label):
            captured["user"] = user
            return "# NEXUS Action\nprint('FINDING: test')"

        fake_rates = {
            "market_low": 30.0, "market_high": 90.0,
            "competitive_quote": 55.0, "sample_size": 8,
            "currency": "USD", "context": "Market rates: $30–$90/hr. Quote $55/hr (competitive mid-band).",
        }

        with patch.object(HouseC, "_call_llm", fake_llm), \
             patch.object(HouseC, "_research_market_rates", return_value=fake_rates):
            hc._generate_action_script(_sso("find freelance work on r/forhire"))

        prompt = captured.get("user", "")
        assert "55" in prompt or "competitive" in prompt.lower() or "market" in prompt.lower(), (
            "Prompt must include market rate context"
        )

    def test_generate_script_prompt_says_not_cheapest(self, tmp_path):
        hc = _hc(tmp_path)
        captured = {}

        def fake_llm(self_inner, system, user, label):
            captured["user"] = user
            return "# NEXUS Action\nprint('FINDING: test')"

        with patch.object(HouseC, "_call_llm", fake_llm), \
             patch.object(HouseC, "_research_market_rates", return_value={
                 "market_low": 20.0, "market_high": 100.0,
                 "competitive_quote": 55.0, "sample_size": 5,
                 "currency": "USD", "context": "Quote competitively.",
             }):
            hc._generate_action_script(_sso("find freelance work"))

        prompt = captured.get("user", "")
        assert "not the cheapest" in prompt.lower() or "competitive" in prompt.lower() or \
               "mid" in prompt.lower() or "55" in prompt, (
            "Prompt must discourage lowest-price quoting"
        )


# ══════════════════════════════════════════════════════════════════
#  Behavior 2 — Template constants
# ══════════════════════════════════════════════════════════════════

class TestFollowupEmailConstants:
    def test_template_exists(self):
        assert hasattr(house_c_mod, "FOLLOWUP_EMAIL_TEMPLATE")

    def test_subject_exists(self):
        assert hasattr(house_c_mod, "FOLLOWUP_EMAIL_SUBJECT")

    def test_subject_is_non_empty_string(self):
        assert isinstance(FOLLOWUP_EMAIL_SUBJECT, str)
        assert len(FOLLOWUP_EMAIL_SUBJECT.strip()) > 0

    def test_template_is_non_empty_string(self):
        assert isinstance(FOLLOWUP_EMAIL_TEMPLATE, str)
        assert len(FOLLOWUP_EMAIL_TEMPLATE.strip()) > 0

    def test_template_contains_thank(self):
        assert "thank" in FOLLOWUP_EMAIL_TEMPLATE.lower()

    def test_template_contains_referral(self):
        assert "referral" in FOLLOWUP_EMAIL_TEMPLATE.lower() or \
               "refer" in FOLLOWUP_EMAIL_TEMPLATE.lower()

    def test_template_contains_discount(self):
        assert "discount" in FOLLOWUP_EMAIL_TEMPLATE.lower()

    def test_template_contains_10_percent(self):
        assert "10%" in FOLLOWUP_EMAIL_TEMPLATE or "10 percent" in FOLLOWUP_EMAIL_TEMPLATE.lower()

    def test_template_has_placeholder_for_client_name(self):
        assert "{client_name}" in FOLLOWUP_EMAIL_TEMPLATE or \
               "{name}" in FOLLOWUP_EMAIL_TEMPLATE or \
               "{{client}}" in FOLLOWUP_EMAIL_TEMPLATE

    def test_template_has_placeholder_for_job(self):
        assert "{job" in FOLLOWUP_EMAIL_TEMPLATE or "{project" in FOLLOWUP_EMAIL_TEMPLATE


# ══════════════════════════════════════════════════════════════════
#  Behavior 2 — _send_followup_email()
# ══════════════════════════════════════════════════════════════════

class TestSendFollowupEmail:
    def test_method_exists(self, tmp_path):
        assert hasattr(_hc(tmp_path), "_send_followup_email")

    def test_returns_true_on_smtp_success(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GMAIL_USER", "sender@example.com")
        monkeypatch.setenv("GMAIL_APP_PASS", "secret")
        hc = _hc(tmp_path)
        mock_smtp = MagicMock()
        mock_smtp.__enter__ = lambda s: s
        mock_smtp.__exit__ = MagicMock(return_value=False)
        with patch("smtplib.SMTP_SSL", return_value=mock_smtp):
            result = hc._send_followup_email(
                client_email="client@example.com",
                client_name="Alice",
                job_summary="Built a landing page",
            )
        assert result is True

    def test_returns_false_on_smtp_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GMAIL_USER", "sender@example.com")
        monkeypatch.setenv("GMAIL_APP_PASS", "secret")
        hc = _hc(tmp_path)
        with patch("smtplib.SMTP_SSL", side_effect=smtplib.SMTPException("auth failed")):
            result = hc._send_followup_email(
                client_email="client@example.com",
                client_name="Bob",
                job_summary="SEO audit",
            )
        assert result is False

    def test_returns_false_when_no_credentials(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GMAIL_USER", raising=False)
        monkeypatch.delenv("GMAIL_ADDRESS", raising=False)
        monkeypatch.delenv("GMAIL_APP_PASS", raising=False)
        monkeypatch.delenv("GMAIL_PASS", raising=False)
        hc = _hc(tmp_path)
        result = hc._send_followup_email(
            client_email="client@example.com",
            client_name="Carol",
            job_summary="Content writing",
        )
        assert result is False

    def test_smtp_ssl_called_with_gmail_host(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GMAIL_USER", "sender@example.com")
        monkeypatch.setenv("GMAIL_APP_PASS", "secret")
        hc = _hc(tmp_path)
        mock_smtp = MagicMock()
        mock_smtp.__enter__ = lambda s: s
        mock_smtp.__exit__ = MagicMock(return_value=False)
        with patch("smtplib.SMTP_SSL", return_value=mock_smtp) as smtp_cls:
            hc._send_followup_email("c@example.com", "Dave", "web dev")
        args = smtp_cls.call_args
        host = args[0][0] if args[0] else args[1].get("host", "")
        assert "smtp.gmail.com" in host

    def test_email_body_contains_10_percent_discount(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GMAIL_USER", "sender@example.com")
        monkeypatch.setenv("GMAIL_APP_PASS", "secret")
        hc = _hc(tmp_path)
        sent_messages = []
        mock_smtp = MagicMock()
        mock_smtp.__enter__ = lambda s: s
        mock_smtp.__exit__ = MagicMock(return_value=False)
        mock_smtp.sendmail.side_effect = lambda frm, to, msg: sent_messages.append(msg)
        with patch("smtplib.SMTP_SSL", return_value=mock_smtp):
            hc._send_followup_email("c@example.com", "Eve", "logo design")
        assert sent_messages, "sendmail was never called"
        body = sent_messages[0]
        assert "10%" in body or "10 percent" in body.lower(), "Email must mention 10% discount"

    def test_email_body_mentions_referral(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GMAIL_USER", "sender@example.com")
        monkeypatch.setenv("GMAIL_APP_PASS", "secret")
        hc = _hc(tmp_path)
        sent_messages = []
        mock_smtp = MagicMock()
        mock_smtp.__enter__ = lambda s: s
        mock_smtp.__exit__ = MagicMock(return_value=False)
        mock_smtp.sendmail.side_effect = lambda frm, to, msg: sent_messages.append(msg)
        with patch("smtplib.SMTP_SSL", return_value=mock_smtp):
            hc._send_followup_email("c@example.com", "Frank", "api integration")
        body = sent_messages[0]
        assert "refer" in body.lower(), "Email must mention referral"

    def test_uses_gmail_address_fallback(self, tmp_path, monkeypatch):
        """GMAIL_ADDRESS should be used when GMAIL_USER is absent."""
        monkeypatch.delenv("GMAIL_USER", raising=False)
        monkeypatch.setenv("GMAIL_ADDRESS", "alt@example.com")
        monkeypatch.setenv("GMAIL_APP_PASS", "secret")
        hc = _hc(tmp_path)
        mock_smtp = MagicMock()
        mock_smtp.__enter__ = lambda s: s
        mock_smtp.__exit__ = MagicMock(return_value=False)
        with patch("smtplib.SMTP_SSL", return_value=mock_smtp):
            result = hc._send_followup_email("c@example.com", "Grace", "copywriting")
        assert result is True


# ══════════════════════════════════════════════════════════════════
#  Behavior 2 — build() integration
# ══════════════════════════════════════════════════════════════════

class TestBuildFollowupIntegration:
    def _make_proof_with_email(self) -> str:
        return "FINDING: Great project completed\nClient: client@example.com\nWork done."

    def _make_proof_no_email(self) -> str:
        return "FINDING: Great project completed\nWork done."

    def test_followup_sent_on_successful_build(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GMAIL_USER", "sender@example.com")
        monkeypatch.setenv("GMAIL_APP_PASS", "secret")
        sso = _sso("complete a freelance job for client@example.com")
        hc = HouseC(knowledge_graph=_graph(), workspace_dir=str(tmp_path))

        passing_artifact = BuildArtifact(
            sso=sso,
            passed_validation=True,
            execution_proof=self._make_proof_with_email(),
        )

        with patch.object(hc, "_generate_action_script", return_value="# NEXUS Action\nprint('FINDING: done client@example.com')"), \
             patch.object(hc, "_execute_action", return_value=passing_artifact), \
             patch.object(hc, "_send_followup_email", return_value=True) as mock_followup:
            hc.build(sso, _dr())

        mock_followup.assert_called_once()

    def test_followup_not_sent_on_failed_build(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GMAIL_USER", "sender@example.com")
        monkeypatch.setenv("GMAIL_APP_PASS", "secret")
        sso = _sso("complete a job for client@example.com")
        hc = HouseC(knowledge_graph=_graph(), workspace_dir=str(tmp_path))

        failing_artifact = BuildArtifact(
            sso=sso,
            passed_validation=False,
            validation_errors=["NO_DATA: nothing found"],
        )

        with patch.object(hc, "_generate_action_script", return_value="# NEXUS Action\nprint('NO_DATA: x')"), \
             patch.object(hc, "_execute_action", return_value=failing_artifact), \
             patch.object(hc, "_send_followup_email", return_value=False) as mock_followup:
            hc.build(sso, _dr())

        mock_followup.assert_not_called()

    def test_followup_skipped_silently_when_no_email_in_proof(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GMAIL_USER", "sender@example.com")
        monkeypatch.setenv("GMAIL_APP_PASS", "secret")
        sso = _sso("complete a freelance job — no email in output")
        hc = HouseC(knowledge_graph=_graph(), workspace_dir=str(tmp_path))

        passing_no_email = BuildArtifact(
            sso=sso,
            passed_validation=True,
            execution_proof=self._make_proof_no_email(),
        )

        with patch.object(hc, "_generate_action_script", return_value="# NEXUS Action\nprint('FINDING: done')"), \
             patch.object(hc, "_execute_action", return_value=passing_no_email), \
             patch.object(hc, "_send_followup_email") as mock_followup:
            result = hc.build(sso, _dr())

        mock_followup.assert_not_called()
        assert result is not None  # build still returns normally

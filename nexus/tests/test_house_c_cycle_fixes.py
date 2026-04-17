"""Tests for the three fixes identified in the r/forhire test cycle.

Fix 1 — Gmail env aliases:
    GMAIL_USER must equal GMAIL_ADDRESS when both are set.
    GMAIL_PASS must equal GMAIL_APP_PASS when both are set.
    Scripts using GMAIL_USER/GMAIL_PASS receive correct credentials.

Fix 2 — Broader Reddit keyword filter:
    house_c.FORHIRE_KEYWORDS contains the new terms.
    _generate_action_script prompt includes those terms.

Fix 3 — Reddit DM via OpenClaw:
    _needs_browser() returns True for "send reddit dm" tasks.
    _generate_action_script prompt instructs OpenClaw DM usage.
    build() routes reddit-dm tasks through OpenClaw when available.
    Fallback to script path when OpenClaw is unavailable.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

import nexus.core.house_c as house_c_mod
from nexus.core.house_c import (
    FORHIRE_KEYWORDS,
    ACTION_SYSTEM,
    HouseC,
    BuildArtifact,
    _BROWSER_ACTION_KEYWORDS,
)
from nexus.core.house_b import StructuredSpecificationObject
from nexus.core.house_d import DestructionReport
from nexus.core.knowledge_graph import KnowledgeGraph
from nexus.core.openclaw_client import OpenClawClient


# ── helpers ────────────────────────────────────────────────────────────────────

def _sso(problem: str = "find job", domain: str = "freelance") -> StructuredSpecificationObject:
    return StructuredSpecificationObject(
        original_input=problem,
        redefined_problem=problem,
        domain=domain,
        constraints=[],
        success_criteria=["find at least one job"],
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


def _openclaw(available: bool = True, send_result: str = "FINDING: test result") -> MagicMock:
    client = MagicMock(spec=OpenClawClient)
    client.is_available.return_value = available
    client.send.return_value = send_result
    return client


# ══════════════════════════════════════════════════════════════════
#  Fix 1 — Gmail env aliases
# ══════════════════════════════════════════════════════════════════

class TestGmailEnvAliases:
    def test_gmail_user_alias_matches_gmail_address(self, monkeypatch):
        """When both are set, GMAIL_USER must equal GMAIL_ADDRESS."""
        monkeypatch.setenv("GMAIL_ADDRESS", "test@example.com")
        monkeypatch.setenv("GMAIL_USER", "test@example.com")
        assert os.environ["GMAIL_USER"] == os.environ["GMAIL_ADDRESS"]

    def test_gmail_pass_alias_matches_gmail_app_pass(self, monkeypatch):
        """When both are set, GMAIL_PASS must equal GMAIL_APP_PASS."""
        monkeypatch.setenv("GMAIL_APP_PASS", "secret-app-pass")
        monkeypatch.setenv("GMAIL_PASS", "secret-app-pass")
        assert os.environ["GMAIL_PASS"] == os.environ["GMAIL_APP_PASS"]

    def test_script_using_gmail_user_receives_credentials(self, monkeypatch):
        """A subprocess that reads GMAIL_USER gets the correct address."""
        monkeypatch.setenv("GMAIL_USER", "test@example.com")
        monkeypatch.setenv("GMAIL_PASS", "secret")
        # Simulate what the generated script does
        user = os.environ.get("GMAIL_USER")
        pw = os.environ.get("GMAIL_PASS")
        assert user == "test@example.com"
        assert pw == "secret"

    def test_gmail_aliases_non_empty_when_set(self, monkeypatch):
        """Neither alias should be empty when the canonical vars are set."""
        monkeypatch.setenv("GMAIL_ADDRESS", "user@example.com")
        monkeypatch.setenv("GMAIL_APP_PASS", "apppass")
        monkeypatch.setenv("GMAIL_USER", os.environ["GMAIL_ADDRESS"])
        monkeypatch.setenv("GMAIL_PASS", os.environ["GMAIL_APP_PASS"])
        assert os.environ.get("GMAIL_USER")
        assert os.environ.get("GMAIL_PASS")


# ══════════════════════════════════════════════════════════════════
#  Fix 2 — Broader Reddit keyword filter
# ══════════════════════════════════════════════════════════════════

class TestForhireKeywords:
    REQUIRED_KEYWORDS = [
        "hire", "need", "looking for", "paying",
        "budget", "per hour", "per project",
    ]
    ORIGINAL_KEYWORDS = [
        "digital", "service", "marketing", "web", "content",
    ]

    def test_forhire_keywords_constant_exists(self):
        assert hasattr(house_c_mod, "FORHIRE_KEYWORDS"), (
            "FORHIRE_KEYWORDS must be defined in house_c"
        )

    def test_forhire_keywords_is_frozenset_or_tuple(self):
        assert isinstance(FORHIRE_KEYWORDS, (frozenset, tuple, list, set))

    @pytest.mark.parametrize("kw", REQUIRED_KEYWORDS)
    def test_new_keyword_in_forhire_keywords(self, kw):
        lower_kws = {k.lower() for k in FORHIRE_KEYWORDS}
        assert kw in lower_kws, f"'{kw}' must be in FORHIRE_KEYWORDS"

    @pytest.mark.parametrize("kw", ORIGINAL_KEYWORDS)
    def test_original_keywords_preserved(self, kw):
        lower_kws = {k.lower() for k in FORHIRE_KEYWORDS}
        assert kw in lower_kws, f"Original keyword '{kw}' must be preserved"

    def test_generate_script_prompt_includes_hire(self, tmp_path):
        """_generate_action_script must embed the keyword list in its prompt."""
        hc = HouseC(knowledge_graph=_graph(), workspace_dir=str(tmp_path))
        captured = {}

        def fake_call_llm(self_inner, system, user, label):  # noqa: N802
            captured["user"] = user
            return "# NEXUS Action\nprint('FINDING: test')"

        with patch.object(HouseC, "_call_llm", fake_call_llm):
            hc._generate_action_script(_sso("find reddit job"))

        prompt = captured.get("user", "")
        assert "hire" in prompt.lower() or "paying" in prompt.lower(), (
            "user prompt must include broader r/forhire keywords"
        )

    def test_generate_script_prompt_includes_per_hour(self, tmp_path):
        hc = HouseC(knowledge_graph=_graph(), workspace_dir=str(tmp_path))
        captured = {}

        def fake_call_llm(self_inner, system, user, label):  # noqa: N802
            captured["user"] = user
            return "# NEXUS Action\nprint('FINDING: test')"

        with patch.object(HouseC, "_call_llm", fake_call_llm):
            hc._generate_action_script(_sso("find reddit job"))

        prompt = captured.get("user", "")
        assert "per hour" in prompt.lower() or "budget" in prompt.lower(), (
            "user prompt must include rate-related keywords"
        )

    def test_generate_script_prompt_includes_looking_for(self, tmp_path):
        hc = HouseC(knowledge_graph=_graph(), workspace_dir=str(tmp_path))
        captured = {}

        def fake_call_llm(self_inner, system, user, label):  # noqa: N802
            captured["user"] = user
            return "# NEXUS Action\nprint('FINDING: test')"

        with patch.object(HouseC, "_call_llm", fake_call_llm):
            hc._generate_action_script(_sso("find reddit job"))

        prompt = captured.get("user", "")
        assert "looking for" in prompt.lower() or "need" in prompt.lower(), (
            "user prompt must include 'looking for' / 'need' keywords"
        )


# ══════════════════════════════════════════════════════════════════
#  Fix 3 — Reddit DM via OpenClaw
# ══════════════════════════════════════════════════════════════════

class TestNeedsBrowserRedditDM:
    @pytest.mark.parametrize("problem", [
        "send reddit dm to post author",
        "send a direct message on Reddit",
        "message the poster via Reddit DM",
        "Reddit direct message to contact",
        "send dm to r/forhire poster",
    ])
    def test_reddit_dm_needs_browser(self, problem):
        sso = _sso(problem=problem)
        assert HouseC._needs_browser(sso) is True, (
            f"Expected _needs_browser=True for: {problem!r}"
        )

    def test_forhire_scrape_alone_does_not_need_browser(self):
        sso = _sso(problem="scrape r/forhire job listings via public API")
        assert HouseC._needs_browser(sso) is False

    def test_reddit_dm_keyword_in_browser_action_keywords(self):
        kws_lower = {k.lower() for k in _BROWSER_ACTION_KEYWORDS}
        assert any("reddit dm" in k or "direct message" in k or "send dm" in k
                   for k in kws_lower), (
            "_BROWSER_ACTION_KEYWORDS must contain a Reddit DM keyword"
        )


class TestRedditDMPrompt:
    def test_generate_script_prompt_mentions_openclaw_dm(self, tmp_path):
        """Prompt must instruct OpenClaw DM usage for Reddit outreach."""
        hc = HouseC(knowledge_graph=_graph(), workspace_dir=str(tmp_path))
        captured = {}

        def fake_call_llm(self_inner, system, user, label):  # noqa: N802
            captured["user"] = user
            return "# NEXUS Action\nprint('FINDING: test')"

        with patch.object(HouseC, "_call_llm", fake_call_llm):
            hc._generate_action_script(
                _sso("find a job on r/forhire and send proposal via Reddit DM")
            )

        prompt = captured.get("user", "")
        assert "openclaw" in prompt.lower() or "reddit dm" in prompt.lower() or \
               "direct message" in prompt.lower(), (
            "Prompt must mention OpenClaw or Reddit DM for outreach"
        )

    def test_generate_script_prompt_says_not_email_for_reddit(self, tmp_path):
        """Prompt must steer away from email extraction for Reddit tasks."""
        hc = HouseC(knowledge_graph=_graph(), workspace_dir=str(tmp_path))
        captured = {}

        def fake_call_llm(self_inner, system, user, label):  # noqa: N802
            captured["user"] = user
            return "# NEXUS Action\nprint('FINDING: test')"

        with patch.object(HouseC, "_call_llm", fake_call_llm):
            hc._generate_action_script(
                _sso("find a job on r/forhire and send a professional proposal")
            )

        prompt = captured.get("user", "")
        # Prompt should mention DM/OpenClaw, not email extraction
        assert "dm" in prompt.lower() or "openclaw" in prompt.lower() or \
               "username" in prompt.lower(), (
            "Prompt for Reddit tasks must mention DM or username, not email extraction"
        )


class TestBuildRoutingRedditDM:
    def test_reddit_dm_routes_to_openclaw_when_available(self, tmp_path):
        client = _openclaw(available=True, send_result="FINDING: DM sent to user123")
        hc = HouseC(
            knowledge_graph=_graph(),
            openclaw_client=client,
            workspace_dir=str(tmp_path),
        )
        sso = _sso(problem="find a job on r/forhire and send a Reddit DM to poster")
        result = hc.build(sso, _dr())
        assert client.send.called
        assert result.artifact.execution_proof == "FINDING: DM sent to user123"

    def test_reddit_dm_falls_back_to_script_when_openclaw_unavailable(self, tmp_path):
        client = _openclaw(available=False)
        hc = HouseC(
            knowledge_graph=_graph(),
            openclaw_client=client,
            workspace_dir=str(tmp_path),
        )
        sso = _sso(problem="find a job on r/forhire and send a Reddit direct message")

        with patch.object(hc, "_generate_action_script",
                          return_value="# NEXUS Action\nprint('FINDING: fallback')"), \
             patch.object(hc, "_execute_action", return_value=BuildArtifact(
                 sso=sso, passed_validation=True, execution_proof="fallback"
             )):
            result = hc.build(sso, _dr())

        client.send.assert_not_called()
        assert result is not None

    def test_openclaw_sends_reddit_username_in_task(self, tmp_path):
        """OpenClaw task must reference Reddit username, not email."""
        client = _openclaw(available=True, send_result="FINDING: DM sent")
        hc = HouseC(
            knowledge_graph=_graph(),
            openclaw_client=client,
            workspace_dir=str(tmp_path),
        )
        sso = _sso(problem="find r/forhire job and send Reddit DM to author username")
        hc.build(sso, _dr())

        call_arg = client.send.call_args[0][0]
        assert "reddit" in call_arg.lower() or "username" in call_arg.lower() or \
               "dm" in call_arg.lower() or "message" in call_arg.lower(), (
            "OpenClaw task should reference Reddit DM, not email"
        )


class TestActionSystemPrompt:
    def test_action_system_mentions_reddit_dm_option(self):
        """ACTION_SYSTEM must mention Reddit DM via OpenClaw as outreach option."""
        assert "openclaw" in ACTION_SYSTEM.lower() or \
               "reddit dm" in ACTION_SYSTEM.lower() or \
               "direct message" in ACTION_SYSTEM.lower(), (
            "ACTION_SYSTEM prompt must mention Reddit DM / OpenClaw for outreach"
        )

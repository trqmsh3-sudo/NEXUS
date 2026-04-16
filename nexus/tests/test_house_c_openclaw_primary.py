"""Tests for House C OpenClaw-primary routing (TDD — written before implementation).

New behavior:
  - OpenClaw is PRIMARY whenever the client is present and available.
  - _needs_browser() is DELETED — no keyword gate on routing.
  - Generic tasks ("find business opportunity") use OpenClaw when available.
  - Reddit/Google-Trends tasks also go to OpenClaw — no special casing.
  - Skill library match still wins over OpenClaw.
  - Script path is pure fallback: client=None or is_available()=False.
  - Neither the browser prompt nor the script-fallback prompt prescribes
    Reddit or Google Trends as required sources.
  - Both prompts explicitly grant PROXY freedom to choose sources.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nexus.core.house_b import StructuredSpecificationObject
from nexus.core.house_c import BuildArtifact, HouseC
from nexus.core.house_d import DestructionReport
from nexus.core.knowledge_graph import KnowledgeGraph
from nexus.core.openclaw_client import OpenClawClient
from nexus.core.skill_library import Skill, SkillLibrary


# ─────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────

def _make_graph() -> KnowledgeGraph:
    return MagicMock(spec=KnowledgeGraph)


def _make_sso(problem: str = "find one business opportunity anywhere online") -> StructuredSpecificationObject:
    return StructuredSpecificationObject(
        original_input=problem,
        redefined_problem=problem,
        domain="Business Intelligence",
        constraints=[],
        success_criteria=["find a real opportunity"],
    )


def _make_dr(survived: bool = True) -> DestructionReport:
    return DestructionReport(
        target_description="test target",
        survived=survived,
        survival_score=0.85,
        cycles_survived=1,
        recommendation="PROMOTE",
        attacks=[],
    )


def _make_client(available: bool = True, send_result: str = "FINDING: opportunity found") -> MagicMock:
    client = MagicMock(spec=OpenClawClient)
    client.is_available.return_value = available
    client.send.return_value = send_result
    return client


def _passing_artifact(sso: StructuredSpecificationObject) -> BuildArtifact:
    return BuildArtifact(sso=sso, passed_validation=True, execution_proof="FINDING: result")


# ═══════════════════════════════════════════════════════════════
#  1. _needs_browser is gone
# ═══════════════════════════════════════════════════════════════

class TestNeedsBrowserDeleted:
    def test_needs_browser_method_does_not_exist(self):
        """_needs_browser() must be removed — it is no longer part of the design."""
        hc = HouseC(knowledge_graph=_make_graph())
        assert not hasattr(hc, "_needs_browser"), (
            "_needs_browser still exists on HouseC — delete it"
        )

    def test_browser_required_sites_constant_gone(self):
        """_BROWSER_REQUIRED_SITES module constant must be removed."""
        import nexus.core.house_c as house_c_module
        assert not hasattr(house_c_module, "_BROWSER_REQUIRED_SITES"), (
            "_BROWSER_REQUIRED_SITES still defined in house_c — delete it"
        )

    def test_browser_action_keywords_constant_gone(self):
        """_BROWSER_ACTION_KEYWORDS module constant must be removed."""
        import nexus.core.house_c as house_c_module
        assert not hasattr(house_c_module, "_BROWSER_ACTION_KEYWORDS"), (
            "_BROWSER_ACTION_KEYWORDS still defined in house_c — delete it"
        )


# ═══════════════════════════════════════════════════════════════
#  2. OpenClaw is primary — routing tests
# ═══════════════════════════════════════════════════════════════

class TestOpenClawPrimaryRouting:

    def test_generic_task_uses_openclaw_when_available(self, tmp_path):
        """A generic 'find business opportunity' task (no site keywords) must
        route to OpenClaw when the client is available."""
        client = _make_client()
        hc = HouseC(
            knowledge_graph=_make_graph(),
            openclaw_client=client,
            workspace_dir=str(tmp_path),
        )
        hc.build(_make_sso("find one business opportunity anywhere online"), _make_dr())
        assert client.send.called, "OpenClaw was not used for a generic business task"

    def test_reddit_task_uses_openclaw_when_available(self, tmp_path):
        """Reddit tasks must also route through OpenClaw, not the script path."""
        client = _make_client(send_result="FINDING: r/forhire post found")
        hc = HouseC(
            knowledge_graph=_make_graph(),
            openclaw_client=client,
            workspace_dir=str(tmp_path),
        )
        hc.build(_make_sso("scrape r/forhire for freelance opportunities"), _make_dr())
        assert client.send.called, "Reddit task did not route to OpenClaw"

    def test_google_trends_task_uses_openclaw_when_available(self, tmp_path):
        """Google Trends tasks must also route through OpenClaw."""
        client = _make_client(send_result="FINDING: trending niche data")
        hc = HouseC(
            knowledge_graph=_make_graph(),
            openclaw_client=client,
            workspace_dir=str(tmp_path),
        )
        hc.build(_make_sso("check Google Trends for profitable niches"), _make_dr())
        assert client.send.called, "Google Trends task did not route to OpenClaw"

    def test_script_fallback_when_openclaw_offline(self, tmp_path):
        """When is_available()=False, must fall back to script generation."""
        client = _make_client(available=False)
        hc = HouseC(
            knowledge_graph=_make_graph(),
            openclaw_client=client,
            workspace_dir=str(tmp_path),
        )
        sso = _make_sso()
        with patch.object(hc, "_generate_action_script", return_value="# NEXUS Action\nprint('FINDING: fallback')") as mock_gen, \
             patch.object(hc, "_execute_action", return_value=_passing_artifact(sso)):
            hc.build(sso, _make_dr())

        assert mock_gen.called, "Script generation was not called when OpenClaw offline"
        assert not client.send.called, "OpenClaw.send() was called despite gateway being offline"

    def test_script_fallback_when_no_openclaw_client(self, tmp_path):
        """When openclaw_client=None, must use script path without error."""
        hc = HouseC(knowledge_graph=_make_graph(), workspace_dir=str(tmp_path))
        sso = _make_sso()
        with patch.object(hc, "_generate_action_script", return_value="# NEXUS Action\nprint('FINDING: result')") as mock_gen, \
             patch.object(hc, "_execute_action", return_value=_passing_artifact(sso)):
            hc.build(sso, _make_dr())

        assert mock_gen.called, "Script generation was not called when openclaw_client=None"

    def test_skill_path_takes_priority_over_openclaw(self, tmp_path):
        """Skill library match must still win over OpenClaw."""
        client = _make_client()
        skill_lib = MagicMock(spec=SkillLibrary)
        mock_skill = MagicMock(spec=Skill)
        mock_skill.function_code = "# NEXUS Action\nprint('FINDING: from skill')"
        mock_skill.name = "test_skill"
        skill_lib.get_relevant_skills.return_value = [mock_skill]

        sso = _make_sso()
        hc = HouseC(
            knowledge_graph=_make_graph(),
            openclaw_client=client,
            skill_library=skill_lib,
            workspace_dir=str(tmp_path),
        )
        with patch.object(hc, "_execute_action", return_value=_passing_artifact(sso)):
            hc.build(sso, _make_dr())

        assert not client.send.called, "OpenClaw was called despite skill library match"


# ═══════════════════════════════════════════════════════════════
#  3. Browser prompt gives PROXY full source autonomy
# ═══════════════════════════════════════════════════════════════

class TestBrowserPromptAutonomy:

    def _capture_browser_prompt(self, tmp_path: object, problem: str = "find business opportunity") -> str:
        """Run build() with a live OpenClaw client mock and capture the task string."""
        captured: dict[str, str] = {}
        client = MagicMock(spec=OpenClawClient)
        client.is_available.return_value = True

        def capture_send(task: str, **kwargs: object) -> str:
            captured["task"] = task
            return "FINDING: captured"

        client.send.side_effect = capture_send
        hc = HouseC(
            knowledge_graph=_make_graph(),
            openclaw_client=client,
            workspace_dir=str(tmp_path),
        )
        hc.build(_make_sso(problem), _make_dr())
        return captured.get("task", "")

    def test_browser_prompt_does_not_prescribe_reddit_as_source(self, tmp_path):
        """OpenClaw task must not instruct PROXY to 'Use Reddit' as a required source."""
        prompt = self._capture_browser_prompt(tmp_path)
        assert "Use Reddit" not in prompt, (
            "Browser prompt still prescribes 'Use Reddit'"
        )

    def test_browser_prompt_does_not_prescribe_forhire(self, tmp_path):
        """OpenClaw task must not hardcode r/forhire as a required source."""
        prompt = self._capture_browser_prompt(tmp_path)
        assert "r/forhire" not in prompt, (
            "Browser prompt still hardcodes r/forhire"
        )

    def test_browser_prompt_does_not_prescribe_google_trends_rss(self, tmp_path):
        """OpenClaw task must not hardcode Google Trends RSS as a required source."""
        prompt = self._capture_browser_prompt(tmp_path)
        assert "Google Trends RSS" not in prompt, (
            "Browser prompt still hardcodes Google Trends RSS"
        )

    def test_browser_prompt_grants_source_freedom(self, tmp_path):
        """OpenClaw task must give PROXY explicit freedom to choose its own sources."""
        prompt = self._capture_browser_prompt(tmp_path)
        freedom_signals = [
            "choose", "decide", "any source", "you choose",
            "anywhere", "relevant", "your own", "pick",
        ]
        assert any(sig in prompt.lower() for sig in freedom_signals), (
            f"Browser prompt does not grant source freedom.\nPrompt:\n{prompt[:600]}"
        )

    def test_browser_prompt_includes_task_description(self, tmp_path):
        """OpenClaw task must include the SSO's redefined problem."""
        problem = "find profitable freelance niches in AI tooling"
        prompt = self._capture_browser_prompt(tmp_path, problem=problem)
        assert "profitable freelance niches in AI tooling" in prompt


# ═══════════════════════════════════════════════════════════════
#  4. Script-fallback prompt also free of hardcoded sources
# ═══════════════════════════════════════════════════════════════

class TestScriptFallbackPromptAutonomy:

    def _capture_script_prompts(self, tmp_path: object) -> tuple[str, str]:
        """Run build() with no OpenClaw client, capture user + system prompt."""
        captured: dict[str, str] = {}
        hc = HouseC(knowledge_graph=_make_graph(), workspace_dir=str(tmp_path))
        sso = _make_sso()

        def capture_llm(system: str, user: str, label: str) -> str:
            if label == "generate_action_script":
                captured["user"] = user
                captured["system"] = system
            return "# NEXUS Action\nprint('FINDING: result')"

        with patch.object(hc, "_call_llm", side_effect=capture_llm), \
             patch.object(hc, "_execute_action", return_value=_passing_artifact(sso)):
            hc.build(sso, _make_dr())

        return captured.get("user", ""), captured.get("system", "")

    def test_script_user_prompt_does_not_prescribe_reddit(self, tmp_path):
        """Script generation user prompt must not hardcode Reddit as required source."""
        user_prompt, _ = self._capture_script_prompts(tmp_path)
        assert "r/entrepreneur, r/forhire" not in user_prompt, (
            "Script user prompt still prescribes Reddit subreddits"
        )
        assert "Use Reddit" not in user_prompt, (
            "Script user prompt still says 'Use Reddit'"
        )

    def test_script_user_prompt_does_not_prescribe_google_trends_rss(self, tmp_path):
        """Script generation user prompt must not hardcode Google Trends RSS."""
        user_prompt, _ = self._capture_script_prompts(tmp_path)
        assert "Google Trends RSS" not in user_prompt, (
            "Script user prompt still hardcodes Google Trends RSS"
        )

    def test_script_system_prompt_does_not_mandate_reddit_filter(self, tmp_path):
        """ACTION_SYSTEM must not mandate Reddit r/forhire keyword filtering."""
        _, system_prompt = self._capture_script_prompts(tmp_path)
        assert "Reddit r/forhire keyword filter" not in system_prompt, (
            "ACTION_SYSTEM still mandates Reddit r/forhire keyword filtering"
        )

"""Tests for scraper threshold fix.

The LLM was generating scripts with a >=10 guard:
    if len(findings) >= 10:
        for f in findings: print(f)
    else:
        print(f"NO_DATA: Only {len(findings)} relevant posts found")

This silences valid results when fewer than 10 posts match.
The fix: ACTION_SYSTEM must explicitly forbid minimum-count guards and
instruct the script to print any finding (minimum 1).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from nexus.core.house_c import ACTION_SYSTEM, HouseC
from nexus.core.house_b import StructuredSpecificationObject
from nexus.core.knowledge_graph import KnowledgeGraph
from unittest.mock import MagicMock


def _sso(problem: str = "find jobs on r/forhire") -> StructuredSpecificationObject:
    return StructuredSpecificationObject(
        original_input=problem,
        redefined_problem=problem,
        domain="freelance",
        constraints=[],
        success_criteria=["find at least one job"],
    )


def _hc(tmp_path) -> HouseC:
    return HouseC(knowledge_graph=MagicMock(spec=KnowledgeGraph), workspace_dir=str(tmp_path))


# ── ACTION_SYSTEM constant ─────────────────────────────────────────────────────

class TestActionSystemThresholdGuard:
    def test_action_system_explicitly_says_1_result_is_enough(self):
        """Must use phrasing like 'even 1' or 'at least 1' to override SSO success criteria."""
        low = ACTION_SYSTEM.lower()
        assert "even 1" in low or "at least 1 result" in low or "minimum of 1" in low or "1 result" in low, (
            "ACTION_SYSTEM must explicitly say 1 result is sufficient — "
            "the LLM uses SSO success criteria ('at least 10') otherwise"
        )

    def test_action_system_explicitly_forbids_count_gating(self):
        """Must contain explicit 'do not' or 'never' about minimum-count guards."""
        low = ACTION_SYSTEM.lower()
        assert (
            "do not add" in low
            or "do not gate" in low
            or "do not require" in low
            or "never require" in low
            or "no minimum count" in low
            or "not gate" in low
        ), (
            "ACTION_SYSTEM must explicitly forbid 'if len(findings) >= N' guards"
        )

    def test_action_system_no_data_only_when_zero(self):
        """NO_DATA must be reserved for zero results, not 'fewer than N'."""
        low = ACTION_SYSTEM.lower()
        assert "zero" in low or "0 result" in low or "no posts found" in low or "no results" in low, (
            "ACTION_SYSTEM must clarify NO_DATA is only for zero results"
        )


# ── Generated prompt ────────────────────────────────────────────────────────────

class TestGeneratedPromptThreshold:
    def test_user_prompt_explicitly_says_1_result_is_enough(self, tmp_path):
        hc = _hc(tmp_path)
        captured = {}

        def fake_llm(self_inner, system, user, label):
            captured["user"] = user
            return "# NEXUS Action\nprint('FINDING: test')"

        with patch.object(HouseC, "_call_llm", fake_llm), \
             patch.object(HouseC, "_research_market_rates", return_value={
                 "market_low": 30.0, "market_high": 90.0,
                 "competitive_quote": 55.0, "sample_size": 0,
                 "currency": "USD", "context": "no data",
             }):
            hc._generate_action_script(_sso("find jobs on r/forhire"))

        prompt = captured.get("user", "").lower()
        assert "even 1" in prompt or "at least 1 result" in prompt or "1 result" in prompt, (
            "User prompt must explicitly say 1 result is enough"
        )

    def test_system_prompt_forbids_count_gating(self, tmp_path):
        hc = _hc(tmp_path)
        captured = {}

        def fake_llm(self_inner, system, user, label):
            captured["system"] = system
            return "# NEXUS Action\nprint('FINDING: test')"

        with patch.object(HouseC, "_call_llm", fake_llm), \
             patch.object(HouseC, "_research_market_rates", return_value={
                 "market_low": 30.0, "market_high": 90.0,
                 "competitive_quote": 55.0, "sample_size": 0,
                 "currency": "USD", "context": "no data",
             }):
            hc._generate_action_script(_sso())

        system = captured.get("system", "").lower()
        assert (
            "do not add" in system
            or "do not gate" in system
            or "do not require" in system
            or "never require" in system
            or "no minimum count" in system
        ), "System prompt must explicitly forbid minimum-count guards"

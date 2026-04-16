"""Tests for House B concrete-output rule (TDD — written before implementation).

Rule: When the input is a simple action task ("find a gig", "search for X"),
the redefined_problem must stay concrete and actionable — never a system design.

Coverage:
  1. HOUSE_B_SYSTEM prompt contains concrete-output instructions
  2. HouseB._is_system_design() detects system-design language
  3. redefine() falls back to original input when LLM returns system design
  4. Concrete LLM responses pass through unchanged
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from nexus.core.house_b import HOUSE_B_SYSTEM, HouseB, StructuredSpecificationObject
from nexus.core.knowledge_graph import KnowledgeGraph


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def _make_graph() -> KnowledgeGraph:
    kg = MagicMock(spec=KnowledgeGraph)
    kg.beliefs_snapshot.return_value = []
    kg.query_domain.return_value = []
    return kg


def _llm_json(**kwargs) -> str:
    """Build a minimal valid House B JSON response."""
    base = {
        "redefined_problem": "Find one paid gig on Upwork",
        "assumptions": [],
        "constraints": [],
        "success_criteria": ["Find at least one listing", "Listing has a stated rate", "Source is accessible"],
        "required_inputs": [],
        "expected_outputs": [],
        "domain": "Freelance",
        "confidence": 0.85,
    }
    base.update(kwargs)
    return json.dumps(base)


def _make_hb(llm_response: str) -> HouseB:
    hb = HouseB(knowledge_graph=_make_graph())
    hb._call_llm = MagicMock(return_value=llm_response)
    return hb


# ═══════════════════════════════════════════════════════════════
#  1. HOUSE_B_SYSTEM prompt contains concrete-output instructions
# ═══════════════════════════════════════════════════════════════

class TestHouseBSystemPrompt:

    def test_system_prompt_instructs_concrete_output(self):
        """HOUSE_B_SYSTEM must explicitly tell the LLM to keep output concrete and actionable."""
        lower = HOUSE_B_SYSTEM.lower()
        assert "concrete" in lower or "actionable" in lower, (
            "HOUSE_B_SYSTEM does not instruct concrete/actionable output"
        )

    def test_system_prompt_forbids_system_design(self):
        """HOUSE_B_SYSTEM must explicitly forbid turning tasks into system designs."""
        lower = HOUSE_B_SYSTEM.lower()
        forbids = any(phrase in lower for phrase in [
            "do not design", "not a system", "no system design",
            "avoid system design", "never design",
        ])
        assert forbids, (
            "HOUSE_B_SYSTEM does not forbid system-design responses"
        )

    def test_system_prompt_requires_action_verb(self):
        """HOUSE_B_SYSTEM must instruct the redefined_problem to start with an action verb."""
        lower = HOUSE_B_SYSTEM.lower()
        assert "action verb" in lower or "verb" in lower, (
            "HOUSE_B_SYSTEM does not require an action verb at the start of redefined_problem"
        )

    def test_system_prompt_includes_good_example(self):
        """HOUSE_B_SYSTEM must include a concrete example to anchor the LLM."""
        assert "Find" in HOUSE_B_SYSTEM or "Search" in HOUSE_B_SYSTEM or "Scrape" in HOUSE_B_SYSTEM, (
            "HOUSE_B_SYSTEM has no concrete example of the expected output format"
        )


# ═══════════════════════════════════════════════════════════════
#  2. HouseB._is_system_design() guard
# ═══════════════════════════════════════════════════════════════

class TestIsSystemDesign:

    def test_method_exists_as_static(self):
        """_is_system_design must exist as a static method on HouseB."""
        assert callable(getattr(HouseB, "_is_system_design", None)), (
            "_is_system_design does not exist on HouseB"
        )

    def test_detects_design_a_systematic_process(self):
        text = "Design a systematic process to identify and validate paid opportunities"
        assert HouseB._is_system_design(text) is True

    def test_detects_architecture_language(self):
        text = "Build an architecture for scalable opportunity monitoring"
        assert HouseB._is_system_design(text) is True

    def test_detects_tiered_verification(self):
        text = "Create a tiered verification framework for online opportunities"
        assert HouseB._is_system_design(text) is True

    def test_detects_framework_language(self):
        text = "Implement a framework with continuous adversarial adaptation"
        assert HouseB._is_system_design(text) is True

    def test_detects_scalable_system_language(self):
        text = "Develop a scalable monitoring system with robust safety protocols"
        assert HouseB._is_system_design(text) is True

    def test_detects_design_a_process(self):
        text = "Design a process to systematically evaluate paid opportunities"
        assert HouseB._is_system_design(text) is True

    def test_detects_systematically_evaluate(self):
        text = "Design a process to systematically evaluate and document online paid opportunities"
        assert HouseB._is_system_design(text) is True

    def test_detects_over_length_as_system_design(self):
        """Anything over 160 chars is almost certainly over-engineered."""
        text = "Find one paid gig " + ("x" * 150)  # clearly too long
        assert HouseB._is_system_design(text) is True

    def test_concrete_find_task_is_not_system_design(self):
        text = "Find one paid freelance gig on Upwork or Fiverr"
        assert HouseB._is_system_design(text) is False

    def test_concrete_search_task_is_not_system_design(self):
        text = "Search HackerNews job board for remote Python contracts"
        assert HouseB._is_system_design(text) is False

    def test_concrete_scrape_task_is_not_system_design(self):
        text = "Scrape r/forhire for posts offering payment above $30/hr"
        assert HouseB._is_system_design(text) is False

    def test_case_insensitive(self):
        text = "DESIGN A SYSTEMATIC PROCESS for identifying opportunities"
        assert HouseB._is_system_design(text) is True


# ═══════════════════════════════════════════════════════════════
#  3. redefine() falls back when LLM returns system design
# ═══════════════════════════════════════════════════════════════

class TestRedefineSystemDesignFallback:

    def test_system_design_redefined_problem_is_replaced(self):
        """When LLM returns system-design language, redefined_problem must be sanitized."""
        system_design_response = _llm_json(
            redefined_problem=(
                "Design a systematic process to identify and validate legitimate paid "
                "opportunities with tiered verification mechanisms and scalable monitoring"
            )
        )
        hb = _make_hb(system_design_response)
        sso = hb.redefine("find one paid gig online")
        assert "systematic process" not in sso.redefined_problem.lower(), (
            "System-design language survived into redefined_problem"
        )
        assert "tiered verification" not in sso.redefined_problem.lower()
        assert "scalable monitoring" not in sso.redefined_problem.lower()

    def test_fallback_stays_concrete(self):
        """After sanitization, the redefined_problem must still be actionable."""
        system_design_response = _llm_json(
            redefined_problem="Design a framework for scalable opportunity architecture"
        )
        hb = _make_hb(system_design_response)
        sso = hb.redefine("find one paid gig online")
        # Must be short and not empty
        assert len(sso.redefined_problem) > 0
        assert len(sso.redefined_problem) < 200, (
            f"Fallback redefined_problem is too long ({len(sso.redefined_problem)} chars)"
        )

    def test_original_input_used_as_fallback(self):
        """Fallback redefined_problem must derive from or equal the original input."""
        original = "find one paid gig online"
        system_design_response = _llm_json(
            redefined_problem="Design a systematic process to discover paid gigs"
        )
        hb = _make_hb(system_design_response)
        sso = hb.redefine(original)
        # The fallback should be grounded in the original input
        assert original.lower() in sso.redefined_problem.lower() or \
               any(word in sso.redefined_problem.lower() for word in ["find", "gig", "paid", "online"]), (
            f"Fallback does not relate to original input. Got: {sso.redefined_problem!r}"
        )


# ═══════════════════════════════════════════════════════════════
#  4. Concrete LLM responses pass through unchanged
# ═══════════════════════════════════════════════════════════════

class TestConcreteResponsePassthrough:

    def test_concrete_problem_is_not_modified(self):
        """A concrete, action-oriented redefined_problem must pass through exactly."""
        concrete = "Find one paid freelance contract posted on Upwork or HackerNews today"
        hb = _make_hb(_llm_json(redefined_problem=concrete))
        sso = hb.redefine("find one paid gig")
        assert sso.redefined_problem == concrete

    def test_short_direct_problem_is_not_modified(self):
        """A short direct task must not be altered."""
        direct = "Search Fiverr for buyers posting Python automation requests"
        hb = _make_hb(_llm_json(redefined_problem=direct))
        sso = hb.redefine("look for gigs on Fiverr")
        assert sso.redefined_problem == direct

    def test_sso_original_input_always_preserved(self):
        """original_input on the SSO must always equal what was passed to redefine()."""
        user_input = "find one real paid opportunity anywhere"
        hb = _make_hb(_llm_json())
        sso = hb.redefine(user_input)
        assert sso.original_input == user_input

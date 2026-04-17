"""Tests for BeliefQualityFilter (TDD — written before implementation).

Rule: Before storing any belief, ask the LLM:
  "Will this belief help make a future decision?"
  - Store if yes (LLM confidence >= 0.7)
  - Discard if no, or confidence < 0.7

Coverage:
  1. BeliefQualityFilter class exists and has is_actionable()
  2. Status/operational messages are discarded
  3. Actionable findings (prices, patterns, opportunities) are stored
  4. System metrics are discarded
  5. Business insights are stored
  6. LLM confidence < 0.7 → discard regardless of decision
  7. LLM failure → discard by default (safe fallback)
  8. knowledge_graph.add_belief() honours the filter when one is set
  9. knowledge_graph.add_belief() works unchanged when no filter is set
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from nexus.core.belief_certificate import BeliefCertificate
from nexus.core.belief_quality_filter import BeliefQualityFilter
from nexus.core.knowledge_graph import KnowledgeGraph
from nexus.core.model_router import ModelRouter


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def _make_router(llm_json: str) -> ModelRouter:
    router = MagicMock(spec=ModelRouter)
    router.complete.return_value = llm_json
    return router


def _llm_verdict(decision: str, confidence: float, reason: str = "") -> str:
    return json.dumps({
        "decision": decision,
        "confidence": confidence,
        "reason": reason or f"Test reason for {decision}",
    })


def _belief(claim: str, proof: str = "print('ok')") -> BeliefCertificate:
    return BeliefCertificate(
        claim=claim,
        source="test",
        confidence=0.8,
        domain="Test",
        executable_proof=proof,
    )


# ═══════════════════════════════════════════════════════════════
#  1. Class and interface
# ═══════════════════════════════════════════════════════════════

class TestBeliefQualityFilterInterface:

    def test_class_is_importable(self):
        assert BeliefQualityFilter is not None

    def test_has_is_actionable_method(self):
        router = _make_router(_llm_verdict("store", 0.9))
        f = BeliefQualityFilter(router=router)
        assert callable(getattr(f, "is_actionable", None))

    def test_accepts_router_in_constructor(self):
        router = _make_router(_llm_verdict("store", 0.9))
        f = BeliefQualityFilter(router=router)
        assert f.router is router


# ═══════════════════════════════════════════════════════════════
#  2. Status / operational messages → discard
# ═══════════════════════════════════════════════════════════════

class TestStatusMessagesDiscarded:

    def test_browser_operational_discarded(self):
        router = _make_router(_llm_verdict("discard", 0.95, "status message"))
        f = BeliefQualityFilter(router=router)
        result = f.is_actionable(_belief("Browser operational — loaded: Google"))
        assert result is False

    def test_system_loaded_discarded(self):
        router = _make_router(_llm_verdict("discard", 0.90))
        f = BeliefQualityFilter(router=router)
        result = f.is_actionable(_belief("System loaded successfully"))
        assert result is False

    def test_connected_message_discarded(self):
        router = _make_router(_llm_verdict("discard", 0.88))
        f = BeliefQualityFilter(router=router)
        result = f.is_actionable(_belief("Database connected: YES"))
        assert result is False

    def test_operational_status_discarded(self):
        router = _make_router(_llm_verdict("discard", 0.85))
        f = BeliefQualityFilter(router=router)
        result = f.is_actionable(_belief("OpenClaw gateway operational"))
        assert result is False


# ═══════════════════════════════════════════════════════════════
#  3. Actionable findings → store
# ═══════════════════════════════════════════════════════════════

class TestActionableFindingsStored:

    def test_price_finding_stored(self):
        router = _make_router(_llm_verdict("store", 0.92))
        f = BeliefQualityFilter(router=router)
        result = f.is_actionable(_belief("Upwork Python contracts average $120/hr in April 2026"))
        assert result is True

    def test_pattern_finding_stored(self):
        router = _make_router(_llm_verdict("store", 0.88))
        f = BeliefQualityFilter(router=router)
        result = f.is_actionable(_belief("AI automation gigs on Fiverr have 40% fewer applicants than design gigs"))
        assert result is True

    def test_obstacle_finding_stored(self):
        router = _make_router(_llm_verdict("store", 0.85))
        f = BeliefQualityFilter(router=router)
        result = f.is_actionable(_belief("Upwork blocks scraping — requires login to view full job listings"))
        assert result is True

    def test_opportunity_finding_stored(self):
        router = _make_router(_llm_verdict("store", 0.91))
        f = BeliefQualityFilter(router=router)
        result = f.is_actionable(_belief("HackerNews 'Who is hiring' thread has 300+ active listings in April"))
        assert result is True


# ═══════════════════════════════════════════════════════════════
#  4. System metrics → discard
# ═══════════════════════════════════════════════════════════════

class TestSystemMetricsDiscarded:

    def test_memory_metric_discarded(self):
        router = _make_router(_llm_verdict("discard", 0.93))
        f = BeliefQualityFilter(router=router)
        result = f.is_actionable(_belief("RSS memory: 193 MB at cycle end"))
        assert result is False

    def test_cycle_time_metric_discarded(self):
        router = _make_router(_llm_verdict("discard", 0.87))
        f = BeliefQualityFilter(router=router)
        result = f.is_actionable(_belief("Average cycle time: 91 seconds"))
        assert result is False

    def test_api_count_metric_discarded(self):
        router = _make_router(_llm_verdict("discard", 0.89))
        f = BeliefQualityFilter(router=router)
        result = f.is_actionable(_belief("14 total API calls made today"))
        assert result is False


# ═══════════════════════════════════════════════════════════════
#  5. Business insights → store
# ═══════════════════════════════════════════════════════════════

class TestBusinessInsightsStored:

    def test_market_insight_stored(self):
        router = _make_router(_llm_verdict("store", 0.90))
        f = BeliefQualityFilter(router=router)
        result = f.is_actionable(_belief("No-code tool demand on Product Hunt doubled in Q1 2026"))
        assert result is True

    def test_client_behaviour_stored(self):
        router = _make_router(_llm_verdict("store", 0.87))
        f = BeliefQualityFilter(router=router)
        result = f.is_actionable(_belief("Clients posting same-week budgets pay 25% more than those with 2-week timelines"))
        assert result is True


# ═══════════════════════════════════════════════════════════════
#  6. Low LLM confidence → discard regardless of decision
# ═══════════════════════════════════════════════════════════════

class TestLowConfidenceDiscarded:

    def test_store_decision_with_low_confidence_discarded(self):
        """Even if LLM says 'store', confidence < 0.7 means discard."""
        router = _make_router(_llm_verdict("store", 0.65))
        f = BeliefQualityFilter(router=router)
        result = f.is_actionable(_belief("Some potentially useful market insight"))
        assert result is False

    def test_exactly_07_confidence_is_accepted(self):
        router = _make_router(_llm_verdict("store", 0.70))
        f = BeliefQualityFilter(router=router)
        result = f.is_actionable(_belief("Python automation demand rising on Upwork"))
        assert result is True

    def test_below_07_discard_decision_also_discarded(self):
        router = _make_router(_llm_verdict("discard", 0.60))
        f = BeliefQualityFilter(router=router)
        result = f.is_actionable(_belief("Some claim"))
        assert result is False


# ═══════════════════════════════════════════════════════════════
#  7. LLM failure → discard by default
# ═══════════════════════════════════════════════════════════════

class TestLLMFailureSafeDiscard:

    def test_llm_returns_empty_string_discards(self):
        router = _make_router("")
        f = BeliefQualityFilter(router=router)
        result = f.is_actionable(_belief("Some claim"))
        assert result is False

    def test_llm_returns_invalid_json_discards(self):
        router = _make_router("not valid json {{{")
        f = BeliefQualityFilter(router=router)
        result = f.is_actionable(_belief("Some claim"))
        assert result is False

    def test_llm_exception_discards(self):
        router = MagicMock(spec=ModelRouter)
        router.complete.side_effect = RuntimeError("LLM unavailable")
        f = BeliefQualityFilter(router=router)
        result = f.is_actionable(_belief("Some claim"))
        assert result is False

    def test_llm_missing_fields_discards(self):
        """JSON without required fields → discard."""
        router = _make_router('{"reason": "missing decision and confidence"}')
        f = BeliefQualityFilter(router=router)
        result = f.is_actionable(_belief("Some claim"))
        assert result is False


# ═══════════════════════════════════════════════════════════════
#  8. KnowledgeGraph honours the filter
# ═══════════════════════════════════════════════════════════════

class TestKnowledgeGraphHonoursFilter:

    def _make_kg(self, filter_result: bool) -> KnowledgeGraph:
        mock_filter = MagicMock(spec=BeliefQualityFilter)
        mock_filter.is_actionable.return_value = filter_result
        return KnowledgeGraph(quality_filter=mock_filter)

    def test_belief_stored_when_filter_approves(self):
        kg = self._make_kg(filter_result=True)
        b = _belief("Python automation demand rising on Upwork")
        result = kg.add_belief(b)
        assert result is True

    def test_belief_discarded_when_filter_rejects(self):
        kg = self._make_kg(filter_result=False)
        b = _belief("Browser operational — loaded: Google")
        result = kg.add_belief(b)
        assert result is False

    def test_filter_is_called_with_the_belief(self):
        mock_filter = MagicMock(spec=BeliefQualityFilter)
        mock_filter.is_actionable.return_value = True
        kg = KnowledgeGraph(quality_filter=mock_filter)
        b = _belief("Some claim")
        kg.add_belief(b)
        mock_filter.is_actionable.assert_called_once_with(b)

    def test_rejected_belief_not_in_graph(self):
        kg = self._make_kg(filter_result=False)
        b = _belief("Browser operational — loaded: Google")
        kg.add_belief(b)
        assert b.claim not in kg.beliefs


# ═══════════════════════════════════════════════════════════════
#  9. Backward compatibility — no filter set
# ═══════════════════════════════════════════════════════════════

class TestNoFilterBackwardCompatible:

    def test_add_belief_works_without_filter(self):
        """KnowledgeGraph with no quality_filter stores beliefs as before."""
        kg = KnowledgeGraph()
        b = _belief("Clean code is better than clever code")
        result = kg.add_belief(b)
        assert result is True

    def test_default_quality_filter_is_none(self):
        kg = KnowledgeGraph()
        assert kg.quality_filter is None

"""Tests for House Omega — The Governor.

All LLM and subprocess calls are mocked so tests run without Ollama
or real file I/O.  Tests cover the full NEXUS cycle, sleep cycles,
external injection, health metrics, cycle history, and all Iron Laws.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from nexus.core.belief_certificate import BeliefCertificate
from nexus.core.house_b import HouseB, MinorityReport, StructuredSpecificationObject
from nexus.core.house_c import BuildArtifact, BuildResult, HouseC
from nexus.core.house_d import AttackResult, DestructionReport, HouseD
from nexus.core.house_omega import (
    CycleResult,
    HouseOmega,
    SystemHealth,
)
from nexus.core.knowledge_graph import KnowledgeGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cert(
    claim: str = "test belief",
    confidence: float = 0.9,
    domain: str = "Testing",
) -> BeliefCertificate:
    return BeliefCertificate(
        claim=claim,
        source="unit-test",
        confidence=confidence,
        decay_rate=0.0,
        created_at=datetime.now(timezone.utc),
        last_verified=datetime.now(timezone.utc),
        executable_proof="assert True",
        domain=domain,
    )


def _make_sso(problem: str = "test problem") -> StructuredSpecificationObject:
    return StructuredSpecificationObject(
        original_input="raw input",
        redefined_problem=problem,
        assumptions=["A"],
        constraints=["C"],
        success_criteria=["S"],
        domain="Testing",
        confidence=0.85,
    )


def _survived_report() -> DestructionReport:
    return DestructionReport(
        target_description="test",
        survived=True,
        survival_score=0.9,
        cycles_survived=3,
        recommendation="PROMOTE",
    )


def _failed_report() -> DestructionReport:
    return DestructionReport(
        target_description="test",
        survived=False,
        survival_score=0.2,
        cycles_survived=3,
        recommendation="REJECT",
    )


def _revise_report() -> DestructionReport:
    return DestructionReport(
        target_description="test",
        survived=False,
        survival_score=0.5,
        cycles_survived=3,
        recommendation="REVISE",
    )


def _good_artifact(sso: StructuredSpecificationObject | None = None) -> BuildArtifact:
    return BuildArtifact(
        sso=sso or _make_sso(),
        code="def f(): return 1",
        tests="def test_f(): assert f() == 1",
        passed_validation=True,
        execution_proof="1 passed",
    )


def _good_build_result(
    sso: StructuredSpecificationObject | None = None,
) -> BuildResult:
    return BuildResult(
        artifact=_good_artifact(sso),
        success=True,
        house_d_report=_survived_report(),
        ready_for_house_a=True,
    )


def _failed_build_result() -> BuildResult:
    return BuildResult(
        artifact=BuildArtifact(passed_validation=False),
        success=False,
        house_d_report=_survived_report(),
        ready_for_house_a=False,
    )


def _make_omega(
    sleep_interval: int = 50,
) -> HouseOmega:
    """Build a HouseOmega with real KnowledgeGraph and mocked Houses."""
    kg = KnowledgeGraph()
    return HouseOmega(
        knowledge_graph=kg,
        house_b=MagicMock(spec=HouseB),
        house_c=MagicMock(spec=HouseC),
        house_d=MagicMock(spec=HouseD),
        sleep_cycle_interval=sleep_interval,
    )


# ---------------------------------------------------------------------------
# CycleResult dataclass
# ---------------------------------------------------------------------------

class TestCycleResult:
    """Tests for the CycleResult dataclass."""

    def test_defaults(self) -> None:
        cr = CycleResult()
        assert cr.success is False
        assert cr.belief_added is False
        assert cr.failure_reason is None
        assert cr.refinement_attempts == 0

    def test_to_dict(self) -> None:
        cr = CycleResult(user_input="hi", success=True, refinement_attempts=2)
        d = cr.to_dict()
        assert d["user_input"] == "hi"
        assert d["success"] is True
        assert d["refinement_attempts"] == 2
        assert "cycle_id" in d


# ---------------------------------------------------------------------------
# SystemHealth dataclass
# ---------------------------------------------------------------------------

class TestSystemHealth:
    """Tests for the SystemHealth dataclass."""

    def test_defaults(self) -> None:
        sh = SystemHealth()
        assert sh.total_cycles == 0
        assert sh.system_score == 0.0

    def test_to_dict(self) -> None:
        sh = SystemHealth(total_cycles=5, system_score=0.8)
        d = sh.to_dict()
        assert d["total_cycles"] == 5
        assert d["system_score"] == 0.8


# ---------------------------------------------------------------------------
# HouseOmega.run — full success path
# ---------------------------------------------------------------------------

class TestOmegaRunSuccess:
    """Tests for the happy path through the main loop."""

    @patch.object(HouseOmega, "_persist_history")
    def test_full_cycle_success(self, mock_persist: MagicMock) -> None:
        omega = _make_omega()
        sso = _make_sso()
        omega.house_b.redefine.return_value = sso
        omega.house_d.attack_sso.return_value = _survived_report()
        omega.house_d.attack_belief.return_value = _survived_report()
        omega.house_c.build.return_value = _good_build_result(sso)
        omega.house_c.to_belief_certificate.return_value = _make_cert()

        result = omega.run("build something")

        assert result.success is True
        assert result.belief_added is True
        assert result.sso is sso
        assert result.failure_reason is None
        assert result.refinement_attempts == 0
        assert omega.cycle_count == 1

    @patch.object(HouseOmega, "_persist_history")
    def test_belief_injected_into_graph(self, mock_persist: MagicMock) -> None:
        omega = _make_omega()
        sso = _make_sso()
        cert = _make_cert(claim="new knowledge")
        omega.house_b.redefine.return_value = sso
        omega.house_d.attack_sso.return_value = _survived_report()
        omega.house_d.attack_belief.return_value = _survived_report()
        omega.house_c.build.return_value = _good_build_result(sso)
        omega.house_c.to_belief_certificate.return_value = cert

        omega.run("test")

        assert "new knowledge" in omega.knowledge_graph


# ---------------------------------------------------------------------------
# HouseOmega.run — failure paths
# ---------------------------------------------------------------------------

class TestOmegaRunFailures:
    """Tests for failure scenarios in the main loop."""

    @patch.object(HouseOmega, "_persist_history")
    def test_sso_destroyed_by_house_d(self, mock_persist: MagicMock) -> None:
        omega = _make_omega()
        omega.house_b.redefine.return_value = _make_sso()
        omega.house_d.attack_sso.return_value = _failed_report()

        result = omega.run("input")

        assert result.success is False
        assert "destroyed by House D" in (result.failure_reason or "")
        assert result.build_result is None

    @patch.object(HouseOmega, "_persist_history")
    def test_build_fails_validation(self, mock_persist: MagicMock) -> None:
        omega = _make_omega()
        sso = _make_sso()
        omega.house_b.redefine.return_value = sso
        omega.house_d.attack_sso.return_value = _survived_report()
        omega.house_c.build.return_value = _failed_build_result()

        result = omega.run("input")

        assert result.success is False
        assert "build failed" in (result.failure_reason or "")

    @patch.object(HouseOmega, "_persist_history")
    def test_belief_destroyed_by_house_d(self, mock_persist: MagicMock) -> None:
        omega = _make_omega()
        sso = _make_sso()
        omega.house_b.redefine.return_value = sso
        omega.house_d.attack_sso.return_value = _survived_report()
        omega.house_c.build.return_value = _good_build_result(sso)
        omega.house_c.to_belief_certificate.return_value = _make_cert()
        omega.house_d.attack_belief.return_value = _failed_report()

        result = omega.run("input")

        assert result.success is False
        assert "Belief destroyed" in (result.failure_reason or "")

    @patch.object(HouseOmega, "_persist_history")
    def test_exception_logged_gracefully(self, mock_persist: MagicMock) -> None:
        omega = _make_omega()
        omega.house_b.redefine.side_effect = RuntimeError("LLM down")

        result = omega.run("input")

        assert result.success is False
        assert "RuntimeError" in (result.failure_reason or "")
        assert omega.cycle_count == 1


# ---------------------------------------------------------------------------
# HouseOmega.run — iterative refinement
# ---------------------------------------------------------------------------

class TestOmegaRefinement:
    """Tests for the iterative refinement loop."""

    @patch.object(HouseOmega, "_persist_history")
    def test_refine_once_then_survive(self, mock_persist: MagicMock) -> None:
        """SSO fails with REVISE, gets refined once, then survives."""
        omega = _make_omega()
        sso_v1 = _make_sso("v1")
        sso_v2 = _make_sso("v2 refined")
        omega.house_b.redefine.return_value = sso_v1
        omega.house_b.refine.return_value = sso_v2
        omega.house_d.attack_sso.side_effect = [
            _revise_report(),
            _survived_report(),
        ]
        omega.house_d.attack_belief.return_value = _survived_report()
        omega.house_c.build.return_value = _good_build_result(sso_v2)
        omega.house_c.to_belief_certificate.return_value = _make_cert()

        result = omega.run("test")

        assert result.success is True
        assert result.refinement_attempts == 1
        assert result.sso is sso_v2
        omega.house_b.refine.assert_called_once_with(sso_v1, _revise_report())

    @patch.object(HouseOmega, "_persist_history")
    def test_refine_twice_then_survive(self, mock_persist: MagicMock) -> None:
        """SSO needs two refinements before surviving."""
        omega = _make_omega()
        sso_v1 = _make_sso("v1")
        sso_v2 = _make_sso("v2")
        sso_v3 = _make_sso("v3 final")
        omega.house_b.redefine.return_value = sso_v1
        omega.house_b.refine.side_effect = [sso_v2, sso_v3]
        omega.house_d.attack_sso.side_effect = [
            _revise_report(),
            _revise_report(),
            _survived_report(),
        ]
        omega.house_d.attack_belief.return_value = _survived_report()
        omega.house_c.build.return_value = _good_build_result(sso_v3)
        omega.house_c.to_belief_certificate.return_value = _make_cert()

        result = omega.run("test")

        assert result.success is True
        assert result.refinement_attempts == 2
        assert result.sso is sso_v3

    @patch.object(HouseOmega, "_persist_history")
    def test_max_refinements_exhausted(self, mock_persist: MagicMock) -> None:
        """SSO keeps getting REVISE but hits the max_refinements cap."""
        omega = _make_omega()
        omega.max_refinements = 3
        omega.house_b.redefine.return_value = _make_sso("v1")
        omega.house_b.refine.return_value = _make_sso("still bad")
        omega.house_d.attack_sso.return_value = _revise_report()

        result = omega.run("test")

        assert result.success is False
        assert result.refinement_attempts == 3
        assert omega.house_b.refine.call_count == 3
        assert omega.house_d.attack_sso.call_count == 4
        assert "after 3 refinement(s)" in (result.failure_reason or "")

    @patch.object(HouseOmega, "_persist_history")
    def test_reject_skips_refinement(self, mock_persist: MagicMock) -> None:
        """SSO destroyed with REJECT — no refinement attempted."""
        omega = _make_omega()
        omega.house_b.redefine.return_value = _make_sso()
        omega.house_d.attack_sso.return_value = _failed_report()

        result = omega.run("test")

        assert result.success is False
        assert result.refinement_attempts == 0
        omega.house_b.refine.assert_not_called()

    @patch.object(HouseOmega, "_persist_history")
    def test_revise_then_reject_stops(self, mock_persist: MagicMock) -> None:
        """First attempt gets REVISE, refinement gets REJECT — stops."""
        omega = _make_omega()
        omega.house_b.redefine.return_value = _make_sso("v1")
        omega.house_b.refine.return_value = _make_sso("v2")
        omega.house_d.attack_sso.side_effect = [
            _revise_report(),
            _failed_report(),
        ]

        result = omega.run("test")

        assert result.success is False
        assert result.refinement_attempts == 1
        assert omega.house_b.refine.call_count == 1

    @patch.object(HouseOmega, "_persist_history")
    def test_zero_max_refinements_disables_loop(
        self, mock_persist: MagicMock,
    ) -> None:
        """With max_refinements=0, REVISE is treated as final rejection."""
        omega = _make_omega()
        omega.max_refinements = 0
        omega.house_b.redefine.return_value = _make_sso()
        omega.house_d.attack_sso.return_value = _revise_report()

        result = omega.run("test")

        assert result.success is False
        assert result.refinement_attempts == 0
        omega.house_b.refine.assert_not_called()


# ---------------------------------------------------------------------------
# HouseOmega.run — cycle counting & sleep trigger
# ---------------------------------------------------------------------------

class TestOmegaCycleSleep:
    """Tests for automatic sleep cycle triggering."""

    @patch.object(HouseOmega, "_persist_history")
    @patch.object(HouseOmega, "run_sleep_cycle")
    def test_sleep_triggered_at_interval(
        self, mock_sleep: MagicMock, mock_persist: MagicMock,
    ) -> None:
        omega = _make_omega(sleep_interval=3)
        omega.house_b.redefine.side_effect = RuntimeError("fail fast")

        for _ in range(3):
            omega.run("x")

        assert omega.cycle_count == 3
        mock_sleep.assert_called_once()

    @patch.object(HouseOmega, "_persist_history")
    @patch.object(HouseOmega, "run_sleep_cycle")
    def test_no_sleep_before_interval(
        self, mock_sleep: MagicMock, mock_persist: MagicMock,
    ) -> None:
        omega = _make_omega(sleep_interval=10)
        omega.house_b.redefine.side_effect = RuntimeError("fail fast")

        for _ in range(5):
            omega.run("x")

        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# HouseOmega.run_sleep_cycle
# ---------------------------------------------------------------------------

class TestRunSleepCycle:
    """Tests for the sleep cycle logic."""

    @patch.object(HouseOmega, "_persist_history")
    def test_sleep_returns_summary(self, mock_persist: MagicMock) -> None:
        omega = _make_omega()
        result = omega.run_sleep_cycle()
        assert "pruned" in result
        assert "flagged" in result
        assert "contradictions" in result
        assert "duration" in result

    @patch.object(HouseOmega, "_persist_history")
    def test_sleep_updates_last_sleep(self, mock_persist: MagicMock) -> None:
        omega = _make_omega()
        assert omega._last_sleep is None
        omega.run_sleep_cycle()
        assert omega._last_sleep is not None


# ---------------------------------------------------------------------------
# HouseOmega.inject_external_knowledge
# ---------------------------------------------------------------------------

class TestInjectExternalKnowledge:
    """Tests for external knowledge injection with House D gating."""

    @patch.object(HouseOmega, "_persist_history")
    def test_survived_beliefs_added(self, mock_persist: MagicMock) -> None:
        omega = _make_omega()
        omega.house_d.attack_belief.return_value = _survived_report()

        result = omega.inject_external_knowledge([
            _make_cert(claim="ext1"),
            _make_cert(claim="ext2"),
        ])

        assert result["submitted"] == 2
        assert result["survived_d"] == 2
        assert result["added_to_a"] == 2

    @patch.object(HouseOmega, "_persist_history")
    def test_destroyed_beliefs_rejected(self, mock_persist: MagicMock) -> None:
        omega = _make_omega()
        omega.house_d.attack_belief.return_value = _failed_report()

        result = omega.inject_external_knowledge([_make_cert()])

        assert result["submitted"] == 1
        assert result["survived_d"] == 0
        assert result["rejected"] == 1

    @patch.object(HouseOmega, "_persist_history")
    def test_mixed_batch(self, mock_persist: MagicMock) -> None:
        omega = _make_omega()
        omega.house_d.attack_belief.side_effect = [
            _survived_report(),
            _failed_report(),
        ]

        result = omega.inject_external_knowledge([
            _make_cert(claim="good"),
            _make_cert(claim="bad"),
        ])

        assert result["survived_d"] == 1
        assert result["rejected"] == 1


# ---------------------------------------------------------------------------
# HouseOmega.get_health
# ---------------------------------------------------------------------------

class TestGetHealth:
    """Tests for the health snapshot."""

    @patch.object(HouseOmega, "_persist_history")
    def test_empty_system(self, mock_persist: MagicMock) -> None:
        omega = _make_omega()
        health = omega.get_health()
        assert health.total_cycles == 0
        assert health.system_score == 0.0
        assert health.total_beliefs == 0

    @patch.object(HouseOmega, "_persist_history")
    def test_after_cycles(self, mock_persist: MagicMock) -> None:
        omega = _make_omega()
        sso = _make_sso()
        omega.house_b.redefine.return_value = sso
        omega.house_d.attack_sso.return_value = _survived_report()
        omega.house_d.attack_belief.return_value = _survived_report()
        omega.house_c.build.return_value = _good_build_result(sso)
        omega.house_c.to_belief_certificate.return_value = _make_cert()

        omega.run("test1")

        omega.house_b.redefine.side_effect = RuntimeError("fail")
        omega.run("test2")

        health = omega.get_health()
        assert health.total_cycles == 2
        assert health.successful_cycles == 1
        assert health.failed_cycles == 1
        assert health.system_score == 0.5


# ---------------------------------------------------------------------------
# HouseOmega.get_cycle_history
# ---------------------------------------------------------------------------

class TestGetCycleHistory:
    """Tests for cycle history retrieval."""

    @patch.object(HouseOmega, "_persist_history")
    def test_returns_last_n(self, mock_persist: MagicMock) -> None:
        omega = _make_omega()
        omega.house_b.redefine.side_effect = RuntimeError("fast")

        for _ in range(7):
            omega.run("x")

        assert len(omega.get_cycle_history(last_n=3)) == 3
        assert len(omega.get_cycle_history(last_n=100)) == 7

    @patch.object(HouseOmega, "_persist_history")
    def test_default_last_10(self, mock_persist: MagicMock) -> None:
        omega = _make_omega()
        omega.house_b.redefine.side_effect = RuntimeError("fast")

        for _ in range(15):
            omega.run("x")

        assert len(omega.get_cycle_history()) == 10


# ---------------------------------------------------------------------------
# HouseOmega._log_cycle
# ---------------------------------------------------------------------------

class TestLogCycle:
    """Tests for cycle logging."""

    @patch.object(HouseOmega, "_persist_history")
    def test_appends_to_history(self, mock_persist: MagicMock) -> None:
        omega = _make_omega()
        result = CycleResult(user_input="test")
        omega._log_cycle(result)
        assert len(omega.cycle_history) == 1
        assert omega.cycle_history[0].user_input == "test"

    @patch.object(HouseOmega, "_persist_history")
    def test_calls_persist(self, mock_persist: MagicMock) -> None:
        omega = _make_omega()
        omega._log_cycle(CycleResult())
        mock_persist.assert_called_once()


# ---------------------------------------------------------------------------
# Iron Law enforcement
# ---------------------------------------------------------------------------

class TestIronLaws:
    """Tests verifying the Iron Laws are enforced."""

    @patch.object(HouseOmega, "_persist_history")
    def test_sso_always_passes_through_house_d(
        self, mock_persist: MagicMock,
    ) -> None:
        omega = _make_omega()
        omega.house_b.redefine.return_value = _make_sso()
        omega.house_d.attack_sso.return_value = _survived_report()
        omega.house_d.attack_belief.return_value = _survived_report()
        omega.house_c.build.return_value = _good_build_result()
        omega.house_c.to_belief_certificate.return_value = _make_cert()

        omega.run("test")

        omega.house_d.attack_sso.assert_called_once()

    @patch.object(HouseOmega, "_persist_history")
    def test_belief_always_passes_through_house_d(
        self, mock_persist: MagicMock,
    ) -> None:
        omega = _make_omega()
        sso = _make_sso()
        omega.house_b.redefine.return_value = sso
        omega.house_d.attack_sso.return_value = _survived_report()
        omega.house_d.attack_belief.return_value = _survived_report()
        omega.house_c.build.return_value = _good_build_result(sso)
        omega.house_c.to_belief_certificate.return_value = _make_cert()

        omega.run("test")

        omega.house_d.attack_belief.assert_called_once()

    @patch.object(HouseOmega, "_persist_history")
    def test_external_knowledge_passes_house_d(
        self, mock_persist: MagicMock,
    ) -> None:
        omega = _make_omega()
        omega.house_d.attack_belief.return_value = _survived_report()

        omega.inject_external_knowledge([_make_cert()])

        omega.house_d.attack_belief.assert_called_once()

    @patch.object(HouseOmega, "_persist_history")
    def test_every_cycle_logged(self, mock_persist: MagicMock) -> None:
        omega = _make_omega()
        omega.house_b.redefine.side_effect = RuntimeError("boom")

        omega.run("a")
        omega.run("b")

        assert len(omega.cycle_history) == 2

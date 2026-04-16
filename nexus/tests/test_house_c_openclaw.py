"""Tests for House C OpenClaw integration (TDD — written before implementation).

Coverage:
  1. HouseC.openclaw_client field
  2. HouseC._needs_browser() — keyword detection
  3. HouseC._execute_browser_task() — delegates to client, sets proof
  4. build() routing — uses OpenClaw when available & browser task
  5. Graceful fallback — falls back to script when OpenClaw offline or not needed
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nexus.core.house_b import StructuredSpecificationObject
from nexus.core.house_c import BuildArtifact, BuildResult, HouseC
from nexus.core.house_d import DestructionReport
from nexus.core.knowledge_graph import KnowledgeGraph
from nexus.core.openclaw_client import OpenClawClient


# ─────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────

def _make_graph() -> KnowledgeGraph:
    return MagicMock(spec=KnowledgeGraph)


def _make_sso(problem: str = "research market", domain: str = "market") -> StructuredSpecificationObject:
    return StructuredSpecificationObject(
        original_input=problem,
        redefined_problem=problem,
        domain=domain,
        constraints=[],
        success_criteria=["find data"],
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


def _make_client(available: bool = True, send_result: str = "FINDING: test result") -> MagicMock:
    client = MagicMock(spec=OpenClawClient)
    client.is_available.return_value = available
    client.send.return_value = send_result
    return client


# ═══════════════════════════════════════════════════════════════
#  1. HouseC.openclaw_client field
# ═══════════════════════════════════════════════════════════════

class TestOpenClawClientField:
    def test_field_exists(self):
        hc = HouseC(knowledge_graph=_make_graph())
        assert hasattr(hc, "openclaw_client")

    def test_default_is_none(self):
        hc = HouseC(knowledge_graph=_make_graph())
        assert hc.openclaw_client is None

    def test_accepts_openclaw_client(self):
        client = _make_client()
        hc = HouseC(knowledge_graph=_make_graph(), openclaw_client=client)
        assert hc.openclaw_client is client


# ═══════════════════════════════════════════════════════════════
#  2. HouseC._execute_browser_task()
# ═══════════════════════════════════════════════════════════════

class TestExecuteBrowserTask:
    def test_method_exists(self):
        hc = HouseC(knowledge_graph=_make_graph())
        assert callable(getattr(hc, "_execute_browser_task", None))

    def test_sets_passed_validation_on_success(self):
        hc = HouseC(knowledge_graph=_make_graph())
        client = _make_client(send_result="FINDING: $120/hr on Upwork")
        artifact = BuildArtifact(sso=_make_sso())
        result = hc._execute_browser_task(artifact, client)
        assert result.passed_validation is True

    def test_sets_execution_proof_from_client_response(self):
        hc = HouseC(knowledge_graph=_make_graph())
        client = _make_client(send_result="FINDING: top gig pays $200")
        artifact = BuildArtifact(sso=_make_sso())
        result = hc._execute_browser_task(artifact, client)
        assert result.execution_proof == "FINDING: top gig pays $200"

    def test_fails_when_client_returns_empty(self):
        hc = HouseC(knowledge_graph=_make_graph())
        client = _make_client(send_result="")
        artifact = BuildArtifact(sso=_make_sso())
        result = hc._execute_browser_task(artifact, client)
        assert result.passed_validation is False
        assert result.execution_proof is None

    def test_fails_when_client_returns_no_data(self):
        hc = HouseC(knowledge_graph=_make_graph())
        client = _make_client(send_result="NO_DATA: site blocked")
        artifact = BuildArtifact(sso=_make_sso())
        result = hc._execute_browser_task(artifact, client)
        assert result.passed_validation is False

    def test_sets_validation_error_on_empty(self):
        hc = HouseC(knowledge_graph=_make_graph())
        client = _make_client(send_result="")
        artifact = BuildArtifact(sso=_make_sso())
        result = hc._execute_browser_task(artifact, client)
        assert len(result.validation_errors) > 0

    def test_sends_redefined_problem_to_client(self):
        hc = HouseC(knowledge_graph=_make_graph())
        client = _make_client(send_result="FINDING: result")
        sso = _make_sso(problem="find Upwork gigs for Python developers")
        artifact = BuildArtifact(sso=sso)
        hc._execute_browser_task(artifact, client)
        call_args = client.send.call_args[0][0]
        assert "Upwork gigs for Python developers" in call_args


# ═══════════════════════════════════════════════════════════════
#  4. build() routing — uses OpenClaw when available + browser task
# ═══════════════════════════════════════════════════════════════

class TestBuildRouting:
    def test_uses_openclaw_for_browser_task_when_available(self, tmp_path):
        client = _make_client(available=True, send_result="FINDING: LinkedIn gig $150/hr")
        hc = HouseC(
            knowledge_graph=_make_graph(),
            openclaw_client=client,
            workspace_dir=str(tmp_path),
        )
        sso = _make_sso(problem="find leads on LinkedIn for AI consulting")
        dr = _make_dr(survived=True)

        result = hc.build(sso, dr)

        assert client.send.called
        assert result.success is True
        assert result.artifact.execution_proof == "FINDING: LinkedIn gig $150/hr"

    def test_falls_back_to_script_when_openclaw_unavailable(self, tmp_path):
        client = _make_client(available=False)
        hc = HouseC(
            knowledge_graph=_make_graph(),
            openclaw_client=client,
            workspace_dir=str(tmp_path),
        )
        sso = _make_sso(problem="find leads on LinkedIn")

        script_generated = []
        with patch.object(hc, "_generate_action_script", side_effect=lambda s: (script_generated.append(True), "# NEXUS Action\nprint('fallback')")[1]), \
             patch.object(hc, "_execute_action", return_value=BuildArtifact(
                 sso=sso, passed_validation=True, execution_proof="fallback result"
             )):
            result = hc.build(sso, _make_dr())

        assert not client.send.called
        assert script_generated  # fell back to script generation

    def test_falls_back_when_no_openclaw_client(self, tmp_path):
        hc = HouseC(knowledge_graph=_make_graph(), workspace_dir=str(tmp_path))
        sso = _make_sso(problem="find LinkedIn leads")

        with patch.object(hc, "_generate_action_script", return_value="# NEXUS Action\nprint('ok')"), \
             patch.object(hc, "_execute_action", return_value=BuildArtifact(
                 sso=sso, passed_validation=True, execution_proof="result"
             )):
            result = hc.build(sso, _make_dr())

        assert result is not None  # no crash

    def test_openclaw_failure_still_saves_artifact(self, tmp_path):
        """If OpenClaw returns empty, artifact is saved and success=False."""
        client = _make_client(available=True, send_result="")
        hc = HouseC(
            knowledge_graph=_make_graph(),
            openclaw_client=client,
            workspace_dir=str(tmp_path),
        )
        sso = _make_sso(problem="find LinkedIn profiles")

        result = hc.build(sso, _make_dr())

        assert result.success is False
        assert result.ready_for_house_a is False
        # Artifact should still be saved
        import pathlib
        artifact_files = list(pathlib.Path(tmp_path).rglob("artifact.json"))
        assert len(artifact_files) == 1


# ═══════════════════════════════════════════════════════════════
#  5. Graceful fallback — gateway offline
# ═══════════════════════════════════════════════════════════════

class TestGracefulFallback:
    def test_is_available_false_triggers_fallback(self, tmp_path):
        client = MagicMock(spec=OpenClawClient)
        client.is_available.return_value = False
        hc = HouseC(
            knowledge_graph=_make_graph(),
            openclaw_client=client,
            workspace_dir=str(tmp_path),
        )
        sso = _make_sso(problem="scrape Upwork for Python gigs")

        fallback_artifact = BuildArtifact(
            sso=sso, passed_validation=True, execution_proof="fallback data"
        )
        with patch.object(hc, "_generate_action_script", return_value="# NEXUS Action\nprint('ok')") as mock_gen, \
             patch.object(hc, "_execute_action", return_value=fallback_artifact):
            result = hc.build(sso, _make_dr())

        client.send.assert_not_called()
        assert mock_gen.called  # fell back to script generation
        assert result.success is True

    def test_openclaw_exception_during_send_still_works(self, tmp_path):
        """If send() raises unexpectedly, build must not crash (falls back to script)."""
        client = MagicMock(spec=OpenClawClient)
        client.is_available.return_value = True
        client.send.side_effect = RuntimeError("gateway crashed")

        hc = HouseC(
            knowledge_graph=_make_graph(),
            openclaw_client=client,
            workspace_dir=str(tmp_path),
        )
        sso = _make_sso(problem="find Upwork gigs")

        fallback_artifact = BuildArtifact(
            sso=sso, passed_validation=True, execution_proof="fallback result"
        )
        # Mock fallback so we don't hit the real LLM router
        with patch.object(hc, "_generate_action_script", return_value="# NEXUS Action\nprint('ok')"), \
             patch.object(hc, "_execute_action", return_value=fallback_artifact):
            try:
                result = hc.build(sso, _make_dr())
            except Exception as exc:
                pytest.fail(f"build() raised unexpectedly: {exc}")

        assert result.success is True  # fallback succeeded

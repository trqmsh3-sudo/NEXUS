"""Tests for House C — The Business Action Executor.

LLM calls are mocked via the ModelRouter. Subprocess calls for
action execution are mocked so tests run without network access.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from nexus.core.belief_certificate import BeliefCertificate
from nexus.core.house_b import StructuredSpecificationObject
from nexus.core.house_c import (
    BuildArtifact,
    BuildResult,
    HouseC,
)
from nexus.core.house_d import DestructionReport
from nexus.core.knowledge_graph import KnowledgeGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sso(
    problem: str = "Find profitable freelance opportunities",
    domain: str = "Business Intelligence",
) -> StructuredSpecificationObject:
    return StructuredSpecificationObject(
        original_input=problem,
        redefined_problem=problem,
        assumptions=["Internet available"],
        constraints=["Use only free sources"],
        success_criteria=["Identify at least one opportunity"],
        required_inputs=["public data"],
        expected_outputs=["list of opportunities"],
        domain=domain,
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


def _make_graph() -> KnowledgeGraph:
    path = str(
        pathlib.Path(tempfile.gettempdir()) / f"nexus_hc_{uuid.uuid4().hex}.json"
    )
    return KnowledgeGraph(storage_path=path)


SAMPLE_SCRIPT: str = (
    "# NEXUS Action\n"
    "import urllib.request, json\n\n"
    "req = urllib.request.Request(\n"
    "    'https://www.reddit.com/r/forhire/hot.json?limit=3',\n"
    "    headers={'User-Agent': 'NEXUS/2.8', 'Accept': 'application/json'},\n"
    ")\n"
    "try:\n"
    "    with urllib.request.urlopen(req, timeout=10) as r:\n"
    "        data = json.loads(r.read())\n"
    "    children = data.get('data', {}).get('children', [])\n"
    "    for c in children[:3]:\n"
    "        p = c.get('data', {})\n"
    "        title = p.get('title', '')\n"
    "        if title:\n"
    "            print(f'OPPORTUNITY: {title}')\n"
    "except Exception as e:\n"
    "    print(f'NO_DATA: {e}')\n"
)

SAMPLE_OUTPUT: str = (
    "OPPORTUNITY: [HIRING] AI integration specialist $80/hr remote\n"
    "OPPORTUNITY: [FOR HIRE] Business automation consulting\n"
)


def _mock_popen_success(cmd, **kwargs):
    proc = MagicMock()
    proc.communicate.return_value = (SAMPLE_OUTPUT, "")
    proc.returncode = 0
    return proc


def _mock_popen_failure(cmd, **kwargs):
    proc = MagicMock()
    proc.communicate.return_value = ("", "ConnectionError: network unreachable")
    proc.returncode = 1
    return proc


# ---------------------------------------------------------------------------
# BuildArtifact unit tests
# ---------------------------------------------------------------------------

class TestBuildArtifact:
    """Tests for the BuildArtifact dataclass."""

    def test_default_id_is_uuid(self) -> None:
        a = BuildArtifact()
        assert len(a.artifact_id) == 36
        assert a.artifact_id.count("-") == 4

    def test_to_dict(self) -> None:
        a = BuildArtifact(code="x = 1", language="python")
        d = a.to_dict()
        assert d["code"] == "x = 1"
        assert d["language"] == "python"
        assert "artifact_id" in d

    def test_defaults(self) -> None:
        a = BuildArtifact()
        assert a.passed_validation is False
        assert a.execution_proof is None
        assert a.validation_errors == []


# ---------------------------------------------------------------------------
# BuildResult unit tests
# ---------------------------------------------------------------------------

class TestBuildResult:
    """Tests for the BuildResult dataclass."""

    def test_to_dict(self) -> None:
        br = BuildResult(
            artifact=BuildArtifact(),
            success=True,
            house_d_report=_survived_report(),
            ready_for_house_a=True,
        )
        d = br.to_dict()
        assert d["success"] is True
        assert d["ready_for_house_a"] is True
        assert "artifact" in d
        assert "house_d_report" in d


# ---------------------------------------------------------------------------
# HouseC._strip_fences
# ---------------------------------------------------------------------------

class TestStripFences:
    """Tests for markdown fence stripping."""

    def test_strips_python_fence(self) -> None:
        raw = "```python\nprint('hi')\n```"
        assert HouseC._strip_fences(raw) == "print('hi')"

    def test_strips_plain_fence(self) -> None:
        raw = "```\ncode here\n```"
        assert HouseC._strip_fences(raw) == "code here"

    def test_no_fence_passthrough(self) -> None:
        raw = "x = 1"
        assert HouseC._strip_fences(raw) == "x = 1"

    def test_strips_py_fence(self) -> None:
        raw = "```py\nfoo()\n```"
        assert HouseC._strip_fences(raw) == "foo()"


# ---------------------------------------------------------------------------
# HouseC.build — gate check
# ---------------------------------------------------------------------------

class TestBuildGate:
    """Tests that build() rejects SSOs that didn't survive House D."""

    def test_raises_on_failed_report(self) -> None:
        hc = HouseC(knowledge_graph=_make_graph())
        with pytest.raises(ValueError, match="did not survive House D"):
            hc.build(_make_sso(), _failed_report())

    def test_accepts_survived_report(self, tmp_path: Any) -> None:
        router = MagicMock()
        router.complete.return_value = SAMPLE_SCRIPT
        hc = HouseC(
            knowledge_graph=_make_graph(),
            router=router,
            workspace_dir=str(tmp_path),
        )
        with patch("nexus.core.house_c.subprocess.Popen", _mock_popen_success):
            result = hc.build(_make_sso(), _survived_report())
        assert result.success is True
        assert result.ready_for_house_a is True


# ---------------------------------------------------------------------------
# HouseC._execute_action (mocked subprocess)
# ---------------------------------------------------------------------------

class TestExecuteAction:
    """Tests for the action execution runner."""

    def test_success_on_zero_exit_with_output(self, tmp_path: Any) -> None:
        hc = HouseC(
            knowledge_graph=_make_graph(),
            workspace_dir=str(tmp_path),
        )
        artifact = BuildArtifact(code=SAMPLE_SCRIPT)
        with patch("nexus.core.house_c.subprocess.Popen", _mock_popen_success):
            result = hc._execute_action(artifact)
        assert result.passed_validation is True
        assert result.execution_proof is not None
        assert "OPPORTUNITY" in result.execution_proof

    def test_failure_on_nonzero_exit(self, tmp_path: Any) -> None:
        hc = HouseC(
            knowledge_graph=_make_graph(),
            workspace_dir=str(tmp_path),
        )
        artifact = BuildArtifact(code=SAMPLE_SCRIPT)
        with patch("nexus.core.house_c.subprocess.Popen", _mock_popen_failure):
            result = hc._execute_action(artifact)
        assert result.passed_validation is False
        assert len(result.validation_errors) == 1

    def test_failure_on_no_data_output(self, tmp_path: Any) -> None:
        def popen_no_data(cmd, **kwargs):
            proc = MagicMock()
            proc.communicate.return_value = ("NO_DATA: connection refused\n", "")
            proc.returncode = 0
            return proc

        hc = HouseC(
            knowledge_graph=_make_graph(),
            workspace_dir=str(tmp_path),
        )
        artifact = BuildArtifact(code=SAMPLE_SCRIPT)
        with patch("nexus.core.house_c.subprocess.Popen", popen_no_data):
            result = hc._execute_action(artifact)
        assert result.passed_validation is False

    def test_creates_workspace_dir(self, tmp_path: Any) -> None:
        workspace = tmp_path / "deep" / "nested"
        hc = HouseC(
            knowledge_graph=_make_graph(),
            workspace_dir=str(workspace),
        )
        artifact = BuildArtifact(code=SAMPLE_SCRIPT)
        with patch("nexus.core.house_c.subprocess.Popen", _mock_popen_success):
            hc._execute_action(artifact)
        build_dir = workspace / artifact.artifact_id
        assert build_dir.exists()
        assert (build_dir / "action.py").exists()


# ---------------------------------------------------------------------------
# HouseC._save_to_workspace
# ---------------------------------------------------------------------------

class TestSaveToWorkspace:
    """Tests for artifact persistence."""

    def test_saves_json(self, tmp_path: Any) -> None:
        hc = HouseC(
            knowledge_graph=_make_graph(),
            workspace_dir=str(tmp_path),
        )
        artifact = BuildArtifact(code=SAMPLE_SCRIPT)
        path = hc._save_to_workspace(artifact)
        assert path.endswith("artifact.json")

        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        assert data["artifact_id"] == artifact.artifact_id
        assert data["code"] == SAMPLE_SCRIPT


# ---------------------------------------------------------------------------
# HouseC.to_belief_certificate
# ---------------------------------------------------------------------------

class TestToBeliefCertificate:
    """Tests for converting artifacts to BeliefCertificates."""

    def test_passed_artifact_high_confidence(self) -> None:
        hc = HouseC(knowledge_graph=_make_graph())
        artifact = BuildArtifact(
            sso=_make_sso(problem="find opportunities"),
            code=SAMPLE_SCRIPT,
            passed_validation=True,
            execution_proof="OPPORTUNITY: AI tools $80/hr",
        )
        cert = hc.to_belief_certificate(artifact)
        assert cert.confidence == 0.88
        # execution_proof set → static print-snippet, not the raw script
        assert cert.executable_proof is not None
        assert "# NEXUS verified findings" in cert.executable_proof
        assert "OPPORTUNITY" in cert.executable_proof
        assert cert.is_valid() is True

    def test_failed_artifact_low_confidence(self) -> None:
        hc = HouseC(knowledge_graph=_make_graph())
        artifact = BuildArtifact(
            sso=_make_sso(),
            passed_validation=False,
            execution_proof=None,
        )
        cert = hc.to_belief_certificate(artifact)
        assert cert.confidence == 0.3
        assert cert.is_valid() is False

    def test_domain_propagated(self) -> None:
        hc = HouseC(knowledge_graph=_make_graph())
        artifact = BuildArtifact(
            sso=_make_sso(domain="Market Research"),
            code=SAMPLE_SCRIPT,
            passed_validation=True,
            execution_proof="OPPORTUNITY: SaaS tools",
        )
        cert = hc.to_belief_certificate(artifact)
        assert cert.domain == "Market Research"
        # execution_proof set → static print-snippet, not the raw script
        assert cert.executable_proof is not None
        assert "# NEXUS verified findings" in cert.executable_proof
        assert "SaaS tools" in cert.executable_proof


# ---------------------------------------------------------------------------
# Full pipeline (mocked router + mocked subprocess)
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """End-to-end tests for the build pipeline."""

    def test_success_pipeline(self, tmp_path: Any) -> None:
        router = MagicMock()
        router.complete.return_value = SAMPLE_SCRIPT
        hc = HouseC(
            knowledge_graph=_make_graph(),
            router=router,
            workspace_dir=str(tmp_path),
        )
        with patch("nexus.core.house_c.subprocess.Popen", _mock_popen_success):
            result = hc.build(_make_sso(), _survived_report())

        assert result.success is True
        assert result.ready_for_house_a is True
        assert result.artifact.passed_validation is True
        assert result.artifact.execution_proof is not None
        # Only one LLM call now (generate_action_script, no test generation)
        assert router.complete.call_count == 1

    def test_failed_validation_not_ready(self, tmp_path: Any) -> None:
        router = MagicMock()
        router.complete.return_value = SAMPLE_SCRIPT
        hc = HouseC(
            knowledge_graph=_make_graph(),
            router=router,
            workspace_dir=str(tmp_path),
        )
        with patch("nexus.core.house_c.subprocess.Popen", _mock_popen_failure):
            result = hc.build(_make_sso(), _survived_report())

        assert result.success is False
        assert result.ready_for_house_a is False

    def test_artifact_saved_to_workspace(self, tmp_path: Any) -> None:
        router = MagicMock()
        router.complete.return_value = SAMPLE_SCRIPT
        hc = HouseC(
            knowledge_graph=_make_graph(),
            router=router,
            workspace_dir=str(tmp_path),
        )
        with patch("nexus.core.house_c.subprocess.Popen", _mock_popen_success):
            result = hc.build(_make_sso(), _survived_report())

        artifact_dir = pathlib.Path(tmp_path) / result.artifact.artifact_id
        assert (artifact_dir / "action.py").exists()
        assert (artifact_dir / "artifact.json").exists()


class TestDirectJobFetcherFallback:
    """direct fetch empty → NO_DATA immediately, no AI controller."""

    def test_no_data_immediately_when_direct_fetch_empty(self):
        from nexus.core.openclaw_client import OpenClawClient
        mock_client = MagicMock(spec=OpenClawClient)
        mock_client.is_available.return_value = True

        hc = HouseC(
            knowledge_graph=_make_graph(),
            router=MagicMock(),
            openclaw_client=mock_client,
        )

        sso = StructuredSpecificationObject(
            original_input="find jobs",
            redefined_problem="Find Python remote gigs",
            success_criteria=["find at least one listing"],
        )

        with patch("nexus.core.house_c.DirectJobFetcher") as mock_fetcher_cls, \
             patch("nexus.core.house_c.OpenClawAIController") as mock_ctrl_cls:
            mock_fetcher_cls.return_value.fetch.return_value = ""
            result = hc.build(sso, _survived_report())

        mock_ctrl_cls.assert_not_called()
        assert result.success is False

    def test_ai_controller_not_called_when_direct_fetch_empty(self):
        """OpenClawAIController.run() must never be called when direct fetch returns ''."""
        from nexus.core.openclaw_client import OpenClawClient
        mock_client = MagicMock(spec=OpenClawClient)
        mock_client.is_available.return_value = True

        hc = HouseC(
            knowledge_graph=_make_graph(),
            router=MagicMock(),
            openclaw_client=mock_client,
        )

        with patch("nexus.core.house_c.DirectJobFetcher") as mock_fetcher_cls, \
             patch("nexus.core.house_c.OpenClawAIController") as mock_ctrl_cls:
            mock_fetcher_cls.return_value.fetch.return_value = ""
            hc.build(
                StructuredSpecificationObject(
                    original_input="x",
                    redefined_problem="Find design gigs",
                    success_criteria=["one listing"],
                ),
                _survived_report(),
            )

        mock_ctrl_cls.return_value.run.assert_not_called()

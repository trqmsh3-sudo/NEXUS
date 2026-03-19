"""Tests for House C — The Builder.

LLM calls are mocked via litellm.completion. Subprocess calls for
validation are mocked where needed so tests run without an API key
and without spawning pytest sub-processes.
"""

from __future__ import annotations

import json
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
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
    problem: str = "build a fibonacci function",
    domain: str = "General",
) -> StructuredSpecificationObject:
    return StructuredSpecificationObject(
        original_input="user request",
        redefined_problem=problem,
        assumptions=["Input is non-negative integer"],
        constraints=["Must be pure Python"],
        success_criteria=["Returns correct Fibonacci number"],
        required_inputs=["integer n"],
        expected_outputs=["integer"],
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
    import uuid
    from pathlib import Path
    path = str(Path(tempfile.gettempdir()) / f"nexus_hc_{uuid.uuid4().hex}.json")
    return KnowledgeGraph(storage_path=path)


def _fake_response(content: str) -> MagicMock:
    """Build a mock litellm completion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


SAMPLE_CODE: str = (
    "def fibonacci(n: int) -> int:\n"
    "    if n <= 1:\n"
    "        return n\n"
    "    a, b = 0, 1\n"
    "    for _ in range(2, n + 1):\n"
    "        a, b = b, a + b\n"
    "    return b\n"
)

SAMPLE_TESTS: str = (
    "from main import fibonacci\n\n"
    "def test_fib_zero():\n"
    "    assert fibonacci(0) == 0\n\n"
    "def test_fib_one():\n"
    "    assert fibonacci(1) == 1\n\n"
    "def test_fib_ten():\n"
    "    assert fibonacci(10) == 55\n"
)


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

    @patch("nexus.core.model_router.litellm")
    @patch("nexus.core.house_c.subprocess.run")
    def test_accepts_survived_report(
        self, mock_run: MagicMock, mock_litellm: MagicMock, tmp_path: Any,
    ) -> None:
        mock_litellm.completion.return_value = _fake_response(SAMPLE_CODE)
        mock_run.return_value = MagicMock(
            returncode=0, stdout="all passed", stderr="",
        )

        hc = HouseC(
            knowledge_graph=_make_graph(),
            workspace_dir=str(tmp_path),
        )
        result = hc.build(_make_sso(), _survived_report())
        assert result.success is True
        assert result.ready_for_house_a is True


# ---------------------------------------------------------------------------
# HouseC._generate_code (mocked LLM)
# ---------------------------------------------------------------------------

class TestGenerateCode:
    """Tests for code generation."""

    @patch("nexus.core.model_router.litellm")
    def test_returns_stripped_code(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _fake_response("```python\ndef foo(): pass\n```")
        hc = HouseC(knowledge_graph=_make_graph())
        code = hc._generate_code(_make_sso())
        assert "def foo():" in code
        assert "```" not in code

    @patch("nexus.core.model_router.litellm")
    def test_prompt_includes_constraints(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _fake_response("def foo(): pass")
        hc = HouseC(knowledge_graph=_make_graph())
        hc._generate_code(_make_sso())
        user_msg = mock_litellm.completion.call_args[1]["messages"][1]["content"]
        assert "Must be pure Python" in user_msg


# ---------------------------------------------------------------------------
# HouseC._generate_tests (mocked LLM)
# ---------------------------------------------------------------------------

class TestGenerateTests:
    """Tests for test generation."""

    @patch("nexus.core.model_router.litellm")
    def test_returns_stripped_tests(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _fake_response("```python\ndef test_x(): assert True\n```")
        hc = HouseC(knowledge_graph=_make_graph())
        tests = hc._generate_tests(_make_sso(), "def x(): return 1")
        assert "def test_x():" in tests
        assert "```" not in tests

    @patch("nexus.core.model_router.litellm")
    def test_prompt_includes_code(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _fake_response("def test_x(): assert True")
        hc = HouseC(knowledge_graph=_make_graph())
        hc._generate_tests(_make_sso(), "def fibonacci(n): return n")
        user_msg = mock_litellm.completion.call_args[1]["messages"][1]["content"]
        assert "def fibonacci(n)" in user_msg


# ---------------------------------------------------------------------------
# HouseC._validate (mocked subprocess)
# ---------------------------------------------------------------------------

class TestValidate:
    """Tests for the validation subprocess runner."""

    def test_passed_validation(self, tmp_path: Any) -> None:
        hc = HouseC(
            knowledge_graph=_make_graph(),
            workspace_dir=str(tmp_path),
        )
        artifact = BuildArtifact(code=SAMPLE_CODE, tests=SAMPLE_TESTS)

        with patch("nexus.core.house_c.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="3 passed", stderr="",
            )
            result = hc._validate(artifact)

        assert result.passed_validation is True
        assert result.execution_proof == "3 passed"
        assert result.validation_errors == []

    def test_failed_validation(self, tmp_path: Any) -> None:
        hc = HouseC(
            knowledge_graph=_make_graph(),
            workspace_dir=str(tmp_path),
        )
        artifact = BuildArtifact(code="bad code", tests="bad tests")

        with (
            patch("nexus.core.house_c.subprocess.run") as mock_run,
            patch("nexus.core.model_router.litellm") as mock_litellm,
        ):
            mock_litellm.completion.return_value = _fake_response(
                "def test_healed(): assert True",
            )
            mock_run.return_value = MagicMock(
                returncode=1, stdout="FAILED", stderr="Error details",
            )
            result = hc._validate(artifact)

        assert result.passed_validation is False
        assert len(result.validation_errors) == 1
        assert "FAILED" in result.validation_errors[0]
        assert result.healing_attempts == 2

    def test_creates_workspace_dir(self, tmp_path: Any) -> None:
        workspace = tmp_path / "deep" / "nested"
        hc = HouseC(
            knowledge_graph=_make_graph(),
            workspace_dir=str(workspace),
        )
        artifact = BuildArtifact(code="x=1", tests="pass")

        with patch("nexus.core.house_c.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="ok", stderr="",
            )
            hc._validate(artifact)

        build_dir = workspace / artifact.artifact_id
        assert build_dir.exists()


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
        artifact = BuildArtifact(code="x=1", tests="assert True")
        path = hc._save_to_workspace(artifact)
        assert path.endswith("artifact.json")

        import pathlib
        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        assert data["artifact_id"] == artifact.artifact_id
        assert data["code"] == "x=1"


# ---------------------------------------------------------------------------
# HouseC.to_belief_certificate
# ---------------------------------------------------------------------------

class TestToBeliefCertificate:
    """Tests for converting artifacts to BeliefCertificates."""

    def test_passed_artifact_high_confidence(self) -> None:
        hc = HouseC(knowledge_graph=_make_graph())
        artifact = BuildArtifact(
            sso=_make_sso(problem="build fib"),
            code="def fib(n): ...",
            passed_validation=True,
            execution_proof="3 passed in 0.01s",
        )
        cert = hc.to_belief_certificate(artifact)
        assert cert.confidence == 0.9
        assert cert.executable_proof == "def fib(n): ..."
        assert cert.is_valid() is True
        assert "build fib" in cert.claim

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
            sso=_make_sso(domain="ML Architecture"),
            code="x = 1\n",
            passed_validation=True,
            execution_proof="ok",
        )
        cert = hc.to_belief_certificate(artifact)
        assert cert.domain == "ML Architecture"
        assert cert.executable_proof == "x = 1"


# ---------------------------------------------------------------------------
# Full pipeline (mocked LLM + mocked subprocess)
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """End-to-end tests for the build pipeline."""

    @patch("nexus.core.model_router.litellm")
    @patch("nexus.core.house_c.subprocess.run")
    def test_success_pipeline(
        self, mock_run: MagicMock, mock_litellm: MagicMock, tmp_path: Any,
    ) -> None:
        mock_litellm.completion.return_value = _fake_response(SAMPLE_CODE)
        mock_run.return_value = MagicMock(
            returncode=0, stdout="3 passed", stderr="",
        )

        hc = HouseC(
            knowledge_graph=_make_graph(),
            workspace_dir=str(tmp_path),
        )
        result = hc.build(_make_sso(), _survived_report())

        assert result.success is True
        assert result.ready_for_house_a is True
        assert result.artifact.passed_validation is True
        assert result.artifact.execution_proof == "3 passed"
        assert mock_litellm.completion.call_count == 2

    @patch("nexus.core.model_router.litellm")
    @patch("nexus.core.house_c.subprocess.run")
    def test_failed_validation_not_ready(
        self, mock_run: MagicMock, mock_litellm: MagicMock, tmp_path: Any,
    ) -> None:
        mock_litellm.completion.return_value = _fake_response(SAMPLE_CODE)
        mock_run.return_value = MagicMock(
            returncode=1, stdout="FAILED", stderr="err",
        )

        hc = HouseC(
            knowledge_graph=_make_graph(),
            workspace_dir=str(tmp_path),
        )
        result = hc.build(_make_sso(), _survived_report())

        assert result.success is False
        assert result.ready_for_house_a is False

    @patch("nexus.core.model_router.litellm")
    @patch("nexus.core.house_c.subprocess.run")
    def test_artifact_saved_to_workspace(
        self, mock_run: MagicMock, mock_litellm: MagicMock, tmp_path: Any,
    ) -> None:
        mock_litellm.completion.return_value = _fake_response(SAMPLE_CODE)
        mock_run.return_value = MagicMock(
            returncode=0, stdout="ok", stderr="",
        )

        hc = HouseC(
            knowledge_graph=_make_graph(),
            workspace_dir=str(tmp_path),
        )
        result = hc.build(_make_sso(), _survived_report())

        import pathlib
        artifact_dir = pathlib.Path(tmp_path) / result.artifact.artifact_id
        assert (artifact_dir / "main.py").exists()
        assert (artifact_dir / "test_main.py").exists()
        assert (artifact_dir / "artifact.json").exists()

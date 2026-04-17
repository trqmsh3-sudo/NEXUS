"""Tests for House C — Business Action Executor.

These tests FAIL against the current Python-code-generator implementation
and PASS after house_c.py is restructured as a business action executor.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nexus.core.house_b import StructuredSpecificationObject
from nexus.core.house_c import HouseC, BuildResult, BuildArtifact
from nexus.core.house_d import DestructionReport
from nexus.core.knowledge_graph import KnowledgeGraph
from nexus.core.model_router import ModelRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sso(problem: str = "Find profitable freelance opportunities on Reddit") -> StructuredSpecificationObject:
    return StructuredSpecificationObject(
        original_input=problem,
        redefined_problem=problem,
        domain="Business Intelligence",
        success_criteria=["Identify at least one opportunity"],
        assumptions=["Internet access available"],
        constraints=["Use only free sources"],
        expected_outputs=["List of opportunities with revenue potential"],
        confidence=0.75,
    )


def _make_survived_report() -> DestructionReport:
    return DestructionReport(
        target_description="Find profitable freelance opportunities",
        survived=True,
        survival_score=0.72,
        recommendation="PROMOTE",
        attacks=[],
    )


def _make_failed_report() -> DestructionReport:
    return DestructionReport(
        target_description="Find profitable freelance opportunities",
        survived=False,
        survival_score=0.20,
        recommendation="REJECT",
        attacks=[],
    )


_ACTION_SCRIPT = """\
# NEXUS Action
import urllib.request, json

req = urllib.request.Request(
    "https://www.reddit.com/r/forhire/hot.json?limit=3",
    headers={"User-Agent": "NEXUS/2.8", "Accept": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read())
    children = data.get("data", {}).get("children", [])
    for c in children[:3]:
        p = c.get("data", {})
        title = p.get("title", "")
        if title:
            print(f"OPPORTUNITY: {title}")
except Exception as e:
    print(f"NO_DATA: {e}")
"""

_ACTION_OUTPUT = (
    "OPPORTUNITY: [HIRING] AI integration specialist $80/hr remote\n"
    "OPPORTUNITY: [FOR HIRE] Business automation consulting\n"
)


def _mock_popen_success(cmd, **kwargs):
    proc = MagicMock()
    proc.communicate.return_value = (_ACTION_OUTPUT, "")
    proc.returncode = 0
    return proc


def _mock_popen_failure(cmd, **kwargs):
    proc = MagicMock()
    proc.communicate.return_value = ("", "ConnectionError: network unreachable")
    proc.returncode = 1
    return proc


def _mock_popen_no_data(cmd, **kwargs):
    proc = MagicMock()
    proc.communicate.return_value = ("NO_DATA: connection refused\n", "")
    proc.returncode = 0
    return proc


# ---------------------------------------------------------------------------
# Module surface — new symbols present, old ones removed
# ---------------------------------------------------------------------------

def test_action_system_exists() -> None:
    import nexus.core.house_c as hc
    assert hasattr(hc, "ACTION_SYSTEM"), "ACTION_SYSTEM must be defined in house_c"


def test_code_system_removed() -> None:
    import nexus.core.house_c as hc
    assert not hasattr(hc, "CODE_SYSTEM"), "CODE_SYSTEM must be removed from house_c"


def test_test_system_removed() -> None:
    import nexus.core.house_c as hc
    assert not hasattr(hc, "TEST_SYSTEM"), "TEST_SYSTEM must be removed from house_c"


def test_generate_tests_method_removed() -> None:
    assert not hasattr(HouseC, "_generate_tests"), "_generate_tests() must be removed"


def test_heal_tests_method_removed() -> None:
    assert not hasattr(HouseC, "_heal_tests"), "_heal_tests() must be removed"


def test_build_method_exists() -> None:
    assert hasattr(HouseC, "build"), "HouseC must still have build()"


def test_generate_action_script_method_exists() -> None:
    assert hasattr(HouseC, "_generate_action_script"), (
        "HouseC must have _generate_action_script()"
    )


def test_execute_action_method_exists() -> None:
    assert hasattr(HouseC, "_execute_action"), (
        "HouseC must have _execute_action()"
    )


# ---------------------------------------------------------------------------
# build() — raises on failed destruction report
# ---------------------------------------------------------------------------

def test_build_raises_when_sso_not_survived(tmp_path) -> None:
    kg = KnowledgeGraph()
    router = MagicMock(spec=ModelRouter)
    hc = HouseC(knowledge_graph=kg, router=router, workspace_dir=str(tmp_path))

    with pytest.raises(ValueError):
        hc.build(_make_sso(), _make_failed_report())


# ---------------------------------------------------------------------------
# build() — happy path
# ---------------------------------------------------------------------------

def test_build_returns_build_result(tmp_path) -> None:
    kg = KnowledgeGraph()
    router = MagicMock(spec=ModelRouter)
    router.complete.return_value = _ACTION_SCRIPT
    hc = HouseC(knowledge_graph=kg, router=router, workspace_dir=str(tmp_path))

    with patch("nexus.core.house_c.subprocess.Popen", _mock_popen_success):
        result = hc.build(_make_sso(), _make_survived_report())

    assert isinstance(result, BuildResult)


def test_build_success_sets_ready_for_house_a(tmp_path) -> None:
    kg = KnowledgeGraph()
    router = MagicMock(spec=ModelRouter)
    router.complete.return_value = _ACTION_SCRIPT
    hc = HouseC(knowledge_graph=kg, router=router, workspace_dir=str(tmp_path))

    with patch("nexus.core.house_c.subprocess.Popen", _mock_popen_success):
        result = hc.build(_make_sso(), _make_survived_report())

    assert result.success is True
    assert result.ready_for_house_a is True


def test_build_stores_execution_proof(tmp_path) -> None:
    kg = KnowledgeGraph()
    router = MagicMock(spec=ModelRouter)
    router.complete.return_value = _ACTION_SCRIPT
    hc = HouseC(knowledge_graph=kg, router=router, workspace_dir=str(tmp_path))

    with patch("nexus.core.house_c.subprocess.Popen", _mock_popen_success):
        result = hc.build(_make_sso(), _make_survived_report())

    assert result.artifact.execution_proof is not None
    assert "OPPORTUNITY" in result.artifact.execution_proof


def test_build_failure_when_script_nonzero_exit(tmp_path) -> None:
    kg = KnowledgeGraph()
    router = MagicMock(spec=ModelRouter)
    router.complete.return_value = _ACTION_SCRIPT
    hc = HouseC(knowledge_graph=kg, router=router, workspace_dir=str(tmp_path))

    with patch("nexus.core.house_c.subprocess.Popen", _mock_popen_failure):
        result = hc.build(_make_sso(), _make_survived_report())

    assert result.success is False
    assert result.ready_for_house_a is False


def test_build_failure_when_only_no_data(tmp_path) -> None:
    kg = KnowledgeGraph()
    router = MagicMock(spec=ModelRouter)
    router.complete.return_value = _ACTION_SCRIPT
    hc = HouseC(knowledge_graph=kg, router=router, workspace_dir=str(tmp_path))

    with patch("nexus.core.house_c.subprocess.Popen", _mock_popen_no_data):
        result = hc.build(_make_sso(), _make_survived_report())

    assert result.success is False


# ---------------------------------------------------------------------------
# Action script content
# ---------------------------------------------------------------------------

def test_generated_script_starts_with_nexus_action(tmp_path) -> None:
    kg = KnowledgeGraph()
    router = MagicMock(spec=ModelRouter)
    router.complete.return_value = _ACTION_SCRIPT
    hc = HouseC(knowledge_graph=kg, router=router, workspace_dir=str(tmp_path))

    with patch("nexus.core.house_c.subprocess.Popen", _mock_popen_success):
        result = hc.build(_make_sso(), _make_survived_report())

    assert result.artifact.code.strip().startswith("# NEXUS Action")


def test_generated_script_stored_in_artifact(tmp_path) -> None:
    kg = KnowledgeGraph()
    router = MagicMock(spec=ModelRouter)
    router.complete.return_value = _ACTION_SCRIPT
    hc = HouseC(knowledge_graph=kg, router=router, workspace_dir=str(tmp_path))

    with patch("nexus.core.house_c.subprocess.Popen", _mock_popen_success):
        result = hc.build(_make_sso(), _make_survived_report())

    assert result.artifact.code.strip() == _ACTION_SCRIPT.strip()


# ---------------------------------------------------------------------------
# to_belief_certificate
# ---------------------------------------------------------------------------

def test_to_belief_certificate_valid_on_success(tmp_path) -> None:
    kg = KnowledgeGraph()
    router = MagicMock(spec=ModelRouter)
    router.complete.return_value = _ACTION_SCRIPT
    hc = HouseC(knowledge_graph=kg, router=router, workspace_dir=str(tmp_path))

    with patch("nexus.core.house_c.subprocess.Popen", _mock_popen_success):
        result = hc.build(_make_sso(), _make_survived_report())

    belief = hc.to_belief_certificate(result.artifact)
    assert belief.is_valid()
    assert belief.confidence >= 0.8


def test_to_belief_certificate_domain_matches_sso(tmp_path) -> None:
    kg = KnowledgeGraph()
    router = MagicMock(spec=ModelRouter)
    router.complete.return_value = _ACTION_SCRIPT
    hc = HouseC(knowledge_graph=kg, router=router, workspace_dir=str(tmp_path))
    sso = _make_sso()

    with patch("nexus.core.house_c.subprocess.Popen", _mock_popen_success):
        result = hc.build(sso, _make_survived_report())

    belief = hc.to_belief_certificate(result.artifact)
    assert belief.domain == sso.domain


def test_to_belief_certificate_low_confidence_on_failure(tmp_path) -> None:
    kg = KnowledgeGraph()
    router = MagicMock(spec=ModelRouter)
    router.complete.return_value = _ACTION_SCRIPT
    hc = HouseC(knowledge_graph=kg, router=router, workspace_dir=str(tmp_path))

    with patch("nexus.core.house_c.subprocess.Popen", _mock_popen_failure):
        result = hc.build(_make_sso(), _make_survived_report())

    belief = hc.to_belief_certificate(result.artifact)
    assert belief.confidence < 0.6

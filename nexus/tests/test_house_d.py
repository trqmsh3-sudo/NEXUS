"""Tests for House D — The Destroyer.

All LLM calls are mocked via litellm so tests run without
an API key. Tests cover the enum, dataclasses, cycle logic,
contradiction-based auto-fatal injection, scoring, recommendation,
and promotion gating.
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
from nexus.core.house_d import (
    AttackResult,
    AttackType,
    DestructionReport,
    HouseD,
)
from nexus.core.knowledge_graph import KnowledgeGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cert(
    claim: str = "test claim",
    confidence: float = 0.9,
    domain: str = "Testing",
    contradictions: list[str] | None = None,
    executable_proof: str | None = "assert True",
) -> BeliefCertificate:
    return BeliefCertificate(
        claim=claim,
        source="unit-test",
        confidence=confidence,
        contradictions=contradictions or [],
        decay_rate=0.0,
        created_at=datetime.now(timezone.utc),
        last_verified=datetime.now(timezone.utc),
        executable_proof=executable_proof,
        domain=domain,
    )


def _make_graph(*certs: BeliefCertificate) -> KnowledgeGraph:
    import tempfile
    import uuid
    from pathlib import Path
    path = str(Path(tempfile.gettempdir()) / f"nexus_hd_{uuid.uuid4().hex}.json")
    graph = KnowledgeGraph(storage_path=path)
    for c in certs:
        graph.add_belief(c)
    return graph


def _make_sso(
    problem: str = "test problem",
    domain: str = "General",
    confidence: float = 0.8,
) -> StructuredSpecificationObject:
    return StructuredSpecificationObject(
        original_input="user said something",
        redefined_problem=problem,
        assumptions=["A1"],
        constraints=["C1"],
        success_criteria=["S1"],
        required_inputs=["I1"],
        expected_outputs=["O1"],
        domain=domain,
        confidence=confidence,
    )


def _fake_response(payload: dict[str, Any]) -> MagicMock:
    msg = MagicMock()
    msg.content = json.dumps(payload)
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _cycle_payload(
    attacks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a valid LLM response payload for a single cycle."""
    if attacks is None:
        attacks = [
            {
                "attack_type": "LOGIC_FLAW",
                "severity": 0.3,
                "description": "Minor logical gap",
                "is_fatal": False,
            },
        ]
    return {"attacks": attacks}


def _fatal_cycle_payload() -> dict[str, Any]:
    return {
        "attacks": [
            {
                "attack_type": "SECURITY_VULNERABILITY",
                "severity": 0.95,
                "description": "SQL injection possible",
                "is_fatal": True,
            },
        ],
    }


def _empty_cycle_payload() -> dict[str, Any]:
    return {"attacks": []}


# ---------------------------------------------------------------------------
# AttackType enum
# ---------------------------------------------------------------------------

class TestAttackType:
    """Tests for the AttackType enum."""

    def test_all_members_exist(self) -> None:
        expected = {
            "LOGIC_FLAW", "SECURITY_VULNERABILITY", "SCALABILITY_FAILURE",
            "HIDDEN_ASSUMPTION", "ETHICAL_VIOLATION",
            "CONTRADICTS_KNOWN_TRUTH", "UNMAINTAINABLE", "UNSOLVABLE_PROBLEM",
        }
        assert set(AttackType.__members__.keys()) == expected

    def test_value_matches_name(self) -> None:
        for member in AttackType:
            assert member.value == member.name


# ---------------------------------------------------------------------------
# AttackResult dataclass
# ---------------------------------------------------------------------------

class TestAttackResult:
    """Tests for the AttackResult dataclass."""

    def test_basic_construction(self) -> None:
        ar = AttackResult(
            target="x", attack_type="LOGIC_FLAW",
            severity=0.5, description="flaw", is_fatal=False,
        )
        assert ar.severity == 0.5
        assert ar.is_fatal is False

    def test_severity_clamped_high(self) -> None:
        ar = AttackResult(
            target="x", attack_type="LOGIC_FLAW",
            severity=1.5, description="flaw", is_fatal=False,
        )
        assert ar.severity == 1.0

    def test_severity_clamped_low(self) -> None:
        ar = AttackResult(
            target="x", attack_type="LOGIC_FLAW",
            severity=-0.3, description="flaw", is_fatal=False,
        )
        assert ar.severity == 0.0

    def test_to_dict(self) -> None:
        ar = AttackResult(
            target="t", attack_type="ETHICAL_VIOLATION",
            severity=0.8, description="bad", is_fatal=True,
        )
        d = ar.to_dict()
        assert d["target"] == "t"
        assert d["attack_type"] == "ETHICAL_VIOLATION"
        assert d["is_fatal"] is True


# ---------------------------------------------------------------------------
# DestructionReport dataclass
# ---------------------------------------------------------------------------

class TestDestructionReport:
    """Tests for the DestructionReport dataclass."""

    def test_defaults(self) -> None:
        dr = DestructionReport(target_description="x")
        assert dr.survived is True
        assert dr.attacks == []
        assert dr.recommendation == "REJECT"

    def test_to_dict(self) -> None:
        attack = AttackResult(
            target="x", attack_type="LOGIC_FLAW",
            severity=0.4, description="gap", is_fatal=False,
        )
        dr = DestructionReport(
            target_description="test",
            attacks=[attack],
            survived=True,
            survival_score=0.8,
            cycles_survived=3,
            recommendation="PROMOTE",
        )
        d = dr.to_dict()
        assert len(d["attacks"]) == 1
        assert d["recommendation"] == "PROMOTE"


# ---------------------------------------------------------------------------
# HouseD._parse_json
# ---------------------------------------------------------------------------

class TestHouseDParseJson:
    """Tests for the JSON parser."""

    def test_valid_json(self) -> None:
        hd = HouseD(knowledge_graph=_make_graph())
        result = hd._parse_json('{"a": 1}', label="test")
        assert result == {"a": 1}

    def test_strips_markdown_fences(self) -> None:
        hd = HouseD(knowledge_graph=_make_graph())
        result = hd._parse_json('```json\n{"b": 2}\n```', label="test")
        assert result == {"b": 2}

    def test_raises_on_garbage(self) -> None:
        hd = HouseD(knowledge_graph=_make_graph())
        with pytest.raises(ValueError, match="invalid JSON"):
            hd._parse_json("totally broken {{{", label="test")


# ---------------------------------------------------------------------------
# HouseD.run_cycle (mocked LLM)
# ---------------------------------------------------------------------------

class TestRunCycle:
    """Tests for a single attack cycle."""

    @patch("nexus.core.model_router.litellm")
    def test_returns_attacks(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _fake_response(_cycle_payload())
        hd = HouseD(knowledge_graph=_make_graph())
        attacks = hd.run_cycle("target text", cycle_num=1)
        assert len(attacks) == 1
        assert attacks[0].attack_type == "LOGIC_FLAW"

    @patch("nexus.core.model_router.litellm")
    def test_empty_cycle(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _fake_response(_empty_cycle_payload())
        hd = HouseD(knowledge_graph=_make_graph())
        attacks = hd.run_cycle("target", cycle_num=1)
        assert attacks == []

    @patch("nexus.core.model_router.litellm")
    def test_unknown_attack_type_defaults(self, mock_litellm: MagicMock) -> None:
        payload = {"attacks": [
            {"attack_type": "ALIEN_INVASION", "severity": 0.5,
             "description": "lol", "is_fatal": False},
        ]}
        mock_litellm.completion.return_value = _fake_response(payload)
        hd = HouseD(knowledge_graph=_make_graph())
        attacks = hd.run_cycle("target", cycle_num=1)
        assert attacks[0].attack_type == "LOGIC_FLAW"

    @patch("nexus.core.model_router.litellm")
    def test_cycle_prompt_includes_cycle_number(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _fake_response(_empty_cycle_payload())
        hd = HouseD(knowledge_graph=_make_graph())
        hd.run_cycle("target", cycle_num=7)
        user_msg = mock_litellm.completion.call_args[1]["messages"][1]["content"]
        assert "CYCLE 7" in user_msg


# ---------------------------------------------------------------------------
# HouseD.attack_sso (mocked LLM)
# ---------------------------------------------------------------------------

class TestAttackSSO:
    """Integration tests for full SSO attack campaign."""

    @patch("nexus.core.model_router.litellm")
    def test_no_fatal_promotes(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _fake_response(_cycle_payload([
            {"attack_type": "HIDDEN_ASSUMPTION", "severity": 0.2,
             "description": "minor", "is_fatal": False},
        ]))
        hd = HouseD(knowledge_graph=_make_graph(), min_cycles=3)
        report = hd.attack_sso(_make_sso())

        assert report.survived is True
        assert report.recommendation == "PROMOTE"
        assert len(report.attacks) == 3

    @patch("nexus.core.model_router.litellm")
    def test_fatal_rejects(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _fake_response(_fatal_cycle_payload())
        hd = HouseD(knowledge_graph=_make_graph(), min_cycles=3)
        report = hd.attack_sso(_make_sso())

        assert report.survived is False
        assert report.recommendation in ("REVISE", "REJECT")

    @patch("nexus.core.model_router.litellm")
    def test_runs_min_cycles(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _fake_response(_empty_cycle_payload())
        hd = HouseD(knowledge_graph=_make_graph(), min_cycles=5)
        hd.attack_sso(_make_sso())
        assert mock_litellm.completion.call_count == 5

    @patch("nexus.core.model_router.litellm")
    def test_survival_score_perfect_on_empty(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _fake_response(_empty_cycle_payload())
        hd = HouseD(knowledge_graph=_make_graph(), min_cycles=3)
        report = hd.attack_sso(_make_sso())
        assert report.survival_score == 1.0
        assert report.recommendation == "PROMOTE"


# ---------------------------------------------------------------------------
# HouseD.attack_belief (mocked LLM)
# ---------------------------------------------------------------------------

class TestAttackBelief:
    """Integration tests for belief attack with contradiction detection."""

    @patch("nexus.core.model_router.litellm")
    def test_contradiction_auto_fatal(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _fake_response(_empty_cycle_payload())

        graph = _make_graph(
            _make_cert(claim="X is true", contradictions=["X is false"]),
        )
        hd = HouseD(knowledge_graph=graph, min_cycles=3)
        target = _make_cert(claim="X is false")
        report = hd.attack_belief(target)

        assert report.survived is False
        fatal = [a for a in report.attacks if a.is_fatal]
        assert len(fatal) >= 1
        assert fatal[0].attack_type == "CONTRADICTS_KNOWN_TRUTH"

    @patch("nexus.core.model_router.litellm")
    def test_no_contradiction_clean(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _fake_response(_empty_cycle_payload())
        hd = HouseD(knowledge_graph=_make_graph(), min_cycles=3)
        report = hd.attack_belief(_make_cert(claim="harmless"))

        assert report.survived is True
        assert report.survival_score == 1.0

    @patch("nexus.core.model_router.litellm")
    def test_runs_min_cycles_on_belief(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _fake_response(_empty_cycle_payload())
        hd = HouseD(knowledge_graph=_make_graph(), min_cycles=4)
        hd.attack_belief(_make_cert())
        assert mock_litellm.completion.call_count == 4


# ---------------------------------------------------------------------------
# HouseD.should_promote
# ---------------------------------------------------------------------------

class TestShouldPromote:
    """Tests for the promotion gate."""

    def test_promotes_when_all_criteria_met(self) -> None:
        hd = HouseD(knowledge_graph=_make_graph(), min_cycles=3)
        report = DestructionReport(
            target_description="x",
            survived=True,
            survival_score=0.85,
            cycles_survived=3,
            recommendation="PROMOTE",
        )
        assert hd.should_promote(report) is True

    def test_rejects_fatal(self) -> None:
        hd = HouseD(knowledge_graph=_make_graph(), min_cycles=3)
        report = DestructionReport(
            target_description="x",
            survived=False,
            survival_score=0.85,
            cycles_survived=3,
        )
        assert hd.should_promote(report) is False

    def test_rejects_low_score(self) -> None:
        hd = HouseD(knowledge_graph=_make_graph(), min_cycles=3)
        report = DestructionReport(
            target_description="x",
            survived=True,
            survival_score=0.5,
            cycles_survived=3,
        )
        assert hd.should_promote(report) is False

    def test_rejects_insufficient_cycles(self) -> None:
        hd = HouseD(knowledge_graph=_make_graph(), min_cycles=5)
        report = DestructionReport(
            target_description="x",
            survived=True,
            survival_score=0.9,
            cycles_survived=3,
        )
        assert hd.should_promote(report) is False

    def test_boundary_score_exactly_0_7(self) -> None:
        hd = HouseD(knowledge_graph=_make_graph(), min_cycles=3)
        report = DestructionReport(
            target_description="x",
            survived=True,
            survival_score=0.7,
            cycles_survived=3,
        )
        assert hd.should_promote(report) is True


# ---------------------------------------------------------------------------
# HouseD._build_report scoring
# ---------------------------------------------------------------------------

class TestBuildReport:
    """Tests for the internal report builder logic."""

    def test_no_attacks_perfect_score(self) -> None:
        hd = HouseD(knowledge_graph=_make_graph())
        report = hd._build_report("target", [])
        assert report.survival_score == 1.0
        assert report.survived is True
        assert report.recommendation == "PROMOTE"

    def test_all_fatal_low_score(self) -> None:
        hd = HouseD(knowledge_graph=_make_graph())
        attacks = [
            AttackResult("t", "LOGIC_FLAW", 0.9, "bad", True),
            AttackResult("t", "SECURITY_VULNERABILITY", 1.0, "worse", True),
        ]
        report = hd._build_report("target", attacks)
        assert report.survived is False
        assert report.survival_score < 0.2

    def test_mixed_severity_revise(self) -> None:
        hd = HouseD(knowledge_graph=_make_graph())
        attacks = [
            AttackResult("t", "LOGIC_FLAW", 0.6, "medium", False),
            AttackResult("t", "HIDDEN_ASSUMPTION", 0.4, "minor", False),
        ]
        report = hd._build_report("target", attacks)
        assert report.survived is True
        assert report.recommendation == "REVISE"

    def test_high_severity_nonfatal_reject(self) -> None:
        hd = HouseD(knowledge_graph=_make_graph())
        attacks = [
            AttackResult("t", "LOGIC_FLAW", 0.8, "bad", False),
            AttackResult("t", "LOGIC_FLAW", 0.9, "bad", False),
        ]
        report = hd._build_report("target", attacks)
        assert report.survived is True
        assert report.recommendation == "REJECT"

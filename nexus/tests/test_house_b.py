"""Tests for House B — The Oracle.

These tests mock litellm.completion so they run without an API
key. They verify prompt construction, JSON parsing, retry logic,
knowledge context building, and the contract that an SSO is never
returned without a MinorityReport.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from nexus.core.belief_certificate import BeliefCertificate
from nexus.core.house_b import (
    HouseB,
    MinorityReport,
    StructuredSpecificationObject,
)
from nexus.core.knowledge_graph import KnowledgeGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cert(
    claim: str = "test claim",
    confidence: float = 0.9,
    domain: str = "Testing",
) -> BeliefCertificate:
    """Factory for valid test certificates."""
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


def _make_graph(*certs: BeliefCertificate) -> KnowledgeGraph:
    """Build a KnowledgeGraph from a list of certificates."""
    graph = KnowledgeGraph()
    for c in certs:
        graph.add_belief(c)
    return graph


def _fake_response(payload: dict[str, Any]) -> MagicMock:
    """Build a mock litellm completion response.

    litellm.completion() returns a response where
    response.choices[0].message.content is a string.
    """
    msg = MagicMock()
    msg.content = json.dumps(payload)
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


SAMPLE_REDEFINE: dict[str, Any] = {
    "redefined_problem": "The real problem is X",
    "assumptions": ["A1", "A2"],
    "constraints": ["C1"],
    "success_criteria": ["S1"],
    "required_inputs": ["I1"],
    "expected_outputs": ["O1"],
    "domain": "ML Architecture",
    "confidence": 0.85,
}

SAMPLE_MINORITY: dict[str, Any] = {
    "reasons_to_fail": ["R1", "R2"],
    "risks": ["Risk1"],
    "hidden_assumptions": ["H1"],
    "better_alternatives": ["Alt1"],
}


# ---------------------------------------------------------------------------
# MinorityReport unit tests
# ---------------------------------------------------------------------------

class TestMinorityReport:
    """Tests for the MinorityReport dataclass."""

    def test_round_trip(self) -> None:
        mr = MinorityReport(
            reasons_to_fail=["will crash"],
            risks=["data loss"],
            hidden_assumptions=["unlimited memory"],
            better_alternatives=["use SQLite"],
        )
        restored = MinorityReport.from_dict(mr.to_dict())
        assert restored.reasons_to_fail == mr.reasons_to_fail
        assert restored.risks == mr.risks
        assert restored.hidden_assumptions == mr.hidden_assumptions
        assert restored.better_alternatives == mr.better_alternatives

    def test_defaults(self) -> None:
        mr = MinorityReport()
        assert mr.reasons_to_fail == []
        assert mr.risks == []

    def test_from_dict_missing_keys(self) -> None:
        mr = MinorityReport.from_dict({})
        assert mr.reasons_to_fail == []


# ---------------------------------------------------------------------------
# StructuredSpecificationObject unit tests
# ---------------------------------------------------------------------------

class TestSSO:
    """Tests for the StructuredSpecificationObject dataclass."""

    def test_round_trip(self) -> None:
        sso = StructuredSpecificationObject(
            original_input="make it fast",
            redefined_problem="optimise latency",
            assumptions=["low traffic"],
            constraints=["budget $0"],
            success_criteria=["p99 < 100ms"],
            required_inputs=["metrics"],
            expected_outputs=["config"],
            domain="Infra",
            confidence=0.9,
            minority_report=MinorityReport(reasons_to_fail=["cold starts"]),
        )
        data = sso.to_dict()
        restored = StructuredSpecificationObject.from_dict(data)
        assert restored.redefined_problem == sso.redefined_problem
        assert restored.minority_report.reasons_to_fail == ["cold starts"]

    def test_default_minority_report(self) -> None:
        sso = StructuredSpecificationObject(
            original_input="x", redefined_problem="y",
        )
        assert isinstance(sso.minority_report, MinorityReport)


# ---------------------------------------------------------------------------
# HouseB._build_knowledge_context
# ---------------------------------------------------------------------------

class TestBuildKnowledgeContext:
    """Tests for knowledge context assembly."""

    def test_empty_graph(self) -> None:
        graph = _make_graph()
        hb = HouseB(knowledge_graph=graph)
        ctx = hb._build_knowledge_context("Anything")
        assert ctx == "(no verified knowledge available)"

    def test_returns_top_5_sorted_by_confidence(self) -> None:
        certs = [
            _make_cert(claim=f"claim-{i}", confidence=round(0.6 + i * 0.05, 2))
            for i in range(8)
        ]
        graph = _make_graph(*certs)
        hb = HouseB(knowledge_graph=graph)
        ctx = hb._build_knowledge_context("Testing")
        lines = ctx.strip().split("\n")
        assert len(lines) == 5
        assert "claim-7" in lines[0]

    def test_mixes_domains_when_fewer_than_5(self) -> None:
        graph = _make_graph(
            _make_cert(claim="sec1", domain="Security"),
            _make_cert(claim="ml1", domain="ML"),
        )
        hb = HouseB(knowledge_graph=graph)
        ctx = hb._build_knowledge_context("Security")
        assert "sec1" in ctx
        assert "ml1" in ctx


# ---------------------------------------------------------------------------
# HouseB._parse_json
# ---------------------------------------------------------------------------

class TestParseJson:
    """Tests for the JSON parser with retry logic."""

    def test_valid_json(self) -> None:
        hb = HouseB(knowledge_graph=_make_graph())
        result = hb._parse_json('{"a": 1}', label="test")
        assert result == {"a": 1}

    def test_strips_markdown_fences(self) -> None:
        hb = HouseB(knowledge_graph=_make_graph())
        raw = '```json\n{"b": 2}\n```'
        result = hb._parse_json(raw, label="test")
        assert result == {"b": 2}

    def test_raises_after_retry(self) -> None:
        hb = HouseB(knowledge_graph=_make_graph())
        with pytest.raises(ValueError, match="invalid JSON"):
            hb._parse_json("not json at all {{{", label="test")


# ---------------------------------------------------------------------------
# HouseB.redefine (mocked LLM)
# ---------------------------------------------------------------------------

class TestHouseBRedefine:
    """Integration tests for the full redefine pipeline with mocked LLM."""

    @patch("nexus.core.model_router.litellm")
    def test_redefine_returns_complete_sso(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.side_effect = [
            _fake_response(SAMPLE_REDEFINE),
            _fake_response(SAMPLE_MINORITY),
        ]

        graph = _make_graph(_make_cert(claim="known fact", domain="ML Architecture"))
        hb = HouseB(knowledge_graph=graph)
        sso = hb.redefine("Build me an ML pipeline")

        assert sso.original_input == "Build me an ML pipeline"
        assert sso.redefined_problem == "The real problem is X"
        assert sso.domain == "ML Architecture"
        assert sso.confidence == 0.85
        assert sso.assumptions == ["A1", "A2"]
        assert sso.constraints == ["C1"]

    @patch("nexus.core.model_router.litellm")
    def test_minority_report_always_attached(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.side_effect = [
            _fake_response(SAMPLE_REDEFINE),
            _fake_response(SAMPLE_MINORITY),
        ]

        hb = HouseB(knowledge_graph=_make_graph())
        sso = hb.redefine("anything")

        assert isinstance(sso.minority_report, MinorityReport)
        assert sso.minority_report.reasons_to_fail == ["R1", "R2"]
        assert sso.minority_report.risks == ["Risk1"]
        assert sso.minority_report.hidden_assumptions == ["H1"]
        assert sso.minority_report.better_alternatives == ["Alt1"]

    @patch("nexus.core.model_router.litellm")
    def test_two_llm_calls_made(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.side_effect = [
            _fake_response(SAMPLE_REDEFINE),
            _fake_response(SAMPLE_MINORITY),
        ]

        hb = HouseB(knowledge_graph=_make_graph())
        hb.redefine("test")

        assert mock_litellm.completion.call_count == 2
        calls = mock_litellm.completion.call_args_list
        first_system = calls[0][1]["messages"][0]["content"]
        second_system = calls[1][1]["messages"][0]["content"]
        assert "House B" in first_system
        assert "destroyer" in second_system

    @patch("nexus.core.model_router.litellm")
    def test_knowledge_context_included_in_prompt(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.side_effect = [
            _fake_response(SAMPLE_REDEFINE),
            _fake_response(SAMPLE_MINORITY),
        ]

        graph = _make_graph(
            _make_cert(claim="Transformers use attention", domain="ML"),
        )
        hb = HouseB(knowledge_graph=graph)
        hb.redefine("Help with ML")

        user_msg = mock_litellm.completion.call_args_list[0][1]["messages"][1]["content"]
        assert "Transformers use attention" in user_msg

    @patch("nexus.core.model_router.litellm")
    def test_redefine_raises_on_invalid_json(self, mock_litellm: MagicMock) -> None:
        bad_msg = MagicMock()
        bad_msg.content = "not valid json {{{"
        bad_choice = MagicMock()
        bad_choice.message = bad_msg
        bad_resp = MagicMock()
        bad_resp.choices = [bad_choice]
        mock_litellm.completion.return_value = bad_resp

        hb = HouseB(knowledge_graph=_make_graph())
        with pytest.raises(ValueError, match="invalid JSON"):
            hb.redefine("test")


# ---------------------------------------------------------------------------
# HouseB._generate_minority_report (mocked LLM)
# ---------------------------------------------------------------------------

class TestGenerateMinorityReport:
    """Tests for the isolated minority report generation."""

    @patch("nexus.core.model_router.litellm")
    def test_generates_report(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _fake_response(SAMPLE_MINORITY)

        hb = HouseB(knowledge_graph=_make_graph())
        sso = StructuredSpecificationObject(
            original_input="x",
            redefined_problem="y",
            domain="General",
        )
        report = hb._generate_minority_report(sso)

        assert report.reasons_to_fail == ["R1", "R2"]
        assert report.risks == ["Risk1"]
        assert report.better_alternatives == ["Alt1"]

    @patch("nexus.core.model_router.litellm")
    def test_system_prompt_is_house_d(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _fake_response(SAMPLE_MINORITY)

        hb = HouseB(knowledge_graph=_make_graph())
        sso = StructuredSpecificationObject(
            original_input="x", redefined_problem="y",
        )
        hb._generate_minority_report(sso)

        system = mock_litellm.completion.call_args[1]["messages"][0]["content"]
        assert "destroyer" in system
        assert "House D" in system

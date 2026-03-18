"""Tests for the NEXUS knowledge graph, House-A engine, and supporting types."""

from __future__ import annotations

import json
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from unittest.mock import patch

from nexus.core.belief_certificate import BeliefCertificate
from nexus.core.house_a import AuditEntry, HouseA
from nexus.core.knowledge_graph import KnowledgeGraph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_cert(
    claim: str = "test claim",
    source: str = "unit-test",
    confidence: float = 0.9,
    decay_rate: float = 0.0,
    executable_proof: str | None = "assert True",
    domain: str = "Testing",
    contradictions: list[str] | None = None,
    downstream_dependents: list[str] | None = None,
    last_verified: datetime | None = None,
) -> BeliefCertificate:
    """Factory for test BeliefCertificates."""
    return BeliefCertificate(
        claim=claim,
        source=source,
        confidence=confidence,
        contradictions=contradictions or [],
        decay_rate=decay_rate,
        created_at=datetime.now(timezone.utc),
        last_verified=last_verified or datetime.now(timezone.utc),
        downstream_dependents=downstream_dependents or [],
        executable_proof=executable_proof,
        domain=domain,
    )


def _make_graph(storage_path: str | None = None) -> KnowledgeGraph:
    """Return an empty KnowledgeGraph with isolated storage."""
    if storage_path is None:
        storage_path = str(
            Path(tempfile.gettempdir()) / f"nexus_test_{uuid.uuid4().hex}.json"
        )
    return KnowledgeGraph(storage_path=storage_path)


# ---------------------------------------------------------------------------
# BeliefCertificate unit tests
# ---------------------------------------------------------------------------

class TestBeliefCertificate:
    """Tests for the BeliefCertificate dataclass."""

    def test_is_valid_with_proof_and_high_confidence(self) -> None:
        cert = _make_cert(confidence=0.8, executable_proof="assert 1 == 1")
        assert cert.is_valid() is True

    def test_is_valid_fails_low_confidence(self) -> None:
        cert = _make_cert(confidence=0.3, executable_proof="assert True")
        assert cert.is_valid() is False

    def test_is_valid_fails_no_proof(self) -> None:
        cert = _make_cert(confidence=0.9, executable_proof=None)
        assert cert.is_valid() is False

    def test_is_expired_fresh_certificate(self) -> None:
        cert = _make_cert(decay_rate=0.5)
        assert cert.is_expired() is False

    def test_is_expired_old_certificate(self) -> None:
        long_ago = datetime.now(timezone.utc) - timedelta(days=400)
        cert = _make_cert(decay_rate=0.0, last_verified=long_ago)
        assert cert.is_expired() is True

    def test_is_expired_high_decay_rate(self) -> None:
        recently = datetime.now(timezone.utc) - timedelta(days=40)
        cert = _make_cert(decay_rate=0.9, last_verified=recently)
        assert cert.is_expired() is True

    def test_round_trip_serialisation(self) -> None:
        cert = _make_cert(contradictions=["counter-claim"], downstream_dependents=["dep-1"])
        data = cert.to_dict()
        restored = BeliefCertificate.from_dict(data)
        assert restored.claim == cert.claim
        assert restored.confidence == cert.confidence
        assert restored.contradictions == cert.contradictions
        assert restored.downstream_dependents == cert.downstream_dependents
        assert restored.executable_proof == cert.executable_proof

    def test_confidence_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            _make_cert(confidence=1.5)

    def test_decay_rate_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="decay_rate"):
            _make_cert(decay_rate=-0.1)


# ---------------------------------------------------------------------------
# KnowledgeGraph — add_belief gating
# ---------------------------------------------------------------------------

class TestKnowledgeGraphAddBelief:
    """Tests for the gated add_belief method."""

    def test_accept_valid_belief(self) -> None:
        graph = _make_graph()
        cert = _make_cert(claim="good belief", confidence=0.9)
        assert graph.add_belief(cert) is True
        assert "good belief" in graph

    @patch("nexus.core.knowledge_graph.run_executable_proof_in_subprocess")
    def test_reject_when_subprocess_proof_fails(
        self, mock_proof: object,
    ) -> None:
        mock_proof.return_value = (False, "proof failed")
        graph = _make_graph()
        assert graph.add_belief(_make_cert(claim="bad proof")) is False
        assert "bad proof" not in graph

    @patch("nexus.core.knowledge_graph.run_executable_proof_in_subprocess")
    def test_subprocess_proof_sets_last_verified_at(
        self, mock_proof: object,
    ) -> None:
        mock_proof.return_value = (True, "")
        graph = _make_graph()
        assert graph.add_belief(_make_cert(claim="verified")) is True
        b = graph.get_belief("verified")
        assert b is not None
        assert b.last_verified_at is not None
        assert b.verification_status == "VERIFIED"

    def test_reverify_quarantines_and_governor_alert(self) -> None:
        graph = _make_graph()
        alerts: list[dict] = []
        graph.governor_alert = lambda a: alerts.append(a)
        assert graph.add_belief(_make_cert(claim="reverify-me")) is True
        b = graph.get_belief("reverify-me")
        assert b is not None
        b.last_verified_at = datetime.now(timezone.utc) - timedelta(hours=25)

        with patch(
            "nexus.core.knowledge_graph.run_executable_proof_in_subprocess",
            return_value=(False, "re-verify failed"),
        ):
            r = graph.reverify_beliefs_past_due()
        assert r["quarantined"] == 1
        b2 = graph.get_belief("reverify-me")
        assert b2 is not None
        assert b2.quarantined is True
        assert b2.verification_status == "UNVERIFIED"
        assert any(x.get("type") == "UNVERIFIED" for x in alerts)

    def test_reject_expired_belief(self) -> None:
        graph = _make_graph()
        old = _make_cert(
            claim="stale",
            decay_rate=0.0,
            last_verified=datetime.now(timezone.utc) - timedelta(days=400),
        )
        assert graph.add_belief(old) is False
        assert "stale" not in graph

    def test_reject_invalid_belief_no_proof(self) -> None:
        graph = _make_graph()
        no_proof = _make_cert(claim="no proof", executable_proof=None)
        assert graph.add_belief(no_proof) is False
        assert "no proof" not in graph

    def test_reject_low_confidence(self) -> None:
        graph = _make_graph()
        low = _make_cert(claim="low conf", confidence=0.3, executable_proof="assert True")
        assert graph.add_belief(low) is False

    def test_reject_contradiction(self) -> None:
        graph = _make_graph()
        graph.add_belief(_make_cert(
            claim="A is true",
            contradictions=["A is false"],
        ))
        conflicting = _make_cert(claim="A is false")
        assert graph.add_belief(conflicting) is False
        assert "A is false" not in graph

    def test_updates_domain_index(self) -> None:
        graph = _make_graph()
        graph.add_belief(_make_cert(claim="sec1", domain="Security"))
        assert "Security" in graph.domain_index
        assert "sec1" in graph.domain_index["Security"]

    def test_updates_graph_adjacency(self) -> None:
        graph = _make_graph()
        graph.add_belief(_make_cert(
            claim="parent",
            downstream_dependents=["child"],
        ))
        assert "parent" in graph.graph.get("child", set())


# ---------------------------------------------------------------------------
# KnowledgeGraph — get_belief
# ---------------------------------------------------------------------------

class TestKnowledgeGraphGetBelief:
    """Tests for get_belief with auto-pruning."""

    def test_get_existing_belief(self) -> None:
        graph = _make_graph()
        cert = _make_cert(claim="alpha")
        graph.add_belief(cert)
        assert graph.get_belief("alpha") is cert

    def test_get_missing_returns_none(self) -> None:
        graph = _make_graph()
        assert graph.get_belief("nonexistent") is None

    def test_get_auto_prunes_expired(self) -> None:
        graph = _make_graph()
        cert = _make_cert(claim="will-expire")
        graph.add_belief(cert)
        cert.last_verified = datetime.now(timezone.utc) - timedelta(days=400)
        assert graph.get_belief("will-expire") is None
        assert "will-expire" not in graph


# ---------------------------------------------------------------------------
# KnowledgeGraph — query_domain
# ---------------------------------------------------------------------------

class TestKnowledgeGraphQueryDomain:
    """Tests for domain-scoped queries."""

    def test_returns_matching_domain(self) -> None:
        graph = _make_graph()
        graph.add_belief(_make_cert(claim="s1", domain="Security"))
        graph.add_belief(_make_cert(claim="m1", domain="ML"))
        results = graph.query_domain("Security")
        assert len(results) == 1
        assert results[0].claim == "s1"

    def test_empty_domain_returns_empty(self) -> None:
        graph = _make_graph()
        assert graph.query_domain("Nonexistent") == []

    def test_auto_prunes_expired_during_query(self) -> None:
        graph = _make_graph()
        cert = _make_cert(claim="expires-soon", domain="Test")
        graph.add_belief(cert)
        cert.last_verified = datetime.now(timezone.utc) - timedelta(days=400)
        results = graph.query_domain("Test")
        assert len(results) == 0
        assert "expires-soon" not in graph


# ---------------------------------------------------------------------------
# KnowledgeGraph — get_dependents
# ---------------------------------------------------------------------------

class TestKnowledgeGraphGetDependents:
    """Tests for dependency lookups."""

    def test_returns_dependents(self) -> None:
        graph = _make_graph()
        graph.add_belief(_make_cert(
            claim="parent",
            downstream_dependents=["child"],
        ))
        graph.add_belief(_make_cert(claim="child"))
        deps = graph.get_dependents("child")
        assert len(deps) == 1
        assert deps[0].claim == "parent"

    def test_no_dependents(self) -> None:
        graph = _make_graph()
        graph.add_belief(_make_cert(claim="lonely"))
        assert graph.get_dependents("lonely") == []


# ---------------------------------------------------------------------------
# KnowledgeGraph — prune_expired
# ---------------------------------------------------------------------------

class TestKnowledgeGraphPruneExpired:
    """Tests for bulk expiration pruning."""

    def test_prune_removes_expired(self) -> None:
        graph = _make_graph()
        cert = _make_cert(claim="old")
        graph.add_belief(cert)
        cert.last_verified = datetime.now(timezone.utc) - timedelta(days=400)
        count = graph.prune_expired()
        assert count == 1
        assert "old" not in graph

    def test_prune_keeps_fresh(self) -> None:
        graph = _make_graph()
        graph.add_belief(_make_cert(claim="fresh"))
        count = graph.prune_expired()
        assert count == 0
        assert "fresh" in graph

    def test_prune_cleans_domain_index(self) -> None:
        graph = _make_graph()
        cert = _make_cert(claim="idx-test", domain="TestDomain")
        graph.add_belief(cert)
        cert.last_verified = datetime.now(timezone.utc) - timedelta(days=400)
        graph.prune_expired()
        assert "TestDomain" not in graph.domain_index


# ---------------------------------------------------------------------------
# KnowledgeGraph — contradiction_check
# ---------------------------------------------------------------------------

class TestKnowledgeGraphContradictionCheck:
    """Tests for contradiction detection."""

    def test_detects_contradiction(self) -> None:
        graph = _make_graph()
        existing = _make_cert(claim="X is true", contradictions=["X is false"])
        graph.add_belief(existing)
        conflicts = graph.contradiction_check("X is false", graph.beliefs_snapshot())
        assert "X is true" in conflicts

    def test_no_contradiction(self) -> None:
        graph = _make_graph()
        existing = _make_cert(claim="Y is true")
        graph.add_belief(existing)
        conflicts = graph.contradiction_check("Z is true", graph.beliefs_snapshot())
        assert conflicts == []


# ---------------------------------------------------------------------------
# KnowledgeGraph — inject_external_signal (IRON LAW)
# ---------------------------------------------------------------------------

class TestKnowledgeGraphInject:
    """Tests for the IRON LAW external injection method."""

    def test_inject_adds_valid_beliefs(self) -> None:
        graph = _make_graph()
        batch = [
            _make_cert(claim="ext-1"),
            _make_cert(claim="ext-2"),
        ]
        result = graph.inject_external_signal(batch)
        assert result["added"] == 2
        assert result["rejected"] == 0
        assert result["contradictions"] == []

    def test_inject_rejects_invalid(self) -> None:
        graph = _make_graph()
        batch = [
            _make_cert(claim="good", confidence=0.9),
            _make_cert(claim="bad", confidence=0.3, executable_proof="assert True"),
        ]
        result = graph.inject_external_signal(batch)
        assert result["added"] == 1
        assert result["rejected"] == 1

    def test_inject_reports_contradictions(self) -> None:
        graph = _make_graph()
        graph.add_belief(_make_cert(
            claim="earth is round",
            contradictions=["earth is flat"],
        ))
        batch = [_make_cert(claim="earth is flat")]
        result = graph.inject_external_signal(batch)
        assert result["rejected"] == 1
        assert "earth is flat" in result["contradictions"]

    def test_inject_mixed_batch(self) -> None:
        graph = _make_graph()
        graph.add_belief(_make_cert(
            claim="existing",
            contradictions=["conflicting"],
        ))
        batch = [
            _make_cert(claim="valid-new"),
            _make_cert(claim="conflicting"),
            _make_cert(claim="no-proof", executable_proof=None),
        ]
        result = graph.inject_external_signal(batch)
        assert result["added"] == 1
        assert result["rejected"] == 2
        assert "conflicting" in result["contradictions"]


# ---------------------------------------------------------------------------
# KnowledgeGraph — health_report
# ---------------------------------------------------------------------------

class TestKnowledgeGraphHealthReport:
    """Tests for the diagnostic health report."""

    def test_empty_graph_report(self) -> None:
        graph = _make_graph()
        report = graph.health_report()
        assert report["total_beliefs"] == 0
        assert report["expired_count"] == 0
        assert report["average_confidence"] == 0.0
        assert report["without_executable_proof"] == 0

    def test_populated_graph_report(self) -> None:
        graph = _make_graph()
        graph.add_belief(_make_cert(claim="a", confidence=0.8, domain="D1"))
        graph.add_belief(_make_cert(claim="b", confidence=0.6, domain="D1"))
        graph.add_belief(_make_cert(claim="c", confidence=0.9, domain="D2"))
        report = graph.health_report()
        assert report["total_beliefs"] == 3
        assert report["domain_breakdown"]["D1"] == 2
        assert report["domain_breakdown"]["D2"] == 1
        assert report["average_confidence"] == pytest.approx(
            round((0.8 + 0.6 + 0.9) / 3, 4),
        )
        assert report["without_executable_proof"] == 0

    def test_report_counts_without_proof(self) -> None:
        graph = _make_graph()
        cert = _make_cert(claim="proved", confidence=0.9)
        graph.add_belief(cert)
        cert.executable_proof = None
        report = graph.health_report()
        assert report["without_executable_proof"] == 1


# ---------------------------------------------------------------------------
# KnowledgeGraph — persistence
# ---------------------------------------------------------------------------

class TestKnowledgeGraphPersistence:
    """Tests for persistence round-trip via PersistenceManager."""

    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "beliefs.json")
            graph = KnowledgeGraph(storage_path=path)
            graph.add_belief(_make_cert(claim="persist-me", domain="Persist"))
            new_graph = KnowledgeGraph(storage_path=path)
            assert "persist-me" in new_graph
            assert "Persist" in new_graph.domain_index

    def test_load_rebuilds_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "beliefs.json")
            graph = KnowledgeGraph(storage_path=path)
            graph.add_belief(_make_cert(
                claim="parent",
                downstream_dependents=["child"],
                domain="Deps",
            ))
            loaded = KnowledgeGraph(storage_path=path)
            assert "parent" in loaded.graph.get("child", set())
            assert "Deps" in loaded.domain_index


# ---------------------------------------------------------------------------
# KnowledgeGraph — dunder methods
# ---------------------------------------------------------------------------

class TestKnowledgeGraphDunders:
    """Tests for __len__, __iter__, __contains__."""

    def test_len(self) -> None:
        graph = _make_graph()
        graph.add_belief(_make_cert(claim="a"))
        graph.add_belief(_make_cert(claim="b"))
        assert len(graph) == 2

    def test_contains(self) -> None:
        graph = _make_graph()
        graph.add_belief(_make_cert(claim="exists"))
        assert "exists" in graph
        assert "missing" not in graph

    def test_iter(self) -> None:
        graph = _make_graph()
        graph.add_belief(_make_cert(claim="x"))
        graph.add_belief(_make_cert(claim="y"))
        claims = {b.claim for b in graph}
        assert claims == {"x", "y"}


# ---------------------------------------------------------------------------
# HouseA integration tests
# ---------------------------------------------------------------------------

class TestHouseA:
    """Tests for the House-A reasoning engine."""

    def test_prune_expired(self) -> None:
        graph = _make_graph()
        old = _make_cert(claim="stale")
        graph.add_belief(old)
        old.last_verified = datetime.now(timezone.utc) - timedelta(days=400)
        graph.add_belief(_make_cert(claim="fresh"))

        engine = HouseA(graph=graph)
        pruned = engine.prune_expired()

        assert len(pruned) == 1
        assert pruned[0].claim == "stale"
        assert "stale" not in graph

    def test_detect_contradictions(self) -> None:
        graph = _make_graph()
        graph.add_belief(_make_cert(claim="A is true", contradictions=["A is false"]))
        # Insert the contradicting belief *first* under a non-conflicting name,
        # then add the contradiction reference.  We bypass the gate by writing
        # directly to beliefs so HouseA can test its own detection logic.
        contra = _make_cert(claim="A is false")
        graph.register_belief_bypass_gates(contra)

        engine = HouseA(graph=graph)
        results = engine.detect_contradictions()

        assert len(results) == 1
        cert, contras = results[0]
        assert cert.claim == "A is true"
        assert contras[0].claim == "A is false"

    def test_propagate_decay(self) -> None:
        graph = _make_graph()
        weak = _make_cert(claim="weak parent", confidence=0.9)
        graph.add_belief(weak)
        weak.confidence = 0.3
        weak.executable_proof = None

        child = _make_cert(
            claim="strong child",
            confidence=0.9,
            downstream_dependents=["weak parent"],
        )
        graph.add_belief(child)

        engine = HouseA(graph=graph, propagation_factor=0.5)
        affected = engine.propagate_decay()

        assert len(affected) == 1
        assert affected[0].claim == "strong child"
        assert affected[0].confidence == pytest.approx(0.45)

    def test_evaluate_returns_summary(self) -> None:
        graph = _make_graph()
        graph.add_belief(_make_cert(claim="ok"))
        engine = HouseA(graph=graph)
        summary = engine.evaluate()
        assert "pruned_count" in summary
        assert "valid_beliefs" in summary

    def test_audit_log_populated(self) -> None:
        graph = _make_graph()
        old = _make_cert(claim="old-claim")
        graph.add_belief(old)
        old.last_verified = datetime.now(timezone.utc) - timedelta(days=400)

        engine = HouseA(graph=graph)
        engine.evaluate()

        log = engine.get_audit_log()
        assert len(log) >= 1
        assert log[0]["action"] == "pruned"

"""Tests for PROXY architectural beliefs — TDD.

Coverage:
  1. Module + list exist
  2. Exactly 5 beliefs defined
  3. Each belief: is_axiom=True, confidence=0.95, decay_rate=0.0
  4. Each belief: claim text matches spec (key phrases present)
  5. Each belief: executable_proof passes when exec'd
  6. Each belief: is_valid() returns True
  7. Integration: inject_external_signal adds all 5 to KnowledgeGraph
  8. Integration: build_nexus() seeds architecture beliefs on boot
"""

from __future__ import annotations

import textwrap
from unittest.mock import MagicMock, patch

import pytest

import nexus.core.architecture_beliefs as arch_mod
from nexus.core.architecture_beliefs import ARCHITECTURE_BELIEFS
from nexus.core.belief_certificate import BeliefCertificate
from nexus.core.knowledge_graph import KnowledgeGraph


# ══════════════════════════════════════════════════════════════════
#  1. Module and list exist
# ══════════════════════════════════════════════════════════════════

class TestModuleExists:
    def test_module_importable(self):
        import nexus.core.architecture_beliefs  # noqa: F401

    def test_architecture_beliefs_exported(self):
        assert hasattr(arch_mod, "ARCHITECTURE_BELIEFS")

    def test_architecture_beliefs_is_list(self):
        assert isinstance(ARCHITECTURE_BELIEFS, list)


# ══════════════════════════════════════════════════════════════════
#  2. Exactly 5 beliefs
# ══════════════════════════════════════════════════════════════════

class TestBeliefCount:
    def test_exactly_five_beliefs(self):
        assert len(ARCHITECTURE_BELIEFS) == 5, (
            f"Expected 5 architectural beliefs, got {len(ARCHITECTURE_BELIEFS)}"
        )

    def test_all_are_belief_certificates(self):
        for b in ARCHITECTURE_BELIEFS:
            assert isinstance(b, BeliefCertificate), f"Not a BeliefCertificate: {b!r}"


# ══════════════════════════════════════════════════════════════════
#  3. Axiom flags and numeric fields
# ══════════════════════════════════════════════════════════════════

class TestBeliefFlags:
    @pytest.mark.parametrize("belief", ARCHITECTURE_BELIEFS)
    def test_is_axiom_true(self, belief):
        assert belief.is_axiom is True, f"{belief.claim[:40]!r}: is_axiom must be True"

    @pytest.mark.parametrize("belief", ARCHITECTURE_BELIEFS)
    def test_confidence_095(self, belief):
        assert belief.confidence == 0.95, (
            f"{belief.claim[:40]!r}: confidence must be 0.95, got {belief.confidence}"
        )

    @pytest.mark.parametrize("belief", ARCHITECTURE_BELIEFS)
    def test_decay_rate_zero(self, belief):
        assert belief.decay_rate == 0.0, (
            f"{belief.claim[:40]!r}: decay_rate must be 0.0 (axioms never expire)"
        )

    @pytest.mark.parametrize("belief", ARCHITECTURE_BELIEFS)
    def test_source_is_architecture_axiom(self, belief):
        assert "architecture" in belief.source.lower() or "axiom" in belief.source.lower(), (
            f"{belief.claim[:40]!r}: source should reference 'architecture axiom'"
        )


# ══════════════════════════════════════════════════════════════════
#  4. Claim text — key phrases from spec
# ══════════════════════════════════════════════════════════════════

class TestClaimContent:
    def _claims(self) -> list[str]:
        return [b.claim.lower() for b in ARCHITECTURE_BELIEFS]

    def test_ooda_loop_belief_exists(self):
        claims = self._claims()
        assert any("ooda" in c for c in claims), "Missing OODA loop belief"

    def test_ooda_includes_observe_orient_decide_act(self):
        ooda = next(b for b in ARCHITECTURE_BELIEFS if "ooda" in b.claim.lower())
        low = ooda.claim.lower()
        for keyword in ("observe", "orient", "decide", "act"):
            assert keyword in low, f"OODA belief missing '{keyword}'"

    def test_routing_belief_exists(self):
        claims = self._claims()
        assert any(
            ("route" in c or "routing" in c or "complexity" in c) for c in claims
        ), "Missing task routing belief"

    def test_routing_mentions_cheap_and_quality(self):
        routing = next(
            b for b in ARCHITECTURE_BELIEFS
            if "route" in b.claim.lower() or "routing" in b.claim.lower() or "complexity" in b.claim.lower()
        )
        low = routing.claim.lower()
        assert "simple" in low or "complex" in low or "cheap" in low or "quality" in low, (
            "Routing belief should mention simple/complex tiers"
        )

    def test_confidence_routing_belief_exists(self):
        claims = self._claims()
        assert any("confidence" in c for c in claims), "Missing confidence routing belief"

    def test_confidence_routing_has_thresholds(self):
        conf = next(b for b in ARCHITECTURE_BELIEFS if "confidence" in b.claim.lower())
        low = conf.claim.lower()
        assert "0.85" in low or "0.6" in low or "85" in low or "60" in low, (
            "Confidence routing belief should reference numeric thresholds"
        )

    def test_skill_library_belief_exists(self):
        claims = self._claims()
        assert any(
            "skill" in c or "reusable" in c or "anti-belief" in c or "failure" in c
            for c in claims
        ), "Missing skill library / failure learning belief"

    def test_skill_library_mentions_success_and_failure(self):
        skill = next(
            b for b in ARCHITECTURE_BELIEFS
            if "skill" in b.claim.lower() or "reusable" in b.claim.lower()
            or "anti-belief" in b.claim.lower()
        )
        low = skill.claim.lower()
        assert "success" in low or "failure" in low, (
            "Skill library belief must mention success and/or failure"
        )

    def test_ai_tools_belief_exists(self):
        claims = self._claims()
        assert any(
            ("ai" in c and ("tool" in c or "capabilit" in c)) or "weekly" in c
            for c in claims
        ), "Missing AI tools / continuous improvement belief"

    def test_ai_tools_mentions_weekly_cadence(self):
        ai_belief = next(
            b for b in ARCHITECTURE_BELIEFS
            if ("ai" in b.claim.lower() and "tool" in b.claim.lower()) or "weekly" in b.claim.lower()
        )
        assert "weekly" in ai_belief.claim.lower(), (
            "AI tools belief should specify weekly cadence"
        )


# ══════════════════════════════════════════════════════════════════
#  5. Executable proofs compile and run without error
# ══════════════════════════════════════════════════════════════════

class TestExecutableProofs:
    @pytest.mark.parametrize("belief", ARCHITECTURE_BELIEFS)
    def test_proof_is_non_empty(self, belief):
        assert belief.executable_proof, f"{belief.claim[:40]!r}: executable_proof is empty"

    @pytest.mark.parametrize("belief", ARCHITECTURE_BELIEFS)
    def test_proof_executes_without_error(self, belief):
        code = textwrap.dedent(belief.executable_proof)
        try:
            exec(compile(code, "<proof>", "exec"), {})
        except Exception as exc:
            pytest.fail(
                f"executable_proof for {belief.claim[:40]!r} raised: {exc}\n"
                f"Code:\n{code}"
            )


# ══════════════════════════════════════════════════════════════════
#  6. is_valid() returns True for all beliefs
# ══════════════════════════════════════════════════════════════════

class TestBeliefValidity:
    @pytest.mark.parametrize("belief", ARCHITECTURE_BELIEFS)
    def test_is_valid(self, belief):
        assert belief.is_valid(), (
            f"{belief.claim[:40]!r}: is_valid() returned False — "
            "check confidence, executable_proof, and is_expired()"
        )


# ══════════════════════════════════════════════════════════════════
#  7. Integration: inject_external_signal adds all 5 to graph
# ══════════════════════════════════════════════════════════════════

class TestKnowledgeGraphInjection:
    def test_all_beliefs_injected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NEXUS_DATA_DIR", str(tmp_path))
        graph = KnowledgeGraph()
        result = graph.inject_external_signal(ARCHITECTURE_BELIEFS)
        assert result["added"] == 5, (
            f"Expected 5 beliefs added, got {result['added']} "
            f"(rejected={result['rejected']})"
        )

    def test_beliefs_retrievable_from_graph(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NEXUS_DATA_DIR", str(tmp_path))
        graph = KnowledgeGraph()
        graph.inject_external_signal(ARCHITECTURE_BELIEFS)
        claims_in_graph = {b.claim for b in graph.beliefs_snapshot()}
        for belief in ARCHITECTURE_BELIEFS:
            assert belief.claim in claims_in_graph, (
                f"Belief not found in graph: {belief.claim[:60]!r}"
            )

    def test_injected_beliefs_are_axioms_in_graph(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NEXUS_DATA_DIR", str(tmp_path))
        graph = KnowledgeGraph()
        graph.inject_external_signal(ARCHITECTURE_BELIEFS)
        arch_claims = {b.claim for b in ARCHITECTURE_BELIEFS}
        for b in graph.beliefs_snapshot():
            if b.claim in arch_claims:
                assert b.is_axiom is True, f"Axiom flag lost for: {b.claim[:40]!r}"

    def test_idempotent_injection(self, tmp_path, monkeypatch):
        """Injecting twice must not produce duplicate beliefs in the graph."""
        monkeypatch.setenv("NEXUS_DATA_DIR", str(tmp_path))
        graph = KnowledgeGraph()
        graph.inject_external_signal(ARCHITECTURE_BELIEFS)
        graph.inject_external_signal(ARCHITECTURE_BELIEFS)
        # The graph stores beliefs by claim (dict key) — no duplicates possible.
        arch_claims = [b.claim for b in ARCHITECTURE_BELIEFS]
        count = sum(1 for b in graph.beliefs_snapshot() if b.claim in arch_claims)
        assert count == 5, f"Expected exactly 5 architecture beliefs in graph, got {count}"


# ══════════════════════════════════════════════════════════════════
#  8. build_nexus() seeds architecture beliefs on boot
# ══════════════════════════════════════════════════════════════════

class TestBuildNexusIntegration:
    def test_build_nexus_injects_architecture_beliefs(self, tmp_path, monkeypatch):
        import nexus.main as main_mod
        from nexus.core.guardian import Guardian, GuardianReport, GuardianVault

        # Minimal clean Guardian mock
        mock_vault = MagicMock(spec=GuardianVault)
        mock_vault.has.return_value = False
        mock_report = MagicMock(spec=GuardianReport)
        mock_report.secret_findings = []
        mock_report.passed = True
        mock_guardian = MagicMock(spec=Guardian)
        mock_guardian.vault = mock_vault
        mock_guardian.audit.return_value = mock_report

        beliefs_file = str(tmp_path / "beliefs.json")
        monkeypatch.setattr(
            "nexus.core.persistence.PersistenceManager.__init__",
            lambda self, storage_path=beliefs_file: setattr(
                self, "storage_path", beliefs_file
            ) or setattr(self, "last_load_count", 0),
        )

        with patch("nexus.main.Guardian", return_value=mock_guardian), \
             patch("nexus.main.migrate_key_to_vault"):
            omega = main_mod.build_nexus(
                guardian_vault_path=str(tmp_path / "vault.enc"),
                guardian_master_key="test-key",
                guardian_scan_paths=[str(tmp_path)],
            )

        arch_claims = {b.claim for b in ARCHITECTURE_BELIEFS}
        graph_claims = {b.claim for b in omega.knowledge_graph.beliefs_snapshot()}
        missing = arch_claims - graph_claims
        assert not missing, (
            f"Architecture beliefs missing from graph after build_nexus: {missing}"
        )

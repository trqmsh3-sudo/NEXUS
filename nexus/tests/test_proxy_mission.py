"""Tests for PROXY mission BeliefCertificates.

Defines exactly what PROXY_MISSION_BELIEFS must look like.
All tests fail until nexus/core/proxy_mission.py is written.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from typing import Set

import pytest

from nexus.core.proxy_mission import PROXY_MISSION_BELIEFS
from nexus.core.belief_certificate import BeliefCertificate


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def test_nine_beliefs_defined() -> None:
    """One certificate per mission rule."""
    assert len(PROXY_MISSION_BELIEFS) == 9, (
        f"Expected 9 mission beliefs, got {len(PROXY_MISSION_BELIEFS)}"
    )


def test_all_are_belief_certificate_instances() -> None:
    for b in PROXY_MISSION_BELIEFS:
        assert isinstance(b, BeliefCertificate), f"Not a BeliefCertificate: {b!r}"


# ---------------------------------------------------------------------------
# Axiom flags
# ---------------------------------------------------------------------------

def test_all_flagged_as_axioms() -> None:
    for b in PROXY_MISSION_BELIEFS:
        assert b.is_axiom is True, f"Not flagged as axiom: {b.claim!r}"


def test_all_have_zero_decay_rate() -> None:
    """Mission rules must never expire."""
    for b in PROXY_MISSION_BELIEFS:
        assert b.decay_rate == 0.0, (
            f"Axiom should not decay (decay_rate={b.decay_rate}): {b.claim!r}"
        )


def test_source_is_proxy_mission() -> None:
    for b in PROXY_MISSION_BELIEFS:
        assert "PROXY" in b.source, (
            f"Source should reference PROXY: {b.source!r} -- {b.claim!r}"
        )


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------

def test_all_valid() -> None:
    """is_valid() requires confidence > 0.5 AND executable_proof is not None."""
    for b in PROXY_MISSION_BELIEFS:
        assert b.is_valid(), (
            f"Belief not valid (conf={b.confidence}, proof={b.executable_proof!r}): "
            f"{b.claim!r}"
        )


def test_all_high_confidence() -> None:
    for b in PROXY_MISSION_BELIEFS:
        assert b.confidence >= 0.9, (
            f"Mission rule confidence too low ({b.confidence}): {b.claim!r}"
        )


def test_none_expired() -> None:
    for b in PROXY_MISSION_BELIEFS:
        assert not b.is_expired(), f"Axiom must not start expired: {b.claim!r}"


# ---------------------------------------------------------------------------
# Domain coverage
# ---------------------------------------------------------------------------

EXPECTED_DOMAINS: Set[str] = {
    "Risk Management",
    "Capital Efficiency",
    "Revenue Strategy",
    "Business Intelligence",
    "Business Strategy",
}


def test_domains_are_business_oriented() -> None:
    for b in PROXY_MISSION_BELIEFS:
        assert b.domain in EXPECTED_DOMAINS, (
            f"Unexpected domain {b.domain!r} for: {b.claim!r}"
        )


def test_all_expected_domains_represented() -> None:
    """Every domain in EXPECTED_DOMAINS must appear at least once."""
    actual = {b.domain for b in PROXY_MISSION_BELIEFS}
    missing = EXPECTED_DOMAINS - actual
    assert not missing, f"These domains are missing from mission beliefs: {missing}"


# ---------------------------------------------------------------------------
# Claims content
# ---------------------------------------------------------------------------

CODING_KEYWORDS = {"python", "function", "class", "test", "pytest", "code", "debug"}


def test_claims_contain_no_coding_keywords() -> None:
    for b in PROXY_MISSION_BELIEFS:
        words = set(b.claim.lower().split())
        overlap = words & CODING_KEYWORDS
        assert not overlap, (
            f"Claim contains coding keyword(s) {overlap}: {b.claim!r}"
        )


def test_claims_are_unique() -> None:
    claims = [b.claim for b in PROXY_MISSION_BELIEFS]
    assert len(claims) == len(set(claims)), "Duplicate claims found"


def test_claims_are_non_empty() -> None:
    for b in PROXY_MISSION_BELIEFS:
        assert b.claim.strip(), "Empty claim found"


# ---------------------------------------------------------------------------
# Key rules present
# ---------------------------------------------------------------------------

def _any_claim_contains(keyword: str) -> bool:
    kw = keyword.lower()
    return any(kw in b.claim.lower() for b in PROXY_MISSION_BELIEFS)


def test_legal_rule_present() -> None:
    assert _any_claim_contains("legal"), "No belief covering the legal rule"


def test_risk_cap_rule_present() -> None:
    assert _any_claim_contains("5%") or _any_claim_contains("five percent"), (
        "No belief covering the 5% risk cap"
    )


def test_revenue_validation_rule_present() -> None:
    assert _any_claim_contains("payment") or _any_claim_contains("paid"), (
        "No belief covering real-payment validation"
    )


def test_guaranteed_returns_rule_present() -> None:
    assert _any_claim_contains("guaranteed") or _any_claim_contains("scam"), (
        "No belief covering the guaranteed-returns = scam rule"
    )


def test_bootstrap_rule_present() -> None:
    assert _any_claim_contains("bootstrap") or _any_claim_contains("safety net"), (
        "No belief covering the bootstrap-only rule"
    )


# ---------------------------------------------------------------------------
# Contradictions populated on rules that need them
# ---------------------------------------------------------------------------

def test_risk_belief_has_contradictions() -> None:
    risk_beliefs = [b for b in PROXY_MISSION_BELIEFS if "5%" in b.claim or "risk" in b.claim.lower()]
    assert risk_beliefs, "No risk-cap belief found"
    for b in risk_beliefs:
        assert b.contradictions, f"Risk belief has no contradictions: {b.claim!r}"


def test_legal_belief_has_contradictions() -> None:
    legal = [b for b in PROXY_MISSION_BELIEFS if "legal" in b.claim.lower()]
    assert legal, "No legal belief found"
    for b in legal:
        assert b.contradictions, f"Legal belief has no contradictions: {b.claim!r}"


# ---------------------------------------------------------------------------
# Executable proofs are valid Python and pass when run
# ---------------------------------------------------------------------------

def test_proofs_are_valid_python() -> None:
    for b in PROXY_MISSION_BELIEFS:
        assert b.executable_proof, f"No proof on: {b.claim!r}"
        try:
            ast.parse(b.executable_proof)
        except SyntaxError as exc:
            pytest.fail(f"Proof is not valid Python for {b.claim!r}: {exc}")


def test_proofs_execute_without_error() -> None:
    """Each proof must pass when run in a clean subprocess."""
    for b in PROXY_MISSION_BELIEFS:
        result = subprocess.run(
            [sys.executable, "-c", b.executable_proof],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, (
            f"Proof failed for {b.claim!r}:\n"
            f"  stdout: {result.stdout}\n"
            f"  stderr: {result.stderr}"
        )


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

def test_round_trip_serialization() -> None:
    """to_dict() -> from_dict() must preserve all key fields."""
    for b in PROXY_MISSION_BELIEFS:
        d = b.to_dict()
        restored = BeliefCertificate.from_dict(d)
        assert restored.claim == b.claim
        assert restored.confidence == b.confidence
        assert restored.domain == b.domain
        assert restored.is_axiom == b.is_axiom
        assert restored.decay_rate == b.decay_rate
        assert restored.executable_proof == b.executable_proof

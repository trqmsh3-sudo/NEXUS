"""Tests for PROXY mission injection in build_nexus().

Verifies that build_nexus() seeds the knowledge graph with all 9 PROXY
mission beliefs and none of the old coding axioms. All tests fail until
the coding axioms in main.py are replaced.
"""
from __future__ import annotations

import inspect

import pytest

from nexus.core.knowledge_graph import KnowledgeGraph
from nexus.core.proxy_mission import PROXY_MISSION_BELIEFS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OLD_CODING_CLAIMS = {
    "Clean code is better than clever code",
    "Tests must run before code is trusted",
    "Every system needs a kill switch",
}

PROXY_CLAIMS = {b.claim for b in PROXY_MISSION_BELIEFS}


# ---------------------------------------------------------------------------
# Injection unit tests — KnowledgeGraph in isolation
# ---------------------------------------------------------------------------

def test_inject_adds_all_nine(tmp_path: Path) -> None:
    """inject_external_signal must accept all 9 PROXY beliefs."""
    graph = KnowledgeGraph(storage_path=str(tmp_path / "kg.json"))
    result = graph.inject_external_signal(PROXY_MISSION_BELIEFS)
    assert result["added"] == 9, (
        f"Expected 9 beliefs added, got {result['added']} "
        f"(rejected={result['rejected']})"
    )


def test_inject_rejects_none(tmp_path: Path) -> None:
    """All 9 PROXY beliefs must pass every gate in inject_external_signal."""
    graph = KnowledgeGraph(storage_path=str(tmp_path / "kg.json"))
    result = graph.inject_external_signal(PROXY_MISSION_BELIEFS)
    assert result["rejected"] == 0, (
        f"Expected 0 rejections, got {result['rejected']}"
    )


def test_all_nine_retrievable_from_graph(tmp_path: Path) -> None:
    """Every PROXY claim must be findable in the graph after injection."""
    graph = KnowledgeGraph(storage_path=str(tmp_path / "kg.json"))
    graph.inject_external_signal(PROXY_MISSION_BELIEFS)
    for belief in PROXY_MISSION_BELIEFS:
        assert belief.claim in graph, (
            f"Claim not found in graph after injection: {belief.claim!r}"
        )


def test_injected_beliefs_are_valid(tmp_path: Path) -> None:
    """All injected beliefs must pass is_valid() inside the graph."""
    graph = KnowledgeGraph(storage_path=str(tmp_path / "kg.json"))
    graph.inject_external_signal(PROXY_MISSION_BELIEFS)
    for b in graph.beliefs_snapshot():
        if b.claim in PROXY_CLAIMS:
            assert b.is_valid(), f"Injected belief not valid: {b.claim!r}"


def test_injected_beliefs_retain_axiom_flag(tmp_path: Path) -> None:
    """is_axiom must survive the inject_external_signal round-trip."""
    graph = KnowledgeGraph(storage_path=str(tmp_path / "kg.json"))
    graph.inject_external_signal(PROXY_MISSION_BELIEFS)
    for b in graph.beliefs_snapshot():
        if b.claim in PROXY_CLAIMS:
            assert b.is_axiom is True, (
                f"is_axiom flag lost after injection: {b.claim!r}"
            )


def test_injected_beliefs_retain_zero_decay(tmp_path: Path) -> None:
    graph = KnowledgeGraph(storage_path=str(tmp_path / "kg.json"))
    graph.inject_external_signal(PROXY_MISSION_BELIEFS)
    for b in graph.beliefs_snapshot():
        if b.claim in PROXY_CLAIMS:
            assert b.decay_rate == 0.0, (
                f"decay_rate changed after injection: {b.claim!r}"
            )


# ---------------------------------------------------------------------------
# build_nexus() integration tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def nexus_omega(monkeypatch):
    """build_nexus() with persistence stubbed out — no disk/Supabase I/O."""
    monkeypatch.setattr(
        "nexus.core.persistence.PersistenceManager.load",
        lambda self: [],
    )
    monkeypatch.setattr(
        "nexus.core.persistence.PersistenceManager.save",
        lambda self, graph: None,
    )
    from nexus.main import build_nexus
    return build_nexus()


def test_build_nexus_graph_contains_all_nine_proxy_claims(nexus_omega) -> None:
    """build_nexus() graph must contain all 9 PROXY mission claims."""
    graph_claims = {b.claim for b in nexus_omega.knowledge_graph.beliefs_snapshot()}
    missing = PROXY_CLAIMS - graph_claims
    assert not missing, (
        "These PROXY claims are missing from the graph:\n"
        + "\n".join(f"  - {c}" for c in sorted(missing))
    )


def test_build_nexus_graph_has_no_coding_axioms(nexus_omega) -> None:
    """build_nexus() must NOT inject the old coding axioms."""
    graph_claims = {b.claim for b in nexus_omega.knowledge_graph.beliefs_snapshot()}
    stale = OLD_CODING_CLAIMS & graph_claims
    assert not stale, (
        "Old coding axioms still present in graph:\n"
        + "\n".join(f"  - {c}" for c in sorted(stale))
    )


def test_build_nexus_all_proxy_beliefs_are_valid(nexus_omega) -> None:
    """Every PROXY belief in the live graph must pass is_valid()."""
    for b in nexus_omega.knowledge_graph.beliefs_snapshot():
        if b.claim in PROXY_CLAIMS:
            assert b.is_valid(), f"PROXY belief not valid in live graph: {b.claim!r}"


def test_build_nexus_all_proxy_beliefs_are_axioms(nexus_omega) -> None:
    """Every PROXY belief in the live graph must have is_axiom=True."""
    for b in nexus_omega.knowledge_graph.beliefs_snapshot():
        if b.claim in PROXY_CLAIMS:
            assert b.is_axiom is True, (
                f"PROXY belief lost is_axiom in live graph: {b.claim!r}"
            )


# ---------------------------------------------------------------------------
# Source-level guard — coding axioms must not exist in main.py source
# ---------------------------------------------------------------------------

def test_main_py_no_longer_contains_coding_axiom_claims() -> None:
    """The literal coding claim strings must not appear in main.py source."""
    import nexus.main as main_mod
    source = inspect.getsource(main_mod)
    for claim in OLD_CODING_CLAIMS:
        assert claim not in source, (
            f"Old coding axiom still literal in main.py source: {claim!r}"
        )


def test_main_py_imports_proxy_mission() -> None:
    """main.py must import from nexus.core.proxy_mission."""
    import nexus.main as main_mod
    source = inspect.getsource(main_mod)
    assert "proxy_mission" in source, (
        "main.py does not import from nexus.core.proxy_mission"
    )

"""Knowledge Graph — the main knowledge store for NEXUS House A.

Manages BeliefCertificate objects in a graph structure with an adjacency
list for dependency edges and a domain index for fast domain-scoped
queries. Every mutation is gated: no belief enters the graph without
passing ``is_valid()`` and ``is_expired() == False``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterator

from nexus.core.belief_certificate import BeliefCertificate
from nexus.core.persistence import PersistenceManager

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class KnowledgeGraph:
    """Directed graph of BeliefCertificates with gated insertion.

    Three parallel storage structures are maintained in lockstep:

    * ``beliefs`` — canonical store keyed by claim text.
    * ``graph`` — adjacency list mapping a claim to the set of claims
      it depends on (derived from ``downstream_dependents``).
    * ``domain_index`` — maps each domain string to the set of claims
      currently filed under that domain.

    Attributes:
        beliefs: Primary claim -> BeliefCertificate mapping.
        graph: Adjacency list of dependency edges (claim -> dependents).
        domain_index: Domain -> set of claims belonging to that domain.
    """

    beliefs: dict[str, BeliefCertificate] = field(default_factory=dict)
    graph: dict[str, set[str]] = field(default_factory=dict)
    domain_index: dict[str, set[str]] = field(default_factory=dict)
    storage_path: str | None = None
    persistence: PersistenceManager = field(init=False)

    def __post_init__(self) -> None:
        path = self.storage_path or "data/knowledge_store/beliefs.json"
        self.persistence = PersistenceManager(storage_path=path)
        loaded = self.persistence.load()
        for belief in loaded:
            self.beliefs[belief.claim] = belief
            self._index_belief(belief)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _index_belief(self, belief: BeliefCertificate) -> None:
        """Populate the graph and domain_index for a newly added belief."""
        self.graph.setdefault(belief.claim, set())
        for dep in belief.downstream_dependents:
            self.graph.setdefault(dep, set()).add(belief.claim)

        self.domain_index.setdefault(belief.domain, set()).add(belief.claim)

    def _deindex_belief(self, belief: BeliefCertificate) -> None:
        """Remove a belief from the graph and domain_index."""
        self.graph.pop(belief.claim, None)
        for dep_set in self.graph.values():
            dep_set.discard(belief.claim)

        domain_claims = self.domain_index.get(belief.domain)
        if domain_claims is not None:
            domain_claims.discard(belief.claim)
            if not domain_claims:
                del self.domain_index[belief.domain]

    # ------------------------------------------------------------------
    # 1. add_belief
    # ------------------------------------------------------------------

    def add_belief(self, belief: BeliefCertificate) -> bool:
        """Attempt to add a belief to the graph after full validation.

        The belief is rejected (and the rejection logged) if any of the
        following conditions hold:

        * ``belief.is_expired()`` is True.
        * ``belief.is_valid()`` is False.
        * ``belief.confidence < 0.5``.
        * The belief contradicts one or more existing beliefs in the
          graph (checked via :meth:`contradiction_check`).

        When all checks pass the belief is inserted into ``beliefs``,
        ``graph``, and ``domain_index``.

        Args:
            belief: The BeliefCertificate to add.

        Returns:
            True if the belief was accepted and stored, False otherwise.
        """
        if belief.is_expired():
            logger.warning(
                "REJECTED [expired] claim=%r  decay_rate=%s  last_verified=%s",
                belief.claim, belief.decay_rate, belief.last_verified.isoformat(),
            )
            return False

        if not belief.is_valid():
            logger.warning(
                "REJECTED [invalid] claim=%r  confidence=%s  has_proof=%s",
                belief.claim, belief.confidence, belief.executable_proof is not None,
            )
            return False

        if belief.confidence < 0.5:
            logger.warning(
                "REJECTED [low confidence] claim=%r  confidence=%s",
                belief.claim, belief.confidence,
            )
            return False

        existing = list(self.beliefs.values())
        contradictions = self.contradiction_check(belief.claim, existing)
        if contradictions:
            logger.warning(
                "REJECTED [contradiction] claim=%r  conflicts_with=%s",
                belief.claim, contradictions,
            )
            return False

        self.beliefs[belief.claim] = belief
        self._index_belief(belief)
        logger.info("ACCEPTED claim=%r  domain=%r  confidence=%s", belief.claim, belief.domain, belief.confidence)
        self.persistence.auto_save(self)
        return True

    # ------------------------------------------------------------------
    # 2. get_belief
    # ------------------------------------------------------------------

    def get_belief(self, claim: str) -> BeliefCertificate | None:
        """Retrieve a belief by its claim, auto-pruning if expired.

        If the belief exists but has expired since it was stored, it is
        removed from all indexes and ``None`` is returned.

        Args:
            claim: The exact claim string to look up.

        Returns:
            The matching BeliefCertificate, or None if absent or expired.
        """
        belief = self.beliefs.get(claim)
        if belief is None:
            return None
        if belief.is_expired():
            logger.info("AUTO-PRUNED [expired on access] claim=%r", claim)
            self._deindex_belief(belief)
            del self.beliefs[claim]
            return None
        return belief

    # ------------------------------------------------------------------
    # 3. query_domain
    # ------------------------------------------------------------------

    def query_domain(self, domain: str) -> list[BeliefCertificate]:
        """Return all valid, non-expired beliefs in a domain.

        Expired beliefs discovered during the query are automatically
        pruned from the graph.

        Args:
            domain: The domain name to query (case-sensitive).

        Returns:
            List of healthy BeliefCertificates belonging to *domain*.
        """
        claims = list(self.domain_index.get(domain, set()))
        results: list[BeliefCertificate] = []
        for claim in claims:
            belief = self.beliefs.get(claim)
            if belief is None:
                continue
            if belief.is_expired():
                logger.info("AUTO-PRUNED [domain query] claim=%r", claim)
                self._deindex_belief(belief)
                del self.beliefs[claim]
                continue
            if belief.is_valid():
                results.append(belief)
        return results

    # ------------------------------------------------------------------
    # 4. get_dependents
    # ------------------------------------------------------------------

    def get_dependents(self, claim: str) -> list[BeliefCertificate]:
        """Return all beliefs that depend on the given claim.

        Dependency edges are stored in ``graph[claim]`` — each member
        of that set is a claim whose ``downstream_dependents`` list
        includes *claim*.

        Args:
            claim: The claim whose dependents to retrieve.

        Returns:
            List of BeliefCertificates that depend on *claim*.
        """
        dependent_claims = self.graph.get(claim, set())
        results: list[BeliefCertificate] = []
        for dep_claim in dependent_claims:
            belief = self.beliefs.get(dep_claim)
            if belief is not None:
                results.append(belief)
        return results

    # ------------------------------------------------------------------
    # 5. prune_expired
    # ------------------------------------------------------------------

    def prune_expired(self) -> int:
        """Remove all expired beliefs from the graph.

        Returns:
            The number of beliefs that were pruned.
        """
        expired_claims = [
            claim for claim, belief in self.beliefs.items()
            if belief.is_expired()
        ]
        for claim in expired_claims:
            belief = self.beliefs.pop(claim)
            self._deindex_belief(belief)
            logger.info("PRUNED [expired] claim=%r", claim)
        return len(expired_claims)

    # ------------------------------------------------------------------
    # 6. contradiction_check
    # ------------------------------------------------------------------

    def contradiction_check(
        self,
        new_claim: str,
        existing_beliefs: list[BeliefCertificate],
    ) -> list[str]:
        """Check whether *new_claim* contradicts any existing beliefs.

        A contradiction is detected when an existing belief explicitly
        lists *new_claim* in its ``contradictions`` field, or when
        *new_claim* matches the ``claim`` of an existing belief that is
        already listed as a contradiction by another stored belief.

        Args:
            new_claim: The claim text being proposed for insertion.
            existing_beliefs: The beliefs to check against.

        Returns:
            List of claim strings from *existing_beliefs* that conflict
            with *new_claim*. Empty if no contradictions found.
        """
        conflicts: list[str] = []
        for belief in existing_beliefs:
            if new_claim in belief.contradictions:
                conflicts.append(belief.claim)
            if belief.claim in self._contradictions_for_new(new_claim):
                conflicts.append(belief.claim)
        return list(dict.fromkeys(conflicts))

    def _contradictions_for_new(self, new_claim: str) -> set[str]:
        """Gather contradiction sets that mention *new_claim* as a source.

        Scans stored beliefs to find any whose ``claim`` is listed as a
        contradiction *by* the belief being added. Since the new belief
        isn't stored yet, we check if any existing belief's claim appears
        in the contradiction lists that mention *new_claim*.
        """
        result: set[str] = set()
        for belief in self.beliefs.values():
            if new_claim in belief.contradictions:
                result.add(belief.claim)
        return result

    # ------------------------------------------------------------------
    # 7. inject_external_signal  (IRON LAW)
    # ------------------------------------------------------------------

    def inject_external_signal(
        self, beliefs: list[BeliefCertificate],
    ) -> dict[str, Any]:
        """Inject a batch of external beliefs — the IRON LAW method.

        House A must receive fresh external data every cycle. This method
        accepts a batch of new beliefs from an outside source and attempts
        to add each one through the standard ``add_belief`` gate.

        Args:
            beliefs: Batch of BeliefCertificates from an external source.

        Returns:
            A summary dict with keys:

            * ``added`` — number of beliefs accepted.
            * ``rejected`` — number of beliefs rejected.
            * ``contradictions`` — list of claim strings that were
              rejected specifically due to contradictions.
        """
        added: int = 0
        rejected: int = 0
        contradictions: list[str] = []

        for belief in beliefs:
            existing = list(self.beliefs.values())
            conflicts = self.contradiction_check(belief.claim, existing)

            if not self.add_belief(belief):
                rejected += 1
                if conflicts:
                    contradictions.append(belief.claim)
            else:
                added += 1

        logger.info(
            "INJECT complete  added=%d  rejected=%d  contradictions=%d",
            added, rejected, len(contradictions),
        )
        return {
            "added": added,
            "rejected": rejected,
            "contradictions": contradictions,
        }

    # ------------------------------------------------------------------
    # 8. health_report
    # ------------------------------------------------------------------

    def health_report(self) -> dict[str, Any]:
        """Generate a diagnostic health report for the knowledge store.

        Returns:
            A dictionary containing:

            * ``total_beliefs`` — current number of stored beliefs.
            * ``expired_count`` — how many are currently expired.
            * ``domain_breakdown`` — ``{domain: count}`` mapping.
            * ``average_confidence`` — mean confidence across all beliefs.
            * ``without_executable_proof`` — count of beliefs lacking a
              machine-verifiable proof.
        """
        total = len(self.beliefs)
        expired = sum(1 for b in self.beliefs.values() if b.is_expired())

        domain_breakdown: dict[str, int] = {}
        for domain, claims in self.domain_index.items():
            live = claims & self.beliefs.keys()
            if live:
                domain_breakdown[domain] = len(live)

        avg_confidence: float = 0.0
        if total > 0:
            avg_confidence = round(
                sum(b.confidence for b in self.beliefs.values()) / total, 4,
            )

        without_proof = sum(
            1 for b in self.beliefs.values() if b.executable_proof is None
        )

        return {
            "total_beliefs": total,
            "expired_count": expired,
            "domain_breakdown": domain_breakdown,
            "average_confidence": avg_confidence,
            "without_executable_proof": without_proof,
        }

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of beliefs in the graph."""
        return len(self.beliefs)

    def __iter__(self) -> Iterator[BeliefCertificate]:
        """Iterate over all stored beliefs."""
        return iter(self.beliefs.values())

    def __contains__(self, claim: str) -> bool:
        """Check if a claim exists in the graph."""
        return claim in self.beliefs

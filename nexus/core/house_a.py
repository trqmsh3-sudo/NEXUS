"""House-A reasoning engine for NEXUS.

House-A implements a forward-chaining reasoning loop over the
KnowledgeGraph. It evaluates belief validity, detects contradictions,
propagates confidence decay, and produces an audit trail of decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from nexus.core.belief_certificate import BeliefCertificate
from nexus.core.knowledge_graph import KnowledgeGraph


@dataclass
class AuditEntry:
    """A single entry in the House-A reasoning audit trail.

    Attributes:
        timestamp: When this entry was created (UTC).
        action: The type of action taken (e.g. "pruned", "flagged", "resolved").
        claim: The claim the action pertains to.
        detail: Human-readable explanation.
    """

    timestamp: datetime
    action: str
    claim: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "claim": self.claim,
            "detail": self.detail,
        }


@dataclass
class HouseA:
    """Forward-chaining reasoning engine over a KnowledgeGraph.

    House-A performs three core operations in each evaluation cycle:

    1. **Prune** — remove expired certificates.
    2. **Detect** — find unresolved contradictions.
    3. **Propagate** — lower confidence on dependents of invalid beliefs.

    Attributes:
        graph: The KnowledgeGraph to reason over.
        audit_log: Accumulated audit entries from all evaluation cycles.
        propagation_factor: Multiplier applied when lowering dependent
            confidence (0.0-1.0). Lower means harsher penalty.
    """

    graph: KnowledgeGraph
    audit_log: list[AuditEntry] = field(default_factory=list)
    propagation_factor: float = 0.8

    def _log(self, action: str, claim: str, detail: str) -> None:
        """Append an entry to the audit trail."""
        self.audit_log.append(
            AuditEntry(
                timestamp=datetime.now(timezone.utc),
                action=action,
                claim=claim,
                detail=detail,
            )
        )

    def prune_expired(self) -> list[BeliefCertificate]:
        """Remove expired beliefs and log each removal.

        Returns:
            List of pruned BeliefCertificates.
        """
        expired = [b for b in self.graph.beliefs_snapshot() if b.is_expired()]
        self.graph.prune_expired()
        for cert in expired:
            self._log(
                "pruned",
                cert.claim,
                f"Expired (decay_rate={cert.decay_rate}, "
                f"last_verified={cert.last_verified.isoformat()})",
            )
        return expired

    def detect_contradictions(self) -> list[tuple[BeliefCertificate, list[BeliefCertificate]]]:
        """Find beliefs with live contradictions in the graph.

        A contradiction is "live" when both the belief and the
        contradicting belief are present in the graph and valid.

        Returns:
            List of (belief, [contradicting_beliefs]) tuples
            for every belief that has at least one live contradiction.
        """
        results: list[tuple[BeliefCertificate, list[BeliefCertificate]]] = []
        for cert in self.graph:
            live = []
            for c in cert.contradictions:
                other = self.graph.get_belief(c)
                if other is not None and other.is_valid():
                    live.append(other)
            if live:
                self._log(
                    "flagged",
                    cert.claim,
                    f"Has {len(live)} live contradiction(s): "
                    f"{[c.claim for c in live]}",
                )
                results.append((cert, live))
        return results

    def propagate_decay(self) -> list[BeliefCertificate]:
        """Lower confidence on dependents of invalid or expired beliefs.

        For each belief that is not valid or is expired, every belief
        that depends on it has its confidence multiplied by
        ``propagation_factor``.

        Returns:
            List of beliefs whose confidence was reduced.
        """
        affected: list[BeliefCertificate] = []
        for cert in list(self.graph):
            if cert.is_valid() and not cert.is_expired():
                continue
            dependents = self.graph.get_dependents(cert.claim)
            for dep in dependents:
                old_conf = dep.confidence
                dep.confidence = round(dep.confidence * self.propagation_factor, 6)
                self._log(
                    "decayed",
                    dep.claim,
                    f"Confidence {old_conf} -> {dep.confidence} "
                    f"(caused by invalid/expired '{cert.claim}')",
                )
                affected.append(dep)
        return affected

    def evaluate(self) -> dict[str, Any]:
        """Run a full evaluation cycle: prune, detect, propagate.

        Returns:
            Summary dictionary with counts and details from each phase.
        """
        pruned = self.prune_expired()
        contradictions = self.detect_contradictions()
        decayed = self.propagate_decay()

        valid = [
            b for b in self.graph.beliefs_snapshot()
            if b.is_valid() and not b.is_expired()
        ]

        return {
            "pruned_count": len(pruned),
            "contradiction_count": len(contradictions),
            "decayed_count": len(decayed),
            "remaining_beliefs": len(self.graph),
            "valid_beliefs": len(valid),
        }

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Return the full audit log as serialisable dictionaries.

        Returns:
            List of audit entry dicts.
        """
        return [entry.to_dict() for entry in self.audit_log]

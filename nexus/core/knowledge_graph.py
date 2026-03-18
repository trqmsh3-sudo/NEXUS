"""Knowledge Graph — the main knowledge store for NEXUS House A.

Manages BeliefCertificate objects in a graph structure with an adjacency
list for dependency edges and a domain index for fast domain-scoped
queries. Every mutation is gated: no belief enters the graph without
passing ``is_valid()`` and ``is_expired() == False``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterator

from nexus.core.belief_certificate import BeliefCertificate
from nexus.core.domain_normalizer import normalize_domain
from nexus.core.persistence import PersistenceManager
from nexus.core.proof_runner import run_executable_proof_in_subprocess

logger: logging.Logger = logging.getLogger(__name__)


def _is_pytest_session() -> bool:
    return "pytest" in sys.modules or bool(os.getenv("PYTEST_CURRENT_TEST"))


def _reject_unit_test_outside_pytest(belief: BeliefCertificate) -> bool:
    """FIX 5: block unit-test beliefs in non-test runs."""
    return (belief.source or "").strip() == "unit-test" and not _is_pytest_session()


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
    governor_alert: Callable[[dict[str, Any]], None] | None = field(
        default=None, init=False, repr=False,
    )
    _belief_lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False,
    )

    @contextmanager
    def belief_lock(self) -> Iterator[None]:
        """Serialize all reads/writes of beliefs and indexes."""
        with self._belief_lock:
            yield

    def __post_init__(self) -> None:
        path = self.storage_path or "data/knowledge_store/beliefs.json"
        self.persistence = PersistenceManager(storage_path=path)
        loaded = self.persistence.load()
        with self._belief_lock:
            for belief in loaded:
                belief.domain = normalize_domain(belief.domain)
                self.beliefs[belief.claim] = belief
                self._index_belief(belief)

    def register_belief_bypass_gates(self, belief: BeliefCertificate) -> None:
        """Insert a belief without contradiction gate (founding axioms). Proof still runs in subprocess."""
        if _reject_unit_test_outside_pytest(belief):
            return
        ok, err = run_executable_proof_in_subprocess((belief.executable_proof or "").strip())
        if not ok:
            logger.warning(
                "SEED REJECTED [executable_proof] claim=%r err=%s",
                belief.claim, (err or "")[:300],
            )
            return
        belief.domain = normalize_domain(belief.domain)
        now = datetime.now(timezone.utc)
        belief.last_verified_at = now
        belief.last_verified = now
        belief.verification_status = "VERIFIED"
        belief.quarantined = False
        with self._belief_lock:
            self.beliefs[belief.claim] = belief
            self._index_belief(belief)

    def beliefs_snapshot(self) -> list[BeliefCertificate]:
        with self._belief_lock:
            return list(self.beliefs.values())

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
        * ``executable_proof`` fails when re-executed in a fresh subprocess.

        When all checks pass the belief is inserted into ``beliefs``,
        ``graph``, and ``domain_index``.

        Args:
            belief: The BeliefCertificate to add.

        Returns:
            True if the belief was accepted and stored, False otherwise.
        """
        if _reject_unit_test_outside_pytest(belief):
            return False
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

        belief.domain = normalize_domain(belief.domain)

        ok, err = run_executable_proof_in_subprocess((belief.executable_proof or "").strip())
        if not ok:
            logger.warning(
                "REJECTED [executable_proof] claim=%r err=%s",
                belief.claim, (err or "")[:300],
            )
            return False
        now = datetime.now(timezone.utc)
        belief.last_verified_at = now
        belief.last_verified = now
        belief.verification_status = "VERIFIED"
        belief.quarantined = False

        with self._belief_lock:
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
        logger.info(
            "ACCEPTED claim=%r  domain=%r  confidence=%s",
            belief.claim, belief.domain, belief.confidence,
        )
        try:
            _semantic_contradiction_after_add(self, belief)
        except Exception as exc:
            logger.warning("Semantic contradiction check failed (non-fatal): %s", exc)
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
        with self._belief_lock:
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
        with self._belief_lock:
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
                if belief.is_valid() and not belief.quarantined:
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
        with self._belief_lock:
            dependent_claims = set(self.graph.get(claim, set()))
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
        with self._belief_lock:
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
        with self._belief_lock:
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
        proof_rejected: int = 0

        admitted: list[BeliefCertificate] = []
        for belief in beliefs:
            if _reject_unit_test_outside_pytest(belief):
                rejected += 1
                continue
            if belief.is_expired():
                rejected += 1
                continue
            if not belief.is_valid():
                rejected += 1
                continue
            if belief.confidence < 0.5:
                rejected += 1
                continue
            ok, err = run_executable_proof_in_subprocess((belief.executable_proof or "").strip())
            if not ok:
                logger.warning(
                    "INJECT REJECTED [executable_proof] claim=%r err=%s",
                    belief.claim, (err or "")[:300],
                )
                proof_rejected += 1
                rejected += 1
                continue
            belief.domain = normalize_domain(belief.domain)
            now = datetime.now(timezone.utc)
            belief.last_verified_at = now
            belief.last_verified = now
            belief.verification_status = "VERIFIED"
            belief.quarantined = False
            admitted.append(belief)

        to_semantic: list[BeliefCertificate] = []
        with self._belief_lock:
            for belief in admitted:
                existing = list(self.beliefs.values())
                confl = self.contradiction_check(belief.claim, existing)
                if confl:
                    rejected += 1
                    contradictions.append(belief.claim)
                    continue
                self.beliefs[belief.claim] = belief
                self._index_belief(belief)
                added += 1
                to_semantic.append(belief)

        for belief in to_semantic:
            try:
                _semantic_contradiction_after_add(self, belief)
            except Exception as exc:
                logger.warning("Semantic contradiction check failed (non-fatal): %s", exc)
        if to_semantic:
            self.persistence.auto_save(self)

        logger.info(
            "INJECT complete  added=%d  rejected=%d  contradictions=%d",
            added, rejected, len(contradictions),
        )
        return {
            "added": added,
            "rejected": rejected,
            "contradictions": contradictions,
            "proof_rejected": proof_rejected,
        }

    # ------------------------------------------------------------------
    # 7b. Periodic proof re-verification (24h)
    # ------------------------------------------------------------------

    def reverify_beliefs_past_due(self) -> dict[str, Any]:
        """Re-run ``executable_proof`` in a fresh subprocess every 24h per belief.

        On failure: quarantine, set ``verification_status`` to UNVERIFIED,
        and notify ``governor_alert``.
        """
        now = datetime.now(timezone.utc)
        due = [b for b in self.beliefs_snapshot() if b.proof_reverification_due(24.0)]
        reverified = 0
        quarantined_n = 0
        for b in due:
            claim = b.claim
            ok, err = run_executable_proof_in_subprocess((b.executable_proof or "").strip())
            with self._belief_lock:
                live = self.beliefs.get(claim)
                if live is None:
                    continue
                if ok:
                    live.last_verified_at = now
                    live.last_verified = now
                    live.verification_status = "VERIFIED"
                    live.quarantined = False
                    reverified += 1
                else:
                    live.quarantined = True
                    live.verification_status = "UNVERIFIED"
                    quarantined_n += 1
                    logger.warning(
                        "QUARANTINE [re-verification failed] claim=%r err=%s",
                        claim,
                        (err or "")[:300],
                    )
                    if self.governor_alert:
                        try:
                            self.governor_alert({
                                "type": "UNVERIFIED",
                                "claim": claim,
                                "reason": "executable_proof re-verification failed",
                                "detail": (err or "")[:500],
                            })
                        except Exception:
                            pass
        if reverified or quarantined_n:
            self.persistence.auto_save(self)
        return {
            "reverified": reverified,
            "quarantined": quarantined_n,
            "checked": len(due),
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
        with self._belief_lock:
            total = len(self.beliefs)
            beliefs_list = list(self.beliefs.values())
            expired = sum(1 for b in beliefs_list if b.is_expired())

            domain_breakdown: dict[str, int] = {}
            for domain, claims in self.domain_index.items():
                live = claims & self.beliefs.keys()
                if live:
                    domain_breakdown[domain] = len(live)

            avg_confidence: float = 0.0
            if total > 0:
                avg_confidence = round(
                    sum(b.confidence for b in beliefs_list) / total, 4,
                )

            without_proof = sum(
                1 for b in beliefs_list if b.executable_proof is None
            )
            quarantined_count = sum(1 for b in beliefs_list if b.quarantined)
            unverified_count = sum(
                1 for b in beliefs_list if b.verification_status == "UNVERIFIED"
            )

            return {
                "total_beliefs": total,
                "expired_count": expired,
                "domain_breakdown": domain_breakdown,
                "average_confidence": avg_confidence,
                "without_executable_proof": without_proof,
                "quarantined_count": quarantined_count,
                "unverified_count": unverified_count,
            }

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of beliefs in the graph."""
        with self._belief_lock:
            return len(self.beliefs)

    def __iter__(self) -> Iterator[BeliefCertificate]:
        """Iterate over all stored beliefs."""
        with self._belief_lock:
            return iter(list(self.beliefs.values()))

    def __contains__(self, claim: str) -> bool:
        """Check if a claim exists in the graph."""
        with self._belief_lock:
            return claim in self.beliefs


# ------------------------------------------------------------------
# Semantic contradiction (Groq free model)
# ------------------------------------------------------------------


def _groq_completion(messages: list[dict[str, str]], max_tokens: int = 800) -> str:
    """Single Groq call; returns empty string on missing key or error."""
    import os
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return ""
    try:
        import litellm
        litellm.suppress_debug_info = True
        r = litellm.completion(
            model="groq/llama-3.1-8b-instant",
            messages=messages,
            api_key=key,
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.debug("groq completion failed: %s", exc)
        return ""


def _parse_json_array(text: str) -> list[dict[str, Any]]:
    """Extract a JSON array from model output; return list of dicts."""
    text = text.strip()
    for wrapper in ("```json", "```"):
        if wrapper in text:
            idx = text.find(wrapper) + len(wrapper)
            text = text[idx:].lstrip()
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []
    try:
        arr = json.loads(match.group(0))
        if not isinstance(arr, list):
            return []
        return [x for x in arr if isinstance(x, dict)]
    except json.JSONDecodeError:
        return []


def _extract_triples(claim: str) -> list[dict[str, Any]]:
    """Extract up to 3 (subject, predicate, object) triples from a claim via Groq."""
    out = _groq_completion([
        {"role": "system", "content": "Return ONLY a JSON array of up to 3 objects. Each object has keys: subject, predicate, object (strings). No other text."},
        {"role": "user", "content": f"Claim:\n{claim[:2000]}\n\nExtract up to 3 factual triples."},
    ], max_tokens=600)
    arr = _parse_json_array(out)
    triples = []
    for t in arr[:3]:
        s = t.get("subject") or t.get("s")
        p = t.get("predicate") or t.get("p")
        o = t.get("object") or t.get("o")
        if isinstance(s, str) and isinstance(p, str) and isinstance(o, str):
            triples.append({"subject": s[:200], "predicate": p[:200], "object": o[:200]})
    return triples[:3]


def _groq_detect_contradiction(
    graph: KnowledgeGraph,
    new_claim: str,
    new_triples: list[dict[str, Any]],
    others: list[tuple[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    """Ask Groq if new belief contradicts any existing; return {contradiction, claim_a, claim_b, reason}."""
    if not others:
        return {"contradiction": False}
    payload = json.dumps({
        "new_claim": new_claim[:800],
        "new_triples": new_triples,
        "existing": [{"claim": c[:400], "triples": t} for c, t in others[:30]],
    }, ensure_ascii=False)
    out = _groq_completion([
        {"role": "system", "content": "Return ONLY valid JSON with keys: contradiction (boolean), claim_a (string), claim_b (string), reason (string). If no contradiction, contradiction=false and others empty."},
        {"role": "user", "content": f"Does the new belief contradict any existing? Payload:\n{payload}"},
    ], max_tokens=400)
    try:
        # Find first { ... }
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", out)
        if match:
            obj = json.loads(match.group(0))
            return {
                "contradiction": bool(obj.get("contradiction")),
                "claim_a": (obj.get("claim_a") or "").strip()[:500],
                "claim_b": (obj.get("claim_b") or "").strip()[:500],
                "reason": (obj.get("reason") or "").strip()[:300],
            }
    except (json.JSONDecodeError, TypeError):
        pass
    return {"contradiction": False}


def _semantic_contradiction_after_add(graph: KnowledgeGraph, belief: BeliefCertificate) -> None:
    """After a belief is added: extract triples, compare to existing, flag CONFLICT and alert governor."""
    triples = _extract_triples(belief.claim)
    belief.semantic_triples = triples

    with graph._belief_lock:
        others = [
            (b.claim, list(getattr(b, "semantic_triples", []) or []))
            for b in graph.beliefs.values()
            if b.claim != belief.claim
        ]
    if not others:
        return

    result = _groq_detect_contradiction(graph, belief.claim, triples, others)
    if not result.get("contradiction"):
        return

    claim_a = result.get("claim_a") or ""
    claim_b = result.get("claim_b") or ""
    with graph._belief_lock:
        ba = graph.beliefs.get(claim_a)
        bb = graph.beliefs.get(claim_b)
        if ba is not None and bb is not None:
            ba.conflict_flag = "CONFLICT"
            bb.conflict_flag = "CONFLICT"
        elif belief.claim in (claim_a, claim_b):
            belief.conflict_flag = "CONFLICT"
            other_claim = claim_b if claim_a == belief.claim else claim_a
            bo = graph.beliefs.get(other_claim)
            if bo is not None:
                bo.conflict_flag = "CONFLICT"

    alert = {
        "type": "CONFLICT",
        "claims": [claim_a, claim_b],
        "reason": result.get("reason", ""),
    }
    logger.warning("GOVERNOR CONFLICT ALERT: semantic contradiction %s", alert)
    if graph.governor_alert:
        try:
            graph.governor_alert(alert)
        except Exception:
            pass

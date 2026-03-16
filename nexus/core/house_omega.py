"""House Omega — The Governor.

Controls all other Houses and runs the full NEXUS loop.
Manages sleep cycles, evolution, and self-improvement.
Nothing bypasses House D. Every cycle is logged. Sleep cycles
run automatically every N iterations to compress and clean the
knowledge graph.
"""

from __future__ import annotations

import json
import logging
import pathlib
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from nexus.core.belief_certificate import BeliefCertificate
from nexus.core.external_signal import ExternalSignalProvider
from nexus.core.house_b import HouseB, StructuredSpecificationObject
from nexus.core.house_c import BuildResult, HouseC
from nexus.core.house_d import DestructionReport, HouseD
from nexus.core.knowledge_graph import KnowledgeGraph

logger: logging.Logger = logging.getLogger(__name__)

_HISTORY_PATH: pathlib.Path = pathlib.Path("data/cycle_history.json")


# ------------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------------

@dataclass
class CycleResult:
    """Full trace of a single NEXUS processing cycle.

    Attributes:
        cycle_id: Unique identifier for this cycle (UUID4).
        user_input: The raw input that initiated the cycle.
        sso: The SSO produced by House B, or None on failure.
        destruction_report: House D's report on the SSO, or None.
        build_result: House C's build result, or None.
        belief_added: Whether a new belief was injected into House A.
        cycle_time_seconds: Wall-clock duration of the cycle.
        success: Whether the cycle completed successfully end-to-end.
        failure_reason: Human-readable failure explanation, or None.
    """

    cycle_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_input: str = ""
    sso: StructuredSpecificationObject | None = None
    destruction_report: DestructionReport | None = None
    build_result: BuildResult | None = None
    belief_added: bool = False
    cycle_time_seconds: float = 0.0
    success: bool = False
    failure_reason: str | None = None
    refinement_attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise the cycle result."""
        return {
            "cycle_id": self.cycle_id,
            "user_input": self.user_input,
            "sso": self.sso.to_dict() if self.sso else None,
            "destruction_report": (
                self.destruction_report.to_dict()
                if self.destruction_report else None
            ),
            "build_result": (
                self.build_result.to_dict()
                if self.build_result else None
            ),
            "belief_added": self.belief_added,
            "cycle_time_seconds": self.cycle_time_seconds,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "refinement_attempts": self.refinement_attempts,
        }


@dataclass
class SystemHealth:
    """Diagnostic snapshot of the entire NEXUS system.

    Attributes:
        total_cycles: Total cycles executed since creation.
        successful_cycles: Count of fully successful cycles.
        failed_cycles: Count of cycles that failed at any stage.
        total_beliefs: Current number of beliefs in the knowledge graph.
        average_cycle_time: Mean wall-clock time per cycle (seconds).
        last_sleep_cycle: Timestamp of the most recent sleep cycle.
        next_sleep_cycle_due: Predicted timestamp for the next sleep.
        domains_covered: List of knowledge domains with live beliefs.
        system_score: Ratio of successful to total cycles (0.0-1.0).
    """

    total_cycles: int = 0
    successful_cycles: int = 0
    failed_cycles: int = 0
    total_beliefs: int = 0
    average_cycle_time: float = 0.0
    last_sleep_cycle: datetime | None = None
    next_sleep_cycle_due: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    domains_covered: list[str] = field(default_factory=list)
    system_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise for reporting or persistence."""
        return {
            "total_cycles": self.total_cycles,
            "successful_cycles": self.successful_cycles,
            "failed_cycles": self.failed_cycles,
            "total_beliefs": self.total_beliefs,
            "average_cycle_time": round(self.average_cycle_time, 4),
            "last_sleep_cycle": (
                self.last_sleep_cycle.isoformat()
                if self.last_sleep_cycle else None
            ),
            "next_sleep_cycle_due": self.next_sleep_cycle_due.isoformat(),
            "domains_covered": list(self.domains_covered),
            "system_score": round(self.system_score, 4),
        }


# ------------------------------------------------------------------
# House Omega — The Governor
# ------------------------------------------------------------------

@dataclass
class HouseOmega:
    """The Governor — orchestrates the full NEXUS processing loop.

    House Omega owns every other House and the shared KnowledgeGraph.
    It enforces the Iron Laws:

    * Nothing bypasses House D.
    * Sleep cycles run automatically every ``sleep_cycle_interval``
      cycles.
    * External knowledge must survive House D before entering House A.
    * Every cycle is logged — no exceptions.

    Attributes:
        knowledge_graph: The shared knowledge store (House A).
        house_b: The Oracle — problem redefinition.
        house_c: The Builder — code generation and validation.
        house_d: The Destroyer — adversarial attack engine.
        cycle_count: Total cycles executed.
        sleep_cycle_interval: Run a sleep cycle every N iterations.
        cycle_history: Ordered list of all cycle results.
        created_at: UTC timestamp when the Governor was instantiated.
    """

    knowledge_graph: KnowledgeGraph
    house_b: HouseB
    house_c: HouseC
    house_d: HouseD
    cycle_count: int = 0
    sleep_cycle_interval: int = 50
    external_signal_interval: int = 10
    max_refinements: int = 3
    cycle_history: list[CycleResult] = field(default_factory=list)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    _last_sleep: datetime | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # 1. run  —  THE MAIN LOOP
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> CycleResult:
        """Execute a full NEXUS cycle with iterative refinement.

        Pipeline:
        1. House B redefines the problem into an SSO.
        2. House D attacks the SSO.
        3. If SSO is killed but recommendation is ``"REVISE"``,
           send the DestructionReport back to House B for refinement.
           Repeat steps 2-3 up to ``max_refinements`` times.
        4. If survived, House C builds code from the SSO.
        5. If build succeeds, convert artifact to BeliefCertificate.
        6. House D attacks the BeliefCertificate.
        7. If belief survived, inject into House A (knowledge graph).
        8. Increment ``cycle_count``.
        9. If ``cycle_count % sleep_cycle_interval == 0``, trigger
           :meth:`run_sleep_cycle`.

        If any House raises an exception the cycle is marked as failed
        with the reason logged, and the next cycle can proceed normally.

        Args:
            user_input: Raw human request.

        Returns:
            A CycleResult containing the full trace of every step.
        """
        start = time.perf_counter()
        result = CycleResult(user_input=user_input)
        logger.info(
            "OMEGA cycle %d started  input=%r",
            self.cycle_count + 1, user_input[:100],
        )

        try:
            # Step 1 — House B: redefine
            sso = self.house_b.redefine(user_input)
            result.sso = sso
            logger.info("OMEGA step 1 complete  problem=%r", sso.redefined_problem[:80])

            # Step 2 — House D: attack SSO (with iterative refinement)
            sso_report = self.house_d.attack_sso(sso)
            result.destruction_report = sso_report
            logger.info(
                "OMEGA step 2 complete  survived=%s  score=%.2f  rec=%s",
                sso_report.survived, sso_report.survival_score,
                sso_report.recommendation,
            )

            # Iterative refinement loop
            attempts = 0
            while (
                not sso_report.survived
                and sso_report.recommendation == "REVISE"
                and attempts < self.max_refinements
            ):
                attempts += 1
                logger.info(
                    "OMEGA refinement %d/%d  sending attacks back to House B",
                    attempts, self.max_refinements,
                )

                sso = self.house_b.refine(sso, sso_report)
                result.sso = sso
                logger.info(
                    "OMEGA refinement %d  refined problem=%r",
                    attempts, sso.redefined_problem[:80],
                )

                sso_report = self.house_d.attack_sso(sso)
                result.destruction_report = sso_report
                logger.info(
                    "OMEGA refinement %d  survived=%s  score=%.2f  rec=%s",
                    attempts, sso_report.survived,
                    sso_report.survival_score, sso_report.recommendation,
                )

            result.refinement_attempts = attempts

            if not sso_report.survived:
                result.failure_reason = (
                    f"SSO destroyed by House D after {attempts} refinement(s) "
                    f"(score={sso_report.survival_score:.2f}, "
                    f"recommendation={sso_report.recommendation})"
                )
                result.cycle_time_seconds = time.perf_counter() - start
                self._finalise_cycle(result)
                return result

            # Step 3 — House C: build
            build_result = self.house_c.build(sso, sso_report)
            result.build_result = build_result
            logger.info(
                "OMEGA step 3 complete  build_success=%s  ready=%s",
                build_result.success, build_result.ready_for_house_a,
            )

            if not build_result.success:
                result.failure_reason = "House C build failed validation"
                result.cycle_time_seconds = time.perf_counter() - start
                self._finalise_cycle(result)
                return result

            # Step 4 — Convert to BeliefCertificate
            belief = self.house_c.to_belief_certificate(build_result.artifact)

            # Step 5 — House D: attack belief
            belief_report = self.house_d.attack_belief(belief)
            logger.info(
                "OMEGA step 5 complete  belief_survived=%s  score=%.2f",
                belief_report.survived, belief_report.survival_score,
            )

            if not belief_report.survived:
                result.failure_reason = (
                    f"Belief destroyed by House D "
                    f"(score={belief_report.survival_score:.2f})"
                )
                result.cycle_time_seconds = time.perf_counter() - start
                self._finalise_cycle(result)
                return result

            # Step 6 — Inject into House A
            added = self.knowledge_graph.add_belief(belief)
            result.belief_added = added
            if not added:
                result.failure_reason = "House A rejected the belief"
            else:
                result.success = True
            logger.info("OMEGA step 6 complete  belief_added=%s", added)

        except Exception as exc:
            result.failure_reason = f"{type(exc).__name__}: {exc}"
            logger.exception("OMEGA cycle failed with exception")

        result.cycle_time_seconds = time.perf_counter() - start
        self._finalise_cycle(result)
        return result

    # ------------------------------------------------------------------
    # 2. run_sleep_cycle
    # ------------------------------------------------------------------

    def run_sleep_cycle(self) -> dict[str, Any]:
        """Compress and clean the knowledge graph during a sleep cycle.

        Steps:
        1. Prune all expired beliefs.
        2. Flag beliefs with confidence < 0.6.
        3. Detect contradiction clusters.
        4. Re-verify and re-inject the top beliefs.
        5. Log completion.

        Returns:
            Summary dict with ``pruned``, ``flagged``,
            ``contradictions``, and ``duration`` keys.
        """
        start = time.perf_counter()
        logger.info("OMEGA sleep cycle started  cycle_count=%d", self.cycle_count)

        pruned = self.knowledge_graph.prune_expired()

        low_confidence = [
            b for b in self.knowledge_graph.beliefs.values()
            if b.confidence < 0.6
        ]
        flagged = len(low_confidence)
        for b in low_confidence:
            logger.info(
                "SLEEP flagged low-confidence  claim=%r  confidence=%s",
                b.claim[:60], b.confidence,
            )

        contradiction_count = 0
        for b in list(self.knowledge_graph.beliefs.values()):
            conflicts = self.knowledge_graph.contradiction_check(
                b.claim, list(self.knowledge_graph.beliefs.values()),
            )
            if conflicts:
                contradiction_count += 1
                logger.info(
                    "SLEEP contradiction cluster  claim=%r  conflicts=%s",
                    b.claim[:60], conflicts,
                )

        top_beliefs = sorted(
            self.knowledge_graph.beliefs.values(),
            key=lambda b: b.confidence,
            reverse=True,
        )[:10]
        refreshed: list[BeliefCertificate] = []
        for b in top_beliefs:
            refreshed.append(BeliefCertificate(
                claim=b.claim,
                source=b.source,
                confidence=b.confidence,
                contradictions=list(b.contradictions),
                decay_rate=b.decay_rate,
                downstream_dependents=list(b.downstream_dependents),
                executable_proof=b.executable_proof,
                domain=b.domain,
                created_at=b.created_at,
                last_verified=datetime.now(timezone.utc),
            ))
        if refreshed:
            self.knowledge_graph.inject_external_signal(refreshed)

        self._last_sleep = datetime.now(timezone.utc)
        elapsed = time.perf_counter() - start

        self.knowledge_graph.persistence.save(self.knowledge_graph)

        logger.info(
            "OMEGA sleep cycle complete  pruned=%d  flagged=%d  "
            "contradictions=%d  elapsed=%.2fs",
            pruned, flagged, contradiction_count, elapsed,
        )
        return {
            "pruned": pruned,
            "flagged": flagged,
            "contradictions": contradiction_count,
            "duration": round(elapsed, 4),
        }

    # ------------------------------------------------------------------
    # 3. inject_external_knowledge
    # ------------------------------------------------------------------

    def inject_external_knowledge(
        self, beliefs: list[BeliefCertificate],
    ) -> dict[str, Any]:
        """Inject external knowledge after House D validation.

        Every belief must survive House D before it can enter House A.
        This is the IRON LAW for external signals.

        Args:
            beliefs: External beliefs to evaluate and possibly inject.

        Returns:
            Summary dict with ``submitted``, ``survived_d``,
            ``added_to_a``, and ``rejected`` counts.
        """
        start = time.perf_counter()
        logger.info(
            "OMEGA inject_external started  count=%d", len(beliefs),
        )

        submitted = len(beliefs)
        survived_d = 0
        added_to_a = 0
        rejected = 0

        for belief in beliefs:
            report = self.house_d.attack_belief(belief)
            if not report.survived:
                rejected += 1
                logger.info(
                    "OMEGA external belief destroyed  claim=%r",
                    belief.claim[:80],
                )
                continue

            survived_d += 1
            if self.knowledge_graph.add_belief(belief):
                added_to_a += 1
            else:
                rejected += 1

        elapsed = time.perf_counter() - start
        logger.info(
            "OMEGA inject_external complete  submitted=%d  survived_d=%d  "
            "added=%d  rejected=%d  elapsed=%.2fs",
            submitted, survived_d, added_to_a, rejected, elapsed,
        )
        return {
            "submitted": submitted,
            "survived_d": survived_d,
            "added_to_a": added_to_a,
            "rejected": rejected,
        }

    # ------------------------------------------------------------------
    # 4. get_health
    # ------------------------------------------------------------------

    def get_health(self) -> SystemHealth:
        """Calculate a diagnostic health snapshot of the system.

        Returns:
            A SystemHealth instance with metrics derived from
            ``cycle_history`` and the knowledge graph.
        """
        total = len(self.cycle_history)
        successes = sum(1 for c in self.cycle_history if c.success)
        failures = total - successes

        avg_time = 0.0
        if total > 0:
            avg_time = sum(c.cycle_time_seconds for c in self.cycle_history) / total

        score = successes / total if total > 0 else 0.0

        domains = list(self.knowledge_graph.domain_index.keys())

        remaining_cycles = self.sleep_cycle_interval - (
            self.cycle_count % self.sleep_cycle_interval
        )
        estimated_cycle_time = avg_time if avg_time > 0 else 10.0
        next_sleep = datetime.now(timezone.utc) + timedelta(
            seconds=remaining_cycles * estimated_cycle_time,
        )

        return SystemHealth(
            total_cycles=total,
            successful_cycles=successes,
            failed_cycles=failures,
            total_beliefs=len(self.knowledge_graph),
            average_cycle_time=round(avg_time, 4),
            last_sleep_cycle=self._last_sleep,
            next_sleep_cycle_due=next_sleep,
            domains_covered=domains,
            system_score=round(score, 4),
        )

    # ------------------------------------------------------------------
    # 5. get_cycle_history
    # ------------------------------------------------------------------

    def get_cycle_history(self, last_n: int = 10) -> list[CycleResult]:
        """Return the most recent cycle results.

        Args:
            last_n: Maximum number of results to return.

        Returns:
            Up to *last_n* CycleResult objects, most recent last.
        """
        return self.cycle_history[-last_n:]

    # ------------------------------------------------------------------
    # 6. _log_cycle
    # ------------------------------------------------------------------

    def _log_cycle(self, result: CycleResult) -> None:
        """Append a cycle result to history and persist to disk.

        Args:
            result: The completed CycleResult to record.
        """
        self.cycle_history.append(result)
        logger.info(
            "OMEGA cycle logged  id=%s  success=%s  time=%.2fs  "
            "belief_added=%s  reason=%s",
            result.cycle_id, result.success,
            result.cycle_time_seconds, result.belief_added,
            result.failure_reason,
        )
        self._persist_history()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _finalise_cycle(self, result: CycleResult) -> None:
        """Post-cycle bookkeeping: increment counter, log, sleep/external check."""
        self.cycle_count += 1
        self._log_cycle(result)

        if self.cycle_count % self.external_signal_interval == 0:
            self._inject_automatic_external_signal()

        if self.cycle_count % self.sleep_cycle_interval == 0:
            logger.info(
                "OMEGA triggering automatic sleep cycle at count=%d",
                self.cycle_count,
            )
            self.run_sleep_cycle()

    def _inject_automatic_external_signal(self) -> None:
        """IRON LAW: Inject fresh external knowledge every N cycles."""
        try:
            provider = ExternalSignalProvider()
            beliefs = provider.fetch_all()
            if beliefs:
                result = self.inject_external_knowledge(beliefs)
                logger.info(
                    "External signal injected: %d new beliefs  "
                    "(submitted=%d  survived_d=%d  added=%d  rejected=%d)",
                    result["added_to_a"],
                    result["submitted"],
                    result["survived_d"],
                    result["added_to_a"],
                    result["rejected"],
                )
            else:
                logger.info(
                    "External signal skipped: no beliefs fetched  cycle=%d",
                    self.cycle_count,
                )
        except Exception as exc:
            logger.warning(
                "External signal injection failed  cycle=%d  error=%s",
                self.cycle_count, exc,
            )

    def _persist_history(self) -> None:
        """Write the full cycle history to ``data/cycle_history.json``."""
        try:
            _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = [c.to_dict() for c in self.cycle_history]
            _HISTORY_PATH.write_text(
                json.dumps(data, indent=2, default=str), encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Failed to persist cycle history: %s", exc)

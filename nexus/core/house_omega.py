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
import os
import pathlib
import time
import uuid
import gc
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from nexus.core.belief_certificate import BeliefCertificate
from nexus.core.external_signal import ExternalSignalProvider
from nexus.core.house_b import HouseB, StructuredSpecificationObject
from nexus.core.house_c import BuildArtifact, BuildResult, HouseC
from nexus.core.house_d import AttackResult, DestructionReport, HouseD
from nexus.core import database as nexus_db
from nexus.core.database import load_cycle_history, save_cycle_history
from nexus.core.knowledge_graph import KnowledgeGraph
from nexus.core.anti_belief import AntiBeliefGraph
from nexus.core.bounty import BountySystem
from nexus.core.skill_library import SkillLibrary
from nexus.core.counterfactual import CounterfactualLog, groq_validate_rejected_prediction

logger: logging.Logger = logging.getLogger(__name__)

_HISTORY_PATH: pathlib.Path = pathlib.Path("data/cycle_history.json")
_BOUNDARY_PATH: pathlib.Path = pathlib.Path("data/boundary_pairs.json")
_GOVERNOR_ALERTS_PATH: pathlib.Path = pathlib.Path("data/governor_alerts.json")

_MAX_CYCLE_HISTORY: int = 50  # memory guard: keep only last N cycles in RAM + persistence
_MAX_LOG_CHARS: int = 4000    # trim large stdout/stderr/proofs stored on cycles


def _rss_mb() -> float:
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
    except Exception:
        return -1.0


def _compact_cycle(result: "CycleResult") -> None:
    """Trim heavy fields on build artifacts to prevent per-cycle memory growth."""
    br = result.build_result
    if not br or not getattr(br, "artifact", None):
        return
    art = br.artifact
    # Keep only metadata required for debugging; drop big strings.
    if isinstance(getattr(art, "code", None), str):
        art.code = art.code[: min(len(art.code), _MAX_LOG_CHARS)]
    if isinstance(getattr(art, "tests", None), str):
        art.tests = art.tests[: min(len(art.tests), _MAX_LOG_CHARS)]
    if isinstance(getattr(art, "documentation", None), str):
        art.documentation = art.documentation[: min(len(art.documentation), _MAX_LOG_CHARS)]
    if isinstance(getattr(art, "execution_proof", None), str) and art.execution_proof:
        art.execution_proof = art.execution_proof[: min(len(art.execution_proof), _MAX_LOG_CHARS)]
    # Validation errors can be huge (full pytest output); keep only a tail/head slice.
    if isinstance(getattr(art, "validation_errors", None), list) and art.validation_errors:
        trimmed: list[str] = []
        for e in art.validation_errors[:3]:
            if isinstance(e, str):
                trimmed.append(e[: min(len(e), _MAX_LOG_CHARS)])
        art.validation_errors = trimmed


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


def cycle_result_from_dict(d: dict[str, Any]) -> CycleResult:
    """Rehydrate CycleResult from persisted JSON (best-effort)."""
    cr = CycleResult(
        cycle_id=str(d.get("cycle_id") or uuid.uuid4()),
        user_input=str(d.get("user_input") or ""),
        belief_added=bool(d.get("belief_added")),
        cycle_time_seconds=float(d.get("cycle_time_seconds") or 0),
        success=bool(d.get("success")),
        failure_reason=d.get("failure_reason"),
        refinement_attempts=int(d.get("refinement_attempts") or 0),
    )
    try:
        if d.get("sso") and isinstance(d["sso"], dict):
            cr.sso = StructuredSpecificationObject.from_dict(d["sso"])
    except Exception:
        pass
    try:
        if d.get("destruction_report") and isinstance(d["destruction_report"], dict):
            dr = d["destruction_report"]
            attacks = [
                AttackResult(
                    target=str(a.get("target", "")),
                    attack_type=str(a.get("attack_type", "")),
                    severity=float(a.get("severity", 0)),
                    description=str(a.get("description", "")),
                    is_fatal=bool(a.get("is_fatal")),
                )
                for a in dr.get("attacks", [])
                if isinstance(a, dict)
            ]
            cr.destruction_report = DestructionReport(
                target_description=str(dr.get("target_description", "")),
                attacks=attacks,
                survived=bool(dr.get("survived", True)),
                survival_score=float(dr.get("survival_score", 1.0)),
                cycles_survived=int(dr.get("cycles_survived", 0)),
                recommendation=str(dr.get("recommendation", "REJECT")),
            )
    except Exception:
        pass
    try:
        if d.get("build_result") and isinstance(d["build_result"], dict):
            br = d["build_result"]
            ad = br.get("artifact") or {}
            sso_raw = ad.get("sso") or {}
            if isinstance(sso_raw, dict) and sso_raw.get("original_input") is not None:
                sso = StructuredSpecificationObject.from_dict(sso_raw)
            else:
                sso = StructuredSpecificationObject(
                    original_input="", redefined_problem="",
                )
            cat = ad.get("created_at")
            try:
                created_at = datetime.fromisoformat(str(cat)) if cat else datetime.now(timezone.utc)
            except Exception:
                created_at = datetime.now(timezone.utc)
            art = BuildArtifact(
                artifact_id=str(ad.get("artifact_id") or uuid.uuid4()),
                sso=sso,
                code=str(ad.get("code") or ""),
                language=str(ad.get("language") or "python"),
                tests=str(ad.get("tests") or ""),
                documentation=str(ad.get("documentation") or ""),
                created_at=created_at,
                passed_validation=bool(ad.get("passed_validation")),
                validation_errors=list(ad.get("validation_errors") or []),
                execution_proof=ad.get("execution_proof"),
                healing_attempts=int(ad.get("healing_attempts") or 0),
            )
            hdr = br.get("house_d_report") or {}
            h_attacks = [
                AttackResult(
                    target=str(a.get("target", "")),
                    attack_type=str(a.get("attack_type", "")),
                    severity=float(a.get("severity", 0)),
                    description=str(a.get("description", "")),
                    is_fatal=bool(a.get("is_fatal")),
                )
                for a in hdr.get("attacks", [])
                if isinstance(a, dict)
            ]
            hdr_obj = DestructionReport(
                target_description=str(hdr.get("target_description", "")),
                attacks=h_attacks,
                survived=bool(hdr.get("survived", True)),
                survival_score=float(hdr.get("survival_score", 1.0)),
                cycles_survived=int(hdr.get("cycles_survived", 0)),
                recommendation=str(hdr.get("recommendation", "REJECT")),
            )
            cr.build_result = BuildResult(
                artifact=art,
                success=bool(br.get("success")),
                house_d_report=hdr_obj,
                ready_for_house_a=bool(br.get("ready_for_house_a")),
            )
    except Exception:
        pass
    return cr


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
    self_built_beliefs: int = 0
    autonomy_ratio: float = 0.0
    daily_cost: float = 0.0
    average_cycle_time: float = 0.0
    last_sleep_cycle: datetime | None = None
    next_sleep_cycle_due: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    domains_covered: list[str] = field(default_factory=list)
    system_score: float = 0.0
    total_skills: int = 0
    skills_used_this_cycle: int = 0
    total_counterfactuals: int = 0
    wrong_predictions: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise for reporting or persistence."""
        return {
            "total_cycles": self.total_cycles,
            "successful_cycles": self.successful_cycles,
            "failed_cycles": self.failed_cycles,
            "total_beliefs": self.total_beliefs,
            "self_built_beliefs": self.self_built_beliefs,
            "autonomy_ratio": round(self.autonomy_ratio, 4),
            "daily_cost": round(self.daily_cost, 4),
            "average_cycle_time": round(self.average_cycle_time, 4),
            "last_sleep_cycle": (
                self.last_sleep_cycle.isoformat()
                if self.last_sleep_cycle else None
            ),
            "next_sleep_cycle_due": self.next_sleep_cycle_due.isoformat(),
            "domains_covered": list(self.domains_covered),
            "system_score": round(self.system_score, 4),
            "total_skills": self.total_skills,
            "skills_used_this_cycle": self.skills_used_this_cycle,
            "total_counterfactuals": self.total_counterfactuals,
            "wrong_predictions": self.wrong_predictions,
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
    anti_beliefs: AntiBeliefGraph = field(default_factory=AntiBeliefGraph, repr=False)
    bounty_system: BountySystem = field(default_factory=BountySystem, repr=False)
    conflict_alerts: list[dict[str, Any]] = field(default_factory=list, repr=False)
    skill_library: SkillLibrary = field(default_factory=SkillLibrary, repr=False)
    counterfactual_log: CounterfactualLog = field(default_factory=CounterfactualLog, repr=False)

    def __post_init__(self) -> None:
        """Wire governor alert callback and skill library into Houses."""
        self.knowledge_graph.governor_alert = self._governor_conflict_alert
        self.house_b.skill_library = self.skill_library
        self.house_c.skill_library = self.skill_library
        self.house_b.counterfactual_log = self.counterfactual_log
        # Under pytest, do not hydrate from global cycle history (would leak
        # real runs into unit tests and break length assertions).
        if os.getenv("PYTEST_CURRENT_TEST"):
            self.cycle_history = []
            self.cycle_count = 0
        else:
            raw_hist = load_cycle_history(_HISTORY_PATH)
            self.cycle_history = [
                cycle_result_from_dict(x) for x in raw_hist if isinstance(x, dict)
            ]
            # Memory guard: keep only last N cycles, and compact heavy artifact fields.
            if len(self.cycle_history) > _MAX_CYCLE_HISTORY:
                self.cycle_history = self.cycle_history[-_MAX_CYCLE_HISTORY:]
            for c in self.cycle_history:
                _compact_cycle(c)
            self.cycle_count = len(self.cycle_history)

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
        self.skill_library.reset_usage_this_cycle()
        logger.info("MEM rss_mb=%.2f  stage=start", _rss_mb())
        logger.info(
            "OMEGA cycle %d started  input=%r",
            self.cycle_count + 1, user_input[:100],
        )

        try:
            # Anti-belief pre-check: build a minimal SSO-shaped object and ask if blocked.
            pre_sso = StructuredSpecificationObject(
                original_input=user_input,
                redefined_problem=user_input,
            )
            if self.anti_beliefs.is_blocked(pre_sso):
                result.failure_reason = "BLOCKED BY ANTI-BELIEF"
                logger.warning("OMEGA cycle blocked by anti-belief graph  input=%r", user_input[:100])
                result.cycle_time_seconds = time.perf_counter() - start
                self._finalise_cycle(result)
                return result

            # Determine bounty for this task (for router tier control).
            task_key = user_input.strip().lower()
            bounty = self.bounty_system.get_bounty(task_key)

            # Step 1 — House B: redefine
            sso = self.house_b.redefine(user_input, cycle_id=result.cycle_id)
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
                # Record failure into anti-belief graph and bounty system.
                self.anti_beliefs.add_failure(result)
                self.bounty_system.record_failure(task_key)
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
                self.anti_beliefs.add_failure(result)
                self.bounty_system.record_failure(task_key)
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
                self.anti_beliefs.add_failure(result)
                self.bounty_system.record_failure(task_key)
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
                self.bounty_system.record_success(task_key)
                skill = self.skill_library.compile_from_belief(belief)
                if skill:
                    logger.info("OMEGA skill compiled and added to library  name=%s", skill.name)
            logger.info("OMEGA step 6 complete  belief_added=%s", added)

        except Exception as exc:
            result.failure_reason = f"{type(exc).__name__}: {exc}"
            logger.exception("OMEGA cycle failed with exception")

        result.cycle_time_seconds = time.perf_counter() - start
        self._finalise_cycle(result)
        return result

    # ------------------------------------------------------------------
    # 2. run_light_sleep (pruning only, no LLM) — every 10 cycles
    # ------------------------------------------------------------------

    def run_light_sleep(self) -> dict[str, Any]:
        """Pruning-only sleep: no LLM calls. Run every 10 cycles."""
        start = time.perf_counter()
        logger.info("OMEGA light sleep started  cycle_count=%d", self.cycle_count)

        pruned = self.knowledge_graph.prune_expired()
        rev = self.knowledge_graph.reverify_beliefs_past_due()
        logger.info(
            "OMEGA light sleep proof re-verify  checked=%d reverified=%d quarantined=%d",
            rev.get("checked", 0),
            rev.get("reverified", 0),
            rev.get("quarantined", 0),
        )

        snap = self.knowledge_graph.beliefs_snapshot()
        low_confidence = [b for b in snap if b.confidence < 0.6]
        flagged = len(low_confidence)
        for b in low_confidence:
            logger.info(
                "SLEEP flagged low-confidence  claim=%r  confidence=%s",
                b.claim[:60], b.confidence,
            )

        contradiction_count = 0
        for b in snap:
            conflicts = self.knowledge_graph.contradiction_check(
                b.claim, snap,
            )
            if conflicts:
                contradiction_count += 1
                logger.info(
                    "SLEEP contradiction cluster  claim=%r  conflicts=%s",
                    b.claim[:60], conflicts,
                )

        elapsed = time.perf_counter() - start
        logger.info(
            "OMEGA light sleep complete  pruned=%d  flagged=%d  "
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
    # 3. run_sleep_cycle (deep sleep: light + chained inference) — every 50 cycles
    # ------------------------------------------------------------------

    def run_sleep_cycle(self) -> dict[str, Any]:
        """Deep sleep: run light_sleep (pruning) then chained inference + LLM.

        Steps:
        1. Run light_sleep (prune, flag, contradiction check).
        2. Re-verify and re-inject the top beliefs.
        3. Chained inference: synthesise up to 5 new beliefs from pairs (LLM).
        4. Persist and log.
        """
        start = time.perf_counter()
        logger.info("OMEGA deep sleep started  cycle_count=%d", self.cycle_count)

        light_result = self.run_light_sleep()
        pruned = light_result["pruned"]
        flagged = light_result["flagged"]
        contradiction_count = light_result["contradictions"]

        top_beliefs = sorted(
            self.knowledge_graph.beliefs_snapshot(),
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
                attempts=list(b.attempts),
                lessons_learned=list(b.lessons_learned),
                semantic_triples=list(getattr(b, "semantic_triples", []) or []),
                conflict_flag=getattr(b, "conflict_flag", None),
            ))
        if refreshed:
            self.knowledge_graph.inject_external_signal(refreshed)

        # Chained inference: synthesise up to 5 new beliefs from pairs.
        inferred = 0
        max_inferred = 5
        by_domain: dict[str, list[BeliefCertificate]] = {}
        for b in self.knowledge_graph.beliefs_snapshot():
            if not b.is_valid() or b.is_expired():
                continue
            by_domain.setdefault(b.domain, []).append(b)

        for domain, beliefs in by_domain.items():
            if inferred >= max_inferred:
                break
            n = len(beliefs)
            for i in range(n):
                for j in range(i + 1, n):
                    if inferred >= max_inferred:
                        break
                    b1, b2 = beliefs[i], beliefs[j]
                    payload = self._synthesise_belief(b1, b2)
                    if not payload or "claim" not in payload:
                        continue
                    candidate = BeliefCertificate(
                        claim=payload["claim"],
                        source="nexus:omega:inference",
                        confidence=float(payload.get("confidence", 0.7)),
                        contradictions=[],
                        decay_rate=0.05,
                        downstream_dependents=[],
                        executable_proof=payload.get("executable_proof"),
                        domain=payload.get("domain") or domain,
                    )
                    report = self.house_d.attack_belief(candidate)
                    if not report.survived:
                        continue
                    added = self.knowledge_graph.add_belief(candidate)
                    if added:
                        inferred += 1


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

    def _synthesise_belief(
        self,
        b1: BeliefCertificate,
        b2: BeliefCertificate,
    ) -> dict[str, Any] | None:
        """Ask House B's router to synthesise a higher-order belief from two inputs."""
        system = (
            "You are a knowledge synthesis engine.\n"
            "Given two verified beliefs, generate ONE new higher-order belief that combines them.\n"
            "Return strict JSON with keys: claim, domain, confidence, executable_proof.\n"
            "The claim must be precise, falsifiable, and suitable for automated testing."
        )
        user_payload = {
            "belief_1": {"claim": b1.claim, "domain": b1.domain},
            "belief_2": {"claim": b2.claim, "domain": b2.domain},
        }
        try:
            raw = self.house_b.router.complete(
                house="house_b",
                system=system,
                user=json.dumps(user_payload, ensure_ascii=False),
                label="inference",
            )
            return json.loads(raw)
        except Exception:
            return None

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

        total_beliefs = len(self.knowledge_graph)
        self_built = sum(
            1 for b in self.knowledge_graph.beliefs_snapshot()
            if b.source.startswith("nexus:house_c:artifact:")
        )
        autonomy_ratio = self_built / total_beliefs if total_beliefs > 0 else 0.0

        # Read daily LLM cost from the router's tracking file, if present.
        daily_cost = 0.0
        cost_path = pathlib.Path("data/daily_cost.json")
        try:
            dc = nexus_db.load_daily_cost(cost_path)
            date_str = str(dc.get("date") or "")
            total_cost = float(dc.get("total_cost") or 0.0)
            if date_str:
                today = datetime.now(timezone.utc).date().isoformat()
                if date_str == today:
                    daily_cost = total_cost
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("HEALTH cost load failed: %s", exc)

        domains = list(self.knowledge_graph.domain_index.keys())

        remaining_cycles = self.sleep_cycle_interval - (
            self.cycle_count % self.sleep_cycle_interval
        )
        estimated_cycle_time = avg_time if avg_time > 0 else 10.0
        next_sleep = datetime.now(timezone.utc) + timedelta(
            seconds=remaining_cycles * estimated_cycle_time,
        )

        total_skills = len(self.skill_library.skills)
        skills_used_this_cycle = self.skill_library.usage_this_cycle
        total_counterfactuals = len(self.counterfactual_log.entries)
        wrong_predictions = self.counterfactual_log.wrong_predictions

        return SystemHealth(
            total_cycles=total,
            successful_cycles=successes,
            failed_cycles=failures,
            total_beliefs=total_beliefs,
            self_built_beliefs=self_built,
            autonomy_ratio=round(autonomy_ratio, 4),
            daily_cost=round(daily_cost, 4),
            average_cycle_time=round(avg_time, 4),
            last_sleep_cycle=self._last_sleep,
            next_sleep_cycle_due=next_sleep,
            domains_covered=domains,
            system_score=round(score, 4),
            total_skills=total_skills,
            skills_used_this_cycle=skills_used_this_cycle,
            total_counterfactuals=total_counterfactuals,
            wrong_predictions=wrong_predictions,
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
        if len(self.cycle_history) > _MAX_CYCLE_HISTORY:
            self.cycle_history = self.cycle_history[-_MAX_CYCLE_HISTORY:]
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

    def _governor_conflict_alert(self, payload: dict[str, Any]) -> None:
        """Called by KnowledgeGraph when semantic contradiction is detected."""
        self.conflict_alerts.append(payload)
        logger.warning("GOVERNOR CONFLICT ALERT: %s", payload)
        try:
            existing = nexus_db.load_governor_alerts(_GOVERNOR_ALERTS_PATH)
            existing.append({**payload, "timestamp": datetime.now(timezone.utc).isoformat()})
            nexus_db.save_governor_alerts(existing, _GOVERNOR_ALERTS_PATH)
        except OSError as exc:
            logger.warning("Failed to persist governor alert: %s", exc)

    def _record_boundary_pair(self, result: CycleResult) -> None:
        """Ask Gemini free: minimal change that would flip this outcome; store in boundary_pairs.json."""
        try:
            summary = (
                f"success={result.success} belief_added={result.belief_added} "
                f"reason={result.failure_reason or 'none'}"
            )
            user_snip = (result.user_input or "")[:500]
            prompt = (
                f"NEXUS cycle outcome: {summary}\n\n"
                f"User input: {user_snip}\n\n"
                "What is the minimal change that would flip this outcome? "
                "Answer in one or two sentences."
            )
            raw = self.house_b.router.complete(
                house="house_b",
                system="Answer concisely. One or two sentences only.",
                user=prompt,
                label="boundary",
            )
            minimal_flip = (raw or "").strip()[:1000]
            pair = {
                "cycle_id": result.cycle_id,
                "success": result.success,
                "user_input": user_snip,
                "outcome_summary": summary,
                "minimal_flip_change": minimal_flip,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            existing = nexus_db.load_boundary_pairs(_BOUNDARY_PATH)
            existing.append(pair)
            nexus_db.save_boundary_pairs(existing, _BOUNDARY_PATH)
        except Exception as exc:
            logger.debug("Boundary pair recording failed (non-fatal): %s", exc)

    def _run_background_counterfactual_check(self) -> None:
        """Every 10 cycles: probe 2 rejected redefinitions via House D mini-cycle."""
        pairs = self.counterfactual_log.pick_rejected_pairs(max_pairs=2)
        for e, rc, idx in pairs:
            self.counterfactual_log.mark_background_seen(e, idx)
            action = (rc.get("action") or "")[:3000]
            if not action.strip():
                continue
            pred = (rc.get("predicted_outcome") or "").lower()
            try:
                alt_sso = StructuredSpecificationObject(
                    original_input=e.chosen_action[:500] or "counterfactual",
                    redefined_problem=action,
                    domain="General",
                    confidence=0.5,
                )
                dr = self.house_d.attack_sso(alt_sso)
                survived = dr.survived
                predicted_failure = any(
                    w in pred for w in ("fail", "reject", "destroy", "invalid", "weak")
                )
                predicted_success = any(
                    w in pred for w in ("success", "pass", "survive", "work")
                )
                wrong = False
                if predicted_failure and survived:
                    wrong = True
                elif predicted_success and not survived:
                    wrong = True
                if not wrong:
                    g = groq_validate_rejected_prediction(
                        e.actual_outcome or "unknown",
                        action,
                        rc.get("predicted_outcome", ""),
                    )
                    if g is False:
                        wrong = True
                if wrong:
                    self.counterfactual_log.wrong_predictions += 1
                    self.counterfactual_log.save()
                    logger.warning(
                        "COUNTERFACTUAL: prediction was wrong — updating world model  "
                        "cycle_id=%s  action=%r",
                        e.cycle_id,
                        action[:80],
                    )
            except Exception as exc:
                logger.debug("background counterfactual check failed: %s", exc)
        if pairs:
            self.counterfactual_log.save()

    def _finalise_cycle(self, result: CycleResult) -> None:
        """Post-cycle bookkeeping: increment counter, log, sleep/external check."""
        # Trim heavy fields before storing in history/persistence.
        _compact_cycle(result)
        rss_before = _rss_mb()
        self.cycle_count += 1
        self._log_cycle(result)
        actual = (
            "SUCCESS: belief added"
            if result.success
            else f"FAILURE: {result.failure_reason or 'unknown'}"
        )
        try:
            self.counterfactual_log.validate_predictions(result.cycle_id, actual)
        except Exception as exc:
            logger.debug("validate_predictions failed: %s", exc)
        try:
            self._record_boundary_pair(result)
        except Exception as exc:
            logger.debug("Boundary recording failed: %s", exc)

        if self.cycle_count % 10 == 0:
            try:
                self._run_background_counterfactual_check()
            except Exception as exc:
                logger.debug("background counterfactual: %s", exc)

        if self.cycle_count % self.external_signal_interval == 0:
            self._inject_automatic_external_signal()

        if self.cycle_count % self.sleep_cycle_interval == 0:
            logger.info(
                "OMEGA triggering deep sleep at count=%d",
                self.cycle_count,
            )
            self.run_sleep_cycle()
        elif self.cycle_count % 10 == 0:
            logger.info(
                "OMEGA triggering light sleep at count=%d",
                self.cycle_count,
            )
            self.run_light_sleep()

        # Explicit GC to combat RSS growth on small dynos (requested).
        gc.collect()
        logger.info(
            "MEM rss_mb=%.2f  stage=finalise  delta_mb=%.2f  history=%d",
            _rss_mb(),
            (_rss_mb() - rss_before) if rss_before >= 0 else 0.0,
            len(self.cycle_history),
        )

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
        """Persist cycle history to Supabase (if configured) or JSON file."""
        try:
            data = [c.to_dict() for c in self.cycle_history[-_MAX_CYCLE_HISTORY:]]
            save_cycle_history(data, _HISTORY_PATH)
        except OSError as exc:
            logger.warning("Failed to persist cycle history: %s", exc)

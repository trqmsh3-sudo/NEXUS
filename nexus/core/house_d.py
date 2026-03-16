"""House D — The Destroyer.

House D's ONLY job is to attack and destroy outputs before they enter
House A.  It does NOT build.  It does NOT help.  It KILLS weak ideas
and code.

Nothing gets promoted to House A without passing through House D.
A single fatal attack is enough to reject.
"""

from __future__ import annotations

import enum
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from nexus.core.belief_certificate import BeliefCertificate
from nexus.core.house_b import StructuredSpecificationObject
from nexus.core.knowledge_graph import KnowledgeGraph
from nexus.core.model_router import ModelRouter

logger: logging.Logger = logging.getLogger(__name__)

_MAX_RETRIES: int = 1

_FATAL_ATTACK_TYPES: frozenset[str] = frozenset({
    "SECURITY_VULNERABILITY",
    "UNSOLVABLE_PROBLEM",
    "ETHICAL_VIOLATION",
})

# ------------------------------------------------------------------
# System prompt
# ------------------------------------------------------------------

DESTROYER_SYSTEM: str = (
    "You are the Destroyer. Your only goal is to find real flaws.\n"
    "Attack the logic. Attack the assumptions. Attack the ethics.\n"
    "Be specific. Be honest. Return ONLY valid JSON.\n\n"
    "Severity scoring rules -- follow strictly:\n"
    "- 0.9-1.0: ONLY for attacks that make the entire system "
    "fundamentally impossible to build\n"
    "- 0.7-0.89: Serious flaws that require major redesign\n"
    "- 0.5-0.69: Significant issues that need addressing but "
    "system can still work\n"
    "- 0.3-0.49: Minor flaws, easily fixable\n"
    "- 0.1-0.29: Nitpicks, style issues\n\n"
    "Most real specs have flaws in the 0.5-0.7 range.\n"
    "Reserve 0.9+ for truly fatal architectural impossibilities.\n"
    "Not every flaw is catastrophic.\n"
    "is_fatal should be true ONLY for flaws that make the system "
    "completely undeliverable.\n\n"
    "The JSON must have exactly this structure:\n"
    '{\n'
    '  "attacks": [\n'
    '    {\n'
    '      "attack_type": one of "LOGIC_FLAW", "SECURITY_VULNERABILITY", '
    '"SCALABILITY_FAILURE", "HIDDEN_ASSUMPTION", "ETHICAL_VIOLATION", '
    '"CONTRADICTS_KNOWN_TRUTH", "UNMAINTAINABLE", "UNSOLVABLE_PROBLEM",\n'
    '      "severity": float 0.0 to 1.0,\n'
    '      "description": string (exact description of the flaw),\n'
    '      "is_fatal": boolean\n'
    '    }\n'
    '  ]\n'
    '}\n'
)


# ------------------------------------------------------------------
# Enum
# ------------------------------------------------------------------

class AttackType(enum.Enum):
    """Categories of destructive attacks House D can launch."""

    LOGIC_FLAW = "LOGIC_FLAW"
    SECURITY_VULNERABILITY = "SECURITY_VULNERABILITY"
    SCALABILITY_FAILURE = "SCALABILITY_FAILURE"
    HIDDEN_ASSUMPTION = "HIDDEN_ASSUMPTION"
    ETHICAL_VIOLATION = "ETHICAL_VIOLATION"
    CONTRADICTS_KNOWN_TRUTH = "CONTRADICTS_KNOWN_TRUTH"
    UNMAINTAINABLE = "UNMAINTAINABLE"
    UNSOLVABLE_PROBLEM = "UNSOLVABLE_PROBLEM"


# ------------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------------

@dataclass
class AttackResult:
    """A single flaw discovered during a destruction cycle.

    Attributes:
        target: Description of what was attacked.
        attack_type: Category of the attack (see :class:`AttackType`).
        severity: How bad the flaw is, from 0.0 (trivial) to 1.0
            (catastrophic).
        description: Precise description of the flaw found.
        is_fatal: True if this flaw alone should kill the target.
    """

    target: str
    attack_type: str
    severity: float
    description: str
    is_fatal: bool

    def __post_init__(self) -> None:
        """Clamp severity to [0.0, 1.0]."""
        self.severity = max(0.0, min(1.0, self.severity))

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "target": self.target,
            "attack_type": self.attack_type,
            "severity": self.severity,
            "description": self.description,
            "is_fatal": self.is_fatal,
        }


@dataclass
class DestructionReport:
    """Aggregated result of all attack cycles against a target.

    Attributes:
        target_description: Human-readable description of the target.
        attacks: Every individual attack discovered across all cycles.
        survived: True only when zero fatal attacks were found.
        survival_score: Aggregate score from 0.0 (destroyed) to 1.0
            (bulletproof).
        cycles_survived: Number of attack cycles the target endured.
        recommendation: One of ``"PROMOTE"``, ``"REVISE"``, or
            ``"REJECT"``.
    """

    target_description: str
    attacks: list[AttackResult] = field(default_factory=list)
    survived: bool = True
    survival_score: float = 1.0
    cycles_survived: int = 0
    recommendation: str = "REJECT"

    def to_dict(self) -> dict[str, Any]:
        """Serialise the full report."""
        return {
            "target_description": self.target_description,
            "attacks": [a.to_dict() for a in self.attacks],
            "survived": self.survived,
            "survival_score": self.survival_score,
            "cycles_survived": self.cycles_survived,
            "recommendation": self.recommendation,
        }


# ------------------------------------------------------------------
# House D — The Destroyer
# ------------------------------------------------------------------

@dataclass
class HouseD:
    """The Destroyer — attacks and kills weak ideas before they reach House A.

    House D runs multiple attack cycles against an SSO or a
    BeliefCertificate, using a cloud LLM (Mistral AI) to discover
    fatal flaws. A single fatal flaw is enough to reject the target.

    Attributes:
        knowledge_graph: The shared NEXUS knowledge store, used for
            contradiction checks against verified beliefs.
        router: ModelRouter for LLM calls.
        min_cycles: Minimum number of attack rounds before promotion
            is possible.
    """

    knowledge_graph: KnowledgeGraph
    router: ModelRouter = field(default_factory=ModelRouter)
    min_cycles: int = 3

    # ------------------------------------------------------------------
    # 1. attack_sso
    # ------------------------------------------------------------------

    def attack_sso(
        self, sso: StructuredSpecificationObject,
    ) -> DestructionReport:
        """Launch a full destruction campaign against an SSO.

        Runs :attr:`min_cycles` rounds of LLM-powered attacks.
        Aggregates every flaw into a :class:`DestructionReport` and
        computes a survival score and recommendation.

        Args:
            sso: The StructuredSpecificationObject to attack.

        Returns:
            A DestructionReport summarising all discovered flaws.
        """
        start = time.perf_counter()
        target_desc = (
            f"SSO: {sso.redefined_problem} "
            f"(domain={sso.domain}, confidence={sso.confidence})"
        )
        logger.info("HOUSE-D attack_sso started  target=%r", target_desc[:120])

        target_text = (
            f"Problem: {sso.redefined_problem}\n"
            f"Assumptions: {json.dumps(sso.assumptions)}\n"
            f"Constraints: {json.dumps(sso.constraints)}\n"
            f"Success criteria: {json.dumps(sso.success_criteria)}\n"
            f"Required inputs: {json.dumps(sso.required_inputs)}\n"
            f"Expected outputs: {json.dumps(sso.expected_outputs)}\n"
            f"Domain: {sso.domain}\n"
            f"Confidence: {sso.confidence}"
        )

        all_attacks: list[AttackResult] = []
        for cycle in range(1, self.min_cycles + 1):
            cycle_attacks = self.run_cycle(target_text, cycle)
            for a in cycle_attacks:
                a.target = target_desc
            all_attacks.extend(cycle_attacks)

        report = self._build_report(target_desc, all_attacks)
        elapsed = time.perf_counter() - start
        logger.info(
            "HOUSE-D attack_sso complete  survived=%s  score=%.2f  "
            "recommendation=%s  attacks=%d  elapsed=%.2fs",
            report.survived, report.survival_score,
            report.recommendation, len(report.attacks), elapsed,
        )
        return report

    # ------------------------------------------------------------------
    # 2. attack_belief
    # ------------------------------------------------------------------

    def attack_belief(
        self, belief: BeliefCertificate,
    ) -> DestructionReport:
        """Launch a full destruction campaign against a BeliefCertificate.

        First checks for contradictions against the knowledge graph.
        If any are found, an automatic fatal attack is injected. Then
        runs :attr:`min_cycles` rounds of LLM-powered attacks.

        Args:
            belief: The BeliefCertificate to attack.

        Returns:
            A DestructionReport summarising all discovered flaws.
        """
        start = time.perf_counter()
        target_desc = (
            f"Belief: {belief.claim} "
            f"(source={belief.source}, confidence={belief.confidence})"
        )
        logger.info("HOUSE-D attack_belief started  target=%r", target_desc[:120])

        all_attacks: list[AttackResult] = []

        contradictions = self.knowledge_graph.contradiction_check(
            belief.claim, list(self.knowledge_graph.beliefs.values()),
        )
        if contradictions:
            all_attacks.append(AttackResult(
                target=target_desc,
                attack_type=AttackType.CONTRADICTS_KNOWN_TRUTH.value,
                severity=1.0,
                description=(
                    f"Contradicts verified knowledge: {contradictions}"
                ),
                is_fatal=True,
            ))
            logger.warning(
                "HOUSE-D auto-fatal  claim=%r contradicts %s",
                belief.claim, contradictions,
            )

        target_text = (
            f"Belief claim: {belief.claim}\n"
            f"Source: {belief.source}\n"
            f"Confidence: {belief.confidence}\n"
            f"Domain: {belief.domain}\n"
            f"Executable proof: {belief.executable_proof or '(none)'}\n"
            f"Contradictions: {json.dumps(belief.contradictions)}\n"
            f"Decay rate: {belief.decay_rate}"
        )

        for cycle in range(1, self.min_cycles + 1):
            cycle_attacks = self.run_cycle(target_text, cycle)
            for a in cycle_attacks:
                a.target = target_desc
            all_attacks.extend(cycle_attacks)

        report = self._build_report(target_desc, all_attacks)
        elapsed = time.perf_counter() - start
        logger.info(
            "HOUSE-D attack_belief complete  survived=%s  score=%.2f  "
            "recommendation=%s  attacks=%d  elapsed=%.2fs",
            report.survived, report.survival_score,
            report.recommendation, len(report.attacks), elapsed,
        )
        return report

    # ------------------------------------------------------------------
    # 3. run_cycle
    # ------------------------------------------------------------------

    def run_cycle(
        self, target_description: str, cycle_num: int,
    ) -> list[AttackResult]:
        """Execute a single attack cycle against the target.

        Each cycle sends a fresh prompt to the LLM that includes the
        cycle number, forcing the model to explore different angles.

        Args:
            target_description: Textual description of the target.
            cycle_num: The 1-based cycle index (included in the prompt
                so the LLM tries different attack vectors each round).

        Returns:
            List of :class:`AttackResult` objects found in this cycle.

        Raises:
            ValueError: If the LLM returns invalid JSON after one retry.
        """
        start = time.perf_counter()
        logger.info("HOUSE-D cycle %d started", cycle_num)

        user_prompt = (
            f"ATTACK CYCLE {cycle_num}. "
            f"Find NEW flaws you have not found before.\n\n"
            f"Target:\n{target_description}\n\n"
            "Destroy it. Return ONLY the JSON object."
        )

        raw = self._call_llm(user_prompt, label=f"cycle-{cycle_num}")
        parsed = self._parse_json(raw, label=f"cycle-{cycle_num}")

        raw_attacks: list[dict[str, Any]] = parsed.get("attacks", [])
        results: list[AttackResult] = []
        for item in raw_attacks:
            attack_type_str = item.get("attack_type", "LOGIC_FLAW")
            if attack_type_str not in AttackType.__members__:
                attack_type_str = "LOGIC_FLAW"

            severity = float(item.get("severity", 0.5))
            is_fatal = (
                severity >= 0.85
                and attack_type_str in _FATAL_ATTACK_TYPES
            )

            result = AttackResult(
                target="",
                attack_type=attack_type_str,
                severity=severity,
                description=str(item.get("description", "No description")),
                is_fatal=is_fatal,
            )
            results.append(result)
            logger.info(
                "HOUSE-D cycle %d  attack_type=%s  severity=%.2f  "
                "fatal=%s  desc=%r",
                cycle_num, result.attack_type, result.severity,
                result.is_fatal, result.description[:100],
            )

        elapsed = time.perf_counter() - start
        logger.info(
            "HOUSE-D cycle %d complete  attacks_found=%d  elapsed=%.2fs",
            cycle_num, len(results), elapsed,
        )
        return results

    # ------------------------------------------------------------------
    # 4. should_promote
    # ------------------------------------------------------------------

    def should_promote(self, report: DestructionReport) -> bool:
        """Determine whether a target should be promoted to House A.

        Promotion requires all three conditions:

        1. ``report.survived`` is True (zero fatal attacks).
        2. ``report.cycles_survived`` >= :attr:`min_cycles`.
        3. ``report.survival_score`` >= 0.7.

        Args:
            report: The DestructionReport to evaluate.

        Returns:
            True only if all promotion criteria are met.
        """
        return (
            report.survived is True
            and report.cycles_survived >= self.min_cycles
            and report.survival_score >= 0.7
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_report(
        self,
        target_desc: str,
        attacks: list[AttackResult],
    ) -> DestructionReport:
        """Compute aggregated metrics and build a DestructionReport.

        Args:
            target_desc: Human-readable description of the target.
            attacks: All attacks collected across every cycle.

        Returns:
            A fully populated DestructionReport with survival score
            and recommendation.
        """
        has_fatal = any(a.is_fatal for a in attacks)
        survived = not has_fatal

        if attacks:
            avg_severity = sum(a.severity for a in attacks) / len(attacks)
            survival_score = round(max(0.0, 1.0 - avg_severity), 4)
        else:
            survival_score = 1.0

        if has_fatal and survival_score <= 0.2:
            recommendation = "REJECT"
        elif has_fatal:
            recommendation = "REVISE"
        elif survived and survival_score > 0.7:
            recommendation = "PROMOTE"
        elif survived and survival_score > 0.3:
            recommendation = "REVISE"
        else:
            recommendation = "REJECT"

        return DestructionReport(
            target_description=target_desc,
            attacks=attacks,
            survived=survived,
            survival_score=survival_score,
            cycles_survived=self.min_cycles,
            recommendation=recommendation,
        )

    def _call_llm(self, user: str, label: str) -> str:
        """Route an LLM call through the ModelRouter.

        Args:
            user: The user prompt.
            label: Human-readable label for logging.

        Returns:
            The raw text content from the LLM response.
        """
        return self.router.complete(
            house="house_d", system=DESTROYER_SYSTEM, user=user, label=label,
        )

    def _parse_json(self, raw: str, label: str) -> dict[str, Any]:
        """Parse a JSON string with one retry on failure.

        Args:
            raw: The raw string that should be valid JSON.
            label: Label for log and error messages.

        Returns:
            The parsed dictionary.

        Raises:
            ValueError: If parsing fails after the retry.
        """
        try:
            return json.loads(raw)
        except json.JSONDecodeError as first_err:
            logger.warning(
                "JSON parse failed [%s] (attempt 1): %s  raw=%r",
                label, first_err, raw[:200],
            )

        stripped = raw.strip()
        for prefix in ("```json", "```"):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        stripped = stripped.strip()

        try:
            return json.loads(stripped)
        except json.JSONDecodeError as second_err:
            logger.error(
                "JSON parse failed [%s] (attempt 2, giving up): %s",
                label, second_err,
            )
            raise ValueError(
                f"House D LLM returned invalid JSON for '{label}' after "
                f"{_MAX_RETRIES} retry. Raw output: {raw[:300]}"
            ) from second_err

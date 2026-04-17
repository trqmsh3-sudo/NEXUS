"""BeliefQualityFilter — LLM-based gate before belief storage.

Asks the LLM: "Will this belief help make a future decision?"
- store  + confidence >= 0.7 → True  (accept)
- discard OR confidence < 0.7 → False (reject)
- LLM failure / bad JSON      → False (safe default)
"""

from __future__ import annotations

import json
import logging

from nexus.core.belief_certificate import BeliefCertificate
from nexus.core.model_router import ModelRouter

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a belief quality evaluator for an autonomous business intelligence system.

Your job: decide whether a candidate belief is worth storing for future decision-making.

STORE if the belief contains:
- Actionable market data (prices, rates, demand levels)
- Discovered patterns or trends relevant to earning money
- Real obstacles or constraints a decision-maker must know
- Business insights that change what actions to take
- Specific opportunities with verifiable details

DISCARD if the belief is:
- A system status message ("loaded", "connected", "operational", "started")
- A technical metric (memory usage, cycle time, API call counts)
- A generic observation with no decision value
- Vague or unverifiable

Return ONLY valid JSON, no other text:
{"decision": "store" | "discard", "confidence": 0.0-1.0, "reason": "<one sentence>"}
"""


class BeliefQualityFilter:
    """LLM-powered filter that classifies beliefs as actionable or not."""

    def __init__(self, router: ModelRouter) -> None:
        self.router = router

    def is_actionable(self, belief: BeliefCertificate) -> bool:
        """Return True if the belief should be stored, False if it should be discarded.

        Calls the LLM with the belief claim. Discards by default on any failure.
        """
        user_prompt = (
            f"Belief claim: {belief.claim}\n"
            f"Domain: {belief.domain}\n"
            f"Source: {belief.source}\n\n"
            "Will this belief help make a future decision? Reply with JSON only."
        )
        try:
            raw = self.router.complete(
                system=_SYSTEM_PROMPT,
                user=user_prompt,
                house="belief_quality_filter",
                label="classify",
            )
            if not raw or not raw.strip():
                logger.info(
                    "QUALITY_FILTER discarded [empty LLM response]  claim=%r",
                    belief.claim[:80],
                )
                return False

            # Strip markdown fences if present
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            data = json.loads(text)
            decision = data.get("decision", "").lower()
            confidence = float(data.get("confidence", 0.0))

            if confidence < 0.7:
                logger.info(
                    "QUALITY_FILTER discarded [low confidence=%.2f]  claim=%r  reason=%s",
                    confidence, belief.claim[:80], data.get("reason", "")[:100],
                )
                return False

            if decision != "store":
                logger.info(
                    "QUALITY_FILTER discarded [decision=%r confidence=%.2f]  claim=%r  reason=%s",
                    decision, confidence, belief.claim[:80], data.get("reason", "")[:100],
                )
                return False

            logger.info(
                "QUALITY_FILTER approved [confidence=%.2f]  claim=%r",
                confidence, belief.claim[:80],
            )
            return True

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.info(
                "QUALITY_FILTER discarded [parse error: %s]  claim=%r",
                exc, belief.claim[:80],
            )
            return False
        except Exception as exc:
            logger.warning(
                "QUALITY_FILTER discarded [unexpected error: %s]  claim=%r",
                exc, belief.claim[:80],
            )
            return False

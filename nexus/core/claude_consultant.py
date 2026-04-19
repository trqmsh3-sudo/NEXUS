"""ClaudeConsultant — strategic reasoning via Claude API.

Used by StrategicAgent to consult Claude on complex decisions:
  - What domain to try next when multiple domains are failing
  - How to reframe a task that keeps producing no results
  - Summarising findings for the owner

Requires ANTHROPIC_API_KEY in the environment or GuardianVault.
Gracefully returns "" when no key is configured.
"""

from __future__ import annotations

import logging
import os

import litellm

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"
_DEFAULT_SYSTEM = (
    "You are a strategic advisor for an autonomous AI earning agent called PROXY. "
    "PROXY earns money by finding remote gig opportunities online and submitting proposals. "
    "Give short, actionable advice. Never ask clarifying questions. "
    "Respond in 1-3 sentences maximum."
)


class ClaudeConsultant:
    """Consults Claude API for strategic decisions.

    Args:
        api_key:       Anthropic API key. Falls back to ANTHROPIC_API_KEY env var if not provided.
        model:         litellm model string (default: anthropic/claude-sonnet-4-6).
        system_prompt: System prompt sent with every request.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        system_prompt: str = _DEFAULT_SYSTEM,
    ) -> None:
        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self._api_key = api_key or ""
        self.model = model
        self.system_prompt = system_prompt

    # ──────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True when an API key is configured."""
        return bool(self._api_key)

    def consult(self, question: str, max_tokens: int = 200) -> str:
        """Ask Claude a question. Returns "" when unavailable or on error."""
        if not self.is_available():
            return ""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                api_key=self._api_key,
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("ClaudeConsultant: API call failed — %s", exc)
            return ""

    def suggest_strategy(
        self,
        recent_failures: list[str],
        recent_successes: list[str],
    ) -> str:
        """Ask Claude to suggest the next task strategy based on recent history.

        Returns a task string, or "" when Claude is unavailable.
        """
        if not self.is_available():
            return ""

        failure_summary = ", ".join(recent_failures[-5:]) if recent_failures else "none"
        success_summary = ", ".join(recent_successes[-5:]) if recent_successes else "none"

        question = (
            f"PROXY has been running cycles to find remote gig opportunities. "
            f"Recent failing domains: {failure_summary}. "
            f"Recent successful domains: {success_summary}. "
            f"Suggest ONE specific task for the next cycle. "
            f"Format: 'Find [N] [role] remote gigs on [platform]. Extract title, URL, rate.' "
            f"Use remoteok.com or weworkremotely.com as platforms."
        )
        return self.consult(question, max_tokens=100)

"""StrategicAgent — adaptive task selection for NEXUS background cycles.

Maintains a portfolio of focused earning tasks. Tracks per-domain failure
counts and skips domains that are consistently failing. Rotates through
the portfolio so each cycle has a different angle.

State is persisted to data/strategy_state.json so restarts don't lose
domain failure history.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_STATE_FILE = Path("data/strategy_state.json")

# Imported lazily to avoid circular imports
def _get_claude_consultant():  # noqa: ANN202
    try:
        from nexus.core.claude_consultant import ClaudeConsultant
        return ClaudeConsultant()
    except Exception:
        return None

# (task_string, domain_tag, platform)
# domain_tag groups related tasks so failures are tracked together.
_PORTFOLIO: list[tuple[str, str, str]] = [
    (
        "Find 5 Python developer or data science remote gig listings on remoteok.com. "
        "Extract title, company, URL, and salary if shown.",
        "python",
        "remoteok",
    ),
    (
        "Find 5 JavaScript or React or Node.js remote developer job listings on remoteok.com. "
        "Extract title, company, URL, and salary if shown.",
        "javascript",
        "remoteok",
    ),
    (
        "Find 5 writing, copywriting, or content remote gig listings on remoteok.com. "
        "Extract title, company, URL, and rate if shown.",
        "writing",
        "remoteok",
    ),
    (
        "Find 5 data entry, virtual assistant, or customer support remote gig listings "
        "on remoteok.com. Extract title, company, URL, and rate if shown.",
        "support",
        "remoteok",
    ),
    (
        "Find 5 Python developer or backend engineer remote job listings on weworkremotely.com. "
        "Extract title, company, URL, and salary if shown.",
        "python",
        "weworkremotely",
    ),
    (
        "Find 5 design, UI/UX, or front-end remote job listings on remoteok.com. "
        "Extract title, company, URL, and salary if shown.",
        "design",
        "remoteok",
    ),
    (
        "Find 5 DevOps, cloud, or infrastructure remote job listings on remoteok.com. "
        "Extract title, company, URL, and salary if shown.",
        "devops",
        "remoteok",
    ),
    (
        "Find 5 marketing, SEO, or social media remote gig listings on remoteok.com. "
        "Extract title, company, URL, and rate if shown.",
        "marketing",
        "remoteok",
    ),
]


class StrategicAgent:
    """Selects and rotates task strategies for NEXUS background cycles.

    Args:
        state_file: Path to JSON file for persisting domain failure counts.
        max_domain_failures: Consecutive failures before a domain is skipped.
    """

    TASK_PORTFOLIO: list[tuple[str, str, str]] = _PORTFOLIO
    MAX_DOMAIN_FAILURES: int = 3

    def __init__(
        self,
        state_file: Path = _DEFAULT_STATE_FILE,
        max_domain_failures: int | None = None,
    ) -> None:
        self.state_file = state_file
        if max_domain_failures is not None:
            self.MAX_DOMAIN_FAILURES = max_domain_failures
        self._state: dict = self._load_state()

    # ──────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────

    def next_task(self) -> str:
        """Return the next task string to run.

        Skips domains with consecutive failures ≥ MAX_DOMAIN_FAILURES.
        If all domains are exhausted, picks the least-failed domain.
        """
        self._state = self._load_state()
        failures = self._state.get("domain_failures", {})
        index = self._state.get("task_index", 0)

        # Try to find a non-blocked task starting from current index
        n = len(self.TASK_PORTFOLIO)
        for offset in range(n):
            idx = (index + offset) % n
            task_str, domain, platform = self.TASK_PORTFOLIO[idx]
            if failures.get(domain, 0) < self.MAX_DOMAIN_FAILURES:
                self._state["task_index"] = (idx + 1) % n
                self._save_state()
                return task_str

        # All domains blocked — reset failures and pick least-failed
        logger.warning("StrategicAgent: all domains exhausted, resetting failures")
        min_failures = min(failures.get(d, 0) for _, d, _ in self.TASK_PORTFOLIO)
        for idx, (task_str, domain, _) in enumerate(self.TASK_PORTFOLIO):
            if failures.get(domain, 0) == min_failures:
                self._state["task_index"] = (idx + 1) % n
                self._save_state()
                return task_str

        # Absolute fallback — should never reach this
        return self.TASK_PORTFOLIO[0][0]

    def record_outcome(self, task: str, success: bool) -> None:
        """Record the result of a cycle for the given task string."""
        self._state = self._load_state()
        domain = self._task_domain(task)
        failures = self._state.setdefault("domain_failures", {})
        successes = self._state.setdefault("domain_successes", {})

        if success:
            failures[domain] = 0
            successes[domain] = successes.get(domain, 0) + 1
            logger.info("StrategicAgent: success on domain=%r — failures reset", domain)
        else:
            failures[domain] = failures.get(domain, 0) + 1
            logger.info(
                "StrategicAgent: failure on domain=%r — consecutive_failures=%d",
                domain, failures[domain],
            )
            # When a domain just hit the failure threshold, consult Claude
            if failures[domain] == self.MAX_DOMAIN_FAILURES:
                self._consult_claude_for_strategy(failures, successes)

        self._save_state()

    def _consult_claude_for_strategy(
        self,
        failures: dict[str, int],
        successes: dict[str, int],
    ) -> None:
        """Ask Claude for a new strategy when a domain is blocked. Logs suggestion only."""
        consultant = _get_claude_consultant()
        if consultant is None or not consultant.is_available():
            return
        recent_failures = [d for d, n in failures.items() if n >= self.MAX_DOMAIN_FAILURES]
        recent_successes = [d for d, n in successes.items() if n > 0]
        suggestion = consultant.suggest_strategy(
            recent_failures=recent_failures,
            recent_successes=recent_successes,
        )
        if suggestion:
            logger.info("StrategicAgent: Claude suggests — %s", suggestion)

    def _task_domain(self, task: str) -> str:
        """Return the domain tag for a task string, or 'unknown'."""
        for task_str, domain, _ in self.TASK_PORTFOLIO:
            if task_str == task:
                return domain
        # Fuzzy match: check if any domain keyword appears in task
        task_lower = task.lower()
        for _, domain, _ in self.TASK_PORTFOLIO:
            if domain in task_lower:
                return domain
        return "unknown"

    # ──────────────────────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────────────────────

    def _load_state(self) -> dict:
        if not self.state_file.exists():
            return {"task_index": 0, "domain_failures": {}}
        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("StrategicAgent: state file corrupt, resetting — %s", exc)
            return {"task_index": 0, "domain_failures": {}}

    def _save_state(self) -> None:
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(
                json.dumps(self._state, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("StrategicAgent: failed to save state — %s", exc)

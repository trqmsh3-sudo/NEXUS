from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import logging

from nexus.core import database as nexus_db

logger = logging.getLogger(__name__)


@dataclass
class BountySystem:
    """Tracks per-task bounties that control router tier usage."""

    bounties: Dict[str, float] = field(default_factory=dict)
    failures: Dict[str, int] = field(default_factory=dict)
    storage_path: str = "data/bounty_system.json"

    def __post_init__(self) -> None:
        self.load()

    # --------------- persistence ---------------
    def load(self) -> None:
        try:
            data = nexus_db.load_bounty_system(Path(self.storage_path))
            self.bounties = dict(data.get("bounties") or {})
            self.failures = dict(data.get("failures") or {})
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("BOUNTY load failed: %s", exc)

    def save(self) -> None:
        try:
            nexus_db.save_bounty_system(
                {"bounties": self.bounties, "failures": self.failures},
                Path(self.storage_path),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("BOUNTY save failed: %s", exc)

    # --------------- core API ---------------
    def get_bounty(self, task_key: str) -> float:
        return float(self.bounties.get(task_key, 0.001))

    def record_failure(self, task_key: str) -> float:
        count = self.failures.get(task_key, 0) + 1
        self.failures[task_key] = count

        if count >= 5:
            bounty = 0.10
        elif count >= 2:
            bounty = 0.01
        else:
            bounty = 0.001

        self.bounties[task_key] = bounty
        self.save()
        return bounty

    def record_success(self, task_key: str) -> None:
        """On success we reset to the default low bounty."""
        self.bounties[task_key] = 0.001
        # Optionally reset failures as well.
        self.failures[task_key] = 0
        self.save()


from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

import json
import logging
import re

from nexus.core.house_b import StructuredSpecificationObject

logger = logging.getLogger(__name__)


@dataclass
class AntiBeliefCertificate:
    task_description: str
    domain: str
    keywords: List[str]
    error_trace: str
    approach_tried: str
    failure_count: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_description": self.task_description,
            "domain": self.domain,
            "keywords": list(self.keywords),
            "error_trace": self.error_trace,
            "approach_tried": self.approach_tried,
            "failure_count": self.failure_count,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AntiBeliefCertificate":
        return cls(
            task_description=data["task_description"],
            domain=data.get("domain", "General"),
            keywords=list(data.get("keywords", [])),
            error_trace=data.get("error_trace", ""),
            approach_tried=data.get("approach_tried", ""),
            failure_count=int(data.get("failure_count", 1)),
            created_at=datetime.fromisoformat(data["created_at"]),
        )


class AntiBeliefGraph:
    """Stores repeated failure patterns and blocks similar future tasks."""

    def __init__(self, storage_path: str = "data/anti_beliefs.json") -> None:
        self._path = Path(storage_path)
        self._items: list[AntiBeliefCertificate] = []
        self._load()

    # ---------------- persistence -----------------
    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw or "[]")
            self._items = [AntiBeliefCertificate.from_dict(d) for d in data]
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ANTI-BELIEF load failed: %s", exc)

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = [a.to_dict() for a in self._items]
            self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ANTI-BELIEF save failed: %s", exc)

    # ---------------- public API ------------------
    def add_failure(self, cycle_result: Any) -> AntiBeliefCertificate:
        """Create or update an AntiBeliefCertificate from a failed cycle."""
        task = (cycle_result.user_input or "").strip()
        domain = getattr(cycle_result.sso, "domain", "General") or "General"
        error_trace = (cycle_result.failure_reason or "").strip()

        approach_tried = ""
        build_result = getattr(cycle_result, "build_result", None)
        if build_result and getattr(build_result, "artifact", None):
            approach_tried = (build_result.artifact.code or "")[:1000]

        existing = self._find(task, domain)
        if existing:
            existing.failure_count += 1
            if error_trace:
                existing.error_trace = (existing.error_trace or "") + "\n\n" + error_trace
            self._save()
            return existing

        keywords = list(self._extract_keywords(task))
        cert = AntiBeliefCertificate(
            task_description=task,
            domain=domain,
            keywords=keywords,
            error_trace=error_trace,
            approach_tried=approach_tried,
        )
        self._items.append(cert)
        self._save()
        return cert

    def is_blocked(self, sso: StructuredSpecificationObject) -> bool:
        """Return True if a similar failure exists in the same domain."""
        desc = (sso.redefined_problem or sso.original_input or "").strip()
        domain = sso.domain or "General"
        candidate = self._extract_keywords(desc)

        for anti in self._items:
            if anti.domain.lower() != domain.lower():
                continue
            overlap = len(candidate & set(anti.keywords))
            union = len(candidate | set(anti.keywords))
            if union == 0:
                continue
            jaccard = overlap / union
            if jaccard >= 0.4:
                logger.warning(
                    "BLOCKED BY ANTI-BELIEF  domain=%s  task=%r  matched=%r",
                    domain,
                    desc[:80],
                    anti.task_description[:80],
                )
                return True
        return False

    # ---------------- helpers ------------------
    @staticmethod
    def _extract_keywords(text: str) -> set[str]:
        tokens = re.findall(r"[a-zA-Z_]+", text.lower())
        stop = {"the", "a", "an", "and", "or", "for", "with", "that", "this", "from"}
        return {t for t in tokens if t not in stop and len(t) > 2}

    def _find(self, task: str, domain: str) -> AntiBeliefCertificate | None:
        for a in self._items:
            if a.task_description == task and a.domain == domain:
                return a
        return None


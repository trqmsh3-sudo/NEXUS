"""Counterfactual logging — rejected alternatives and prediction validation."""

from __future__ import annotations

import json
import logging
import os
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class CounterfactualEntry:
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    cycle_id: str = ""
    house: str = ""
    chosen_action: str = ""
    rejected_candidates: list[dict[str, Any]] = field(default_factory=list)
    actual_outcome: str = ""
    was_prediction_correct: bool | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    background_checked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "cycle_id": self.cycle_id,
            "house": self.house,
            "chosen_action": self.chosen_action,
            "rejected_candidates": list(self.rejected_candidates),
            "actual_outcome": self.actual_outcome,
            "was_prediction_correct": self.was_prediction_correct,
            "created_at": self.created_at.isoformat(),
            "background_checked": self.background_checked,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CounterfactualEntry:
        return cls(
            entry_id=data.get("entry_id", str(uuid.uuid4())),
            cycle_id=data.get("cycle_id", ""),
            house=data.get("house", ""),
            chosen_action=data.get("chosen_action", ""),
            rejected_candidates=list(data.get("rejected_candidates", [])),
            actual_outcome=data.get("actual_outcome", ""),
            was_prediction_correct=data.get("was_prediction_correct"),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.now(timezone.utc),
            background_checked=bool(data.get("background_checked", False)),
        )


class CounterfactualLog:
    def __init__(self, storage_path: str = "data/counterfactual_log.json") -> None:
        self.entries: list[CounterfactualEntry] = []
        self.storage_path: str = storage_path
        self.wrong_predictions: int = 0
        self._background_seen: set[str] = set()
        self.load()

    def pick_rejected_pairs(
        self, max_pairs: int = 2,
    ) -> list[tuple[CounterfactualEntry, dict[str, Any], int]]:
        """Up to max_pairs (entry, rejected_candidate, index) not yet background-checked."""
        pool: list[tuple[CounterfactualEntry, dict[str, Any], int]] = []
        for e in self.entries:
            if e.house != "house_b":
                continue
            for i, rc in enumerate(e.rejected_candidates):
                if not isinstance(rc, dict):
                    continue
                key = f"{e.entry_id}:{i}"
                if key in self._background_seen:
                    continue
                pool.append((e, rc, i))
        random.shuffle(pool)
        return pool[:max_pairs]

    def mark_background_seen(self, entry: CounterfactualEntry, rc_index: int) -> None:
        self._background_seen.add(f"{entry.entry_id}:{rc_index}")

    def add_entry(self, entry: CounterfactualEntry) -> None:
        self.entries.append(entry)
        self.save()

    def get_entries_for_cycle(self, cycle_id: str) -> list[CounterfactualEntry]:
        return [e for e in self.entries if e.cycle_id == cycle_id]

    def validate_predictions(self, cycle_id: str, actual_outcome: str) -> int:
        """Mark was_prediction_correct for entries in this cycle. Returns count updated."""
        updated = 0
        for e in self.entries:
            if e.cycle_id != cycle_id:
                continue
            if e.was_prediction_correct is not None:
                continue
            e.actual_outcome = actual_outcome[:2000]
            # Chosen path was "correct" if full cycle succeeded (belief added).
            success = "SUCCESS" in actual_outcome.upper() or "belief added" in actual_outcome.lower()
            e.was_prediction_correct = success
            updated += 1
        if updated:
            self.save()
        return updated

    def save(self) -> None:
        try:
            path = Path(self.storage_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "wrong_predictions": self.wrong_predictions,
                "background_seen": sorted(self._background_seen),
                "entries": [e.to_dict() for e in self.entries],
            }
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        except OSError as exc:
            logger.warning("CounterfactualLog save failed: %s", exc)

    def load(self) -> None:
        try:
            path = Path(self.storage_path)
            if not path.exists():
                return
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw or "{}")
            if isinstance(data, list):
                self.entries = [CounterfactualEntry.from_dict(x) for x in data if isinstance(x, dict)]
                self.wrong_predictions = 0
                return
            self.wrong_predictions = int(data.get("wrong_predictions", 0))
            self._background_seen = set(data.get("background_seen", []))
            for item in data.get("entries", []):
                if isinstance(item, dict):
                    self.entries.append(CounterfactualEntry.from_dict(item))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("CounterfactualLog load failed: %s", exc)


def groq_counterfactual_alternatives(user_input: str, chosen_redefinition: str) -> list[dict[str, str]]:
    """Ask Groq for 2 rejected alternative redefinitions. Returns list of dicts."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return []
    try:
        import litellm
        litellm.suppress_debug_info = True
        user = (
            f"Human input:\n{user_input[:1500]}\n\n"
            f"Chosen redefinition:\n{chosen_redefinition[:1500]}\n\n"
            "What are 2 alternative ways you considered redefining this problem but rejected? "
            "Return ONLY a JSON array of 2 objects, each with keys: "
            "action, predicted_outcome, rejection_reason (all strings)."
        )
        r = litellm.completion(
            model="groq/llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON array. No markdown."},
                {"role": "user", "content": user},
            ],
            api_key=key,
            temperature=0.3,
            max_tokens=600,
        )
        text = (r.choices[0].message.content or "").strip()
        start = text.find("[")
        end = text.rfind("]") + 1
        if start < 0 or end <= start:
            return []
        arr = json.loads(text[start:end])
        if not isinstance(arr, list):
            return []
        out = []
        for x in arr[:2]:
            if not isinstance(x, dict):
                continue
            out.append({
                "action": str(x.get("action", ""))[:800],
                "predicted_outcome": str(x.get("predicted_outcome", ""))[:500],
                "rejection_reason": str(x.get("rejection_reason", ""))[:500],
            })
        return out
    except Exception as exc:
        logger.debug("groq_counterfactual_alternatives: %s", exc)
        return []


def groq_validate_rejected_prediction(
    actual_outcome: str,
    rejected_action: str,
    predicted_outcome: str,
) -> bool | None:
    """Returns True if the model thinks the rejection prediction was accurate, False if wrong, None on error."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    try:
        import litellm
        litellm.suppress_debug_info = True
        user = (
            f"Actual cycle outcome: {actual_outcome[:800]}\n\n"
            f"Rejected alternative framing: {rejected_action[:600]}\n"
            f"If that alternative had been chosen, the model predicted: {predicted_outcome[:400]}\n\n"
            "Was that prediction about the alternative likely accurate given what we know? "
            'Reply ONLY JSON: {{"accurate": true or false}}'
        )
        r = litellm.completion(
            model="groq/llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Return ONLY JSON with key accurate (boolean)."},
                {"role": "user", "content": user},
            ],
            api_key=key,
            temperature=0.1,
            max_tokens=80,
        )
        text = (r.choices[0].message.content or "").strip()
        if "{" in text:
            text = text[text.find("{") : text.rfind("}") + 1]
        obj = json.loads(text)
        return bool(obj.get("accurate"))
    except Exception as exc:
        logger.debug("groq_validate_rejected_prediction: %s", exc)
        return None

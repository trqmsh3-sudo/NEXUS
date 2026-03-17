"""Model Router — routes LLM calls via Mistral, Groq, and OpenRouter.

Uses LiteLLM. Primary: Mistral (reliable, no encoding issues).
Groq and OpenRouter free tier as fallback.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import litellm

from nexus.core.text_utils import clean_text

logger: logging.Logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True

# ------------------------------------------------------------------
# Model tiers — cost-aware routing
# ------------------------------------------------------------------

TIER_1_FREE: list[str] = [
    # Cheap but reliable default for all Houses
    "openrouter/google/gemini-2.5-flash",
    # Verified working free models
    "openrouter/mistralai/mistral-small-3.1-24b-instruct:free",
    "openrouter/liquid/lfm-7b:free",
]

TIER_2_CHEAP: list[str] = [
    "openrouter/google/gemini-2.5-flash",
    "openrouter/deepseek/deepseek-chat",
]

TIER_3_POWERFUL: list[str] = [
    "openrouter/anthropic/claude-sonnet-4-5",
    "openrouter/google/gemini-2.5-pro",
]

MAX_DAILY_COST: float = 0.30  # USD
_COST_FILE: pathlib.Path = pathlib.Path("data/daily_cost.json")
_BLACKLIST_FILE: pathlib.Path = pathlib.Path("data/model_blacklist.json")

OPENROUTER_HEADERS: dict[str, str] = {
    "HTTP-Referer": "https://github.com/trqmsh3-sudo/NEXUS",
    "X-Title": "NEXUS",
}

_MIN_RESPONSE_CHARS: int = 10


@dataclass
class ModelRouter:
    """Routes LLM calls through LiteLLM with cost-aware model selection.

    Houses are routed through three tiers:

    * Tier 1 — free OpenRouter models.
    * Tier 2 — cheap, fast models.
    * Tier 3 — most powerful models (used only when cheaper tiers fail).

    A simple daily budget is enforced:

    * MAX_DAILY_COST caps spend at ~30 cents/day.
    * Costs are tracked in ``data/daily_cost.json``.
    * When the budget is exceeded, only Tier 1 models are used.

    Attributes:
        call_log: History of (house, model_used, elapsed_s, ok) tuples.
    """

    call_log: list[tuple[str, str, float, bool]] = field(default_factory=list)
    _failure_counts: dict[str, int] = field(default_factory=dict, repr=False)
    _blacklist: set[str] = field(default_factory=set, repr=False)

    def complete(
        self,
        *,
        house: str,
        system: str,
        user: str,
        label: str = "",
        max_tokens: int = 2000,
    ) -> str:
        """Send a chat completion through the best model for *house*.

        Tries multiple models if the preferred one fails or returns empty.

        Args:
            house: House identifier (``"house_b"``, ``"house_c"``, etc.).
            system: The system prompt.
            user: The user prompt.
            label: Human-readable label for logging.
            max_tokens: Maximum tokens in the response.

        Returns:
            The raw text content from the LLM response.

        Raises:
            ValueError: If all models failed or returned empty.
        """
        today = self._today_str()
        cost_date, total_cost = self._load_daily_cost()
        bl_date, blacklist = self._load_blacklist()
        if cost_date != today:
            cost_date, total_cost = today, 0.0
        if bl_date != today:
            # Reset blacklist daily
            bl_date, blacklist = today, set()
        # Cache blacklist in-memory for this session
        self._blacklist = set(blacklist)

        over_budget = total_cost >= MAX_DAILY_COST

        # Build tiered list per house, respecting budget.
        if over_budget:
            models_to_try: list[str] = list(TIER_1_FREE)
        else:
            if house in {"house_b", "house_c", "house_d"}:
                models_to_try = list(TIER_1_FREE) + list(TIER_2_CHEAP)
            else:
                # Fallback for any future houses
                models_to_try = list(TIER_2_CHEAP) + list(TIER_1_FREE)
            # Tier 3 is only attempted when cheaper tiers fail.
            models_to_try += list(TIER_3_POWERFUL)

        # De-duplicate while preserving order.
        seen: set[str] = set()
        models_to_try = [
            m for m in models_to_try
            if m and m not in seen and not seen.add(m)
            and m not in self._blacklist
        ]

        messages: list[dict[str, str]] = [
            {"role": "system", "content": clean_text(system)},
            {"role": "user", "content": clean_text(user)},
        ]

        last_error: Exception | None = None
        for model in models_to_try:
            result = self._try_model(model, messages, max_tokens, house, label)
            if result is None:
                # Increment failure counter and possibly blacklist.
                count = self._failure_counts.get(model, 0) + 1
                self._failure_counts[model] = count
                if count >= 2 and model not in self._blacklist:
                    self._blacklist.add(model)
                    self._save_blacklist(today, self._blacklist)
                    logger.warning(
                        "ROUTER blacklist  model=%s  failures=%d", model, count,
                    )
                continue
            text = result.strip()
            if len(text) < _MIN_RESPONSE_CHARS:
                logger.warning(
                    "ROUTER empty  house=%s  model=%s  label=%s  chars=%d  trying next",
                    house, model, label, len(text),
                )
                continue

            # Successful, non-empty response — record cost and return.
            call_cost = self._estimate_cost(model)
            total_cost += call_cost
            self._save_daily_cost(today, total_cost)
            logger.info(
                "ROUTER ok  house=%s  model=%s  label=%s  chars=%d  "
                "cost=%.5f  daily_total=%.5f",
                house, model, label, len(text), call_cost, total_cost,
            )
            logger.info(
                "ROUTER budget  remaining=%.5f  max=%.2f  over_budget=%s",
                max(MAX_DAILY_COST - total_cost, 0.0),
                MAX_DAILY_COST,
                total_cost >= MAX_DAILY_COST,
            )
            return text

        raise ValueError(
            f"{label}: all models failed or returned empty"
        )

    def _try_model(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        house: str,
        label: str,
    ) -> str | None:
        """Attempt a single model call. Returns None on failure or empty."""
        start = time.perf_counter()
        is_openrouter = model.startswith("openrouter/")
        kw: dict[str, Any] = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
        )
        if is_openrouter:
            kw["api_base"] = "https://openrouter.ai/api/v1"
            kw["api_key"] = os.getenv("OPENROUTER_API_KEY")
            kw["extra_headers"] = OPENROUTER_HEADERS
        try:
            logger.info("ROUTER call  house=%s  model=%s  label=%s", house, model, label)
            response = litellm.completion(**kw)
            content: str = response.choices[0].message.content or ""
            elapsed = time.perf_counter() - start
            self.call_log.append((house, model, round(elapsed, 3), bool(content.strip())))
            return content
        except Exception as exc:
            elapsed = time.perf_counter() - start
            self.call_log.append((house, model, round(elapsed, 3), False))
            logger.warning(
                "ROUTER failed  house=%s  model=%s  label=%s  error=%s",
                house, model, label, exc,
            )
            return None

    # ------------------------------------------------------------------
    # Cost helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _today_str() -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def _load_daily_cost(self) -> tuple[str, float]:
        """Load the current daily cost tracking file."""
        try:
            if _COST_FILE.exists():
                raw = _COST_FILE.read_text(encoding="utf-8")
                data = json.loads(raw or "{}")
                date = str(data.get("date") or "")
                total = float(data.get("total_cost") or 0.0)
                return date, total
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ROUTER cost load failed: %s", exc)
        return self._today_str(), 0.0

    def _save_daily_cost(self, date: str, total_cost: float) -> None:
        """Persist the daily cost tracker to disk."""
        try:
            _COST_FILE.parent.mkdir(parents=True, exist_ok=True)
            payload = {"date": date, "total_cost": round(total_cost, 6)}
            _COST_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ROUTER cost save failed: %s", exc)

    def _load_blacklist(self) -> tuple[str, set[str]]:
        """Load the per-day model blacklist."""
        try:
            if _BLACKLIST_FILE.exists():
                raw = _BLACKLIST_FILE.read_text(encoding="utf-8")
                data = json.loads(raw or "{}")
                date = str(data.get("date") or "")
                models = set(data.get("models") or [])
                return date, models
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ROUTER blacklist load failed: %s", exc)
        return self._today_str(), set()

    def _save_blacklist(self, date: str, models: set[str]) -> None:
        """Persist the blacklist to disk."""
        try:
            _BLACKLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
            payload = {"date": date, "models": sorted(models)}
            _BLACKLIST_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ROUTER blacklist save failed: %s", exc)

    @staticmethod
    def _estimate_cost(model: str) -> float:
        """Rough per-call cost estimate in USD based on tier.

        This is deliberately conservative and does not attempt to model
        exact token usage. It is only used for coarse daily budgeting.
        """
        if model in TIER_1_FREE or model.endswith(":free"):
            return 0.0
        if model in TIER_2_CHEAP:
            return 0.002  # ~0.2 cents per call
        if model in TIER_3_POWERFUL:
            return 0.02   # ~2 cents per call
        # Unknown model: assume cheap-ish.
        return 0.002

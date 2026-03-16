"""Model Router — routes LLM calls to the best provider per House.

Uses LiteLLM as the universal interface so any supported provider
can be swapped in without changing calling code. Each House is
assigned a preferred model. If the preferred model fails, the
router falls back to a reliable default.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import litellm

logger: logging.Logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True

HOUSE_MODELS: dict[str, str] = {
    "house_b": "gemini/gemini-2.0-flash",
    "house_c": "mistral/mistral-large-latest",
    "house_d": "mistral/mistral-large-latest",
}

FALLBACK_MODEL: str = "mistral/mistral-large-latest"


@dataclass
class ModelRouter:
    """Routes LLM calls through LiteLLM with per-House model selection.

    Each House is assigned a preferred model via :data:`HOUSE_MODELS`.
    If the preferred model raises an exception the router automatically
    retries with :data:`FALLBACK_MODEL`.

    Attributes:
        call_log: History of (house, model_used, elapsed_s, ok) tuples.
    """

    call_log: list[tuple[str, str, float, bool]] = field(default_factory=list)

    def complete(
        self,
        *,
        house: str,
        system: str,
        user: str,
        label: str = "",
        max_tokens: int = 4096,
    ) -> str:
        """Send a chat completion through the best model for *house*.

        Args:
            house: House identifier (``"house_b"``, ``"house_c"``, etc.).
            system: The system prompt.
            user: The user prompt.
            label: Human-readable label for logging.
            max_tokens: Maximum tokens in the response.

        Returns:
            The raw text content from the LLM response.

        Raises:
            Exception: If both the preferred model and the fallback fail.
        """
        preferred = HOUSE_MODELS.get(house, FALLBACK_MODEL)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        content = self._try_model(preferred, messages, max_tokens, house, label)
        if content is not None:
            return content

        if preferred != FALLBACK_MODEL:
            logger.warning(
                "ROUTER fallback  house=%s  from=%s  to=%s  label=%s",
                house, preferred, FALLBACK_MODEL, label,
            )
            content = self._try_model(FALLBACK_MODEL, messages, max_tokens, house, label)
            if content is not None:
                return content

        raise RuntimeError(
            f"All models failed for {house} [{label}]: "
            f"tried {preferred} and {FALLBACK_MODEL}"
        )

    def _try_model(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        house: str,
        label: str,
    ) -> str | None:
        """Attempt a single model call, returning None on failure."""
        start = time.perf_counter()
        try:
            logger.info(
                "ROUTER call  house=%s  model=%s  label=%s",
                house, model, label,
            )
            response = litellm.completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )
            content: str = response.choices[0].message.content or ""
            elapsed = time.perf_counter() - start
            self.call_log.append((house, model, round(elapsed, 3), True))
            logger.info(
                "ROUTER ok  house=%s  model=%s  chars=%d  elapsed=%.2fs",
                house, model, len(content), elapsed,
            )
            return content
        except Exception as exc:
            elapsed = time.perf_counter() - start
            self.call_log.append((house, model, round(elapsed, 3), False))
            logger.warning(
                "ROUTER failed  house=%s  model=%s  error=%s  elapsed=%.2fs",
                house, model, exc, elapsed,
            )
            return None

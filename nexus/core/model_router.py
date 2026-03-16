"""Model Router — routes LLM calls via Mistral, Groq, and OpenRouter.

Uses LiteLLM. Primary: Mistral (reliable, no encoding issues).
Groq and OpenRouter free tier as fallback.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import litellm

from nexus.core.text_utils import clean_text

logger: logging.Logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True

# Primary model — Mistral (proven to work, no encoding issues)
PRIMARY_MODELS: list[str] = [
    "mistral/mistral-large-latest",
]

# Per-house model assignments (Mistral only for reliability)
HOUSE_MODELS: dict[str, str] = {
    "house_b": "mistral/mistral-large-latest",
    "house_c": "mistral/mistral-large-latest",
    "house_d": "mistral/mistral-large-latest",
}

FALLBACK_MODEL: str = "mistral/mistral-large-latest"

# Fallback models — Groq and OpenRouter (last resort)
WORKING_FREE_MODELS: list[str] = [
    "groq/llama-3.3-70b-versatile",
    "openrouter/liquid/lfm-2.5-1.2b-instruct:free",
    "openrouter/liquid/lfm-2.5-1.2b-thinking:free",
    "openrouter/nvidia/nemotron-nano-9b-v2:free",
    "openrouter/meta-llama/llama-3.2-3b-instruct:free",
    "openrouter/meta-llama/llama-3.3-70b-instruct:free",
    "openrouter/qwen/qwen3-coder:free",
]

OPENROUTER_HEADERS: dict[str, str] = {
    "HTTP-Referer": "https://github.com/trqmsh3-sudo/NEXUS",
    "X-Title": "NEXUS",
}

_MIN_RESPONSE_CHARS: int = 10


@dataclass
class ModelRouter:
    """Routes LLM calls through LiteLLM with per-House model selection.

    If the preferred model fails or returns empty, the router tries
    alternative models. Free models sometimes return empty strings.

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
        preferred = HOUSE_MODELS.get(house, FALLBACK_MODEL)
        # Try primary (Groq/Mistral) first, then OpenRouter as last resort
        primary = [preferred] + [m for m in PRIMARY_MODELS if m != preferred]
        if FALLBACK_MODEL not in primary:
            primary.append(FALLBACK_MODEL)
        models_to_try: list[str] = primary + WORKING_FREE_MODELS
        seen: set[str] = set()
        models_to_try = [m for m in models_to_try if m and m not in seen and not seen.add(m)]

        messages: list[dict[str, str]] = [
            {"role": "system", "content": clean_text(system)},
            {"role": "user", "content": clean_text(user)},
        ]

        last_error: Exception | None = None
        for model in models_to_try:
            result = self._try_model(model, messages, max_tokens, house, label)
            if result is None:
                continue
            text = result.strip()
            if len(text) < _MIN_RESPONSE_CHARS:
                logger.warning(
                    "ROUTER empty  house=%s  model=%s  label=%s  chars=%d  trying next",
                    house, model, label, len(text),
                )
                continue
            logger.info(
                "ROUTER ok  house=%s  model=%s  label=%s  chars=%d",
                house, model, label, len(text),
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

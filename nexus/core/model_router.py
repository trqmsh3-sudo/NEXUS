"""Model Router — routes LLM calls via DeepSeek, Gemini, Groq, and OpenRouter.

Uses LiteLLM. Primary: DeepSeek V3 (deepseek-chat).
Gemini/Groq/OpenRouter as fallback tiers.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import litellm

from nexus.core import database as nexus_db
from nexus.core.text_utils import clean_text

# GuardianVault import is deferred to avoid a circular import at module level.
# Type annotations use a string literal; isinstance checks use the lazy import.
_GuardianVault: type | None = None

def _get_guardian_vault_class() -> type:
    global _GuardianVault
    if _GuardianVault is None:
        from nexus.core.guardian import GuardianVault  # noqa: PLC0415
        _GuardianVault = GuardianVault
    return _GuardianVault

logger: logging.Logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True

# OpenRouter API key (optional at import; Tier 2 models skipped if missing).
_openrouter_key = os.getenv("OPENROUTER_API_KEY")
if _openrouter_key:
    os.environ["OPENROUTER_API_KEY"] = _openrouter_key
    logger.info("OpenRouter key loaded: %s...", _openrouter_key[:8])
else:
    logger.warning(
        "OPENROUTER_API_KEY not set — OpenRouter models will be skipped.",
    )

# Gemini direct API key (used by litellm's Gemini backend).
_gemini_key = os.getenv("GEMINI_API_KEY")
if _gemini_key:
    os.environ["GEMINI_API_KEY"] = _gemini_key
    logger.info("Gemini key loaded: %s...", _gemini_key[:8])
else:
    logger.warning("GEMINI_API_KEY not set — Tier 0 Gemini models will be skipped.")

# Groq API key (for direct Groq models via LiteLLM).
_groq_key = os.getenv("GROQ_API_KEY")
if _groq_key:
    os.environ["GROQ_API_KEY"] = _groq_key
    logger.info("Groq key loaded: %s...", _groq_key[:8])
else:
    logger.warning("GROQ_API_KEY not set — Groq models may not be usable.")

# DeepSeek API key — primary LLM provider.
_deepseek_key = os.getenv("DEEPSEEK_API_KEY")
if _deepseek_key:
    os.environ["DEEPSEEK_API_KEY"] = _deepseek_key
    logger.info("DeepSeek key loaded: %s...", _deepseek_key[:8])
else:
    logger.info(
        "DEEPSEEK_API_KEY not in environment — "
        "will use vault if configured, else DeepSeek models skipped."
    )

# ------------------------------------------------------------------
# Model tiers — cost-aware routing
# ------------------------------------------------------------------

# Tier Primary — DeepSeek V3 direct (used first when key is available).
TIER_PRIMARY_DEEPSEEK: list[str] = [
    "deepseek/deepseek-chat",  # DeepSeek V3
]

# Tier 0 — direct Gemini free-tier models (used first when available).
TIER_0_GEMINI_FREE: list[str] = [
    "gemini/gemini-2.0-flash",      # 15 RPM free
    "gemini/gemini-1.5-flash",      # 15 RPM free
    "gemini/gemini-2.5-pro",        # low free RPM, use sparingly
]

# Tier 1 — free / very cheap defaults (OpenRouter + direct where appropriate).
TIER_1_FREE: list[str] = [
    "gemini/gemini-2.0-flash",      # direct Gemini
    "groq/llama-3.3-70b-versatile", # Groq - fast + strong
    "groq/llama-3.1-8b-instant",    # Groq - fastest
    "openrouter/google/gemini-2.5-flash",
    "openrouter/mistralai/mistral-small-3.1-24b-instruct:free",
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
_GEMINI_RL_FILE: pathlib.Path = pathlib.Path("data/gemini_rate_limits.json")

# Entries expire automatically so transient API errors do not block all day.
_BLACKLIST_TTL_SECONDS: int = int(
    os.getenv("NEXUS_MODEL_BLACKLIST_TTL_SECONDS", "3600"),
)

_GROQ_RL_FILE: pathlib.Path = pathlib.Path("data/groq_rate_limits.json")

# Per-minute RPM limits for Groq free-tier models.
_GROQ_RATE_LIMITS: dict[str, int] = {
    "groq/llama-3.3-70b-versatile": 30,
    "groq/llama-3.1-8b-instant": 30,
}

OPENROUTER_HEADERS: dict[str, str] = {
    "HTTP-Referer": "https://github.com/trqmsh3-sudo/NEXUS",
    "X-Title": "NEXUS",
}

_MIN_RESPONSE_CHARS: int = 10

# Rough per-call costs used for budgeting and "cheapest available" selection.
MODEL_COSTS: dict[str, float] = {
    "deepseek/deepseek-chat": 0.001,  # ~$0.27/M input, $1.10/M output
    "gemini/gemini-2.0-flash": 0.0,
    "gemini/gemini-1.5-flash": 0.0,
    "gemini/gemini-2.5-pro": 0.0,  # treat as free for now (RPM-limited)
    "groq/llama-3.3-70b-versatile": 0.0,
    "groq/llama-3.1-8b-instant": 0.0,
    "openrouter/google/gemini-2.5-flash": 0.001,
    "openrouter/deepseek/deepseek-chat": 0.002,
    "openrouter/anthropic/claude-sonnet-4-5": 0.05,
}


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
    vault: Any = field(default=None, repr=False)  # GuardianVault | None

    def complete(
        self,
        *,
        house: str,
        system: str,
        user: str,
        label: str = "",
        max_tokens: int = 2000,
        bounty_amount: float | None = None,
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
        if cost_date != today:
            cost_date, total_cost = today, 0.0
        # TTL blacklist: prune expired entries on disk, cache active set
        bl_entries = self._read_and_prune_blacklist_entries()
        self._blacklist = set(bl_entries.keys())

        over_budget = total_cost >= MAX_DAILY_COST

        # Build tiered list per house, respecting budget and bounty.
        # DeepSeek is always first when the key is available; remaining
        # tiers are Gemini/Groq/OpenRouter as fallback.
        base_models: list[str] = list(TIER_PRIMARY_DEEPSEEK) + list(TIER_0_GEMINI_FREE)

        if over_budget:
            base_models += list(TIER_1_FREE)
        else:
            if bounty_amount is not None:
                if bounty_amount < 0.005:
                    base_models += list(TIER_1_FREE)
                elif bounty_amount < 0.05:
                    base_models += list(TIER_1_FREE) + list(TIER_2_CHEAP)
                else:
                    base_models += list(TIER_1_FREE) + list(TIER_2_CHEAP) + list(TIER_3_POWERFUL)
            else:
                if house in {"house_b", "house_c", "house_d"}:
                    base_models += list(TIER_1_FREE) + list(TIER_2_CHEAP)
                else:
                    base_models += list(TIER_2_CHEAP) + list(TIER_1_FREE)
                base_models += list(TIER_3_POWERFUL)

        models_to_try: list[str] = base_models

        # De-duplicate while preserving order.
        seen: set[str] = set()
        models_to_try = [
            m for m in models_to_try
            if m and m not in seen and not seen.add(m)
            and m not in self._blacklist
        ]
        if not self._get_deepseek_key():
            models_to_try = [m for m in models_to_try if not m.startswith("deepseek/")]
        if not _openrouter_key:
            models_to_try = [
                m for m in models_to_try if not m.startswith("openrouter/")
            ]
        if not _groq_key:
            models_to_try = [m for m in models_to_try if not m.startswith("groq/")]
        if not _gemini_key:
            models_to_try = [m for m in models_to_try if not m.startswith("gemini/")]

        messages: list[dict[str, str]] = [
            {"role": "system", "content": clean_text(system)},
            {"role": "user", "content": clean_text(user)},
        ]

        last_error: Exception | None = None
        for model in models_to_try:
            # Respect Gemini per-minute rate limits before calling.
            if model.startswith("gemini/") and not self._can_use_gemini(model):
                logger.info(
                    "ROUTER skip  house=%s  model=%s  label=%s  reason=gemini_rate_limit",
                    house, model, label,
                )
                continue

            # Respect Groq per-minute rate limits before calling.
            if model.startswith("groq/") and not self._can_use_groq(model):
                logger.info(
                    "ROUTER skip  house=%s  model=%s  label=%s  reason=groq_rate_limit",
                    house, model, label,
                )
                continue

            result = self._try_model(model, messages, max_tokens, house, label)
            if result is None:
                # Increment failure counter and possibly blacklist.
                count = self._failure_counts.get(model, 0) + 1
                self._failure_counts[model] = count
                if count >= 2 and model not in self._blacklist:
                    self._blacklist_model(model)
                    logger.warning(
                        "ROUTER blacklist  model=%s  failures=%d  ttl_s=%d",
                        model,
                        count,
                        _BLACKLIST_TTL_SECONDS,
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

    def _get_deepseek_key(self) -> str | None:
        """Return the DeepSeek API key from vault (preferred) or env fallback."""
        if self.vault is not None and self.vault.has("DEEPSEEK_API_KEY"):
            return self.vault.get("DEEPSEEK_API_KEY")
        return os.getenv("DEEPSEEK_API_KEY")

    def _get_gemini_key(self) -> str | None:
        """Return the Gemini API key from vault (preferred) or env fallback."""
        if self.vault is not None and self.vault.has("GEMINI_API_KEY"):
            return self.vault.get("GEMINI_API_KEY")
        return os.getenv("GEMINI_API_KEY") or None

    def _get_groq_key(self) -> str | None:
        """Return the Groq API key from vault (preferred) or env fallback."""
        if self.vault is not None and self.vault.has("GROQ_API_KEY"):
            return self.vault.get("GROQ_API_KEY")
        return os.getenv("GROQ_API_KEY") or None

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
        is_deepseek = model.startswith("deepseek/")
        is_openrouter = model.startswith("openrouter/")
        is_gemini = model.startswith("gemini/")
        is_groq = model.startswith("groq/")
        kw: dict[str, Any] = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
        )
        if is_deepseek:
            ds_key = self._get_deepseek_key()
            if not ds_key:
                logger.info("ROUTER skip DeepSeek call — no API key")
                return None
            kw["api_base"] = "https://api.deepseek.com"
            kw["api_key"] = ds_key
        elif is_openrouter:
            or_key = os.getenv("OPENROUTER_API_KEY")
            if not or_key:
                logger.info("ROUTER skip OpenRouter call — no API key")
                return None
            kw["api_base"] = "https://openrouter.ai/api/v1"
            kw["api_key"] = or_key
            kw["extra_headers"] = OPENROUTER_HEADERS
        elif is_gemini:
            gem_key = self._get_gemini_key()
            if not gem_key:
                logger.info("ROUTER skip Gemini call — no API key")
                return None
            kw["api_key"] = gem_key
        elif is_groq:
            groq_key = self._get_groq_key()
            if not groq_key:
                logger.info("ROUTER skip Groq call — no API key")
                return None
            kw["api_key"] = groq_key
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
        """Load daily cost from Supabase (if configured) else file."""
        try:
            d = nexus_db.load_daily_cost(_COST_FILE)
            date = str(d.get("date") or "")
            total = float(d.get("total_cost") or 0.0)
            return date, total
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ROUTER cost load failed: %s", exc)
        return self._today_str(), 0.0

    def _save_daily_cost(self, date: str, total_cost: float) -> None:
        """Persist daily cost to Supabase (if configured) else file."""
        try:
            nexus_db.save_daily_cost(
                {"date": date, "total_cost": round(total_cost, 6)},
                _COST_FILE,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ROUTER cost save failed: %s", exc)

    def _read_and_prune_blacklist_entries(self) -> dict[str, datetime]:
        """Load blacklist entries and drop those past ``expires_at`` (UTC)."""
        now = datetime.now(timezone.utc)
        if not _BLACKLIST_FILE.exists():
            return {}
        try:
            raw_text = _BLACKLIST_FILE.read_text(encoding="utf-8")
            data = json.loads(raw_text or "{}")
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("ROUTER blacklist read failed: %s", exc)
            return {}

        if isinstance(data.get("models"), list):
            logger.info(
                "ROUTER blacklist: retiring legacy list (%d models) → TTL entries",
                len(data["models"]),
            )
            self._write_blacklist_entries({})
            return {}

        raw_entries: dict[str, str] = {
            str(k): str(v)
            for k, v in (data.get("entries") or {}).items()
            if k and v
        }
        entries: dict[str, datetime] = {}
        for model, iso in raw_entries.items():
            try:
                exp = datetime.fromisoformat(iso.replace("Z", "+00:00"))
                if exp.tzinfo is None:
                    exp = exp.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if exp > now:
                entries[model] = exp

        if len(entries) != len(raw_entries):
            self._write_blacklist_entries(entries)
        return entries

    def _write_blacklist_entries(self, entries: dict[str, datetime]) -> None:
        """Persist TTL blacklist (version 2)."""
        try:
            _BLACKLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
            payload: dict[str, Any] = {
                "version": 2,
                "entries": {
                    m: exp.isoformat()
                    for m, exp in sorted(
                        entries.items(),
                        key=lambda kv: kv[0],
                    )
                },
            }
            _BLACKLIST_FILE.write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning("ROUTER blacklist write failed: %s", exc)

    def _blacklist_model(self, model: str) -> None:
        """Block *model* until TTL elapses (default 1 hour)."""
        until = datetime.now(timezone.utc) + timedelta(
            seconds=_BLACKLIST_TTL_SECONDS,
        )
        entries = self._read_and_prune_blacklist_entries()
        entries[model] = until
        self._write_blacklist_entries(entries)
        self._blacklist.add(model)

    # ------------------------------------------------------------------
    # Gemini rate limiting helpers
    # ------------------------------------------------------------------

    def _load_gemini_window(self) -> tuple[datetime, dict[str, int]]:
        """Load current Gemini rate-limit window and per-model counts."""
        now = datetime.now(timezone.utc)
        try:
            if _GEMINI_RL_FILE.exists():
                raw = _GEMINI_RL_FILE.read_text(encoding="utf-8")
                data = json.loads(raw or "{}")
                window_start_raw = data.get("window_start")
                if window_start_raw:
                    window_start = datetime.fromisoformat(window_start_raw)
                else:
                    window_start = now
                counters = {str(k): int(v) for k, v in (data.get("counters") or {}).items()}
                return window_start, counters
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ROUTER gemini RL load failed: %s", exc)
        return now, {}

    def _save_gemini_window(self, window_start: datetime, counters: dict[str, int]) -> None:
        """Persist Gemini rate-limit window and counters."""
        try:
            _GEMINI_RL_FILE.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "window_start": window_start.isoformat(),
                "counters": counters,
            }
            _GEMINI_RL_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ROUTER gemini RL save failed: %s", exc)

    def _can_use_gemini(self, model: str) -> bool:
        """Return True if model is within per-minute Gemini limits and key is set."""
        if not _gemini_key:
            # No key: never use Gemini direct models.
            return False

        # Per-minute limits (RPM) for each Gemini model.
        limits: dict[str, int] = {
            "gemini/gemini-2.0-flash": 15,
            "gemini/gemini-1.5-flash": 15,
            "gemini/gemini-2.5-pro": 2,
        }
        max_rpm = limits.get(model)
        if max_rpm is None:
            # Unknown Gemini model: be conservative and allow it.
            return True

        window_start, counters = self._load_gemini_window()
        now = datetime.now(timezone.utc)

        # Reset window if more than 60 seconds have passed.
        if (now - window_start).total_seconds() >= 60:
            window_start = now
            counters = {}

        used = int(counters.get(model, 0))
        if used >= max_rpm:
            logger.info(
                "ROUTER gemini RL hit  model=%s  used=%d  limit=%d",
                model,
                used,
                max_rpm,
            )
            # Do not update counters/window; just signal caller to skip.
            return False

        # Within limit: increment and persist.
        counters[model] = used + 1
        self._save_gemini_window(window_start, counters)
        return True

    # ------------------------------------------------------------------
    # Groq rate limiting helpers  (mirrors Gemini RL)
    # ------------------------------------------------------------------

    def _load_groq_window(self) -> tuple[datetime, dict[str, int]]:
        """Load current Groq rate-limit window and per-model counts."""
        now = datetime.now(timezone.utc)
        try:
            if _GROQ_RL_FILE.exists():
                raw = _GROQ_RL_FILE.read_text(encoding="utf-8")
                data = json.loads(raw or "{}")
                window_start_raw = data.get("window_start")
                if window_start_raw:
                    window_start = datetime.fromisoformat(window_start_raw)
                    if window_start.tzinfo is None:
                        window_start = window_start.replace(tzinfo=timezone.utc)
                else:
                    window_start = now
                counters = {str(k): int(v) for k, v in (data.get("counters") or {}).items()}
                return window_start, counters
        except Exception as exc:
            logger.warning("ROUTER groq RL load failed: %s", exc)
        return now, {}

    def _save_groq_window(self, window_start: datetime, counters: dict[str, int]) -> None:
        """Persist Groq rate-limit window and counters."""
        try:
            _GROQ_RL_FILE.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "window_start": window_start.isoformat(),
                "counters": counters,
            }
            _GROQ_RL_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("ROUTER groq RL save failed: %s", exc)

    def _can_use_groq(self, model: str) -> bool:
        """Return True if model is within per-minute Groq limits and key is set."""
        if not self._get_groq_key():
            return False

        max_rpm = _GROQ_RATE_LIMITS.get(model)
        if max_rpm is None:
            return True  # Unknown model: allow it

        window_start, counters = self._load_groq_window()
        now = datetime.now(timezone.utc)

        if (now - window_start).total_seconds() >= 60:
            window_start = now
            counters = {}

        used = int(counters.get(model, 0))
        if used >= max_rpm:
            logger.info(
                "ROUTER groq RL hit  model=%s  used=%d  limit=%d",
                model, used, max_rpm,
            )
            return False

        counters[model] = used + 1
        self._save_groq_window(window_start, counters)
        return True

    # ------------------------------------------------------------------
    # Cost helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_cost(model: str) -> float:
        """Rough per-call cost estimate in USD based on tier.

        This is deliberately conservative and does not attempt to model
        exact token usage. It is only used for coarse daily budgeting.
        """
        if model in MODEL_COSTS:
            return MODEL_COSTS[model]
        if model in TIER_1_FREE or model.endswith(":free"):
            return 0.0
        if model in TIER_2_CHEAP:
            return 0.002  # ~0.2 cents per call
        if model in TIER_3_POWERFUL:
            return 0.02   # ~2 cents per call
        # Unknown model: assume cheap-ish.
        return 0.002

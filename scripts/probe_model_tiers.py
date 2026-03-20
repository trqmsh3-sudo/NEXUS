#!/usr/bin/env python3
"""Probe Tier 0 (Gemini direct), Tier 1 (Groq), Tier 2 (OpenRouter).

Requires respective env vars. Run from repo root with PYTHONPATH=.

    python scripts/probe_model_tiers.py
"""

from __future__ import annotations

import os
import sys

# ASCII-only: some providers error if non-ASCII appears in prompts/env.
MESSAGES = [
    {"role": "system", "content": "Reply with exactly: OK"},
    {"role": "user", "content": "ping"},
]

# Minimize tokens
KW = {"max_tokens": 16, "temperature": 0.2}


def _try_litellm(model: str, **extra: object) -> tuple[bool, str]:
    import litellm

    litellm.suppress_debug_info = True
    kw = dict(model=model, messages=MESSAGES, **KW)
    kw.update(extra)
    try:
        resp = litellm.completion(**kw)
        content = (resp.choices[0].message.content or "").strip()
        return True, content[:200]
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"[:500]


def main() -> int:
    results: list[tuple[str, str, bool, str]] = []

    if os.getenv("GEMINI_API_KEY"):
        ok, msg = _try_litellm("gemini/gemini-2.0-flash")
        results.append(("Tier 0", "Gemini direct gemini-2.0-flash", ok, msg))
    else:
        results.append(("Tier 0", "Gemini direct", False, "GEMINI_API_KEY not set"))

    if os.getenv("GROQ_API_KEY"):
        ok, msg = _try_litellm("groq/llama-3.1-8b-instant")
        results.append(("Tier 1", "Groq llama-3.1-8b-instant", ok, msg))
    else:
        results.append(("Tier 1", "Groq", False, "GROQ_API_KEY not set"))

    or_key = os.getenv("OPENROUTER_API_KEY")
    if or_key:
        ok, msg = _try_litellm(
            "openrouter/google/gemini-2.5-flash",
            api_base="https://openrouter.ai/api/v1",
            api_key=or_key,
            extra_headers={
                "HTTP-Referer": "https://github.com/trqmsh3-sudo/NEXUS",
                "X-Title": "NEXUS",
            },
        )
        results.append(("Tier 2", "OpenRouter google/gemini-2.5-flash", ok, msg))
    else:
        results.append(("Tier 2", "OpenRouter", False, "OPENROUTER_API_KEY not set"))

    for tier, label, ok, detail in results:
        status = "OK" if ok else "FAIL"
        print(f"{tier:8} | {status:4} | {label}")
        print(f"          {detail}")
        print()

    return 0 if any(r[2] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

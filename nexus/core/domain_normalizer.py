"""Normalize belief domains before indexing (FIX 4)."""

from __future__ import annotations

import re


def _key(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("_", " ")
    return re.sub(r"\s+", " ", s).lower()


# alias_key -> canonical display name
_DOMAIN_ALIASES: dict[str, str] = {}
for _canonical, _aliases in (
    (
        "Software Engineering",
        (
            "software engineering",
            "software development",
            "software dev",
        ),
    ),
    (
        "AI/ML",
        (
            "ml",
            "machine learning",
            "ml architecture",
            "ai research",
        ),
    ),
    (
        "System Architecture",
        (
            "system architecture",
            "system design",
        ),
    ),
    (
        "Security",
        ("security",),
    ),
    (
        "Testing",
        ("testing",),
    ),
    (
        "Algorithms",
        (
            "algorithms",
            "algorithm design",
            "data structures",
        ),
    ),
    (
        "Concurrency",
        (
            "concurrency",
            "concurrent data structures",
        ),
    ),
):
    for _a in _aliases:
        _DOMAIN_ALIASES[_key(_a)] = _canonical
    _DOMAIN_ALIASES[_key(_canonical)] = _canonical


def normalize_domain(domain: str) -> str:
    """Lowercase/strip-style matching; map known variants; else title-style casing."""
    raw = (domain or "").strip()
    if not raw:
        return "General"
    k = _key(raw)
    if k in _DOMAIN_ALIASES:
        return _DOMAIN_ALIASES[k]
    words = raw.split()
    if not words:
        return "General"
    return " ".join(w[:1].upper() + w[1:].lower() for w in words)

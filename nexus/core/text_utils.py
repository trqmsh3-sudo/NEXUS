"""Text utilities for encoding-safe string handling."""

from __future__ import annotations


def clean_text(text: str) -> str:
    """Ensure text is UTF-8 clean for API calls and serialisation.

    Args:
        text: Raw string that may contain problematic encoding.

    Returns:
        UTF-8-safe string (non-encodable chars dropped).
    """
    return text.encode("utf-8", errors="ignore").decode("utf-8")

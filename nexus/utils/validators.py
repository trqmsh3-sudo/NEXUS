"""Shared validation utilities for NEXUS.

All validators raise ``ValueError`` on failure, making them suitable
for use inside ``__post_init__`` or at API boundaries.
"""

from __future__ import annotations


def validate_confidence(value: float) -> float:
    """Ensure *value* is a valid confidence score in [0.0, 1.0].

    Args:
        value: The confidence value to validate.

    Returns:
        The validated value, unchanged.

    Raises:
        TypeError: If *value* is not a float or int.
        ValueError: If *value* is outside [0.0, 1.0].
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"confidence must be a number, got {type(value).__name__}")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"confidence must be between 0.0 and 1.0, got {value}")
    return float(value)


def validate_decay_rate(value: float) -> float:
    """Ensure *value* is a valid decay rate in [0.0, 1.0].

    Args:
        value: The decay rate to validate.

    Returns:
        The validated value, unchanged.

    Raises:
        TypeError: If *value* is not a float or int.
        ValueError: If *value* is outside [0.0, 1.0].
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"decay_rate must be a number, got {type(value).__name__}")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"decay_rate must be between 0.0 and 1.0, got {value}")
    return float(value)


def validate_non_empty_string(value: str, field_name: str) -> str:
    """Ensure *value* is a non-empty string.

    Args:
        value: The string to validate.
        field_name: Name of the field (used in error messages).

    Returns:
        The validated string, stripped of leading/trailing whitespace.

    Raises:
        TypeError: If *value* is not a string.
        ValueError: If *value* is empty or whitespace-only.
    """
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be empty or whitespace-only")
    return stripped

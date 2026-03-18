"""FIX 4 — domain normalization."""

from nexus.core.domain_normalizer import normalize_domain


def test_alias_software_engineering() -> None:
    assert normalize_domain("software dev") == "Software Engineering"
    assert normalize_domain("SOFTWARE_ENGINEERING") == "Software Engineering"


def test_alias_ai_ml() -> None:
    assert normalize_domain("ml") == "AI/ML"
    assert normalize_domain("Machine Learning") == "AI/ML"


def test_unknown_title_case() -> None:
    assert normalize_domain("  my custom TOPIC ") == "My Custom Topic"


def test_empty_general() -> None:
    assert normalize_domain("") == "General"
    assert normalize_domain("   ") == "General"

"""Tests for business-oriented training_problems_v2.json.

All tests fail on the current coding-task file and pass after replacement.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRAINING_FILE = REPO_ROOT / "data" / "training_problems_v2.json"

CODING_PATTERNS = [
    r"Write a Python (function|class)",
    r"Design a Python class",
    r"def ",
    r"->\s*(bool|int|str|float|list|dict|tuple|None)",
    r"list\[int\]",
    r"dict\[str",
    r"using (re|threading) module",
    r"O\(1\) time",
]

BUSINESS_KEYWORDS = {
    "opportunity", "opportunities", "market", "niche", "revenue",
    "buyer", "buyers", "validate", "validation", "product", "service",
    "freelance", "affiliate", "monetize", "monetization", "sell", "selling",
    "income", "profit", "profitable", "customers", "price", "pricing",
    "demand", "research", "identify", "find", "trending", "audience",
}


@pytest.fixture(scope="module")
def tasks() -> list[str]:
    assert TRAINING_FILE.exists(), f"Training file not found: {TRAINING_FILE}"
    data = json.loads(TRAINING_FILE.read_text(encoding="utf-8"))
    return [str(t) for t in data.get("tasks", [])]


@pytest.fixture(scope="module")
def raw_data() -> dict:
    return json.loads(TRAINING_FILE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_file_exists() -> None:
    assert TRAINING_FILE.exists()


def test_version_preserved(raw_data) -> None:
    assert raw_data.get("version") == 2, "version field must remain 2"


def test_at_least_thirty_tasks(tasks) -> None:
    assert len(tasks) >= 30, f"Expected >= 30 tasks, got {len(tasks)}"


def test_all_tasks_are_non_empty_strings(tasks) -> None:
    for t in tasks:
        assert isinstance(t, str) and t.strip(), f"Empty or non-string task: {t!r}"


def test_all_tasks_are_unique(tasks) -> None:
    assert len(tasks) == len(set(tasks)), "Duplicate tasks found"


# ---------------------------------------------------------------------------
# No coding tasks remain
# ---------------------------------------------------------------------------

def test_no_python_function_tasks(tasks) -> None:
    for task in tasks:
        assert "Write a Python function" not in task, (
            f"Coding task still present: {task!r}"
        )


def test_no_python_class_tasks(tasks) -> None:
    for task in tasks:
        assert "Design a Python class" not in task, (
            f"Coding task still present: {task!r}"
        )


def test_no_coding_patterns(tasks) -> None:
    for task in tasks:
        for pattern in CODING_PATTERNS:
            assert not re.search(pattern, task), (
                f"Task matches coding pattern {pattern!r}: {task!r}"
            )


# ---------------------------------------------------------------------------
# Business coverage — all four categories must be present
# ---------------------------------------------------------------------------

def test_opportunity_finding_tasks_present(tasks) -> None:
    hits = [t for t in tasks if any(
        kw in t.lower() for kw in ("opportunit", "untapped", "trending", "find 3", "find 5")
    )]
    assert len(hits) >= 3, (
        f"Expected >= 3 opportunity-finding tasks, found {len(hits)}"
    )


def test_market_research_tasks_present(tasks) -> None:
    hits = [t for t in tasks if any(
        kw in t.lower() for kw in ("research", "market", "niche", "rates", "average")
    )]
    assert len(hits) >= 3, (
        f"Expected >= 3 market-research tasks, found {len(hits)}"
    )


def test_buyer_identification_tasks_present(tasks) -> None:
    hits = [t for t in tasks if any(
        kw in t.lower() for kw in ("buyer", "customer", "audience", "who pays", "persona")
    )]
    assert len(hits) >= 3, (
        f"Expected >= 3 buyer-identification tasks, found {len(hits)}"
    )


def test_idea_validation_tasks_present(tasks) -> None:
    hits = [t for t in tasks if any(
        kw in t.lower() for kw in ("validate", "validation", "test demand", "pre-sell", "first sale", "7 days", "48-hour", "fast")
    )]
    assert len(hits) >= 3, (
        f"Expected >= 3 idea-validation tasks, found {len(hits)}"
    )


def test_tasks_use_business_vocabulary(tasks) -> None:
    business_task_count = sum(
        1 for t in tasks
        if any(kw in t.lower() for kw in BUSINESS_KEYWORDS)
    )
    ratio = business_task_count / len(tasks)
    assert ratio >= 0.8, (
        f"Only {ratio:.0%} of tasks use business vocabulary (need >= 80%)"
    )

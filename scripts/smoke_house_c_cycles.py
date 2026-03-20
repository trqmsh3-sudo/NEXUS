"""Smoke: 5 House C cycles with real pytest; LLM mocked (task key explicit)."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

from nexus.core.house_b import StructuredSpecificationObject
from nexus.core.house_c import HouseC
from nexus.core.house_d import DestructionReport
from nexus.core.knowledge_graph import KnowledgeGraph


def _sso(prompt: str) -> StructuredSpecificationObject:
    return StructuredSpecificationObject(
        original_input=prompt,
        redefined_problem=prompt,
        assumptions=["Valid types"],
        constraints=["stdlib only"],
        success_criteria=["Tests pass", "Correct API", "Edge cases"],
        required_inputs=[],
        expected_outputs=[],
        domain="General",
        confidence=0.9,
    )


def _survived() -> DestructionReport:
    return DestructionReport(
        target_description="x",
        survived=True,
        survival_score=0.9,
        cycles_survived=1,
        recommendation="PROMOTE",
    )


def _kg() -> KnowledgeGraph:
    return KnowledgeGraph(
        storage_path=str(Path(tempfile.gettempdir()) / f"nxd_{uuid.uuid4().hex}.json"),
    )


CODE = {
    "is_even": """# NEXUS Build
def _warmup():
    return 0

def is_even(n: int) -> bool:
    return n % 2 == 0
""",
    "add": """# NEXUS Build
def add(a: int, b: int) -> int:
    return a + b
""",
    "reverse_string": """# NEXUS Build
def reverse_string(s: str) -> str:
    return s[::-1]
""",
}

TESTS_BAD = {
    "is_even": """import pytest
from main import is_even

def test_basic():
    assert is_even(2) is True

def test_typeerror_on_int():
    with pytest.raises(TypeError):
        is_even(4)
""",
    "add": """import pytest
from main import add

def test_add():
    assert add(1, 2) == 3

def test_typeerror_ints():
    with pytest.raises(TypeError):
        add(1, 2)
""",
    "reverse_string": """from main import reverse_string

def test_rev():
    assert reverse_string("ab") == "ba"

def test_empty_wrong():
    assert reverse_string("") is None
""",
}


def main() -> None:
    calls: list[str] = []

    tasks: list[tuple[str, str]] = [
        ("Write a Python function is_even(n: int) -> bool", "is_even"),
        ("Write a Python function add(a: int, b: int) -> int", "add"),
        ("Write a Python function reverse_string(s: str) -> str", "reverse_string"),
        ("Write a Python function is_even(n: int) -> bool", "is_even"),
        ("Write a Python function add(a: int, b: int) -> int", "add"),
    ]

    def make_llm(task_key: str):
        def llm_side_effect(system: str, user: str, label: str) -> str:
            calls.append(label)
            tk = task_key
            if label == "generate_code":
                return CODE[tk]
            if label == "generate_tests":
                return TESTS_BAD[tk]
            if tk == "is_even":
                return """import pytest
from main import is_even
def test_basic():
    assert is_even(2) is True
def test_odd():
    assert is_even(3) is False
"""
            if tk == "add":
                return """from main import add
def test_add():
    assert add(1, 2) == 3
"""
            return """from main import reverse_string
def test_rev():
    assert reverse_string("ab") == "ba"
def test_empty():
    assert reverse_string("") == ""
"""

        return llm_side_effect

    passed = 0
    for i, (prompt, task_key) in enumerate(tasks):
        calls.clear()
        hc = HouseC(knowledge_graph=_kg(), workspace_dir=str(Path(tempfile.mkdtemp())))
        with patch.object(hc, "_call_llm", side_effect=make_llm(task_key)):
            r = hc.build(_sso(prompt), _survived())
        ok = r.success
        passed += int(ok)
        print(
            f"cycle {i + 1} ({task_key}): success={ok} "
            f"healing={r.artifact.healing_attempts} llm_calls={calls}",
        )
    print(f"SUCCESS RATE: {passed}/{len(tasks)} = {100 * passed / len(tasks):.0f}%")


if __name__ == "__main__":
    main()

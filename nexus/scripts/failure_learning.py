"""Turn every failed cycle into a learning opportunity — generate simpler variants."""

from __future__ import annotations

import json
import re
from pathlib import Path


def _simplify_task(user_input: str, failure_reason: str | None) -> str:
    """Generate a simpler variant of a failed task.

    Heuristics:
    - Remove complex qualifiers (with pytest, with cache, etc.)
    - Extract core: "Write a Python function X that does Y"
    - Shorten long descriptions to one clear function signature
    """
    inp = (user_input or "").strip()
    if not inp:
        return "Write a Python function add(a: int, b: int) -> int"

    # Common simplification patterns
    simplifications = [
        (r"Build a .*? class .*? that .*?\. Use only .*", "Write a Python class with minimal implementation"),
        (r"Design .*? (?:Python )?function.*? (?:that|to) ([^.]+)\.?", r"Write a Python function that \1"),
        (r"Write a Python (?:class|function) .*? (?:with|using) .*", lambda m: m.group(0).split(" with ")[0].split(" using ")[0].rstrip(".,") + ""),
        (r"Must handle .*$", ""),
        (r"No external dependencies\.?$", ""),
        (r"Use only the (?:re|time|json) module\.?$", ""),
        (r"Use only Python standard library\.?$", ""),
    ]

    result = inp
    for pattern, repl in simplifications:
        if callable(repl):
            m = re.search(pattern, result, re.IGNORECASE | re.DOTALL)
            if m:
                result = repl(m) or result
        else:
            result = re.sub(pattern, repl, result, flags=re.IGNORECASE | re.DOTALL)
        result = result.strip()

    # If we have a clear "function X" pattern, keep it minimal
    if "function" in result.lower() and len(result) > 120:
        # Try to extract just the function part
        match = re.search(r"Write a Python function ([^.]+)", result, re.IGNORECASE)
        if match:
            core = match.group(1).strip()
            if len(core) < 80:
                result = f"Write a Python function {core}"

    result = result.strip().rstrip(".,")
    if not result:
        result = "Write a Python function that solves the problem"
    return result


def main() -> None:
    """Read cycle_history.json, find failures, generate simpler variants, add to training queue."""
    history_path = Path("data/cycle_history.json")
    training_path = Path("data/training_problems_v2.json")

    if not history_path.exists():
        print(f"No {history_path} found. Run NEXUS first.")
        return

    with open(history_path, encoding="utf-8") as f:
        cycles = json.load(f)

    failed = [c for c in cycles if c.get("success") is False]
    print(f"Found {len(failed)} failed cycles out of {len(cycles)} total")

    simplified: list[str] = []
    seen: set[str] = set()
    for c in failed:
        user_input = c.get("user_input") or ""
        failure_reason = c.get("failure_reason")
        simpler = _simplify_task(user_input, failure_reason)
        if simpler and simpler not in seen:
            seen.add(simpler)
            simplified.append(simpler)

    # Load existing training problems if present
    if training_path.exists():
        with open(training_path, encoding="utf-8") as f:
            data = json.load(f)
        existing = set(data.get("tasks", []))
    else:
        data = {"version": 2, "tasks": []}
        existing = set()

    added = 0
    for task in simplified:
        if task not in existing:
            data["tasks"].append(task)
            existing.add(task)
            added += 1
            print(f"  + {task[:70]}...")

    training_path.parent.mkdir(parents=True, exist_ok=True)
    with open(training_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\nAdded {added} simpler variants to {training_path}")
    print(f"Total tasks: {len(data['tasks'])}")


if __name__ == "__main__":
    main()

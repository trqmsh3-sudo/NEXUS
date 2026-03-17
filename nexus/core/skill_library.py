"""Skill Library — compile reusable skills from beliefs and inject into prompts."""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nexus.core.belief_certificate import BeliefCertificate

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """A reusable, parameterized Python function compiled from a belief."""

    skill_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    function_code: str = ""
    function_signature: str = ""
    tags: list[str] = field(default_factory=list)
    domain: str = "General"
    usage_count: int = 0
    success_rate: float = 0.0
    created_from_belief: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "function_code": self.function_code,
            "function_signature": self.function_signature,
            "tags": list(self.tags),
            "domain": self.domain,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "created_from_belief": self.created_from_belief,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Skill:
        return cls(
            skill_id=data.get("skill_id", str(uuid.uuid4())),
            name=data.get("name", ""),
            description=data.get("description", ""),
            function_code=data.get("function_code", ""),
            function_signature=data.get("function_signature", ""),
            tags=list(data.get("tags", [])),
            domain=data.get("domain", "General"),
            usage_count=int(data.get("usage_count", 0)),
            success_rate=float(data.get("success_rate", 0.0)),
            created_from_belief=data.get("created_from_belief", ""),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.now(timezone.utc),
        )


def _groq_completion(messages: list[dict[str, str]], max_tokens: int = 1500) -> str:
    """Single Groq call; returns empty string on missing key or error."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return ""
    try:
        import litellm
        litellm.suppress_debug_info = True
        r = litellm.completion(
            model="groq/llama-3.1-8b-instant",
            messages=messages,
            api_key=key,
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.debug("skill_library groq failed: %s", exc)
        return ""


def _extract_first_function_def(code: str) -> tuple[str, str]:
    """Return (signature_line, full_function_code) or ('', '')."""
    code = code.strip()
    for fence in ("```python", "```"):
        if fence in code:
            idx = code.find(fence) + len(fence)
            code = code[idx:].lstrip()
    if "```" in code:
        code = code.split("```")[0].strip()
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                start = node.lineno - 1
                end = node.end_lineno or start + 1
                lines = code.splitlines()
                block = "\n".join(lines[start:end])
                sig = ast.get_source_segment(code, node) or ""
                first_line = sig.split("\n")[0].strip() if sig else ""
                return first_line, block
    except SyntaxError:
        pass
    match = re.search(r"def\s+(\w+)\s*\([^)]*\)\s*(?:->\s*[\w\[\], \.]*)?:", code)
    if match:
        start = code.find(match.group(0))
        end = code.find("\n\n", start)
        if end == -1:
            end = len(code)
        block = code[start:end].strip()
        return match.group(0).strip(), block
    return "", ""


class SkillLibrary:
    """Store and retrieve compiled skills; compile from beliefs via Groq."""

    def __init__(self, storage_path: str = "data/skills/library.json") -> None:
        self.skills: dict[str, Skill] = {}
        self.storage_path: str = storage_path
        self._usage_this_cycle: int = 0
        self.load()

    @property
    def usage_this_cycle(self) -> int:
        return self._usage_this_cycle

    def record_usage_this_cycle(self) -> None:
        self._usage_this_cycle += 1

    def reset_usage_this_cycle(self) -> None:
        self._usage_this_cycle = 0

    def compile_from_belief(self, belief: BeliefCertificate) -> Skill | None:
        """Turn a belief with executable_proof into a reusable Skill via Groq. Returns None if failed."""
        if not belief.executable_proof or not belief.executable_proof.strip():
            return None
        proof = (belief.executable_proof or "").strip()[:8000]
        claim = (belief.claim or "")[:1500]
        domain = belief.domain or "General"

        system = (
            "You are a code refactoring assistant. Generalize the given code into a single "
            "reusable, parameterized Python function. Return ONLY raw Python code: one function "
            "with a clear signature, docstring, and no external imports. No markdown, no explanation."
        )
        user = (
            f"Claim/context:\n{claim}\n\n"
            f"Code or proof to generalize:\n{proof}\n\n"
            "Produce ONE reusable Python function (standard library only)."
        )
        raw = _groq_completion(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=1200,
        )
        if not raw or len(raw) < 20:
            return None

        sig, block = _extract_first_function_def(raw)
        if not block:
            return None

        name = "generalized_fn"
        if sig:
            m = re.search(r"def\s+(\w+)\s*\(", sig)
            if m:
                name = m.group(1)

        skill = Skill(
            name=name,
            description=claim[:500],
            function_code=block,
            function_signature=sig or f"def {name}(...):",
            tags=_tags_from_claim_and_domain(claim, domain),
            domain=domain,
            usage_count=0,
            success_rate=0.0,
            created_from_belief=belief.source or belief.claim[:200],
            created_at=datetime.now(timezone.utc),
        )
        self.skills[skill.skill_id] = skill
        self.save()
        logger.info("SkillLibrary compiled skill  skill_id=%s  name=%s", skill.skill_id, skill.name)
        return skill

    def get_relevant_skills(self, task: str, max_k: int = 3) -> list[Skill]:
        """Simple keyword relevance: task words vs tags + domain. Returns top max_k."""
        task_lower = task.lower()
        task_words = set(re.findall(r"\w+", task_lower)) - {"a", "an", "the", "for", "to", "in", "of", "and", "or"}

        def score(s: Skill) -> int:
            tags_lower = " ".join(s.tags).lower() + " " + (s.domain or "").lower()
            domain_words = set(re.findall(r"\w+", tags_lower))
            return len(task_words & domain_words) + (1 if any(w in task_lower for w in tags_lower.split()) else 0)

        scored = [(score(s), s) for s in self.skills.values()]
        scored.sort(key=lambda x: (-x[0], -x[1].usage_count))
        return [s for _, s in scored[:max_k]]

    def inject_into_prompt(self, task: str, max_k: int = 3) -> str:
        """Formatted string of available tools for prompt injection."""
        skills = self.get_relevant_skills(task, max_k=max_k)
        if not skills:
            return ""
        lines = ["Available tools (verified):"]
        for i, s in enumerate(skills, 1):
            lines.append(f"  {i}. {s.function_signature}")
        return "\n".join(lines)

    def save(self) -> None:
        try:
            path = Path(self.storage_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            data = [s.to_dict() for s in self.skills.values()]
            path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        except OSError as exc:
            logger.warning("SkillLibrary save failed: %s", exc)

    def load(self) -> None:
        try:
            path = Path(self.storage_path)
            if not path.exists():
                return
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw or "[]")
            if not isinstance(data, list):
                return
            for item in data:
                if isinstance(item, dict):
                    try:
                        s = Skill.from_dict(item)
                        self.skills[s.skill_id] = s
                    except (KeyError, TypeError, ValueError):
                        pass
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("SkillLibrary load failed: %s", exc)


def _tags_from_claim_and_domain(claim: str, domain: str) -> list[str]:
    """Derive a few tags from claim text and domain."""
    words = re.findall(r"\w+", (claim + " " + domain).lower())
    stop = {"the", "a", "an", "is", "are", "for", "to", "in", "of", "and", "or", "that", "this"}
    return list(dict.fromkeys(w for w in words if len(w) > 2 and w not in stop))[:15]

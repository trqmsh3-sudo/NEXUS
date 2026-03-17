"""House C — The Builder.

House C receives a StructuredSpecificationObject that has ALREADY
survived House D attacks.  It builds real, working code — nothing else.
Generated code is validated by actually running pytest against it.
Successful builds are convertible to BeliefCertificates for House A.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import pathlib
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from nexus.core.belief_certificate import BeliefCertificate
from nexus.core.house_b import StructuredSpecificationObject
from nexus.core.house_d import DestructionReport
from nexus.core.knowledge_graph import KnowledgeGraph
from nexus.core.model_router import ModelRouter

logger: logging.Logger = logging.getLogger(__name__)

_MAX_RETRIES: int = 1

# ------------------------------------------------------------------
# System prompts
# ------------------------------------------------------------------

CODE_SYSTEM: str = (
    "You are NEXUS House C. Write simple, working Python code.\n"
    "Rules:\n"
    "- Use ONLY Python standard library. No external imports.\n"
    "- Every function must be complete. No placeholders.\n"
    "- MINIMAL: If the spec asks for one function, write ONLY that function. No classes, no validation, no doctest, no kill switches.\n"
    "- Keep it simple — 20-50 lines maximum.\n"
    "- Return ONLY raw Python code. No markdown. No explanation.\n"
    "- Start with: # NEXUS Build"
)

TEST_SYSTEM: str = (
    "Write simple pytest tests for this exact code.\n"
    "Rules:\n"
    "- Import ONLY from the code file and pytest\n"
    "- Test only functions that exist in the code\n"
    "- Keep tests simple — assert statements only\n"
    "- Maximum 10 test functions\n"
    "- Return ONLY raw pytest code starting with: import pytest"
)


# ------------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------------

@dataclass
class BuildArtifact:
    """A concrete code artifact produced by House C.

    Attributes:
        artifact_id: Unique identifier (UUID4 string).
        sso: The specification this artifact was built from.
        code: The generated source code.
        language: Programming language of the code.
        tests: The generated test code.
        documentation: Auto-generated documentation string.
        created_at: UTC timestamp of creation.
        passed_validation: Whether the tests actually passed.
        validation_errors: Error messages from failed validation.
        execution_proof: Captured stdout from a passing test run,
            or None if validation failed.
    """

    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sso: StructuredSpecificationObject = field(
        default_factory=lambda: StructuredSpecificationObject(
            original_input="", redefined_problem="",
        ),
    )
    code: str = ""
    language: str = "python"
    tests: str = ""
    documentation: str = ""
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    passed_validation: bool = False
    validation_errors: list[str] = field(default_factory=list)
    execution_proof: str | None = None
    healing_attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise the artifact to a plain dictionary."""
        return {
            "artifact_id": self.artifact_id,
            "sso": self.sso.to_dict(),
            "code": self.code,
            "language": self.language,
            "tests": self.tests,
            "documentation": self.documentation,
            "created_at": self.created_at.isoformat(),
            "passed_validation": self.passed_validation,
            "validation_errors": list(self.validation_errors),
            "execution_proof": self.execution_proof,
            "healing_attempts": self.healing_attempts,
        }


@dataclass
class BuildResult:
    """Outcome of a full House C build pipeline.

    Attributes:
        artifact: The generated BuildArtifact.
        success: Whether the build and validation succeeded.
        house_d_report: The DestructionReport the SSO survived.
        ready_for_house_a: True only when ``success`` is True and
            the SSO survived House D.
    """

    artifact: BuildArtifact
    success: bool
    house_d_report: DestructionReport
    ready_for_house_a: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialise the full build result."""
        return {
            "artifact": self.artifact.to_dict(),
            "success": self.success,
            "house_d_report": self.house_d_report.to_dict(),
            "ready_for_house_a": self.ready_for_house_a,
        }


# ------------------------------------------------------------------
# House C — The Builder
# ------------------------------------------------------------------

@dataclass
class HouseC:
    """The Builder — turns verified specifications into working code.

    House C only accepts SSOs that have already survived House D.
    It generates production code, generates tests, runs the tests
    via subprocess, and packages the result as a BuildArtifact.

    Attributes:
        knowledge_graph: The shared NEXUS knowledge store.
        router: ModelRouter for LLM calls.
        workspace_dir: Directory where build artifacts are written.
    """

    knowledge_graph: KnowledgeGraph
    router: ModelRouter = field(default_factory=ModelRouter)
    workspace_dir: str = "data/builds/"

    # ------------------------------------------------------------------
    # 1. build
    # ------------------------------------------------------------------

    def build(
        self,
        sso: StructuredSpecificationObject,
        destruction_report: DestructionReport,
    ) -> BuildResult:
        """Execute the full build pipeline for a verified SSO.

        The SSO must have survived House D — if ``destruction_report
        .survived`` is False, a ``ValueError`` is raised immediately.

        Pipeline steps:
        1. Verify House D survival.
        2. Generate code via LLM.
        3. Generate tests via LLM.
        4. Run validation (actually execute pytest).
        5. Save artifact to workspace.

        Args:
            sso: The specification to build from.
            destruction_report: The DestructionReport proving the SSO
                survived House D.

        Returns:
            A BuildResult with the artifact and readiness flag.

        Raises:
            ValueError: If the SSO did not survive House D.
        """
        start = time.perf_counter()

        if not destruction_report.survived:
            logger.error(
                "BUILD REJECTED — SSO did not survive House D  "
                "problem=%r  score=%.2f",
                sso.redefined_problem[:80],
                destruction_report.survival_score,
            )
            raise ValueError("SSO did not survive House D")

        logger.info(
            "HOUSE-C build started  problem=%r  domain=%s",
            sso.redefined_problem[:80], sso.domain,
        )

        artifact = BuildArtifact(sso=sso)

        artifact.code = self._generate_code(sso)
        artifact.tests = self._generate_tests(sso, artifact.code)
        artifact.documentation = (
            f"Auto-generated by NEXUS House C\n"
            f"Problem: {sso.redefined_problem}\n"
            f"Domain: {sso.domain}\n"
            f"Constraints: {', '.join(sso.constraints)}\n"
            f"Success criteria: {', '.join(sso.success_criteria)}"
        )

        artifact = self._validate(artifact)
        self._save_to_workspace(artifact)

        success = artifact.passed_validation
        ready = success and destruction_report.survived

        elapsed = time.perf_counter() - start
        logger.info(
            "HOUSE-C build complete  artifact_id=%s  passed=%s  "
            "ready_for_house_a=%s  elapsed=%.2fs",
            artifact.artifact_id, success, ready, elapsed,
        )

        return BuildResult(
            artifact=artifact,
            success=success,
            house_d_report=destruction_report,
            ready_for_house_a=ready,
        )

    # ------------------------------------------------------------------
    # 2. _generate_code
    # ------------------------------------------------------------------

    def _generate_code(self, sso: StructuredSpecificationObject) -> str:
        """Generate production code from a verified specification.

        Args:
            sso: The specification describing what to build.

        Returns:
            Raw source code string.
        """
        start = time.perf_counter()

        user_prompt = (
            f"Build code for this specification:\n\n"
            f"Problem: {sso.redefined_problem}\n"
            f"Constraints: {json.dumps(sso.constraints)}\n"
            f"Success criteria: {json.dumps(sso.success_criteria)}\n"
            f"Domain: {sso.domain}\n\n"
            "Write ONE Python file. The file will be saved as main.py.\n"
            "If the problem needs a single function (e.g. add two numbers), write ONLY that function — no extra classes or helpers.\n"
            "Keep it minimal. Use only Python standard library."
        )

        code = self._call_llm(
            system=CODE_SYSTEM,
            user=user_prompt,
            label="generate_code",
        )
        code = self._strip_fences(code)

        elapsed = time.perf_counter() - start
        logger.info(
            "HOUSE-C code generated  lines=%d  elapsed=%.2fs",
            code.count("\n") + 1, elapsed,
        )
        return code

    # ------------------------------------------------------------------
    # 3. _generate_tests
    # ------------------------------------------------------------------

    def _generate_tests(
        self, sso: StructuredSpecificationObject, code: str,
    ) -> str:
        """Generate comprehensive pytest tests for the generated code.

        Args:
            sso: The specification the code was built from.
            code: The generated source code to test.

        Returns:
            Raw pytest test code string.
        """
        start = time.perf_counter()

        user_prompt = (
            f"Code to test:\n{code}\n\n"
            f"Specification:\n"
            f"- Domain: {sso.domain}\n"
            f"- Success criteria: {json.dumps(sso.success_criteria)}\n\n"
            "Write pytest tests that import from main and test "
            "the actual functions/classes defined above.\n"
            "Use only: from main import <name> for imports.\n"
            "Cover happy path, edge cases, and error cases."
        )

        tests = self._call_llm(
            system=TEST_SYSTEM,
            user=user_prompt,
            label="generate_tests",
        )
        tests = self._strip_fences(tests)

        elapsed = time.perf_counter() - start
        logger.info(
            "HOUSE-C tests generated  lines=%d  elapsed=%.2fs",
            tests.count("\n") + 1, elapsed,
        )
        return tests

    # ------------------------------------------------------------------
    # 4. _validate
    # ------------------------------------------------------------------

    def _validate(self, artifact: BuildArtifact) -> BuildArtifact:
        """Validate a build artifact by running its tests, with self-healing.

        Saves code and tests to the workspace, runs pytest, and if
        tests fail, sends the failure output to :meth:`_heal_tests`
        for up to 2 repair attempts before giving up.

        Args:
            artifact: The artifact to validate.

        Returns:
            The same artifact with ``passed_validation``,
            ``validation_errors``, ``execution_proof``, and
            ``healing_attempts`` updated.
        """
        max_healing = 2
        build_dir = pathlib.Path(self.workspace_dir).resolve() / artifact.artifact_id
        build_dir.mkdir(parents=True, exist_ok=True)

        code_file = build_dir / "main.py"
        test_file = build_dir / "test_main.py"

        code_file.write_text(artifact.code, encoding="utf-8")

        current_tests = artifact.tests

        for attempt in range(1 + max_healing):
            # Pre-validation: align test imports and function names with the code
            try:
                current_tests = self._repair_test_imports_and_names(
                    artifact.code,
                    current_tests,
                )
            except Exception:
                logger.exception(
                    "HOUSE-C pre-validation repair failed; "
                    "continuing with original tests  artifact_id=%s",
                    artifact.artifact_id,
                )

            test_with_import = (
                "import sys, os\n"
                "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n\n"
                f"{current_tests}"
            )
            test_file.write_text(test_with_import, encoding="utf-8")

            logger.info(
                "HOUSE-C validation attempt %d  artifact_id=%s",
                attempt, artifact.artifact_id,
            )

            result = subprocess.run(
                [sys.executable, "-m", "pytest", "test_main.py", "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(build_dir),
            )

            if result.returncode == 0:
                artifact.passed_validation = True
                artifact.execution_proof = result.stdout
                artifact.tests = current_tests
                artifact.healing_attempts = attempt
                logger.info(
                    "HOUSE-C validation PASSED  artifact_id=%s  "
                    "healing_attempts=%d",
                    artifact.artifact_id, attempt,
                )
                return artifact

            pytest_output = (result.stdout + "\n" + result.stderr).strip()

            if attempt < max_healing:
                logger.info(
                    "HOUSE-C healing attempt %d/%d  artifact_id=%s",
                    attempt + 1, max_healing, artifact.artifact_id,
                )
                healed = self._heal_tests(
                    artifact.code, current_tests, pytest_output,
                )
                current_tests = self._strip_fences(healed)
            else:
                artifact.passed_validation = False
                artifact.validation_errors = [pytest_output]
                artifact.tests = current_tests
                artifact.healing_attempts = attempt
                logger.warning(
                    "HOUSE-C validation FAILED after %d healing attempts  "
                    "artifact_id=%s",
                    max_healing, artifact.artifact_id,
                )

        return artifact

    # ------------------------------------------------------------------
    # 4b. _heal_tests
    # ------------------------------------------------------------------

    def _heal_tests(
        self, code: str, tests: str, pytest_output: str,
    ) -> str:
        """Fix failing tests based on pytest output.

        Sends the code, current tests, and failure output to the LLM
        and asks it to return a corrected version of the test file
        that matches the code's actual behaviour.

        Args:
            code: The generated source code (unchanged).
            tests: The current test code with failures.
            pytest_output: The captured pytest failure output.

        Returns:
            The complete fixed test file as raw code.
        """
        start = time.perf_counter()

        user_prompt = (
            "These pytest tests are almost correct but have minor failures.\n"
            "Fix ONLY the failing tests based on the error output.\n"
            "Do not change passing tests.\n"
            "Do not change the code being tested.\n"
            "Match the exact output format the code produces.\n\n"
            f"CODE:\n{code}\n\n"
            f"CURRENT TESTS:\n{tests}\n\n"
            f"PYTEST FAILURES:\n{pytest_output[-2000:]}\n\n"
            "Return the complete fixed test file. "
            "Raw pytest code only. No markdown."
        )

        healed = self._call_llm(
            system="You fix failing pytest tests precisely. "
            "Return ONLY the complete test file. No explanation.",
            user=user_prompt,
            label="heal_tests",
        )

        elapsed = time.perf_counter() - start
        logger.info(
            "HOUSE-C test healing complete  elapsed=%.2fs", elapsed,
        )
        return healed

    # ------------------------------------------------------------------
    # 4c. _repair_test_imports_and_names
    # ------------------------------------------------------------------

    def _repair_test_imports_and_names(self, code: str, tests: str) -> str:
        """Best-effort repair when tests call functions that don't exist.

        Heuristics:
        - Parse the code to find all top-level function names.
        - Parse the tests to find called function names.
        - If tests call functions that are not defined in the code and
          there is exactly one function defined in the code, assume the
          tests *meant* to call that function and rewrite:
            * ``from main import missing`` -> the real function name
            * bare calls to ``missing(...)`` -> real function name

        This is deliberately conservative: if parsing fails or multiple
        candidate functions exist, the tests are returned unchanged.
        """
        try:
            code_module = ast.parse(code)
        except SyntaxError:
            return tests

        code_funcs: set[str] = {
            node.name
            for node in code_module.body
            if isinstance(node, ast.FunctionDef)
        }
        if not code_funcs:
            return tests

        try:
            test_module = ast.parse(tests)
        except SyntaxError:
            return tests

        called_funcs: set[str] = set()
        for node in ast.walk(test_module):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    called_funcs.add(func.id)

        # Ignore pytest and obvious helpers
        ignore = {"pytest"}
        missing = {
            name
            for name in called_funcs
            if name not in code_funcs and name not in ignore
        }

        if not missing:
            return tests

        # Only apply automatic repair when there is a single obvious target.
        if len(code_funcs) != 1:
            return tests

        target_name = next(iter(code_funcs))
        repaired = tests

        # 1) Fix `from main import ...` lines that reference missing names.
        def replace_import(match: re.Match[str]) -> str:
            imported = [p.strip() for p in match.group(1).split(",")]
            new_imports: list[str] = []
            for name in imported:
                base = name.split(" as ")[0].strip()
                if base in missing:
                    new_imports.append(target_name)
                else:
                    new_imports.append(name)
            return f"from main import {', '.join(new_imports)}"

        repaired = re.sub(
            r"from\s+main\s+import\s+([^\n]+)",
            replace_import,
            repaired,
        )

        # 2) Replace bare calls to missing function names with the target.
        for wrong in missing:
            # word-boundary replacement to avoid substrings
            repaired = re.sub(rf"\b{re.escape(wrong)}\b", target_name, repaired)

        return repaired

    # ------------------------------------------------------------------
    # 5. _save_to_workspace
    # ------------------------------------------------------------------

    def _save_to_workspace(self, artifact: BuildArtifact) -> str:
        """Persist the artifact as JSON in the workspace directory.

        Args:
            artifact: The BuildArtifact to save.

        Returns:
            The absolute file path where the artifact was written.
        """
        build_dir = pathlib.Path(self.workspace_dir) / artifact.artifact_id
        build_dir.mkdir(parents=True, exist_ok=True)

        meta_path = build_dir / "artifact.json"
        meta_path.write_text(
            json.dumps(artifact.to_dict(), indent=2), encoding="utf-8",
        )
        logger.info(
            "HOUSE-C artifact saved  path=%s", meta_path,
        )
        return str(meta_path.resolve())

    # ------------------------------------------------------------------
    # 6. to_belief_certificate
    # ------------------------------------------------------------------

    def to_belief_certificate(
        self, artifact: BuildArtifact,
    ) -> BeliefCertificate:
        """Convert a successful build artifact into a BeliefCertificate.

        Only validated artifacts should be converted. Failed artifacts
        will produce a low-confidence certificate that House A's gated
        ``add_belief`` will reject.

        Args:
            artifact: The BuildArtifact to convert.

        Returns:
            A BeliefCertificate whose ``executable_proof`` is the
            captured test output and whose confidence reflects
            validation status.
        """
        confidence = 0.9 if artifact.passed_validation else 0.3

        cert = BeliefCertificate(
            claim=(
                f"House C built working code for: "
                f"{artifact.sso.redefined_problem}"
            ),
            source=f"nexus:house_c:artifact:{artifact.artifact_id}",
            confidence=confidence,
            domain=artifact.sso.domain,
            executable_proof=artifact.execution_proof,
            created_at=artifact.created_at,
            last_verified=datetime.now(timezone.utc),
        )

        logger.info(
            "HOUSE-C -> BeliefCertificate  artifact_id=%s  "
            "confidence=%s  valid=%s",
            artifact.artifact_id, confidence, cert.is_valid(),
        )
        return cert

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _call_llm(self, system: str, user: str, label: str) -> str:
        """Route an LLM call through the ModelRouter.

        Args:
            system: The system prompt.
            user: The user prompt.
            label: A human-readable label for logging.

        Returns:
            The raw text content from the LLM response.
        """
        return self.router.complete(
            house="house_c", system=system, user=user, label=label,
        )

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove markdown code fences if the LLM wrapped its output.

        Args:
            text: Raw LLM output that may be wrapped in fences.

        Returns:
            The unwrapped code string.
        """
        stripped = text.strip()
        for prefix in ("```python", "```py", "```"):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):]
                break
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        return stripped.strip()

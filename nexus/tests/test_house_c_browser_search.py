"""Tests for House C browser search quality (TDD — written before implementation).

Problems being fixed:
  1. Timeout too short (30s) — browser automation needs 120s+
  2. Prompt too vague — OpenClaw needs concrete browser action steps
  3. No validation — "Browser operational" passes as a real finding

Coverage:
  1. send() timeout is at least 120s
  2. Task prompt contains step-by-step browser instructions (navigate, search, extract)
  3. Task prompt includes the actual search terms from the SSO
  4. Task prompt tells OpenClaw what data to extract (title, URL, rate/price)
  5. Task prompt tells OpenClaw the output format (FINDING: title | URL | ...)
  6. Task prompt explicitly forbids returning browser-state messages
  7. "FINDING: Browser operational" → rejected as NO_DATA (status, not a finding)
  8. "FINDING: Browser loaded" variants → rejected
  9. Real FINDING with URL → accepted
  10. Real FINDING with price/rate data → accepted
  11. Multiple real findings → all accepted
  12. Empty result still fails validation
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nexus.core.house_b import StructuredSpecificationObject
from nexus.core.house_c import BuildArtifact, HouseC
from nexus.core.house_d import DestructionReport
from nexus.core.knowledge_graph import KnowledgeGraph
from nexus.core.openclaw_client import OpenClawClient


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def _make_graph() -> KnowledgeGraph:
    return MagicMock(spec=KnowledgeGraph)


def _make_sso(problem: str = "find paid Python freelance gigs on Upwork") -> StructuredSpecificationObject:
    return StructuredSpecificationObject(
        original_input=problem,
        redefined_problem=problem,
        domain="Freelance",
        constraints=[],
        success_criteria=["find at least one listing with a rate"],
    )


def _make_dr(survived: bool = True) -> DestructionReport:
    return DestructionReport(
        target_description="test",
        survived=survived,
        survival_score=0.8,
        cycles_survived=1,
        recommendation="PROMOTE",
        attacks=[],
    )


def _capture_send_call(tmp_path, problem: str = "find paid Python freelance gigs on Upwork") -> tuple[str, int]:
    """Run _execute_browser_task and capture the task string and timeout sent to client.send()."""
    captured: dict = {}

    client = MagicMock(spec=OpenClawClient)
    client.is_available.return_value = True

    def capture(task: str, timeout: int = 30) -> str:
        captured["task"] = task
        captured["timeout"] = timeout
        return "FINDING: Python contract | https://upwork.com/job/123 | $120/hr | Remote"

    client.send.side_effect = capture

    hc = HouseC(knowledge_graph=_make_graph(), openclaw_client=client, workspace_dir=str(tmp_path))
    sso = _make_sso(problem)
    artifact = BuildArtifact(sso=sso)
    hc._execute_browser_task(artifact, client)

    return captured.get("task", ""), captured.get("timeout", 0)


# ═══════════════════════════════════════════════════════════════
#  1. Timeout
# ═══════════════════════════════════════════════════════════════

class TestBrowserTimeout:

    def test_send_called_with_timeout_at_least_120(self, tmp_path):
        """Browser automation must have >= 120s to navigate, search, and extract."""
        _, timeout = _capture_send_call(tmp_path)
        assert timeout >= 120, (
            f"send() called with timeout={timeout}s — must be >= 120s for browser automation"
        )


# ═══════════════════════════════════════════════════════════════
#  2-6. Task prompt content
# ═══════════════════════════════════════════════════════════════

class TestTaskPromptContent:

    def test_prompt_includes_navigate_step(self, tmp_path):
        """Prompt must tell OpenClaw to navigate to a URL."""
        task, _ = _capture_send_call(tmp_path)
        nav_signals = ["navigate", "go to", "open", "visit"]
        assert any(s in task.lower() for s in nav_signals), (
            f"Prompt has no navigation step.\nPrompt:\n{task[:500]}"
        )

    def test_prompt_includes_search_step(self, tmp_path):
        """Prompt must tell OpenClaw to type and execute a search query."""
        task, _ = _capture_send_call(tmp_path)
        search_signals = ["search", "type", "query", "enter"]
        assert any(s in task.lower() for s in search_signals), (
            f"Prompt has no search step.\nPrompt:\n{task[:500]}"
        )

    def test_prompt_includes_actual_problem_text(self, tmp_path):
        """Prompt must embed the SSO's redefined_problem as the search terms."""
        problem = "find paid Python freelance gigs on Upwork"
        task, _ = _capture_send_call(tmp_path, problem=problem)
        # Key words from the problem must appear in the task
        assert "Python" in task or "freelance" in task.lower(), (
            f"Prompt does not include search terms from the SSO.\nPrompt:\n{task[:500]}"
        )

    def test_prompt_asks_to_extract_url(self, tmp_path):
        """Prompt must ask OpenClaw to extract URLs from results."""
        task, _ = _capture_send_call(tmp_path)
        assert "url" in task.lower() or "link" in task.lower(), (
            f"Prompt does not ask for URLs.\nPrompt:\n{task[:500]}"
        )

    def test_prompt_asks_to_extract_price_or_rate(self, tmp_path):
        """Prompt must ask OpenClaw to extract price or rate information."""
        task, _ = _capture_send_call(tmp_path)
        assert any(s in task.lower() for s in ["price", "rate", "pay", "salary", "budget"]), (
            f"Prompt does not ask for price/rate data.\nPrompt:\n{task[:500]}"
        )

    def test_prompt_specifies_finding_format(self, tmp_path):
        """Prompt must tell OpenClaw to start each result with 'FINDING:'."""
        task, _ = _capture_send_call(tmp_path)
        assert "FINDING:" in task, (
            f"Prompt does not specify 'FINDING:' output format.\nPrompt:\n{task[:500]}"
        )

    def test_prompt_forbids_status_messages(self, tmp_path):
        """Prompt must explicitly forbid returning browser-state messages."""
        task, _ = _capture_send_call(tmp_path)
        forbid_signals = [
            "do not return status", "not status", "no status",
            "only return data", "only actual data", "not browser state",
            "do not report browser", "avoid status",
        ]
        assert any(s in task.lower() for s in forbid_signals), (
            f"Prompt does not forbid status messages.\nPrompt:\n{task[:500]}"
        )


# ═══════════════════════════════════════════════════════════════
#  7-8. Status findings rejected
# ═══════════════════════════════════════════════════════════════

class TestStatusFindingsRejected:

    def _run_with_result(self, tmp_path, client_result: str) -> BuildArtifact:
        client = MagicMock(spec=OpenClawClient)
        client.is_available.return_value = True
        client.send.return_value = client_result
        hc = HouseC(knowledge_graph=_make_graph(), openclaw_client=client, workspace_dir=str(tmp_path))
        artifact = BuildArtifact(sso=_make_sso())
        return hc._execute_browser_task(artifact, client)

    def test_browser_operational_finding_fails_validation(self, tmp_path):
        """'FINDING: Browser operational — loaded: Google' must not pass as a real result."""
        result = self._run_with_result(tmp_path, "FINDING: Browser operational — loaded: Google")
        assert result.passed_validation is False, (
            "Status message 'Browser operational' must be rejected, not stored as a finding"
        )

    def test_browser_loaded_finding_fails_validation(self, tmp_path):
        result = self._run_with_result(tmp_path, "FINDING: Browser loaded successfully")
        assert result.passed_validation is False

    def test_page_loaded_finding_fails_validation(self, tmp_path):
        result = self._run_with_result(tmp_path, "FINDING: Page loaded: google.com")
        assert result.passed_validation is False

    def test_navigation_complete_fails_validation(self, tmp_path):
        result = self._run_with_result(tmp_path, "FINDING: Navigation complete — at google.com")
        assert result.passed_validation is False

    def test_status_only_finding_has_validation_error(self, tmp_path):
        result = self._run_with_result(tmp_path, "FINDING: Browser operational — loaded: Google")
        assert len(result.validation_errors) > 0


# ═══════════════════════════════════════════════════════════════
#  9-11. Real findings accepted
# ═══════════════════════════════════════════════════════════════

class TestRealFindingsAccepted:

    def _run_with_result(self, tmp_path, client_result: str) -> BuildArtifact:
        client = MagicMock(spec=OpenClawClient)
        client.is_available.return_value = True
        client.send.return_value = client_result
        hc = HouseC(knowledge_graph=_make_graph(), openclaw_client=client, workspace_dir=str(tmp_path))
        artifact = BuildArtifact(sso=_make_sso())
        return hc._execute_browser_task(artifact, client)

    def test_finding_with_url_passes(self, tmp_path):
        result = self._run_with_result(
            tmp_path,
            "FINDING: Senior Python Dev | https://upwork.com/job/abc | $150/hr | Remote, 40h/week"
        )
        assert result.passed_validation is True

    def test_finding_with_price_passes(self, tmp_path):
        result = self._run_with_result(
            tmp_path,
            "FINDING: AI automation contract paying $200/hr on Toptal — apply by Friday"
        )
        assert result.passed_validation is True

    def test_finding_with_budget_passes(self, tmp_path):
        result = self._run_with_result(
            tmp_path,
            "FINDING: Flask API project — budget $3,000 fixed — HackerNews thread April 2026"
        )
        assert result.passed_validation is True

    def test_multiple_findings_passes(self, tmp_path):
        result = self._run_with_result(
            tmp_path,
            "FINDING: Python gig | https://upwork.com/1 | $120/hr\n"
            "FINDING: Data pipeline contract | https://freelancer.com/2 | $95/hr"
        )
        assert result.passed_validation is True
        assert result.execution_proof is not None

    def test_real_finding_sets_execution_proof(self, tmp_path):
        proof_text = "FINDING: Python contract | https://upwork.com/job/123 | $130/hr | Full remote"
        result = self._run_with_result(tmp_path, proof_text)
        assert result.execution_proof == proof_text

    def test_empty_result_still_fails(self, tmp_path):
        result = self._run_with_result(tmp_path, "")
        assert result.passed_validation is False

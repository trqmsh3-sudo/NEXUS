"""TDD tests for StrategicAgent — adaptive task selection layer.

Coverage:
  1.  next_task() returns a non-empty task string
  2.  next_task() rotates through the portfolio (not always the same task)
  3.  record_outcome() persists success/failure to state file
  4.  after MAX_DOMAIN_FAILURES consecutive failures on a domain, that domain
      is skipped in the next pick
  5.  domain skip resets after a success on that domain
  6.  all domains exhausted → still returns a task (fallback to least-failed)
  7.  state file missing → no crash, returns task from fresh state
  8.  state file corrupt → no crash, falls back to fresh state
  9.  task strings contain actionable keywords (remoteok, weworkremotely, etc.)
 10.  consecutive_failures counter increments per domain on failure
 11.  next_task never returns empty string
 12.  portfolio has at least 6 distinct tasks
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from nexus.core.strategic_agent import StrategicAgent


def _make_agent(tmp_path: Path) -> StrategicAgent:
    state_file = tmp_path / "strategy_state.json"
    return StrategicAgent(state_file=state_file)


class TestNextTask:
    def test_returns_non_empty_string(self, tmp_path):
        agent = _make_agent(tmp_path)
        task = agent.next_task()
        assert isinstance(task, str)
        assert len(task) > 10

    def test_never_returns_empty(self, tmp_path):
        agent = _make_agent(tmp_path)
        for _ in range(20):
            task = agent.next_task()
            assert task, "next_task() returned empty string"

    def test_rotates_through_portfolio(self, tmp_path):
        agent = _make_agent(tmp_path)
        tasks = [agent.next_task() for _ in range(len(agent.TASK_PORTFOLIO) + 1)]
        assert len(set(tasks)) > 1, "next_task() returned the same task every time"

    def test_task_contains_actionable_keywords(self, tmp_path):
        agent = _make_agent(tmp_path)
        task = agent.next_task()
        actionable = any(
            kw in task.lower()
            for kw in ["find", "search", "fetch", "get", "remoteok", "weworkremotely",
                       "remote", "gig", "job", "listing"]
        )
        assert actionable, f"Task lacks actionable keywords: {task!r}"

    def test_portfolio_has_at_least_6_entries(self, tmp_path):
        agent = _make_agent(tmp_path)
        assert len(agent.TASK_PORTFOLIO) >= 6


class TestRecordOutcome:
    def test_persists_to_state_file(self, tmp_path):
        agent = _make_agent(tmp_path)
        task = agent.next_task()
        agent.record_outcome(task, success=False)
        state = json.loads(agent.state_file.read_text())
        assert "domain_failures" in state

    def test_consecutive_failures_increments(self, tmp_path):
        agent = _make_agent(tmp_path)
        task = agent.next_task()
        domain = agent._task_domain(task)
        agent.record_outcome(task, success=False)
        agent.record_outcome(task, success=False)
        state = json.loads(agent.state_file.read_text())
        assert state["domain_failures"].get(domain, 0) >= 2

    def test_success_resets_failures(self, tmp_path):
        agent = _make_agent(tmp_path)
        task = agent.next_task()
        domain = agent._task_domain(task)
        agent.record_outcome(task, success=False)
        agent.record_outcome(task, success=False)
        agent.record_outcome(task, success=True)
        state = json.loads(agent.state_file.read_text())
        assert state["domain_failures"].get(domain, 0) == 0


class TestDomainSkipping:
    def test_domain_skipped_after_max_failures(self, tmp_path):
        agent = _make_agent(tmp_path)
        # Force failures on the first domain until it should be skipped
        first_task = agent.TASK_PORTFOLIO[0]
        task_str, domain, _ = first_task
        for _ in range(agent.MAX_DOMAIN_FAILURES):
            agent.record_outcome(task_str, success=False)

        # Next task should not come from the exhausted domain
        # Run enough picks to cover one full rotation
        seen_domains = set()
        for _ in range(len(agent.TASK_PORTFOLIO) * 2):
            t = agent.next_task()
            seen_domains.add(agent._task_domain(t))

        # The exhausted domain should NOT appear in the results
        assert domain not in seen_domains, \
            f"Domain {domain!r} still appears after {agent.MAX_DOMAIN_FAILURES} consecutive failures"

    def test_all_domains_exhausted_still_returns_task(self, tmp_path):
        agent = _make_agent(tmp_path)
        # Exhaust every domain
        for task_str, domain, _ in agent.TASK_PORTFOLIO:
            for _ in range(agent.MAX_DOMAIN_FAILURES):
                agent.record_outcome(task_str, success=False)
        task = agent.next_task()
        assert task, "next_task() returned empty when all domains exhausted"

    def test_success_unblocks_domain(self, tmp_path):
        agent = _make_agent(tmp_path)
        first_task_str, domain, _ = agent.TASK_PORTFOLIO[0]
        for _ in range(agent.MAX_DOMAIN_FAILURES):
            agent.record_outcome(first_task_str, success=False)
        # Now record a success
        agent.record_outcome(first_task_str, success=True)

        # Domain should be selectable again
        seen_domains = set()
        for _ in range(len(agent.TASK_PORTFOLIO) * 3):
            t = agent.next_task()
            seen_domains.add(agent._task_domain(t))
        assert domain in seen_domains, \
            f"Domain {domain!r} was not unblocked after success"


class TestStateFileFaults:
    def test_missing_state_file_no_crash(self, tmp_path):
        state_file = tmp_path / "nonexistent.json"
        agent = StrategicAgent(state_file=state_file)
        task = agent.next_task()
        assert task

    def test_corrupt_state_file_no_crash(self, tmp_path):
        state_file = tmp_path / "corrupt.json"
        state_file.write_text("not valid json {{{{")
        agent = StrategicAgent(state_file=state_file)
        task = agent.next_task()
        assert task

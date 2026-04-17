"""NEXUS entry point — boots the full system and runs the demo loop."""

from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (one directory above this file)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from nexus.core.text_utils import clean_text
from nexus.core.guardian import Guardian, GuardianReport, migrate_key_to_vault
from nexus.core.house_b import HouseB
from nexus.core.house_c import HouseC
from nexus.core.house_d import HouseD
from nexus.core.house_omega import CycleResult, HouseOmega, SystemHealth
from nexus.core.knowledge_graph import KnowledgeGraph
from nexus.core.model_router import ModelRouter
from nexus.core.proxy_mission import PROXY_MISSION_BELIEFS
from nexus.core.architecture_beliefs import ARCHITECTURE_BELIEFS
from nexus.core.identity_manager import IdentityManager
from nexus.core.proposal_sender import ProposalSender
from nexus.core.telegram_relay import TelegramRelay

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)-8s %(name)s: %(message)s",
)


# ------------------------------------------------------------------
# Banner
# ------------------------------------------------------------------

BANNER: str = r"""
================================================================
    _   _  _____  __  __  _   _  ____
   | \ | || ____| \ \/ / | | | |/ ___|
   |  \| ||  _|   \  /  | | | |\___ \
   | |\  || |___  /  \  | |_| | ___) |
   |_| \_||_____|/_/\_\  \___/ |____/

   Knowledge Reasoning System - All Houses Online
================================================================
"""


# ------------------------------------------------------------------
# fail_fast_on_critical_findings
# ------------------------------------------------------------------

def fail_fast_on_critical_findings(report: GuardianReport) -> None:
    """Abort startup if the Guardian audit found any CRITICAL secrets.

    Args:
        report: The GuardianReport produced by Guardian.audit().

    Raises:
        SystemExit: If the report contains one or more CRITICAL secret findings.
    """
    critical = [f for f in report.secret_findings if f.severity == "CRITICAL"]
    if critical:
        logging.critical(
            "GUARDIAN: %d CRITICAL secret(s) exposed — aborting NEXUS startup. "
            "Remove plaintext credentials before restarting.",
            len(critical),
        )
        raise SystemExit(
            f"GUARDIAN audit failed: {len(critical)} CRITICAL secret(s) found. "
            "Migrate credentials to the vault and remove them from source files."
        )


# ------------------------------------------------------------------
# build_nexus
# ------------------------------------------------------------------

def build_nexus(
    *,
    guardian_vault_path: str | None = None,
    guardian_master_key: str | None = None,
    guardian_scan_paths: list[str] | None = None,
) -> HouseOmega:
    """Initialise the complete NEXUS system.

    Creates the shared KnowledgeGraph, boots every House, wires
    them into House Omega, and injects the nine PROXY mission axioms
    as the foundational belief set.

    Guardian runs a security audit before any LLM components are created.
    If CRITICAL secrets are found in the scan paths, startup is aborted.

    Args:
        guardian_vault_path: Path to the encrypted vault file.
            Defaults to the ``NEXUS_VAULT_PATH`` env var, then
            ``"data/guardian_vault.enc"``.
        guardian_master_key: Master key for the vault.
            Defaults to the ``NEXUS_VAULT_KEY`` env var.
        guardian_scan_paths: Directories to scan for exposed secrets.
            Defaults to ``["nexus/", "scripts/"]``.

    Returns:
        A fully initialised HouseOmega ready to accept cycles.

    Raises:
        SystemExit: If Guardian finds CRITICAL exposed secrets.
        ValueError: If no master key is available for the vault.
    """
    # ── 1. Guardian security check ────────────────────────────
    vault_path = (
        guardian_vault_path
        or os.getenv("NEXUS_VAULT_PATH")
        or "data/guardian_vault.enc"
    )
    master_key = guardian_master_key or os.getenv("NEXUS_VAULT_KEY")
    scan_paths = guardian_scan_paths  # None → Guardian uses its own defaults

    guardian = Guardian(
        vault_path=vault_path,
        master_key=master_key,
        scan_paths=scan_paths,
    )

    # Migrate DEEPSEEK_API_KEY from .env into vault on first run.
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        try:
            migrate_key_to_vault(_env_path, guardian.vault, "DEEPSEEK_API_KEY")
        except KeyError:
            pass  # Key absent or already migrated — nothing to do.

    report = guardian.audit()
    fail_fast_on_critical_findings(report)

    # ── 2. NEXUS components ────────────────────────────────────
    graph = KnowledgeGraph()
    loaded = graph.persistence.last_load_count
    print(f"  Beliefs loaded from disk: {loaded}")

    router = ModelRouter(vault=guardian.vault)

    identity_manager = IdentityManager(data_dir="data", vault=guardian.vault)
    telegram_relay = TelegramRelay.from_env()
    proposal_sender = ProposalSender(
        router=router,
        identity_manager=identity_manager,
        telegram=telegram_relay,
    )

    house_b = HouseB(knowledge_graph=graph, router=router)
    house_c = HouseC(
        knowledge_graph=graph,
        router=router,
        proposal_sender=proposal_sender,
    )
    house_d = HouseD(knowledge_graph=graph, router=router, min_cycles=1)

    omega = HouseOmega(
        knowledge_graph=graph,
        house_b=house_b,
        house_c=house_c,
        house_d=house_d,
        sleep_cycle_interval=50,
    )

    print("  Injecting PROXY mission axioms...")
    result = graph.inject_external_signal(PROXY_MISSION_BELIEFS)
    print(f"  New beliefs added: {result['added']}  Rejected: {result['rejected']}")

    print("  Injecting architecture beliefs...")
    arch_result = graph.inject_external_signal(ARCHITECTURE_BELIEFS)
    print(f"  Architecture beliefs added: {arch_result['added']}  Rejected: {arch_result['rejected']}")
    print()

    print("  Knowledge Graph seeded:")
    for belief in graph:
        print(f"    [{belief.domain}] {belief.claim} (conf={belief.confidence})")
    print()

    return omega


# ------------------------------------------------------------------
# print helpers
# ------------------------------------------------------------------

def _print_separator(label: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {label}")
    print(f"{'-' * 60}")


def _print_cycle_result(i: int, result: CycleResult) -> None:
    """Print a human-readable summary of a single cycle."""
    _print_separator(f"CYCLE {i} RESULT")

    status = "SUCCESS" if result.success else "FAILED"
    print(f"  Status:    {status}")
    print(f"  Input:     {result.user_input}")
    print(f"  Time:      {result.cycle_time_seconds:.2f}s")

    if result.sso:
        print(f"  Redefined: {result.sso.redefined_problem[:80]}")
        print(f"  Domain:    {result.sso.domain}")

    if result.refinement_attempts > 0:
        print(f"  Refined:   {result.refinement_attempts} attempt(s)")

    if result.destruction_report:
        dr = result.destruction_report
        print(f"  House D:   survived={dr.survived}  "
              f"score={dr.survival_score:.2f}  "
              f"rec={dr.recommendation}")
        if not dr.survived:
            print(f"  >> KILLED BY HOUSE D: {dr.recommendation}")

    if result.build_result:
        br = result.build_result
        print(f"  Build:     success={br.success}  "
              f"ready={br.ready_for_house_a}")

    if result.belief_added:
        print(f"  >> BELIEF ADDED TO HOUSE A")
    elif result.failure_reason:
        from nexus.core.text_utils import clean_text
        print(f"  Reason:    {clean_text(result.failure_reason)}")

    print()


def _print_health(health: SystemHealth) -> None:
    """Print the system health report."""
    _print_separator("SYSTEM HEALTH REPORT")
    print(f"  Total cycles:      {health.total_cycles}")
    print(f"  Successful:        {health.successful_cycles}")
    print(f"  Failed:            {health.failed_cycles}")
    print(f"  System score:      {health.system_score:.2%}")
    print(f"  Total beliefs:     {health.total_beliefs}")
    print(f"  Autonomy ratio:    {health.autonomy_ratio:.2%}")
    print(f"  Daily LLM cost:    ${health.daily_cost:.4f}")
    print(f"  Counterfactuals:   {health.total_counterfactuals}")
    print(f"  Wrong predictions: {health.wrong_predictions}")
    print(f"  Avg cycle time:    {health.average_cycle_time:.2f}s")
    print(f"  Domains:           {', '.join(health.domains_covered) or '(none)'}")
    if health.last_sleep_cycle:
        print(f"  Last sleep:        {health.last_sleep_cycle.isoformat()}")
    else:
        print(f"  Last sleep:        (not yet)")
    print()


# ------------------------------------------------------------------
# run_demo
# ------------------------------------------------------------------

def run_demo(
    nexus: HouseOmega,
    inputs: list[str] | None = None,
) -> None:
    """Run the NEXUS demo with sample cycles.

    Args:
        nexus: A fully initialised HouseOmega instance.
        inputs: Optional list of user inputs. If None, uses default 3-cycle demo.
    """
    if inputs is None:
        inputs = [
        "Build a Python function called validate_email that takes a "
        "string and returns True if it is a valid email address and "
        "False otherwise. Must handle plus-addressing like user+tag@domain.com "
        "and subdomains like user@mail.corp.example.com. "
        "Use only the re module from Python standard library. "
        "No external dependencies.",
        "Build a Python class called TokenBucket that implements "
        "rate limiting. Constructor takes max_tokens (int) and "
        "refill_rate (float tokens per second). Methods: consume() "
        "returns True if a token is available and False if rate limited, "
        "and get_tokens() returns current token count as float. "
        "Use only time module. No external dependencies.",
        "Build a Python class called JsonValidator that validates "
        "dictionaries. Constructor takes a schema dict mapping field "
        "names to Python types. Method validate(data: dict) returns "
        "a dict with 'valid' (bool) and 'errors' (list of strings). "
        "Use only Python standard library. No external dependencies.",
    ]

    _print_separator("STARTING NEXUS DEMO - 3 CYCLES")

    for i, user_input in enumerate(inputs, 1):
        print(f"\n  >>> Cycle {i}: \"{user_input}\"")
        result = nexus.run(user_input)
        _print_cycle_result(i, result)

    _print_health(nexus.get_health())

    _print_separator("KNOWLEDGE GRAPH - FINAL STATE")
    if len(nexus.knowledge_graph) == 0:
        print("  (empty)")
    else:
        for belief in nexus.knowledge_graph:
            valid = "valid" if belief.is_valid() else "invalid"
            print(f"  [{belief.domain}] {belief.claim}")
            print(f"    confidence={belief.confidence}  {valid}  "
                  f"source={belief.source}")
    print()


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def main() -> int:
    """Boot NEXUS and run the demo.

    If command-line args are provided, runs a single cycle with that input.
    Otherwise runs the default 3-cycle demo.

    Returns:
        Exit code (0 for success).
    """
    # Ensure UTF-8 output on Windows (safe here — not at import time)
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

    print(BANNER)

    print("  Booting NEXUS...")
    nexus = build_nexus()

    if len(sys.argv) > 1:
        inputs = sys.argv[1:]
        health_before = nexus.get_health()
        _print_separator("AUTONOMY RATIO (BEFORE)")
        print(f"  autonomy_ratio: {health_before.autonomy_ratio:.2%}")
        print(f"  self_built_beliefs: {health_before.self_built_beliefs}")
        print(f"  total_beliefs: {health_before.total_beliefs}")
        print()

        _print_separator("RUNNING CYCLES")
        for i, user_input in enumerate(inputs, 1):
            print(f"\n  >>> Cycle {i}: \"{user_input}\"")
            result = nexus.run(user_input)
            _print_cycle_result(i, result)

        _print_health(nexus.get_health())
        _print_separator("KNOWLEDGE GRAPH - FINAL STATE")
        if len(nexus.knowledge_graph) == 0:
            print("  (empty)")
        else:
            for belief in nexus.knowledge_graph:
                valid = "valid" if belief.is_valid() else "invalid"
                print(f"  [{belief.domain}] {belief.claim}")
                print(f"    confidence={belief.confidence}  {valid}  source={belief.source}")
        print()
    else:
        run_demo(nexus)

    _print_separator("FINAL HEALTH REPORT")
    health = nexus.get_health()
    print(f"  Score:   {health.system_score:.2%}")
    print(f"  Cycles:  {health.total_cycles}")
    print(f"  Beliefs: {health.total_beliefs}")
    beliefs_added = sum(1 for c in nexus.cycle_history if c.belief_added)
    print(f"  Beliefs added this run: {beliefs_added}")
    self_built_added = sum(
        1 for c in nexus.cycle_history if c.belief_added
    )  # each belief_added is a self-built artifact
    print(f"  Self-built beliefs added: {self_built_added}")
    print(f"  Autonomy ratio (after):  {health.autonomy_ratio:.2%}")
    total_time = sum(c.cycle_time_seconds for c in nexus.cycle_history)
    print(f"  Total cycle time: {total_time:.2f}s")
    if len(sys.argv) > 1:
        print(f"  Tasks succeeded: {sum(1 for c in nexus.cycle_history[-len(sys.argv[1:]):] if c.success)}/{len(sys.argv[1:])}")
        for i, (inp, c) in enumerate(zip(sys.argv[1:], nexus.cycle_history[-len(sys.argv[1:]):]), 1):
            status = "OK" if c.success else "FAILED"
            print(f"    {i}. [{status}] {inp[:60]}...")
    print(f"  Domains: {', '.join(health.domains_covered) or '(none)'}")
    print()

    nexus.knowledge_graph.persistence.save(nexus.knowledge_graph)
    saved = len([b for b in nexus.knowledge_graph.beliefs_snapshot()
                 if b.is_valid() and not b.is_expired()])
    print(f"  Saved to disk: {saved} beliefs")
    print()

    router = nexus.house_b.router
    if router.call_log:
        _print_separator("MODEL USAGE (per House call)")
        for i, (house, model, elapsed, ok) in enumerate(router.call_log, 1):
            status = "ok" if ok else "FAILED"
            print(f"  {i}. {house}  ->  {model}  ({elapsed}s)  [{status}]")
        print()

    print("  NEXUS shutdown complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

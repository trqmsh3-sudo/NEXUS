"""NEXUS entry point — boots the full system and runs the demo loop."""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from nexus.core.belief_certificate import BeliefCertificate  # noqa: E402
from nexus.core.text_utils import clean_text
from nexus.core.house_b import HouseB
from nexus.core.house_c import HouseC
from nexus.core.house_d import HouseD
from nexus.core.house_omega import CycleResult, HouseOmega, SystemHealth
from nexus.core.knowledge_graph import KnowledgeGraph
from nexus.core.model_router import ModelRouter

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
# build_nexus
# ------------------------------------------------------------------

def build_nexus() -> HouseOmega:
    """Initialise the complete NEXUS system.

    Creates the shared KnowledgeGraph, boots every House, wires
    them into House Omega, and injects three foundational beliefs
    as the initial external signal.

    Returns:
        A fully initialised HouseOmega ready to accept cycles.
    """
    graph = KnowledgeGraph()
    loaded = graph.persistence.last_load_count
    print(f"  Beliefs loaded from disk: {loaded}")

    router = ModelRouter()
    house_b = HouseB(knowledge_graph=graph, router=router)
    house_c = HouseC(knowledge_graph=graph, router=router)
    house_d = HouseD(knowledge_graph=graph, router=router, min_cycles=1)

    omega = HouseOmega(
        knowledge_graph=graph,
        house_b=house_b,
        house_c=house_c,
        house_d=house_d,
        sleep_cycle_interval=50,
    )

    starter_beliefs = [
        BeliefCertificate(
            claim="Clean code is better than clever code",
            source="NEXUS founding axiom",
            confidence=0.9,
            domain="Software Engineering",
            executable_proof="print('clean')",
            decay_rate=0.05,
        ),
        BeliefCertificate(
            claim="Tests must run before code is trusted",
            source="NEXUS founding axiom",
            confidence=0.95,
            domain="Software Engineering",
            executable_proof="assert True",
            decay_rate=0.05,
        ),
        BeliefCertificate(
            claim="Every system needs a kill switch",
            source="NEXUS founding axiom",
            confidence=0.99,
            domain="System Architecture",
            executable_proof="assert True",
            decay_rate=0.05,
        ),
    ]

    print("  Injecting starter beliefs...")
    result = graph.inject_external_signal(starter_beliefs)
    print(f"  New beliefs added: {result['added']}  Rejected: {result['rejected']}")
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
    saved = len([b for b in nexus.knowledge_graph.beliefs.values()
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

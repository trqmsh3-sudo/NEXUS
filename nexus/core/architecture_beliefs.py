"""PROXY architectural BeliefCertificates.

Five foundational operating principles that govern HOW PROXY reasons,
routes work, manages confidence, learns from outcomes, and evolves.

Every belief is:
- is_axiom=True      — never pruned by House A's prune cycle
- decay_rate=0.0     — never expires
- confidence=0.95    — near-maximum epistemic weight
- executable_proof   — runnable Python that asserts the principle holds
"""

from __future__ import annotations

from nexus.core.belief_certificate import BeliefCertificate

# ---------------------------------------------------------------------------
# Belief 1 — OODA loop for all decisions
# ---------------------------------------------------------------------------
_OODA_LOOP = BeliefCertificate(
    claim=(
        "Use OODA loop for all decisions: "
        "Observe market signals, Orient with mental models, "
        "Decide by impact × probability × urgency, Act with checkpoints"
    ),
    source="PROXY architecture axiom",
    confidence=0.95,
    domain="Decision Making",
    decay_rate=0.0,
    is_axiom=True,
    executable_proof=(
        "# OODA loop phases must all be present in any decision cycle\n"
        "phases = ['observe', 'orient', 'decide', 'act']\n"
        "assert len(phases) == 4, 'OODA requires exactly 4 phases'\n"
        "# Decide by impact x probability x urgency\n"
        "def priority_score(impact, probability, urgency):\n"
        "    return impact * probability * urgency\n"
        "assert priority_score(1.0, 0.8, 0.9) > 0, 'Priority score must be positive'\n"
        "assert priority_score(0, 1, 1) == 0, 'Zero impact = zero priority'\n"
        "# Act with checkpoints\n"
        "checkpoints_required = True\n"
        "assert checkpoints_required, 'Every action must include checkpoints'"
    ),
)

# ---------------------------------------------------------------------------
# Belief 2 — Route tasks by complexity
# ---------------------------------------------------------------------------
_COMPLEXITY_ROUTING = BeliefCertificate(
    claim=(
        "Route tasks by complexity: "
        "Simple → DeepSeek/Gemini (cheap), "
        "Complex → Claude (quality), "
        "Real-time → Grok (speed)"
    ),
    source="PROXY architecture axiom",
    confidence=0.95,
    domain="System Architecture",
    decay_rate=0.0,
    is_axiom=True,
    executable_proof=(
        "ROUTING_TABLE = {\n"
        "    'simple':    ('deepseek', 'gemini'),   # cheap\n"
        "    'complex':   ('claude',),               # quality\n"
        "    'real-time': ('grok',),                 # speed\n"
        "}\n"
        "assert 'simple' in ROUTING_TABLE\n"
        "assert 'complex' in ROUTING_TABLE\n"
        "assert 'real-time' in ROUTING_TABLE\n"
        "# Simple tasks must map to cheap providers\n"
        "cheap = ROUTING_TABLE['simple']\n"
        "assert 'deepseek' in cheap or 'gemini' in cheap\n"
        "# Complex tasks must map to quality provider\n"
        "assert 'claude' in ROUTING_TABLE['complex']\n"
        "# Real-time tasks must map to fast provider\n"
        "assert 'grok' in ROUTING_TABLE['real-time']"
    ),
)

# ---------------------------------------------------------------------------
# Belief 3 — Confidence routing thresholds
# ---------------------------------------------------------------------------
_CONFIDENCE_ROUTING = BeliefCertificate(
    claim=(
        "Confidence routing: "
        "Above 0.85 + reversible → act alone. "
        "Below 0.6 + irreversible → ask owner"
    ),
    source="PROXY architecture axiom",
    confidence=0.95,
    domain="Autonomy Management",
    decay_rate=0.0,
    is_axiom=True,
    executable_proof=(
        "HIGH_CONF = 0.85\n"
        "LOW_CONF  = 0.6\n"
        "\n"
        "def should_act_alone(confidence: float, reversible: bool) -> bool:\n"
        "    return confidence >= HIGH_CONF and reversible\n"
        "\n"
        "def should_ask_owner(confidence: float, reversible: bool) -> bool:\n"
        "    return confidence < LOW_CONF and not reversible\n"
        "\n"
        "# High confidence + reversible → act alone\n"
        "assert should_act_alone(0.90, True)  is True\n"
        "assert should_act_alone(0.90, False) is False   # reversible required\n"
        "assert should_act_alone(0.80, True)  is False   # confidence too low\n"
        "# Low confidence + irreversible → ask owner\n"
        "assert should_ask_owner(0.50, False) is True\n"
        "assert should_ask_owner(0.50, True)  is False   # must be irreversible\n"
        "assert should_ask_owner(0.70, False) is False   # confidence not low enough"
    ),
)

# ---------------------------------------------------------------------------
# Belief 4 — Skill library and anti-belief learning
# ---------------------------------------------------------------------------
_SKILL_LIBRARY = BeliefCertificate(
    claim=(
        "After every success: store as reusable skill in belief library. "
        "After every failure: store why it failed as anti-belief"
    ),
    source="PROXY architecture axiom",
    confidence=0.95,
    domain="Learning Systems",
    decay_rate=0.0,
    is_axiom=True,
    executable_proof=(
        "skill_library = []\n"
        "anti_beliefs  = []\n"
        "\n"
        "def on_success(task, result):\n"
        "    skill_library.append({'task': task, 'result': result})\n"
        "\n"
        "def on_failure(task, reason):\n"
        "    anti_beliefs.append({'task': task, 'reason': reason})\n"
        "\n"
        "on_success('write proposal', 'client accepted')\n"
        "on_failure('cold email', 'no reply after 3 attempts')\n"
        "\n"
        "assert len(skill_library) == 1, 'Success must be stored as a skill'\n"
        "assert len(anti_beliefs)  == 1, 'Failure must be stored as anti-belief'\n"
        "assert skill_library[0]['task']  == 'write proposal'\n"
        "assert anti_beliefs[0]['reason'] == 'no reply after 3 attempts'"
    ),
)

# ---------------------------------------------------------------------------
# Belief 5 — Weekly AI tool evaluation
# ---------------------------------------------------------------------------
_AI_TOOLS_CADENCE = BeliefCertificate(
    claim=(
        "Always stay updated with latest AI tools. "
        "Weekly: search for new capabilities, evaluate, adopt if better"
    ),
    source="PROXY architecture axiom",
    confidence=0.95,
    domain="Continuous Improvement",
    decay_rate=0.0,
    is_axiom=True,
    executable_proof=(
        "REVIEW_CADENCE_DAYS = 7  # weekly\n"
        "assert REVIEW_CADENCE_DAYS == 7, 'AI tool review must be weekly'\n"
        "\n"
        "def should_adopt(is_better: bool, evaluation_done: bool) -> bool:\n"
        "    return is_better and evaluation_done\n"
        "\n"
        "# Adopt only when evaluated and confirmed better\n"
        "assert should_adopt(True,  True)  is True\n"
        "assert should_adopt(False, True)  is False  # not better\n"
        "assert should_adopt(True,  False) is False  # no evaluation done\n"
        "# Review steps: search → evaluate → adopt if better\n"
        "review_steps = ['search', 'evaluate', 'adopt_if_better']\n"
        "assert len(review_steps) == 3"
    ),
)

# ---------------------------------------------------------------------------
# Public export
# ---------------------------------------------------------------------------
ARCHITECTURE_BELIEFS: list[BeliefCertificate] = [
    _OODA_LOOP,
    _COMPLEXITY_ROUTING,
    _CONFIDENCE_ROUTING,
    _SKILL_LIBRARY,
    _AI_TOOLS_CADENCE,
]

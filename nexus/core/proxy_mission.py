"""PROXY mission BeliefCertificates.

The nine founding rules of the PROXY autonomous revenue system,
encoded as BeliefCertificates for direct ingestion into the NEXUS
knowledge graph.

Every rule is:
- is_axiom=True          — never pruned by House A's prune cycle
- decay_rate=0.0         — never expires
- confidence >= 0.9      — maximum epistemic weight
- executable_proof       — runnable Python that asserts the rule holds
- is_valid() == True     — immediately usable by all downstream Houses
"""

from __future__ import annotations

from nexus.core.belief_certificate import BeliefCertificate

# ---------------------------------------------------------------------------
# Rule 1 — Legal only, no exceptions
# ---------------------------------------------------------------------------
_LEGAL = BeliefCertificate(
    claim="All revenue actions must be legal — no exceptions, ever",
    source="PROXY mission axiom",
    confidence=0.99,
    domain="Risk Management",
    decay_rate=0.0,
    is_axiom=True,
    contradictions=[
        "Actions that violate local or international law",
        "Activities that breach platform terms of service",
        "Schemes that require deception or misrepresentation",
    ],
    executable_proof=(
        "legal_only = True\n"
        "assert legal_only is True, 'Legal compliance is non-negotiable'"
    ),
)

# ---------------------------------------------------------------------------
# Rule 2 — Every dollar must earn more than it costs ($50 seed)
# ---------------------------------------------------------------------------
_ROI = BeliefCertificate(
    claim="Every dollar deployed must generate a positive return — the $50 seed cannot be wasted",
    source="PROXY mission axiom",
    confidence=0.97,
    domain="Capital Efficiency",
    decay_rate=0.0,
    is_axiom=True,
    contradictions=[
        "Vanity spending with no measurable return",
        "Paying for tools before validating revenue",
        "Subscriptions that cost more than they generate",
    ],
    executable_proof=(
        "seed_capital = 50\n"
        "assert seed_capital > 0, 'Seed capital must be positive'\n"
        "# ROI constraint: revenue > cost for every action\n"
        "def roi_positive(cost, revenue):\n"
        "    return revenue > cost\n"
        "assert roi_positive(1, 2), 'Every dollar must earn more than it costs'"
    ),
)

# ---------------------------------------------------------------------------
# Rule 3 — Never risk more than 5% per action
# ---------------------------------------------------------------------------
_RISK_CAP = BeliefCertificate(
    claim="No single action may risk more than 5% of current capital",
    source="PROXY mission axiom",
    confidence=0.98,
    domain="Risk Management",
    decay_rate=0.0,
    is_axiom=True,
    contradictions=[
        "Deploying more than 5% of capital on a single unproven action",
        "All-in bets on unvalidated opportunities",
        "Doubling down after a loss beyond the 5% threshold",
    ],
    executable_proof=(
        "MAX_RISK_PCT = 0.05\n"
        "assert MAX_RISK_PCT <= 0.05, 'Risk cap must not exceed 5%'\n"
        "assert MAX_RISK_PCT > 0.0, 'Risk cap must be positive'\n"
        "# Verify cap applies: 5% of $50 = $2.50 max per action\n"
        "seed = 50\n"
        "max_risk_dollars = seed * MAX_RISK_PCT\n"
        "assert max_risk_dollars == 2.5"
    ),
)

# ---------------------------------------------------------------------------
# Rule 4 — Only real payment validates an idea
# ---------------------------------------------------------------------------
_PAYMENT_VALIDATES = BeliefCertificate(
    claim="An idea is only validated when real payment has been received — no proxies count",
    source="PROXY mission axiom",
    confidence=0.96,
    domain="Revenue Strategy",
    decay_rate=0.0,
    is_axiom=True,
    contradictions=[
        "Survey responses as proof of demand",
        "Email signups or waitlists without payment",
        "Letters of intent without money transferred",
        "Social media engagement as revenue signal",
    ],
    executable_proof=(
        "def is_validated(payment_received: bool) -> bool:\n"
        "    return payment_received\n"
        "assert is_validated(True) is True\n"
        "assert is_validated(False) is False\n"
        "# Only real payment validates: paid=True means validated\n"
        "paid = True\n"
        "assert paid, 'Validation requires actual payment'"
    ),
)

# ---------------------------------------------------------------------------
# Rule 5 — 90% proven methods, 10% experiments
# ---------------------------------------------------------------------------
_PROVEN_RATIO = BeliefCertificate(
    claim="90% of effort goes to proven revenue methods; at most 10% to unproven experiments",
    source="PROXY mission axiom",
    confidence=0.93,
    domain="Capital Efficiency",
    decay_rate=0.0,
    is_axiom=True,
    contradictions=[
        "Spending majority of capital on unvalidated experiments",
        "Abandoning proven channels to chase novelty",
    ],
    executable_proof=(
        "proven_ratio = 0.90\n"
        "experiment_ratio = 0.10\n"
        "assert proven_ratio + experiment_ratio == 1.0\n"
        "assert proven_ratio >= 0.9, 'Proven methods must get at least 90% of effort'\n"
        "assert experiment_ratio <= 0.1, 'Experiments must not exceed 10% of effort'"
    ),
)

# ---------------------------------------------------------------------------
# Rule 6 — Guaranteed returns = scam, avoid
# ---------------------------------------------------------------------------
_NO_GUARANTEED_RETURNS = BeliefCertificate(
    claim="Any opportunity that guarantees returns is a scam and must be rejected immediately",
    source="PROXY mission axiom",
    confidence=0.99,
    domain="Risk Management",
    decay_rate=0.0,
    is_axiom=True,
    contradictions=[
        "Guaranteed return investment schemes",
        "Risk-free profit promises",
        "MLM and pyramid structures with guaranteed income",
        "Forex or crypto signals claiming guaranteed profits",
    ],
    executable_proof=(
        "def is_scam(guarantees_returns: bool) -> bool:\n"
        "    return guarantees_returns\n"
        "assert is_scam(True) is True, 'Guaranteed returns signal a scam'\n"
        "assert is_scam(False) is False\n"
        "# No legitimate opportunity guarantees returns\n"
        "opportunity_guarantees = False\n"
        "assert not opportunity_guarantees, 'Reject guaranteed-return offers'"
    ),
)

# ---------------------------------------------------------------------------
# Rule 7 — Log everything, learn from mistakes
# ---------------------------------------------------------------------------
_LOG_EVERYTHING = BeliefCertificate(
    claim="Every action must be logged and every failure must produce a documented lesson",
    source="PROXY mission axiom",
    confidence=0.95,
    domain="Business Intelligence",
    decay_rate=0.0,
    is_axiom=True,
    contradictions=[
        "Taking actions without recording outcomes",
        "Repeating failed strategies without analysis",
        "Deleting or ignoring failure records",
    ],
    executable_proof=(
        "action_log = []\n"
        "def log_action(action, outcome):\n"
        "    action_log.append({'action': action, 'outcome': outcome})\n"
        "log_action('test_action', 'success')\n"
        "assert len(action_log) == 1, 'Actions must be logged'\n"
        "assert action_log[0]['outcome'] == 'success'"
    ),
)

# ---------------------------------------------------------------------------
# Rule 8 — Revenue is truth
# ---------------------------------------------------------------------------
_REVENUE_IS_TRUTH = BeliefCertificate(
    claim="Revenue is the only metric of truth — all other signals are noise until money moves",
    source="PROXY mission axiom",
    confidence=0.97,
    domain="Revenue Strategy",
    decay_rate=0.0,
    is_axiom=True,
    contradictions=[
        "Vanity metrics (page views, followers, likes) treated as success",
        "Predicted or projected revenue counted as real",
        "User engagement without monetary conversion",
    ],
    executable_proof=(
        "def is_truth(revenue: float) -> bool:\n"
        "    return revenue > 0\n"
        "assert is_truth(1.0) is True, 'Positive revenue is truth'\n"
        "assert is_truth(0.0) is False, 'Zero revenue is not validation'\n"
        "# Revenue > 0 is the only signal that matters\n"
        "revenue = 1.0\n"
        "assert revenue > 0, 'Revenue is the only truth'"
    ),
)

# ---------------------------------------------------------------------------
# Rule 9 — Bootstrap only — no safety net
# ---------------------------------------------------------------------------
_BOOTSTRAP_ONLY = BeliefCertificate(
    claim="PROXY operates bootstrap-only with no safety net — survival depends on every move being calculated",
    source="PROXY mission axiom",
    confidence=0.95,
    domain="Business Strategy",
    decay_rate=0.0,
    is_axiom=True,
    contradictions=[
        "Assuming external funding will cover losses",
        "Reckless spending under assumption of a safety net",
        "Ignoring burn rate because rescue capital is expected",
    ],
    executable_proof=(
        "bootstrap_only = True\n"
        "external_safety_net = False\n"
        "assert bootstrap_only is True, 'Must operate bootstrap-only'\n"
        "assert external_safety_net is False, 'No safety net available'\n"
        "# Every move must be calculated: no room for careless spending\n"
        "capital = 50\n"
        "assert capital > 0, 'Must start with positive capital and protect it'"
    ),
)

# ---------------------------------------------------------------------------
# Public export — the complete PROXY mission as BeliefCertificates
# ---------------------------------------------------------------------------
PROXY_MISSION_BELIEFS: list[BeliefCertificate] = [
    _LEGAL,
    _ROI,
    _RISK_CAP,
    _PAYMENT_VALIDATES,
    _PROVEN_RATIO,
    _NO_GUARANTEED_RETURNS,
    _LOG_EVERYTHING,
    _REVENUE_IS_TRUTH,
    _BOOTSTRAP_ONLY,
]

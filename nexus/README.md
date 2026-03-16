# NEXUS

A knowledge reasoning system built around **BeliefCertificates** — auditable, time-decaying, machine-verifiable units of knowledge.

## Project structure

```
nexus/
├── core/
│   ├── belief_certificate.py   # BeliefCertificate dataclass
│   ├── knowledge_graph.py      # Directed graph of certificates
│   └── house_a.py              # Forward-chaining reasoning engine
├── utils/
│   └── validators.py           # Shared validation helpers
├── data/
│   └── knowledge_store/        # Persistent storage directory
├── tests/
│   └── test_house_a.py         # Full test suite
├── main.py                     # Demo entry point
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.11+
- No external dependencies beyond `pytest` for testing

## Quick start

```bash
# Install test dependencies
pip install -r requirements.txt

# Run the demo
python -m nexus.main

# Run the test suite
pytest nexus/tests/ -v
```

## Core concepts

### BeliefCertificate

The atomic unit of knowledge. Each certificate binds a human-readable **claim** to:

- A **source** (provenance)
- A **confidence** score (0.0–1.0)
- A **decay rate** controlling how fast the knowledge expires
- An optional **executable proof** — code that can programmatically verify the claim
- Known **contradictions** and **downstream dependents**

A certificate is *valid* only when `confidence > 0.5` **and** an executable proof is present.

### KnowledgeGraph

A directed graph where nodes are certificates. Edges are derived from `downstream_dependents` and `contradictions`. Supports querying by domain, filtering valid/expired certificates, and persistence via JSON.

### House-A

A forward-chaining reasoning engine that operates over the knowledge graph in three phases:

1. **Prune** — remove expired certificates
2. **Detect** — flag unresolved contradictions
3. **Propagate** — decay confidence on dependents of invalid beliefs

Every action is recorded in a serialisable audit log.

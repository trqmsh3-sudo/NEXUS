"""NEXUS core package.

Keep this module import-safe (no environment-variable requirements).
Heavy objects are available via lazy attribute access to avoid import-time
side effects where possible (ModelRouter logs if optional keys are missing).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__: list[str] = [
    "AttackResult",
    "AttackType",
    "BeliefCertificate",
    "BuildArtifact",
    "BuildResult",
    "CycleResult",
    "DestructionReport",
    "ExternalSignalProvider",
    "HouseB",
    "HouseC",
    "HouseD",
    "HouseOmega",
    "KnowledgeGraph",
    "ModelRouter",
    "PersistenceManager",
    "MinorityReport",
    "StructuredSpecificationObject",
    "SystemHealth",
]


_LAZY: dict[str, tuple[str, str]] = {
    "BeliefCertificate": ("nexus.core.belief_certificate", "BeliefCertificate"),
    "KnowledgeGraph": ("nexus.core.knowledge_graph", "KnowledgeGraph"),
    "PersistenceManager": ("nexus.core.persistence", "PersistenceManager"),
    "ExternalSignalProvider": ("nexus.core.external_signal", "ExternalSignalProvider"),
    "ModelRouter": ("nexus.core.model_router", "ModelRouter"),
    "HouseB": ("nexus.core.house_b", "HouseB"),
    "MinorityReport": ("nexus.core.house_b", "MinorityReport"),
    "StructuredSpecificationObject": ("nexus.core.house_b", "StructuredSpecificationObject"),
    "HouseC": ("nexus.core.house_c", "HouseC"),
    "BuildArtifact": ("nexus.core.house_c", "BuildArtifact"),
    "BuildResult": ("nexus.core.house_c", "BuildResult"),
    "HouseD": ("nexus.core.house_d", "HouseD"),
    "AttackResult": ("nexus.core.house_d", "AttackResult"),
    "AttackType": ("nexus.core.house_d", "AttackType"),
    "DestructionReport": ("nexus.core.house_d", "DestructionReport"),
    "HouseOmega": ("nexus.core.house_omega", "HouseOmega"),
    "CycleResult": ("nexus.core.house_omega", "CycleResult"),
    "SystemHealth": ("nexus.core.house_omega", "SystemHealth"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if not target:
        raise AttributeError(name)
    mod_name, attr = target
    mod = import_module(mod_name)
    value = getattr(mod, attr)
    globals()[name] = value
    return value


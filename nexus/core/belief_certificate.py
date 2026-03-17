"""Belief Certificate module for NEXUS knowledge validation.

Provides the BeliefCertificate dataclass — the atomic unit of knowledge
in the NEXUS system. Each certificate encapsulates a claim, its provenance,
confidence scoring, temporal decay, and an optional executable proof that
can programmatically verify the claim still holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from nexus.core.text_utils import clean_text


@dataclass
class BeliefCertificate:
    """An auditable certificate representing a single knowledge claim.

    A BeliefCertificate binds a human-readable claim to its source,
    a confidence score, known contradictions, temporal decay metadata,
    downstream dependents, and an optional executable proof.

    Attributes:
        claim: The knowledge statement this certificate represents.
        source: Origin of the claim (URL, paper, person, system, etc.).
        confidence: Trust level in the range [0.0, 1.0].
        contradictions: Known statements that conflict with this claim.
        decay_rate: Speed at which this knowledge expires, in [0.0, 1.0].
            A decay_rate of 0.0 means the knowledge never expires;
            1.0 means it expires almost immediately.
        created_at: UTC timestamp when the certificate was first created.
        last_verified: UTC timestamp of the most recent verification.
        downstream_dependents: IDs or descriptions of beliefs that
            logically depend on this certificate being valid.
        executable_proof: Optional Python code or test identifier that,
            when executed, proves the claim is still true.
        domain: Knowledge domain, e.g. "ML Architecture", "Security".
    """

    claim: str
    source: str
    confidence: float
    contradictions: list[str] = field(default_factory=list)
    decay_rate: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_verified: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    downstream_dependents: list[str] = field(default_factory=list)
    executable_proof: str | None = None
    domain: str = "General"
    attempts: list[dict[str, Any]] = field(default_factory=list)
    lessons_learned: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate field constraints after initialisation."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )
        if not 0.0 <= self.decay_rate <= 1.0:
            raise ValueError(
                f"decay_rate must be between 0.0 and 1.0, got {self.decay_rate}"
            )

    def is_valid(self) -> bool:
        """Determine whether this certificate meets minimum validity criteria.

        A certificate is valid when confidence exceeds 0.5 **and** an
        executable proof is attached. This ensures that high-confidence
        claims without machine-verifiable evidence are not trusted blindly.

        Returns:
            True if confidence > 0.5 and executable_proof is not None.
        """
        return self.confidence > 0.5 and self.executable_proof is not None

    def is_expired(self) -> bool:
        """Check whether this certificate has exceeded its time-to-live.

        The effective TTL in days is ``365 * (1 - decay_rate)``.  A
        decay_rate of 0.0 gives a full year; 0.9 gives ~36.5 days.

        Returns:
            True if the elapsed time since last verification exceeds
            the computed TTL.
        """
        now = datetime.now(timezone.utc)
        last = self.last_verified.replace(tzinfo=timezone.utc) if self.last_verified.tzinfo is None else self.last_verified
        elapsed_days: float = (now - last).total_seconds() / 86400
        ttl_days: float = 365.0 * (1.0 - self.decay_rate)
        return elapsed_days > ttl_days

    def to_dict(self) -> dict[str, Any]:
        """Serialise this certificate to a plain dictionary.

        Datetime fields are converted to ISO-8601 strings so the
        result is directly JSON-serialisable.

        Returns:
            A dictionary containing all certificate fields.
        """
        def _clean(val: Any) -> Any:
            if isinstance(val, str):
                return clean_text(val)
            if isinstance(val, list):
                return [_clean(v) for v in val]
            return val

        return {
            "claim": _clean(self.claim),
            "source": _clean(self.source),
            "confidence": self.confidence,
            "contradictions": _clean(list(self.contradictions)),
            "decay_rate": self.decay_rate,
            "created_at": self.created_at.isoformat(),
            "last_verified": self.last_verified.isoformat(),
            "downstream_dependents": _clean(list(self.downstream_dependents)),
            "executable_proof": _clean(self.executable_proof) if self.executable_proof else None,
            "domain": _clean(self.domain),
            "attempts": list(self.attempts),
            "lessons_learned": _clean(list(self.lessons_learned)),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BeliefCertificate:
        """Reconstruct a BeliefCertificate from a dictionary.

        This is the inverse of :meth:`to_dict`.  ISO-8601 datetime
        strings are parsed back into aware ``datetime`` objects.

        Args:
            data: Dictionary previously produced by ``to_dict()``.

        Returns:
            A fully hydrated BeliefCertificate instance.

        Raises:
            KeyError: If a required field is missing from *data*.
            ValueError: If field values are out of range.
        """
        attempts_raw = data.get("attempts", [])
        attempts = [a if isinstance(a, dict) else {} for a in attempts_raw]
        lessons = [clean_text(s) for s in data.get("lessons_learned", [])]
        return cls(
            claim=clean_text(data["claim"]),
            source=clean_text(data["source"]),
            confidence=data["confidence"],
            contradictions=[clean_text(c) for c in data.get("contradictions", [])],
            decay_rate=data.get("decay_rate", 0.0),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_verified=datetime.fromisoformat(data["last_verified"]),
            downstream_dependents=[clean_text(d) for d in data.get("downstream_dependents", [])],
            executable_proof=clean_text(data["executable_proof"]) if data.get("executable_proof") else None,
            domain=clean_text(data.get("domain", "General")),
            attempts=attempts,
            lessons_learned=lessons,
        )

    def __repr__(self) -> str:
        """Concise developer-friendly representation."""
        valid = "valid" if self.is_valid() else "invalid"
        expired = "expired" if self.is_expired() else "active"
        return (
            f"BeliefCertificate(claim={self.claim!r}, "
            f"confidence={self.confidence}, "
            f"domain={self.domain!r}, "
            f"{valid}, {expired})"
        )

"""Identity Manager for PROXY.

Manages freelancer identities used for all outreach, proposals, and
follow-up emails.  Identities are stored in data/identities.json.

Email and PayPal fields may reference vault keys via the sentinel
``"vault:<KEY_NAME>"`` — resolved at runtime by IdentityManager.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import logging

logger = logging.getLogger(__name__)

_IDENTITIES_FILE = "identities.json"
_VAULT_PREFIX = "vault:"


@dataclass
class Identity:
    """A freelancer identity used for PROXY outreach.

    Fields that reference vault secrets use the sentinel format
    ``"vault:<KEY_NAME>"``, e.g. ``"vault:GMAIL_ADDRESS"``.
    Call :meth:`IdentityManager.resolve_field` to get the real value.
    """

    id: str
    name: str
    business: str
    niche: list[str]
    email: str
    bio: str
    paypal: str
    active: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Identity":
        return cls(
            id=data["id"],
            name=data["name"],
            business=data["business"],
            niche=list(data.get("niche", [])),
            email=data["email"],
            bio=data["bio"],
            paypal=data["paypal"],
            active=bool(data.get("active", False)),
        )


class IdentityManager:
    """Load, save, and resolve PROXY identities.

    Args:
        data_dir: Directory where ``identities.json`` lives.
        vault:    Optional GuardianVault instance for secret resolution.
    """

    def __init__(self, data_dir: str = "data", vault: Any = None) -> None:
        self._data_dir = Path(data_dir)
        self._vault = vault

    @property
    def _ids_path(self) -> Path:
        return self._data_dir / _IDENTITIES_FILE

    # ------------------------------------------------------------------
    # Load / query
    # ------------------------------------------------------------------

    def load_identities(self) -> list[Identity]:
        """Return all stored identities, or [] if file missing."""
        if not self._ids_path.exists():
            return []
        try:
            data = json.loads(self._ids_path.read_text(encoding="utf-8"))
            return [Identity.from_dict(d) for d in data]
        except Exception as exc:
            logger.warning("identity_manager: failed to load identities: %s", exc)
            return []

    def get_active_identity(self) -> Identity | None:
        """Return the currently active identity, or None."""
        for identity in self.load_identities():
            if identity.active:
                return identity
        return None

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_identity(self, identity: Identity) -> None:
        """Persist identity to identities.json, replacing any entry with same id."""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        identities = self.load_identities()
        # Replace existing or append
        replaced = False
        for i, existing in enumerate(identities):
            if existing.id == identity.id:
                identities[i] = identity
                replaced = True
                break
        if not replaced:
            identities.append(identity)
        self._ids_path.write_text(
            json.dumps([i.to_dict() for i in identities], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def add_identity(self, identity: Identity) -> None:
        """Add identity, setting it active only if no active identity exists."""
        existing = self.load_identities()
        has_active = any(i.active for i in existing)
        if not has_active:
            identity.active = True
        else:
            identity.active = False
        self.save_identity(identity)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve_field(self, value: str) -> str:
        """Resolve a vault sentinel (``vault:KEY``) to the real value.

        Resolution order:
        1. Vault (if vault provided and key exists)
        2. Environment variable with same key name
        3. Raw value as-is (not a sentinel)
        """
        if not value.startswith(_VAULT_PREFIX):
            return value
        key = value[len(_VAULT_PREFIX):]
        if self._vault is not None and self._vault.has(key):
            return self._vault.get(key)
        env_val = os.getenv(key, "").strip()
        if env_val:
            return env_val
        logger.warning("identity_manager: could not resolve vault sentinel %r", value)
        return value  # return sentinel as fallback

    def resolve_identity(self, identity: Identity) -> dict[str, Any]:
        """Return identity as dict with vault sentinels resolved to real values."""
        d = identity.to_dict()
        d["email"] = self.resolve_field(identity.email)
        d["paypal"] = self.resolve_field(identity.paypal)
        return d

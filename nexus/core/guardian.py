"""GUARDIAN — Security layer for NEXUS/PROXY.

Five components:

GuardianVault    Fernet-encrypted credential store backed by a single
                 file.  Master key comes from GUARDIAN_MASTER_KEY env
                 var or an explicit argument.  Raw values are *never*
                 logged, repr-ed, or written to disk in plaintext.

SecretScanner    Regex-based scanner that detects common secret shapes
                 (API keys, tokens, passwords) in source files and log
                 strings.  Matched values are redacted in all output.

CVEScanner       Delegates to ``pip-audit`` via subprocess.  Returns an
                 empty list and logs a warning when pip-audit is absent.

GuardianAlert    Sends breach notifications through the Telegram Bot API
                 using only the stdlib.  Never raises; returns bool.

Guardian         Thin facade that wires the four components together and
                 exposes a single ``audit()`` method returning a
                 ``GuardianReport``.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import pathlib
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

# Forward declaration — defined after GuardianVault.
__all__ = [
    "SecretFinding", "CVEFinding", "GuardianReport",
    "GuardianVault", "SecretScanner", "CVEScanner",
    "GuardianAlert", "Guardian",
    "migrate_key_to_vault",
]

# ──────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────

_PBKDF2_ITERATIONS = 480_000
_SALT_BYTES        = 16

# Patterns: (regex, severity)
_SECRET_PATTERNS: dict[str, tuple[str, str]] = {
    "deepseek_key":       (r"sk-[a-zA-Z0-9]{32,}", "CRITICAL"),
    "openai_key":         (r"sk-[a-zA-Z0-9]{20,}", "CRITICAL"),
    "telegram_token":     (r"\d{8,10}:[A-Za-z0-9_-]{35}", "CRITICAL"),
    "openclaw_token":     (r"oc-tok-[a-zA-Z0-9]{16,}", "CRITICAL"),
    "supabase_key":       (r"eyJ[A-Za-z0-9_-]{50,}", "HIGH"),
    "api_key_assignment": (
        r'[Aa][Pp][Ii][_-]?[Kk][Ee][Yy]\s*[=:]\s*["\'][^"\']{8,}["\']',
        "HIGH",
    ),
    "password_assignment": (
        r'[Pp][Aa][Ss][Ss][Ww][Oo][Rr][Dd]\s*[=:]\s*["\'][^"\']{4,}["\']',
        "HIGH",
    ),
}

_SCAN_EXTENSIONS = {".py", ".log", ".env", ".json", ".txt", ".yaml", ".yml"}
_SKIP_FILENAMES  = {"guardian_vault.enc"}

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


# ══════════════════════════════════════════════════════════
#  Data classes
# ══════════════════════════════════════════════════════════

@dataclass
class SecretFinding:
    """A single detected secret in a file or string."""
    file_path:     str
    line_number:   int
    pattern_name:  str
    redacted_match: str   # first-6…last-3 only — never the full secret
    severity:      str    # "CRITICAL" | "HIGH" | "MEDIUM"


@dataclass
class CVEFinding:
    """A single vulnerability returned by pip-audit."""
    package:           str
    installed_version: str
    vulnerability_id:  str
    description:       str
    fix_versions:      list[str]
    severity:          str


@dataclass
class GuardianReport:
    """Full result of Guardian.audit()."""
    timestamp:       datetime
    secret_findings: list[SecretFinding]
    cve_findings:    list[CVEFinding]
    vault_healthy:   bool
    alerts_sent:     int
    passed:          bool   # True iff zero CRITICAL secret findings


# ══════════════════════════════════════════════════════════
#  1. GuardianVault
# ══════════════════════════════════════════════════════════

class _RedactedStr(str):
    """A str subclass whose repr/str show only a placeholder."""
    __slots__ = ()
    def __repr__(self) -> str: return "<redacted>"
    def __str__(self)  -> str: return "<redacted>"


class GuardianVault:
    """Fernet-encrypted credential store.

    The vault is a single encrypted file.  All plaintext values are
    kept only in memory, never logged, and never exposed via repr/str.

    Usage::

        vault = GuardianVault("data/guardian_vault.enc")
        vault.set("DEEPSEEK_API_KEY", raw_key)
        key = vault.get("DEEPSEEK_API_KEY")   # returns str

    The master key must be supplied as ``GUARDIAN_MASTER_KEY`` env var
    or the ``master_key`` constructor argument.
    """

    def __init__(
        self,
        vault_path: str,
        master_key: str | None = None,
    ) -> None:
        raw_key = master_key or os.environ.get("GUARDIAN_MASTER_KEY")
        if not raw_key:
            raise ValueError(
                "GUARDIAN_MASTER_KEY not set — "
                "pass master_key= or set the environment variable."
            )

        self._path    = pathlib.Path(vault_path)
        self._fernet  = self._build_fernet(raw_key)
        self._data: dict[str, str] = self._load()

        # Zero the raw key from local scope immediately.
        del raw_key

    # ── Fernet setup ────────────────────────────────────────

    def _salt_path(self) -> pathlib.Path:
        return self._path.with_suffix(".salt")

    def _load_or_create_salt(self) -> bytes:
        sp = self._salt_path()
        if sp.exists():
            return base64.b64decode(sp.read_text().strip())
        salt = os.urandom(_SALT_BYTES)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(base64.b64encode(salt).decode())
        return salt

    def _build_fernet(self, master_key: str) -> Fernet:
        salt = self._load_or_create_salt()
        kdf  = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=_PBKDF2_ITERATIONS,
        )
        derived = base64.urlsafe_b64encode(
            kdf.derive(master_key.encode("utf-8"))
        )
        return Fernet(derived)

    # ── Persistence ─────────────────────────────────────────

    def _load(self) -> dict[str, str]:
        if not self._path.exists():
            return {}
        try:
            ciphertext = self._path.read_bytes()
            plaintext  = self._fernet.decrypt(ciphertext)
            return json.loads(plaintext.decode("utf-8"))
        except InvalidToken:
            raise ValueError(
                "Vault decryption failed — wrong master key or corrupted vault."
            )

    def _save(self) -> None:
        plaintext  = json.dumps(self._data).encode("utf-8")
        ciphertext = self._fernet.encrypt(plaintext)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_bytes(ciphertext)

    # ── Public API ──────────────────────────────────────────

    def set(self, name: str, value: str) -> None:
        """Encrypt and store *value* under *name*."""
        self._data[name] = value
        self._save()

    def get(self, name: str) -> str:
        """Decrypt and return the credential stored under *name*.

        The caller is responsible for handling the returned plaintext
        safely.  The value is never logged inside this method.
        """
        if name not in self._data:
            raise KeyError(name)
        # Return a plain str — but log at debug without the value.
        logger.debug("vault.get called  key=%s", name)
        return self._data[name]

    def has(self, name: str) -> bool:
        """Return True if *name* exists in the vault."""
        return name in self._data

    def delete(self, name: str) -> None:
        """Remove *name* from the vault.  Silent if missing."""
        if name in self._data:
            del self._data[name]
            self._save()

    def list_keys(self) -> list[str]:
        """Return credential names — never the values."""
        return list(self._data.keys())

    # ── Safe repr ───────────────────────────────────────────

    def __repr__(self) -> str:
        return f"GuardianVault(path={self._path!r}, keys={self.list_keys()!r})"

    def __str__(self) -> str:
        return repr(self)


# ══════════════════════════════════════════════════════════
#  2. SecretScanner
# ══════════════════════════════════════════════════════════

class SecretScanner:
    """Scans files and strings for common secret patterns.

    Matched values are *always* redacted (first-6…last-3) before being
    stored in :class:`SecretFinding`.  The raw secret never leaves this
    class.
    """

    @staticmethod
    def _redact(value: str) -> str:
        if len(value) > 12:
            return value[:6] + "..." + value[-3:]
        return value[:3] + "***"

    def scan_string(
        self, text: str, label: str = "<string>"
    ) -> list[SecretFinding]:
        """Return all secret findings in *text*."""
        findings: list[SecretFinding] = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            for name, (pattern, severity) in _SECRET_PATTERNS.items():
                m = re.search(pattern, line)
                if m:
                    findings.append(
                        SecretFinding(
                            file_path=label,
                            line_number=lineno,
                            pattern_name=name,
                            redacted_match=self._redact(m.group(0)),
                            severity=severity,
                        )
                    )
        return findings

    def scan_file(self, path: str | pathlib.Path) -> list[SecretFinding]:
        """Scan a single file.  Returns [] for skipped/unreadable files."""
        p = pathlib.Path(path)
        if p.name in _SKIP_FILENAMES:
            return []
        if p.suffix not in _SCAN_EXTENSIONS:
            return []
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except (OSError, PermissionError) as exc:
            logger.debug("SecretScanner: cannot read %s — %s", p, exc)
            return []
        return self.scan_string(text, label=str(p))

    def scan_directory(
        self, path: str | pathlib.Path
    ) -> list[SecretFinding]:
        """Recursively scan all files under *path*."""
        findings: list[SecretFinding] = []
        for p in pathlib.Path(path).rglob("*"):
            if p.is_file():
                findings.extend(self.scan_file(p))
        return findings


# ══════════════════════════════════════════════════════════
#  3. CVEScanner
# ══════════════════════════════════════════════════════════

class CVEScanner:
    """Wraps ``pip-audit`` to check installed packages for known CVEs.

    Falls back gracefully (returns []) when pip-audit is not installed
    or when the subprocess times out.
    """

    _TIMEOUT = 120  # seconds

    def scan(self) -> list[CVEFinding]:
        """Run pip-audit and return findings.  Never raises."""
        try:
            proc = subprocess.run(
                ["pip-audit", "--format=json", "--progress-spinner=off"],
                capture_output=True,
                text=True,
                timeout=self._TIMEOUT,
            )
            data = json.loads(proc.stdout)
        except FileNotFoundError:
            logger.warning(
                "CVEScanner: pip-audit not found — install with "
                "'pip install pip-audit' to enable CVE scanning."
            )
            return []
        except subprocess.TimeoutExpired:
            logger.warning("CVEScanner: pip-audit timed out after %ds", self._TIMEOUT)
            return []
        except json.JSONDecodeError as exc:
            logger.warning("CVEScanner: failed to parse pip-audit output — %s", exc)
            return []
        except Exception as exc:
            logger.warning("CVEScanner: unexpected error — %s", exc)
            return []

        findings: list[CVEFinding] = []
        for dep in data.get("dependencies", []):
            for vuln in dep.get("vulns", []):
                findings.append(
                    CVEFinding(
                        package=dep.get("name", ""),
                        installed_version=dep.get("version", ""),
                        vulnerability_id=vuln.get("id", ""),
                        description=vuln.get("description", "")[:300],
                        fix_versions=list(vuln.get("fix_versions", [])),
                        severity=str(vuln.get("severity", "UNKNOWN")).upper(),
                    )
                )
        return findings


# ══════════════════════════════════════════════════════════
#  4. GuardianAlert
# ══════════════════════════════════════════════════════════

class GuardianAlert:
    """Sends breach notifications via Telegram Bot API.

    Credentials (bot token, chat ID) are retrieved from the vault at
    send-time — they are never stored as instance attributes.

    ``send()`` never raises.  Network failures are logged and return
    False.
    """

    def __init__(self, vault: GuardianVault) -> None:
        self._vault = vault

    def send(self, message: str, severity: str = "INFO") -> bool:
        """Send *message* to the configured Telegram chat.

        Returns True on HTTP 200, False on any failure.
        """
        try:
            token   = self._vault.get("TELEGRAM_BOT_TOKEN")
            chat_id = self._vault.get("TELEGRAM_CHAT_ID")
        except KeyError:
            logger.warning(
                "GuardianAlert: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID "
                "not in vault — alert not sent."
            )
            return False

        text    = f"[GUARDIAN {severity}] {message}"
        url     = _TELEGRAM_API.format(token=token)
        payload = urllib.parse.urlencode(
            {"chat_id": chat_id, "text": text}
        ).encode()

        try:
            req = urllib.request.Request(url, data=payload, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                ok = resp.status == 200
                logger.debug(
                    "GuardianAlert: sent  severity=%s  ok=%s", severity, ok
                )
                return ok
        except Exception as exc:
            logger.warning("GuardianAlert: failed to send alert — %s", exc)
            return False


# ══════════════════════════════════════════════════════════
#  5. Guardian — facade
# ══════════════════════════════════════════════════════════

class Guardian:
    """Security facade for NEXUS.

    Wires vault, scanner, CVE checker, and alert channel together.
    Call :meth:`audit` to run a full security check.

    Example::

        guardian = Guardian()
        guardian.vault.set("DEEPSEEK_API_KEY", raw_key)
        report = guardian.audit()
        if not report.passed:
            print("Security issues found:", report.secret_findings)
    """

    def __init__(
        self,
        vault_path: str = "data/guardian_vault.enc",
        master_key: str | None = None,
        scan_paths: list[str] | None = None,
    ) -> None:
        self.vault   = GuardianVault(vault_path, master_key=master_key)
        self.scanner = SecretScanner()
        self.cve     = CVEScanner()
        self.alert   = GuardianAlert(self.vault)
        self._scan_paths: list[str] = scan_paths if scan_paths is not None else [
            "nexus/",
            "scripts/",
        ]

    def audit(self) -> GuardianReport:
        """Run all security checks and return a :class:`GuardianReport`.

        Steps:
        1. Scan configured paths for exposed secrets.
        2. Run CVE scan via pip-audit.
        3. Send Telegram alerts for CRITICAL secrets and HIGH/CRITICAL CVEs.
        4. Return the aggregated report.
        """
        # 1. Secret scan
        secret_findings: list[SecretFinding] = []
        for path in self._scan_paths:
            if pathlib.Path(path).exists():
                secret_findings.extend(self.scanner.scan_directory(path))

        # 2. CVE scan
        cve_findings = self.cve.scan()

        # 3. Alerts
        alerts_sent = 0
        for f in secret_findings:
            if f.severity == "CRITICAL":
                msg = (
                    f"Secret exposed: `{f.pattern_name}` "
                    f"in `{f.file_path}` line {f.line_number} "
                    f"({f.redacted_match})"
                )
                if self.alert.send(msg, severity="CRITICAL"):
                    alerts_sent += 1

        for f in cve_findings:
            if f.severity in ("HIGH", "CRITICAL"):
                msg = (
                    f"CVE: {f.vulnerability_id} in "
                    f"{f.package} {f.installed_version} — "
                    f"{f.description[:120]}"
                )
                if self.alert.send(msg, severity=f.severity):
                    alerts_sent += 1

        critical_secrets = [f for f in secret_findings if f.severity == "CRITICAL"]
        passed = len(critical_secrets) == 0

        if secret_findings or cve_findings:
            logger.warning(
                "GUARDIAN audit  secrets=%d  cves=%d  passed=%s",
                len(secret_findings), len(cve_findings), passed,
            )
        else:
            logger.info("GUARDIAN audit clean — no findings.")

        return GuardianReport(
            timestamp=datetime.now(timezone.utc),
            secret_findings=secret_findings,
            cve_findings=cve_findings,
            vault_healthy=True,
            alerts_sent=alerts_sent,
            passed=passed,
        )


# ══════════════════════════════════════════════════════════
#  Utility — one-time key migration helper
# ══════════════════════════════════════════════════════════

def migrate_key_to_vault(
    env_path: str | pathlib.Path,
    vault: GuardianVault,
    key_name: str,
) -> None:
    """Move *key_name* from a dotenv file into *vault* and blank it in the file.

    Idempotent — if *key_name* is already present in the vault, nothing is
    changed (neither the vault nor the env file is touched).

    Args:
        env_path: Path to the ``.env`` file containing the plaintext key.
        vault:    The destination :class:`GuardianVault` instance.
        key_name: The variable name to migrate (e.g. ``"DEEPSEEK_API_KEY"``).

    Raises:
        KeyError: If *key_name* is absent from the env file or its value
                  is empty, and the vault does not already hold it.
    """
    if vault.has(key_name):
        # Already migrated — nothing to do.
        return

    env_path = pathlib.Path(env_path)
    lines = env_path.read_text(encoding="utf-8").splitlines(keepends=True)

    value: str | None = None
    new_lines: list[str] = []
    for line in lines:
        stripped = line.rstrip("\r\n")
        if stripped.startswith(f"{key_name}="):
            value = stripped[len(key_name) + 1:]
            if value:
                # Blank the value — keep the key name so the file stays valid.
                new_lines.append(f"{key_name}=\n")
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    if not value:
        raise KeyError(
            f"{key_name!r} not found or empty in {env_path} "
            "— cannot migrate to vault."
        )

    vault.set(key_name, value)
    env_path.write_text("".join(new_lines), encoding="utf-8")
    logger.info("GUARDIAN: migrated %s from %s to vault", key_name, env_path)

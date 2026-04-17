"""Tests for GUARDIAN — security module. Written before implementation (TDD).

Coverage:
- GuardianVault   : encrypted storage, key isolation, wrong-key rejection
- SecretScanner   : pattern detection, redaction, file/directory scanning
- CVEScanner      : subprocess delegation, graceful fallback
- GuardianAlert   : Telegram dispatch, never-raises contract
- Guardian facade : audit report, passed/failed logic
"""

from __future__ import annotations

import json
import logging
import pathlib
import subprocess
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call

import pytest

from nexus.core.guardian import (
    CVEFinding,
    CVEScanner,
    Guardian,
    GuardianAlert,
    GuardianReport,
    GuardianVault,
    SecretFinding,
    SecretScanner,
)


# ─────────────────────────────────────────────
#  Shared fixtures
# ─────────────────────────────────────────────

MASTER_KEY = "test-master-key-for-guardian-unit-tests"
ALT_KEY    = "different-wrong-key-should-fail-decryption"


@pytest.fixture()
def vault_path(tmp_path) -> pathlib.Path:
    return tmp_path / "test_vault.enc"


@pytest.fixture()
def vault(vault_path) -> GuardianVault:
    return GuardianVault(str(vault_path), master_key=MASTER_KEY)


# ═══════════════════════════════════════════════════════════
#  1. GuardianVault
# ═══════════════════════════════════════════════════════════

class TestGuardianVaultInit:
    def test_raises_without_master_key(self, vault_path):
        with pytest.raises((ValueError, KeyError)):
            GuardianVault(str(vault_path), master_key=None)

    def test_creates_with_master_key(self, vault_path):
        v = GuardianVault(str(vault_path), master_key=MASTER_KEY)
        assert v is not None

    def test_creates_with_env_var(self, vault_path, monkeypatch):
        monkeypatch.setenv("GUARDIAN_MASTER_KEY", MASTER_KEY)
        v = GuardianVault(str(vault_path))
        assert v is not None


class TestGuardianVaultSetGet:
    def test_set_and_get_roundtrip(self, vault):
        vault.set("MY_KEY", "super-secret-value")
        assert vault.get("MY_KEY") == "super-secret-value"

    def test_get_missing_key_raises_keyerror(self, vault):
        with pytest.raises(KeyError):
            vault.get("NONEXISTENT")

    def test_multiple_credentials(self, vault):
        vault.set("KEY_A", "value_a")
        vault.set("KEY_B", "value_b")
        assert vault.get("KEY_A") == "value_a"
        assert vault.get("KEY_B") == "value_b"

    def test_overwrite_updates_value(self, vault):
        vault.set("KEY", "old")
        vault.set("KEY", "new")
        assert vault.get("KEY") == "new"

    def test_has_returns_true_for_existing(self, vault):
        vault.set("EXISTS", "yes")
        assert vault.has("EXISTS") is True

    def test_has_returns_false_for_missing(self, vault):
        assert vault.has("MISSING") is False

    def test_delete_removes_key(self, vault):
        vault.set("TEMP", "value")
        vault.delete("TEMP")
        assert vault.has("TEMP") is False

    def test_delete_missing_key_is_silent(self, vault):
        vault.delete("NEVER_SET")  # must not raise

    def test_list_keys_returns_names(self, vault):
        vault.set("ALPHA", "1")
        vault.set("BETA",  "2")
        keys = vault.list_keys()
        assert "ALPHA" in keys
        assert "BETA"  in keys

    def test_list_keys_never_contains_values(self, vault):
        vault.set("SECRET", "plaintext_value_xyz")
        keys = vault.list_keys()
        assert "plaintext_value_xyz" not in keys
        assert "plaintext_value_xyz" not in str(keys)


class TestGuardianVaultEncryption:
    def test_vault_file_is_not_plaintext(self, vault, vault_path):
        vault.set("API_KEY", "sk-supersecret12345678901234567890")
        raw = vault_path.read_bytes()
        assert b"sk-supersecret" not in raw
        assert b"API_KEY" not in raw

    def test_wrong_master_key_raises_on_get(self, vault, vault_path):
        vault.set("KEY", "value")
        with pytest.raises(Exception):
            v2 = GuardianVault(str(vault_path), master_key=ALT_KEY)
            v2.get("KEY")

    def test_persists_across_instances(self, vault_path):
        v1 = GuardianVault(str(vault_path), master_key=MASTER_KEY)
        v1.set("PERSISTENT", "round-trip-value")
        del v1
        v2 = GuardianVault(str(vault_path), master_key=MASTER_KEY)
        assert v2.get("PERSISTENT") == "round-trip-value"


class TestGuardianVaultNoLeak:
    def test_get_does_not_log_plaintext(self, vault, caplog):
        secret = "sk-4ac0b8f0a9ed4e7ab19f5132f8e77c95"
        vault.set("DS_KEY", secret)
        with caplog.at_level(logging.DEBUG, logger="nexus.core.guardian"):
            vault.get("DS_KEY")
        assert secret not in caplog.text

    def test_set_does_not_log_plaintext(self, vault, caplog):
        secret = "telegram-token-9999999:AbCdEfGhIjKlMnOpQrStUvWxYz01234567"
        with caplog.at_level(logging.DEBUG, logger="nexus.core.guardian"):
            vault.set("TG_TOKEN", secret)
        assert secret not in caplog.text

    def test_repr_does_not_expose_secrets(self, vault):
        vault.set("HIDDEN", "do-not-show-me")
        r = repr(vault)
        assert "do-not-show-me" not in r

    def test_str_does_not_expose_secrets(self, vault):
        vault.set("HIDDEN", "do-not-show-me")
        assert "do-not-show-me" not in str(vault)


# ═══════════════════════════════════════════════════════════
#  2. SecretScanner
# ═══════════════════════════════════════════════════════════

@pytest.fixture()
def scanner() -> SecretScanner:
    return SecretScanner()


class TestSecretScannerPatterns:
    def test_detects_deepseek_key(self, scanner):
        findings = scanner.scan_string(
            'DEEPSEEK_API_KEY = "sk-4ac0b8f0a9ed4e7ab19f5132f8e77c95"'
        )
        assert len(findings) >= 1
        assert any(f.severity == "CRITICAL" for f in findings)

    def test_detects_telegram_token(self, scanner):
        findings = scanner.scan_string(
            "BOT_TOKEN = '123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef012'"
        )
        assert len(findings) >= 1

    def test_detects_api_key_assignment(self, scanner):
        findings = scanner.scan_string('api_key = "my-secret-api-key-value"')
        assert len(findings) >= 1

    def test_detects_password_assignment(self, scanner):
        findings = scanner.scan_string("password = 'hunter2secure'")
        assert len(findings) >= 1

    def test_clean_string_returns_empty(self, scanner):
        findings = scanner.scan_string("x = 1\nprint('hello world')\n")
        assert findings == []

    def test_finding_redacts_match(self, scanner):
        findings = scanner.scan_string(
            'KEY = "sk-4ac0b8f0a9ed4e7ab19f5132f8e77c95"'
        )
        assert findings
        raw_secret = "sk-4ac0b8f0a9ed4e7ab19f5132f8e77c95"
        for f in findings:
            assert raw_secret not in f.redacted_match

    def test_finding_has_correct_line_number(self, scanner):
        text = "# comment\n# comment\nAPI_KEY = 'sk-abcdef1234567890abcdef1234567890'\n"
        findings = scanner.scan_string(text)
        assert findings
        assert findings[0].line_number == 3

    def test_finding_has_pattern_name(self, scanner):
        findings = scanner.scan_string('sk-abcdef1234567890abcdef1234567890abcdef')
        assert findings
        assert findings[0].pattern_name != ""

    def test_finding_is_secret_finding_instance(self, scanner):
        findings = scanner.scan_string('sk-abcdef1234567890abcdef1234567890abcdef')
        assert findings
        assert isinstance(findings[0], SecretFinding)


class TestSecretScannerFiles:
    def test_scan_file_with_secret(self, scanner, tmp_path):
        f = tmp_path / "config.py"
        f.write_text('API_KEY = "sk-abcdefabcdefabcdefabcdefabcdefab"\n')
        findings = scanner.scan_file(f)
        assert len(findings) >= 1

    def test_scan_file_without_secret(self, scanner, tmp_path):
        f = tmp_path / "clean.py"
        f.write_text("def add(a, b):\n    return a + b\n")
        findings = scanner.scan_file(f)
        assert findings == []

    def test_scan_file_records_path(self, scanner, tmp_path):
        f = tmp_path / "leaky.py"
        f.write_text('TOKEN = "sk-abcdefabcdefabcdefabcdefabcdefab"\n')
        findings = scanner.scan_file(f)
        assert findings
        assert str(f) in findings[0].file_path

    def test_scan_file_skips_vault_file(self, scanner, tmp_path):
        vault = tmp_path / "guardian_vault.enc"
        vault.write_bytes(b"\x00\x01ENCRYPTED_BINARY_GARBAGE\xFF")
        findings = scanner.scan_file(vault)
        assert findings == []

    def test_scan_file_ignores_unknown_extension(self, scanner, tmp_path):
        f = tmp_path / "binary.exe"
        f.write_bytes(b"sk-abcdefabcdefabcdefabcdefabcdefab")
        findings = scanner.scan_file(f)
        assert findings == []

    def test_scan_directory_finds_secrets_in_subtree(self, scanner, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "secrets.py").write_text(
            'DS_KEY = "sk-abcdefabcdefabcdefabcdefabcdefab"\n'
        )
        (tmp_path / "clean.py").write_text("x = 1\n")
        findings = scanner.scan_directory(tmp_path)
        assert len(findings) >= 1

    def test_scan_directory_empty_returns_empty(self, scanner, tmp_path):
        findings = scanner.scan_directory(tmp_path)
        assert findings == []


# ═══════════════════════════════════════════════════════════
#  3. CVEScanner
# ═══════════════════════════════════════════════════════════

_PIP_AUDIT_CLEAN = json.dumps({"dependencies": []})

_PIP_AUDIT_VULN = json.dumps({
    "dependencies": [
        {
            "name": "requests",
            "version": "2.27.0",
            "vulns": [
                {
                    "id": "PYSEC-2023-74",
                    "description": "Unintended leak of Proxy-Authorization header",
                    "fix_versions": ["2.31.0"],
                    "severity": "HIGH",
                }
            ],
        }
    ]
})


class TestCVEScanner:
    def test_returns_list(self):
        cve = CVEScanner()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=_PIP_AUDIT_CLEAN, returncode=0)
            result = cve.scan()
        assert isinstance(result, list)

    def test_parses_vulnerability(self):
        cve = CVEScanner()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=_PIP_AUDIT_VULN, returncode=1)
            findings = cve.scan()
        assert len(findings) == 1
        f = findings[0]
        assert f.package == "requests"
        assert f.installed_version == "2.27.0"
        assert f.vulnerability_id == "PYSEC-2023-74"
        assert f.severity == "HIGH"
        assert "2.31.0" in f.fix_versions

    def test_finding_is_cve_finding_instance(self):
        cve = CVEScanner()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=_PIP_AUDIT_VULN, returncode=1)
            findings = cve.scan()
        assert isinstance(findings[0], CVEFinding)

    def test_graceful_when_pip_audit_missing(self):
        cve = CVEScanner()
        with patch("subprocess.run", side_effect=FileNotFoundError):
            findings = cve.scan()
        assert findings == []

    def test_graceful_on_timeout(self):
        cve = CVEScanner()
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("pip-audit", 120)):
            findings = cve.scan()
        assert findings == []

    def test_graceful_on_bad_json(self):
        cve = CVEScanner()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="not json {{", returncode=0)
            findings = cve.scan()
        assert findings == []

    def test_clean_scan_returns_empty(self):
        cve = CVEScanner()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=_PIP_AUDIT_CLEAN, returncode=0)
            findings = cve.scan()
        assert findings == []


# ═══════════════════════════════════════════════════════════
#  4. GuardianAlert
# ═══════════════════════════════════════════════════════════

def _make_vault_with_telegram(tmp_path) -> GuardianVault:
    v = GuardianVault(str(tmp_path / "vault.enc"), master_key=MASTER_KEY)
    v.set("TELEGRAM_BOT_TOKEN", "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef012")
    v.set("TELEGRAM_CHAT_ID", "-1001234567890")
    return v


class TestGuardianAlert:
    def test_send_returns_true_on_success(self, tmp_path):
        vault = _make_vault_with_telegram(tmp_path)
        alert = GuardianAlert(vault)
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 200
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = alert.send("test message", severity="INFO")
        assert result is True

    def test_send_returns_false_on_network_error(self, tmp_path):
        import urllib.error
        vault = _make_vault_with_telegram(tmp_path)
        alert = GuardianAlert(vault)
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("connection refused")):
            result = alert.send("test", severity="HIGH")
        assert result is False

    def test_send_never_raises(self, tmp_path):
        vault = _make_vault_with_telegram(tmp_path)
        alert = GuardianAlert(vault)
        with patch("urllib.request.urlopen", side_effect=Exception("unexpected")):
            try:
                result = alert.send("boom", severity="CRITICAL")
                assert result is False
            except Exception as exc:
                pytest.fail(f"GuardianAlert.send() raised unexpectedly: {exc}")

    def test_send_returns_false_when_credentials_missing(self, vault_path):
        vault = GuardianVault(str(vault_path), master_key=MASTER_KEY)
        # Telegram credentials NOT set
        alert = GuardianAlert(vault)
        result = alert.send("no creds test")
        assert result is False

    def test_send_includes_severity_in_message(self, tmp_path):
        vault = _make_vault_with_telegram(tmp_path)
        alert = GuardianAlert(vault)
        captured = {}
        def fake_urlopen(req, **kw):
            captured["body"] = req.data.decode()
            resp = MagicMock()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            resp.status = 200
            return resp
        with patch("urllib.request.urlopen", fake_urlopen):
            alert.send("breach detected", severity="CRITICAL")
        assert "CRITICAL" in captured.get("body", "")

    def test_send_does_not_expose_bot_token_in_logs(self, tmp_path, caplog):
        vault = _make_vault_with_telegram(tmp_path)
        alert = GuardianAlert(vault)
        token = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef012"
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 200
        with caplog.at_level(logging.DEBUG, logger="nexus.core.guardian"):
            with patch("urllib.request.urlopen", return_value=mock_resp):
                alert.send("check log safety")
        assert token not in caplog.text


# ═══════════════════════════════════════════════════════════
#  5. Guardian facade
# ═══════════════════════════════════════════════════════════

class TestGuardianFacade:
    def test_has_vault(self, vault_path):
        g = Guardian(vault_path=str(vault_path), master_key=MASTER_KEY)
        assert isinstance(g.vault, GuardianVault)

    def test_has_scanner(self, vault_path):
        g = Guardian(vault_path=str(vault_path), master_key=MASTER_KEY)
        assert isinstance(g.scanner, SecretScanner)

    def test_has_cve_scanner(self, vault_path):
        g = Guardian(vault_path=str(vault_path), master_key=MASTER_KEY)
        assert isinstance(g.cve, CVEScanner)

    def test_has_alert(self, vault_path):
        g = Guardian(vault_path=str(vault_path), master_key=MASTER_KEY)
        assert isinstance(g.alert, GuardianAlert)


class TestGuardianAudit:
    def _make_guardian(self, tmp_path, scan_dir=None) -> Guardian:
        return Guardian(
            vault_path=str(tmp_path / "vault.enc"),
            master_key=MASTER_KEY,
            scan_paths=[str(scan_dir)] if scan_dir else [],
        )

    def test_audit_returns_guardian_report(self, tmp_path):
        g = self._make_guardian(tmp_path)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=_PIP_AUDIT_CLEAN, returncode=0)
            report = g.audit()
        assert isinstance(report, GuardianReport)

    def test_audit_report_has_timestamp(self, tmp_path):
        g = self._make_guardian(tmp_path)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=_PIP_AUDIT_CLEAN, returncode=0)
            report = g.audit()
        assert isinstance(report.timestamp, datetime)

    def test_audit_passed_when_no_critical_findings(self, tmp_path):
        scan_dir = tmp_path / "src"
        scan_dir.mkdir()
        (scan_dir / "clean.py").write_text("x = 1\n")
        g = self._make_guardian(tmp_path, scan_dir=scan_dir)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=_PIP_AUDIT_CLEAN, returncode=0)
            report = g.audit()
        assert report.passed is True
        assert report.secret_findings == []

    def test_audit_fails_when_critical_secret_found(self, tmp_path):
        scan_dir = tmp_path / "src"
        scan_dir.mkdir()
        (scan_dir / "leak.py").write_text(
            'DS = "sk-abcdef1234567890abcdef1234567890ab"\n'
        )
        g = self._make_guardian(tmp_path, scan_dir=scan_dir)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=_PIP_AUDIT_CLEAN, returncode=0)
            report = g.audit()
        assert report.passed is False
        assert len(report.secret_findings) >= 1

    def test_audit_includes_cve_findings(self, tmp_path):
        g = self._make_guardian(tmp_path)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=_PIP_AUDIT_VULN, returncode=1)
            report = g.audit()
        assert len(report.cve_findings) == 1
        assert report.cve_findings[0].package == "requests"

    def test_audit_sends_alert_on_critical_secret(self, tmp_path):
        scan_dir = tmp_path / "src"
        scan_dir.mkdir()
        (scan_dir / "leak.py").write_text(
            'API = "sk-abcdef1234567890abcdef1234567890ab"\n'
        )
        vault = GuardianVault(str(tmp_path / "vault.enc"), master_key=MASTER_KEY)
        vault.set("TELEGRAM_BOT_TOKEN", "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef012")
        vault.set("TELEGRAM_CHAT_ID", "-1001234567890")
        g = Guardian(
            vault_path=str(tmp_path / "vault.enc"),
            master_key=MASTER_KEY,
            scan_paths=[str(scan_dir)],
        )
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 200
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=_PIP_AUDIT_CLEAN, returncode=0)
            with patch("urllib.request.urlopen", return_value=mock_resp):
                report = g.audit()
        assert report.alerts_sent >= 1

    def test_audit_vault_healthy_flag(self, tmp_path):
        g = self._make_guardian(tmp_path)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=_PIP_AUDIT_CLEAN, returncode=0)
            report = g.audit()
        assert report.vault_healthy is True


# ═══════════════════════════════════════════════════════════
#  6. Integration — vault credentials never reach other modules
# ═══════════════════════════════════════════════════════════

class TestCredentialIsolation:
    def test_vault_get_value_not_in_list_keys(self, vault):
        vault.set("SECRET_TOKEN", "plaintext_secret_xyz")
        assert "plaintext_secret_xyz" not in vault.list_keys()

    def test_vault_file_does_not_contain_key_names(self, vault, vault_path):
        vault.set("DEEPSEEK_API_KEY", "sk-test1234567890test1234567890test12")
        raw = vault_path.read_bytes()
        assert b"DEEPSEEK_API_KEY" not in raw

    def test_wrong_key_on_populated_vault_raises(self, vault_path):
        v1 = GuardianVault(str(vault_path), master_key=MASTER_KEY)
        v1.set("CRED", "sensitive-data")
        with pytest.raises(Exception):
            v2 = GuardianVault(str(vault_path), master_key=ALT_KEY)
            v2.get("CRED")

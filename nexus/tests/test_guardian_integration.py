"""Tests for GUARDIAN ↔ NEXUS integration (TDD — written before implementation).

Coverage:
  1. migrate_key_to_vault  — reads key from .env, stores in vault, blanks line
  2. ModelRouter.vault     — vault-first key resolution, env fallback, model filter
  3. fail_fast_on_critical_findings — raises SystemExit on CRITICAL secrets
  4. build_nexus guardian wiring    — guardian created, vault passed to router,
                                       fail-fast fires on exposed secrets
"""

from __future__ import annotations

import pathlib
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from nexus.core.guardian import (
    Guardian,
    GuardianReport,
    GuardianVault,
    SecretFinding,
    migrate_key_to_vault,
)
from nexus.core.model_router import ModelRouter


MASTER_KEY = "integration-test-master-key-for-nexus-guardian"


# ─────────────────────────────────────────────────────────────
#  Shared fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def vault(tmp_path):
    return GuardianVault(str(tmp_path / "vault.enc"), master_key=MASTER_KEY)


@pytest.fixture
def env_file(tmp_path):
    ef = tmp_path / ".env"
    ef.write_text(
        "DEEPSEEK_API_KEY=sk-test1234567890abcdef1234567890ab\n"
        "OTHER_KEY=some-other-value\n"
    )
    return ef


def _make_report(passed: bool, *, critical: bool = False) -> GuardianReport:
    findings = []
    if critical:
        findings.append(SecretFinding(
            file_path="leak.py", line_number=1,
            pattern_name="deepseek_key",
            redacted_match="sk-te...0ab",
            severity="CRITICAL",
        ))
    return GuardianReport(
        timestamp=datetime.now(timezone.utc),
        secret_findings=findings,
        cve_findings=[],
        vault_healthy=True,
        alerts_sent=0,
        passed=passed,
    )


# ═══════════════════════════════════════════════════════════════
#  1. TestMigrateKeyToVault
# ═══════════════════════════════════════════════════════════════

class TestMigrateKeyToVault:
    """migrate_key_to_vault reads key from .env, stores in vault, blanks line."""

    def test_stores_key_in_vault(self, vault, env_file):
        migrate_key_to_vault(env_file, vault, "DEEPSEEK_API_KEY")
        assert vault.get("DEEPSEEK_API_KEY") == "sk-test1234567890abcdef1234567890ab"

    def test_blanks_key_value_in_env_file(self, vault, env_file):
        migrate_key_to_vault(env_file, vault, "DEEPSEEK_API_KEY")
        content = env_file.read_text()
        assert "sk-test1234567890abcdef1234567890ab" not in content

    def test_preserves_other_lines_in_env_file(self, vault, env_file):
        migrate_key_to_vault(env_file, vault, "DEEPSEEK_API_KEY")
        content = env_file.read_text()
        assert "OTHER_KEY=some-other-value" in content

    def test_idempotent_when_already_in_vault(self, vault, env_file):
        """If key already in vault, do nothing — do not overwrite vault or .env."""
        vault.set("DEEPSEEK_API_KEY", "sk-already-in-vault")
        migrate_key_to_vault(env_file, vault, "DEEPSEEK_API_KEY")
        # Vault value unchanged
        assert vault.get("DEEPSEEK_API_KEY") == "sk-already-in-vault"
        # .env file untouched
        assert "sk-test1234567890abcdef1234567890ab" in env_file.read_text()

    def test_raises_keyerror_when_key_absent_from_env(self, vault, tmp_path):
        ef = tmp_path / "no_ds.env"
        ef.write_text("OTHER_KEY=value\n")
        with pytest.raises(KeyError):
            migrate_key_to_vault(ef, vault, "DEEPSEEK_API_KEY")

    def test_raises_keyerror_when_value_blank_and_not_in_vault(self, vault, tmp_path):
        ef = tmp_path / "blank.env"
        ef.write_text("DEEPSEEK_API_KEY=\n")
        with pytest.raises(KeyError):
            migrate_key_to_vault(ef, vault, "DEEPSEEK_API_KEY")


# ═══════════════════════════════════════════════════════════════
#  2. TestModelRouterVaultKey
# ═══════════════════════════════════════════════════════════════

class TestModelRouterVaultKey:
    """ModelRouter must expose a vault field and prefer it over env for keys."""

    def test_vault_field_exists(self):
        assert hasattr(ModelRouter(), "vault")

    def test_vault_is_none_by_default(self):
        assert ModelRouter().vault is None

    def test_get_deepseek_key_method_exists(self):
        assert callable(getattr(ModelRouter(), "_get_deepseek_key", None))

    def test_vault_value_returned_when_vault_present(self, tmp_path):
        v = GuardianVault(str(tmp_path / "v.enc"), master_key=MASTER_KEY)
        v.set("DEEPSEEK_API_KEY", "sk-vault-value-1234567890abcdef")
        router = ModelRouter(vault=v)
        assert router._get_deepseek_key() == "sk-vault-value-1234567890abcdef"

    def test_env_fallback_when_no_vault(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env-fallback")
        router = ModelRouter()
        assert router._get_deepseek_key() == "sk-env-fallback"

    def test_vault_takes_precedence_over_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env-should-lose")
        v = GuardianVault(str(tmp_path / "v.enc"), master_key=MASTER_KEY)
        v.set("DEEPSEEK_API_KEY", "sk-vault-wins-1234567890abcdef")
        router = ModelRouter(vault=v)
        assert router._get_deepseek_key() == "sk-vault-wins-1234567890abcdef"

    def test_env_fallback_when_key_not_in_vault(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env-used-as-fallback")
        v = GuardianVault(str(tmp_path / "v.enc"), master_key=MASTER_KEY)
        # vault exists but DEEPSEEK_API_KEY not stored
        router = ModelRouter(vault=v)
        assert router._get_deepseek_key() == "sk-env-used-as-fallback"

    def test_none_returned_when_neither_vault_nor_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        v = GuardianVault(str(tmp_path / "v.enc"), master_key=MASTER_KEY)
        router = ModelRouter(vault=v)
        assert router._get_deepseek_key() is None

    def test_deepseek_key_available_via_vault_when_env_absent(self, tmp_path, monkeypatch):
        """DeepSeek must not be filtered out when key exists only in vault."""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        v = GuardianVault(str(tmp_path / "v.enc"), master_key=MASTER_KEY)
        v.set("DEEPSEEK_API_KEY", "sk-only-in-vault-1234567890abcd")
        router = ModelRouter(vault=v)
        assert router._get_deepseek_key() == "sk-only-in-vault-1234567890abcd"

    def test_try_model_passes_vault_key_to_litellm(self, tmp_path, monkeypatch):
        """_try_model must pass the vault key as api_key, not the env key."""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        v = GuardianVault(str(tmp_path / "v.enc"), master_key=MASTER_KEY)
        v.set("DEEPSEEK_API_KEY", "sk-vault-only-abcdef1234567890ab")
        router = ModelRouter(vault=v)

        captured: list[str] = []

        def mock_completion(**kw):
            captured.append(kw.get("api_key", ""))
            resp = MagicMock()
            resp.choices[0].message.content = "vault key worked"
            return resp

        with patch("nexus.core.model_router.litellm.completion", side_effect=mock_completion):
            msgs = [{"role": "user", "content": "hi"}]
            result = router._try_model(
                "deepseek/deepseek-chat", msgs, 100, "test_house", "test_label"
            )

        assert result == "vault key worked"
        assert captured[0] == "sk-vault-only-abcdef1234567890ab"


# ═══════════════════════════════════════════════════════════════
#  3. TestFailFastOnCriticalFindings
# ═══════════════════════════════════════════════════════════════

class TestFailFastOnCriticalFindings:
    """fail_fast_on_critical_findings must raise SystemExit on CRITICAL, pass on clean."""

    def test_raises_system_exit_when_report_not_passed(self):
        from nexus.main import fail_fast_on_critical_findings
        report = _make_report(passed=False, critical=True)
        with pytest.raises(SystemExit):
            fail_fast_on_critical_findings(report)

    def test_system_exit_message_mentions_critical(self):
        from nexus.main import fail_fast_on_critical_findings
        report = _make_report(passed=False, critical=True)
        with pytest.raises(SystemExit) as exc_info:
            fail_fast_on_critical_findings(report)
        assert "CRITICAL" in str(exc_info.value).upper() or exc_info.value.code != 0

    def test_does_not_raise_on_clean_report(self):
        from nexus.main import fail_fast_on_critical_findings
        fail_fast_on_critical_findings(_make_report(passed=True))  # must not raise

    def test_does_not_raise_on_high_only_findings(self):
        from nexus.main import fail_fast_on_critical_findings
        high = SecretFinding(
            file_path="x.py", line_number=1,
            pattern_name="api_key_assignment",
            redacted_match="abc...xyz", severity="HIGH",
        )
        report = GuardianReport(
            timestamp=datetime.now(timezone.utc),
            secret_findings=[high],
            cve_findings=[],
            vault_healthy=True,
            alerts_sent=0,
            passed=True,
        )
        fail_fast_on_critical_findings(report)  # HIGH alone must not abort


# ═══════════════════════════════════════════════════════════════
#  4. TestBuildNexusGuardian
# ═══════════════════════════════════════════════════════════════

def _nexus_mocks():
    """Context manager that stubs out the heavy NEXUS components."""
    return (
        patch("nexus.main.KnowledgeGraph"),
        patch("nexus.main.HouseB"),
        patch("nexus.main.HouseC"),
        patch("nexus.main.HouseD"),
        patch("nexus.main.HouseOmega"),
    )


def _configure_kg(MockKG):
    MockKG.return_value.persistence.last_load_count = 0
    MockKG.return_value.__iter__ = lambda s: iter([])
    MockKG.return_value.inject_external_signal = MagicMock(
        return_value={"added": 0, "rejected": 0}
    )


class TestBuildNexusGuardian:
    """build_nexus must accept guardian params and wire Guardian into startup."""

    def test_accepts_guardian_params(self, tmp_path):
        """build_nexus must accept guardian_vault_path / master_key / scan_paths."""
        from nexus.main import build_nexus
        with (
            patch("nexus.main.KnowledgeGraph") as MockKG,
            patch("nexus.main.HouseB"),
            patch("nexus.main.HouseC"),
            patch("nexus.main.HouseD"),
            patch("nexus.main.HouseOmega"),
        ):
            _configure_kg(MockKG)
            build_nexus(
                guardian_vault_path=str(tmp_path / "vault.enc"),
                guardian_master_key=MASTER_KEY,
                guardian_scan_paths=[str(tmp_path)],
            )  # must not raise

    def test_router_gets_vault_instance(self, tmp_path):
        """ModelRouter created inside build_nexus must have vault set."""
        from nexus.main import build_nexus
        captured_routers: list[ModelRouter] = []

        real_ModelRouter = ModelRouter

        def capturing_router(**kw):
            r = real_ModelRouter(**kw)
            captured_routers.append(r)
            return r

        with (
            patch("nexus.main.KnowledgeGraph") as MockKG,
            patch("nexus.main.HouseB"),
            patch("nexus.main.HouseC"),
            patch("nexus.main.HouseD"),
            patch("nexus.main.HouseOmega"),
            patch("nexus.main.ModelRouter", side_effect=capturing_router),
        ):
            _configure_kg(MockKG)
            build_nexus(
                guardian_vault_path=str(tmp_path / "vault.enc"),
                guardian_master_key=MASTER_KEY,
                guardian_scan_paths=[str(tmp_path)],
            )

        assert len(captured_routers) == 1
        assert captured_routers[0].vault is not None
        assert isinstance(captured_routers[0].vault, GuardianVault)

    def test_fails_fast_on_critical_secret_in_scan_path(self, tmp_path):
        """build_nexus must raise SystemExit when scan finds CRITICAL secret."""
        from nexus.main import build_nexus
        (tmp_path / "leak.py").write_text(
            'DEEPSEEK_API_KEY = "sk-leaked1234567890abcdef1234567890ab"\n'
        )
        with (
            patch("nexus.main.KnowledgeGraph") as MockKG,
            patch("nexus.main.HouseB"),
            patch("nexus.main.HouseC"),
            patch("nexus.main.HouseD"),
            patch("nexus.main.HouseOmega"),
        ):
            _configure_kg(MockKG)
            with pytest.raises(SystemExit):
                build_nexus(
                    guardian_vault_path=str(tmp_path / "vault.enc"),
                    guardian_master_key=MASTER_KEY,
                    guardian_scan_paths=[str(tmp_path)],
                )

    def test_passes_with_clean_scan_path(self, tmp_path):
        """build_nexus must not raise when no CRITICAL secrets are present."""
        from nexus.main import build_nexus
        with (
            patch("nexus.main.KnowledgeGraph") as MockKG,
            patch("nexus.main.HouseB"),
            patch("nexus.main.HouseC"),
            patch("nexus.main.HouseD"),
            patch("nexus.main.HouseOmega"),
        ):
            _configure_kg(MockKG)
            build_nexus(
                guardian_vault_path=str(tmp_path / "vault.enc"),
                guardian_master_key=MASTER_KEY,
                guardian_scan_paths=[str(tmp_path)],
            )  # must not raise

    def test_build_nexus_no_guardian_params_still_works(self, tmp_path, monkeypatch):
        """build_nexus must remain backwards-compatible when no guardian params given.

        The audit itself is mocked here — this test is about API compatibility
        (no crash on missing params), not about audit correctness.  The
        other tests cover actual audit behaviour.
        """
        from nexus.main import build_nexus
        monkeypatch.setenv("NEXUS_VAULT_KEY", MASTER_KEY)
        monkeypatch.setenv("NEXUS_VAULT_PATH", str(tmp_path / "vault.enc"))

        clean_vault = GuardianVault(str(tmp_path / "vault.enc"), master_key=MASTER_KEY)

        with (
            patch("nexus.main.KnowledgeGraph") as MockKG,
            patch("nexus.main.HouseB"),
            patch("nexus.main.HouseC"),
            patch("nexus.main.HouseD"),
            patch("nexus.main.HouseOmega"),
            patch("nexus.main.Guardian") as MockGuardian,
        ):
            _configure_kg(MockKG)
            mock_g = MockGuardian.return_value
            mock_g.vault = clean_vault
            mock_g.audit.return_value = _make_report(passed=True)
            build_nexus()  # no guardian kwargs — must not raise

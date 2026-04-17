"""Tests for SMS verification wiring in _execute_browser_task.

When OpenClaw hits an SMS verification screen it cannot complete on its own,
it returns a response containing:  WAITING_FOR_SMS: <site>

_execute_browser_task must:
  1. Detect the WAITING_FOR_SMS signal in the OpenClaw response.
  2. Extract the site name from the signal.
  3. Build a TelegramRelay (from env) and call request_sms_code(site).
  4. Send a second OpenClaw task with the SMS code to complete verification.
  5. Use the second response as the final result.
  6. If TelegramRelay is unavailable or times out — fail gracefully (no crash).

Coverage:
  WAITING_FOR_SMS signal detected in OpenClaw response
  Site name correctly extracted from WAITING_FOR_SMS signal
  TelegramRelay.request_sms_code called with extracted site
  Second OpenClaw call made when SMS code received
  Second OpenClaw response used as final execution_proof
  Artifact passes validation after successful SMS relay
  Graceful failure when TelegramRelay returns None (timeout)
  Graceful failure when TelegramRelay unavailable (no env vars)
  No crash when OpenClaw returns None
  Normal (non-SMS) OpenClaw responses bypass the relay entirely
  WAITING_FOR_SMS with multi-word site name parsed correctly
  Second OpenClaw call includes the SMS code in the task text
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest

from nexus.core.house_c import HouseC, BuildArtifact
from nexus.core.house_b import StructuredSpecificationObject
from nexus.core.knowledge_graph import KnowledgeGraph
from nexus.core.openclaw_client import OpenClawClient


# ── helpers ────────────────────────────────────────────────────────────────────

def _sso(problem: str = "send Reddit DM to r/forhire posters") -> StructuredSpecificationObject:
    return StructuredSpecificationObject(
        original_input=problem,
        redefined_problem=problem,
        domain="freelance",
        constraints=[],
        success_criteria=["send at least one DM"],
    )


def _hc(tmp_path) -> HouseC:
    client = MagicMock(spec=OpenClawClient)
    return HouseC(
        knowledge_graph=MagicMock(spec=KnowledgeGraph),
        workspace_dir=str(tmp_path),
        openclaw_client=client,
    )


def _artifact(hc: HouseC) -> BuildArtifact:
    return BuildArtifact(sso=_sso())


# ══════════════════════════════════════════════════════════════════
#  Signal detection
# ══════════════════════════════════════════════════════════════════

class TestWaitingForSmsSignal:
    def test_signal_detected_in_response(self, tmp_path):
        """WAITING_FOR_SMS in response triggers the relay path."""
        hc = _hc(tmp_path)
        artifact = _artifact(hc)

        relay = MagicMock()
        relay.request_sms_code.return_value = "123456"

        hc.openclaw_client.send.side_effect = [
            "WAITING_FOR_SMS: reddit.com",   # first call — needs SMS
            "FINDING: u/alice — job post",   # second call — success
        ]

        with patch("nexus.core.house_c.TelegramRelay") as MockRelay:
            MockRelay.from_env.return_value = relay
            result = hc._execute_browser_task(artifact, hc.openclaw_client)

        assert relay.request_sms_code.called

    def test_site_extracted_from_signal(self, tmp_path):
        """The site name after 'WAITING_FOR_SMS:' is passed to request_sms_code."""
        hc = _hc(tmp_path)
        artifact = _artifact(hc)

        relay = MagicMock()
        relay.request_sms_code.return_value = "999999"

        hc.openclaw_client.send.side_effect = [
            "WAITING_FOR_SMS: reddit.com",
            "FINDING: u/bob — job",
        ]

        with patch("nexus.core.house_c.TelegramRelay") as MockRelay:
            MockRelay.from_env.return_value = relay
            hc._execute_browser_task(artifact, hc.openclaw_client)

        relay.request_sms_code.assert_called_once_with("reddit.com")

    def test_multiword_site_parsed(self, tmp_path):
        """Sites like 'google.com/accounts' are passed through intact."""
        hc = _hc(tmp_path)
        artifact = _artifact(hc)

        relay = MagicMock()
        relay.request_sms_code.return_value = "000001"

        hc.openclaw_client.send.side_effect = [
            "WAITING_FOR_SMS: google.com/accounts",
            "FINDING: logged in",
        ]

        with patch("nexus.core.house_c.TelegramRelay") as MockRelay:
            MockRelay.from_env.return_value = relay
            hc._execute_browser_task(artifact, hc.openclaw_client)

        relay.request_sms_code.assert_called_once_with("google.com/accounts")

    def test_normal_response_bypasses_relay(self, tmp_path):
        """When OpenClaw returns a normal FINDING, TelegramRelay is never called."""
        hc = _hc(tmp_path)
        artifact = _artifact(hc)

        hc.openclaw_client.send.return_value = "FINDING: u/charlie — great post"

        with patch("nexus.core.house_c.TelegramRelay") as MockRelay:
            hc._execute_browser_task(artifact, hc.openclaw_client)

        MockRelay.from_env.assert_not_called()


# ══════════════════════════════════════════════════════════════════
#  Second OpenClaw call with SMS code
# ══════════════════════════════════════════════════════════════════

class TestSecondOpenClawCall:
    def test_second_call_made_after_code_received(self, tmp_path):
        """After getting the SMS code, a second send() call is made to OpenClaw."""
        hc = _hc(tmp_path)
        artifact = _artifact(hc)

        relay = MagicMock()
        relay.request_sms_code.return_value = "654321"

        hc.openclaw_client.send.side_effect = [
            "WAITING_FOR_SMS: reddit.com",
            "FINDING: u/dave — job post",
        ]

        with patch("nexus.core.house_c.TelegramRelay") as MockRelay:
            MockRelay.from_env.return_value = relay
            hc._execute_browser_task(artifact, hc.openclaw_client)

        assert hc.openclaw_client.send.call_count == 2

    def test_second_call_includes_sms_code(self, tmp_path):
        """The second OpenClaw task text contains the SMS code."""
        hc = _hc(tmp_path)
        artifact = _artifact(hc)

        relay = MagicMock()
        relay.request_sms_code.return_value = "654321"

        hc.openclaw_client.send.side_effect = [
            "WAITING_FOR_SMS: reddit.com",
            "FINDING: u/dave — job post",
        ]

        with patch("nexus.core.house_c.TelegramRelay") as MockRelay:
            MockRelay.from_env.return_value = relay
            hc._execute_browser_task(artifact, hc.openclaw_client)

        second_call_arg = hc.openclaw_client.send.call_args_list[1][0][0]
        assert "654321" in second_call_arg

    def test_second_response_is_execution_proof(self, tmp_path):
        """The second OpenClaw response becomes the artifact's execution_proof."""
        hc = _hc(tmp_path)
        artifact = _artifact(hc)

        relay = MagicMock()
        relay.request_sms_code.return_value = "111222"

        hc.openclaw_client.send.side_effect = [
            "WAITING_FOR_SMS: reddit.com",
            "FINDING: u/eve — great opportunity",
        ]

        with patch("nexus.core.house_c.TelegramRelay") as MockRelay:
            MockRelay.from_env.return_value = relay
            result = hc._execute_browser_task(artifact, hc.openclaw_client)

        assert result.execution_proof == "FINDING: u/eve — great opportunity"

    def test_artifact_passes_validation_after_relay(self, tmp_path):
        """Artifact.passed_validation is True after a successful SMS relay."""
        hc = _hc(tmp_path)
        artifact = _artifact(hc)

        relay = MagicMock()
        relay.request_sms_code.return_value = "333444"

        hc.openclaw_client.send.side_effect = [
            "WAITING_FOR_SMS: reddit.com",
            "FINDING: u/frank — hiring now",
        ]

        with patch("nexus.core.house_c.TelegramRelay") as MockRelay:
            MockRelay.from_env.return_value = relay
            result = hc._execute_browser_task(artifact, hc.openclaw_client)

        assert result.passed_validation is True


# ══════════════════════════════════════════════════════════════════
#  Graceful failure paths
# ══════════════════════════════════════════════════════════════════

class TestGracefulFailure:
    def test_timeout_returns_failed_artifact(self, tmp_path):
        """When TelegramRelay times out (returns None), artifact fails gracefully."""
        hc = _hc(tmp_path)
        artifact = _artifact(hc)

        relay = MagicMock()
        relay.request_sms_code.return_value = None   # timeout

        hc.openclaw_client.send.return_value = "WAITING_FOR_SMS: reddit.com"

        with patch("nexus.core.house_c.TelegramRelay") as MockRelay:
            MockRelay.from_env.return_value = relay
            result = hc._execute_browser_task(artifact, hc.openclaw_client)

        assert result.passed_validation is False

    def test_timeout_does_not_crash(self, tmp_path):
        """A relay timeout never raises an exception."""
        hc = _hc(tmp_path)
        artifact = _artifact(hc)

        relay = MagicMock()
        relay.request_sms_code.return_value = None

        hc.openclaw_client.send.return_value = "WAITING_FOR_SMS: reddit.com"

        with patch("nexus.core.house_c.TelegramRelay") as MockRelay:
            MockRelay.from_env.return_value = relay
            try:
                hc._execute_browser_task(artifact, hc.openclaw_client)
            except Exception as exc:
                pytest.fail(f"_execute_browser_task raised on timeout: {exc}")

    def test_no_relay_env_fails_gracefully(self, tmp_path):
        """When TelegramRelay.from_env() returns None, artifact fails gracefully."""
        hc = _hc(tmp_path)
        artifact = _artifact(hc)

        hc.openclaw_client.send.return_value = "WAITING_FOR_SMS: reddit.com"

        with patch("nexus.core.house_c.TelegramRelay") as MockRelay:
            MockRelay.from_env.return_value = None   # no credentials
            result = hc._execute_browser_task(artifact, hc.openclaw_client)

        assert result.passed_validation is False

    def test_no_relay_env_sets_validation_error(self, tmp_path):
        """When relay is unavailable, validation_errors explains why."""
        hc = _hc(tmp_path)
        artifact = _artifact(hc)

        hc.openclaw_client.send.return_value = "WAITING_FOR_SMS: reddit.com"

        with patch("nexus.core.house_c.TelegramRelay") as MockRelay:
            MockRelay.from_env.return_value = None
            result = hc._execute_browser_task(artifact, hc.openclaw_client)

        assert result.validation_errors, "validation_errors must be non-empty"
        combined = " ".join(result.validation_errors).lower()
        assert "sms" in combined or "telegram" in combined or "verification" in combined

    def test_openclaw_none_response_fails_gracefully(self, tmp_path):
        """An OpenClaw None response is handled without raising."""
        hc = _hc(tmp_path)
        artifact = _artifact(hc)

        hc.openclaw_client.send.return_value = None

        with patch("nexus.core.house_c.TelegramRelay") as MockRelay:
            try:
                result = hc._execute_browser_task(artifact, hc.openclaw_client)
            except Exception as exc:
                pytest.fail(f"None response raised: {exc}")

        assert result.passed_validation is False

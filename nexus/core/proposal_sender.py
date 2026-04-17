"""ProposalSender — writes and dispatches proposals for every found opportunity.

After House C finds opportunities (FINDING: lines in execution_proof),
ProposalSender:
  1. Parses each FINDING: line into job_title / job_url / details
  2. Uses the LLM to write a professional proposal as the active identity
  3. Sends via Gmail SMTP when a contact email is present in the finding
  4. Notifies Telegram: "Proposal sent to [site/job]"
"""

from __future__ import annotations

import email.mime.multipart
import email.mime.text
import logging
import re
import smtplib
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_FINDING_PREFIX = "FINDING:"


@dataclass
class ProposalResult:
    """Result of one proposal attempt for a single finding."""

    job_title: str
    job_url: str
    proposal_text: str
    sent: bool = False       # True if emailed via SMTP
    notified: bool = False   # True if Telegram message sent
    error: str | None = None


class ProposalSender:
    """Writes and sends proposals for every FINDING discovered by House C.

    Args:
        router:           ModelRouter for LLM proposal generation.
        identity_manager: IdentityManager to resolve active identity.
        telegram:         Optional TelegramRelay for Telegram notifications.
    """

    def __init__(
        self,
        router: Any,
        identity_manager: Any,
        telegram: Any = None,
    ) -> None:
        self._router = router
        self._identity_manager = identity_manager
        self._telegram = telegram

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_proposal(
        self,
        job_title: str,
        job_description: str,
        identity: Any,
    ) -> str:
        """Use the LLM to write a professional proposal as the identity.

        Args:
            job_title:       Title of the job/opportunity.
            job_description: Full text of the job listing.
            identity:        The Identity to use as sender.

        Returns:
            Proposal text string.
        """
        system = (
            f"You are {identity.name}, a freelancer at {identity.business}.\n"
            f"Your bio: {identity.bio}\n"
            f"Your specialties: {', '.join(identity.niche)}\n\n"
            "Write a short, professional proposal (3-5 sentences) for the job below. "
            "Be specific, mention your relevant skills, and end with your availability. "
            "Do not add a subject line — just the proposal body."
        )
        user = (
            f"Job title: {job_title}\n\n"
            f"Job description:\n{job_description}"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            return self._router.complete(messages, label="proposal_writer")
        except Exception as exc:
            logger.warning("proposal_sender: LLM call failed: %s", exc)
            return (
                f"Hi, I am {identity.name} from {identity.business}. "
                f"I am interested in your {job_title} listing and believe my skills "
                f"in {', '.join(identity.niche[:2])} are a great fit. "
                f"Please feel free to reach out to discuss further."
            )

    def send_via_gmail(
        self,
        to_email: str,
        subject: str,
        body: str,
        identity: Any,
        gmail_password: str,
    ) -> bool:
        """Send proposal via Gmail SMTP as the identity's email address.

        Returns True on success, False on any error (never raises).
        """
        resolved_email = self._identity_manager.resolve_field(identity.email)
        if not resolved_email or not gmail_password:
            logger.warning(
                "proposal_sender: skipping Gmail send — "
                "email=%r password=%s",
                resolved_email, "set" if gmail_password else "missing",
            )
            return False

        msg = email.mime.multipart.MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = resolved_email
        msg["To"] = to_email
        msg.attach(email.mime.text.MIMEText(body, "plain", "utf-8"))

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                smtp.login(resolved_email, gmail_password)
                smtp.sendmail(resolved_email, to_email, msg.as_string())
            logger.info(
                "proposal_sender: email sent  from=%s  to=%s",
                resolved_email, to_email,
            )
            return True
        except Exception as exc:
            logger.warning("proposal_sender: Gmail SMTP failed: %s", exc)
            return False

    def notify_telegram(self, site_or_job: str) -> bool:
        """Send a Telegram notification about a proposal.

        Returns True if sent, False if relay unavailable (never raises).
        """
        if self._telegram is None:
            return False
        msg = f"Proposal sent to {site_or_job}"
        try:
            return bool(self._telegram.send_message(msg))
        except Exception as exc:
            logger.warning("proposal_sender: Telegram notify failed: %s", exc)
            return False

    def process_findings(
        self,
        findings_text: str,
        identity: Any,
    ) -> list[ProposalResult]:
        """Parse FINDING: lines and dispatch proposals for each.

        For each FINDING:
          - Generates a proposal via LLM
          - Sends via Gmail if a contact email is found in the finding
          - Notifies Telegram regardless of email presence

        Args:
            findings_text: Raw execution_proof string containing FINDING: lines.
            identity:      The Identity to send proposals as.

        Returns:
            List of ProposalResult, one per FINDING: line.
        """
        results: list[ProposalResult] = []

        if not findings_text:
            return results

        # Resolve Gmail credentials once
        gmail_pass = self._identity_manager.resolve_field(
            "vault:GMAIL_APP_PASS"
        )
        if gmail_pass.startswith("vault:"):
            # Not found in vault — try env
            import os
            gmail_pass = (
                os.getenv("GMAIL_APP_PASS", "").strip()
                or os.getenv("GMAIL_PASS", "").strip()
            )

        for line in findings_text.splitlines():
            line = line.strip()
            if not line.startswith(_FINDING_PREFIX):
                continue

            content = line[len(_FINDING_PREFIX):].strip()
            parts = [p.strip() for p in content.split("|")]

            job_title = parts[0] if len(parts) > 0 else content
            job_url = parts[1] if len(parts) > 1 else ""
            details = " | ".join(parts[2:]) if len(parts) > 2 else content

            # Generate proposal
            proposal = self.generate_proposal(job_title, details, identity)

            # Extract contact email from finding
            contact_email_match = _EMAIL_RE.search(content)
            contact_email = contact_email_match.group(0) if contact_email_match else None

            # Send via Gmail if contact email found
            sent = False
            if contact_email:
                subject = f"Proposal: {job_title} — {identity.name}, {identity.business}"
                sent = self.send_via_gmail(
                    to_email=contact_email,
                    subject=subject,
                    body=proposal,
                    identity=identity,
                    gmail_password=gmail_pass,
                )

            # Notify Telegram
            target = job_url or job_title
            notified = self.notify_telegram(target)

            results.append(ProposalResult(
                job_title=job_title,
                job_url=job_url,
                proposal_text=proposal,
                sent=sent,
                notified=notified,
            ))

            logger.info(
                "proposal_sender: processed finding  title=%r  sent=%s  notified=%s",
                job_title[:60], sent, notified,
            )

        return results

"""Tests for the two blockers identified after the r/forhire cycle run.

Fix 1 — _needs_browser() must check original_input, not just redefined_problem.
    Problem: House B rewrites the task and strips "Reddit DM" from the
    redefined_problem.  _needs_browser() sees only the redefined text and
    returns False, so the browser path is never taken.
    Fix: include sso.original_input in the text that is scanned.

Fix 2 — Reddit credentials in vault.
    vault_store_credentials.CREDENTIALS must prompt for
    REDDIT_USERNAME and REDDIT_PASSWORD so the operator can
    store them without editing source files.
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock

import pytest

from nexus.core.house_c import HouseC, _BROWSER_ACTION_KEYWORDS
from nexus.core.house_b import StructuredSpecificationObject
from nexus.core.knowledge_graph import KnowledgeGraph
import scripts.vault_store_credentials as vault_script


# ── helpers ────────────────────────────────────────────────────────────────────

def _sso(
    original: str,
    redefined: str | None = None,
    domain: str = "freelance",
) -> StructuredSpecificationObject:
    return StructuredSpecificationObject(
        original_input=original,
        redefined_problem=redefined if redefined is not None else original,
        domain=domain,
        constraints=[],
        success_criteria=["succeed"],
    )


# ══════════════════════════════════════════════════════════════════
#  Fix 1 — _needs_browser() checks original_input
# ══════════════════════════════════════════════════════════════════

class TestNeedsBrowserChecksOriginalInput:
    """_needs_browser must return True based on original_input even when
    the redefined_problem no longer contains the browser trigger phrase."""

    def test_true_when_original_says_reddit_dm_and_redefined_does_not(self):
        sso = _sso(
            original="Find a job on r/forhire and send a professional proposal via Reddit DM",
            redefined="Create a compliance-first freelance opportunity discovery system",
        )
        assert HouseC._needs_browser(sso) is True, (
            "_needs_browser must catch 'Reddit DM' in original_input "
            "even when redefined_problem omits it"
        )

    def test_true_when_original_says_send_dm(self):
        sso = _sso(
            original="send dm to r/forhire poster about my services",
            redefined="Build an outreach pipeline for freelance prospects",
        )
        assert HouseC._needs_browser(sso) is True

    def test_true_when_original_says_direct_message(self):
        sso = _sso(
            original="Send a direct message to the top poster on r/forhire",
            redefined="Identify the highest-engagement post on a freelance subreddit",
        )
        assert HouseC._needs_browser(sso) is True

    def test_true_when_original_says_send_reddit_dm(self):
        sso = _sso(
            original="send reddit dm to freelancers hiring devs",
            redefined="Research development job postings",
        )
        assert HouseC._needs_browser(sso) is True

    def test_true_when_original_says_message_the_poster(self):
        sso = _sso(
            original="message the poster and pitch my writing services",
            redefined="Generate a pitch for a freelance writing opportunity",
        )
        assert HouseC._needs_browser(sso) is True

    def test_still_true_when_redefined_contains_keyword(self):
        """Backward-compat: redefined_problem alone must still trigger True."""
        sso = _sso(
            original="find some gigs",
            redefined="navigate to r/forhire and send dm to relevant posters",
        )
        assert HouseC._needs_browser(sso) is True

    def test_false_when_neither_original_nor_redefined_contain_keyword(self):
        sso = _sso(
            original="find jobs on r/forhire via public JSON API",
            redefined="scrape r/forhire using the public Reddit API and return results",
        )
        assert HouseC._needs_browser(sso) is False, (
            "Pure scraping without DM/browser keywords must remain False"
        )

    def test_exact_failing_input_from_production_cycle(self):
        """Regression: the exact input that previously slipped through."""
        sso = _sso(
            original="Find a job on r/forhire and send a professional proposal via Reddit DM",
            redefined=(
                "Create a system that helps freelancers manually identify and evaluate "
                "suitable job opportunities on Reddit's r/forhire subreddit, providing "
                "structured analysis and draft response templates that users can manually "
                "adapt and send through legitimate Reddit channels."
            ),
        )
        assert HouseC._needs_browser(sso) is True, (
            "This exact production input must route to the browser path"
        )

    def test_domain_alone_does_not_override_false(self):
        """A domain that contains 'freelance' must not be treated as a browser signal."""
        sso = _sso(
            original="research top earners on r/forhire",
            redefined="analyse salary data from the r/forhire subreddit",
            domain="Freelance Research",
        )
        assert HouseC._needs_browser(sso) is False


# ══════════════════════════════════════════════════════════════════
#  Fix 2 — Reddit credentials in vault_store_credentials
# ══════════════════════════════════════════════════════════════════

class TestRedditCredentialsInVaultScript:
    """vault_store_credentials.CREDENTIALS must include Reddit keys."""

    def _keys(self) -> list[str]:
        return [key for key, _ in vault_script.CREDENTIALS]

    def test_reddit_username_key_present(self):
        assert "REDDIT_USERNAME" in self._keys(), (
            "CREDENTIALS must include REDDIT_USERNAME"
        )

    def test_reddit_password_key_present(self):
        assert "REDDIT_PASSWORD" in self._keys(), (
            "CREDENTIALS must include REDDIT_PASSWORD"
        )

    def test_reddit_username_has_label(self):
        labels = {k: v for k, v in vault_script.CREDENTIALS}
        assert labels.get("REDDIT_USERNAME"), "REDDIT_USERNAME must have a non-empty label"

    def test_reddit_password_has_label(self):
        labels = {k: v for k, v in vault_script.CREDENTIALS}
        assert labels.get("REDDIT_PASSWORD"), "REDDIT_PASSWORD must have a non-empty label"

    def test_reddit_entries_are_after_existing_entries(self):
        """Reddit entries should appear after the original credential set."""
        keys = self._keys()
        gmail_idx = keys.index("GMAIL_ADDRESS")
        reddit_idx = keys.index("REDDIT_USERNAME")
        assert reddit_idx > gmail_idx, (
            "Reddit credentials should be added after existing entries"
        )

    def test_all_credentials_have_nonempty_key_and_label(self):
        for key, label in vault_script.CREDENTIALS:
            assert key and key.strip(), "Every credential must have a non-empty key"
            assert label and label.strip(), f"Credential {key!r} must have a non-empty label"

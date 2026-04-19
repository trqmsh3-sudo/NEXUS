"""OpenClawAIController — DeepSeek vision-guided browser automation loop.

Replaces keyword-matching with a screenshot→DeepSeek→action feedback loop:
  1. Take screenshot via OpenClaw /screenshot
  2. Send screenshot + task to DeepSeek vision API
  3. Parse the JSON action DeepSeek returns
  4. Execute the action via OpenClaw /action
  5. Repeat until DeepSeek returns task_complete or MAX_STEPS is reached

Supported action types:
  click        {"type": "click", "x": int, "y": int}
  type         {"type": "type", "text": str}
  scroll       {"type": "scroll", "direction": "up"|"down", "amount": int}
  extract_data {"type": "extract_data", "data": str}   — data collected, not forwarded
  task_complete {"type": "task_complete", "data": str} — ends the loop
"""

from __future__ import annotations

import json
import logging

import litellm

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a browser automation AI. You will be shown a screenshot of a web page "
    "and a task. Analyze the screenshot and decide the SINGLE next action to take.\n\n"
    "Return ONLY valid JSON — no markdown, no explanation:\n"
    '{"type": "click", "x": 150, "y": 300, "description": "clicking the login button"}\n'
    '{"type": "type", "text": "hello world", "description": "typing the search query"}\n'
    '{"type": "scroll", "direction": "down", "amount": 500, "description": "scrolling to see more"}\n'
    '{"type": "extract_data", "data": "FINDING: Python dev | example.com | $5000/mo | Remote", "description": "extracting job listing"}\n'
    '{"type": "task_complete", "data": "summary of all findings", "description": "task is done"}\n\n'
    "Rules:\n"
    "- For extract_data: start data with 'FINDING:' so downstream parsers recognise it\n"
    "- Return task_complete when: the goal is achieved, nothing more to do, or after 3+ extractions\n"
    "- Return task_complete if the page has no relevant data and there are no obvious next steps\n"
    "- IMPORTANT: If you see a login form, sign-in page, signup wall, CAPTCHA, or access-denied page, "
    "respond IMMEDIATELY with: {\"type\": \"task_complete\", \"data\": \"NO_DATA: login required\", "
    "\"description\": \"blocked by login wall\"} — do NOT try to fill in credentials\n"
    "- Never return status messages — only actionable JSON"
)

_NON_FORWARDED = {"extract_data", "task_complete"}

_DEFAULT_START_URL = "https://remoteok.com"

# Ordered: first match wins. Keys are lowercase substrings to detect in task.
_SITE_MAP: list[tuple[str, str]] = [
    ("remoteok.com",        "https://remoteok.com"),
    ("remote.co",           "https://remote.co/remote-jobs/"),
    ("weworkremotely",      "https://weworkremotely.com"),
    ("indeed.com",          "https://www.indeed.com"),
    ("freelancer.com",      "https://www.freelancer.com/projects"),
    ("fiverr.com",          "https://www.fiverr.com"),
    ("peopleperhour",       "https://www.peopleperhour.com"),
]

# Fallback URLs tried when stuck (login walls, no content). No-login sites first.
_FALLBACK_URLS: list[str] = [
    "https://remoteok.com",
    "https://weworkremotely.com",
    "https://remote.co/remote-jobs/",
    "https://www.indeed.com",
]

_STUCK_THRESHOLD = 3  # consecutive same-type forwarded actions → re-navigate


class OpenClawAIController:
    """Vision-based browser control loop powered by DeepSeek + OpenClaw.

    Args:
        client:  An :class:`~nexus.core.openclaw_client.OpenClawClient` instance.
        api_key: DeepSeek API key (read from GuardianVault by callers).
        model:   litellm model string; defaults to ``deepseek/deepseek-chat``.
    """

    MAX_STEPS: int = 15

    def __init__(
        self,
        client,
        api_key: str,
        model: str = "deepseek/deepseek-chat",
    ) -> None:
        self.client  = client
        self.api_key = api_key
        self.model   = model

    # ──────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────

    def _extract_start_url(self, task: str) -> str:
        """Return the best start URL for the task, or the default."""
        lower = task.lower()
        for keyword, url in _SITE_MAP:
            if keyword in lower:
                return url
        return _DEFAULT_START_URL

    def run(self, task: str) -> str:
        """Navigate to the task URL then execute the vision-action loop.

        Returns all FINDING lines joined by newlines, or an empty string
        if nothing was found or the loop failed before completing.
        """
        findings: list[str] = []

        start_url = self._extract_start_url(task)
        logger.info("OpenClawAIController: navigating to %s", start_url)
        self.client.execute_action({"type": "navigate", "url": start_url})

        # Stuck detection: track consecutive forwarded action type
        consecutive_type: str | None = None
        consecutive_count: int = 0
        current_url: str = start_url
        fallback_index: int = 0

        for step in range(self.MAX_STEPS):
            screenshot_b64 = self.client.screenshot()
            if not screenshot_b64:
                logger.warning(
                    "OpenClawAIController: empty screenshot at step %d — aborting", step
                )
                break

            action = self._decide_action(task, screenshot_b64, findings)
            if action is None:
                break

            action_type = action.get("type", "")
            logger.info("OpenClawAIController: step=%d action=%s", step, action_type)

            if action_type == "task_complete":
                data = action.get("data", "")
                if data and not findings:
                    findings.append(data)
                break

            if action_type == "extract_data":
                data = action.get("data", "")
                if data:
                    findings.append(data)
                # extract_data does not count toward stuck
                consecutive_type = None
                consecutive_count = 0
            elif action_type not in _NON_FORWARDED:
                # Track consecutive forwarded actions
                if action_type == consecutive_type:
                    consecutive_count += 1
                else:
                    consecutive_type = action_type
                    consecutive_count = 1

                if consecutive_count >= _STUCK_THRESHOLD:
                    # Pick next fallback that isn't the current URL
                    fallback_url: str | None = None
                    while fallback_index < len(_FALLBACK_URLS):
                        candidate = _FALLBACK_URLS[fallback_index]
                        fallback_index += 1
                        if candidate != current_url:
                            fallback_url = candidate
                            break
                    if fallback_url is not None:
                        logger.warning(
                            "OpenClawAIController: stuck (%d× %s) at step %d — "
                            "re-navigating to %s",
                            consecutive_count, action_type, step, fallback_url,
                        )
                        self.client.execute_action({"type": "navigate", "url": fallback_url})
                        current_url = fallback_url
                        consecutive_type = None
                        consecutive_count = 0
                    else:
                        logger.warning(
                            "OpenClawAIController: stuck and all fallbacks exhausted — aborting"
                        )
                        break
                else:
                    self.client.execute_action(action)

        return "\n".join(findings)

    # ──────────────────────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────────────────────

    def _decide_action(
        self,
        task: str,
        screenshot_b64: str,
        findings_so_far: list[str],
    ) -> dict | None:
        """Call DeepSeek vision API and return a parsed action dict, or None."""
        context = ""
        if findings_so_far:
            recent = findings_so_far[-3:]
            context = (
                f"\n\nFindings so far ({len(findings_so_far)}):\n"
                + "\n".join(recent)
            )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{screenshot_b64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": f"TASK: {task}{context}\n\nWhat is the single next action?",
                    },
                ],
            },
        ]

        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                api_key=self.api_key,
                max_tokens=200,
                temperature=0,
            )
            content = response.choices[0].message.content.strip()
            return self._parse_action(content)
        except Exception as exc:
            logger.warning("OpenClawAIController: DeepSeek call failed — %s", exc)
            return None

    def _parse_action(self, content: str) -> dict | None:
        """Parse action JSON from a DeepSeek response, stripping markdown fences."""
        content = content.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            # drop opening fence line; drop closing fence if present
            inner = lines[1:]
            if inner and inner[-1].strip() == "```":
                inner = inner[:-1]
            content = "\n".join(inner).strip()

        try:
            action = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.warning(
                "OpenClawAIController: JSON parse failed — %s — content=%r",
                exc, content[:120],
            )
            return None

        if not isinstance(action, dict) or "type" not in action:
            logger.warning(
                "OpenClawAIController: invalid action structure — content=%r", content[:120]
            )
            return None

        return action

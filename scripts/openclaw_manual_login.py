"""manual_login.py — One-time manual Google login for OpenClaw.

Deployed to: /opt/openclaw/manual_login.py

HOW TO USE:
  1. On your local machine, open a new terminal and run:
         ssh -L 9222:localhost:9222 root@46.101.115.140
     Keep this terminal open.

  2. On the server:
         systemctl stop openclaw
         python3 /opt/openclaw/manual_login.py

  3. Open Google Chrome (NOT Firefox) on your local machine.
     Go to:  chrome://inspect
     Click: "Configure..."  → add  localhost:9222  → Done
     Under "Remote Target", click "inspect" on the page that appears.

  4. In the DevTools window that opens:
     - Click the "Console" tab
     - Type: window.location.href = 'https://accounts.google.com'
     - Press Enter — the browser will navigate to Google login
     - Complete the login manually in the DevTools window

  5. After login succeeds, go back to the server terminal and press Ctrl+C.
     The session is now saved to the persistent profile.

  6. Restart OpenClaw:
         systemctl start openclaw

  7. Verify session:
         python3 /opt/openclaw/session_manager.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("manual_login")

PROFILE_DIR = Path("/opt/openclaw/browser_profile")
REMOTE_DEBUG_PORT = 9222

LAUNCH_ARGS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-blink-features=AutomationControlled",
    f"--remote-debugging-port={REMOTE_DEBUG_PORT}",
    f"--remote-debugging-address=127.0.0.1",
    "--disable-features=IsolateOrigins,site-per-process",
]

ANTI_DETECT_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
window.chrome = {runtime: {}};
"""


async def run_login_session() -> None:
    from playwright.async_api import async_playwright

    log.info("=" * 60)
    log.info("OpenClaw Manual Login Tool")
    log.info("=" * 60)
    log.info("Profile: %s", PROFILE_DIR)
    log.info("")
    log.info("INSTRUCTIONS:")
    log.info("  1. On your LOCAL machine, open a new terminal:")
    log.info("     ssh -L %d:localhost:%d root@<server-ip>", REMOTE_DEBUG_PORT, REMOTE_DEBUG_PORT)
    log.info("  2. In Chrome on your local machine:")
    log.info("     Go to: chrome://inspect")
    log.info("     Configure → add: localhost:%d", REMOTE_DEBUG_PORT)
    log.info("     Click 'inspect' under Remote Target")
    log.info("  3. In the DevTools console, navigate to Google and log in.")
    log.info("  4. When done, press Ctrl+C here.")
    log.info("=" * 60)

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as pw:
        ctx = await pw.chromium.launch_persistent_context(
            str(PROFILE_DIR),
            headless=True,
            args=LAUNCH_ARGS,
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.6099.109 Safari/537.36"
            ),
            locale="en-US",
            timezone_id="America/New_York",
            viewport={"width": 1280, "height": 900},
        )
        await ctx.add_init_script(ANTI_DETECT_SCRIPT)

        page = await ctx.new_page()
        await page.goto("https://accounts.google.com", timeout=30000)
        log.info("Browser launched. Waiting for you to log in remotely...")
        log.info("Remote debugging available at: http://localhost:%d", REMOTE_DEBUG_PORT)

        # Keep alive until Ctrl+C
        try:
            while True:
                await asyncio.sleep(5)
                title = await page.title()
                url = page.url
                log.info("Current page: %s | %s", title[:60], url[:80])

                # Detect successful Google login
                if "myaccount.google.com" in url or (
                    "google.com" in url and "signin" not in url and "accounts" not in url
                ):
                    log.info("Google login detected! Session saved.")
        except asyncio.CancelledError:
            pass

        log.info("Saving session cookies...")
        try:
            cookies = await ctx.cookies()
            google_cookies = [c for c in cookies if "google" in c.get("domain", "")]
            log.info("Google cookies captured: %d", len(google_cookies))
        except Exception as e:
            log.warning("Cookie capture error: %s", e)

        await ctx.close()
        log.info("Session saved to profile: %s", PROFILE_DIR)
        log.info("Run 'python3 session_manager.py' to verify.")


def main() -> None:
    if os.geteuid() != 0:
        log.warning("Run as root for best results (profile may be root-owned).")

    loop = asyncio.new_event_loop()

    task = loop.create_task(run_login_session())

    def _shutdown(sig, frame):
        log.info("Ctrl+C received — saving and closing browser...")
        task.cancel()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        loop.run_until_complete(task)
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        loop.close()

    log.info("Done. Restart OpenClaw: systemctl start openclaw")


if __name__ == "__main__":
    main()

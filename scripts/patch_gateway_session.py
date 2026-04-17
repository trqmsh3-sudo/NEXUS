"""Patch gateway.py to:
1. Import session_manager at top
2. In run_browser_task, check session before doing anything Google-related
3. Add session status to health endpoint
"""
from pathlib import Path
import re

src = Path('/opt/openclaw/gateway.py').read_text()

# 1. Add import after existing imports
old_import = 'from generic_handler import run_generic_browser_task'
new_import = (
    'from generic_handler import run_generic_browser_task\n'
    'from session_manager import (\n'
    '    is_google_session_valid, google_session_hours_remaining,\n'
    '    session_summary, backup_session, PROFILE_DIR,\n'
    ')'
)
if old_import not in src:
    print("ERROR: import anchor not found")
    exit(1)
src = src.replace(old_import, new_import, 1)

# 2. Patch run_browser_task to check session and log status
old_task_start = (
    'async def run_browser_task(task: str) -> str:\n'
    '    log.info(\'Task: %s\', task[:120])\n'
    '    async with async_playwright() as pw:\n'
    '        ctx = await new_context(pw)\n'
    '        try:\n'
    '            task_lower = task.lower()\n'
    '            if any(kw in task_lower for kw in [\'reddit\', \'r/forhire\', \'dm\', \'direct message\']):\n'
    '                logged_in = await ensure_reddit_logged_in(ctx)\n'
    '                if not logged_in:\n'
    '                    return \'WAITING_FOR_SMS: reddit.com\'\n'
    '                return await find_and_dm_forhire(ctx)\n'
    '            else:\n'
    '                return await run_generic_browser_task(ctx, task)'
)

new_task_start = (
    'async def run_browser_task(task: str) -> str:\n'
    '    log.info(\'Task: %s\', task[:120])\n'
    '    # Log session status on each task for visibility\n'
    '    hrs = google_session_hours_remaining()\n'
    '    if hrs > 0:\n'
    '        log.info(\'Google session valid — %.0fh remaining\', hrs)\n'
    '    else:\n'
    '        log.warning(\'Google session INVALID — run manual_login.py to re-authenticate\')\n'
    '    async with async_playwright() as pw:\n'
    '        ctx = await new_context(pw)\n'
    '        try:\n'
    '            task_lower = task.lower()\n'
    '            if any(kw in task_lower for kw in [\'reddit\', \'r/forhire\', \'dm\', \'direct message\']):\n'
    '                logged_in = await ensure_reddit_logged_in(ctx)\n'
    '                if not logged_in:\n'
    '                    return \'WAITING_FOR_SMS: reddit.com\'\n'
    '                return await find_and_dm_forhire(ctx)\n'
    '            else:\n'
    '                return await run_generic_browser_task(ctx, task)'
)

if old_task_start not in src:
    print("ERROR: run_browser_task anchor not found — printing candidates:")
    idx = src.find('async def run_browser_task')
    print(repr(src[idx:idx+400]))
else:
    src = src.replace(old_task_start, new_task_start, 1)
    print("PATCHED: run_browser_task with session check")

# 3. Add session status to /health endpoint
old_health = (
    "@app.get('/health')\n"
    "async def health():\n"
    "    return {'status': 'ok', 'service': 'openclaw-gateway', 'version': '2'}"
)
new_health = (
    "@app.get('/health')\n"
    "async def health():\n"
    "    sess = session_summary()\n"
    "    return {\n"
    "        'status': 'ok',\n"
    "        'service': 'openclaw-gateway',\n"
    "        'version': '3',\n"
    "        'google_session': sess,\n"
    "    }"
)
if old_health in src:
    src = src.replace(old_health, new_health, 1)
    print("PATCHED: /health endpoint with session status")
else:
    print("WARNING: /health anchor not found — skipping")

Path('/opt/openclaw/gateway.py').write_text(src)
print("gateway.py written OK")

# Verify import
import importlib.util, sys
spec = importlib.util.spec_from_file_location('sm', '/opt/openclaw/session_manager.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print("session_manager import OK")
print("Session valid:", mod.is_google_session_valid())
print("Hours remaining:", round(mod.google_session_hours_remaining(), 0))

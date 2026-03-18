"""Run belief executable_proof in an isolated subprocess (FIX 2)."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from typing import Final

logger = logging.getLogger(__name__)

DEFAULT_PROOF_TIMEOUT: Final[float] = 60.0


def run_executable_proof_in_subprocess(
    code: str,
    *,
    timeout: float = DEFAULT_PROOF_TIMEOUT,
) -> tuple[bool, str]:
    """Execute *code* as a Python script in a fresh interpreter process.

    Returns:
        (True, "") on success; (False, error_message) on failure or timeout.
    """
    text = (code or "").strip()
    if not text:
        return False, "empty executable_proof"

    fd, path = tempfile.mkstemp(suffix="_nexus_proof.py", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
            return False, msg[:800]
        return True, ""
    except subprocess.TimeoutExpired:
        return False, f"executable_proof timed out after {timeout}s"
    except OSError as exc:
        return False, str(exc)
    except Exception as exc:  # pragma: no cover
        logger.debug("proof subprocess error: %s", exc)
        return False, str(exc)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

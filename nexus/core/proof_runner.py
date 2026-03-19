"""Run belief executable_proof in an isolated subprocess (FIX 2). STEP 2: 30s timeout, kill/wait, semaphore."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import threading
from typing import Final

logger = logging.getLogger(__name__)

DEFAULT_PROOF_TIMEOUT: Final[float] = 30.0
_subprocess_semaphore: threading.Semaphore = threading.Semaphore(1)


def run_executable_proof_in_subprocess(
    code: str,
    *,
    timeout: float = DEFAULT_PROOF_TIMEOUT,
) -> tuple[bool, str]:
    """Execute *code* as a Python script. At most one subprocess at a time; 30s timeout; explicit kill/wait."""
    text = (code or "").strip()
    if not text:
        return False, "empty executable_proof"

    with _subprocess_semaphore:
        fd, path = tempfile.mkstemp(suffix="_nexus_proof.py", text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
                if not text.endswith("\n"):
                    f.write("\n")
            proc = subprocess.Popen(
                [sys.executable, path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                return False, f"executable_proof timed out after {timeout}s"
            if proc.returncode is None:
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            if proc.returncode != 0:
                msg = (stderr or stdout or "").strip() or f"exit {proc.returncode}"
                return False, msg[:800]
            return True, ""
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

"""Persistent storage for the NEXUS Knowledge Graph.

Provides atomic saves and crash-resistant loads. Only valid, non-expired
beliefs are ever persisted. Corrupted storage never crashes the system.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from nexus.core.belief_certificate import BeliefCertificate

if TYPE_CHECKING:
    from nexus.core.knowledge_graph import KnowledgeGraph

logger: logging.Logger = logging.getLogger(__name__)


class PersistenceManager:
    """Manages durable storage of BeliefCertificates to disk.

    Uses atomic writes (temp file + rename) to prevent corruption on
    crash. Invalid or expired beliefs are never saved. Corrupted files
    are skipped with a warning, and the system starts fresh.
    """

    def __init__(self, storage_path: str = "data/knowledge_store/beliefs.json") -> None:
        self.storage_path: str = storage_path

        # Set by load() for callers to inspect
        self.last_load_count: int = 0
        self.last_skip_count: int = 0

    def save(self, graph: KnowledgeGraph) -> None:
        """Serialize valid, non-expired beliefs to JSON and save atomically.

        Writes to a temp file in the same directory, then renames over
        the target. Prevents partial writes from corrupting the store.

        Args:
            graph: The KnowledgeGraph whose beliefs to persist.
        """
        valid_beliefs = [
            b for b in graph.beliefs.values()
            if b.is_valid() and not b.is_expired()
        ]
        data = [b.to_dict() for b in valid_beliefs]
        path = Path(self.storage_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(
            dir=path.parent,
            prefix=".beliefs.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except OSError:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        logger.info(
            "PERSISTENCE saved  path=%s  beliefs=%d",
            self.storage_path, len(valid_beliefs),
        )

    def load(self) -> list[BeliefCertificate]:
        """Load beliefs from disk, skipping invalid or corrupted entries.

        Skips any belief that:
        - Fails deserialization (logs and continues)
        - Is expired
        - Fails is_valid()

        On corrupted storage or missing file, returns [] and logs a
        warning. Never raises.

        Returns:
            List of clean, valid, non-expired BeliefCertificates.
        """
        path = Path(self.storage_path)
        loaded: list[BeliefCertificate] = []
        skipped = 0

        if not path.exists():
            logger.info("PERSISTENCE no file  path=%s  loaded=0  skipped=0", path)
            self.last_load_count = 0
            self.last_skip_count = 0
            return loaded

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "PERSISTENCE corrupted or unreadable  path=%s  error=%s  "
                "starting fresh",
                path, exc,
            )
            self.last_load_count = 0
            self.last_skip_count = 0
            return []

        if not isinstance(raw, list):
            logger.warning(
                "PERSISTENCE invalid format (expected list)  path=%s  "
                "starting fresh",
                path,
            )
            self.last_load_count = 0
            self.last_skip_count = 1
            return []

        for item in raw:
            if not isinstance(item, dict):
                skipped += 1
                continue
            try:
                belief = BeliefCertificate.from_dict(item)
            except (KeyError, ValueError, TypeError) as exc:
                logger.debug("PERSISTENCE skip deserialization  item=%s  error=%s", item, exc)
                skipped += 1
                continue
            if belief.is_expired():
                skipped += 1
                continue
            if not belief.is_valid():
                skipped += 1
                continue
            loaded.append(belief)

        self.last_load_count = len(loaded)
        self.last_skip_count = skipped
        logger.info(
            "PERSISTENCE loaded  path=%s  loaded=%d  skipped=%d",
            path, len(loaded), skipped,
        )
        return loaded

    def auto_save(self, graph: KnowledgeGraph) -> None:
        """Persist the graph after a successful belief injection.

        Args:
            graph: The KnowledgeGraph to save.
        """
        self.save(graph)

"""Supabase PostgreSQL persistence with JSON file fallback.

Set SUPABASE_URL and SUPABASE_KEY in the environment. Run supabase_schema.sql
in the Supabase SQL editor to create tables.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SINGLETON_ID = "singleton"

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
logger.info("SUPABASE_URL: %s", "SET" if url else "NOT SET")
logger.info("SUPABASE_KEY: %s", "SET" if key else "NOT SET")


def is_supabase_enabled() -> bool:
    url = (os.getenv("SUPABASE_URL") or "").strip()
    key = (os.getenv("SUPABASE_KEY") or "").strip()
    return bool(url and key)


def _claim_hash(claim: str) -> str:
    return hashlib.sha256(claim.encode("utf-8")).hexdigest()


def _client() -> Any:
    from supabase import create_client
    return create_client(
        os.environ["SUPABASE_URL"].strip(),
        os.environ["SUPABASE_KEY"].strip(),
    )


# ---------------------------------------------------------------------------
# Beliefs
# ---------------------------------------------------------------------------


def save_beliefs(belief_dicts: list[dict[str, Any]]) -> bool:
    """Persist beliefs to Supabase. Returns True on success."""
    if not is_supabase_enabled():
        return False
    try:
        client = _client()
        want: dict[str, dict[str, Any]] = {}
        for d in belief_dicts:
            c = d.get("claim") or ""
            if not c:
                continue
            want[_claim_hash(c)] = d
        for h, d in want.items():
            client.table("beliefs").upsert(
                {"claim_hash": h, "data": d},
            ).execute()
        resp = client.table("beliefs").select("claim_hash").execute()
        for row in resp.data or []:
            ch = row.get("claim_hash")
            if ch and ch not in want:
                client.table("beliefs").delete().eq("claim_hash", ch).execute()
        logger.info("DATABASE beliefs saved to Supabase  count=%d", len(want))
        return True
    except Exception as exc:
        logger.warning("DATABASE beliefs Supabase save failed: %s", exc)
        return False


def load_belief_dicts() -> list[dict[str, Any]] | None:
    """Load belief dicts from Supabase. Returns None to signal fallback to JSON file."""
    if not is_supabase_enabled():
        return None
    try:
        client = _client()
        resp = client.table("beliefs").select("data").execute()
        rows = resp.data or []
        out = [r["data"] for r in rows if isinstance(r.get("data"), dict)]
        logger.info("DATABASE beliefs loaded from Supabase  count=%d", len(out))
        return out
    except Exception as exc:
        logger.warning("DATABASE beliefs Supabase load failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Cycle history
# ---------------------------------------------------------------------------


def save_cycle_history(entries: list[dict[str, Any]], fallback_path: Path) -> None:
    if is_supabase_enabled():
        try:
            _client().table("cycle_history").upsert(
                {"id": _SINGLETON_ID, "entries": entries},
            ).execute()
            logger.info("DATABASE cycle_history saved to Supabase  n=%d", len(entries))
            return
        except Exception as exc:
            logger.warning("DATABASE cycle_history Supabase failed, using file: %s", exc)
    try:
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        fallback_path.write_text(
            json.dumps(entries, indent=2, default=str), encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("DATABASE cycle_history file save failed: %s", exc)


def load_cycle_history(fallback_path: Path) -> list[dict[str, Any]]:
    if is_supabase_enabled():
        try:
            resp = (
                _client()
                .table("cycle_history")
                .select("entries")
                .eq("id", _SINGLETON_ID)
                .limit(1)
                .execute()
            )
            if resp.data and len(resp.data) > 0:
                ent = resp.data[0].get("entries")
                if isinstance(ent, list):
                    logger.info(
                        "DATABASE cycle_history loaded from Supabase  n=%d", len(ent),
                    )
                    return ent
        except Exception as exc:
            logger.warning("DATABASE cycle_history Supabase load failed: %s", exc)
    try:
        if fallback_path.exists():
            raw = json.loads(fallback_path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, list) else []
    except (OSError, json.JSONDecodeError):
        pass
    return []


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------


def save_skills(skill_dicts: list[dict[str, Any]], fallback_path: Path) -> None:
    if is_supabase_enabled():
        try:
            client = _client()
            want_ids = set()
            for s in skill_dicts:
                sid = s.get("skill_id")
                if not sid:
                    continue
                want_ids.add(sid)
                client.table("skills").upsert(
                    {"skill_id": sid, "data": s},
                ).execute()
            resp = client.table("skills").select("skill_id").execute()
            for row in resp.data or []:
                sid = row.get("skill_id")
                if sid and sid not in want_ids:
                    client.table("skills").delete().eq("skill_id", sid).execute()
            logger.info("DATABASE skills saved to Supabase  n=%d", len(want_ids))
            return
        except Exception as exc:
            logger.warning("DATABASE skills Supabase failed, using file: %s", exc)
    try:
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        fallback_path.write_text(
            json.dumps(skill_dicts, indent=2, default=str), encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("DATABASE skills file save failed: %s", exc)


def load_skills(fallback_path: Path) -> list[dict[str, Any]]:
    if is_supabase_enabled():
        try:
            resp = _client().table("skills").select("data").execute()
            rows = resp.data or []
            out = [r["data"] for r in rows if isinstance(r.get("data"), dict)]
            logger.info("DATABASE skills loaded from Supabase  n=%d", len(out))
            return out
        except Exception as exc:
            logger.warning("DATABASE skills Supabase load failed: %s", exc)
    try:
        if fallback_path.exists():
            raw = json.loads(fallback_path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, list) else []
    except (OSError, json.JSONDecodeError):
        pass
    return []

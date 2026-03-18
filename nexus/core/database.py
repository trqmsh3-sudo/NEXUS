"""Supabase PostgreSQL persistence with JSON file fallback.

Set SUPABASE_URL and SUPABASE_KEY in the environment. Run supabase_schema.sql
in the Supabase SQL editor to create tables.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SINGLETON_ID = "singleton"
_UNIT_TEST_MIGRATION_FLAG = Path("data/.nexus_migration_remove_unit_test_beliefs_v1")


def migrate_remove_unit_test_beliefs_once() -> None:
    """One-time: strip source=unit-test from Supabase beliefs + default JSON file (FIX 5)."""
    if "pytest" in sys.modules or os.getenv("PYTEST_CURRENT_TEST"):
        return
    try:
        _UNIT_TEST_MIGRATION_FLAG.parent.mkdir(parents=True, exist_ok=True)
        if _UNIT_TEST_MIGRATION_FLAG.exists():
            return
    except OSError:
        return
    removed = 0
    try:
        if is_supabase_enabled():
            client = _client()
            resp = client.table("beliefs").select("claim_hash, data").execute()
            for row in resp.data or []:
                d = row.get("data")
                if isinstance(d, dict) and d.get("source") == "unit-test":
                    ch = row.get("claim_hash")
                    if ch:
                        client.table("beliefs").delete().eq("claim_hash", ch).execute()
                        removed += 1
        default_json = Path("data/knowledge_store/beliefs.json")
        if default_json.exists():
            raw = json.loads(default_json.read_text(encoding="utf-8") or "[]")
            if isinstance(raw, list):
                filt = [
                    x for x in raw
                    if isinstance(x, dict) and x.get("source") != "unit-test"
                ]
                if len(filt) != len(raw):
                    default_json.write_text(
                        json.dumps(filt, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
    except Exception as exc:
        logger.warning("migrate_remove_unit_test_beliefs_once failed: %s", exc)
        return
    try:
        _UNIT_TEST_MIGRATION_FLAG.touch()
    except OSError:
        pass
    if removed:
        logger.info(
            "DATABASE FIX5 migration: removed %d unit-test row(s) from Supabase",
            removed,
        )


def describe_supabase_key(key: str | None) -> str:
    """Detect whether env uses service_role JWT, anon JWT, or non-JWT (e.g. publishable).

    Server-side Render should use **service_role** JWT from Supabase Dashboard
    → Project Settings → API → ``service_role`` secret. Anon/publishable keys
    are subject to RLS and often return only a subset of rows.
    """
    k = (key or "").strip()
    if not k:
        return "missing"
    if k.startswith("eyJ") and "." in k:
        try:
            parts = k.split(".")
            seg = parts[1]
            pad_len = (4 - len(seg) % 4) % 4
            raw = base64.urlsafe_b64decode(seg + "=" * pad_len)
            payload = json.loads(raw.decode("utf-8"))
            role = str(payload.get("role") or "")
            if role == "service_role":
                return "service_role (bypasses RLS)"
            if role == "anon":
                return "anon (RLS applies — use service_role on server if blocked)"
            return f"jwt role={role!r}"
        except Exception:
            return "jwt (unparseable)"
    return "publishable/non-JWT (not service_role — RLS likely limits reads)"


url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
logger.info("SUPABASE_URL: %s", "SET" if url else "NOT SET")
logger.info("SUPABASE_KEY: %s", "SET" if key else "NOT SET")
if key:
    logger.info("SUPABASE_KEY kind: %s", describe_supabase_key(key))


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
        logger.info(
            "DATABASE load beliefs  key_kind=%s",
            describe_supabase_key(os.getenv("SUPABASE_KEY")),
        )
        client = _client()
        logger.info('DATABASE query: supabase.table("beliefs").select("*").execute()')
        result = client.table("beliefs").select("*").execute()
        rows = list(result.data or [])
        n = len(rows)
        logger.info("Supabase returned: %d rows from beliefs", n)
        if n == 0:
            logger.error(
                "Supabase returned 0 rows — check RLS policy, table name, or project URL. "
                "On Render use SUPABASE_KEY=service_role JWT; or run DISABLE ROW LEVEL SECURITY in SQL.",
            )
        out = [r["data"] for r in rows if isinstance(r.get("data"), dict)]
        if n > 0 and len(out) < n:
            logger.warning(
                "DATABASE beliefs: %d rows missing valid JSONB data column",
                n - len(out),
            )
        kind = describe_supabase_key(os.getenv("SUPABASE_KEY"))
        if n > 0 and n < 20 and "service_role" not in kind:
            logger.warning(
                "Few beliefs returned (%d) but key is not service_role — likely RLS. "
                "Set SUPABASE_KEY to service_role JWT on Render, or disable RLS in SQL.",
                n,
            )
        logger.info("DATABASE beliefs usable dicts: %d", len(out))
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


# ---------------------------------------------------------------------------
# FIX 3 — anti_beliefs, counterfactuals, daily_cost, boundary_pairs,
#          governor_alerts, bounty_system (Supabase primary; file fallback)
# ---------------------------------------------------------------------------


def _read_json_file(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            raw = path.read_text(encoding="utf-8")
            if default == []:
                return json.loads(raw or "[]")
            return json.loads(raw or "{}")
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return default


def _write_json_file(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _supabase_get_data(table: str) -> Any | None:
    resp = (
        _client()
        .table(table)
        .select("data")
        .eq("id", _SINGLETON_ID)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        return None
    return rows[0].get("data")


def _supabase_upsert_data(table: str, data: Any) -> None:
    _client().table(table).upsert(
        {"id": _SINGLETON_ID, "data": data},
    ).execute()


def load_anti_beliefs(fallback_path: Path) -> list[dict[str, Any]]:
    file_rows = _read_json_file(fallback_path, [])
    if not isinstance(file_rows, list):
        file_rows = []
    file_rows = [x for x in file_rows if isinstance(x, dict)]
    if not is_supabase_enabled():
        return file_rows
    try:
        d = _supabase_get_data("anti_beliefs")
        if isinstance(d, list) and len(d) > 0:
            return [x for x in d if isinstance(x, dict)]
        if file_rows:
            _supabase_upsert_data("anti_beliefs", file_rows)
            logger.info(
                "DATABASE anti_beliefs migrated from file to Supabase  n=%d", len(file_rows),
            )
            return file_rows
        return []
    except Exception as exc:
        logger.warning("DATABASE anti_beliefs Supabase load failed: %s", exc)
        return file_rows


def save_anti_beliefs(items: list[dict[str, Any]], fallback_path: Path) -> None:
    if is_supabase_enabled():
        try:
            _supabase_upsert_data("anti_beliefs", items)
            logger.info("DATABASE anti_beliefs saved Supabase  n=%d", len(items))
            return
        except Exception as exc:
            logger.warning("DATABASE anti_beliefs Supabase save failed: %s", exc)
    try:
        _write_json_file(fallback_path, items)
    except OSError as exc:
        logger.warning("DATABASE anti_beliefs file save failed: %s", exc)


def load_counterfactuals(fallback_path: Path) -> dict[str, Any]:
    default: dict[str, Any] = {
        "wrong_predictions": 0,
        "background_seen": [],
        "entries": [],
    }
    raw = _read_json_file(fallback_path, {})
    file_state = default.copy()
    if isinstance(raw, list):
        file_state["entries"] = [x for x in raw if isinstance(x, dict)]
    elif isinstance(raw, dict):
        file_state["wrong_predictions"] = int(raw.get("wrong_predictions", 0))
        file_state["background_seen"] = list(raw.get("background_seen", []))
        ent = raw.get("entries", [])
        file_state["entries"] = [x for x in ent if isinstance(x, dict)] if isinstance(ent, list) else []
    if not is_supabase_enabled():
        return file_state
    try:
        d = _supabase_get_data("counterfactuals")
        if isinstance(d, dict) and (
            d.get("entries") or int(d.get("wrong_predictions") or 0) > 0 or d.get("background_seen")
        ):
            ent = d.get("entries", [])
            return {
                "wrong_predictions": int(d.get("wrong_predictions", 0)),
                "background_seen": list(d.get("background_seen", [])),
                "entries": [x for x in ent if isinstance(x, dict)] if isinstance(ent, list) else [],
            }
        if file_state.get("entries") or file_state.get("wrong_predictions"):
            _supabase_upsert_data("counterfactuals", file_state)
            logger.info("DATABASE counterfactuals migrated from file to Supabase")
            return file_state
        return default.copy()
    except Exception as exc:
        logger.warning("DATABASE counterfactuals Supabase load failed: %s", exc)
        return file_state


def save_counterfactuals(state: dict[str, Any], fallback_path: Path) -> None:
    payload = {
        "wrong_predictions": int(state.get("wrong_predictions", 0)),
        "background_seen": list(state.get("background_seen", [])),
        "entries": list(state.get("entries", [])),
    }
    if is_supabase_enabled():
        try:
            _supabase_upsert_data("counterfactuals", payload)
            return
        except Exception as exc:
            logger.warning("DATABASE counterfactuals Supabase save failed: %s", exc)
    try:
        _write_json_file(fallback_path, payload)
    except OSError as exc:
        logger.warning("DATABASE counterfactuals file save failed: %s", exc)


def load_daily_cost(fallback_path: Path) -> dict[str, Any]:
    file_d = _read_json_file(fallback_path, {})
    if not isinstance(file_d, dict):
        file_d = {}
    file_state = {
        "date": str(file_d.get("date") or ""),
        "total_cost": float(file_d.get("total_cost") or 0.0),
    }
    if not is_supabase_enabled():
        return file_state
    try:
        d = _supabase_get_data("daily_cost")
        if isinstance(d, dict) and str(d.get("date") or ""):
            return {
                "date": str(d.get("date") or ""),
                "total_cost": float(d.get("total_cost") or 0.0),
            }
        if file_state.get("date"):
            _supabase_upsert_data("daily_cost", file_state)
        return file_state
    except Exception as exc:
        logger.warning("DATABASE daily_cost Supabase load failed: %s", exc)
        return file_state


def save_daily_cost(state: dict[str, Any], fallback_path: Path) -> None:
    payload = {
        "date": str(state.get("date") or ""),
        "total_cost": round(float(state.get("total_cost") or 0.0), 6),
    }
    if is_supabase_enabled():
        try:
            _supabase_upsert_data("daily_cost", payload)
            return
        except Exception as exc:
            logger.warning("DATABASE daily_cost Supabase save failed: %s", exc)
    try:
        _write_json_file(fallback_path, payload)
    except OSError as exc:
        logger.warning("DATABASE daily_cost file save failed: %s", exc)


def load_boundary_pairs(fallback_path: Path) -> list[dict[str, Any]]:
    file_rows = _read_json_file(fallback_path, [])
    if not isinstance(file_rows, list):
        file_rows = []
    file_rows = [x for x in file_rows if isinstance(x, dict)]
    if not is_supabase_enabled():
        return file_rows
    try:
        d = _supabase_get_data("boundary_pairs")
        if isinstance(d, list) and len(d) > 0:
            return [x for x in d if isinstance(x, dict)]
        if file_rows:
            _supabase_upsert_data("boundary_pairs", file_rows)
            return file_rows
        return []
    except Exception as exc:
        logger.warning("DATABASE boundary_pairs Supabase load failed: %s", exc)
        return file_rows


def save_boundary_pairs(pairs: list[dict[str, Any]], fallback_path: Path) -> None:
    if is_supabase_enabled():
        try:
            _supabase_upsert_data("boundary_pairs", pairs)
            return
        except Exception as exc:
            logger.warning("DATABASE boundary_pairs Supabase save failed: %s", exc)
    try:
        _write_json_file(fallback_path, pairs)
    except OSError as exc:
        logger.warning("DATABASE boundary_pairs file save failed: %s", exc)


def load_governor_alerts(fallback_path: Path) -> list[dict[str, Any]]:
    file_rows = _read_json_file(fallback_path, [])
    if not isinstance(file_rows, list):
        file_rows = []
    file_rows = [x for x in file_rows if isinstance(x, dict)]
    if not is_supabase_enabled():
        return file_rows
    try:
        d = _supabase_get_data("governor_alerts")
        if isinstance(d, list) and len(d) > 0:
            return [x for x in d if isinstance(x, dict)]
        if file_rows:
            _supabase_upsert_data("governor_alerts", file_rows)
            return file_rows
        return []
    except Exception as exc:
        logger.warning("DATABASE governor_alerts Supabase load failed: %s", exc)
        return file_rows


def save_governor_alerts(alerts: list[dict[str, Any]], fallback_path: Path) -> None:
    if is_supabase_enabled():
        try:
            _supabase_upsert_data("governor_alerts", alerts)
            return
        except Exception as exc:
            logger.warning("DATABASE governor_alerts Supabase save failed: %s", exc)
    try:
        _write_json_file(fallback_path, alerts)
    except OSError as exc:
        logger.warning("DATABASE governor_alerts file save failed: %s", exc)


def load_bounty_system(fallback_path: Path) -> dict[str, Any]:
    raw = _read_json_file(fallback_path, {})
    if not isinstance(raw, dict):
        raw = {}
    file_state = {
        "bounties": {str(k): float(v) for k, v in (raw.get("bounties") or {}).items()},
        "failures": {str(k): int(v) for k, v in (raw.get("failures") or {}).items()},
    }
    if not is_supabase_enabled():
        return file_state
    try:
        d = _supabase_get_data("bounty_system")
        if isinstance(d, dict) and (d.get("bounties") or d.get("failures")):
            return {
                "bounties": {str(k): float(v) for k, v in (d.get("bounties") or {}).items()},
                "failures": {str(k): int(v) for k, v in (d.get("failures") or {}).items()},
            }
        if file_state["bounties"] or file_state["failures"]:
            _supabase_upsert_data("bounty_system", file_state)
        return file_state
    except Exception as exc:
        logger.warning("DATABASE bounty_system Supabase load failed: %s", exc)
        return file_state


def save_bounty_system(state: dict[str, Any], fallback_path: Path) -> None:
    payload = {
        "bounties": {str(k): float(v) for k, v in (state.get("bounties") or {}).items()},
        "failures": {str(k): int(v) for k, v in (state.get("failures") or {}).items()},
    }
    if is_supabase_enabled():
        try:
            _supabase_upsert_data("bounty_system", payload)
            return
        except Exception as exc:
            logger.warning("DATABASE bounty_system Supabase save failed: %s", exc)
    try:
        _write_json_file(fallback_path, payload)
    except OSError as exc:
        logger.warning("DATABASE bounty_system file save failed: %s", exc)

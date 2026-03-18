"""One-time migration: local JSON stores -> Supabase.

Run:
  python -m nexus.scripts.migrate_to_supabase

Reads:
  - data/knowledge_store/beliefs.json
  - data/skills/library.json
  - data/cycle_history.json

Writes to Supabase tables:
  - beliefs
  - skills
  - cycle_history (singleton row id='singleton')
"""

from __future__ import annotations

import json
from pathlib import Path

from nexus.core import database as nexus_db


_BELIEFS_PATH = Path("data/knowledge_store/beliefs.json")
_SKILLS_PATH = Path("data/skills/library.json")
_CYCLE_HISTORY_PATH = Path("data/cycle_history.json")


def _load_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    return [x for x in raw if isinstance(x, dict)]


def main() -> None:
    if not nexus_db.is_supabase_enabled():
        raise SystemExit(
            "SUPABASE_URL / SUPABASE_KEY not set; cannot migrate to Supabase.",
        )

    local_beliefs = _load_list(_BELIEFS_PATH)
    local_skills = _load_list(_SKILLS_PATH)
    local_cycles = _load_list(_CYCLE_HISTORY_PATH)

    # Persist
    beliefs_ok = nexus_db.save_beliefs(local_beliefs)
    if not beliefs_ok:
        raise SystemExit("Supabase beliefs save failed; see logs above.")

    nexus_db.save_skills(local_skills, _SKILLS_PATH)
    nexus_db.save_cycle_history(local_cycles, _CYCLE_HISTORY_PATH)

    # Verify counts from destination (best-effort)
    beliefs_migrated = len(nexus_db.load_belief_dicts() or [])
    skills_migrated = len(nexus_db.load_skills(_SKILLS_PATH))
    cycles_migrated = len(nexus_db.load_cycle_history(_CYCLE_HISTORY_PATH))

    print("Migration complete.")
    print(f"Beliefs migrated: {beliefs_migrated}")
    print(f"Skills migrated:  {skills_migrated}")
    print(f"Cycles migrated:  {cycles_migrated}")


if __name__ == "__main__":
    main()


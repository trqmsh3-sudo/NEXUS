# NEXUS — Project Context

## Current Version: v2.7

## Phase 1 Metrics (updated: 2026-03-16)
- autonomy_ratio: 79.6% ✅ (target: 70%)
- self_built_beliefs: 43 (target: 100)
- multi_file_systems: 0 (target: 5)
- success_rate: ~37% ⚠️ → **House C validation smoke: 5/5 pass locally** (target: 75% sustained in production)
- monthly_cost: $0 ✅ (target: <$30)

## What Works
- Thread safety (RLock on KnowledgeGraph)
- Executable proof re-verification (subprocess)
- Supabase persistence for all data stores
- LRU cache (50 beliefs max in RAM)
- Memory stable at ~254MB idle
- Kill switch obsession fixed
- Domain normalization working
- **House C v2.7:** primary function resolution from SSO (fixes helper-before-main false signature)
- **House C v2.7:** TypeError test sanitizer uses **per-function** hints from `main.py` (not “first function only”)
- **House C v2.7:** re-sanitize tests after LLM heal (stops bad `pytest.raises(TypeError)` coming back)
- **House C v2.7:** import/name repair can target SSO-primary when multiple functions exist
- **House C v2.7:** code prompt example reworded (“combine two integers”) to avoid accidental `add` substring false positives in tooling
- Regression tests in `nexus/tests/test_house_c.py` (`TestHouseCTestSanitizer`)
- Optional smoke: `scripts/smoke_house_c_cycles.py` (real pytest, mocked LLM)

## Known Issues
- **Production success_rate** still gated by model routing / rate limits / API keys (`redefine`, `all models failed`) — orthogonal to House C pytest logic
- Multi-file systems not yet supported
- `str` vs `None` assertions still rely on heal or model quality (not auto-stripped)

## Next Steps
1. Re-measure **production** success_rate after deploying v2.7; confirm 75%+ when models are healthy
2. Verify v2.7 in production (Render / live keys)
3. Add multi-file build support (Phase 2 prerequisite)

## Commit workflow (mandatory)
After **every** commit from now on:
1. Update `CONTEXT.md` with what changed (summary of the commit).
2. Update **Phase 1 Metrics** (or other metrics) if any numbers or statuses changed.
3. Add any **new issues** discovered during the work (Known Issues).
4. Stage and fold into the same commit:  
   `git add CONTEXT.md && git commit --amend --no-edit`

**`CONTEXT.md` is never optional.** Every code/docs/config change that gets committed must include a matching `CONTEXT.md` update in that same commit (via amend as above).

## Recent changes
- **2026-03-16 — v2.7:** House C test pipeline fixes (primary function selection, per-function TypeError sanitizer, post-heal sanitize, multi-fn import repair). Prompt wording tweak. Tests + smoke script. Metrics note: local validation 5/5; prod TBD on API health.

## Rules
- **Every** change → update `CONTEXT.md` (not only “significant” ones); then amend the commit.
- Mark completed items with ✅
- Add discoveries and new issues immediately
- Run all tests before marking anything complete in this file

Last updated: 2026-03-16

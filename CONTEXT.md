# NEXUS — Project Context

## Current Version: v2.8

## Phase 1 Metrics (updated: 2026-03-16)
- autonomy_ratio: 79.6% ✅ (target: 70%)
- self_built_beliefs: 43 (target: 100)
- multi_file_systems: 0 (target: 5)
- success_rate: ~37% ⚠️ (target: 75% — **unblock with valid keys + v2.8 routing**)
- monthly_cost: $0 ✅ (target: <$30)

## What Works
- Thread safety (RLock on KnowledgeGraph)
- Executable proof re-verification (subprocess)
- Supabase persistence for all data stores
- LRU cache (50 beliefs max in RAM)
- Memory stable at ~254MB idle
- Kill switch obsession fixed
- Domain normalization working
- **House C v2.7** — test generation / validation pipeline (see prior release notes)
- **Model router v2.8:**
  - **TTL blacklist** (default **3600s**): entries auto-expire; legacy `{date, models[]}` files **migrate to empty** on load (clears stale all-day blocks)
  - Env **`NEXUS_MODEL_BLACKLIST_TTL_SECONDS`** overrides TTL
  - **OpenRouter** no longer required at import — missing key logs a warning; OpenRouter models **skipped** in routing
  - **Groq** models skipped if `GROQ_API_KEY` unset
  - OpenRouter calls guard empty API key before `litellm.completion`
- **Gemini RPM file** `data/gemini_rate_limits.json` reset in repo (stale maxed counters no longer block Tier 0 on fresh deploy)
- **`scripts/probe_model_tiers.py`** — live probe Tier 0 (Gemini) / Tier 1 (Groq) / Tier 2 (OpenRouter)
- Tests: `nexus/tests/test_model_router.py`

## Known Issues
- **Production keys must be valid:** expired `GEMINI_API_KEY`, missing/invalid `OPENROUTER_API_KEY`, or Groq upstream errors still yield failures — v2.8 prevents **permanent** self-blacklisting from one bad session
- Multi-file systems not yet supported
- Local probe run (2026-03-16 dev env): Tier 0 **expired key**; Tier 1 **Groq InternalServerError** (encoding); Tier 2 **401** missing OpenRouter header/key — **fix env on Render** and re-run `python scripts/probe_model_tiers.py`

## Next Steps
1. Deploy v2.8; set **`OPENROUTER_API_KEY`**, renew **`GEMINI_API_KEY`**, confirm **`GROQ_API_KEY`** on hosting
2. Run `probe_model_tiers.py` on production shell or CI with secrets
3. Re-measure success_rate toward 75%+
4. Add multi-file build support (Phase 2 prerequisite)

## Commit workflow (mandatory)
After **every** commit from now on:
1. Update `CONTEXT.md` with what changed (summary of the commit).
2. Update **Phase 1 Metrics** (or other metrics) if any numbers or statuses changed.
3. Add any **new issues** discovered during the work (Known Issues).
4. Stage and fold into the same commit:  
   `git add CONTEXT.md && git commit --amend --no-edit`

**`CONTEXT.md` is never optional.** Every code/docs/config change that gets committed must include a matching `CONTEXT.md` update in that same commit (via amend as above).

## Recent changes
- **2026-03-16 — v2.8:** Model routing — TTL blacklist + legacy migration, optional OpenRouter at import, skip missing-tier models, reset Gemini RPM snapshot in repo, `probe_model_tiers.py`, router tests.
- **2026-03-16 — v2.7:** House C test pipeline fixes (prior).

### Reference: pre-v2.8 `data/model_blacklist.json` (legacy)
Previously **all** routing targets could be blacklisted at once under daily `models[]` (e.g. 2026-03-19 list including every Gemini, Groq, and OpenRouter model). v2.8 replaces this with **per-model `entries` → ISO expiry** or clears legacy format on read.

## Rules
- **Every** change → update `CONTEXT.md` (not only “significant” ones); then amend the commit.
- Mark completed items with ✅
- Add discoveries and new issues immediately
- Run all tests before marking anything complete in this file

Last updated: 2026-03-16

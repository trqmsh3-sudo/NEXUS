-- FIX 3 — Run in Supabase SQL Editor if you already have beliefs/cycle_history/skills.
-- (Or run full supabase_schema.sql which includes these.)

CREATE TABLE IF NOT EXISTS anti_beliefs (
  id TEXT PRIMARY KEY,
  data JSONB NOT NULL DEFAULT '[]'::jsonb,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS counterfactuals (
  id TEXT PRIMARY KEY,
  data JSONB NOT NULL DEFAULT '{}'::jsonb,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS daily_cost (
  id TEXT PRIMARY KEY,
  data JSONB NOT NULL DEFAULT '{}'::jsonb,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS boundary_pairs (
  id TEXT PRIMARY KEY,
  data JSONB NOT NULL DEFAULT '[]'::jsonb,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS governor_alerts (
  id TEXT PRIMARY KEY,
  data JSONB NOT NULL DEFAULT '[]'::jsonb,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS bounty_system (
  id TEXT PRIMARY KEY,
  data JSONB NOT NULL DEFAULT '{}'::jsonb,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE anti_beliefs DISABLE ROW LEVEL SECURITY;
ALTER TABLE counterfactuals DISABLE ROW LEVEL SECURITY;
ALTER TABLE daily_cost DISABLE ROW LEVEL SECURITY;
ALTER TABLE boundary_pairs DISABLE ROW LEVEL SECURITY;
ALTER TABLE governor_alerts DISABLE ROW LEVEL SECURITY;
ALTER TABLE bounty_system DISABLE ROW LEVEL SECURITY;

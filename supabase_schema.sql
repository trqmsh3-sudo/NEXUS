-- Run in Supabase: Dashboard → SQL → New query
-- Set NEXUS env: SUPABASE_URL + SUPABASE_KEY (service_role key recommended for server)

CREATE TABLE IF NOT EXISTS beliefs (
  claim_hash TEXT PRIMARY KEY,
  data JSONB NOT NULL,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cycle_history (
  id TEXT PRIMARY KEY,
  entries JSONB NOT NULL DEFAULT '[]'::jsonb,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS skills (
  skill_id TEXT PRIMARY KEY,
  data JSONB NOT NULL,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

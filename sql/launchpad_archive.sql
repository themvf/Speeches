-- Graduation Archive (2026-09-19): a complete record of Robinhood Chain launchpad activity.
-- Spec: docs/graduation-archive-spec.md. Python-owned schema, applied on every sweep.
--
-- V1 records and surfaces nothing. Roughly one launch in sixty graduates, so thresholds picked
-- before outcomes are labelled would be confidently wrong; the radar is a later reader of these
-- tables.

-- One row per launchpad token, updated in place as its state advances.
CREATE TABLE IF NOT EXISTS launchpad_tokens (
 network text NOT NULL,
 token_address text NOT NULL,
 symbol text,
 name text,
 dex text NOT NULL,                              -- launchpad that created the bonding-curve pool
 curve_pool text,
 first_pool_created timestamptz,                 -- pool_created_at of the curve pool = launch
 first_seen_at timestamptz NOT NULL,             -- when WE first observed it
 first_seen_pct double precision,                -- graduation % at first observation
 graduated boolean NOT NULL DEFAULT false,
 graduated_at timestamptz,                       -- launchpad completed_at, or the graduate pool's creation
 graduated_detected_at timestamptz,              -- when WE saw it; the difference is our observation lag
 graduation_pool text,
 last_pct double precision,
 last_seen_at timestamptz NOT NULL,
 moved boolean NOT NULL DEFAULT false,           -- graduation % ever exceeded its first observation
 state text NOT NULL DEFAULT 'live' CHECK(state IN ('live','graduated','stalled','dead')),
 holders integer,                                -- graduates only (per-token /info call)
 top10_share double precision,
 twitter_handle text,
 enriched_at timestamptz,
 PRIMARY KEY (network,token_address)
);
CREATE INDEX IF NOT EXISTS launchpad_tokens_candidates ON launchpad_tokens(state,last_seen_at);
CREATE INDEX IF NOT EXISTS launchpad_tokens_graduated ON launchpad_tokens(graduated_at) WHERE graduated;
CREATE INDEX IF NOT EXISTS launchpad_tokens_curve_pool ON launchpad_tokens(curve_pool);

-- Immutable. One row per observation; never rewritten, same rule as crypto_price_events.
CREATE TABLE IF NOT EXISTS launchpad_observations (
 network text NOT NULL,
 token_address text NOT NULL,
 observed_at timestamptz NOT NULL,
 phase text NOT NULL CHECK(phase IN ('curve','post')),
 rung_minutes integer,                           -- post phase: which ladder rung this row fills
 graduation_pct double precision,
 price_usd double precision,
 fdv_usd double precision,
 liquidity_usd double precision,
 volume_m30 double precision,
 volume_h1 double precision,
 volume_h24 double precision,
 buyers_m30 integer, sellers_m30 integer,
 buyers_h1 integer, sellers_h1 integer,
 txns_h1 integer,
 price_change_h1 double precision,
 holders integer,
 PRIMARY KEY (network,token_address,observed_at)
);
-- One row per rung per token: the ladder is filled once, even if a sweep retries.
CREATE UNIQUE INDEX IF NOT EXISTS launchpad_observations_rung
 ON launchpad_observations(network,token_address,rung_minutes) WHERE rung_minutes IS NOT NULL;
CREATE INDEX IF NOT EXISTS launchpad_observations_token ON launchpad_observations(token_address,observed_at);

-- One row per sweep. The feed reaches back ~19 minutes, so a late or failed sweep loses launches
-- permanently and invisibly; this table is what makes that a recorded fact instead of a silent hole.
CREATE TABLE IF NOT EXISTS launchpad_sweeps (
 id bigserial PRIMARY KEY,
 started_at timestamptz NOT NULL,
 finished_at timestamptz,
 pages_fetched integer NOT NULL DEFAULT 0,
 pools_seen integer NOT NULL DEFAULT 0,
 oldest_pool_at timestamptz,                     -- oldest pool_created_at reached this sweep
 newest_pool_at timestamptz,
 window_seconds integer,                         -- how far back this sweep could see
 new_tokens integer NOT NULL DEFAULT 0,
 graduations integer NOT NULL DEFAULT 0,
 observations integer NOT NULL DEFAULT 0,
 gap_seconds integer,                            -- >0 means launches happened that we never saw
 complete boolean NOT NULL DEFAULT true,
 errors text[] NOT NULL DEFAULT '{}',
 note text
);
CREATE INDEX IF NOT EXISTS launchpad_sweeps_started ON launchpad_sweeps(started_at DESC);
-- Legacy rows cannot be reliably attributed to a chain. Never guess their network.
ALTER TABLE launchpad_sweeps ADD COLUMN IF NOT EXISTS network text;
CREATE INDEX IF NOT EXISTS launchpad_sweeps_network_started ON launchpad_sweeps(network,started_at DESC);

-- One shared public-API pacing row across the three concurrent archive jobs.
-- Per-process sleeps alone let each job consume the entire provider allowance.
CREATE TABLE IF NOT EXISTS launchpad_api_budget (
 name text PRIMARY KEY,
 next_request_at timestamptz NOT NULL
);
INSERT INTO launchpad_api_budget(name,next_request_at) VALUES ('gecko',clock_timestamp())
 ON CONFLICT (name) DO NOTHING;

-- Session advisory locks are not safe through transaction-pooling proxies.
-- Leases survive connection switching and recover after a runner is terminated.
CREATE TABLE IF NOT EXISTS launchpad_worker_leases (
 network text NOT NULL,
 kind text NOT NULL,
 owner text NOT NULL,
 expires_at timestamptz NOT NULL,
 PRIMARY KEY (network,kind)
);

-- Multi-chain + OSINT capture (2026-09-19). Additive: the Robinhood archive is untouched.
-- See docs/solana-pumpfun-archive-spec.md.
ALTER TABLE launchpad_tokens ADD COLUMN IF NOT EXISTS launchpad text;              -- curve pool's DEX id
ALTER TABLE launchpad_tokens ADD COLUMN IF NOT EXISTS measure_pool text;           -- pool the ladder reads
ALTER TABLE launchpad_tokens ADD COLUMN IF NOT EXISTS measure_pool_reason text;    -- and why it was chosen
ALTER TABLE launchpad_tokens ADD COLUMN IF NOT EXISTS cohort_sampled boolean NOT NULL DEFAULT true;
-- Creator layer, free from the /info call the archive already makes for every graduate.
ALTER TABLE launchpad_tokens ADD COLUMN IF NOT EXISTS developer_address text;
ALTER TABLE launchpad_tokens ADD COLUMN IF NOT EXISTS developer_holding double precision;
ALTER TABLE launchpad_tokens ADD COLUMN IF NOT EXISTS is_honeypot text;
ALTER TABLE launchpad_tokens ADD COLUMN IF NOT EXISTS mint_authority text;
ALTER TABLE launchpad_tokens ADD COLUMN IF NOT EXISTS freeze_authority text;
ALTER TABLE launchpad_tokens ADD COLUMN IF NOT EXISTS telegram_handle text;
ALTER TABLE launchpad_tokens ADD COLUMN IF NOT EXISTS website text;
ALTER TABLE launchpad_tokens ADD COLUMN IF NOT EXISTS description text;
ALTER TABLE launchpad_tokens ADD COLUMN IF NOT EXISTS categories text[];
ALTER TABLE launchpad_tokens ADD COLUMN IF NOT EXISTS gt_score double precision;
-- Raw enrichment payload, so a later question can be asked of data already collected.
ALTER TABLE launchpad_tokens ADD COLUMN IF NOT EXISTS info_raw jsonb;
CREATE INDEX IF NOT EXISTS launchpad_tokens_developer ON launchpad_tokens(developer_address) WHERE developer_address IS NOT NULL;

-- Opening trade capture. Deliberately not called "the first N trades": the endpoint returns a
-- recent window whose relationship to the graduation moment is not guaranteed, so the capture
-- records its own boundaries and how much of the opening window it caught is a measurement.
-- This data is perishable - on a busy graduate 300 trades spanned 33 seconds - so it is captured
-- at graduation or not at all.
CREATE TABLE IF NOT EXISTS launchpad_trade_captures (
 id bigserial PRIMARY KEY,
 network text NOT NULL,
 token_address text NOT NULL,
 pool text,
 graduated_at timestamptz,
 capture_started_at timestamptz NOT NULL,
 capture_finished_at timestamptz,
 pages_fetched integer NOT NULL DEFAULT 0,
 trades integer NOT NULL DEFAULT 0,
 wallets integer NOT NULL DEFAULT 0,
 buyers integer NOT NULL DEFAULT 0,
 sellers integer NOT NULL DEFAULT 0,
 top_wallet_share double precision,
 repeat_wallets integer,
 earliest_trade_at timestamptz,                    -- oldest trade the window actually returned
 latest_trade_at timestamptz,
 window_seconds integer,                           -- latest - earliest: the span we hold
 lag_seconds integer,                              -- earliest_trade_at - graduated_at: what we missed
 note text
);
CREATE INDEX IF NOT EXISTS launchpad_trade_captures_token ON launchpad_trade_captures(network,token_address);

-- Raw wallet-level rows. Kept rather than summarised away: which features matter is exactly what
-- we do not know yet.
CREATE TABLE IF NOT EXISTS launchpad_trades (
 capture_id bigint NOT NULL REFERENCES launchpad_trade_captures(id) ON DELETE CASCADE,
 sequence integer NOT NULL,                        -- order within the capture, oldest first
 wallet text,
 traded_at timestamptz NOT NULL,
 kind text,
 token_amount double precision,
 usd double precision,
 tx_hash text,
 block_number bigint,
 PRIMARY KEY (capture_id,sequence)
);
CREATE INDEX IF NOT EXISTS launchpad_trades_wallet ON launchpad_trades(wallet) WHERE wallet IS NOT NULL;

-- One row per enrichment worker run. Health is not "the worker ran": a worker can match the arrival
-- rate exactly while never reaching the back of the queue, so the primary signal is whether the
-- OLDEST pending item is getting older across successive runs. That needs history to answer.
CREATE TABLE IF NOT EXISTS launchpad_enrich_runs (
 id bigserial PRIMARY KEY,
 network text NOT NULL,
 started_at timestamptz NOT NULL,
 finished_at timestamptz,
 processed integer NOT NULL DEFAULT 0,
 rungs_filled integer NOT NULL DEFAULT 0,
 error_count integer NOT NULL DEFAULT 0,
 pending integer,
 oldest_pending_age_seconds integer,
 arrival_rate_per_hour integer,
 service_rate_per_hour integer,
 state text                                        -- idle | healthy | degrading
);
CREATE INDEX IF NOT EXISTS launchpad_enrich_runs_network ON launchpad_enrich_runs(network,id DESC);

-- Data-quality caveat, recorded where the column lives so it travels with the schema.
COMMENT ON COLUMN launchpad_tokens.first_pool_created IS
 'Creation time of the pool we observed. On Solana this is an indexing/migration-adjacent timestamp '
 'and MUST NOT be interpreted as token launch time: measured 2026-09-19, the median gap between it '
 'and graduation was 0s for Pump.fun and 52s for Meteora DBC, because the curve pool is frequently '
 'indexed at or near migration rather than at launch. Any time-since-launch or minutes-to-graduation '
 'calculation on Solana needs a different source (Solana RPC or Bitquery on the launchpad program).';

-- Provenance that survives whichever side we observe first. `launchpad` is only ever a curve DEX,
-- so it is NULL for a graduate first seen arriving at its destination - on the commissioning sample
-- filtering launchpad='pump-fun' returned 9 of 16 real Pump.fun graduates. Family spans both sides.
ALTER TABLE launchpad_tokens ADD COLUMN IF NOT EXISTS launchpad_family text;
CREATE INDEX IF NOT EXISTS launchpad_tokens_family ON launchpad_tokens(network,launchpad_family) WHERE graduated;
COMMENT ON COLUMN launchpad_tokens.launchpad IS
 'The bonding-curve DEX, when we observed it. NULL for a graduate first seen arriving at its '
 'destination. Filter analysis on launchpad_family instead: this column silently undercounts.';
COMMENT ON COLUMN launchpad_tokens.launchpad_family IS
 'Launchpad provenance spanning both sides of a pairing (pump-fun and pumpswap are both pump.fun). '
 'This is the column a launchpad-specific analysis should filter on.';

-- Analytical semantics as a value, not a substring of a prose reason.
ALTER TABLE launchpad_tokens ADD COLUMN IF NOT EXISTS measure_pool_timing text
  CHECK (measure_pool_timing IN ('at_graduation','late'));
COMMENT ON COLUMN launchpad_tokens.measure_pool_timing IS
 'Whether the measurement pool was chosen at graduation or later. A late choice may name a '
 'different market than the one that mattered, since a token can fall to near-zero liquidity '
 'within the hour, so the two must stay separable in analysis.';

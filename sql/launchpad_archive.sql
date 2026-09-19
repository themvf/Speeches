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

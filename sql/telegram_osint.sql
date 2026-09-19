-- Telegram OSINT collector (2026-09-19). Python-owned schema, applied on every run.
-- Spec: docs/telegram-osint-spec.md.
--
-- Three layers, kept deliberately apart:
--   1. raw      - what Telegram said, stored verbatim and never rewritten by a classifier
--   2. derived  - resolution, archive join, outcomes; all recomputable from layer 1 + the archive
--   3. rollup   - channel statistics, every one of which stores its own denominator
-- A derived row must never be the only place a fact lives, because every classification rule here
-- is provisional and will be rewritten once there is enough data to judge it.

-- ---------------------------------------------------------------- layer 1: raw

-- Monitored channels. The collector reads this table, not a hardcoded list; telegram_channels.json
-- is a seed applied with --sync-config, so a channel can be added or paused without a deploy.
CREATE TABLE IF NOT EXISTS telegram_channels (
 channel_id bigint PRIMARY KEY,                  -- Telegram's own peer id, stable across renames
 username text,                                  -- may be NULL (private) and may change
 title text,
 kind text,                                      -- channel | supergroup | group
 active boolean NOT NULL DEFAULT true,
 added_at timestamptz NOT NULL DEFAULT now(),
 -- Collection state. last_message_id is the cursor for incremental collection; backfill_floor_id
 -- is how far back history has been walked, so a resumed backfill does not restart at the top.
 last_message_id bigint,
 last_message_at timestamptz,
 backfill_floor_id bigint,
 backfill_complete boolean NOT NULL DEFAULT false,
 last_poll_at timestamptz,
 last_success_at timestamptz,
 access_state text NOT NULL DEFAULT 'unknown'
   CHECK (access_state IN ('unknown','ok','forbidden','not_found','flood_wait','error')),
 access_note text,
 consecutive_failures integer NOT NULL DEFAULT 0,
 notes text
);
CREATE INDEX IF NOT EXISTS telegram_channels_active ON telegram_channels(active,last_poll_at);

-- Raw messages. (channel_id, message_id) is Telegram's own identity, so collection is idempotent by
-- construction: a re-read of the same window writes the same rows.
CREATE TABLE IF NOT EXISTS telegram_messages (
 channel_id bigint NOT NULL REFERENCES telegram_channels(channel_id) ON DELETE CASCADE,
 message_id bigint NOT NULL,
 posted_at timestamptz NOT NULL,                 -- Telegram's date: when the channel posted it
 ingested_at timestamptz NOT NULL DEFAULT now(), -- when WE saw it; the difference is our lag
 sender_id bigint,                               -- often NULL: channel posts are usually unsigned
 sender_username text,
 author_signature text,
 text text,
 edited_at timestamptz,
 deleted_detected_at timestamptz,                -- set when a re-read finds the id gone
 views integer,
 forwards integer,
 reply_to_message_id bigint,
 -- Forwarding provenance. A token copied across five channels is one discovery, not five, so
 -- forwards must stay separable from original posts in every statistic downstream.
 is_forward boolean NOT NULL DEFAULT false,
 forward_from_channel_id bigint,
 forward_from_name text,
 forward_from_message_id bigint,
 forward_origin_at timestamptz,
 urls text[] NOT NULL DEFAULT '{}',
 raw jsonb,                                      -- the normalised payload, for questions not yet asked
 PRIMARY KEY (channel_id,message_id)
);
CREATE INDEX IF NOT EXISTS telegram_messages_posted ON telegram_messages(posted_at DESC);
CREATE INDEX IF NOT EXISTS telegram_messages_channel_posted ON telegram_messages(channel_id,posted_at DESC);
CREATE INDEX IF NOT EXISTS telegram_messages_unprocessed ON telegram_messages(channel_id,message_id)
 WHERE text IS NOT NULL;

-- One row per collector run per channel. The same reasoning as launchpad_sweeps: a run that never
-- happened leaves no trace anywhere else, so continuity has to be a recorded fact.
CREATE TABLE IF NOT EXISTS telegram_collection_runs (
 id bigserial PRIMARY KEY,
 started_at timestamptz NOT NULL,
 finished_at timestamptz,
 mode text NOT NULL,                             -- backfill | poll
 channels_attempted integer NOT NULL DEFAULT 0,
 channels_ok integer NOT NULL DEFAULT 0,
 messages_seen integer NOT NULL DEFAULT 0,
 messages_new integer NOT NULL DEFAULT 0,
 messages_updated integer NOT NULL DEFAULT 0,
 reconnects integer NOT NULL DEFAULT 0,
 flood_waits integer NOT NULL DEFAULT 0,
 flood_wait_seconds integer NOT NULL DEFAULT 0,
 complete boolean NOT NULL DEFAULT true,
 errors text[] NOT NULL DEFAULT '{}',
 note text
);
CREATE INDEX IF NOT EXISTS telegram_collection_runs_started ON telegram_collection_runs(started_at DESC);

-- ---------------------------------------------------------------- layer 2: derived

-- One row per (message, token reference). Recomputable: drop it and re-derive from telegram_messages
-- plus the archive. `resolution` records HOW the token was identified, so a later audit can remove
-- one resolution class from every statistic without re-reading Telegram.
CREATE TABLE IF NOT EXISTS telegram_token_mentions (
 id bigserial PRIMARY KEY,
 channel_id bigint NOT NULL,
 message_id bigint NOT NULL,
 network text NOT NULL DEFAULT 'solana',
 token_address text,                             -- NULL when a reference could not be resolved
 raw_reference text NOT NULL,                    -- the address or ticker exactly as it appeared
 reference_kind text NOT NULL CHECK (reference_kind IN ('contract','ticker','url')),
 resolution text NOT NULL CHECK (resolution IN
   ('contract','ticker_unique','unresolved_unknown','unresolved_ambiguous','unresolved_not_in_archive')),
 confidence double precision,
 mentioned_at timestamptz NOT NULL,              -- the message's posted_at, denormalised for speed
 is_forward boolean NOT NULL DEFAULT false,
 -- Archive join, copied at derivation time so a statistic is reproducible even if the archive row
 -- is later corrected; re-derive to pick up corrections.
 graduated boolean,
 graduated_at timestamptz,
 launchpad_family text,
 measure_pool text,
 measure_pool_timing text,
 seconds_to_graduation double precision,         -- graduated_at - mentioned_at; POSITIVE = pre-graduation
 -- Ordering among monitored channels, over ORIGINAL posts only (see is_forward).
 is_first_monitored_mention boolean,
 monitored_sequence integer,
 seconds_after_first_mention double precision,
 -- Claim language the message made about itself. Stored for comparison against measured outcomes,
 -- never as an input to one: a channel's own "10x" is a claim, not a result.
 claimed_multiple double precision,
 claimed_market_cap double precision,
 derived_at timestamptz NOT NULL DEFAULT now(),
 FOREIGN KEY (channel_id,message_id) REFERENCES telegram_messages(channel_id,message_id) ON DELETE CASCADE
);
CREATE UNIQUE INDEX IF NOT EXISTS telegram_token_mentions_identity
 ON telegram_token_mentions(channel_id,message_id,network,raw_reference);
CREATE INDEX IF NOT EXISTS telegram_token_mentions_token
 ON telegram_token_mentions(network,token_address,mentioned_at) WHERE token_address IS NOT NULL;
CREATE INDEX IF NOT EXISTS telegram_token_mentions_channel
 ON telegram_token_mentions(channel_id,mentioned_at);

-- Prices anchored to a MENTION, not to graduation. The archive's own ladder is anchored to
-- graduation and samples only a fraction of graduates, so it cannot answer "what happened after
-- this channel posted". OHLCV is historical and can be fetched after the fact, unlike the trade
-- window, which is why this layer can be back-filled at all.
CREATE TABLE IF NOT EXISTS telegram_price_points (
 network text NOT NULL,
 token_address text NOT NULL,
 pool text NOT NULL,
 minute timestamptz NOT NULL,                    -- candle open, truncated to the fetched resolution
 resolution text NOT NULL CHECK (resolution IN ('minute','hour')),
 open double precision, high double precision, low double precision, close double precision,
 volume_usd double precision,
 fetched_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY (network,token_address,pool,resolution,minute)
);
CREATE INDEX IF NOT EXISTS telegram_price_points_lookup
 ON telegram_price_points(network,token_address,minute);

-- One row per mention per horizon rung. `status` is the whole point of this table:
--   observed     - we have a price at the rung and a price at the mention
--   pending      - the horizon has not elapsed yet; this is NOT a loss and must never be counted as one
--   unobserved   - the horizon elapsed and we still have no price (collection gap, dead pool, no pool)
-- Separating pending from unobserved from observed-and-down is the difference between a survival
-- rate and a number that quietly flatters whichever channel posted most recently.
CREATE TABLE IF NOT EXISTS telegram_mention_outcomes (
 mention_id bigint NOT NULL REFERENCES telegram_token_mentions(id) ON DELETE CASCADE,
 rung_minutes integer NOT NULL,
 status text NOT NULL CHECK (status IN ('observed','pending','unobserved')),
 base_price double precision,                    -- price at the mention itself
 rung_price double precision,
 return_pct double precision,                    -- (rung/base - 1) * 100
 position_value double precision,                -- a hypothetical $100 at the mention, at this rung
 price_age_seconds double precision,             -- how far the matched candle sat from the target
 unobserved_reason text,
 computed_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY (mention_id,rung_minutes)
);

-- Peak and base are horizon-independent, so they live on the mention's own summary row rather than
-- being recomputed per rung.
CREATE TABLE IF NOT EXISTS telegram_mention_summary (
 mention_id bigint PRIMARY KEY REFERENCES telegram_token_mentions(id) ON DELETE CASCADE,
 base_price double precision,
 base_price_age_seconds double precision,
 market_cap_at_mention double precision,
 liquidity_at_mention double precision,
 peak_price double precision,
 peak_at timestamptz,
 peak_multiple double precision,
 minutes_to_peak double precision,
 price_points integer NOT NULL DEFAULT 0,        -- how much price history backs the row above
 window_minutes integer,
 computed_at timestamptz NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------- layer 3: rollup

-- Channel statistics. Every rate stores its numerator and denominator, because a percentage whose
-- denominator is not visible is the easiest way to publish a number nobody can check.
CREATE TABLE IF NOT EXISTS telegram_channel_stats (
 channel_id bigint NOT NULL REFERENCES telegram_channels(channel_id) ON DELETE CASCADE,
 network text NOT NULL,
 window_days integer NOT NULL,
 computed_at timestamptz NOT NULL,
 messages integer NOT NULL DEFAULT 0,
 mentions integer NOT NULL DEFAULT 0,
 forwarded_mentions integer NOT NULL DEFAULT 0,
 unresolved_mentions integer NOT NULL DEFAULT 0,
 tokens_distinct integer NOT NULL DEFAULT 0,
 repeat_mentions_median double precision,
 -- Timing. Denominator is resolved mentions of tokens the archive knows graduated.
 graduated_tokens integer NOT NULL DEFAULT 0,
 pre_graduation integer NOT NULL DEFAULT 0,
 median_seconds_to_graduation double precision,
 first_among_monitored integer NOT NULL DEFAULT 0,
 median_sequence double precision,
 median_market_cap_at_mention double precision,
 -- Outcomes. Each rung carries its own observed/pending/unobserved split in `rungs`, so a rate is
 -- never read without the three-way denominator that produced it.
 rungs jsonb NOT NULL DEFAULT '{}'::jsonb,
 median_peak_multiple double precision,
 peak_sample integer NOT NULL DEFAULT 0,
 -- Typology, derived from the columns above. NULL until the sample justifies a label.
 typology text,
 typology_reason text,
 typology_sample integer NOT NULL DEFAULT 0,
 PRIMARY KEY (channel_id,network,window_days,computed_at)
);
CREATE INDEX IF NOT EXISTS telegram_channel_stats_recent
 ON telegram_channel_stats(network,window_days,computed_at DESC);

-- Co-mention edges: pairs of channels that posted the same token within a narrow window. This is
-- the coordination signal, and it is stored as counted evidence rather than as a verdict.
CREATE TABLE IF NOT EXISTS telegram_channel_pairs (
 channel_a bigint NOT NULL,
 channel_b bigint NOT NULL,                      -- always > channel_a, so a pair has one row
 network text NOT NULL,
 window_days integer NOT NULL,
 computed_at timestamptz NOT NULL,
 shared_tokens integer NOT NULL DEFAULT 0,
 tight_pairs integer NOT NULL DEFAULT 0,         -- shared tokens posted within TIGHT_SECONDS
 median_gap_seconds double precision,
 a_first integer NOT NULL DEFAULT 0,             -- how often A led
 b_first integer NOT NULL DEFAULT 0,
 forward_pairs integer NOT NULL DEFAULT 0,       -- at least one side was a forward: not independent
 PRIMARY KEY (channel_a,channel_b,network,window_days,computed_at)
);

COMMENT ON COLUMN telegram_token_mentions.seconds_to_graduation IS
 'graduated_at minus mentioned_at. Positive means the channel posted BEFORE graduation. It is '
 'deliberately not measured against first_pool_created, which on Solana is an indexing/migration '
 'adjacent timestamp and is not a launch time (see launchpad_tokens.first_pool_created).';
COMMENT ON COLUMN telegram_token_mentions.is_first_monitored_mention IS
 'First among MONITORED channels only, counting original posts. It says nothing about whether the '
 'channel was first in the world; the monitored set is a convenience sample of Telegram.';
COMMENT ON TABLE telegram_mention_outcomes IS
 'status separates pending (horizon not yet elapsed) from unobserved (elapsed, no price) from '
 'observed. A pending rung is not a loss. Every rate computed from this table must carry all three.';

-- Derivation watermark. A message with no token reference in it is still a processed message, and
-- without this column it would be re-read on every run forever.
ALTER TABLE telegram_messages ADD COLUMN IF NOT EXISTS references_derived_at timestamptz;
CREATE INDEX IF NOT EXISTS telegram_messages_pending_derivation
 ON telegram_messages(posted_at) WHERE references_derived_at IS NULL;

-- The inputs a typology label was derived from, stored beside the label. A stored verdict whose
-- inputs are not stored cannot be audited after the rule changes, and these rules will change.
ALTER TABLE telegram_channel_stats ADD COLUMN IF NOT EXISTS wallet_alert_messages integer;
ALTER TABLE telegram_channel_stats ADD COLUMN IF NOT EXISTS median_seconds_after_first double precision;
ALTER TABLE telegram_channel_stats ADD COLUMN IF NOT EXISTS tight_pair_share double precision;
ALTER TABLE telegram_channel_stats ADD COLUMN IF NOT EXISTS median_claimed_multiple double precision;
COMMENT ON COLUMN telegram_channel_stats.median_claimed_multiple IS
 'What the channel SAID it achieved, for comparison against the measured rungs. Never an input to '
 'them: performance is computed from price history, not from the post claiming a result.';

-- Relay attribution (2026-09-19). A Telegram forward announces itself in the message header; a
-- repost carrying the same wording does not, and in call channels that shape is the commoner one.
-- Without this a repost becomes a second independent sighting: the token looks discovered twice and
-- the relay channel's "first among monitored" count is inflated by exactly the tokens it was
-- slowest on. Only 'original' mentions take a sequence position.
ALTER TABLE telegram_token_mentions ADD COLUMN IF NOT EXISTS mention_origin text
  NOT NULL DEFAULT 'original' CHECK (mention_origin IN ('original','forward','repost'));
ALTER TABLE telegram_token_mentions ADD COLUMN IF NOT EXISTS relay_of_channel_id bigint;
ALTER TABLE telegram_token_mentions ADD COLUMN IF NOT EXISTS relay_reason text;
CREATE INDEX IF NOT EXISTS telegram_token_mentions_origin
 ON telegram_token_mentions(network,token_address,mention_origin,mentioned_at);
COMMENT ON COLUMN telegram_token_mentions.mention_origin IS
 'original = no earlier monitored post shares this wording; forward = Telegram forward header; '
 'repost = near-duplicate text about the same token from another channel earlier in the window. '
 'Only original mentions are counted as discovery. repost is an observation about SHARED WORDING '
 'and an attribution of sequence - it is NOT a finding that one channel deliberately copied '
 'another, since shared call-bot templates, a quoted launch announcement and a channel reusing its '
 'own boilerplate all produce high overlap with no copying. Intent needs platform metadata or the '
 'wording itself saying so; relay_reason states the measurement and never characterises it.';
ALTER TABLE telegram_channel_stats ADD COLUMN IF NOT EXISTS reposted_mentions integer;

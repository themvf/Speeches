-- Source-neutral intelligence fusion. Additive; existing X and launchpad archives remain authoritative.
-- Raw/derived rows are append-only. Re-running a materializer inserts missing deterministic IDs only.

CREATE TABLE IF NOT EXISTS intelligence_collection_runs (
 id text PRIMARY KEY,
 kind text NOT NULL,
 version text NOT NULL,
 started_at timestamptz NOT NULL,
 finished_at timestamptz,
 status text NOT NULL CHECK(status IN ('running','complete','failed')),
 metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS intelligence_source_actors (
 id text PRIMARY KEY,
 source text NOT NULL CHECK(source IN ('x','telegram','onchain','exchange','official_web','github')),
 source_actor_id text NOT NULL,
 handle text,
 display_name text,
 first_observed_at timestamptz NOT NULL,
 metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
 UNIQUE(source,source_actor_id)
);

CREATE TABLE IF NOT EXISTS intelligence_entities (
 id text PRIMARY KEY,
 kind text NOT NULL CHECK(kind IN ('asset','wallet','exchange','pool','account','repository','organization','other')),
 label text NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(),
 metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS intelligence_asset_identities (
 entity_id text PRIMARY KEY REFERENCES intelligence_entities(id),
 network text NOT NULL,
 contract_address text,
 native_symbol text,
 identity_key text NOT NULL UNIQUE,
 CHECK(contract_address IS NOT NULL OR native_symbol IS NOT NULL)
);

CREATE TABLE IF NOT EXISTS intelligence_observations (
 id text PRIMARY KEY,
 source text NOT NULL CHECK(source IN ('x','telegram','onchain','exchange','official_web','github')),
 source_record_id text NOT NULL,
 revision integer NOT NULL DEFAULT 1 CHECK(revision>0),
 supersedes_observation_id text REFERENCES intelligence_observations(id),
 source_actor_id text REFERENCES intelligence_source_actors(id),
 content text,
 published_at timestamptz,
 observed_at timestamptz NOT NULL,
 raw_hash text NOT NULL,
 collection_run_id text REFERENCES intelligence_collection_runs(id),
 metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
 UNIQUE(source,source_record_id,revision)
);
CREATE INDEX IF NOT EXISTS intelligence_observations_time ON intelligence_observations(published_at,observed_at);

CREATE TABLE IF NOT EXISTS intelligence_observation_entities (
 observation_id text NOT NULL REFERENCES intelligence_observations(id),
 entity_id text NOT NULL REFERENCES intelligence_entities(id),
 relationship text NOT NULL CHECK(relationship IN ('subject','mentions','author','counterparty','venue','pool')),
 match_method text NOT NULL,
 provisional boolean NOT NULL DEFAULT false,
 derivation_version text NOT NULL,
 PRIMARY KEY(observation_id,entity_id,relationship,derivation_version)
);

CREATE TABLE IF NOT EXISTS intelligence_claims (
 id text PRIMARY KEY,
 subject_entity_id text NOT NULL REFERENCES intelligence_entities(id),
 claim_type text NOT NULL CHECK(claim_type IN ('listing','transfer','graduation','pool_creation','launch','security_incident','supply_change','governance','performance_prediction','other')),
 predicate text NOT NULL,
 object jsonb NOT NULL,
 effective_at timestamptz,
 verifiability text NOT NULL CHECK(verifiability IN ('objective','conditional','subjective')),
 canonical_fingerprint text NOT NULL,
 extraction_version text NOT NULL,
 created_at timestamptz NOT NULL,
 UNIQUE(canonical_fingerprint,extraction_version)
);
CREATE INDEX IF NOT EXISTS intelligence_claims_subject ON intelligence_claims(subject_entity_id,created_at);

CREATE TABLE IF NOT EXISTS intelligence_observation_claims (
 observation_id text NOT NULL REFERENCES intelligence_observations(id),
 claim_id text NOT NULL REFERENCES intelligence_claims(id),
 relationship text NOT NULL CHECK(relationship IN ('asserts','independently_corroborates','repeats','quotes','disputes','corrects','retracts')),
 confidence double precision NOT NULL CHECK(confidence>=0 AND confidence<=1),
 derivation_version text NOT NULL,
 evidence jsonb NOT NULL DEFAULT '{}'::jsonb,
 PRIMARY KEY(observation_id,claim_id,derivation_version)
);

CREATE TABLE IF NOT EXISTS intelligence_events (
 id text PRIMARY KEY,
 event_type text NOT NULL,
 occurred_at timestamptz NOT NULL,
 observed_at timestamptz NOT NULL,
 verification_policy text NOT NULL,
 verification_version text NOT NULL,
 attributes jsonb NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(event_type,verification_version,id)
);

CREATE TABLE IF NOT EXISTS intelligence_event_entities (
 event_id text NOT NULL REFERENCES intelligence_events(id),
 entity_id text NOT NULL REFERENCES intelligence_entities(id),
 relationship text NOT NULL CHECK(relationship IN ('subject','actor','counterparty','venue','pool')),
 PRIMARY KEY(event_id,entity_id,relationship)
);

CREATE TABLE IF NOT EXISTS intelligence_event_observations (
 event_id text NOT NULL REFERENCES intelligence_events(id),
 observation_id text NOT NULL REFERENCES intelligence_observations(id),
 relationship text NOT NULL CHECK(relationship IN ('records','confirms','reports')),
 PRIMARY KEY(event_id,observation_id,relationship)
);

CREATE TABLE IF NOT EXISTS intelligence_claim_event_assessments (
 id text PRIMARY KEY,
 claim_id text NOT NULL REFERENCES intelligence_claims(id),
 event_id text NOT NULL REFERENCES intelligence_events(id),
 status text NOT NULL CHECK(status IN ('confirmed','partially_confirmed','contradicted','superseded')),
 assessed_at timestamptz NOT NULL,
 policy text NOT NULL,
 version text NOT NULL,
 rationale text NOT NULL,
 evidence_observation_ids text[] NOT NULL,
 UNIQUE(claim_id,event_id,version)
);

CREATE TABLE IF NOT EXISTS intelligence_conversation_edges (
 id text PRIMARY KEY,
 source_observation_id text NOT NULL REFERENCES intelligence_observations(id),
 target_observation_id text NOT NULL REFERENCES intelligence_observations(id),
 relationship text NOT NULL CHECK(relationship IN ('reply','quote','correction','rebuttal')),
 confidence double precision NOT NULL CHECK(confidence>=0 AND confidence<=1),
 derivation_version text NOT NULL,
 evidence jsonb NOT NULL
);

CREATE TABLE IF NOT EXISTS intelligence_propagation_edges (
 id text PRIMARY KEY,
 source_observation_id text NOT NULL REFERENCES intelligence_observations(id),
 target_observation_id text NOT NULL REFERENCES intelligence_observations(id),
 relationship text NOT NULL CHECK(relationship IN ('exact_copy','near_copy','cross_post','shared_url','shared_media','explicit_forward')),
 confidence double precision NOT NULL CHECK(confidence>=0 AND confidence<=1),
 derivation_version text NOT NULL,
 evidence jsonb NOT NULL
);

CREATE TABLE IF NOT EXISTS intelligence_market_measurements (
 id text PRIMARY KEY,
 entity_id text NOT NULL REFERENCES intelligence_entities(id),
 venue text,
 pool text,
 quote_currency text,
 measured_at timestamptz NOT NULL,
 price_usd double precision,
 liquidity_usd double precision,
 -- A volume is a flow over a window, so it is meaningless without one. This table is
 -- source-neutral and different sources report different windows, so the window travels with the
 -- value as data rather than in the column name. The paired CHECK below makes a windowless volume
 -- impossible to store.
 volume_usd double precision,
 volume_window text,
 source text NOT NULL,
 methodology_version text NOT NULL,
 source_record_id text NOT NULL,
 metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
 UNIQUE(entity_id,source,source_record_id,methodology_version)
);

CREATE TABLE IF NOT EXISTS intelligence_claim_outcomes (
 id text PRIMARY KEY,
 claim_id text NOT NULL REFERENCES intelligence_claims(id),
 measurement_id text NOT NULL REFERENCES intelligence_market_measurements(id),
 anchor_type text NOT NULL CHECK(anchor_type IN ('first_social_observation','first_unlinked_corroboration','event_occurred','verified_at')),
 anchor_at timestamptz NOT NULL,
 horizon_seconds integer NOT NULL,
 methodology_version text NOT NULL,
 UNIQUE(claim_id,measurement_id,anchor_type,methodology_version)
);

CREATE TABLE IF NOT EXISTS intelligence_collection_diagnostics (
 id text PRIMARY KEY,
 collection_run_id text REFERENCES intelligence_collection_runs(id),
 source text NOT NULL,
 asset_entity_id text REFERENCES intelligence_entities(id),
 query_kind text NOT NULL,
 query_version text NOT NULL,
 window_start timestamptz NOT NULL,
 window_end timestamptz NOT NULL,
 pages integer,
 unique_results integer,
 exclusive_results integer,
 exhausted boolean,
 oldest_result_at timestamptz,
 newest_result_at timestamptz,
 provider_failures integer NOT NULL DEFAULT 0,
 budget_exhausted boolean NOT NULL DEFAULT false,
 metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

-- Append-only enforcement for evidence-bearing facts. Corrections are new rows linked by revision/history.
CREATE OR REPLACE FUNCTION prevent_intelligence_fact_mutation() RETURNS trigger AS $$
BEGIN
 RAISE EXCEPTION '% is append-only; insert a revision or assessment instead',TG_TABLE_NAME;
END;
$$ LANGUAGE plpgsql;

DO $$
DECLARE table_name text;
BEGIN
 FOREACH table_name IN ARRAY ARRAY['intelligence_observations','intelligence_claims','intelligence_observation_claims',
  'intelligence_events','intelligence_event_observations','intelligence_claim_event_assessments',
  'intelligence_market_measurements','intelligence_claim_outcomes']
 LOOP
  EXECUTE format('DROP TRIGGER IF EXISTS %I_append_only ON %I',table_name,table_name);
  EXECUTE format('CREATE TRIGGER %I_append_only BEFORE UPDATE OR DELETE ON %I FOR EACH ROW EXECUTE FUNCTION prevent_intelligence_fact_mutation()',table_name,table_name);
 END LOOP;
END $$;

-- Volume windows (2026-09): a flow stored without its window cannot be interpreted. The launchpad
-- adapter reads launchpad_observations.volume_h1 -- an hour -- and wrote it into a column named
-- only volume_usd, so the window was lost the moment it crossed into this source-neutral layer.
-- The window now travels with the value.
--
-- The constraint is added NOT VALID on purpose. This table is append-only: a stored fact is never
-- rewritten, and rows recorded before the column existed genuinely did not record a window.
-- Backfilling them to 'h1' would be inventing a measurement that was never taken, and the
-- append-only trigger rightly refuses it. NOT VALID binds every new row while leaving the
-- historical record exactly as it was written; those rows read as "window not recorded", which is
-- the truth, rather than as a window we inferred later.
ALTER TABLE intelligence_market_measurements ADD COLUMN IF NOT EXISTS volume_window text;
DO $$
BEGIN
 IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='intelligence_market_measurements_volume_window') THEN
  ALTER TABLE intelligence_market_measurements ADD CONSTRAINT intelligence_market_measurements_volume_window
   CHECK ((volume_usd IS NULL) = (volume_window IS NULL)) NOT VALID;
 END IF;
END $$;

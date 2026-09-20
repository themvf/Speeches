-- Source-neutral intelligence fusion substrate.
-- Facts are preserved; interpretations are versioned; assessments are historical.
-- Telegram is intentionally a source placeholder only until its live collector is commissioned.

CREATE TABLE IF NOT EXISTS source_actor (
 id bigserial PRIMARY KEY,
 platform text NOT NULL,
 external_actor_id text NOT NULL,
 handle text,
 display_name text,
 first_seen_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(platform, external_actor_id)
);

CREATE TABLE IF NOT EXISTS source_record (
 id bigserial PRIMARY KEY,
 platform text NOT NULL,
 external_record_id text NOT NULL,
 source_actor_id bigint REFERENCES source_actor(id),
 record_type text NOT NULL,
 canonical_url text,
 first_seen_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(platform, external_record_id)
);

-- Immutable snapshots of the source artifact itself. An edit creates a new row.
CREATE TABLE IF NOT EXISTS source_record_revision (
 id bigserial PRIMARY KEY,
 source_record_id bigint NOT NULL REFERENCES source_record(id),
 source_version text,
 content_sha256 text NOT NULL,
 published_at timestamptz,
 observed_at timestamptz NOT NULL,
 raw_text text,
 raw_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
 supersedes_source_revision_id bigint REFERENCES source_record_revision(id),
 UNIQUE(source_record_id, content_sha256)
);
CREATE INDEX IF NOT EXISTS source_record_revision_record
 ON source_record_revision(source_record_id, observed_at, id);

-- Logical observation identity. Re-derivation never changes this row.
CREATE TABLE IF NOT EXISTS observation (
 id bigserial PRIMARY KEY,
 source_record_id bigint NOT NULL REFERENCES source_record(id),
 observation_type text NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(source_record_id, observation_type)
);

-- Immutable analytical interpretation of one source revision.
-- Source edits and extraction-policy changes both append rows; neither rewrites history.
CREATE TABLE IF NOT EXISTS observation_revision (
 id bigserial PRIMARY KEY,
 observation_id bigint NOT NULL REFERENCES observation(id),
 source_record_revision_id bigint NOT NULL REFERENCES source_record_revision(id),
 derivation_version text NOT NULL,
 extracted_at timestamptz NOT NULL,
 payload jsonb NOT NULL DEFAULT '{}'::jsonb,
 supersedes_observation_revision_id bigint REFERENCES observation_revision(id),
 UNIQUE(observation_id, source_record_revision_id, derivation_version)
);
CREATE INDEX IF NOT EXISTS observation_revision_observation
 ON observation_revision(observation_id, extracted_at, id);

CREATE TABLE IF NOT EXISTS entity (
 id bigserial PRIMARY KEY,
 entity_type text NOT NULL,
 canonical_key text NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(entity_type, canonical_key)
);

-- Contract identity is canonical. Symbol/name remain metadata and are never used as the key.
CREATE TABLE IF NOT EXISTS asset_identity (
 entity_id bigint PRIMARY KEY REFERENCES entity(id),
 chain text NOT NULL,
 contract_address text NOT NULL,
 symbol text,
 name text,
 metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
 UNIQUE(chain, contract_address)
);
COMMENT ON COLUMN asset_identity.contract_address IS
 'Canonical asset identity within a chain. Preserve case on case-sensitive chains such as Solana.';

-- Relationship tables carry their own provenance and derivation metadata.
CREATE TABLE IF NOT EXISTS observation_entity (
 observation_revision_id bigint NOT NULL REFERENCES observation_revision(id),
 entity_id bigint NOT NULL REFERENCES entity(id),
 relation_type text NOT NULL,
 confidence double precision NOT NULL CHECK(confidence BETWEEN 0 AND 1),
 derivation_version text NOT NULL,
 provenance jsonb NOT NULL DEFAULT '{}'::jsonb,
 linked_at timestamptz NOT NULL,
 PRIMARY KEY(observation_revision_id, entity_id, relation_type, derivation_version)
);
CREATE INDEX IF NOT EXISTS observation_entity_entity ON observation_entity(entity_id);

CREATE TABLE IF NOT EXISTS claim (
 id bigserial PRIMARY KEY,
 claim_type text NOT NULL,
 canonical_text text NOT NULL,
 clustering_policy_version text NOT NULL,
 clustering_fingerprint text,
 created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS claim_cluster_lookup
 ON claim(clustering_policy_version, clustering_fingerprint)
 WHERE clustering_fingerprint IS NOT NULL;
COMMENT ON COLUMN claim.clustering_fingerprint IS
 'Versioned clustering aid only. It is not the durable identity of a claim.';

CREATE TABLE IF NOT EXISTS observation_claim (
 observation_revision_id bigint NOT NULL REFERENCES observation_revision(id),
 claim_id bigint NOT NULL REFERENCES claim(id),
 relation_type text NOT NULL,
 confidence double precision NOT NULL CHECK(confidence BETWEEN 0 AND 1),
 derivation_version text NOT NULL,
 provenance jsonb NOT NULL DEFAULT '{}'::jsonb,
 linked_at timestamptz NOT NULL,
 PRIMARY KEY(observation_revision_id, claim_id, relation_type, derivation_version)
);
CREATE INDEX IF NOT EXISTS observation_claim_claim ON observation_claim(claim_id);

CREATE TABLE IF NOT EXISTS claim_relation (
 from_claim_id bigint NOT NULL REFERENCES claim(id),
 to_claim_id bigint NOT NULL REFERENCES claim(id),
 relation_type text NOT NULL,
 confidence double precision NOT NULL CHECK(confidence BETWEEN 0 AND 1),
 derivation_version text NOT NULL,
 provenance jsonb NOT NULL DEFAULT '{}'::jsonb,
 linked_at timestamptz NOT NULL,
 PRIMARY KEY(from_claim_id, to_claim_id, relation_type, derivation_version),
 CHECK(from_claim_id <> to_claim_id)
);

CREATE TABLE IF NOT EXISTS event (
 id bigserial PRIMARY KEY,
 event_type text NOT NULL,
 entity_id bigint REFERENCES entity(id),
 event_key text NOT NULL UNIQUE,
 occurred_at timestamptz,
 first_observed_at timestamptz NOT NULL,
 methodology_version text NOT NULL,
 payload jsonb NOT NULL DEFAULT '{}'::jsonb,
 created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS event_entity_time ON event(entity_id, occurred_at);

CREATE TABLE IF NOT EXISTS event_observation (
 event_id bigint NOT NULL REFERENCES event(id),
 observation_revision_id bigint NOT NULL REFERENCES observation_revision(id),
 relation_type text NOT NULL,
 confidence double precision NOT NULL CHECK(confidence BETWEEN 0 AND 1),
 derivation_version text NOT NULL,
 provenance jsonb NOT NULL DEFAULT '{}'::jsonb,
 linked_at timestamptz NOT NULL,
 PRIMARY KEY(event_id, observation_revision_id, relation_type, derivation_version)
);

-- Assessments are a ledger, not a current-status field. A later assessment appends a row.
CREATE TABLE IF NOT EXISTS claim_event_assessment (
 id bigserial PRIMARY KEY,
 claim_id bigint NOT NULL REFERENCES claim(id),
 event_id bigint REFERENCES event(id),
 status text NOT NULL CHECK(status IN
   ('unresolved','partially_confirmed','confirmed','contradicted','retracted','stale')),
 assessed_at timestamptz NOT NULL,
 methodology_version text NOT NULL,
 rationale text,
 created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS claim_event_assessment_history
 ON claim_event_assessment(claim_id, assessed_at, id);

CREATE TABLE IF NOT EXISTS claim_event_assessment_evidence (
 assessment_id bigint NOT NULL REFERENCES claim_event_assessment(id),
 observation_revision_id bigint REFERENCES observation_revision(id),
 event_id bigint REFERENCES event(id),
 evidence_role text NOT NULL CHECK(evidence_role IN ('supports','disputes','context')),
 confidence double precision NOT NULL CHECK(confidence BETWEEN 0 AND 1),
 derivation_version text NOT NULL,
 provenance jsonb NOT NULL DEFAULT '{}'::jsonb,
 linked_at timestamptz NOT NULL,
 CHECK ((observation_revision_id IS NOT NULL)::integer + (event_id IS NOT NULL)::integer = 1)
);
CREATE UNIQUE INDEX IF NOT EXISTS claim_event_assessment_evidence_observation
 ON claim_event_assessment_evidence(assessment_id, observation_revision_id, evidence_role, derivation_version)
 WHERE observation_revision_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS claim_event_assessment_evidence_event
 ON claim_event_assessment_evidence(assessment_id, event_id, evidence_role, derivation_version)
 WHERE event_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS conversation_edge (
 from_observation_revision_id bigint NOT NULL REFERENCES observation_revision(id),
 to_observation_revision_id bigint NOT NULL REFERENCES observation_revision(id),
 edge_type text NOT NULL,
 confidence double precision NOT NULL CHECK(confidence BETWEEN 0 AND 1),
 derivation_version text NOT NULL,
 provenance jsonb NOT NULL DEFAULT '{}'::jsonb,
 observed_at timestamptz NOT NULL,
 PRIMARY KEY(from_observation_revision_id, to_observation_revision_id, edge_type, derivation_version),
 CHECK(from_observation_revision_id <> to_observation_revision_id)
);

CREATE TABLE IF NOT EXISTS propagation_edge (
 from_observation_revision_id bigint NOT NULL REFERENCES observation_revision(id),
 to_observation_revision_id bigint NOT NULL REFERENCES observation_revision(id),
 claim_id bigint REFERENCES claim(id),
 edge_type text NOT NULL,
 confidence double precision NOT NULL CHECK(confidence BETWEEN 0 AND 1),
 derivation_version text NOT NULL,
 provenance jsonb NOT NULL DEFAULT '{}'::jsonb,
 observed_at timestamptz NOT NULL,
 PRIMARY KEY(from_observation_revision_id, to_observation_revision_id, edge_type, derivation_version),
 CHECK(from_observation_revision_id <> to_observation_revision_id)
);

-- Immutable measured fact. The natural key includes methodology so a revised method appends,
-- rather than replacing, the old measurement.
CREATE TABLE IF NOT EXISTS market_measurement (
 id bigserial PRIMARY KEY,
 asset_entity_id bigint NOT NULL REFERENCES asset_identity(entity_id),
 venue text NOT NULL DEFAULT '',
 pool_address text NOT NULL DEFAULT '',
 anchor_type text NOT NULL,
 anchor_timestamp timestamptz NOT NULL,
 horizon_seconds bigint NOT NULL CHECK(horizon_seconds >= 0),
 price_source text NOT NULL,
 methodology_version text NOT NULL,
 measured_at timestamptz NOT NULL,
 anchor_price_usd double precision,
 price_usd double precision,
 return_pct double precision,
 fdv_usd double precision,
 liquidity_usd double precision,
 volume_usd double precision,
 raw_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
 UNIQUE(asset_entity_id, venue, pool_address, anchor_type, anchor_timestamp,
        horizon_seconds, price_source, methodology_version)
);
CREATE INDEX IF NOT EXISTS market_measurement_asset_anchor
 ON market_measurement(asset_entity_id, anchor_timestamp, horizon_seconds);

CREATE TABLE IF NOT EXISTS claim_outcome (
 claim_id bigint NOT NULL REFERENCES claim(id),
 market_measurement_id bigint NOT NULL REFERENCES market_measurement(id),
 relation_type text NOT NULL,
 confidence double precision NOT NULL CHECK(confidence BETWEEN 0 AND 1),
 derivation_version text NOT NULL,
 provenance jsonb NOT NULL DEFAULT '{}'::jsonb,
 linked_at timestamptz NOT NULL,
 PRIMARY KEY(claim_id, market_measurement_id, relation_type, derivation_version)
);

CREATE TABLE IF NOT EXISTS collection_run (
 id bigserial PRIMARY KEY,
 source_system text NOT NULL,
 collector_version text NOT NULL,
 query_version text,
 identity_policy_version text,
 started_at timestamptz NOT NULL,
 finished_at timestamptz,
 status text NOT NULL,
 returned_records integer,
 saved_records integer,
 budget_units bigint,
 metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS collection_diagnostic (
 id bigserial PRIMARY KEY,
 collection_run_id bigint NOT NULL REFERENCES collection_run(id),
 diagnostic_key text NOT NULL,
 status text,
 numeric_value double precision,
 text_value text,
 observed_at timestamptz NOT NULL,
 metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS collection_diagnostic_run ON collection_diagnostic(collection_run_id);

-- Enforce append-only history at the database boundary. Corrections use a new row/version.
CREATE OR REPLACE FUNCTION fusion_reject_mutation() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
 RAISE EXCEPTION '% is append-only; insert a new revision/version instead', TG_TABLE_NAME;
END;
$$;

DO $$
DECLARE t text;
BEGIN
 FOREACH t IN ARRAY ARRAY[
   'source_record_revision','observation_revision','observation_entity',
   'observation_claim','claim_relation','event_observation',
   'claim_event_assessment','claim_event_assessment_evidence',
   'conversation_edge','propagation_edge','market_measurement',
   'claim_outcome','collection_diagnostic'
 ] LOOP
   IF NOT EXISTS (
     SELECT 1 FROM pg_trigger
     WHERE tgname='fusion_immutable_'||t
       AND tgrelid=to_regclass(t)
   ) THEN
     EXECUTE format(
       'CREATE TRIGGER %I BEFORE UPDATE OR DELETE ON %I FOR EACH ROW EXECUTE FUNCTION fusion_reject_mutation()',
       'fusion_immutable_'||t, t
     );
   END IF;
 END LOOP;
END $$;

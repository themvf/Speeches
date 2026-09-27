-- Additive, collector-owned schema. Web reads never execute DDL.
CREATE TABLE IF NOT EXISTS backpack_assets (
 id bigserial PRIMARY KEY, token_symbol text NOT NULL, token_name text NOT NULL,
 solana_mint text NOT NULL UNIQUE, underlying_symbol text, underlying_exchange text,
 underlying_name text, asset_type text NOT NULL CHECK(asset_type IN ('common_stock','etf','other_security','bp')),
 issuer text NOT NULL, launch_date date, official_source text NOT NULL,
 source_verified_at timestamptz, verification_status text NOT NULL DEFAULT 'pending'
 CHECK(verification_status IN ('pending','official','manual_approved')),
 approval_notes text, active boolean NOT NULL DEFAULT true,
 registry_status text NOT NULL DEFAULT 'registered'
 CHECK(registry_status IN ('registered','launched','paused','redeemed','inactive')),
 first_official_seen_at timestamptz, last_official_seen_at timestamptz,
 launch_evidence_source text, redemption_evidence_source text,
 deposit_enabled boolean, withdraw_enabled boolean, identity_fingerprint text,
 created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
 CHECK(asset_type <> 'bp' OR solana_mint='BPxxfRCXkUVhig4HS1Lh7kZqV6SPJhzfEk4x6fVBjPCy'),
 CHECK(verification_status='pending' OR source_verified_at IS NOT NULL)
);
ALTER TABLE backpack_assets ADD COLUMN IF NOT EXISTS registry_status text NOT NULL DEFAULT 'registered';
ALTER TABLE backpack_assets ADD COLUMN IF NOT EXISTS first_official_seen_at timestamptz;
ALTER TABLE backpack_assets ADD COLUMN IF NOT EXISTS last_official_seen_at timestamptz;
ALTER TABLE backpack_assets ADD COLUMN IF NOT EXISTS launch_evidence_source text;
ALTER TABLE backpack_assets ADD COLUMN IF NOT EXISTS redemption_evidence_source text;
ALTER TABLE backpack_assets ADD COLUMN IF NOT EXISTS deposit_enabled boolean;
ALTER TABLE backpack_assets ADD COLUMN IF NOT EXISTS withdraw_enabled boolean;
ALTER TABLE backpack_assets ADD COLUMN IF NOT EXISTS identity_fingerprint text;
DO $$ BEGIN
 ALTER TABLE backpack_assets ADD CONSTRAINT backpack_assets_registry_status_check
 CHECK(registry_status IN ('registered','launched','paused','redeemed','inactive'));
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
CREATE UNIQUE INDEX IF NOT EXISTS backpack_one_bp ON backpack_assets(asset_type) WHERE asset_type='bp';
CREATE TABLE IF NOT EXISTS backpack_ingestion_runs (
 run_id uuid PRIMARY KEY, snapshot_date date NOT NULL, started_at timestamptz NOT NULL DEFAULT now(),
 completed_at timestamptz, status text NOT NULL DEFAULT 'running',
 assets_attempted int NOT NULL DEFAULT 0, assets_succeeded int NOT NULL DEFAULT 0,
 assets_failed int NOT NULL DEFAULT 0, assets_skipped int NOT NULL DEFAULT 0,
 source_errors text NOT NULL DEFAULT '', data_quality_warnings text NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS backpack_provider_usage (
 run_id uuid REFERENCES backpack_ingestion_runs, provider text, requests int NOT NULL,
 credits numeric, estimated_cost_usd numeric, methodology text NOT NULL,
 PRIMARY KEY(run_id,provider)
);
CREATE TABLE IF NOT EXISTS backpack_wallet_labels (
 wallet_address text PRIMARY KEY, label text NOT NULL CHECK(label IN
 ('Backpack','Treasury','Custody','Market Maker','DEX','Liquidity Pool','Lending Protocol','Bridge','Known Exchange','Protocol','Unknown')),
 entity text NOT NULL, confidence text NOT NULL CHECK(confidence IN ('confirmed','high','medium','low')),
 source text NOT NULL, verified_at timestamptz NOT NULL, notes text NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS backpack_protocol_labels (
 protocol_id text PRIMARY KEY, name text NOT NULL, category text NOT NULL,
 confidence text NOT NULL CHECK(confidence IN ('confirmed','high','medium','low')),
 source text NOT NULL, verified_at timestamptz NOT NULL
);
CREATE TABLE IF NOT EXISTS backpack_asset_daily_snapshots (
 asset_id bigint REFERENCES backpack_assets, date date, run_id uuid NOT NULL REFERENCES backpack_ingestion_runs,
 captured_at timestamptz NOT NULL, slot bigint NOT NULL, block_timestamp timestamptz,
 source text NOT NULL, history_kind text NOT NULL DEFAULT 'captured' CHECK(history_kind IN ('captured','reconstructed')),
 token_supply numeric NOT NULL, decimals int NOT NULL,
 underlying_price numeric, underlying_price_timestamp timestamptz, previous_official_close numeric,
 underlying_market_open boolean, underlying_price_source text,
 onchain_price numeric, onchain_price_timestamp timestamptz,
 reference_aum_usd numeric, onchain_market_value_usd numeric, premium_discount_pct numeric,
 holder_count bigint, unique_holders bigint,
 holders_over_100 bigint, holders_over_1000 bigint, holders_over_10000 bigint, holders_over_100000 bigint,
 top_10_holder_pct numeric, top_20_holder_pct numeric, top_50_holder_pct numeric, top_100_holder_pct numeric,
 economic_top_10_holder_pct numeric, economic_top_20_holder_pct numeric,
 economic_top_50_holder_pct numeric, economic_top_100_holder_pct numeric,
 holders_complete boolean NOT NULL DEFAULT false, holder_start_slot bigint, holder_end_slot bigint,
 daily_swap_volume_usd numeric, observed_swap_volume_usd numeric, daily_transfer_volume_usd numeric,
 unique_traders bigint, observed_unique_traders bigint, unique_buyers bigint, unique_sellers bigint,
 new_holders bigint, lost_holders bigint, net_supply_change_tokens numeric, net_supply_change_usd numeric,
 minted_tokens numeric, burned_tokens numeric, volume_to_aum_ratio numeric,
 defi_value_usd numeric, defi_utilization_pct numeric,
 regular_session_volume_usd numeric, after_hours_volume_usd numeric, after_hours_volume_pct numeric,
 quote_1k_price_impact numeric, quote_10k_price_impact numeric, quote_50k_price_impact numeric, quote_100k_price_impact numeric,
 data_quality_score int NOT NULL, quality_status text NOT NULL CHECK(quality_status IN ('Verified','Estimated','Partial','Stale','Unavailable')),
 PRIMARY KEY(asset_id,date)
);
CREATE INDEX IF NOT EXISTS backpack_snapshots_date ON backpack_asset_daily_snapshots(date);
CREATE TABLE IF NOT EXISTS backpack_asset_holder_daily_snapshots (
 asset_id bigint, date date, wallet_address text,
 balance_tokens numeric NOT NULL, value_usd numeric, excluded boolean NOT NULL,
 label text NOT NULL, source text NOT NULL, slot bigint NOT NULL, ingested_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(asset_id,date,wallet_address),
 FOREIGN KEY(asset_id,date) REFERENCES backpack_asset_daily_snapshots
);
CREATE INDEX IF NOT EXISTS backpack_holders_wallet_date ON backpack_asset_holder_daily_snapshots(wallet_address,date);
CREATE INDEX IF NOT EXISTS backpack_holders_date ON backpack_asset_holder_daily_snapshots(date);
CREATE TABLE IF NOT EXISTS backpack_asset_dex_daily_snapshots (
 asset_id bigint, date date, venue text, observed_volume_usd numeric, trades bigint,
 unique_traders bigint, median_trade_size numeric, average_trade_size numeric,
 p95_trade_size numeric, max_trade_size numeric, coverage_status text NOT NULL,
 source text NOT NULL, PRIMARY KEY(asset_id,date,venue),
 FOREIGN KEY(asset_id,date) REFERENCES backpack_asset_daily_snapshots
);
CREATE TABLE IF NOT EXISTS backpack_asset_liquidity_daily_snapshots (
 asset_id bigint, date date, direction text CHECK(direction IN ('buy','sell')), notional_usd numeric,
 input_amount numeric, expected_output numeric, minimum_output numeric, price_impact_pct numeric,
 route jsonb, route_legs int, quote_at timestamptz NOT NULL, slot bigint,
 status text NOT NULL, limitation text NOT NULL, source text NOT NULL,
 PRIMARY KEY(asset_id,date,direction,notional_usd),
 FOREIGN KEY(asset_id,date) REFERENCES backpack_asset_daily_snapshots
);
CREATE TABLE IF NOT EXISTS backpack_asset_defi_daily_snapshots (
 asset_id bigint, date date, protocol_id text REFERENCES backpack_protocol_labels,
 category text NOT NULL, value_usd numeric, source text NOT NULL, status text NOT NULL,
 PRIMARY KEY(asset_id,date,protocol_id), FOREIGN KEY(asset_id,date) REFERENCES backpack_asset_daily_snapshots
);
CREATE TABLE IF NOT EXISTS backpack_bp_daily_snapshots (
 date date PRIMARY KEY, asset_id bigint NOT NULL, estimated_circulating_supply numeric,
 market_cap_usd numeric, fdv_usd numeric, whale_count bigint, new_whales bigint,
 whale_net_accumulation_tokens numeric, liquidity_usd numeric, staked_bp numeric,
 circulating_source text, staking_source text,
 FOREIGN KEY(asset_id,date) REFERENCES backpack_asset_daily_snapshots,
 CHECK(staked_bp IS NULL OR staking_source IS NOT NULL),
 CHECK(estimated_circulating_supply IS NULL OR circulating_source IS NOT NULL)
);
CREATE TABLE IF NOT EXISTS backpack_ecosystem_daily_snapshots (
 date date PRIMARY KEY, run_id uuid REFERENCES backpack_ingestion_runs,
 assets_expected int NOT NULL, assets_captured int NOT NULL,
 reference_aum_usd numeric, onchain_market_value_usd numeric, meaningful_holders bigint,
 multi_asset_1 bigint, multi_asset_2 bigint, multi_asset_3 bigint, multi_asset_5 bigint,
 multi_asset_adoption_pct numeric, net_supply_change_usd numeric, daily_swap_volume_usd numeric,
 observed_swap_volume_usd numeric, defi_utilization_pct numeric, after_hours_volume_pct numeric,
 captured_at timestamptz NOT NULL DEFAULT now(), quality_status text NOT NULL
);
CREATE TABLE IF NOT EXISTS backpack_market_prices (
 asset_id bigint REFERENCES backpack_assets, price_at timestamptz, source text,
 price numeric NOT NULL, price_kind text NOT NULL, market_open boolean,
 ingested_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(asset_id,price_at,source,price_kind)
);
CREATE TABLE IF NOT EXISTS backpack_data_quality_events (
 id bigserial PRIMARY KEY, run_id uuid REFERENCES backpack_ingestion_runs,
 asset_id bigint REFERENCES backpack_assets, date date, metric text NOT NULL,
 status text NOT NULL CHECK(status IN ('Verified','Estimated','Partial','Stale','Unavailable')),
 source text NOT NULL, calculation text NOT NULL, limitation text NOT NULL,
 observed_at timestamptz, ingested_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS backpack_quality_asset_date ON backpack_data_quality_events(asset_id,date);
CREATE TABLE IF NOT EXISTS backpack_transactions (
 asset_id bigint REFERENCES backpack_assets, signature text, event_kind text,
 slot bigint NOT NULL, timestamp timestamptz NOT NULL, wallet_address text,
 side text, tokens numeric, volume_usd numeric, venue text, source text NOT NULL,
 ingested_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(asset_id,signature,event_kind)
);
CREATE INDEX IF NOT EXISTS backpack_transactions_time ON backpack_transactions(asset_id,timestamp);
CREATE TABLE IF NOT EXISTS backpack_transaction_cursors (
 asset_id bigint REFERENCES backpack_assets, wallet_address text, last_signature text,
 through_at timestamptz NOT NULL, PRIMARY KEY(asset_id,wallet_address)
);
CREATE TABLE IF NOT EXISTS backpack_thesis_milestones (
 id bigserial PRIMARY KEY, metric text NOT NULL, threshold numeric NOT NULL,
 label text NOT NULL, achieved_at date, active boolean NOT NULL DEFAULT true,
 UNIQUE(metric,threshold)
);
CREATE TABLE IF NOT EXISTS backpack_job_leases (
 name text PRIMARY KEY, owner uuid NOT NULL, expires_at timestamptz NOT NULL
);

-- Immutable label revisions and label evidence captured with each holder snapshot.
CREATE TABLE IF NOT EXISTS backpack_wallet_label_revisions (
 id bigserial PRIMARY KEY, wallet_address text NOT NULL,
 label text NOT NULL, entity text NOT NULL, confidence text NOT NULL,
 source text NOT NULL, verified_at timestamptz NOT NULL, notes text NOT NULL,
 recorded_at timestamptz NOT NULL DEFAULT now(), actor text NOT NULL DEFAULT 'authenticated_admin'
);
CREATE INDEX IF NOT EXISTS backpack_label_revision_wallet ON backpack_wallet_label_revisions(wallet_address,recorded_at);
ALTER TABLE backpack_asset_holder_daily_snapshots ADD COLUMN IF NOT EXISTS label_entity text;
ALTER TABLE backpack_asset_holder_daily_snapshots ADD COLUMN IF NOT EXISTS label_confidence text;
ALTER TABLE backpack_asset_holder_daily_snapshots ADD COLUMN IF NOT EXISTS label_source text;
ALTER TABLE backpack_asset_holder_daily_snapshots ADD COLUMN IF NOT EXISTS label_verified_at timestamptz;
CREATE TABLE IF NOT EXISTS backpack_bp_whale_daily_snapshots (
 asset_id bigint NOT NULL, date date NOT NULL, threshold_usd numeric NOT NULL CHECK(threshold_usd>0),
 whale_count bigint, new_whales bigint, exited_whales bigint, whale_net_accumulation_tokens numeric,
 status text NOT NULL, source text NOT NULL, methodology text NOT NULL,
 PRIMARY KEY(asset_id,date,threshold_usd),
 FOREIGN KEY(asset_id,date) REFERENCES backpack_asset_daily_snapshots
);

CREATE TABLE IF NOT EXISTS backpack_readiness_checks (
 run_id uuid NOT NULL, check_name text NOT NULL, status text NOT NULL,
 detail text NOT NULL, checked_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(run_id,check_name)
);
CREATE INDEX IF NOT EXISTS backpack_readiness_recent ON backpack_readiness_checks(checked_at DESC);
CREATE TABLE IF NOT EXISTS backpack_readiness_usage (
 run_id uuid NOT NULL, provider text NOT NULL, requests int NOT NULL,
 PRIMARY KEY(run_id,provider)
);

-- Bounded raw history; permanent state changes and reconciliation checkpoints.
CREATE TABLE IF NOT EXISTS backpack_current_holders (
 asset_id bigint REFERENCES backpack_assets, wallet_address text, balance_tokens numeric NOT NULL,
 value_usd numeric, excluded boolean NOT NULL, label text, label_confidence text,
 label_entity text, label_source text, label_verified_at timestamptz,
 first_seen_at date NOT NULL, last_seen_at date NOT NULL, updated_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(asset_id,wallet_address)
);
CREATE TABLE IF NOT EXISTS backpack_holder_events (
 asset_id bigint REFERENCES backpack_assets, wallet_address text, date date, event_type text,
 previous_balance numeric, new_balance numeric, previous_value_usd numeric, new_value_usd numeric,
 change_tokens numeric, change_usd numeric, previous_label text, new_label text,
 previous_confidence text, new_confidence text, previous_observation_date date,
 source text NOT NULL, observed_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(asset_id,wallet_address,date,event_type)
);
CREATE INDEX IF NOT EXISTS backpack_holder_events_date ON backpack_holder_events(date);
CREATE TABLE IF NOT EXISTS backpack_holder_checkpoints (
 asset_id bigint, date date, owner_count bigint NOT NULL, balance_tokens numeric NOT NULL,
 aggregates_validated boolean NOT NULL, methodology text NOT NULL,
 PRIMARY KEY(asset_id,date), FOREIGN KEY(asset_id,date) REFERENCES backpack_asset_daily_snapshots
);
CREATE TABLE IF NOT EXISTS backpack_transaction_evidence (
 asset_id bigint REFERENCES backpack_assets, signature text, event_kind text,
 reason text NOT NULL, source text NOT NULL, recorded_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(asset_id,signature,event_kind)
);
-- Explicit attestation required before deleting a raw activity day. The current sampled
-- collector does not attest complete market-wide trading aggregates.
CREATE TABLE IF NOT EXISTS backpack_transaction_retention_checks (
 asset_id bigint REFERENCES backpack_assets, activity_date date,
 aggregates_validated boolean NOT NULL DEFAULT false, methodology text NOT NULL,
 validated_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(asset_id,activity_date)
);
CREATE TABLE IF NOT EXISTS backpack_storage_observations (
 date date, relation_name text, table_bytes bigint, index_bytes bigint, total_bytes bigint,
 estimated_rows bigint, database_bytes bigint, database_connections bigint,
 measured_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(date,relation_name)
);
CREATE TABLE IF NOT EXISTS backpack_operational_alerts (
 date date, metric text, detail text NOT NULL, observed_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(date,metric)
);
CREATE TABLE IF NOT EXISTS backpack_analytical_daily_metrics (
 date date, scope_asset_id bigint NOT NULL, metric text, period_days int NOT NULL,
 value numeric, status text NOT NULL, methodology text NOT NULL,
 computed_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(date,scope_asset_id,metric,period_days)
);
CREATE TABLE IF NOT EXISTS backpack_index_observations (
 date date, index_name text, table_name text NOT NULL, bytes bigint NOT NULL,
 measured_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(date,index_name)
);

-- Issuer-neutral coverage foundation; a registry entry is not a market-share denominator.
CREATE TABLE IF NOT EXISTS tokenized_security_issuers (
 id text PRIMARY KEY, name text NOT NULL UNIQUE
);
INSERT INTO tokenized_security_issuers VALUES ('backpack','Backpack'),('xstocks','xStocks'),('ondo','Ondo') ON CONFLICT DO NOTHING;
CREATE TABLE IF NOT EXISTS tokenized_security_assets (
 id bigserial PRIMARY KEY, issuer_id text NOT NULL REFERENCES tokenized_security_issuers,
 network text NOT NULL, mint text NOT NULL, token_symbol text NOT NULL,
 underlying_symbol text NOT NULL, underlying_exchange text NOT NULL, underlying_name text NOT NULL,
 asset_type text NOT NULL CHECK(asset_type IN ('common_stock','etf','other_security')),
 official_source text NOT NULL, verified_at timestamptz NOT NULL,
 verification_status text NOT NULL CHECK(verification_status IN ('official','manual_approved')),
 backpack_asset_id bigint UNIQUE REFERENCES backpack_assets,
 active boolean NOT NULL DEFAULT true, UNIQUE(network,mint)
);
CREATE INDEX IF NOT EXISTS tokenized_security_assets_issuer ON tokenized_security_assets(issuer_id);
CREATE TABLE IF NOT EXISTS tokenized_security_daily_snapshots (
 asset_id bigint REFERENCES tokenized_security_assets, date date,
 reference_aum_usd numeric, net_issuance_usd numeric, meaningful_holders bigint, daily_swap_volume_usd numeric,
 source text NOT NULL, observed_at timestamptz NOT NULL, methodology text NOT NULL,
 PRIMARY KEY(asset_id,date)
);
CREATE TABLE IF NOT EXISTS tokenized_security_market_snapshots (
 date date, universe_id text, reference_aum_usd numeric, net_issuance_usd numeric,
 meaningful_holders bigint, daily_swap_volume_usd numeric,
 coverage_status text NOT NULL CHECK(coverage_status IN ('Verified','Estimated','Partial','Unavailable')),
 methodology text NOT NULL, source text NOT NULL, observed_at timestamptz NOT NULL,
 PRIMARY KEY(date,universe_id)
);
CREATE TABLE IF NOT EXISTS tokenized_security_market_members (
 date date, universe_id text, asset_id bigint REFERENCES tokenized_security_assets,
 PRIMARY KEY(date,universe_id,asset_id),
 FOREIGN KEY(date,universe_id) REFERENCES tokenized_security_market_snapshots
);
CREATE TABLE IF NOT EXISTS backpack_environment_daily (
 date date, period_days int CHECK(period_days IN (7,30)), state text NOT NULL,
 reason text NOT NULL, issuance_aum_pct numeric, aum_growth_pct numeric,
 holder_growth_pct numeric, trading_growth_pct numeric,
 issuance_threshold_pct numeric NOT NULL, trading_threshold_pct numeric NOT NULL,
 growth_threshold_pct numeric NOT NULL, methodology text NOT NULL,
 PRIMARY KEY(date,period_days)
);
CREATE TABLE IF NOT EXISTS backpack_billing_observations (
 date date, provider text, scope text CHECK(scope IN ('backpack','shared')), metric text,
 value numeric NOT NULL CHECK(value>=0), unit text NOT NULL, source text NOT NULL,
 imported_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(date,provider,scope,metric)
);
CREATE TABLE IF NOT EXISTS backpack_adoption_daily (
 date date PRIMARY KEY, data jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS backpack_adoption_assessments (
 date date, period_days int CHECK(period_days IN (7,30,90)), data jsonb NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(date,period_days)
);
CREATE TABLE IF NOT EXISTS backpack_growth_daily (
 date date, period_days int CHECK(period_days IN (7,30,90)), state text NOT NULL, momentum text NOT NULL,
 reason text NOT NULL, issuance_threshold_pct numeric NOT NULL, holder_threshold_pct numeric NOT NULL,
 slowdown_threshold_pp numeric NOT NULL, net_issuance_usd numeric, previous_net_issuance_usd numeric,
 issuance_aum_pct numeric, previous_issuance_aum_pct numeric, holder_growth_pct numeric, previous_holder_growth_pct numeric,
 aum_growth_pct numeric, meaningful_holders bigint, previous_meaningful_holders bigint,
 multi_asset_adoption_pct numeric, previous_multi_asset_adoption_pct numeric,
 top_5_aum_pct numeric, previous_top_5_aum_pct numeric, significant_securities bigint, previous_significant_securities bigint,
 significance_threshold_usd numeric, supply_effect_usd numeric, price_effect_usd numeric, methodology text NOT NULL,
 PRIMARY KEY(date,period_days)
);

-- Phase A: immutable official-universe reconciliation and sourced context.
CREATE TABLE IF NOT EXISTS backpack_asset_registry_daily (
 asset_id bigint REFERENCES backpack_assets, date date,
 official_present boolean NOT NULL, deposit_enabled boolean, withdraw_enabled boolean,
 token_supply numeric, lifecycle_state text NOT NULL
 CHECK(lifecycle_state IN ('registered','launched','paused','redeemed','inactive')),
 source text NOT NULL, observed_at timestamptz NOT NULL,
 methodology_version text NOT NULL, quality_status text NOT NULL
 CHECK(quality_status IN ('Verified','Estimated','Partial','Stale','Unavailable')),
 limitation text NOT NULL DEFAULT '', identity_fingerprint text NOT NULL,
 PRIMARY KEY(asset_id,date)
);
CREATE INDEX IF NOT EXISTS backpack_registry_daily_date ON backpack_asset_registry_daily(date);

CREATE TABLE IF NOT EXISTS backpack_registry_candidates_daily (
 date date, token_symbol text, solana_mint text, token_name text NOT NULL,
 decimals int, deposit_enabled boolean, withdraw_enabled boolean,
 security_name text, cusip text, match_status text NOT NULL
 CHECK(match_status IN ('approved','unresolved','conflict')),
 matched_asset_id bigint REFERENCES backpack_assets, source text NOT NULL,
 observed_at timestamptz NOT NULL, detail text NOT NULL,
 PRIMARY KEY(date,token_symbol,solana_mint)
);
CREATE INDEX IF NOT EXISTS backpack_registry_candidates_status ON backpack_registry_candidates_daily(date,match_status);

CREATE TABLE IF NOT EXISTS backpack_external_observations (
 id bigserial PRIMARY KEY, evidence_key text NOT NULL UNIQUE, metric text NOT NULL,
 scope_type text NOT NULL, scope_id text NOT NULL,
 period_start date, period_end date, comparison_start date, comparison_end date,
 period_days int, value numeric, unit text NOT NULL,
 observation_kind text NOT NULL CHECK(observation_kind IN ('total','change','percentage','share','rank')),
 source_url text NOT NULL, source_publisher text NOT NULL, published_at timestamptz NOT NULL,
 primary_source_url text, methodology text NOT NULL,
 coverage_status text NOT NULL CHECK(coverage_status IN ('Verified','Estimated','Partial','Unavailable')),
 review_status text NOT NULL CHECK(review_status IN ('pending','approved','rejected','superseded')),
 limitation text NOT NULL, recorded_at timestamptz NOT NULL DEFAULT now(),
 recorded_by text NOT NULL DEFAULT 'committed_evidence'
);
CREATE INDEX IF NOT EXISTS backpack_external_observations_metric ON backpack_external_observations(metric,published_at DESC);

CREATE TABLE IF NOT EXISTS backpack_economy_events (
 event_key text PRIMARY KEY, event_at timestamptz NOT NULL, event_type text NOT NULL,
 title text NOT NULL, description text NOT NULL, asset_symbols jsonb NOT NULL DEFAULT '[]'::jsonb,
 source_url text NOT NULL, evidence_status text NOT NULL
 CHECK(evidence_status IN ('Verified','Estimated','Partial','Unavailable')),
 campaign_end_at timestamptz, limitation text NOT NULL DEFAULT '',
 recorded_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS backpack_economy_events_at ON backpack_economy_events(event_at DESC);

-- BP holder intelligence (docs/bp-holder-intelligence-spec.md). Collector-owned and additive.
-- Vesting and Burn join the evidence-backed label set; replaced only while the old definition lacks them,
-- so a routine --migrate never rewrites the constraint.
DO $$ BEGIN
 IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid='backpack_wallet_labels'::regclass
   AND conname='backpack_wallet_labels_label_check' AND pg_get_constraintdef(oid) LIKE '%Vesting%') THEN
  ALTER TABLE backpack_wallet_labels DROP CONSTRAINT IF EXISTS backpack_wallet_labels_label_check;
  ALTER TABLE backpack_wallet_labels ADD CONSTRAINT backpack_wallet_labels_label_check CHECK(label IN
  ('Backpack','Treasury','Custody','Market Maker','DEX','Liquidity Pool','Lending Protocol','Bridge','Known Exchange','Protocol','Vesting','Burn','Unknown'));
 END IF;
END $$;
-- Intelligence runs share the run/usage ledger; readers of the daily capture filter job='daily'.
ALTER TABLE backpack_ingestion_runs ADD COLUMN IF NOT EXISTS job text NOT NULL DEFAULT 'daily';
CREATE INDEX IF NOT EXISTS backpack_runs_job_started ON backpack_ingestion_runs(job,started_at DESC);

-- Stored rankings survive raw holder-snapshot retention; rankings are reproducible from them.
CREATE TABLE IF NOT EXISTS bp_holder_rankings (
 source_date date, ranking text CHECK(ranking IN ('raw','filtered')), rank int,
 wallet_address text NOT NULL, raw_balance numeric NOT NULL CHECK(raw_balance>0), decimals int NOT NULL,
 label text NOT NULL, label_confidence text, label_entity text, label_source text, excluded boolean NOT NULL,
 PRIMARY KEY(source_date,ranking,rank)
);
CREATE TABLE IF NOT EXISTS bp_cohorts (
 version_id bigserial PRIMARY KEY, kind text NOT NULL CHECK(kind IN ('original','current')),
 effective_at timestamptz NOT NULL, source_date date NOT NULL, source_run_id uuid,
 source_slot_start bigint, source_slot_end bigint, derived_from_version bigint REFERENCES bp_cohorts,
 exclusion_fingerprint text NOT NULL, excluded_count int NOT NULL, eligible_count int NOT NULL,
 target_size int NOT NULL, exit_rank int NOT NULL, exit_runs int NOT NULL, entrant_cap int NOT NULL,
 size int NOT NULL, entered int NOT NULL, left_count int NOT NULL, rank_changed int NOT NULL,
 queued int NOT NULL, entrant_cap_bound boolean NOT NULL, bootstrap boolean NOT NULL,
 status text NOT NULL CHECK(status IN ('Estimated','Partial')), methodology text NOT NULL,
 approved_at timestamptz, approved_by text, approval_notes text,
 created_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(kind,source_date),
 CHECK(kind='current' OR (approved_at IS NOT NULL AND derived_from_version IS NOT NULL))
);
CREATE UNIQUE INDEX IF NOT EXISTS bp_one_original_cohort ON bp_cohorts(kind) WHERE kind='original';
CREATE TABLE IF NOT EXISTS bp_cohort_members (
 version_id bigint REFERENCES bp_cohorts, wallet_address text, member boolean NOT NULL,
 rank int, raw_balance numeric, previous_rank int,
 event text NOT NULL CHECK(event IN ('bootstrap','entered','left','rank_changed','unchanged','queued')),
 below_exit_runs int NOT NULL DEFAULT 0, queued_since timestamptz, exit_reason text,
 PRIMARY KEY(version_id,wallet_address)
);
CREATE INDEX IF NOT EXISTS bp_cohort_members_wallet ON bp_cohort_members(wallet_address);
CREATE TABLE IF NOT EXISTS bp_tracked_wallets (
 wallet_address text PRIMARY KEY, tracked_since timestamptz NOT NULL, history_start timestamptz NOT NULL,
 first_version_id bigint NOT NULL REFERENCES bp_cohorts, active boolean NOT NULL,
 updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS bp_tracked_assets (
 network text NOT NULL DEFAULT 'solana-mainnet', mint text,
 asset_class text NOT NULL CHECK(asset_class IN ('native','fungible','nft','position','unknown')),
 token_program text, decimals int, symbol text, name text, metadata_source text,
 extension_flags text[] NOT NULL DEFAULT '{}', extensions jsonb NOT NULL DEFAULT '{}'::jsonb,
 is_sol boolean NOT NULL DEFAULT false, is_stable boolean NOT NULL DEFAULT false, is_bp boolean NOT NULL DEFAULT false,
 spam_class text NOT NULL DEFAULT 'none' CHECK(spam_class IN ('none','suspected_spam')), spam_reason text, class_reason text,
 liquidity_usd numeric, market_created_at timestamptz, market_observed_at timestamptz,
 first_seen_at timestamptz NOT NULL DEFAULT now(), metadata_at timestamptz, updated_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(network,mint)
);
CREATE TABLE IF NOT EXISTS bp_portfolio_wallets (
 run_id uuid REFERENCES backpack_ingestion_runs, wallet_address text,
 status text NOT NULL CHECK(status IN ('complete','partial','unavailable','oversized')),
 sol_status text NOT NULL, spl_status text NOT NULL, token2022_status text NOT NULL,
 token_accounts int, slot_min bigint, slot_max bigint, observed_at timestamptz NOT NULL,
 priced_value_usd numeric, unpriced_holdings int, detail text NOT NULL DEFAULT '',
 PRIMARY KEY(run_id,wallet_address)
);
CREATE TABLE IF NOT EXISTS bp_portfolio_balances (
 run_id uuid, wallet_address text, network text NOT NULL DEFAULT 'solana-mainnet', mint text,
 raw_amount numeric NOT NULL, decimals int NOT NULL, ui_amount text,
 token_accounts jsonb NOT NULL, slot bigint, observed_at timestamptz NOT NULL,
 price numeric, price_at timestamptz, price_source text,
 pricing_status text NOT NULL CHECK(pricing_status IN ('priced','stale','unpriced')),
 value_usd numeric, frozen boolean NOT NULL DEFAULT false,
 balance_visibility text NOT NULL CHECK(balance_visibility IN ('full','partial')),
 holding_class text NOT NULL, spam_class text NOT NULL, dust boolean, classification_reason text NOT NULL DEFAULT '',
 PRIMARY KEY(run_id,wallet_address,network,mint),
 FOREIGN KEY(run_id,wallet_address) REFERENCES bp_portfolio_wallets,
 CHECK(raw_amount>0 OR balance_visibility='partial'),
 CHECK(value_usd IS NULL OR pricing_status='priced')
);
CREATE INDEX IF NOT EXISTS bp_portfolio_balances_mint ON bp_portfolio_balances(run_id,mint);
CREATE INDEX IF NOT EXISTS bp_portfolio_balances_wallet ON bp_portfolio_balances(wallet_address,mint,observed_at DESC);
CREATE TABLE IF NOT EXISTS bp_price_observations (
 network text NOT NULL DEFAULT 'solana-mainnet', mint text, price_at timestamptz, source text,
 price numeric NOT NULL CHECK(price>0), block_id bigint, observed_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(network,mint,price_at,source)
);
CREATE TABLE IF NOT EXISTS bp_raw_transactions (
 network text NOT NULL DEFAULT 'solana-mainnet', signature text, slot bigint NOT NULL, block_time timestamptz,
 finality text NOT NULL CHECK(finality IN ('finalized','confirmed')), source text NOT NULL,
 payload jsonb NOT NULL, enhanced jsonb, parser_version text NOT NULL,
 received_at timestamptz NOT NULL DEFAULT now(), ingested_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(network,signature)
);
CREATE INDEX IF NOT EXISTS bp_raw_transactions_time ON bp_raw_transactions(block_time);
CREATE TABLE IF NOT EXISTS bp_economic_events (
 network text NOT NULL DEFAULT 'solana-mainnet', signature text, wallet_address text, event_index int,
 slot bigint NOT NULL, block_time timestamptz,
 kind text NOT NULL CHECK(kind IN ('swap','transfer_in','transfer_out','wrap','unwrap','liquidity','delegate_transfer','failed','unclassified')),
 tier text NOT NULL CHECK(tier IN ('parsed_swap','inferred_swap','unclassified','not_applicable')),
 input_mint text, input_raw numeric, input_decimals int, output_mint text, output_raw numeric, output_decimals int,
 usd_value numeric, valuation_source text, valuation_status text NOT NULL CHECK(valuation_status IN ('estimated','unpriced','not_applicable')),
 pre_input_raw numeric, post_input_raw numeric, pre_output_raw numeric, post_output_raw numeric,
 pre_balance_source text CHECK(pre_balance_source IN ('transaction','snapshot')),
 new_position boolean, ata_created boolean, first_observed_purchase boolean, re_entry boolean,
 first_group_purchase boolean, recently_launched boolean, pre_membership boolean NOT NULL,
 owner_signed boolean NOT NULL, venue text, programs text[] NOT NULL DEFAULT '{}', detail text NOT NULL DEFAULT '',
 parser_version text NOT NULL, finality text NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(network,signature,wallet_address,event_index),
 CHECK(usd_value IS NULL OR valuation_status='estimated')
);
CREATE INDEX IF NOT EXISTS bp_events_wallet_time ON bp_economic_events(wallet_address,block_time);
CREATE INDEX IF NOT EXISTS bp_events_output_time ON bp_economic_events(output_mint,block_time) WHERE kind='swap';
CREATE INDEX IF NOT EXISTS bp_events_input_time ON bp_economic_events(input_mint,block_time) WHERE kind='swap';
-- Per-wallet history coverage. Separate from backpack_transaction_cursors, whose (asset, wallet) rows
-- carry the daily BP swap sampler's cursor semantics and must not be advanced by this job.
CREATE TABLE IF NOT EXISTS bp_history_coverage (
 wallet_address text PRIMARY KEY, source text NOT NULL, requested_start timestamptz NOT NULL,
 earliest_retrieved timestamptz, earliest_slot bigint, newest_slot bigint, newest_retrieved timestamptz,
 backfill_status text NOT NULL CHECK(backfill_status IN ('pending','incomplete','complete','capped')),
 backfill_pages int NOT NULL DEFAULT 0, poll_status text, last_poll_at timestamptz,
 gaps jsonb NOT NULL DEFAULT '[]'::jsonb, transactions bigint NOT NULL DEFAULT 0,
 updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS bp_alert_rules (
 rule text PRIMARY KEY CHECK(rule IN ('new_position','multiple_buyers','accumulation','major_sale')),
 enabled boolean NOT NULL DEFAULT true, params jsonb NOT NULL, updated_at timestamptz NOT NULL DEFAULT now()
);
INSERT INTO bp_alert_rules(rule,params) VALUES
 ('new_position','{"min_usd":250,"window_minutes":60,"cooldown_minutes":360}'),
 ('multiple_buyers','{"min_wallets":3,"window_minutes":60,"min_usd_per_wallet":250,"cooldown_minutes":360,"exclude_mints":["native","So11111111111111111111111111111111111111112","EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v","Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"]}'),
 ('accumulation','{"min_wallets":5,"window_minutes":1440,"min_usd_per_wallet":250,"cooldown_minutes":1440,"exclude_mints":["native","So11111111111111111111111111111111111111112","EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v","Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"]}'),
 ('major_sale','{"min_fraction":0.5,"window_minutes":60,"cooldown_minutes":360,"exclude_mints":["native","So11111111111111111111111111111111111111112","EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v","Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"]}')
ON CONFLICT DO NOTHING;
CREATE TABLE IF NOT EXISTS bp_alerts (
 alert_key text PRIMARY KEY, rule text NOT NULL, cohort_version_id bigint NOT NULL REFERENCES bp_cohorts,
 mint text NOT NULL, display_name text, window_start timestamptz NOT NULL, window_end timestamptz NOT NULL,
 wallet_count int NOT NULL, wallets jsonb NOT NULL, inferred_wallets int NOT NULL,
 lowest_tier text NOT NULL CHECK(lowest_tier IN ('parsed_swap','inferred_swap')),
 quantity_raw numeric, decimals int, usd_value numeric,
 valuation_status text NOT NULL CHECK(valuation_status IN ('priced','partially_priced','unpriced')),
 signatures jsonb NOT NULL, finality text NOT NULL, data_through timestamptz, params jsonb NOT NULL,
 detail text NOT NULL, first_raised_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS bp_alerts_cohort_time ON bp_alerts(cohort_version_id,window_start DESC);
-- Daily check: token balance change between two complete reads vs the net of classified events between their slots.
CREATE TABLE IF NOT EXISTS bp_reconciliation (
 date date, wallet_address text, mint text, from_run_id uuid NOT NULL, to_run_id uuid NOT NULL,
 from_slot bigint, to_slot bigint, snapshot_delta numeric NOT NULL, event_delta numeric NOT NULL, difference numeric NOT NULL,
 status text NOT NULL CHECK(status IN ('matched','discrepancy')), checked_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(date,wallet_address,mint)
);

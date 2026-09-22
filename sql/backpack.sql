-- Additive, collector-owned schema. Web reads never execute DDL.
CREATE TABLE IF NOT EXISTS backpack_assets (
 id bigserial PRIMARY KEY, token_symbol text NOT NULL, token_name text NOT NULL,
 solana_mint text NOT NULL UNIQUE, underlying_symbol text, underlying_exchange text,
 underlying_name text, asset_type text NOT NULL CHECK(asset_type IN ('common_stock','etf','other_security','bp')),
 issuer text NOT NULL, launch_date date, official_source text NOT NULL,
 source_verified_at timestamptz, verification_status text NOT NULL DEFAULT 'pending'
 CHECK(verification_status IN ('pending','official','manual_approved')),
 approval_notes text, active boolean NOT NULL DEFAULT true,
 created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
 CHECK(asset_type <> 'bp' OR solana_mint='BPxxfRCXkUVhig4HS1Lh7kZqV6SPJhzfEk4x6fVBjPCy'),
 CHECK(verification_status='pending' OR source_verified_at IS NOT NULL)
);
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

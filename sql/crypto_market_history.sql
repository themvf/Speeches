-- Immutable fetches and observations preserve earlier history and provider revisions.
CREATE TABLE IF NOT EXISTS crypto_market_sources (
 id text PRIMARY KEY, coin text NOT NULL,
 provider text NOT NULL, source_url text NOT NULL, metadata jsonb NOT NULL,
 is_default boolean NOT NULL DEFAULT false
);
CREATE UNIQUE INDEX IF NOT EXISTS crypto_market_default ON crypto_market_sources(coin) WHERE is_default;
CREATE TABLE IF NOT EXISTS crypto_market_fetches (
 id bigserial PRIMARY KEY, source_id text NOT NULL REFERENCES crypto_market_sources(id),
 retrieved_at timestamptz NOT NULL, request_url text NOT NULL,
 metadata jsonb NOT NULL, raw_response jsonb NOT NULL
);
CREATE INDEX IF NOT EXISTS crypto_market_fetch_time ON crypto_market_fetches(source_id,retrieved_at DESC);
CREATE TABLE IF NOT EXISTS crypto_market_observations (
 fetch_id bigint NOT NULL REFERENCES crypto_market_fetches(id), day date NOT NULL,
 sample_at timestamptz NOT NULL, close double precision NOT NULL CHECK(close>0),
 -- NULL means the provider does not report this interval's own trading. CoinGecko's market_chart
 -- returns a rolling 24h total, so a price_observation source stores no volume rather than a
 -- number that reads as hourly volume but is not.
 volume double precision CHECK(volume>=0),
 open double precision, high double precision, low double precision,
 complete boolean NOT NULL, kind text NOT NULL CHECK(kind IN ('ohlcv','price_observation')),
 PRIMARY KEY(fetch_id,day)
);
CREATE OR REPLACE VIEW crypto_market_latest AS
 SELECT DISTINCT ON (f.source_id,o.day) f.source_id,f.retrieved_at,o.*
 FROM crypto_market_observations o JOIN crypto_market_fetches f ON f.id=o.fetch_id
 ORDER BY f.source_id,o.day,f.retrieved_at DESC,f.id DESC;


-- Price linkage (2026-09): hourly candles for every tracked coin and an immutable event study.
-- Coin identity is governed by apps/web/lib/crypto-coins.json; the enumerated CHECK was dropped so a new coin needs no migration.
ALTER TABLE crypto_market_sources DROP CONSTRAINT IF EXISTS crypto_market_sources_coin_check;
CREATE TABLE IF NOT EXISTS crypto_market_hourly (
 fetch_id bigint NOT NULL REFERENCES crypto_market_fetches(id), hour timestamptz NOT NULL,
 sample_at timestamptz NOT NULL, close double precision NOT NULL CHECK(close>0),
 volume double precision CHECK(volume>=0),
 open double precision, high double precision, low double precision,
 complete boolean NOT NULL, kind text NOT NULL CHECK(kind IN ('ohlcv','price_observation')),
 PRIMARY KEY(fetch_id,hour)
);
CREATE OR REPLACE VIEW crypto_market_hourly_latest AS
 SELECT DISTINCT ON (f.source_id,o.hour) f.source_id,f.retrieved_at,o.*
 FROM crypto_market_hourly o JOIN crypto_market_fetches f ON f.id=o.fetch_id
 ORDER BY f.source_id,o.hour,f.retrieved_at DESC,f.id DESC;
-- One row per (post, coin, version). Rows are written once the 24-hour horizon has passed and are never rewritten.
CREATE TABLE IF NOT EXISTS crypto_price_events (
 post_id text NOT NULL REFERENCES crypto_social_posts(id), coin text NOT NULL, version text NOT NULL,
 account_id text NOT NULL, posted_at timestamptz NOT NULL, source_id text NOT NULL REFERENCES crypto_market_sources(id),
 hour timestamptz NOT NULL, episode boolean NOT NULL,
 price_before_1h double precision, price_0 double precision NOT NULL,
 price_after_1h double precision, price_after_6h double precision, price_after_24h double precision NOT NULL,
 volume_before_24h double precision, volume_after_24h double precision,
 hours_before_24h integer NOT NULL, hours_after_24h integer NOT NULL,
 computed_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(post_id,coin,version)
);
CREATE INDEX IF NOT EXISTS crypto_price_events_account ON crypto_price_events(coin,account_id,posted_at);

-- Volume correction (2026-09): a price_observation source reports a rolling 24h total, not the
-- interval's own trading, so its volume column is now NULL. Existing tables predate the nullable
-- column; both statements are catalog-only no-ops once applied.
ALTER TABLE crypto_market_observations ALTER COLUMN volume DROP NOT NULL;
ALTER TABLE crypto_market_hourly ALTER COLUMN volume DROP NOT NULL;

-- Clear rows written before that, then make the mistake unrepeatable. Guarding on the constraint's
-- own absence keeps this one-shot: afterwards the check is a catalog lookup, so the sweep path
-- never rescans the observation tables. The UPDATE must precede the constraint, because the rows
-- it corrects are exactly the rows the constraint forbids.
DO $$
DECLARE t text;
BEGIN
 FOREACH t IN ARRAY ARRAY['crypto_market_observations','crypto_market_hourly'] LOOP
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname=t||'_observation_volume') THEN
   EXECUTE format('UPDATE %I SET volume=NULL WHERE kind=''price_observation'' AND volume IS NOT NULL',t);
   EXECUTE format('ALTER TABLE %I ADD CONSTRAINT %I CHECK (kind<>''price_observation'' OR volume IS NULL)',
                  t,t||'_observation_volume');
  END IF;
 END LOOP;
END $$;

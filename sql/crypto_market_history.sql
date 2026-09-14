-- Immutable fetches and observations preserve earlier history and provider revisions.
CREATE TABLE IF NOT EXISTS crypto_market_sources (
 id text PRIMARY KEY, coin text NOT NULL CHECK(coin IN ('ZCAT','ZEC')),
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
 volume double precision NOT NULL CHECK(volume>=0),
 open double precision, high double precision, low double precision,
 complete boolean NOT NULL, kind text NOT NULL CHECK(kind IN ('ohlcv','price_observation')),
 PRIMARY KEY(fetch_id,day)
);
CREATE OR REPLACE VIEW crypto_market_latest AS
 SELECT DISTINCT ON (f.source_id,o.day) f.source_id,f.retrieved_at,o.*
 FROM crypto_market_observations o JOIN crypto_market_fetches f ON f.id=o.fetch_id
 ORDER BY f.source_id,o.day,f.retrieved_at DESC,f.id DESC;

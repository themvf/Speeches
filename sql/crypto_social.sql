-- Additive pilot schema. IDs remain text (X IDs exceed JavaScript safe integers).
CREATE TABLE IF NOT EXISTS crypto_social_pilot (
 id text PRIMARY KEY CHECK (id='zcat-zec-v1'),
 start_at timestamptz NOT NULL, end_at timestamptz NOT NULL,
 credit_limit integer NOT NULL DEFAULT 50000 CHECK (credit_limit BETWEEN 300 AND 50000),
 reserved_credits integer NOT NULL DEFAULT 0 CHECK (reserved_credits BETWEEN 0 AND 50000)
);
CREATE TABLE IF NOT EXISTS crypto_social_coins (
 symbol text PRIMARY KEY, name text NOT NULL, query text NOT NULL, address text,
 identity_status text NOT NULL
);
CREATE TABLE IF NOT EXISTS crypto_social_windows (
 id bigserial PRIMARY KEY, coin text NOT NULL REFERENCES crypto_social_coins(symbol),
 start_at timestamptz NOT NULL, end_at timestamptz NOT NULL, query text NOT NULL,
 cursor text NOT NULL DEFAULT '', pages integer NOT NULL DEFAULT 0,
 status text NOT NULL DEFAULT 'pending', UNIQUE(coin,start_at,end_at)
);
CREATE TABLE IF NOT EXISTS crypto_social_requests (
 id bigserial PRIMARY KEY, window_id bigint NOT NULL REFERENCES crypto_social_windows(id),
 requested_at timestamptz NOT NULL DEFAULT now(), status text NOT NULL DEFAULT 'reserved',
 reserved_credits integer NOT NULL DEFAULT 300, estimated_credits integer,
 returned_count integer, accepted_count integer, error text
);
CREATE TABLE IF NOT EXISTS crypto_social_accounts (
 id text PRIMARY KEY, handle text NOT NULL, name text NOT NULL DEFAULT '',
 followers bigint, observed_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS crypto_social_posts (
 id text PRIMARY KEY, author_id text NOT NULL REFERENCES crypto_social_accounts(id),
 text text NOT NULL, posted_at timestamptz NOT NULL, kind text NOT NULL,
 url text NOT NULL, first_seen_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS crypto_social_matches (
 post_id text NOT NULL REFERENCES crypto_social_posts(id),
 window_id bigint NOT NULL REFERENCES crypto_social_windows(id),
 PRIMARY KEY(post_id,window_id)
);
CREATE TABLE IF NOT EXISTS crypto_social_snapshots (
 post_id text NOT NULL REFERENCES crypto_social_posts(id),
 request_id bigint NOT NULL REFERENCES crypto_social_requests(id),
 observed_at timestamptz NOT NULL DEFAULT now(), likes bigint, replies bigint,
 quotes bigint, reposts bigint, views bigint,
 PRIMARY KEY(post_id,request_id)
);
CREATE TABLE IF NOT EXISTS crypto_social_edges (
 post_id text NOT NULL REFERENCES crypto_social_posts(id), source_id text NOT NULL,
 target_id text NOT NULL, kind text NOT NULL CHECK(kind IN ('reply','quote','repost','mention')),
 PRIMARY KEY(post_id,target_id,kind)
);
CREATE INDEX IF NOT EXISTS crypto_social_matches_window ON crypto_social_matches(window_id);
CREATE INDEX IF NOT EXISTS crypto_social_edges_target ON crypto_social_edges(target_id);

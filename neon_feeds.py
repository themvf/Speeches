"""
Neon (Postgres) helpers for managing rss_feeds, topic rules, and (Phase 1 of
an incremental migration off custom_documents.json) a write-only mirror of
ingested documents. Requires DATABASE_URL env var (Neon connection string).
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import psycopg2
import psycopg2.extras
import streamlit as st

DEFAULT_TOPIC_RULES = [
    {
        "topic_key": "SECURITIES_REGULATION",
        "label": "Securities Regulation",
        "keywords": "sec, securities, disclosure, investor, exchange, registration",
        "sort_order": 10,
    },
    {
        "topic_key": "CAPITAL_FORMATION",
        "label": "Capital Formation",
        "keywords": "ipo, spac, capital, offering, funding, venture, startup",
        "sort_order": 20,
    },
    {
        "topic_key": "AML",
        "label": "AML",
        "keywords": "aml, money laundering, sanctions, bsa, finra, anti-money",
        "sort_order": 30,
    },
    {
        "topic_key": "ENFORCEMENT",
        "label": "Enforcement",
        "keywords": "enforcement, fine, penalty, fraud, charges, lawsuit, settlement, indictment",
        "sort_order": 40,
    },
    {
        "topic_key": "AI_TECH",
        "label": "AI & Tech",
        "keywords": "ai, artificial intelligence, machine learning, technology, fintech, automation",
        "sort_order": 50,
    },
    {
        "topic_key": "CRYPTO",
        "label": "Crypto",
        "keywords": "crypto, bitcoin, blockchain, digital asset, stablecoin, ethereum, defi, nft",
        "sort_order": 60,
    },
    {
        "topic_key": "CREDIT_MARKETS",
        "label": "Credit Markets",
        "keywords": "credit, bond, debt, yield, loan, lending, mortgage, default",
        "sort_order": 70,
    },
    {
        "topic_key": "FINANCIAL_MARKETS",
        "label": "Financial Markets",
        "keywords": "market, stock, equity, trading, volatility, s&p, nasdaq, dow",
        "sort_order": 80,
    },
    {
        "topic_key": "ECONOMIC_GROWTH",
        "label": "Economic Growth",
        "keywords": "economy, gdp, growth, inflation, fed, federal reserve, recession, jobs",
        "sort_order": 90,
    },
]


def get_database_url() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if url:
        return url

    try:
        url = st.secrets.get("DATABASE_URL", "") or st.secrets.get("database_url", "")
        if url:
            return str(url)
    except Exception:
        pass

    try:
        neon_section = st.secrets.get("neon", None)
        if neon_section:
            url = neon_section.get("DATABASE_URL", "") or neon_section.get("database_url", "")
            if url:
                return str(url)
    except Exception:
        pass

    raise RuntimeError(
        "DATABASE_URL is not set. Provide it as an environment variable, "
        "a top-level Streamlit secret, or neon.DATABASE_URL in Streamlit secrets."
    )


def get_database_url_debug_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "env_has_DATABASE_URL": bool(os.environ.get("DATABASE_URL", "")),
        "top_level_has_DATABASE_URL": False,
        "top_level_has_database_url": False,
        "neon_has_DATABASE_URL": False,
        "neon_has_database_url": False,
        "secret_keys": [],
    }

    try:
        info["secret_keys"] = list(st.secrets.keys())
        info["top_level_has_DATABASE_URL"] = bool(st.secrets.get("DATABASE_URL", ""))
        info["top_level_has_database_url"] = bool(st.secrets.get("database_url", ""))
        neon_section = st.secrets.get("neon", None)
        if neon_section:
            info["neon_has_DATABASE_URL"] = bool(neon_section.get("DATABASE_URL", ""))
            info["neon_has_database_url"] = bool(neon_section.get("database_url", ""))
    except Exception as exc:
        info["secrets_error"] = str(exc)

    return info


def _get_conn():
    url = get_database_url()
    return psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)


def _ensure_topic_rule_schema() -> None:
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS rss_topic_rules (
                  id         SERIAL PRIMARY KEY,
                  topic_key  TEXT UNIQUE NOT NULL,
                  label      TEXT NOT NULL,
                  keywords   TEXT NOT NULL DEFAULT '',
                  active     BOOLEAN NOT NULL DEFAULT true,
                  sort_order INTEGER NOT NULL DEFAULT 100,
                  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            conn.commit()
            cur.execute("SELECT COUNT(*) AS n FROM rss_topic_rules")
            existing = int(cur.fetchone()["n"] or 0)
            if existing <= 0:
                for rule in DEFAULT_TOPIC_RULES:
                    cur.execute(
                        """
                        INSERT INTO rss_topic_rules (topic_key, label, keywords, active, sort_order)
                        VALUES (%s, %s, %s, true, %s)
                        ON CONFLICT (topic_key) DO NOTHING
                        """,
                        (
                            rule["topic_key"],
                            rule["label"],
                            rule["keywords"],
                            rule["sort_order"],
                        ),
                    )
                conn.commit()


def _derive_feed_key(feed_url: str) -> str:
    try:
        parsed = urlparse(feed_url)
        raw = (parsed.hostname or "") + (parsed.path or "")
    except Exception:
        raw = feed_url
    key = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    return key[:60]


def get_feeds(only_active: bool = False) -> List[Dict[str, Any]]:
    with _get_conn() as conn:
        with conn.cursor() as cur:
            if only_active:
                cur.execute("SELECT * FROM rss_feeds WHERE active = true ORDER BY added_at ASC")
            else:
                cur.execute("SELECT * FROM rss_feeds ORDER BY added_at ASC")
            return [dict(row) for row in cur.fetchall()]


def add_feed(label: str, feed_url: str) -> Dict[str, Any]:
    feed_key = _derive_feed_key(feed_url)
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rss_feeds (label, feed_url, feed_key)
                VALUES (%s, %s, %s)
                ON CONFLICT (feed_key) DO UPDATE SET
                    label = EXCLUDED.label,
                    feed_url = EXCLUDED.feed_url,
                    active = true
                RETURNING *
                """,
                (label.strip(), feed_url.strip(), feed_key),
            )
            conn.commit()
            return dict(cur.fetchone())


def toggle_feed(feed_id: int, active: bool) -> None:
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE rss_feeds SET active = %s WHERE id = %s", (active, feed_id))
            conn.commit()


def delete_feed(feed_id: int) -> None:
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM rss_feeds WHERE id = %s", (feed_id,))
            conn.commit()


def get_topic_rules(only_active: bool = False) -> List[Dict[str, Any]]:
    _ensure_topic_rule_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            if only_active:
                cur.execute(
                    "SELECT * FROM rss_topic_rules WHERE active = true ORDER BY sort_order ASC, label ASC"
                )
            else:
                cur.execute("SELECT * FROM rss_topic_rules ORDER BY sort_order ASC, label ASC")
            return [dict(row) for row in cur.fetchall()]


def upsert_topic_rule(
    topic_key: str,
    label: str,
    keywords: str,
    active: bool = True,
    sort_order: int = 100,
) -> Dict[str, Any]:
    _ensure_topic_rule_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rss_topic_rules (topic_key, label, keywords, active, sort_order, updated_at)
                VALUES (%s, %s, %s, %s, %s, now())
                ON CONFLICT (topic_key) DO UPDATE
                SET label = EXCLUDED.label,
                    keywords = EXCLUDED.keywords,
                    active = EXCLUDED.active,
                    sort_order = EXCLUDED.sort_order,
                    updated_at = now()
                RETURNING *
                """,
                (topic_key.strip(), label.strip(), keywords.strip(), active, int(sort_order)),
            )
            conn.commit()
            return dict(cur.fetchone())


def delete_topic_rule(rule_id: int) -> None:
    _ensure_topic_rule_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM rss_topic_rules WHERE id = %s", (rule_id,))
            conn.commit()


# ─── Document mirror (Phase 1 of migrating off custom_documents.json) ──────
#
# custom_documents.json is one JSON blob that ~15 scheduled workflows plus
# the web admin routes each download in full, mutate, and re-upload - the
# root cause behind several data-loss and contention bugs already fixed
# elsewhere in this codebase (see CLAUDE.md's Ingestion Pipeline Review).
# rss_articles already proves the alternative (per-row Neon storage) works.
#
# This is Phase 1 only: an additive `documents` table plus a best-effort,
# non-blocking mirror-write called from
# run_financial_news_pipeline.py's _upsert_custom_document_record. No reader
# has been cut over - the blob remains the sole source of truth for every
# existing reader. See CLAUDE.md for the remaining phases (backfill, then a
# one-reader-at-a-time cutover) before treating this table as authoritative.

_DOCUMENTS_SCHEMA_ENSURED = False


def _ensure_documents_schema() -> None:
    global _DOCUMENTS_SCHEMA_ENSURED
    if _DOCUMENTS_SCHEMA_ENSURED:
        return
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                  document_id    TEXT PRIMARY KEY,
                  title          TEXT NOT NULL DEFAULT '',
                  speaker        TEXT NOT NULL DEFAULT '',
                  organization   TEXT NOT NULL DEFAULT '',
                  doc_type       TEXT NOT NULL DEFAULT '',
                  source_kind    TEXT NOT NULL DEFAULT '',
                  url            TEXT NOT NULL DEFAULT '',
                  published_date TEXT NOT NULL DEFAULT '',
                  word_count     INTEGER NOT NULL DEFAULT 0,
                  full_text      TEXT NOT NULL DEFAULT '',
                  metadata       JSONB NOT NULL DEFAULT '{}'::jsonb,
                  updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS documents_source_kind ON documents (source_kind)")
            cur.execute("CREATE INDEX IF NOT EXISTS documents_updated_at ON documents (updated_at DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS documents_metadata_gin ON documents USING GIN (metadata)")
            conn.commit()
    _DOCUMENTS_SCHEMA_ENSURED = True


_DOCUMENTS_UPSERT_COLUMNS = (
    "document_id, title, speaker, organization, doc_type, "
    "source_kind, url, published_date, word_count, full_text, "
    "metadata, updated_at"
)

_DOCUMENTS_UPSERT_CONFLICT_CLAUSE = """
    ON CONFLICT (document_id) DO UPDATE SET
      title = EXCLUDED.title,
      speaker = EXCLUDED.speaker,
      organization = EXCLUDED.organization,
      doc_type = EXCLUDED.doc_type,
      source_kind = EXCLUDED.source_kind,
      url = EXCLUDED.url,
      published_date = EXCLUDED.published_date,
      word_count = EXCLUDED.word_count,
      full_text = EXCLUDED.full_text,
      metadata = EXCLUDED.metadata,
      updated_at = now()
"""


def _strip_nul_bytes(value: str) -> str:
    """Postgres text/JSONB columns reject embedded NUL (0x00) bytes outright
    (a libpq/wire-protocol limitation, not specific to this schema or a bug
    in our SQL) - some scraped/extracted documents contain one, usually a
    PDF or encoding-extraction artifact. Discovered in production: a single
    NUL byte anywhere in a 200-row execute_values batch fails the whole
    batch, not just that row. Stripping is safe - NUL isn't a meaningful
    printable character in any of these fields."""
    return value.replace("\x00", "") if isinstance(value, str) else value


def _sanitize_for_json(value: Any) -> Any:
    """Recursively strip NUL bytes from a metadata dict before it goes into
    a JSONB column - the raw scraper metadata can carry a NUL in any nested
    string field, not just the ones broken out into their own columns."""
    if isinstance(value, str):
        return _strip_nul_bytes(value)
    if isinstance(value, dict):
        return {key: _sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    return value


def _document_record_to_row(record: Dict[str, Any]) -> Optional[tuple]:
    """Extract one (document_id, ...) row tuple from a custom_documents.json-
    shaped record, or None if it has no document_id. Shared by the per-record
    mirror() write path and the bulk backfill path so the two never drift on
    column mapping."""
    metadata = record.get("metadata", {}) if isinstance(record.get("metadata", {}), dict) else {}
    content = record.get("content", {}) if isinstance(record.get("content", {}), dict) else {}
    document_id = _strip_nul_bytes(str(metadata.get("document_id", "") or "").strip())
    if not document_id:
        return None
    return (
        document_id,
        _strip_nul_bytes(str(metadata.get("title", "") or "")),
        _strip_nul_bytes(str(metadata.get("speaker", "") or "")),
        _strip_nul_bytes(str(metadata.get("organization", "") or "")),
        _strip_nul_bytes(str(metadata.get("doc_type", "") or "")),
        _strip_nul_bytes(str(metadata.get("source_kind", "") or "")),
        _strip_nul_bytes(str(metadata.get("url", "") or "")),
        _strip_nul_bytes(str(metadata.get("published_date", "") or metadata.get("date", "") or "")),
        int(metadata.get("word_count", 0) or 0),
        _strip_nul_bytes(str(content.get("full_text", "") or "")),
        psycopg2.extras.Json(_sanitize_for_json(metadata)),
    )


def mirror_document(record: Dict[str, Any]) -> None:
    """Best-effort upsert of one custom_documents.json-shaped record into the
    Neon `documents` table. Callers must treat this as fire-and-forget: any
    exception here (missing DATABASE_URL, connectivity, schema) should be
    caught by the caller and never block or fail the primary blob write."""
    row = _document_record_to_row(record)
    if row is None:
        return

    _ensure_documents_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO documents ({_DOCUMENTS_UPSERT_COLUMNS})
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                {_DOCUMENTS_UPSERT_CONFLICT_CLAUSE}
                """,
                row,
            )
            conn.commit()


def mirror_documents_batch(records: List[Dict[str, Any]]) -> int:
    """Upsert many custom_documents.json-shaped records into `documents` in
    one connection using a single multi-row statement, instead of opening a
    new connection per document like mirror_document() does. Intended for the
    one-time Phase 2 backfill (see CLAUDE.md) where mirror_document()'s
    per-call connection overhead would make backfilling tens of thousands of
    documents impractically slow.

    Unlike mirror_document(), this is NOT fire-and-forget - callers (the
    backfill script) should treat exceptions here as real failures worth
    surfacing, since there's no primary write path to silently fall back to.
    """
    rows = [row for row in (_document_record_to_row(record) for record in records) if row is not None]
    if not rows:
        return 0

    _ensure_documents_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                f"""
                INSERT INTO documents ({_DOCUMENTS_UPSERT_COLUMNS})
                VALUES %s
                {_DOCUMENTS_UPSERT_CONFLICT_CLAUSE}
                """,
                rows,
                template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())",
            )
            conn.commit()
    return len(rows)


def count_documents() -> int:
    """Row count of the Neon `documents` table - used by the backfill script
    to verify the mirror actually matches the source corpus size."""
    _ensure_documents_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS count FROM documents")
            row = cur.fetchone()
            return int(row["count"]) if row else 0


def get_document(document_id: str) -> Optional[Dict[str, Any]]:
    """Fetch one row from the Neon `documents` table by id - used by the
    backfill script's spot-check verification."""
    _ensure_documents_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM documents WHERE document_id = %s", (document_id,))
            return cur.fetchone()


# ─── Reddit attention sweep storage (docs/stock-attention-spec.md §3) ───────

_REDDIT_ATTENTION_SCHEMA_ENSURED = False


def _ensure_reddit_attention_schema() -> None:
    """Creates the tables the Reddit attention sweep writes. Python-owned,
    like `documents` above - deliberately NOT in neon.ts's ensureSchema(),
    since the only writer is reddit_attention_sweep.py.

    intelligence_mentions is normally created by the web app's ensureSchema()
    (neon.ts); the CREATE here is a byte-for-byte copy of that DDL so a fresh
    database can't leave the sweep racing the web app's first request. Kept
    in lockstep with neon.ts - if you change one, change the other.
    """
    global _REDDIT_ATTENTION_SCHEMA_ENSURED
    if _REDDIT_ATTENTION_SCHEMA_ENSURED:
        return
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS reddit_attention_items (
                  source_id   TEXT PRIMARY KEY,
                  kind        TEXT NOT NULL,
                  subreddit   TEXT NOT NULL,
                  author      TEXT NOT NULL,
                  title       TEXT NOT NULL DEFAULT '',
                  permalink   TEXT NOT NULL,
                  created_utc TIMESTAMPTZ NOT NULL,
                  score       INTEGER NOT NULL DEFAULT 0,
                  mood        TEXT NOT NULL DEFAULT 'neutral',
                  swept_at    TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS reddit_attention_items_created ON reddit_attention_items (created_utc)"
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS intelligence_mentions (
                  id               BIGSERIAL PRIMARY KEY,
                  source_type      TEXT NOT NULL,
                  source_id        TEXT NOT NULL,
                  mention_type     TEXT NOT NULL,
                  value            TEXT NOT NULL,
                  normalized_value TEXT NOT NULL,
                  confidence       DOUBLE PRECISION NOT NULL DEFAULT 1,
                  generated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
                  UNIQUE (source_type, source_id, mention_type, normalized_value)
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS intelligence_mentions_lookup ON intelligence_mentions (mention_type, normalized_value)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS intelligence_mentions_source ON intelligence_mentions (source_type, source_id)"
            )
            conn.commit()
    _REDDIT_ATTENTION_SCHEMA_ENSURED = True


_STOCK_ATTENTION_SCHEMA_ENSURED = False


def _ensure_stock_attention_schema() -> None:
    """daily_stock_attention rollup table (spec §3.3). Python-owned like the
    tables above; the web tier only reads it (and falls back naturally if it
    doesn't exist yet, same pattern as getAllMirroredDocumentMetadata)."""
    global _STOCK_ATTENTION_SCHEMA_ENSURED
    if _STOCK_ATTENTION_SCHEMA_ENSURED:
        return
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_stock_attention (
                  id              SERIAL PRIMARY KEY,
                  attention_date  DATE NOT NULL,
                  ticker          TEXT NOT NULL,
                  company         TEXT NOT NULL DEFAULT '',
                  mention_count   INTEGER NOT NULL DEFAULT 0,
                  source_count    INTEGER NOT NULL DEFAULT 0,
                  subreddit_count INTEGER NOT NULL DEFAULT 0,
                  weighted_score  NUMERIC NOT NULL DEFAULT 0,
                  mood            TEXT NOT NULL DEFAULT 'neutral',
                  top_source_ids  TEXT NOT NULL DEFAULT '[]',
                  generated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                  UNIQUE (attention_date, ticker)
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS daily_stock_attention_date_score ON daily_stock_attention (attention_date, weighted_score DESC)"
            )
            # Enhancement items 1-2 (docs/stock-attention-enhancements-spec.md):
            # per-channel counts and market context. Additive, idempotent -
            # ADD COLUMN IF NOT EXISTS is safe to re-run against the table
            # created above by an already-deployed v1.
            cur.execute("ALTER TABLE daily_stock_attention ADD COLUMN IF NOT EXISTS reddit_count INTEGER NOT NULL DEFAULT 0")
            cur.execute("ALTER TABLE daily_stock_attention ADD COLUMN IF NOT EXISTS news_count INTEGER NOT NULL DEFAULT 0")
            cur.execute("ALTER TABLE daily_stock_attention ADD COLUMN IF NOT EXISTS total_mention_count INTEGER NOT NULL DEFAULT 0")
            cur.execute("ALTER TABLE daily_stock_attention ADD COLUMN IF NOT EXISTS price_close NUMERIC")
            cur.execute("ALTER TABLE daily_stock_attention ADD COLUMN IF NOT EXISTS price_pct NUMERIC")
            cur.execute("ALTER TABLE daily_stock_attention ADD COLUMN IF NOT EXISTS volume BIGINT")
            cur.execute("ALTER TABLE daily_stock_attention ADD COLUMN IF NOT EXISTS volume_vs_20d NUMERIC")
            cur.execute("ALTER TABLE daily_stock_attention ADD COLUMN IF NOT EXISTS divergence TEXT NOT NULL DEFAULT ''")
            # Enhancement items 5-6: credibility-weighted counts and
            # data-quality flags (JSON array as TEXT, matching the
            # top_source_ids convention so the TS reader parses both the
            # same way).
            cur.execute("ALTER TABLE daily_stock_attention ADD COLUMN IF NOT EXISTS weighted_mention_count NUMERIC NOT NULL DEFAULT 0")
            cur.execute("ALTER TABLE daily_stock_attention ADD COLUMN IF NOT EXISTS quality_flags TEXT NOT NULL DEFAULT '[]'")
            # Item 5: per-author history rollup, recomputed daily from the
            # raw items window by aggregate_stock_attention.py. account_created/
            # link_karma are filled opportunistically by the sweep (capped
            # PRAW lookups) and must survive the daily recompute.
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS reddit_author_stats (
                  author              TEXT PRIMARY KEY,
                  first_seen          TIMESTAMPTZ,
                  last_seen           TIMESTAMPTZ,
                  items_total         INTEGER NOT NULL DEFAULT 0,
                  tickers_distinct    INTEGER NOT NULL DEFAULT 0,
                  subreddits_distinct INTEGER NOT NULL DEFAULT 0,
                  top_ticker_share    NUMERIC NOT NULL DEFAULT 0,
                  account_created     TIMESTAMPTZ,
                  link_karma          INTEGER,
                  refreshed_at        TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            # Activity/Authors views: which ticker a concentrated account is
            # concentrated on (the Authors leaderboard's most useful cell).
            cur.execute("ALTER TABLE reddit_author_stats ADD COLUMN IF NOT EXISTS top_ticker TEXT NOT NULL DEFAULT ''")
            # Item 6: review queue for tickers newly entering the top of the
            # board - populated by the rollup, worked through the admin UI.
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS attention_review_queue (
                  id                SERIAL PRIMARY KEY,
                  review_date       DATE NOT NULL,
                  ticker            TEXT NOT NULL,
                  status            TEXT NOT NULL DEFAULT 'pending',
                  sample_source_ids TEXT NOT NULL DEFAULT '[]',
                  created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
                  reviewed_at       TIMESTAMPTZ,
                  UNIQUE (review_date, ticker)
                )
                """
            )
            conn.commit()
    _STOCK_ATTENTION_SCHEMA_ENSURED = True


# ─── Attention sweep config (enhancement item 4) ────────────────────────────

# Written by the admin UI (TS side, apps/web/lib/server/neon.ts has the same
# CREATE - kept in lockstep) and read fail-soft by the sweep/rollup. A
# missing/unreachable config never blocks ingestion: callers fall back to
# their in-code defaults.
_ATTENTION_CONFIG_SCHEMA_ENSURED = False

DEFAULT_ATTENTION_SWEEP_CONFIG: Dict[str, Any] = {
    "subreddits": [
        {"name": "wallstreetbets", "tier": 1, "weight": 1.0, "active": True},
        {"name": "stocks", "tier": 1, "weight": 1.0, "active": True},
        {"name": "investing", "tier": 1, "weight": 1.0, "active": True},
        {"name": "StockMarket", "tier": 1, "weight": 1.0, "active": True},
        {"name": "options", "tier": 1, "weight": 1.0, "active": True},
        {"name": "Daytrading", "tier": 1, "weight": 1.0, "active": True},
        # Tier 2 ships active at sub-1.0 weight, per enhancement spec item 4
        # ("enable together with item 6's defenses, weighted < 1.0").
        {"name": "pennystocks", "tier": 2, "weight": 0.7, "active": True},
        {"name": "Shortsqueeze", "tier": 2, "weight": 0.7, "active": True},
        {"name": "SqueezePlays", "tier": 2, "weight": 0.7, "active": True},
        {"name": "smallstreetbets", "tier": 2, "weight": 0.7, "active": True},
        {"name": "ValueInvesting", "tier": 2, "weight": 0.9, "active": True},
        {"name": "dividends", "tier": 2, "weight": 0.9, "active": True},
    ],
    "bot_blocklist": ["automoderator", "visualmod", "wsbvotebot", "flairhelperbot"],
    "symbol_overrides": {"force_ambiguous": [], "force_unambiguous": []},
    "author_weighting": {"low_diversity_share": 0.8, "low_diversity_max_tickers": 2, "discount": 0.25},
}


def _ensure_attention_config_schema() -> None:
    global _ATTENTION_CONFIG_SCHEMA_ENSURED
    if _ATTENTION_CONFIG_SCHEMA_ENSURED:
        return
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS attention_sweep_config (
                  id         SERIAL PRIMARY KEY,
                  config     JSONB NOT NULL,
                  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute("SELECT COUNT(*) AS n FROM attention_sweep_config")
            row = cur.fetchone()
            if int(row["n"] if row else 0) == 0:
                cur.execute(
                    "INSERT INTO attention_sweep_config (config) VALUES (%s)",
                    (psycopg2.extras.Json(DEFAULT_ATTENTION_SWEEP_CONFIG),),
                )
            conn.commit()
    _ATTENTION_CONFIG_SCHEMA_ENSURED = True


def get_attention_sweep_config() -> Optional[Dict[str, Any]]:
    """Latest saved sweep config, or None on any failure (missing table,
    connectivity, malformed row) - callers must treat None as 'use in-code
    defaults', never as an error."""
    try:
        _ensure_attention_config_schema()
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT config FROM attention_sweep_config ORDER BY id DESC LIMIT 1")
                row = cur.fetchone()
                config = row["config"] if row else None
                return config if isinstance(config, dict) else None
    except Exception as exc:
        print(f"[neon_feeds] attention sweep config unavailable, using defaults: {exc}", flush=True)
        return None


# ─── Author stats writers (enhancement item 5) ──────────────────────────────

def upsert_author_stats_batch(rows: List[Dict[str, Any]]) -> int:
    """Daily recompute of per-author aggregates from the raw items window.
    account_created/link_karma are NOT touched here - those are owned by the
    sweep's opportunistic PRAW enrichment (upsert_author_account_info) and
    must survive this recompute."""
    prepared = []
    for row in rows:
        author = _strip_nul_bytes(str(row.get("author", "") or "").strip())
        if not author:
            continue
        prepared.append((
            author,
            row.get("first_seen"),
            row.get("last_seen"),
            int(row.get("items_total", 0) or 0),
            int(row.get("tickers_distinct", 0) or 0),
            int(row.get("subreddits_distinct", 0) or 0),
            float(row.get("top_ticker_share", 0.0) or 0.0),
            _strip_nul_bytes(str(row.get("top_ticker", "") or "")),
        ))
    if not prepared:
        return 0

    _ensure_stock_attention_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO reddit_author_stats
                  (author, first_seen, last_seen, items_total, tickers_distinct, subreddits_distinct, top_ticker_share, top_ticker)
                VALUES %s
                ON CONFLICT (author) DO UPDATE SET
                  first_seen = EXCLUDED.first_seen,
                  last_seen = EXCLUDED.last_seen,
                  items_total = EXCLUDED.items_total,
                  tickers_distinct = EXCLUDED.tickers_distinct,
                  subreddits_distinct = EXCLUDED.subreddits_distinct,
                  top_ticker_share = EXCLUDED.top_ticker_share,
                  top_ticker = EXCLUDED.top_ticker,
                  refreshed_at = now()
                """,
                prepared,
            )
            conn.commit()
    return len(prepared)


def upsert_author_account_info(rows: List[Dict[str, Any]]) -> int:
    """Sweep-side PRAW enrichment: account age/karma for authors we haven't
    looked up yet. Leaves the computed-stats columns alone."""
    prepared = []
    for row in rows:
        author = _strip_nul_bytes(str(row.get("author", "") or "").strip())
        if not author:
            continue
        prepared.append((author, row.get("account_created"), row.get("link_karma")))
    if not prepared:
        return 0

    _ensure_stock_attention_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO reddit_author_stats (author, account_created, link_karma)
                VALUES %s
                ON CONFLICT (author) DO UPDATE SET
                  account_created = EXCLUDED.account_created,
                  link_karma = EXCLUDED.link_karma
                """,
                prepared,
            )
            conn.commit()
    return len(prepared)


def get_authors_missing_account_info(authors: List[str]) -> List[str]:
    """Subset of `authors` with no account_created on record - the sweep's
    per-run PRAW lookup budget is spent on these."""
    candidates = [a for a in {str(a or "").strip() for a in authors} if a and a != "[deleted]"]
    if not candidates:
        return []
    _ensure_stock_attention_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT author FROM reddit_author_stats WHERE author = ANY(%s) AND account_created IS NOT NULL",
                (candidates,),
            )
            known = {row["author"] for row in cur.fetchall()}
    return sorted(a for a in candidates if a not in known)


def upsert_reddit_attention_items(items: List[Dict[str, Any]]) -> int:
    """Batch-upsert item metadata rows. Identity fields never change for a
    given source_id; only score (votes keep moving) and swept_at update on
    conflict."""
    rows = []
    for item in items:
        source_id = _strip_nul_bytes(str(item.get("source_id", "") or "").strip())
        if not source_id:
            continue
        rows.append((
            source_id,
            str(item.get("kind", "") or ""),
            _strip_nul_bytes(str(item.get("subreddit", "") or "")),
            _strip_nul_bytes(str(item.get("author", "") or "")),
            _strip_nul_bytes(str(item.get("title", "") or "")),
            _strip_nul_bytes(str(item.get("permalink", "") or "")),
            item.get("created_utc"),
            int(item.get("score", 0) or 0),
            str(item.get("mood", "neutral") or "neutral"),
        ))
    if not rows:
        return 0

    _ensure_reddit_attention_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO reddit_attention_items
                  (source_id, kind, subreddit, author, title, permalink, created_utc, score, mood)
                VALUES %s
                ON CONFLICT (source_id) DO UPDATE SET
                  score = EXCLUDED.score,
                  swept_at = now()
                """,
                rows,
            )
            conn.commit()
    return len(rows)


def insert_ticker_mentions(mentions: List[Dict[str, Any]]) -> int:
    """Batch-insert ticker mention rows into intelligence_mentions.

    ON CONFLICT DO NOTHING (not DO UPDATE): re-sweeping the same post must
    be a no-op, and unlike the web app's saveIntelligenceMentions (which
    rewrites a source's full mention set per analysis run), the sweep only
    ever discovers additively.
    """
    rows = []
    for mention in mentions:
        source_id = _strip_nul_bytes(str(mention.get("source_id", "") or "").strip())
        value = _strip_nul_bytes(str(mention.get("value", "") or "").strip())
        normalized = _strip_nul_bytes(str(mention.get("normalized_value", "") or "").strip())
        if not source_id or not value or not normalized:
            continue
        rows.append((
            str(mention.get("source_type", "") or ""),
            source_id,
            str(mention.get("mention_type", "ticker") or "ticker"),
            value,
            normalized,
            float(mention.get("confidence", 1.0) or 1.0),
        ))
    if not rows:
        return 0

    _ensure_reddit_attention_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO intelligence_mentions
                  (source_type, source_id, mention_type, value, normalized_value, confidence)
                VALUES %s
                ON CONFLICT (source_type, source_id, mention_type, normalized_value) DO NOTHING
                """,
                rows,
            )
            conn.commit()
    return len(rows)

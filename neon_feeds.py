"""
Neon (Postgres) helpers for managing rss_feeds, topic rules, and (Phase 1 of
an incremental migration off custom_documents.json) a write-only mirror of
ingested documents. Requires DATABASE_URL env var (Neon connection string).
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

import psycopg2
import psycopg2.extras
import streamlit as st

# Keep this byte-for-byte aligned with run_financial_news_pipeline.py; the
# row-level catch-up query must reject the same placeholder bodies as the
# enrichment candidate builder.
METADATA_FALLBACK_TEXT_MARKER = (
    "This metadata-backed record is retained so the item can appear in feed, search, watchlist, "
    "and briefing workflows."
)

import capital_formation_vocabulary

DEFAULT_TOPIC_RULES = [
    {
        "topic_key": "SECURITIES_REGULATION",
        "label": "Securities Regulation",
        "keywords": "sec, securities, disclosure, investor, exchange, registration",
        "sort_order": 10,
    },
    {
        # Vocabulary comes from apps/web/lib/capital-formation-vocabulary.json
        # via capital_formation_vocabulary.py -- the same file the TS side
        # reads. Both writers seed this table, so a forked copy here would
        # make the live rule depend on which side initialized the database.
        "topic_key": capital_formation_vocabulary.topic_key(),
        "label": capital_formation_vocabulary.label(),
        "keywords": capital_formation_vocabulary.keywords_csv(),
        "sort_order": capital_formation_vocabulary.sort_order(),
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
NEON_FULL_BACKFILL_CHECKPOINT = "legacy_gcs_documents_enrichments_v1"


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
            cur.execute("CREATE INDEX IF NOT EXISTS documents_url ON documents (url)")
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
    WHERE (
      documents.title,
      documents.speaker,
      documents.organization,
      documents.doc_type,
      documents.source_kind,
      documents.url,
      documents.published_date,
      documents.word_count,
      documents.full_text,
      documents.metadata
    ) IS DISTINCT FROM (
      EXCLUDED.title,
      EXCLUDED.speaker,
      EXCLUDED.organization,
      EXCLUDED.doc_type,
      EXCLUDED.source_kind,
      EXCLUDED.url,
      EXCLUDED.published_date,
      EXCLUDED.word_count,
      EXCLUDED.full_text,
      EXCLUDED.metadata
    )
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


def get_existing_document_ids(document_ids: List[str]) -> Set[str]:
    """Return the exact subset of requested IDs present in ``documents``."""
    clean_document_ids = list(
        dict.fromkeys(
            clean_id
            for clean_id in (
                _strip_nul_bytes(str(document_id or "").strip())
                for document_id in (document_ids or [])
            )
            if clean_id
        )
    )
    if not clean_document_ids:
        return set()
    _ensure_documents_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT document_id FROM documents WHERE document_id = ANY(%s)",
                (clean_document_ids,),
            )
            return {
                str(row.get("document_id", "") or "").strip()
                for row in cur.fetchall()
                if str(row.get("document_id", "") or "").strip()
            }


# ─── Document enrichment mirror ─────────────────────────────────────────
#
# document_enrichment_state.json stores every enrichment result in one large
# `entries` object.  This additive table mirrors that object one document at a
# time so ingestion jobs can read and update only the documents they are
# processing.  The complete legacy entry is retained in JSONB: no fields are
# discarded, and the blob can remain the rollback source of truth during the
# incremental cutover.

_ENRICHMENTS_SCHEMA_ENSURED = False


def _database_url_is_configured() -> bool:
    """Return False only for the expected optional-Neon configuration case.

    Enrichment mirroring is additive, so local runs and older deployments that
    intentionally have no DATABASE_URL must continue to work.  Connectivity
    and SQL errors are deliberately not swallowed here; callers should still
    see real database failures instead of silently losing a requested write.
    """
    try:
        get_database_url()
        return True
    except RuntimeError:
        return False


def _document_row_to_record(row: Dict[str, Any], include_full_text: bool = True) -> Dict[str, Any]:
    """Rebuild the legacy record shape from one `documents` table row.

    `metadata` is stored losslessly in the mirror.  The explicit columns are
    backfilled only when an older metadata payload omitted them.  The legacy
    content object only has a dedicated `full_text` column in this additive
    schema, so bounded readers intentionally expose just that field.
    """
    raw_metadata = row.get("metadata")
    metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
    column_to_metadata = {
        "document_id": "document_id",
        "title": "title",
        "speaker": "speaker",
        "organization": "organization",
        "doc_type": "doc_type",
        "source_kind": "source_kind",
        "url": "url",
        "published_date": "published_date",
        "word_count": "word_count",
    }
    for column, metadata_key in column_to_metadata.items():
        value = row.get(column)
        if metadata_key not in metadata and value not in (None, ""):
            metadata[metadata_key] = value
    # Some legacy readers use `date`, while the relational column is named
    # `published_date`; preserve both aliases when reconstructing a record.
    if "date" not in metadata and row.get("published_date") not in (None, ""):
        metadata["date"] = row.get("published_date")
    content: Dict[str, Any] = {}
    if include_full_text:
        content["full_text"] = str(row.get("full_text", "") or "")
    return {"metadata": metadata, "content": content}


def get_documents(document_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch explicit document IDs as legacy-shaped records in one query."""
    clean_document_ids = list(
        dict.fromkeys(
            clean_id
            for clean_id in (
                _strip_nul_bytes(str(document_id or "").strip())
                for document_id in (document_ids or [])
            )
            if clean_id
        )
    )
    if not clean_document_ids or not _database_url_is_configured():
        return {}
    _ensure_documents_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM documents WHERE document_id = ANY(%s)",
                (clean_document_ids,),
            )
            result: Dict[str, Dict[str, Any]] = {}
            for row in cur.fetchall():
                document_id = str(row.get("document_id", "") or "")
                if document_id:
                    result[document_id] = _document_row_to_record(row)
            return result


def get_documents_by_urls(
    urls: List[str],
    include_full_text: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Fetch exact URLs, metadata-only by default, with the newest row kept."""
    clean_urls = list(
        dict.fromkeys(
            clean_url
            for clean_url in (
                _strip_nul_bytes(str(url or "").strip()) for url in (urls or [])
            )
            if clean_url
        )
    )
    if not clean_urls or not _database_url_is_configured():
        return {}
    selected_columns = (
        "document_id, title, speaker, organization, doc_type, source_kind, "
        "url, published_date, word_count, metadata, updated_at"
    )
    if include_full_text:
        selected_columns += ", full_text"
    _ensure_documents_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT {selected_columns}
                FROM documents
                WHERE url = ANY(%s)
                ORDER BY updated_at DESC
                """.format(selected_columns=selected_columns),
                (clean_urls,),
            )
            result: Dict[str, Dict[str, Any]] = {}
            for row in cur.fetchall():
                url = str(row.get("url", "") or "")
                if url and url not in result:
                    result[url] = _document_row_to_record(
                        row, include_full_text=include_full_text
                    )
            return result


def get_document_records_by_source_kinds(
    source_kinds: List[str],
    include_full_text: bool = False,
) -> List[Dict[str, Any]]:
    """Fetch records for explicit source kinds, omitting document text by default.

    Discovery/classification only needs metadata, and selecting `full_text`
    accidentally would recreate much of the blob egress inside Neon.  Callers
    must opt in when they are about to extract/enrich a bounded changed set.
    """
    clean_source_kinds = list(
        dict.fromkeys(
            clean_kind
            for clean_kind in (
                _strip_nul_bytes(str(source_kind or "").strip())
                for source_kind in (source_kinds or [])
            )
            if clean_kind
        )
    )
    if not clean_source_kinds or not _database_url_is_configured():
        return []
    selected_columns = (
        "document_id, title, speaker, organization, doc_type, source_kind, "
        "url, published_date, word_count, metadata"
    )
    if include_full_text:
        selected_columns += ", full_text"
    _ensure_documents_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {selected_columns}
                FROM documents
                WHERE source_kind = ANY(%s)
                """,
                (clean_source_kinds,),
            )
            return [
                _document_row_to_record(row, include_full_text=include_full_text)
                for row in cur.fetchall()
            ]


def _ensure_enrichments_schema() -> None:
    global _ENRICHMENTS_SCHEMA_ENSURED
    if _ENRICHMENTS_SCHEMA_ENSURED:
        return
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS document_enrichments (
                  document_id     TEXT PRIMARY KEY,
                  status          TEXT NOT NULL DEFAULT '',
                  pipeline_version TEXT NOT NULL DEFAULT '',
                  entry_updated_at TEXT NOT NULL DEFAULT '',
                  entry           JSONB NOT NULL DEFAULT '{}'::jsonb,
                  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS document_enrichments_status "
                "ON document_enrichments (status)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS document_enrichments_updated_at "
                "ON document_enrichments (updated_at DESC)"
            )
            conn.commit()
    _ENRICHMENTS_SCHEMA_ENSURED = True


def _enrichment_entry_to_row(document_id: str, entry: Any) -> Optional[tuple]:
    """Map one legacy `entries[document_id]` value to a Neon row."""
    clean_document_id = _strip_nul_bytes(str(document_id or "").strip())
    if not clean_document_id or not isinstance(entry, dict):
        return None
    clean_entry = _sanitize_for_json(entry)
    return (
        clean_document_id,
        _strip_nul_bytes(str(clean_entry.get("status", "") or "")),
        _strip_nul_bytes(str(clean_entry.get("pipeline_version", "") or "")),
        _strip_nul_bytes(str(clean_entry.get("updated_at", "") or "")),
        psycopg2.extras.Json(clean_entry),
    )


_ENRICHMENTS_UPSERT_CONFLICT_CLAUSE = """
    ON CONFLICT (document_id) DO UPDATE SET
      status = EXCLUDED.status,
      pipeline_version = EXCLUDED.pipeline_version,
      entry_updated_at = EXCLUDED.entry_updated_at,
      entry = EXCLUDED.entry,
      updated_at = now()
    WHERE (
      document_enrichments.status,
      document_enrichments.pipeline_version,
      document_enrichments.entry_updated_at,
      document_enrichments.entry
    ) IS DISTINCT FROM (
      EXCLUDED.status,
      EXCLUDED.pipeline_version,
      EXCLUDED.entry_updated_at,
      EXCLUDED.entry
    )
"""


def upsert_enrichment_entries(entries: Dict[str, Dict[str, Any]]) -> int:
    """Additively upsert legacy enrichment entries in one database round trip.

    Identical rows are ignored by Postgres, including their `updated_at`, so a
    retry or checkpoint does not create avoidable writes.  The return value is
    the number of valid rows submitted (not the number Postgres found changed).
    Missing DATABASE_URL is an intentional no-op and returns zero.
    """
    if not isinstance(entries, dict) or not entries:
        return 0
    rows = [
        row
        for row in (
            _enrichment_entry_to_row(document_id, entry)
            for document_id, entry in entries.items()
        )
        if row is not None
    ]
    if not rows or not _database_url_is_configured():
        return 0

    _ensure_enrichments_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO document_enrichments (
                  document_id, status, pipeline_version, entry_updated_at,
                  entry, updated_at
                )
                VALUES %s
                """
                + _ENRICHMENTS_UPSERT_CONFLICT_CLAUSE,
                rows,
                template="(%s, %s, %s, %s, %s, now())",
            )
            conn.commit()
    return len(rows)


def get_enrichment_entry(document_id: str) -> Optional[Dict[str, Any]]:
    """Read one legacy-shaped enrichment entry, or None when unavailable."""
    clean_document_id = _strip_nul_bytes(str(document_id or "").strip())
    if not clean_document_id or not _database_url_is_configured():
        return None
    _ensure_enrichments_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT entry FROM document_enrichments WHERE document_id = %s",
                (clean_document_id,),
            )
            row = cur.fetchone()
            entry = row.get("entry") if row else None
            return dict(entry) if isinstance(entry, dict) else None


def get_enrichment_entries(document_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Read a bounded set of legacy-shaped entries keyed by document id.

    Requiring explicit IDs prevents this helper from recreating the full-state
    download pattern it replaces.  Invalid/duplicate IDs are removed before
    querying.  Missing DATABASE_URL and an empty ID list both return `{}`.
    """
    clean_document_ids = list(
        dict.fromkeys(
            clean_id
            for clean_id in (
                _strip_nul_bytes(str(document_id or "").strip())
                for document_id in (document_ids or [])
            )
            if clean_id
        )
    )
    if not clean_document_ids or not _database_url_is_configured():
        return {}
    _ensure_enrichments_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT document_id, entry
                FROM document_enrichments
                WHERE document_id = ANY(%s)
                """,
                (clean_document_ids,),
            )
            result: Dict[str, Dict[str, Any]] = {}
            for row in cur.fetchall():
                document_id = str(row.get("document_id", "") or "")
                entry = row.get("entry")
                if document_id and isinstance(entry, dict):
                    result[document_id] = dict(entry)
            return result


def count_enrichment_entries() -> int:
    """Return the mirror row count for backfill/coverage verification.

    Unlike optional ingestion reads, verification must surface a missing or
    unreachable database so it cannot report a misleading successful count.
    """
    _ensure_enrichments_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS count FROM document_enrichments")
            row = cur.fetchone()
            return int(row["count"]) if row else 0


def get_existing_enrichment_ids(document_ids: List[str]) -> Set[str]:
    """Return the exact subset of requested IDs present in enrichments."""
    clean_document_ids = list(
        dict.fromkeys(
            clean_id
            for clean_id in (
                _strip_nul_bytes(str(document_id or "").strip())
                for document_id in (document_ids or [])
            )
            if clean_id
        )
    )
    if not clean_document_ids:
        return set()
    _ensure_enrichments_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT document_id FROM document_enrichments WHERE document_id = ANY(%s)",
                (clean_document_ids,),
            )
            return {
                str(row.get("document_id", "") or "").strip()
                for row in cur.fetchall()
                if str(row.get("document_id", "") or "").strip()
            }


def get_document_ids_needing_enrichment(
    source_kinds: List[str],
    limit: int = 10,
) -> List[str]:
    """Return a bounded pilot catch-up queue without scanning blob state.

    A document is eligible when enrichment is missing, retryable, or older
    than the document row. The latter recovers a document commit followed by
    a failed targeted enrichment write. Persistent fallbacks stop at the same
    three-attempt cap enforced by the pipeline.
    """
    clean_source_kinds = list(
        dict.fromkeys(
            clean_kind
            for clean_kind in (
                _strip_nul_bytes(str(source_kind or "").strip())
                for source_kind in (source_kinds or [])
            )
            if clean_kind
        )
    )
    bounded_limit = max(0, min(int(limit), 100))
    if not clean_source_kinds or bounded_limit <= 0:
        return []
    _ensure_documents_schema()
    _ensure_enrichments_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT documents.document_id
                FROM documents
                LEFT JOIN document_enrichments enrichment
                  ON enrichment.document_id = documents.document_id
                WHERE documents.source_kind = ANY(%s)
                  AND documents.full_text <> ''
                  AND COALESCE(documents.metadata->>'extraction_mode', '') <> 'metadata_fallback'
                  AND position(%s in documents.full_text) = 0
                  AND (
                    enrichment.document_id IS NULL
                    OR documents.updated_at > enrichment.updated_at
                    OR (
                      lower(COALESCE(enrichment.entry->>'status', ''))
                        NOT IN ('enriched', 'reviewed')
                      AND CASE
                        WHEN COALESCE(enrichment.entry->>'attempt_count', '') ~ '^[0-9]+$'
                          THEN (enrichment.entry->>'attempt_count')::integer
                        ELSE 0
                      END < 3
                    )
                  )
                ORDER BY
                  CASE
                    WHEN enrichment.document_id IS NULL THEN 0
                    WHEN documents.updated_at > enrichment.updated_at THEN 1
                    ELSE 2
                  END,
                  documents.updated_at DESC,
                  documents.document_id
                LIMIT %s
                """,
                (clean_source_kinds, METADATA_FALLBACK_TEXT_MARKER, bounded_limit),
            )
            return [
                str(row.get("document_id", "") or "").strip()
                for row in cur.fetchall()
                if str(row.get("document_id", "") or "").strip()
            ]


_MIGRATION_CHECKPOINT_SCHEMA_ENSURED = False


def _ensure_migration_checkpoint_schema() -> None:
    global _MIGRATION_CHECKPOINT_SCHEMA_ENSURED
    if _MIGRATION_CHECKPOINT_SCHEMA_ENSURED:
        return
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS persistence_migration_checkpoints (
                  checkpoint_key TEXT PRIMARY KEY,
                  status         TEXT NOT NULL,
                  details        JSONB NOT NULL DEFAULT '{}'::jsonb,
                  verified_at    TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            conn.commit()
    _MIGRATION_CHECKPOINT_SCHEMA_ENSURED = True


def set_migration_checkpoint(
    checkpoint_key: str,
    status: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    clean_key = _strip_nul_bytes(str(checkpoint_key or "").strip())
    clean_status = _strip_nul_bytes(str(status or "").strip())
    if not clean_key or not clean_status:
        raise ValueError("checkpoint_key and status are required")
    _ensure_migration_checkpoint_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO persistence_migration_checkpoints (
                  checkpoint_key, status, details, verified_at
                )
                VALUES (%s, %s, %s, now())
                ON CONFLICT (checkpoint_key) DO UPDATE SET
                  status = EXCLUDED.status,
                  details = EXCLUDED.details,
                  verified_at = now()
                """,
                (
                    clean_key,
                    clean_status,
                    psycopg2.extras.Json(_sanitize_for_json(details or {})),
                ),
            )
            conn.commit()


def get_migration_checkpoint(checkpoint_key: str) -> Optional[Dict[str, Any]]:
    clean_key = _strip_nul_bytes(str(checkpoint_key or "").strip())
    if not clean_key:
        return None
    _ensure_migration_checkpoint_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT checkpoint_key, status, details, verified_at
                FROM persistence_migration_checkpoints
                WHERE checkpoint_key = %s
                """,
                (clean_key,),
            )
            row = cur.fetchone()
            return dict(row) if row else None


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
            # Enhancement 1: total upvotes across the deduped threads/comments
            # mentioning a ticker that day. The sweep has always stored a
            # per-item score, but it only ever chose which permalinks the
            # drawer showed; it fed nothing into ranking. Stored as its own
            # column (not folded silently into weighted_score) so the
            # amplifier is auditable and a re-run can be compared against the
            # unweighted history.
            cur.execute("ALTER TABLE daily_stock_attention ADD COLUMN IF NOT EXISTS engagement_score BIGINT NOT NULL DEFAULT 0")
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
            # SEC-4: rss_articles ids behind the news_count, so the drawer
            # can link the actual articles (JSON array as TEXT, same
            # convention as top_source_ids).
            cur.execute("ALTER TABLE daily_stock_attention ADD COLUMN IF NOT EXISTS top_news_ids TEXT NOT NULL DEFAULT '[]'")
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
            # Authors leaderboard: the account's top-3 tickers and top-3
            # subreddits as JSON arrays ([{"ticker","count"}] / [{"subreddit",
            # "count"}]), so the board can name them instead of only counting.
            cur.execute("ALTER TABLE reddit_author_stats ADD COLUMN IF NOT EXISTS top_tickers TEXT NOT NULL DEFAULT '[]'")
            cur.execute("ALTER TABLE reddit_author_stats ADD COLUMN IF NOT EXISTS top_subreddits TEXT NOT NULL DEFAULT '[]'")
            # (The former item-6 attention_review_queue was removed - newcomer
            # tickers now go straight to the board; quality_flags still annotate.)
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


# ─── News connector settings (single-row config, GCS blob retirement) ──────

# Written by the admin UI (TS side, apps/web/lib/server/neon.ts has the same
# CREATE - kept in lockstep) and read fail-soft by run_financial_news_pipeline.
# py's CLI-override merge. Same single-JSONB-row shape and same "None means
# use in-code defaults, never an error" contract as attention_sweep_config
# above; this table replaces news_connector_settings.json rather than sitting
# alongside it, since the blob had exactly one writer (this admin route) and
# one Python reader.
_NEWS_CONNECTOR_SETTINGS_SCHEMA_ENSURED = False


def _ensure_news_connector_settings_schema() -> None:
    global _NEWS_CONNECTOR_SETTINGS_SCHEMA_ENSURED
    if _NEWS_CONNECTOR_SETTINGS_SCHEMA_ENSURED:
        return
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS news_connector_settings (
                  id         SERIAL PRIMARY KEY,
                  config     JSONB NOT NULL,
                  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            conn.commit()
    _NEWS_CONNECTOR_SETTINGS_SCHEMA_ENSURED = True


def get_news_connector_settings() -> Optional[Dict[str, Any]]:
    """Latest saved settings row, or None on any failure (missing table,
    connectivity, no row saved yet, malformed row) - callers must treat None
    as 'use in-code defaults', never as an error."""
    try:
        _ensure_news_connector_settings_schema()
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT config FROM news_connector_settings ORDER BY id DESC LIMIT 1")
                row = cur.fetchone()
                config = row["config"] if row else None
                return config if isinstance(config, dict) else None
    except Exception as exc:
        print(f"[neon_feeds] news connector settings unavailable, using defaults: {exc}", flush=True)
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
            _strip_nul_bytes(str(row.get("top_tickers", "[]") or "[]")),
            _strip_nul_bytes(str(row.get("top_subreddits", "[]") or "[]")),
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
                  (author, first_seen, last_seen, items_total, tickers_distinct, subreddits_distinct, top_ticker_share, top_ticker, top_tickers, top_subreddits)
                VALUES %s
                ON CONFLICT (author) DO UPDATE SET
                  first_seen = EXCLUDED.first_seen,
                  last_seen = EXCLUDED.last_seen,
                  items_total = EXCLUDED.items_total,
                  tickers_distinct = EXCLUDED.tickers_distinct,
                  subreddits_distinct = EXCLUDED.subreddits_distinct,
                  top_ticker_share = EXCLUDED.top_ticker_share,
                  top_ticker = EXCLUDED.top_ticker,
                  top_tickers = EXCLUDED.top_tickers,
                  top_subreddits = EXCLUDED.top_subreddits,
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


def get_authors_needing_account_info(
    recent_authors: List[str], board_budget: int, recent_budget: int
) -> List[str]:
    """Chooses which authors the sweep spends its capped PRAW account-info
    lookups on, split so both consumers of account age get coverage:

    - `board_budget` goes to the top-by-items_total authors still missing
      account_created. These ARE the visible Authors-leaderboard rows, which
      an alphabetical/current-sweep-only selection left permanently blank
      (the board ranks by 90-day item count, so its top rows are rarely the
      authors active in any one sweep).
    - `recent_budget` goes to authors active in the current sweep still
      missing it. Fresh/young accounts never accumulate a high items_total,
      so the board pass alone would never reach them - but they're exactly
      what item 6's young_account_concentration flag needs ages for.

    Board authors first, de-duplicated, so a caller iterating the result
    naturally fills the leaderboard before the young-account reserve.
    """
    _ensure_stock_attention_schema()
    recent = [a for a in {str(a or "").strip() for a in recent_authors} if a and a != "[deleted]"]
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT author FROM reddit_author_stats
                WHERE account_created IS NULL AND author <> '[deleted]'
                ORDER BY items_total DESC, author ASC
                LIMIT %s
                """,
                (max(0, board_budget),),
            )
            board = [row["author"] for row in cur.fetchall()]
            known: set = set()
            if recent:
                cur.execute(
                    "SELECT author FROM reddit_author_stats WHERE author = ANY(%s) AND account_created IS NOT NULL",
                    (recent,),
                )
                known = {row["author"] for row in cur.fetchall()}
    selected = list(board)
    seen = set(board)
    recent_added = 0
    for author in recent:
        if recent_added >= recent_budget:
            break
        if author not in seen and author not in known:
            selected.append(author)
            seen.add(author)
            recent_added += 1
    return selected


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


# ─── Polymarket earnings tracker (SEC-26/27) ─────────────────────────────────
#
# Two-tier retention (SEC-26 success criteria): polymarket_trades is
# EPHEMERAL (fills deletable only after their market is settled - keyed on
# settlement, never fill age, so long-lived open markets keep their full tape
# until resolution); polymarket_markets / polymarket_wallet_market_results /
# polymarket_wallet_stats are DURABLE and accumulate across quarters. Wallet
# stats are always recomputed from the durable results table, never from raw
# fills, so pruning is invisible to the tracked-trader product.

_POLYMARKET_SCHEMA_ENSURED = False


def _ensure_polymarket_schema() -> None:
    global _POLYMARKET_SCHEMA_ENSURED
    if _POLYMARKET_SCHEMA_ENSURED:
        return
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS polymarket_markets (
                  condition_id     TEXT PRIMARY KEY,
                  ticker           TEXT NOT NULL DEFAULT '',
                  question         TEXT NOT NULL DEFAULT '',
                  slug             TEXT NOT NULL DEFAULT '',
                  eps              TEXT,
                  report_date      DATE,
                  end_date         TIMESTAMPTZ,
                  volume           NUMERIC NOT NULL DEFAULT 0,
                  implied_prob_yes NUMERIC,
                  status           TEXT NOT NULL DEFAULT 'open',
                  winner           TEXT,
                  settled_at       TIMESTAMPTZ,
                  fills_pruned_at  TIMESTAMPTZ,
                  first_seen       TIMESTAMPTZ NOT NULL DEFAULT now(),
                  updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute("ALTER TABLE polymarket_markets ADD COLUMN IF NOT EXISTS market_type TEXT NOT NULL DEFAULT 'earnings'")
            cur.execute("ALTER TABLE polymarket_markets ADD COLUMN IF NOT EXISTS cohort TEXT")
            cur.execute("ALTER TABLE polymarket_markets ADD COLUMN IF NOT EXISTS event_key TEXT")
            cur.execute("ALTER TABLE polymarket_markets ADD COLUMN IF NOT EXISTS event_title TEXT")
            cur.execute("ALTER TABLE polymarket_markets ADD COLUMN IF NOT EXISTS release_at TIMESTAMPTZ")
            cur.execute("CREATE INDEX IF NOT EXISTS polymarket_markets_type_event ON polymarket_markets (market_type, event_key)")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS polymarket_trades (
                  fill_key     TEXT PRIMARY KEY,
                  condition_id TEXT NOT NULL,
                  wallet       TEXT NOT NULL,
                  name         TEXT NOT NULL DEFAULT '',
                  outcome      TEXT NOT NULL,
                  side         TEXT NOT NULL,
                  size         NUMERIC NOT NULL,
                  price        NUMERIC NOT NULL,
                  filled_at    TIMESTAMPTZ NOT NULL
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS polymarket_trades_market ON polymarket_trades (condition_id, filled_at)")
            cur.execute("CREATE INDEX IF NOT EXISTS polymarket_trades_wallet ON polymarket_trades (wallet)")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS polymarket_wallet_market_results (
                  condition_id  TEXT NOT NULL,
                  wallet        TEXT NOT NULL,
                  name          TEXT NOT NULL DEFAULT '',
                  ticker        TEXT NOT NULL DEFAULT '',
                  resolved_date DATE,
                  pnl           NUMERIC NOT NULL DEFAULT 0,
                  cost          NUMERIC NOT NULL DEFAULT 0,
                  win_entry_avg NUMERIC,
                  correct       BOOLEAN NOT NULL DEFAULT false,
                  PRIMARY KEY (condition_id, wallet)
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS polymarket_results_wallet ON polymarket_wallet_market_results (wallet)")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS polymarket_wallet_stats (
                  wallet        TEXT PRIMARY KEY,
                  name          TEXT NOT NULL DEFAULT '',
                  markets       INTEGER NOT NULL DEFAULT 0,
                  wins          INTEGER NOT NULL DEFAULT 0,
                  pnl           NUMERIC NOT NULL DEFAULT 0,
                  cost          NUMERIC NOT NULL DEFAULT 0,
                  win_entry_avg NUMERIC,
                  archetype     TEXT NOT NULL DEFAULT 'unclassified',
                  refreshed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS polymarket_sharp_alerts (
                  fill_key     TEXT PRIMARY KEY,
                  condition_id TEXT NOT NULL,
                  ticker       TEXT NOT NULL DEFAULT '',
                  wallet       TEXT NOT NULL,
                  name         TEXT NOT NULL DEFAULT '',
                  archetype    TEXT NOT NULL,
                  side         TEXT NOT NULL,
                  outcome      TEXT NOT NULL DEFAULT '',
                  size         NUMERIC NOT NULL,
                  price        NUMERIC NOT NULL,
                  filled_at    TIMESTAMPTZ NOT NULL,
                  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS polymarket_sharp_alerts_ticker ON polymarket_sharp_alerts (ticker, filled_at DESC)")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS polymarket_macro_wallet_event_results (
                  event_key        TEXT NOT NULL,
                  wallet           TEXT NOT NULL,
                  name             TEXT NOT NULL DEFAULT '',
                  cohort           TEXT NOT NULL,
                  resolved_date    DATE,
                  pnl              NUMERIC NOT NULL DEFAULT 0,
                  cost             NUMERIC NOT NULL DEFAULT 0,
                  early_cost       NUMERIC NOT NULL DEFAULT 0,
                  pre_release_cost NUMERIC NOT NULL DEFAULT 0,
                  late_cost        NUMERIC NOT NULL DEFAULT 0,
                  post_release_cost NUMERIC NOT NULL DEFAULT 0,
                  win_entry_avg    NUMERIC,
                  correct          BOOLEAN NOT NULL DEFAULT false,
                  PRIMARY KEY (event_key, wallet)
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS polymarket_macro_results_wallet ON polymarket_macro_wallet_event_results (wallet, cohort)")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS polymarket_macro_wallet_stats (
                  wallet            TEXT NOT NULL,
                  cohort            TEXT NOT NULL,
                  name              TEXT NOT NULL DEFAULT '',
                  events            INTEGER NOT NULL DEFAULT 0,
                  wins              INTEGER NOT NULL DEFAULT 0,
                  pnl               NUMERIC NOT NULL DEFAULT 0,
                  cost              NUMERIC NOT NULL DEFAULT 0,
                  predictive_cost   NUMERIC NOT NULL DEFAULT 0,
                  timing_cost       NUMERIC NOT NULL DEFAULT 0,
                  win_entry_avg     NUMERIC,
                  archetype         TEXT NOT NULL DEFAULT 'unclassified',
                  refreshed_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                  PRIMARY KEY (wallet, cohort)
                )
                """
            )
            conn.commit()
    _POLYMARKET_SCHEMA_ENSURED = True


def upsert_polymarket_markets(rows: List[Dict[str, Any]]) -> int:
    """Metadata refresh for tracked markets. Never downgrades a resolved
    market back to open, and never touches settlement bookkeeping."""
    prepared = []
    for row in rows:
        condition_id = _strip_nul_bytes(str(row.get("condition_id", "") or "").strip())
        if not condition_id:
            continue
        prepared.append((
            condition_id,
            _strip_nul_bytes(str(row.get("ticker", "") or "")),
            _strip_nul_bytes(str(row.get("question", "") or "")),
            _strip_nul_bytes(str(row.get("slug", "") or "")),
            row.get("eps"),
            row.get("report_date"),
            row.get("end_date") or None,
            float(row.get("volume", 0) or 0),
            row.get("implied_prob_yes"),
            _strip_nul_bytes(str(row.get("market_type", "earnings") or "earnings")),
            _strip_nul_bytes(str(row.get("cohort", "") or "")) or None,
            _strip_nul_bytes(str(row.get("event_key", "") or "")) or None,
            _strip_nul_bytes(str(row.get("event_title", "") or "")) or None,
            row.get("release_at") or None,
        ))
    if not prepared:
        return 0
    _ensure_polymarket_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO polymarket_markets
                  (condition_id, ticker, question, slug, eps, report_date, end_date, volume, implied_prob_yes,
                   market_type, cohort, event_key, event_title, release_at)
                VALUES %s
                ON CONFLICT (condition_id) DO UPDATE SET
                  ticker = EXCLUDED.ticker,
                  question = EXCLUDED.question,
                  slug = EXCLUDED.slug,
                  eps = EXCLUDED.eps,
                  report_date = EXCLUDED.report_date,
                  end_date = EXCLUDED.end_date,
                  volume = EXCLUDED.volume,
                  implied_prob_yes = EXCLUDED.implied_prob_yes,
                  market_type = EXCLUDED.market_type,
                  cohort = EXCLUDED.cohort,
                  event_key = EXCLUDED.event_key,
                  event_title = EXCLUDED.event_title,
                  release_at = EXCLUDED.release_at,
                  updated_at = now()
                """,
                prepared,
            )
            conn.commit()
    return len(prepared)


def mark_polymarket_resolved(condition_id: str, winner: str) -> None:
    _ensure_polymarket_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE polymarket_markets SET status = 'resolved', winner = %s, updated_at = now() WHERE condition_id = %s AND status != 'resolved'",
                (winner, condition_id),
            )
            conn.commit()


def get_polymarket_tracked_markets(market_type: str = "earnings") -> List[Dict[str, Any]]:
    """Every tracked market with its lifecycle fields, plus the per-market
    fill cursor (max stored fill timestamp) in one query."""
    _ensure_polymarket_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT m.condition_id, m.ticker, m.question, m.slug, m.status, m.winner,
                       m.settled_at, m.fills_pruned_at, m.end_date, m.cohort,
                       m.event_key, m.event_title, m.release_at,
                       (SELECT MAX(t.filled_at) FROM polymarket_trades t WHERE t.condition_id = m.condition_id) AS fill_cursor
                FROM polymarket_markets m
                WHERE m.market_type = %s
                """
                , (market_type,)
            )
            return [dict(row) for row in cur.fetchall()]


def insert_polymarket_fills(rows: List[Dict[str, Any]]) -> int:
    """Batch fill insert. fill_key is a content hash computed by the caller
    (the data API exposes no trade id); ON CONFLICT DO NOTHING makes
    re-ingestion of a boundary page a no-op."""
    prepared = []
    for row in rows:
        fill_key = str(row.get("fill_key", "") or "")
        if not fill_key:
            continue
        prepared.append((
            fill_key,
            str(row.get("condition_id", "") or ""),
            _strip_nul_bytes(str(row.get("wallet", "") or "")),
            _strip_nul_bytes(str(row.get("name", "") or "")),
            str(row.get("outcome", "") or ""),
            str(row.get("side", "") or ""),
            float(row.get("size", 0) or 0),
            float(row.get("price", 0) or 0),
            row.get("filled_at"),
        ))
    if not prepared:
        return 0
    _ensure_polymarket_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO polymarket_trades
                  (fill_key, condition_id, wallet, name, outcome, side, size, price, filled_at)
                VALUES %s
                ON CONFLICT (fill_key) DO NOTHING
                """,
                prepared,
            )
            conn.commit()
    return len(prepared)


def get_polymarket_sharp_wallet_set() -> Dict[str, Dict[str, str]]:
    """wallet -> {name, archetype} for the sharp archetypes (early_sharp,
    longshot) - the exact same filter used app-wide for "sharp consensus"
    (predictions route, earnings-week route). SEC-29 alerting reuses this
    directly instead of a separate watchlist config table, since the
    archetype classification on polymarket_wallet_stats already IS the
    watchlist."""
    _ensure_polymarket_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT wallet, name, archetype FROM polymarket_wallet_stats WHERE archetype IN ('early_sharp', 'longshot')"
            )
            return {row["wallet"]: {"name": row["name"], "archetype": row["archetype"]} for row in cur.fetchall()}


def insert_polymarket_sharp_alerts(rows: List[Dict[str, Any]]) -> int:
    """Batch insert of detected sharp-wallet entries into still-open earnings
    markets (SEC-29). fill_key (the same content-hash as polymarket_trades)
    doubles as the dedup key, so re-ingesting a boundary page is a no-op."""
    prepared = []
    for row in rows:
        fill_key = str(row.get("fill_key", "") or "")
        if not fill_key:
            continue
        prepared.append((
            fill_key,
            str(row.get("condition_id", "") or ""),
            _strip_nul_bytes(str(row.get("ticker", "") or "")),
            _strip_nul_bytes(str(row.get("wallet", "") or "")),
            _strip_nul_bytes(str(row.get("name", "") or "")),
            str(row.get("archetype", "") or ""),
            str(row.get("side", "") or ""),
            str(row.get("outcome", "") or ""),
            float(row.get("size", 0) or 0),
            float(row.get("price", 0) or 0),
            row.get("filled_at"),
        ))
    if not prepared:
        return 0
    _ensure_polymarket_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO polymarket_sharp_alerts
                  (fill_key, condition_id, ticker, wallet, name, archetype, side, outcome, size, price, filled_at)
                VALUES %s
                ON CONFLICT (fill_key) DO NOTHING
                """,
                prepared,
                page_size=max(100, len(prepared)),
            )
            conn.commit()
    return len(prepared)


def prune_old_polymarket_sharp_alerts(days: int = 30) -> int:
    """Age-based retention for the alert feed. Unlike polymarket_trades,
    alerts aren't a durable-results source (nothing recomputes stats from
    them), so a simple age cutoff is sufficient - no settle-then-prune
    ordering needed."""
    _ensure_polymarket_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM polymarket_sharp_alerts WHERE filled_at < now() - (%s * INTERVAL '1 day')",
                (int(days),),
            )
            deleted = cur.rowcount
            conn.commit()
    return int(deleted or 0)


def get_polymarket_market_fills(condition_id: str) -> List[Dict[str, Any]]:
    """Full stored tape for one market, shaped for polymarket_pilot.settle_market."""
    _ensure_polymarket_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT wallet AS \"proxyWallet\", name, outcome, side, size, price, filled_at FROM polymarket_trades WHERE condition_id = %s",
                (condition_id,),
            )
            return [dict(row) for row in cur.fetchall()]


def save_polymarket_settlement(condition_id: str, ticker: str, resolved_date: Any, settled: Dict[str, Dict[str, Any]]) -> int:
    """Writes the durable per-wallet results for one resolved market and
    stamps settled_at, in one transaction (SEC-26 settle-then-prune ordering:
    settled_at is what later licenses pruning this market's raw fills)."""
    rows = []
    for wallet, stats in settled.items():
        rows.append((
            condition_id,
            _strip_nul_bytes(str(wallet)),
            _strip_nul_bytes(str(stats.get("name", "") or "")),
            _strip_nul_bytes(str(ticker or "")),
            resolved_date,
            round(float(stats.get("pnl", 0) or 0), 4),
            round(float(stats.get("cost", 0) or 0), 4),
            stats.get("win_entry_avg"),
            bool(float(stats.get("pnl", 0) or 0) > 0),
        ))
    _ensure_polymarket_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            if rows:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO polymarket_wallet_market_results
                      (condition_id, wallet, name, ticker, resolved_date, pnl, cost, win_entry_avg, correct)
                    VALUES %s
                    ON CONFLICT (condition_id, wallet) DO UPDATE SET
                      name = EXCLUDED.name, pnl = EXCLUDED.pnl, cost = EXCLUDED.cost,
                      win_entry_avg = EXCLUDED.win_entry_avg, correct = EXCLUDED.correct,
                      resolved_date = EXCLUDED.resolved_date
                    """,
                    rows,
                )
            cur.execute(
                "UPDATE polymarket_markets SET settled_at = now(), updated_at = now() WHERE condition_id = %s",
                (condition_id,),
            )
            conn.commit()
    return len(rows)


def get_polymarket_wallet_results() -> List[Dict[str, Any]]:
    """Every durable wallet-market result row - the (compact) source of truth
    for wallet-stat recomputes."""
    _ensure_polymarket_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT wallet, name, pnl, cost, win_entry_avg, correct FROM polymarket_wallet_market_results"
            )
            return [dict(row) for row in cur.fetchall()]


def upsert_polymarket_wallet_stats(rows: List[Dict[str, Any]]) -> int:
    prepared = []
    for row in rows:
        wallet = _strip_nul_bytes(str(row.get("wallet", "") or "").strip())
        if not wallet:
            continue
        prepared.append((
            wallet,
            _strip_nul_bytes(str(row.get("name", "") or "")),
            int(row.get("markets", 0) or 0),
            int(row.get("wins", 0) or 0),
            round(float(row.get("pnl", 0) or 0), 4),
            round(float(row.get("cost", 0) or 0), 4),
            row.get("win_entry_avg"),
            str(row.get("archetype", "unclassified") or "unclassified"),
        ))
    if not prepared:
        return 0
    _ensure_polymarket_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO polymarket_wallet_stats
                  (wallet, name, markets, wins, pnl, cost, win_entry_avg, archetype)
                VALUES %s
                ON CONFLICT (wallet) DO UPDATE SET
                  name = EXCLUDED.name, markets = EXCLUDED.markets, wins = EXCLUDED.wins,
                  pnl = EXCLUDED.pnl, cost = EXCLUDED.cost, win_entry_avg = EXCLUDED.win_entry_avg,
                  archetype = EXCLUDED.archetype, refreshed_at = now()
                """,
                prepared,
            )
            conn.commit()
    return len(prepared)


def save_polymarket_macro_event_settlement(
    event_key: str, cohort: str, resolved_date: Any, settled: Dict[str, Dict[str, Any]]
) -> int:
    """Persist one row per wallet/release after all bracket markets have
    been combined. This is the macro sample-size guard: ten brackets from
    one CPI print are still one observation."""
    rows = []
    for wallet, stats in settled.items():
        rows.append((
            event_key, _strip_nul_bytes(str(wallet)),
            _strip_nul_bytes(str(stats.get("name", "") or "")), cohort,
            resolved_date, round(float(stats.get("pnl", 0) or 0), 4),
            round(float(stats.get("cost", 0) or 0), 4),
            round(float(stats.get("early_cost", 0) or 0), 4),
            round(float(stats.get("pre_release_cost", 0) or 0), 4),
            round(float(stats.get("late_cost", 0) or 0), 4),
            round(float(stats.get("post_release_cost", 0) or 0), 4),
            stats.get("win_entry_avg"), bool(float(stats.get("pnl", 0) or 0) > 0),
        ))
    _ensure_polymarket_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            if rows:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO polymarket_macro_wallet_event_results
                      (event_key, wallet, name, cohort, resolved_date, pnl, cost,
                       early_cost, pre_release_cost, late_cost, post_release_cost,
                       win_entry_avg, correct)
                    VALUES %s
                    ON CONFLICT (event_key, wallet) DO UPDATE SET
                      name = EXCLUDED.name, cohort = EXCLUDED.cohort,
                      resolved_date = EXCLUDED.resolved_date, pnl = EXCLUDED.pnl,
                      cost = EXCLUDED.cost, early_cost = EXCLUDED.early_cost,
                      pre_release_cost = EXCLUDED.pre_release_cost,
                      late_cost = EXCLUDED.late_cost,
                      post_release_cost = EXCLUDED.post_release_cost,
                      win_entry_avg = EXCLUDED.win_entry_avg, correct = EXCLUDED.correct
                    """,
                    rows,
                )
            cur.execute(
                "UPDATE polymarket_markets SET settled_at = now(), updated_at = now() WHERE market_type = 'macro' AND event_key = %s",
                (event_key,),
            )
            conn.commit()
    return len(rows)


def get_polymarket_macro_wallet_results() -> List[Dict[str, Any]]:
    _ensure_polymarket_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT event_key, wallet, name, cohort, pnl, cost, early_cost,
                       pre_release_cost, late_cost, post_release_cost,
                       win_entry_avg, correct
                FROM polymarket_macro_wallet_event_results
                """
            )
            return [dict(row) for row in cur.fetchall()]


def upsert_polymarket_macro_wallet_stats(rows: List[Dict[str, Any]]) -> int:
    prepared = []
    for row in rows:
        wallet = _strip_nul_bytes(str(row.get("wallet", "") or "").strip())
        cohort = str(row.get("cohort", "") or "")
        if not wallet or not cohort:
            continue
        prepared.append((
            wallet, cohort, _strip_nul_bytes(str(row.get("name", "") or "")),
            int(row.get("events", 0) or 0), int(row.get("wins", 0) or 0),
            round(float(row.get("pnl", 0) or 0), 4),
            round(float(row.get("cost", 0) or 0), 4),
            round(float(row.get("predictive_cost", 0) or 0), 4),
            round(float(row.get("timing_cost", 0) or 0), 4),
            row.get("win_entry_avg"),
            str(row.get("archetype", "unclassified") or "unclassified"),
        ))
    if not prepared:
        return 0
    _ensure_polymarket_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO polymarket_macro_wallet_stats
                  (wallet, cohort, name, events, wins, pnl, cost,
                   predictive_cost, timing_cost, win_entry_avg, archetype)
                VALUES %s
                ON CONFLICT (wallet, cohort) DO UPDATE SET
                  name = EXCLUDED.name, events = EXCLUDED.events, wins = EXCLUDED.wins,
                  pnl = EXCLUDED.pnl, cost = EXCLUDED.cost,
                  predictive_cost = EXCLUDED.predictive_cost,
                  timing_cost = EXCLUDED.timing_cost,
                  win_entry_avg = EXCLUDED.win_entry_avg,
                  archetype = EXCLUDED.archetype, refreshed_at = now()
                """,
                prepared,
            )
            conn.commit()
    return len(prepared)


# ─── Filing catalyst events (SEC-50) ────────────────────────────────────────
# 8-K and Form 4 filings for the tracked (industry-config) universe, written
# by filing_catalyst_sync.py and read by the movers/attention routes for
# "why is this moving?" chips. Two-phase: rows land immediately with
# detail_status='pending' (chips can render a generic label), then a capped
# per-run enrichment pass fills items/summary and flips to 'done'.

_FILING_EVENTS_SCHEMA_ENSURED = False


def _ensure_filing_events_schema() -> None:
    global _FILING_EVENTS_SCHEMA_ENSURED
    if _FILING_EVENTS_SCHEMA_ENSURED:
        return
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS filing_events (
                  accession     TEXT PRIMARY KEY,
                  cik           TEXT NOT NULL,
                  ticker        TEXT NOT NULL,
                  form          TEXT NOT NULL,
                  filed_at      TIMESTAMPTZ NOT NULL,
                  items         TEXT NOT NULL DEFAULT '',
                  summary       TEXT NOT NULL DEFAULT '',
                  url           TEXT NOT NULL DEFAULT '',
                  detail_status TEXT NOT NULL DEFAULT 'pending',
                  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS filing_events_ticker_time ON filing_events (ticker, filed_at DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS filing_events_filed_at ON filing_events (filed_at DESC)")
            conn.commit()
    _FILING_EVENTS_SCHEMA_ENSURED = True


def insert_filing_events(rows: List[Dict[str, Any]]) -> int:
    """Phase 1: register newly-seen filings. ON CONFLICT DO NOTHING - the
    accession is EDGAR's own id, so re-seeing a filing is a no-op."""
    prepared = []
    for row in rows:
        accession = str(row.get("accession", "") or "").strip()
        if not accession:
            continue
        prepared.append((
            accession,
            str(row.get("cik", "") or ""),
            _strip_nul_bytes(str(row.get("ticker", "") or "")),
            str(row.get("form", "") or ""),
            row.get("filed_at"),
            _strip_nul_bytes(str(row.get("url", "") or "")),
        ))
    if not prepared:
        return 0
    _ensure_filing_events_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO filing_events (accession, cik, ticker, form, filed_at, url)
                VALUES %s
                ON CONFLICT (accession) DO NOTHING
                """,
                prepared,
                # One page so rowcount reflects the whole batch (the default
                # 100-row paging makes rowcount = last page only).
                page_size=max(100, len(prepared)),
            )
            inserted = cur.rowcount
            conn.commit()
    return int(inserted or 0)


def get_pending_filing_events(limit: int = 30) -> List[Dict[str, Any]]:
    """Newest-first enrichment queue (newest first so chips for what's
    moving TODAY get details before old backlog)."""
    _ensure_filing_events_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT accession, cik, ticker, form, filed_at
                FROM filing_events
                WHERE detail_status = 'pending'
                ORDER BY filed_at DESC
                LIMIT %s
                """,
                (int(limit),),
            )
            return [dict(row) for row in cur.fetchall()]


def update_filing_event_detail(accession: str, items: str, summary: str) -> None:
    _ensure_filing_events_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE filing_events
                SET items = %s, summary = %s, detail_status = 'done'
                WHERE accession = %s
                """,
                (_strip_nul_bytes(items or ""), _strip_nul_bytes(summary or ""), accession),
            )
            conn.commit()


def get_known_filing_accessions(accessions: List[str]) -> List[str]:
    if not accessions:
        return []
    _ensure_filing_events_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT accession FROM filing_events WHERE accession = ANY(%s)", (accessions,))
            return [str(row["accession"]) for row in cur.fetchall()]


def prune_old_filing_events(retention_days: int = 30) -> int:
    """Chips only care about recency; a month is plenty."""
    _ensure_filing_events_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM filing_events WHERE filed_at < now() - (%s * INTERVAL '1 day')",
                (int(retention_days),),
            )
            deleted = cur.rowcount
            conn.commit()
    return int(deleted or 0)


def prune_settled_polymarket_fills(days_after_settlement: int = 7) -> int:
    """SEC-26 retention: delete raw fills ONLY for markets settled at least
    N days ago (settle-then-prune - never keyed on fill age, so long-running
    open markets keep their full tape). Returns fills deleted."""
    _ensure_polymarket_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM polymarket_trades t
                USING polymarket_markets m
                WHERE m.condition_id = t.condition_id
                  AND m.settled_at IS NOT NULL
                  AND m.settled_at < now() - (%s * INTERVAL '1 day')
                """,
                (days_after_settlement,),
            )
            deleted = cur.rowcount
            cur.execute(
                """
                UPDATE polymarket_markets SET fills_pruned_at = now()
                WHERE settled_at IS NOT NULL
                  AND settled_at < now() - (%s * INTERVAL '1 day')
                  AND fills_pruned_at IS NULL
                """,
                (days_after_settlement,),
            )
            conn.commit()
    return int(deleted or 0)


# ─── Attention outcomes (enhancement 2: was the attention right?) ───────────
#
# attention_outcomes is durable and one compact row per (attention_date,
# ticker). It stores the contributing subreddits and authors as JSON AT SEED
# TIME, because reddit_attention_items is pruned at 90 days and the
# attribution has to outlive it - the same durable-rollup / ephemeral-detail
# split as polymarket_wallet_market_results vs polymarket_trades.
#
# attention_source_stats is always recomputed from the durable table, never
# from raw items, so pruning stays invisible to the product.

_ATTENTION_OUTCOMES_SCHEMA_ENSURED = False

_ATTENTION_ATTRIBUTION_SUBQUERY = """
                       COALESCE((
                         SELECT json_agg(DISTINCT i.{field})
                         FROM intelligence_mentions m
                         JOIN reddit_attention_items i ON i.source_id = m.source_id
                         WHERE m.mention_type = 'ticker'
                           AND m.value = d.ticker
                           AND m.source_type IN ('reddit_post','reddit_comment')
                           AND i.created_utc >= d.attention_date::timestamptz
                           AND i.created_utc <  d.attention_date::timestamptz + INTERVAL '1 day'
                       ), '[]'::json)::text AS {alias}
"""


def _ensure_attention_outcomes_schema() -> None:
    global _ATTENTION_OUTCOMES_SCHEMA_ENSURED
    if _ATTENTION_OUTCOMES_SCHEMA_ENSURED:
        return
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS attention_outcomes (
                  attention_date  DATE NOT NULL,
                  ticker          TEXT NOT NULL,
                  mood            TEXT NOT NULL DEFAULT 'neutral',
                  weighted_score  NUMERIC NOT NULL DEFAULT 0,
                  mention_count   INTEGER NOT NULL DEFAULT 0,
                  subreddits      TEXT NOT NULL DEFAULT '[]',
                  authors         TEXT NOT NULL DEFAULT '[]',
                  fwd_1d_pct      NUMERIC,
                  fwd_5d_pct      NUMERIC,
                  fwd_20d_pct     NUMERIC,
                  correct_1d      BOOLEAN,
                  correct_5d      BOOLEAN,
                  correct_20d     BOOLEAN,
                  seeded_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
                  resolved_at     TIMESTAMPTZ,
                  PRIMARY KEY (attention_date, ticker)
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS attention_outcomes_unresolved "
                "ON attention_outcomes (attention_date) WHERE fwd_20d_pct IS NULL"
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS attention_source_stats (
                  kind           TEXT NOT NULL,
                  key            TEXT NOT NULL,
                  rows_total     INTEGER NOT NULL DEFAULT 0,
                  scored_1d      INTEGER NOT NULL DEFAULT 0,
                  correct_1d     INTEGER NOT NULL DEFAULT 0,
                  hit_rate_1d    NUMERIC,
                  scored_5d      INTEGER NOT NULL DEFAULT 0,
                  correct_5d     INTEGER NOT NULL DEFAULT 0,
                  hit_rate_5d    NUMERIC,
                  scored_20d     INTEGER NOT NULL DEFAULT 0,
                  correct_20d    INTEGER NOT NULL DEFAULT 0,
                  hit_rate_20d   NUMERIC,
                  refreshed_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                  PRIMARY KEY (kind, key)
                )
                """
            )
            conn.commit()
    _ATTENTION_OUTCOMES_SCHEMA_ENSURED = True


def get_attention_days_missing_outcomes(day_from: Any, day_to: Any) -> List[Dict[str, Any]]:
    """Rolled-up (date, ticker) pairs with no outcome row yet, carrying the
    attribution joined from the still-retained raw items. This join is the
    only chance to capture who said it - the items are pruned at 90 days."""
    _ensure_attention_outcomes_schema()
    sql = (
        """
        SELECT d.attention_date, d.ticker, d.mood, d.weighted_score, d.mention_count,
        """
        + _ATTENTION_ATTRIBUTION_SUBQUERY.format(field="subreddit", alias="subreddits")
        + ","
        + _ATTENTION_ATTRIBUTION_SUBQUERY.format(field="author", alias="authors")
        + """
        FROM daily_stock_attention d
        LEFT JOIN attention_outcomes o
          ON o.attention_date = d.attention_date AND o.ticker = d.ticker
        WHERE d.attention_date >= %(day_from)s
          AND d.attention_date <= %(day_to)s
          AND d.reddit_count > 0
          AND o.ticker IS NULL
        """
    )
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"day_from": day_from, "day_to": day_to})
            return [dict(row) for row in cur.fetchall()]


def upsert_attention_outcomes(rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0
    _ensure_attention_outcomes_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO attention_outcomes
                  (attention_date, ticker, mood, weighted_score, mention_count, subreddits, authors)
                VALUES %s
                ON CONFLICT (attention_date, ticker) DO NOTHING
                """,
                [
                    (
                        r["attention_date"],
                        r["ticker"],
                        str(r.get("mood") or "neutral"),
                        r.get("weighted_score") or 0,
                        r.get("mention_count") or 0,
                        _strip_nul_bytes(r.get("subreddits") or "[]"),
                        _strip_nul_bytes(r.get("authors") or "[]"),
                    )
                    for r in rows
                ],
            )
            conn.commit()
            return len(rows)


def get_unresolved_attention_outcomes(max_horizon_days: int) -> List[Dict[str, Any]]:
    """Rows with at least one unfilled horizon, old enough that the shortest
    horizon could plausibly have resolved."""
    _ensure_attention_outcomes_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT attention_date, ticker, mood, subreddits, authors,
                       fwd_1d_pct, fwd_5d_pct, fwd_20d_pct
                FROM attention_outcomes
                WHERE (fwd_1d_pct IS NULL OR fwd_5d_pct IS NULL OR fwd_20d_pct IS NULL)
                  AND attention_date >= CURRENT_DATE - ((%(window)s)::int * INTERVAL '1 day')
                  AND attention_date <= CURRENT_DATE - INTERVAL '2 days'
                ORDER BY attention_date DESC
                """,
                {"window": max_horizon_days * 3},
            )
            return [dict(row) for row in cur.fetchall()]


def update_attention_outcome_returns(rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0
    _ensure_attention_outcomes_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            for r in rows:
                cur.execute(
                    """
                    UPDATE attention_outcomes
                    SET fwd_1d_pct = %(f1)s, fwd_5d_pct = %(f5)s, fwd_20d_pct = %(f20)s,
                        correct_1d = %(c1)s, correct_5d = %(c5)s, correct_20d = %(c20)s,
                        resolved_at = now()
                    WHERE attention_date = %(day)s AND ticker = %(ticker)s
                    """,
                    {
                        "f1": r.get("fwd_1d_pct"),
                        "f5": r.get("fwd_5d_pct"),
                        "f20": r.get("fwd_20d_pct"),
                        "c1": r.get("correct_1d"),
                        "c5": r.get("correct_5d"),
                        "c20": r.get("correct_20d"),
                        "day": r["attention_date"],
                        "ticker": r["ticker"],
                    },
                )
            conn.commit()
            return len(rows)


def get_resolved_attention_outcomes() -> List[Dict[str, Any]]:
    """Every row with at least one graded horizon - the input to hit rates."""
    _ensure_attention_outcomes_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT attention_date, ticker, mood, subreddits, authors,
                       fwd_1d_pct, fwd_5d_pct, fwd_20d_pct
                FROM attention_outcomes
                WHERE fwd_1d_pct IS NOT NULL
                """
            )
            return [dict(row) for row in cur.fetchall()]


def replace_attention_source_stats(rows: List[Dict[str, Any]]) -> int:
    """Recomputed wholesale: a key that no longer qualifies must disappear
    rather than linger at a stale hit rate."""
    _ensure_attention_outcomes_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM attention_source_stats")
            if rows:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO attention_source_stats
                      (kind, key, rows_total, scored_1d, correct_1d, hit_rate_1d,
                       scored_5d, correct_5d, hit_rate_5d, scored_20d, correct_20d, hit_rate_20d)
                    VALUES %s
                    """,
                    [
                        (
                            r["kind"],
                            r["key"],
                            r.get("rows_total", 0),
                            r.get("scored_1d", 0),
                            r.get("correct_1d", 0),
                            r.get("hit_rate_1d"),
                            r.get("scored_5d", 0),
                            r.get("correct_5d", 0),
                            r.get("hit_rate_5d"),
                            r.get("scored_20d", 0),
                            r.get("correct_20d", 0),
                            r.get("hit_rate_20d"),
                        )
                        for r in rows
                    ],
                )
            conn.commit()
            return len(rows)


# ─── Attention alerts (enhancement 3) ───────────────────────────────────────
#
# Same shape as polymarket_sharp_alerts: content-hash dedup key with
# ON CONFLICT DO NOTHING so a rollup re-run is idempotent, and plain age
# pruning rather than settle-then-prune. Alerts are a display concern; the
# durable record of what happened lives in attention_outcomes.

_ATTENTION_ALERTS_SCHEMA_ENSURED = False

ATTENTION_ALERT_RETENTION_DAYS = 30


def _ensure_attention_alerts_schema() -> None:
    global _ATTENTION_ALERTS_SCHEMA_ENSURED
    if _ATTENTION_ALERTS_SCHEMA_ENSURED:
        return
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS attention_alerts (
                  alert_key      TEXT PRIMARY KEY,
                  attention_date DATE NOT NULL,
                  ticker         TEXT NOT NULL,
                  alert_type     TEXT NOT NULL,
                  rank           INTEGER NOT NULL DEFAULT 0,
                  detail         TEXT NOT NULL DEFAULT '{}',
                  created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS attention_alerts_recent "
                "ON attention_alerts (attention_date DESC, rank ASC)"
            )
            conn.commit()
    _ATTENTION_ALERTS_SCHEMA_ENSURED = True


def insert_attention_alerts(rows: List[Dict[str, Any]]) -> int:
    """Returns rows offered, not rows written: ON CONFLICT DO NOTHING means a
    re-run legitimately writes nothing, and execute_values' rowcount only
    reflects the final page anyway (a trap already hit in SEC-50)."""
    if not rows:
        return 0
    _ensure_attention_alerts_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO attention_alerts
                  (alert_key, attention_date, ticker, alert_type, rank, detail)
                VALUES %s
                ON CONFLICT (alert_key) DO NOTHING
                """,
                [
                    (
                        _strip_nul_bytes(r["alert_key"]),
                        r["attention_date"],
                        _strip_nul_bytes(r["ticker"]),
                        _strip_nul_bytes(r["alert_type"]),
                        int(r.get("rank") or 0),
                        _strip_nul_bytes(r.get("detail") or "{}"),
                    )
                    for r in rows
                ],
            )
            conn.commit()
            return len(rows)


def get_attention_alerts(limit: int = 100, ticker: Optional[str] = None) -> List[Dict[str, Any]]:
    _ensure_attention_alerts_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            if ticker:
                cur.execute(
                    """
                    SELECT alert_key, attention_date, ticker, alert_type, rank, detail, created_at
                    FROM attention_alerts
                    WHERE ticker = %(ticker)s
                    ORDER BY attention_date DESC, rank ASC
                    LIMIT %(limit)s
                    """,
                    {"ticker": ticker.upper(), "limit": max(1, min(limit, 500))},
                )
            else:
                cur.execute(
                    """
                    SELECT alert_key, attention_date, ticker, alert_type, rank, detail, created_at
                    FROM attention_alerts
                    ORDER BY attention_date DESC, rank ASC
                    LIMIT %(limit)s
                    """,
                    {"limit": max(1, min(limit, 500))},
                )
            return [dict(row) for row in cur.fetchall()]


def get_known_attention_tickers(before_date: Any = None) -> List[str]:
    """Every ticker that has appeared in a rollup BEFORE `before_date`, for
    first-appearance detection. Reads the rollup table, which is never pruned,
    so newcomer status stays correct indefinitely.

    Excluding the target day is load-bearing, not defensive: the rollup
    replaces a date wholesale, and a --date re-run would otherwise find the
    day's own tickers already present and never fire a first-appearance alert.
    """
    _ensure_stock_attention_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            if before_date is None:
                cur.execute("SELECT DISTINCT ticker FROM daily_stock_attention")
            else:
                cur.execute(
                    "SELECT DISTINCT ticker FROM daily_stock_attention WHERE attention_date < %s",
                    (before_date,),
                )
            return [str(row["ticker"]) for row in cur.fetchall()]


def get_daily_attention_rows(day: Any) -> List[Dict[str, Any]]:
    """One day's rollup rows, ranked as the board shows them - the prior-day
    baseline for surge detection."""
    _ensure_stock_attention_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ticker, mention_count, total_mention_count, weighted_score
                FROM daily_stock_attention
                WHERE attention_date = %s
                ORDER BY weighted_score DESC, ticker ASC
                """,
                (day,),
            )
            return [dict(row) for row in cur.fetchall()]


def prune_attention_alerts(retention_days: int = ATTENTION_ALERT_RETENTION_DAYS) -> int:
    _ensure_attention_alerts_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM attention_alerts "
                "WHERE attention_date < CURRENT_DATE - ((%s)::int * INTERVAL '1 day')",
                (max(1, retention_days),),
            )
            deleted = cur.rowcount
            conn.commit()
    return int(deleted or 0)

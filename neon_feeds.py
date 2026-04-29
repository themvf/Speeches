"""
Neon (Postgres) helpers for managing rss_feeds.
Requires DATABASE_URL env var (Neon connection string).
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
                ON CONFLICT (feed_url) DO UPDATE SET label = EXCLUDED.label, active = true
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

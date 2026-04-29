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

#!/usr/bin/env python3
"""
Reddit scraper for keyword-based financial/regulatory content discovery.

Uses the public Reddit JSON API (no auth required). Searches across all of
Reddit or within specified subreddits for configurable search terms, then
returns posts in the project's standard document schema.
"""

from __future__ import annotations

import hashlib
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests


REDDIT_SEARCH_URL = "https://www.reddit.com/search.json"
REDDIT_BASE_URL = "https://www.reddit.com"

# Default search terms aligned with the project's financial/regulatory focus
DEFAULT_SEARCH_TERMS: List[str] = [
    "stablecoins",
    "tokenization",
    "prediction markets",
    "crypto regulation",
    "digital assets SEC",
    "DeFi regulation",
    "SEC enforcement",
    "securities fraud",
    "insider trading",
    "FINRA",
    "money market funds",
    "stablecoin regulation",
    "crypto enforcement",
]

DEFAULT_TAGS = "reddit,social-media,financial-regulation,crypto,securities"


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _post_id_from_url(permalink: str) -> str:
    """Extract Reddit post ID from a permalink path."""
    parts = [p for p in str(permalink or "").split("/") if p]
    # Reddit permalinks: /r/<sub>/comments/<id>/<slug>/
    if "comments" in parts:
        idx = parts.index("comments")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return ""


def _document_id(post_id: str) -> str:
    """Stable document ID derived from Reddit post ID."""
    raw = f"reddit_post_{post_id}"
    return hashlib.sha1(raw.encode()).hexdigest()[:24]


def _utc_iso(ts: Any) -> str:
    """Convert a Unix timestamp to ISO-8601 UTC string."""
    try:
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return ""


def _human_date(ts: Any) -> str:
    """Convert a Unix timestamp to a human-readable date string."""
    try:
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        return dt.strftime("%B %d, %Y")
    except Exception:
        return ""


def _build_full_text(post: Dict[str, Any]) -> str:
    """Build full text for a Reddit post (selftext or title + link)."""
    title = _normalize_space(post.get("title", ""))
    selftext = _normalize_space(post.get("selftext", ""))
    subreddit = post.get("subreddit_name_prefixed", "")
    author = post.get("author", "")
    score = post.get("score", 0)
    num_comments = post.get("num_comments", 0)
    url = post.get("url", "")

    header = f"{title}"
    meta = f"Posted in {subreddit} by u/{author} | Score: {score} | Comments: {num_comments}"

    if selftext and selftext not in ("[deleted]", "[removed]"):
        body = selftext
    else:
        body = f"[Link post] {url}"

    return "\n\n".join(part for part in [header, meta, body] if part)


class RedditScraper:
    """
    Discovers Reddit posts matching keyword search terms using the public
    JSON API. No authentication required.

    Rate-limited to ~1 request/sec to stay within Reddit's anonymous limits.
    """

    def __init__(
        self,
        search_terms: Optional[List[str]] = None,
        subreddits: Optional[List[str]] = None,
        sort: str = "new",
        time_filter: str = "week",
        limit_per_term: int = 25,
        tags_csv: str = DEFAULT_TAGS,
        min_delay_seconds: float = 1.2,
    ):
        self.search_terms = search_terms if search_terms is not None else list(DEFAULT_SEARCH_TERMS)
        self.subreddits = subreddits or []
        self.sort = sort if sort in ("new", "relevance", "hot", "top", "comments") else "new"
        self.time_filter = time_filter if time_filter in ("hour", "day", "week", "month", "year", "all") else "week"
        self.limit_per_term = max(1, min(100, int(limit_per_term or 25)))
        self.tags = [t.strip() for t in str(tags_csv or "").split(",") if t.strip()]
        self.min_delay_seconds = max(0.5, float(min_delay_seconds))
        self._last_request_ts = 0.0

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "PolicyResearchHub/1.0 (financial regulatory research; contact via GitHub)",
            "Accept": "application/json",
        })

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_ts
        if elapsed < self.min_delay_seconds:
            time.sleep(self.min_delay_seconds - elapsed)
        self._last_request_ts = time.time()

    def _fetch_search(self, term: str, after: str = "") -> Dict[str, Any]:
        """Fetch one page of search results for a term."""
        self._rate_limit()

        params: Dict[str, Any] = {
            "q": term,
            "sort": self.sort,
            "t": self.time_filter,
            "limit": self.limit_per_term,
            "type": "link",
        }
        if after:
            params["after"] = after

        # Scope to specific subreddits if configured
        if self.subreddits:
            sub_str = "+".join(self.subreddits)
            url = f"https://www.reddit.com/r/{sub_str}/search.json"
            params["restrict_sr"] = "1"
        else:
            url = REDDIT_SEARCH_URL

        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def discover_posts(self) -> List[Dict[str, Any]]:
        """
        Search for all configured terms and return deduplicated post metadata.

        Returns a list of dicts with keys: post_id, title, url, permalink,
        subreddit, author, score, num_comments, created_utc, selftext,
        matched_terms, is_self.
        """
        seen_ids: set[str] = set()
        posts: List[Dict[str, Any]] = []

        for term in self.search_terms:
            try:
                payload = self._fetch_search(term)
            except Exception as exc:
                print(f"[reddit_scraper] Search failed for '{term}': {exc}")
                continue

            children = payload.get("data", {}).get("children", [])
            for child in children:
                data = child.get("data", {})
                post_id = str(data.get("id", "")).strip()
                if not post_id or post_id in seen_ids:
                    continue
                seen_ids.add(post_id)

                # Skip deleted/removed posts with no useful content
                selftext = str(data.get("selftext", "") or "")
                is_self = bool(data.get("is_self", False))
                if is_self and selftext in ("[deleted]", "[removed]", ""):
                    if not data.get("title", "").strip():
                        continue

                posts.append({
                    "post_id": post_id,
                    "title": _normalize_space(data.get("title", "")),
                    "url": str(data.get("url", "") or ""),
                    "permalink": REDDIT_BASE_URL + str(data.get("permalink", "") or ""),
                    "subreddit": str(data.get("subreddit_name_prefixed", "") or f"r/{data.get('subreddit', '')}"),
                    "author": str(data.get("author", "") or "[deleted]"),
                    "score": int(data.get("score", 0) or 0),
                    "num_comments": int(data.get("num_comments", 0) or 0),
                    "created_utc": data.get("created_utc", 0),
                    "selftext": selftext,
                    "is_self": is_self,
                    "matched_terms": [term],
                    "_raw": data,
                })

        # Merge matched_terms for posts discovered by multiple search terms
        # (they were deduped above so this handles only newly discovered ones)
        return posts

    def build_documents(
        self,
        posts: Optional[List[Dict[str, Any]]] = None,
        existing_ids: Optional[set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convert raw post data into the project's standard document schema.

        Args:
            posts: Output of discover_posts(). If None, calls discover_posts().
            existing_ids: Set of document_ids already in the corpus to skip.

        Returns list of dicts with 'metadata' and 'content' keys.
        """
        if posts is None:
            posts = self.discover_posts()

        existing_ids = existing_ids or set()
        documents: List[Dict[str, Any]] = []

        for post in posts:
            post_id = post["post_id"]
            doc_id = _document_id(post_id)

            if doc_id in existing_ids:
                continue

            created_utc = post["created_utc"]
            iso_date = _utc_iso(created_utc)
            human_date = _human_date(created_utc)

            full_text = _build_full_text(post["_raw"])
            word_count = len(full_text.split())

            # Use permalink as the canonical URL so every post is addressable
            canonical_url = post["permalink"]

            # Tags: base tags + subreddit name
            sub_slug = post["subreddit"].lstrip("r/").lower().replace(" ", "-")
            doc_tags = list(self.tags) + [sub_slug]

            metadata: Dict[str, Any] = {
                "document_id": doc_id,
                "title": post["title"],
                "speaker": post["author"],
                "date": human_date,
                "published_date": human_date,
                "url": canonical_url,
                "word_count": word_count,
                "organization": "Reddit",
                "doc_type": "Social Media Post",
                "source_filename": f"reddit_{post_id}.json",
                "source_format": "json",
                "source_local_path": "",
                "source_gcs_path": "",
                "tags": doc_tags,
                "source_kind": "reddit_post",
                "source_family": "reddit_post",
                "source_index_url": REDDIT_SEARCH_URL,
                "subreddit": post["subreddit"],
                "reddit_post_id": post_id,
                "reddit_score": post["score"],
                "reddit_num_comments": post["num_comments"],
                "matched_terms": post["matched_terms"],
                "external_url": post["url"] if not post["is_self"] else "",
            }

            paragraphs = [p for p in full_text.split("\n\n") if p.strip()]

            content: Dict[str, Any] = {
                "full_text": full_text,
                "paragraphs": paragraphs,
                "sentences": [],
            }

            documents.append({"metadata": metadata, "content": content})

        return documents


def scrape_reddit(
    search_terms: Optional[List[str]] = None,
    subreddits: Optional[List[str]] = None,
    sort: str = "new",
    time_filter: str = "week",
    limit_per_term: int = 25,
    tags_csv: str = DEFAULT_TAGS,
    existing_ids: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function: discover and build documents in one call.

    Returns a list of documents in the project's standard schema, skipping
    any whose document_id is already in existing_ids.
    """
    scraper = RedditScraper(
        search_terms=search_terms,
        subreddits=subreddits,
        sort=sort,
        time_filter=time_filter,
        limit_per_term=limit_per_term,
        tags_csv=tags_csv,
    )
    posts = scraper.discover_posts()
    return scraper.build_documents(posts=posts, existing_ids=existing_ids)


if __name__ == "__main__":
    """
    Standalone runner: reads settings from news_connector_settings.json,
    discovers Reddit posts, and upserts them into custom_documents.json.

    Usage:
        python reddit_scraper.py [--dry-run] [--limit N]
    """
    import argparse
    import json
    import sys
    from pathlib import Path

    import run_financial_news_pipeline as core

    parser = argparse.ArgumentParser(description="Ingest Reddit posts into the corpus.")
    parser.add_argument("--dry-run", action="store_true", help="Discover but do not save.")
    parser.add_argument("--limit", type=int, default=0, help="Max posts to save (0 = all).")
    parser.add_argument("--require-remote-persistence", action="store_true")
    args = parser.parse_args()

    # Load GCS storage (optional)
    secrets_payload = core._load_secrets()
    storage = core._init_storage(secrets_payload)

    # Load settings
    settings = core._load_news_connector_settings(storage)
    reddit_cfg = settings.get("reddit", {})

    if not reddit_cfg.get("enabled", True):
        print("[reddit_scraper] Reddit connector disabled in settings. Exiting.")
        sys.exit(0)

    search_terms = reddit_cfg.get("search_terms") or DEFAULT_SEARCH_TERMS
    subreddits = reddit_cfg.get("subreddits") or []
    sort = str(reddit_cfg.get("sort", "new"))
    time_filter = str(reddit_cfg.get("time_filter", "week"))
    limit_per_term = int(reddit_cfg.get("limit_per_term", 25))
    tags_csv = str(reddit_cfg.get("tags_csv", DEFAULT_TAGS))

    # Load existing corpus to compute existing_ids for dedup
    custom_payload = core._load_custom_documents(storage)
    existing_ids: set[str] = set()
    for item in custom_payload.get("documents", []):
        if isinstance(item, dict):
            meta = item.get("metadata", {})
            doc_id = str(meta.get("document_id", "") or "").strip()
            if doc_id:
                existing_ids.add(doc_id)

    print(f"[reddit_scraper] Searching {len(search_terms)} terms (time_filter={time_filter}, limit_per_term={limit_per_term})")
    documents = scrape_reddit(
        search_terms=search_terms,
        subreddits=subreddits,
        sort=sort,
        time_filter=time_filter,
        limit_per_term=limit_per_term,
        tags_csv=tags_csv,
        existing_ids=existing_ids,
    )

    limit = args.limit if args.limit and args.limit > 0 else len(documents)
    documents = documents[:limit]
    print(f"[reddit_scraper] {len(documents)} new posts discovered.")

    if args.dry_run or not documents:
        for doc in documents:
            meta = doc.get("metadata", {})
            print(f"  {meta.get('subreddit')} | {meta.get('title', '')[:80]}")
        sys.exit(0)

    saved = 0
    for doc in documents:
        core._upsert_custom_document_record(custom_payload, doc)
        saved += 1

    core._save_custom_documents(storage, custom_payload, require_remote=args.require_remote_persistence)
    print(f"[reddit_scraper] Saved {saved} posts to custom_documents.json.")

    enrichment_state = core._load_enrichment_state(storage)
    core._rebuild_rule_summaries(
        storage,
        custom_payload=custom_payload,
        enrichment_state=enrichment_state,
        require_remote=args.require_remote_persistence,
    )

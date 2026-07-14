#!/usr/bin/env python3
"""Daily rollup for the stock attention tracker
(docs/stock-attention-spec.md §6; enhancement items 1-2 in
docs/stock-attention-enhancements-spec.md).

Runs at 00:15 UTC and aggregates the just-closed UTC day, bucketing by
reddit_attention_items.created_utc (when the post/comment was actually
written on Reddit) - NOT by the mention row's generated_at (when a sweep
happened to store it), which would misfile late-day items swept after
midnight. News-channel counts (item 1) bucket the same UTC day by
rss_articles' COALESCE(published_at, fetched_at), matching this app's
existing convention elsewhere (e.g. getRecentArticles).

Idempotent per date: rows for the target date are replaced wholesale in
one transaction (delete-then-insert), so a re-run after resolver/blacklist
tuning, or after a market-data fetch failure, fully recomputes the day -
including removing tickers that no longer resolve - without re-sweeping
Reddit or re-scanning news.

Ends with a retention sweep (skipped on --dry-run): raw reddit items and
their mention rows older than the retention window are deleted. The
daily_stock_attention rollups persist indefinitely - they're tiny and
they're the product. This sweep exists because pruneOldRssData (neon.ts)
only prunes source_type='rss_article' mentions; reddit rows would
otherwise grow unbounded (spec §6.3).

News-channel ticker mentions (item 1) are computed on the fly from
rss_articles at rollup time and are NOT persisted into
intelligence_mentions - the spec's "prefer the Python-side pass" choice
keeps this additive (no new write path, no volume added to a table other
features already query) at the cost of no per-article news drill-down for
now (only the daily count survives). Reddit's mention-level persistence
and drill-down are unaffected.

Market context (item 2) is fetched per rolled-up ticker from Yahoo's
unofficial chart endpoint (yahoo_market_data.py) - best-effort; a fetch
failure leaves price/volume columns NULL for that ticker rather than
failing the row or the run. divergence is a simple, documented v1
heuristic (top-N attention rank vs. price move size), not the z-score
approach planned for enhancement item 10.

Usage:
    python aggregate_stock_attention.py [--date YYYY-MM-DD] [--dry-run]
        [--retention-days N] [--skip-news] [--skip-market]
        [--summary-path PATH]

Required env vars: DATABASE_URL
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import UTC, date, datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import neon_feeds
import ticker_resolver
import yahoo_market_data

DEFAULT_RETENTION_DAYS = 90

# Authors leaderboard: how many of an account's tickers/subreddits to name.
TOP_N_AUTHOR_FACETS = 3

# Item 6 quality-flag thresholds - code constants (documented, unit-tested),
# deliberately not admin config: these define what the flags MEAN, and a
# silently-moved threshold would make historical flags incomparable.
CREW_OVERLAP_THRESHOLD = 0.6      # today-vs-yesterday author overlap coefficient
CREW_MIN_AUTHORS = 3
YOUNG_ACCOUNT_DAYS = 30
YOUNG_SHARE_THRESHOLD = 0.5
YOUNG_MIN_KNOWN = 3               # need this many known-age authors before judging
SINGLE_THREAD_SHARE = 0.8
SINGLE_THREAD_MIN_ROWS = 5
REVIEW_QUEUE_TOP_N = 50

_THREAD_KEY_RE = re.compile(r"/comments/([a-z0-9]+)/", re.IGNORECASE)

# Item 2 divergence heuristic thresholds (spec: "simple, documented v1
# heuristic pending item 10's proper z-score work"). Ranks are 1-indexed
# by total_mention_count within the day's rolled-up tickers.
ATTENTION_SPIKE_RANK_THRESHOLD = 20
SMALL_PRICE_MOVE_PCT = 1.5
LARGE_PRICE_MOVE_PCT = 5.0

FETCH_DAY_ROWS_SQL = """
    SELECT m.value AS ticker, m.source_id, i.author, i.subreddit, i.score, i.mood, i.permalink
    FROM intelligence_mentions m
    JOIN reddit_attention_items i ON i.source_id = m.source_id
    WHERE m.mention_type = 'ticker'
      AND m.source_type IN ('reddit_post', 'reddit_comment')
      AND i.created_utc >= %(day_start)s
      AND i.created_utc < %(day_end)s
"""

# Item 5: per-author aggregates over the full retained items window (90d).
FETCH_AUTHOR_ITEM_STATS_SQL = """
    SELECT author, MIN(created_utc) AS first_seen, MAX(created_utc) AS last_seen,
           COUNT(*) AS items_total, COUNT(DISTINCT subreddit) AS subreddits_distinct
    FROM reddit_attention_items
    GROUP BY author
"""

FETCH_AUTHOR_TICKER_COUNTS_SQL = """
    SELECT i.author, m.value AS ticker, COUNT(*) AS cnt
    FROM intelligence_mentions m
    JOIN reddit_attention_items i ON i.source_id = m.source_id
    WHERE m.mention_type = 'ticker'
      AND m.source_type IN ('reddit_post', 'reddit_comment')
    GROUP BY i.author, m.value
"""

# Authors view: per-author-per-subreddit item counts, so the leaderboard can
# name an account's top few subreddits instead of just counting distinct subs.
FETCH_AUTHOR_SUBREDDIT_COUNTS_SQL = """
    SELECT author, subreddit, COUNT(*) AS cnt
    FROM reddit_attention_items
    GROUP BY author, subreddit
"""

FETCH_KNOWN_ACCOUNT_AGES_SQL = """
    SELECT author, account_created FROM reddit_author_stats WHERE account_created IS NOT NULL
"""

FETCH_SEEN_TICKERS_SQL = """
    SELECT DISTINCT ticker FROM daily_stock_attention WHERE attention_date < %(day)s
"""

INSERT_REVIEW_QUEUE_SQL = """
    INSERT INTO attention_review_queue (review_date, ticker, sample_source_ids)
    VALUES %s
    ON CONFLICT (review_date, ticker) DO NOTHING
"""

FETCH_NEWS_ROWS_SQL = """
    SELECT id, title, description
    FROM rss_articles
    WHERE COALESCE(published_at, fetched_at) >= %(day_start)s
      AND COALESCE(published_at, fetched_at) < %(day_end)s
"""

DELETE_DAY_SQL = "DELETE FROM daily_stock_attention WHERE attention_date = %(day)s"

INSERT_DAY_SQL = """
    INSERT INTO daily_stock_attention
      (attention_date, ticker, company, mention_count, source_count, subreddit_count,
       weighted_score, mood, top_source_ids, reddit_count, news_count, total_mention_count,
       price_close, price_pct, volume, volume_vs_20d, divergence,
       weighted_mention_count, quality_flags, top_news_ids)
    VALUES %s
"""

RETENTION_DELETE_MENTIONS_VIA_ITEMS_SQL = """
    DELETE FROM intelligence_mentions m
    USING reddit_attention_items i
    WHERE m.source_id = i.source_id
      AND m.source_type IN ('reddit_post', 'reddit_comment')
      AND i.created_utc < %(cutoff)s
"""

# Backstop for mention rows whose item row is somehow gone already -
# without this, orphans would survive the join-based delete forever.
RETENTION_DELETE_ORPHAN_MENTIONS_SQL = """
    DELETE FROM intelligence_mentions
    WHERE source_type IN ('reddit_post', 'reddit_comment')
      AND generated_at < %(cutoff)s
"""

RETENTION_DELETE_ITEMS_SQL = """
    DELETE FROM reddit_attention_items WHERE created_utc < %(cutoff)s
"""


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def day_bounds(day: date) -> Tuple[datetime, datetime]:
    start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    return start, start + timedelta(days=1)


def _top_n(counts: Dict[str, int], key_name: str, n: int = TOP_N_AUTHOR_FACETS) -> List[Dict[str, Any]]:
    """Top-n (label, count) pairs, ranked by count desc then label asc so
    ties are deterministic. Keyed by `key_name` for the JSON the UI reads."""
    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:n]
    return [{key_name: label, "count": count} for label, count in ranked]


def compute_author_stats(
    item_stat_rows: List[Dict[str, Any]],
    author_ticker_rows: List[Dict[str, Any]],
    author_subreddit_rows: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Item 5, pure: merges per-author item aggregates with per-author
    ticker-concentration figures. top_ticker_share = share of the author's
    ticker-mentions going to their single most-mentioned ticker (1.0 = a
    one-ticker account). top_tickers / top_subreddits carry the account's
    three most-mentioned tickers and most-active subreddits (JSON strings)
    for the Authors leaderboard; top_ticker stays as top_tickers[0] for the
    discount logic and back-compat."""
    ticker_counts: Dict[str, Dict[str, int]] = {}
    for row in author_ticker_rows:
        author = str(row["author"])
        ticker_counts.setdefault(author, {})[str(row["ticker"])] = int(row["cnt"])

    subreddit_counts: Dict[str, Dict[str, int]] = {}
    for row in author_subreddit_rows or []:
        author = str(row["author"])
        subreddit_counts.setdefault(author, {})[str(row["subreddit"])] = int(row["cnt"])

    stats: List[Dict[str, Any]] = []
    for row in item_stat_rows:
        author = str(row["author"])
        counts = ticker_counts.get(author, {})
        total = sum(counts.values())
        top_tickers = _top_n(counts, "ticker")
        top_subreddits = _top_n(subreddit_counts.get(author, {}), "subreddit")
        stats.append({
            "author": author,
            "first_seen": row.get("first_seen"),
            "last_seen": row.get("last_seen"),
            "items_total": int(row.get("items_total", 0) or 0),
            "tickers_distinct": len(counts),
            "subreddits_distinct": int(row.get("subreddits_distinct", 0) or 0),
            "top_ticker_share": (max(counts.values()) / total) if total else 0.0,
            "top_ticker": top_tickers[0]["ticker"] if top_tickers else "",
            "top_tickers": json.dumps(top_tickers),
            "top_subreddits": json.dumps(top_subreddits),
        })
    return stats


def build_author_weights(stats: List[Dict[str, Any]], params: Dict[str, Any] | None = None) -> Dict[str, float]:
    """Item 5, pure: discount low-diversity, high-concentration accounts.
    min_items keeps casual one-off posters (1 item, share 1.0 by
    construction) at full weight - the discount targets the *repeat*
    single-ticker pattern, not everyone who mentioned one stock once."""
    params = params or {}
    share_threshold = float(params.get("low_diversity_share", 0.8))
    max_tickers = int(params.get("low_diversity_max_tickers", 2))
    min_items = int(params.get("min_items", 5))
    discount = float(params.get("discount", 0.25))
    weights: Dict[str, float] = {}
    for stat in stats:
        if (
            stat["items_total"] >= min_items
            and stat["tickers_distinct"] <= max_tickers
            and stat["top_ticker_share"] > share_threshold
        ):
            weights[stat["author"]] = discount
    return weights  # absent = 1.0


def thread_key(permalink: str) -> str:
    """Reddit thread identity from a permalink - comments on the same
    submission share the /comments/<id>/ segment."""
    match = _THREAD_KEY_RE.search(str(permalink or ""))
    return match.group(1).lower() if match else str(permalink or "")


def compute_quality_flags(
    ticker_rows: List[Dict[str, Any]],
    yesterday_authors: Set[str],
    account_ages: Dict[str, datetime],
    now: datetime,
) -> List[str]:
    """Item 6, pure: manipulation-pattern flags for one ticker's day rows.
    Flags annotate, never suppress - the row still ranks normally."""
    flags: List[str] = []
    authors = {str(row["author"]) for row in ticker_rows}

    if len(authors) >= CREW_MIN_AUTHORS and len(yesterday_authors) >= CREW_MIN_AUTHORS:
        overlap = len(authors & yesterday_authors) / min(len(authors), len(yesterday_authors))
        if overlap >= CREW_OVERLAP_THRESHOLD:
            flags.append("same_author_crew")

    known_ages = [(now - account_ages[a]).days for a in authors if a in account_ages]
    if len(known_ages) >= YOUNG_MIN_KNOWN:
        young_share = sum(1 for age_days in known_ages if age_days < YOUNG_ACCOUNT_DAYS) / len(known_ages)
        if young_share >= YOUNG_SHARE_THRESHOLD:
            flags.append("young_account_concentration")

    if len(ticker_rows) >= SINGLE_THREAD_MIN_ROWS:
        thread_counts: Dict[str, int] = {}
        for row in ticker_rows:
            key = thread_key(row.get("permalink", ""))
            thread_counts[key] = thread_counts.get(key, 0) + 1
        if max(thread_counts.values()) / len(ticker_rows) >= SINGLE_THREAD_SHARE:
            flags.append("single_thread_concentration")

    return flags


def compute_weighted_score(mention_count: float, subreddit_count: int, source_count: int) -> float:
    """Spec §6.2: deduped humans talking is the base signal; spread across
    communities amplifies most (harder to fake than volume inside one
    board); spread across threads amplifies mildly. No freshness decay in a
    daily rollup - that belongs in a future intraday view. Unchanged by
    the item 1/2 additions - scored on Reddit mention_count only, since
    news-article counts are a different trust profile (one article isn't
    equivalent to one deduped human) and mixing them would need its own
    calibration, not a guess baked into the existing formula."""
    return round(
        mention_count
        * (1 + 0.15 * min(subreddit_count, 6))
        * (1 + 0.05 * min(source_count, 10)),
        4,
    )


def aggregate_rows(
    rows: List[Dict[str, Any]],
    author_weights: Dict[str, float] | None = None,
    subreddit_weights: Dict[str, float] | None = None,
) -> List[Dict[str, Any]]:
    """Pure aggregation of (ticker, source_id, author, subreddit, score,
    mood) fetch rows into per-ticker rollups. Each input row is one
    item+ticker pair (the mentions unique constraint guarantees that).

    weighted_mention_count (items 4+5) = per deduped author, that author's
    credibility weight x the max weight among the subreddits they mentioned
    the ticker in. Both weight maps default to all-1.0, in which case it
    equals mention_count exactly. Raw counts are always stored unweighted.
    """
    author_weights = author_weights or {}
    subreddit_weights = subreddit_weights or {}

    by_ticker: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        ticker = str(row["ticker"])
        agg = by_ticker.setdefault(ticker, {
            "authors": set(),
            "sources": set(),
            "subreddits": set(),
            "moods": {"bullish": 0, "bearish": 0, "neutral": 0},
            "scored_sources": [],
            "author_subreddits": {},
        })
        author = str(row["author"])
        agg["authors"].add(author)
        agg["sources"].add(str(row["source_id"]))
        agg["subreddits"].add(str(row["subreddit"]))
        agg["author_subreddits"].setdefault(author, set()).add(str(row["subreddit"]))
        mood = str(row.get("mood", "neutral") or "neutral")
        if mood in agg["moods"]:
            agg["moods"][mood] += 1
        agg["scored_sources"].append((int(row.get("score", 0) or 0), str(row["source_id"])))

    out: List[Dict[str, Any]] = []
    for ticker, agg in by_ticker.items():
        moods = agg["moods"]
        directional = {label: count for label, count in moods.items() if label != "neutral" and count > 0}
        if not directional:
            mood = "neutral"
        elif len(directional) == 1:
            mood = next(iter(directional))
        elif moods["bullish"] == moods["bearish"]:
            mood = "mixed"
        else:
            mood = "bullish" if moods["bullish"] > moods["bearish"] else "bearish"

        top_sources = [
            source_id
            for _, source_id in sorted(set(agg["scored_sources"]), key=lambda pair: (-pair[0], pair[1]))[:10]
        ]
        reddit_count = len(agg["authors"])
        source_count = len(agg["sources"])
        subreddit_count = len(agg["subreddits"])
        weighted_mentions = round(
            sum(
                author_weights.get(author, 1.0)
                * max(subreddit_weights.get(sub, 1.0) for sub in subs)
                for author, subs in agg["author_subreddits"].items()
            ),
            4,
        )
        out.append({
            "ticker": ticker,
            "company": ticker_resolver.ticker_title(ticker),
            "mention_count": reddit_count,       # unchanged meaning: Reddit "Real Mentions" dedup count
            "reddit_count": reddit_count,          # explicit alias - see module docstring on item 1
            "news_count": 0,
            "total_mention_count": reddit_count,
            "weighted_mention_count": weighted_mentions,
            "source_count": source_count,
            "subreddit_count": subreddit_count,
            "weighted_score": compute_weighted_score(weighted_mentions, subreddit_count, source_count),
            "mood": mood,
            "top_source_ids": json.dumps(top_sources),
            "quality_flags": "[]",
            "top_news_ids": "[]",
        })
    out.sort(key=lambda row: (-row["weighted_score"], row["ticker"]))
    return out


def compute_news_ticker_mentions(articles: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Pure: per-ticker list of contributing rss_articles ids (SEC-4). One
    article can produce multiple tickers; a ticker mentioned twice in one
    article's title+description still contributes that article once."""
    by_ticker: Dict[str, List[Any]] = {}
    for article in articles:
        text = f"{article.get('title', '') or ''} {article.get('description', '') or ''}"
        for symbol in ticker_resolver.resolve_tickers(text):
            by_ticker.setdefault(symbol, []).append(article.get("id"))
    return by_ticker


def compute_news_ticker_counts(articles: List[Dict[str, Any]]) -> Dict[str, int]:
    """Pure: distinct-article count per ticker resolved from article
    title+description text — the count the leaderboard displays."""
    return {ticker: len(ids) for ticker, ids in compute_news_ticker_mentions(articles).items()}


def merge_news_counts(
    rollups: List[Dict[str, Any]],
    news_counts: Dict[str, int],
    news_article_ids: Dict[str, List[Any]] | None = None,
) -> List[Dict[str, Any]]:
    """Merges news_counts into existing Reddit rollups and appends
    news-only rows (a ticker with zero Reddit mentions but real news
    coverage still ranks - item 1's stated requirement). News-only rows
    carry zeroed Reddit-specific fields and a weighted_score of 0 (the
    score formula is Reddit-only by design; a news-only ticker still shows
    up, just not ranked by a score that isn't measuring it). SEC-4:
    news_article_ids stores up to 10 contributing rss_articles ids per
    ticker so the drawer can link the actual articles."""
    news_article_ids = news_article_ids or {}

    def _top_news(ticker: str) -> str:
        ids = [i for i in news_article_ids.get(ticker, []) if i is not None][:10]
        return json.dumps(ids)

    by_ticker = {row["ticker"]: row for row in rollups}
    for ticker, count in news_counts.items():
        if ticker in by_ticker:
            row = by_ticker[ticker]
            row["news_count"] = count
            row["total_mention_count"] = row["reddit_count"] + count
            row["top_news_ids"] = _top_news(ticker)
        else:
            by_ticker[ticker] = {
                "ticker": ticker,
                "company": ticker_resolver.ticker_title(ticker),
                "mention_count": 0,
                "reddit_count": 0,
                "news_count": count,
                "total_mention_count": count,
                "weighted_mention_count": 0.0,
                "source_count": 0,
                "subreddit_count": 0,
                "weighted_score": 0.0,
                "mood": "neutral",
                "top_source_ids": "[]",
                "quality_flags": "[]",
                "top_news_ids": _top_news(ticker),
            }
    merged = list(by_ticker.values())
    merged.sort(key=lambda row: (-row["weighted_score"], -row["total_mention_count"], row["ticker"]))
    return merged


def compute_divergence(rank: int, price_pct: Optional[float]) -> str:
    """Item 2's v1 divergence heuristic - simple and explicitly documented
    as provisional (see module docstring). rank is 1-indexed by
    total_mention_count among the day's rolled-up tickers; price_pct is
    the day's close-over-close % change. Returns '' when there's nothing
    notable or no price data."""
    if price_pct is None:
        return ""
    if rank <= ATTENTION_SPIKE_RANK_THRESHOLD and abs(price_pct) < SMALL_PRICE_MOVE_PCT:
        return "attention_spike_no_price_move"
    if rank > ATTENTION_SPIKE_RANK_THRESHOLD and abs(price_pct) >= LARGE_PRICE_MOVE_PCT:
        return "price_move_no_attention"
    return ""


def merge_market_context(rollups: List[Dict[str, Any]], market_by_ticker: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attaches price/volume columns and the divergence flag. rollups must
    already be ranked (merge_news_counts/aggregate_rows both sort by
    attention) - rank is read from list position, 1-indexed."""
    for i, row in enumerate(rollups):
        context = market_by_ticker.get(row["ticker"])
        if context is None:
            row["price_close"] = None
            row["price_pct"] = None
            row["volume"] = None
            row["volume_vs_20d"] = None
            row["divergence"] = ""
            continue
        row["price_close"] = context.get("price_close")
        row["price_pct"] = context.get("price_pct")
        row["volume"] = context.get("volume")
        row["volume_vs_20d"] = context.get("volume_vs_20d")
        row["divergence"] = compute_divergence(i + 1, context.get("price_pct"))
    return rollups


def _run(
    target_day: date,
    dry_run: bool,
    retention_days: int,
    skip_news: bool = False,
    skip_market: bool = False,
) -> Dict[str, Any]:
    import psycopg2.extras

    day_start, day_end = day_bounds(target_day)
    prev_start, prev_end = day_bounds(target_day - timedelta(days=1))
    summary: Dict[str, Any] = {
        "ok": True,
        "date": target_day.isoformat(),
        "dry_run": dry_run,
        "config_source": "defaults",
        "mention_rows_seen": 0,
        "news_articles_scanned": 0,
        "news_only_tickers": 0,
        "market_context_fetched": 0,
        "market_context_failed": 0,
        "author_stats_computed": 0,
        "authors_discounted": 0,
        "flagged_tickers": 0,
        "review_queue_added": 0,
        "tickers": 0,
        "rows_written": 0,
        "retention": {"skipped": True},
        "ran_at": _utc_now_iso(),
    }

    # Item 4: admin-managed config - subreddit weights for scoring, symbol
    # overrides for the news-channel resolver pass, author-weight params.
    config = neon_feeds.get_attention_sweep_config()
    if config:
        summary["config_source"] = "db"
    subreddit_weights: Dict[str, float] = {}
    for entry in (config or {}).get("subreddits", []):
        if isinstance(entry, dict) and str(entry.get("name", "") or "").strip():
            try:
                subreddit_weights[str(entry["name"]).strip()] = float(entry.get("weight", 1.0) or 1.0)
            except (TypeError, ValueError):
                continue
    overrides = (config or {}).get("symbol_overrides") or {}
    ticker_resolver.set_runtime_overrides(
        force_ambiguous=overrides.get("force_ambiguous") or [],
        force_unambiguous=overrides.get("force_unambiguous") or [],
    )

    neon_feeds._ensure_stock_attention_schema()
    with neon_feeds._get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(FETCH_DAY_ROWS_SQL, {"day_start": day_start, "day_end": day_end})
            reddit_rows = [dict(row) for row in cur.fetchall()]
            summary["mention_rows_seen"] = len(reddit_rows)

            # Item 5: recompute per-author stats over the full retained
            # window, derive credibility weights, persist the stats (write
            # gated on dry_run; the in-memory weights apply either way so a
            # dry run previews the weighted board faithfully).
            cur.execute(FETCH_AUTHOR_ITEM_STATS_SQL)
            item_stat_rows = [dict(row) for row in cur.fetchall()]
            cur.execute(FETCH_AUTHOR_TICKER_COUNTS_SQL)
            author_ticker_rows = [dict(row) for row in cur.fetchall()]
            cur.execute(FETCH_AUTHOR_SUBREDDIT_COUNTS_SQL)
            author_subreddit_rows = [dict(row) for row in cur.fetchall()]
            author_stats = compute_author_stats(item_stat_rows, author_ticker_rows, author_subreddit_rows)
            author_weights = build_author_weights(author_stats, (config or {}).get("author_weighting"))
            summary["author_stats_computed"] = len(author_stats)
            summary["authors_discounted"] = len(author_weights)

            rollups = aggregate_rows(reddit_rows, author_weights=author_weights, subreddit_weights=subreddit_weights)
            reddit_ticker_count = len(rollups)

            if not skip_news:
                cur.execute(FETCH_NEWS_ROWS_SQL, {"day_start": day_start, "day_end": day_end})
                articles = [dict(row) for row in cur.fetchall()]
                summary["news_articles_scanned"] = len(articles)
                news_ids = compute_news_ticker_mentions(articles)
                news_counts = {ticker: len(ids) for ticker, ids in news_ids.items()}
                rollups = merge_news_counts(rollups, news_counts, news_ids)
                summary["news_only_tickers"] = len(rollups) - reddit_ticker_count

            summary["tickers"] = len(rollups)

            if not skip_market and rollups:
                market_by_ticker = yahoo_market_data.fetch_market_context_batch([r["ticker"] for r in rollups])
                summary["market_context_fetched"] = len(market_by_ticker)
                summary["market_context_failed"] = len(rollups) - len(market_by_ticker)
                rollups = merge_market_context(rollups, market_by_ticker)

            # Item 6: quality flags per ticker - needs today's raw rows
            # grouped by ticker, yesterday's author sets (same query, prior
            # day bounds), and known account ages.
            cur.execute(FETCH_DAY_ROWS_SQL, {"day_start": prev_start, "day_end": prev_end})
            yesterday_authors_by_ticker: Dict[str, Set[str]] = {}
            for row in cur.fetchall():
                yesterday_authors_by_ticker.setdefault(str(row["ticker"]), set()).add(str(row["author"]))
            cur.execute(FETCH_KNOWN_ACCOUNT_AGES_SQL)
            account_ages = {str(row["author"]): row["account_created"] for row in cur.fetchall()}

            today_rows_by_ticker: Dict[str, List[Dict[str, Any]]] = {}
            for row in reddit_rows:
                today_rows_by_ticker.setdefault(str(row["ticker"]), []).append(row)
            now = datetime.now(UTC)
            for rollup in rollups:
                ticker_rows = today_rows_by_ticker.get(rollup["ticker"], [])
                if ticker_rows:
                    flags = compute_quality_flags(
                        ticker_rows,
                        yesterday_authors_by_ticker.get(rollup["ticker"], set()),
                        account_ages,
                        now,
                    )
                    rollup["quality_flags"] = json.dumps(flags)
            summary["flagged_tickers"] = sum(1 for r in rollups if r.get("quality_flags", "[]") != "[]")

            # Item 6: review queue - tickers entering the top of the board
            # that have never appeared in any prior day's rollup.
            cur.execute(FETCH_SEEN_TICKERS_SQL, {"day": target_day})
            seen_tickers = {str(row["ticker"]) for row in cur.fetchall()}
            newcomers = [
                r for r in rollups[:REVIEW_QUEUE_TOP_N] if r["ticker"] not in seen_tickers
            ]

            summary["top_tickers"] = [
                {
                    "ticker": r["ticker"],
                    "mentions": r["total_mention_count"],
                    "weighted": r.get("weighted_mention_count", 0),
                    "reddit": r["reddit_count"],
                    "news": r["news_count"],
                    "score": r["weighted_score"],
                    "price_pct": r.get("price_pct"),
                    "divergence": r.get("divergence", ""),
                    "flags": json.loads(r.get("quality_flags", "[]") or "[]"),
                }
                for r in rollups[:15]
            ]

            if not dry_run:
                # Replace the date wholesale so re-runs after resolver
                # tuning or a market-data hiccup also remove tickers that
                # no longer resolve and refresh price/volume columns.
                cur.execute(DELETE_DAY_SQL, {"day": target_day})
                if rollups:
                    psycopg2.extras.execute_values(
                        cur,
                        INSERT_DAY_SQL,
                        [
                            (
                                target_day,
                                r["ticker"],
                                r["company"],
                                r["mention_count"],
                                r["source_count"],
                                r["subreddit_count"],
                                r["weighted_score"],
                                r["mood"],
                                r["top_source_ids"],
                                r["reddit_count"],
                                r["news_count"],
                                r["total_mention_count"],
                                r.get("price_close"),
                                r.get("price_pct"),
                                r.get("volume"),
                                r.get("volume_vs_20d"),
                                r.get("divergence", ""),
                                r.get("weighted_mention_count", 0.0),
                                r.get("quality_flags", "[]"),
                                r.get("top_news_ids", "[]"),
                            )
                            for r in rollups
                        ],
                    )
                summary["rows_written"] = len(rollups)

                if newcomers:
                    psycopg2.extras.execute_values(
                        cur,
                        INSERT_REVIEW_QUEUE_SQL,
                        [(target_day, r["ticker"], r["top_source_ids"]) for r in newcomers],
                    )
                    summary["review_queue_added"] = len(newcomers)

                # Persist the item-5 author stats (uses its own connection
                # via neon_feeds; account_created/link_karma untouched).
                neon_feeds.upsert_author_stats_batch(author_stats)

                cutoff = datetime.now(UTC) - timedelta(days=retention_days)
                cur.execute(RETENTION_DELETE_MENTIONS_VIA_ITEMS_SQL, {"cutoff": cutoff})
                mentions_deleted = cur.rowcount
                cur.execute(RETENTION_DELETE_ORPHAN_MENTIONS_SQL, {"cutoff": cutoff})
                mentions_deleted += cur.rowcount
                cur.execute(RETENTION_DELETE_ITEMS_SQL, {"cutoff": cutoff})
                summary["retention"] = {
                    "skipped": False,
                    "retention_days": retention_days,
                    "mentions_deleted": mentions_deleted,
                    "items_deleted": cur.rowcount,
                }
        if dry_run:
            conn.rollback()
    return summary


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="", help="UTC day to aggregate (YYYY-MM-DD); default: yesterday")
    parser.add_argument("--dry-run", action="store_true", help="Report the rollup; write and prune nothing")
    parser.add_argument(
        "--retention-days",
        type=int,
        default=int(os.environ.get("REDDIT_ATTENTION_RETENTION_DAYS", DEFAULT_RETENTION_DAYS)),
    )
    parser.add_argument("--skip-news", action="store_true", help="Skip the news-channel scan (item 1)")
    parser.add_argument("--skip-market", action="store_true", help="Skip the Yahoo market-context fetch (item 2)")
    parser.add_argument("--summary-path", default="")
    args = parser.parse_args(argv)

    try:
        if args.date:
            target_day = date.fromisoformat(args.date)
        else:
            target_day = (datetime.now(UTC) - timedelta(days=1)).date()
        if args.retention_days < 7:
            raise ValueError("--retention-days below 7 would delete data the drill-down UI still needs")
        summary = _run(
            target_day,
            dry_run=args.dry_run,
            retention_days=args.retention_days,
            skip_news=args.skip_news,
            skip_market=args.skip_market,
        )
    except Exception as exc:
        summary = {"ok": False, "error": str(exc), "ran_at": _utc_now_iso()}

    output = json.dumps(summary, indent=2, default=str)
    print(output)
    if args.summary_path:
        try:
            with open(args.summary_path, "w", encoding="utf-8") as handle:
                handle.write(output)
        except Exception as exc:
            print(f"[aggregate_stock_attention] could not write summary file: {exc}", file=sys.stderr)
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())

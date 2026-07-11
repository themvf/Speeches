#!/usr/bin/env python3
"""Daily rollup for the stock attention tracker
(docs/stock-attention-spec.md §6).

Runs at 00:15 UTC and aggregates the just-closed UTC day, bucketing by
reddit_attention_items.created_utc (when the post/comment was actually
written on Reddit) - NOT by the mention row's generated_at (when a sweep
happened to store it), which would misfile late-day items swept after
midnight.

Idempotent per date: rows for the target date are replaced wholesale in
one transaction (delete-then-insert), so a re-run after resolver/blacklist
tuning fully recomputes the day - including removing tickers that no
longer resolve - without re-sweeping Reddit.

Ends with a retention sweep (skipped on --dry-run): raw reddit items and
their mention rows older than the retention window are deleted. The
daily_stock_attention rollups persist indefinitely - they're tiny and
they're the product. This sweep exists because pruneOldRssData (neon.ts)
only prunes source_type='rss_article' mentions; reddit rows would
otherwise grow unbounded (spec §6.3).

Usage:
    python aggregate_stock_attention.py [--date YYYY-MM-DD] [--dry-run]
        [--retention-days N] [--summary-path PATH]

Required env vars: DATABASE_URL
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, date, datetime, timedelta
from typing import Any, Dict, List, Tuple

import neon_feeds
import ticker_resolver

DEFAULT_RETENTION_DAYS = 90

FETCH_DAY_ROWS_SQL = """
    SELECT m.value AS ticker, m.source_id, i.author, i.subreddit, i.score, i.mood
    FROM intelligence_mentions m
    JOIN reddit_attention_items i ON i.source_id = m.source_id
    WHERE m.mention_type = 'ticker'
      AND m.source_type IN ('reddit_post', 'reddit_comment')
      AND i.created_utc >= %(day_start)s
      AND i.created_utc < %(day_end)s
"""

DELETE_DAY_SQL = "DELETE FROM daily_stock_attention WHERE attention_date = %(day)s"

INSERT_DAY_SQL = """
    INSERT INTO daily_stock_attention
      (attention_date, ticker, company, mention_count, source_count, subreddit_count, weighted_score, mood, top_source_ids)
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


def compute_weighted_score(mention_count: int, subreddit_count: int, source_count: int) -> float:
    """Spec §6.2: deduped humans talking is the base signal; spread across
    communities amplifies most (harder to fake than volume inside one
    board); spread across threads amplifies mildly. No freshness decay in a
    daily rollup - that belongs in a future intraday view."""
    return round(
        mention_count
        * (1 + 0.15 * min(subreddit_count, 6))
        * (1 + 0.05 * min(source_count, 10)),
        4,
    )


def aggregate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pure aggregation of (ticker, source_id, author, subreddit, score,
    mood) fetch rows into per-ticker rollups. Each input row is one
    item+ticker pair (the mentions unique constraint guarantees that)."""
    by_ticker: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        ticker = str(row["ticker"])
        agg = by_ticker.setdefault(ticker, {
            "authors": set(),
            "sources": set(),
            "subreddits": set(),
            "moods": {"bullish": 0, "bearish": 0, "neutral": 0},
            "scored_sources": [],
        })
        agg["authors"].add(str(row["author"]))
        agg["sources"].add(str(row["source_id"]))
        agg["subreddits"].add(str(row["subreddit"]))
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
        mention_count = len(agg["authors"])
        source_count = len(agg["sources"])
        subreddit_count = len(agg["subreddits"])
        out.append({
            "ticker": ticker,
            "company": ticker_resolver.ticker_title(ticker),
            "mention_count": mention_count,
            "source_count": source_count,
            "subreddit_count": subreddit_count,
            "weighted_score": compute_weighted_score(mention_count, subreddit_count, source_count),
            "mood": mood,
            "top_source_ids": json.dumps(top_sources),
        })
    out.sort(key=lambda row: (-row["weighted_score"], row["ticker"]))
    return out


def _run(target_day: date, dry_run: bool, retention_days: int) -> Dict[str, Any]:
    import psycopg2.extras

    day_start, day_end = day_bounds(target_day)
    summary: Dict[str, Any] = {
        "ok": True,
        "date": target_day.isoformat(),
        "dry_run": dry_run,
        "mention_rows_seen": 0,
        "tickers": 0,
        "rows_written": 0,
        "retention": {"skipped": True},
        "ran_at": _utc_now_iso(),
    }

    neon_feeds._ensure_stock_attention_schema()
    with neon_feeds._get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(FETCH_DAY_ROWS_SQL, {"day_start": day_start, "day_end": day_end})
            rows = [dict(row) for row in cur.fetchall()]
            summary["mention_rows_seen"] = len(rows)

            rollups = aggregate_rows(rows)
            summary["tickers"] = len(rollups)
            summary["top_tickers"] = [
                {"ticker": r["ticker"], "mentions": r["mention_count"], "score": r["weighted_score"]}
                for r in rollups[:15]
            ]

            if not dry_run:
                # Replace the date wholesale so re-runs after resolver
                # tuning also remove tickers that no longer resolve.
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
                            )
                            for r in rollups
                        ],
                    )
                summary["rows_written"] = len(rollups)

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
    parser.add_argument("--summary-path", default="")
    args = parser.parse_args(argv)

    try:
        if args.date:
            target_day = date.fromisoformat(args.date)
        else:
            target_day = (datetime.now(UTC) - timedelta(days=1)).date()
        if args.retention_days < 7:
            raise ValueError("--retention-days below 7 would delete data the drill-down UI still needs")
        summary = _run(target_day, dry_run=args.dry_run, retention_days=args.retention_days)
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

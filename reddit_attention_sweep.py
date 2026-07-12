#!/usr/bin/env python3
"""Hourly Reddit sweep for the stock attention tracker
(docs/stock-attention-spec.md §4).

Pulls new submissions plus top-level comments on hot threads from a
configured subreddit list, resolves ticker mentions via ticker_resolver
(three-tier gating, §5), and writes:

- reddit_attention_items  (item metadata: author/subreddit/created_utc/
                           permalink/mood - only items with >=1 ticker)
- intelligence_mentions   (one row per item+ticker,
                           mention_type='ticker', ON CONFLICT DO NOTHING)

Whole-subreddit sweep, NOT keyword search - this is a different mode than
reddit_scraper.py's RedditScraper (which searches r/all for regulatory
keywords and produces corpus documents). The two coexist.

Usage:
    python reddit_attention_sweep.py [--subreddits a,b,c] [--include-tier2]
        [--limit-new N] [--hot-threads N] [--dry-run] [--summary-path PATH]

Required env vars:
    DATABASE_URL, REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
    (GCS_* additionally required for source-health logging; health logging
    is best-effort and never fails the sweep.)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import UTC, datetime
from typing import Any, Dict, List

import entity_aliases
import neon_feeds
import ticker_resolver
from source_health import record_source_health

# In-code fallbacks only - the live source of truth is the admin-managed
# attention_sweep_config row (enhancement item 4), loaded fail-soft at run
# start. These apply when the config table is unreachable/absent.
TIER1_SUBREDDITS = ["wallstreetbets", "stocks", "investing", "StockMarket", "options", "Daytrading"]
TIER2_SUBREDDITS = ["pennystocks", "Shortsqueeze", "SqueezePlays", "smallstreetbets", "ValueInvesting", "dividends"]

# Known automation accounts (spec §4.3); extended by the admin config's
# bot_blocklist and REDDIT_ATTENTION_BOT_BLOCKLIST env. Compared lowercase.
DEFAULT_BOT_AUTHORS = {"automoderator", "visualmod", "wsbvotebot", "flairhelperbot"}

SOURCE_KEY = "reddit_attention_sweep"

# Item 5: per-sweep budget for PRAW account-age lookups (1 API call each) -
# opportunistic enrichment of reddit_author_stats, never a required step.
# Split between the visible leaderboard (top-by-items_total authors still
# missing an age) and a reserve for fresh/current-sweep authors that the
# young-account manipulation flag needs but that never rank high enough to
# be reached by the board pass alone.
ACCOUNT_INFO_LOOKUPS_PER_SWEEP = 25
ACCOUNT_INFO_RECENT_RESERVE = 8

# Same lightweight keyword-heuristic pattern as RSS tone (inferToneLabel /
# _heuristic_enrichment) - deliberately not a fourth sentiment system and
# not an LLM call (spec §6.2). Directional at best on sarcasm-heavy WSB
# text; the UI renders it de-emphasized.
_BULLISH_RE = re.compile(
    r"\b(calls?|long|moon(?:ing|shot)?|rockets?|buy(?:ing)?|bought|bull(?:ish)?|yolo(?:ed)?|tendies|squeeze|breakout|undervalued|rip(?:ping)?\s+up|to\s+the\s+moon)\b",
    re.IGNORECASE,
)
_BEARISH_RE = re.compile(
    r"\b(puts?|short(?:ing|ed)?|sell(?:ing)?|sold|bear(?:ish)?|crash(?:ing)?|dump(?:ing)?|drill(?:ing)?|tank(?:ing|ed)?|overvalued|baghold(?:er|ing)?|rug(?:ged|pull)?)\b",
    re.IGNORECASE,
)


def infer_reddit_mood(text: str) -> str:
    raw = str(text or "")
    bullish = len(_BULLISH_RE.findall(raw))
    bearish = len(_BEARISH_RE.findall(raw))
    if bullish > bearish:
        return "bullish"
    if bearish > bullish:
        return "bearish"
    return "neutral"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _bot_authors() -> set:
    extra = {
        name.strip().lower()
        for name in os.environ.get("REDDIT_ATTENTION_BOT_BLOCKLIST", "").split(",")
        if name.strip()
    }
    return DEFAULT_BOT_AUTHORS | extra


def _build_reddit():
    """Constructs the PRAW client. Isolated so tests can patch it."""
    import praw

    client_id = os.environ.get("REDDIT_CLIENT_ID", "").strip()
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "").strip()
    user_agent = os.environ.get("REDDIT_USER_AGENT", "").strip() or "PolicyResearchHub/1.0"
    if not client_id or not client_secret:
        raise RuntimeError("REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET are not set")
    return praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)


def _created_dt(created_utc: Any) -> datetime:
    return datetime.fromtimestamp(float(created_utc), tz=UTC)


def _author_name(item: Any) -> str:
    author = getattr(item, "author", None)
    return str(author) if author else "[deleted]"


def _permalink(item: Any) -> str:
    path = str(getattr(item, "permalink", "") or "")
    return f"https://www.reddit.com{path}" if path.startswith("/") else path


def _process_text_item(
    *,
    fullname: str,
    kind: str,
    subreddit: str,
    author: str,
    title: str,
    text: str,
    created_utc: Any,
    score: Any,
    permalink: str,
    items_out: List[Dict[str, Any]],
    mentions_out: List[Dict[str, Any]],
) -> bool:
    """Resolve tickers for one post/comment; append storage rows when any
    are found. Returns True if the item produced mentions."""
    resolved = ticker_resolver.resolve_tickers(text)
    if not resolved:
        return False

    items_out.append({
        "source_id": fullname,
        "kind": kind,
        "subreddit": subreddit,
        "author": author,
        "title": title,
        "permalink": permalink,
        "created_utc": _created_dt(created_utc),
        "score": int(score or 0),
        "mood": infer_reddit_mood(text),
    })
    source_type = "reddit_post" if kind == "post" else "reddit_comment"
    for symbol, confidence in resolved.items():
        mentions_out.append({
            "source_type": source_type,
            "source_id": fullname,
            "mention_type": "ticker",
            "value": symbol,
            "normalized_value": entity_aliases.normalize_mention_value(symbol),
            "confidence": confidence,
        })
    return True


def sweep_subreddit(
    reddit: Any,
    subreddit_name: str,
    *,
    limit_new: int,
    hot_threads: int,
    bot_authors: set,
    items_out: List[Dict[str, Any]],
    mentions_out: List[Dict[str, Any]],
) -> Dict[str, int]:
    """Sweeps one subreddit; appends rows to the shared output lists."""
    stats = {"posts_scanned": 0, "comments_scanned": 0, "items_with_tickers": 0}
    subreddit = reddit.subreddit(subreddit_name)

    # New submissions: title + selftext.
    for submission in subreddit.new(limit=limit_new):
        stats["posts_scanned"] += 1
        author = _author_name(submission)
        if author.lower() in bot_authors:
            continue
        text = f"{submission.title}\n{getattr(submission, 'selftext', '') or ''}"
        if _process_text_item(
            fullname=str(submission.fullname),
            kind="post",
            subreddit=subreddit_name,
            author=author,
            title=str(submission.title or ""),
            text=text,
            created_utc=submission.created_utc,
            score=getattr(submission, "score", 0),
            permalink=_permalink(submission),
            items_out=items_out,
            mentions_out=mentions_out,
        ):
            stats["items_with_tickers"] += 1

    # Top-level comments on current hot threads (where most ticker chat
    # lives, e.g. WSB's daily discussion thread). replace_more(limit=0)
    # drops unexpanded MoreComments stubs instead of fetching them - without
    # it, iteration silently multiplies the API call budget (spec §4.3).
    for submission in subreddit.hot(limit=hot_threads):
        submission.comments.replace_more(limit=0)
        parent_title = str(submission.title or "")
        for comment in submission.comments:
            stats["comments_scanned"] += 1
            author = _author_name(comment)
            if author.lower() in bot_authors:
                continue
            if _process_text_item(
                fullname=str(comment.fullname),
                kind="comment",
                subreddit=subreddit_name,
                author=author,
                title=parent_title,
                text=str(getattr(comment, "body", "") or ""),
                created_utc=comment.created_utc,
                score=getattr(comment, "score", 0),
                permalink=_permalink(comment),
                items_out=items_out,
                mentions_out=mentions_out,
            ):
                stats["items_with_tickers"] += 1

    return stats


def _enrich_author_account_info(reddit: Any, authors: List[str], summary: Dict[str, Any]) -> None:
    """Item 5: opportunistic, budget-capped PRAW lookups for authors whose
    account age we don't know yet. Every failure is per-author and
    non-fatal - a suspended/deleted account just stays unknown."""
    try:
        targets = neon_feeds.get_authors_needing_account_info(
            authors,
            board_budget=ACCOUNT_INFO_LOOKUPS_PER_SWEEP - ACCOUNT_INFO_RECENT_RESERVE,
            recent_budget=ACCOUNT_INFO_RECENT_RESERVE,
        )
    except Exception as exc:
        summary["author_info_error"] = f"author-selection query failed: {exc}"
        return
    rows: List[Dict[str, Any]] = []
    for name in targets[:ACCOUNT_INFO_LOOKUPS_PER_SWEEP]:
        try:
            redditor = reddit.redditor(name)
            created = getattr(redditor, "created_utc", None)
            rows.append({
                "author": name,
                "account_created": _created_dt(created) if created else None,
                "link_karma": int(getattr(redditor, "link_karma", 0) or 0),
            })
        except Exception:
            continue
    if rows:
        try:
            summary["author_info_enriched"] = neon_feeds.upsert_author_account_info(rows)
        except Exception as exc:
            summary["author_info_error"] = f"account-info upsert failed: {exc}"


def _resolve_subreddits(args: argparse.Namespace, config: Dict[str, Any] | None) -> List[str]:
    """Precedence: explicit --subreddits > admin config > in-code tier
    defaults (with --include-tier2)."""
    if args.subreddits:
        return [name.strip() for name in args.subreddits.split(",") if name.strip()]
    if config and isinstance(config.get("subreddits"), list):
        names = [
            str(entry.get("name", "") or "").strip()
            for entry in config["subreddits"]
            if isinstance(entry, dict) and entry.get("active", True)
        ]
        names = [n for n in names if n]
        if names:
            return names
    subreddits = list(TIER1_SUBREDDITS)
    if args.include_tier2:
        subreddits += TIER2_SUBREDDITS
    return subreddits


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    config = neon_feeds.get_attention_sweep_config()
    subreddits = _resolve_subreddits(args, config)

    overrides = (config or {}).get("symbol_overrides") or {}
    ticker_resolver.set_runtime_overrides(
        force_ambiguous=overrides.get("force_ambiguous") or [],
        force_unambiguous=overrides.get("force_unambiguous") or [],
    )

    summary: Dict[str, Any] = {
        "source_key": SOURCE_KEY,
        "connector": SOURCE_KEY,
        "mode": "reddit_attention_sweep",
        "dry_run": args.dry_run,
        "config_source": "db" if config else "defaults",
        "subreddits": subreddits,
        "posts_scanned": 0,
        "comments_scanned": 0,
        "items_with_tickers": 0,
        "item_rows_written": 0,
        "mention_rows_written": 0,
        "unique_tickers": 0,
        "errors": [],
        "ran_at": _utc_now_iso(),
    }

    reddit = _build_reddit()
    bot_authors = _bot_authors()
    if config and isinstance(config.get("bot_blocklist"), list):
        bot_authors = bot_authors | {str(name or "").strip().lower() for name in config["bot_blocklist"] if str(name or "").strip()}
    items: List[Dict[str, Any]] = []
    mentions: List[Dict[str, Any]] = []

    for name in subreddits:
        try:
            stats = sweep_subreddit(
                reddit,
                name,
                limit_new=args.limit_new,
                hot_threads=args.hot_threads,
                bot_authors=bot_authors,
                items_out=items,
                mentions_out=mentions,
            )
            summary["posts_scanned"] += stats["posts_scanned"]
            summary["comments_scanned"] += stats["comments_scanned"]
            summary["items_with_tickers"] += stats["items_with_tickers"]
        except Exception as exc:
            summary["errors"].append(f"r/{name}: {exc}")

    summary["unique_tickers"] = len({mention["value"] for mention in mentions})

    if not args.dry_run and items:
        summary["item_rows_written"] = neon_feeds.upsert_reddit_attention_items(items)
        summary["mention_rows_written"] = neon_feeds.insert_ticker_mentions(mentions)
        _enrich_author_account_info(reddit, [item["author"] for item in items], summary)

    # Health-monitor mapping: every subreddit failing (no rows, only
    # errors) surfaces in failing_sources; partial failures still record
    # the error sample.
    summary["discovered_count"] = summary["posts_scanned"] + summary["comments_scanned"]
    summary["processed_count"] = summary["items_with_tickers"]
    summary["failed_count"] = len(summary["errors"])
    summary["ok"] = len(summary["errors"]) < len(subreddits)
    return summary


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subreddits", default=os.environ.get("REDDIT_ATTENTION_SUBREDDITS", ""),
                        help="Comma-separated override of the tiered default list")
    parser.add_argument("--include-tier2", action="store_true",
                        default=os.environ.get("REDDIT_ATTENTION_INCLUDE_TIER2", "").lower() == "true")
    parser.add_argument("--limit-new", type=int, default=100)
    parser.add_argument("--hot-threads", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true", help="Sweep and resolve but write nothing to Neon")
    parser.add_argument("--summary-path", default="")
    args = parser.parse_args(argv)

    try:
        summary = _run(args)
    except Exception as exc:
        summary = {
            "ok": False,
            "source_key": SOURCE_KEY,
            "connector": SOURCE_KEY,
            "mode": "reddit_attention_sweep",
            "errors": [str(exc)],
            "failed_count": 1,
            "ran_at": _utc_now_iso(),
        }

    record_source_health(summary)

    output = json.dumps(summary, indent=2, default=str)
    print(output)
    if args.summary_path:
        try:
            with open(args.summary_path, "w", encoding="utf-8") as handle:
                handle.write(output)
        except Exception as exc:
            print(f"[reddit_attention_sweep] could not write summary file: {exc}", file=sys.stderr)
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())

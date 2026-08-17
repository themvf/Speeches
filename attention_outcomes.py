#!/usr/bin/env python3
"""Forward-return scoring for Reddit stock attention (enhancement 2).

The attention tracker has always been descriptive: it says what is loud, never
whether loud mattered. This adds the evaluative half - for each (attention
day, ticker) it records what Reddit's mood was, then once enough trading days
have passed it measures what the stock actually did and whether the mood was
directionally right. Those resolved outcomes roll up into hit rates per
subreddit and per author, which is what turns the Authors board from a list of
prolific posters into a list of accurate ones.

Deliberately deterministic end to end - no model calls anywhere. Mood already
comes from the sweep's keyword classifier; this only compares it to price.

Design mirrors the Polymarket sharp-wallet pipeline, for the same reason:

  * attention_outcomes is DURABLE and one compact row per (date, ticker). It
    captures the contributing subreddits and authors AT SEED TIME, because
    reddit_attention_items is pruned at 90 days and the attribution has to
    outlive it. Seeding is therefore not optional bookkeeping - it is the
    only chance to record who said it.
  * attention_source_stats is recomputed from that durable table, never from
    raw items, so pruning is invisible to the product.

Horizons are in TRADING days, walked over the actual close series, so weekends
and holidays do not silently shorten a window.

Usage:
    python attention_outcomes.py [--date YYYY-MM-DD] [--seed-days N]
        [--dry-run] [--summary-path PATH]

Required env vars: DATABASE_URL
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import neon_feeds
import yahoo_market_data

# Trading-day horizons scored for every outcome row.
HORIZONS = (1, 5, 20)

# How far back to look for attention days that still need an outcome row.
# Comfortably inside the 90-day item retention that attribution depends on,
# with room for the job to be down for a while without losing history.
DEFAULT_SEED_DAYS = 45

# A move smaller than this is treated as "no call to grade" rather than a win
# or a loss. Without it, mood would be graded against noise: a 0.03% drift is
# not evidence that a bullish crowd was right.
FLAT_MOVE_PCT = 1.0

# Moods that express a direction. neutral/mixed rows are still stored (they
# are useful context and they carry attribution) but never scored.
DIRECTIONAL_MOODS = ("bullish", "bearish")


def forward_return_pct(
    closes_by_date: Dict[str, float],
    attention_day: date,
    horizon_trading_days: int,
) -> Optional[float]:
    """Percent change from the attention day's close to the close
    `horizon_trading_days` TRADING days later.

    Trading days, not calendar days: the walk steps through the dates the
    series actually contains, so a Friday attention day with a 1-day horizon
    resolves to Monday rather than to a non-existent Saturday bar.

    The baseline is the last close on or before the attention day - Reddit
    chatter on a Saturday is graded against Friday's close, which is the last
    price anyone reading it could have acted on. Returns None when the
    baseline is missing or the horizon has not elapsed yet.
    """
    if horizon_trading_days < 1 or not closes_by_date:
        return None
    day_str = attention_day.isoformat()
    ordered = sorted(closes_by_date)

    baseline_idx = None
    for idx, d in enumerate(ordered):
        if d <= day_str:
            baseline_idx = idx
        else:
            break
    if baseline_idx is None:
        return None

    target_idx = baseline_idx + horizon_trading_days
    if target_idx >= len(ordered):
        return None

    baseline = closes_by_date[ordered[baseline_idx]]
    target = closes_by_date[ordered[target_idx]]
    if not baseline:
        return None
    return round(((target - baseline) / baseline) * 100, 4)


def grade_direction(mood: str, return_pct: Optional[float]) -> Optional[bool]:
    """True/False when the mood made a gradeable directional call, else None.

    None means "not scored", and it covers three distinct cases that must not
    be silently counted as losses: a non-directional mood, an unresolved
    horizon, and a move too small to be evidence either way.
    """
    if return_pct is None:
        return None
    if mood not in DIRECTIONAL_MOODS:
        return None
    if abs(return_pct) < FLAT_MOVE_PCT:
        return None
    return (return_pct > 0) if mood == "bullish" else (return_pct < 0)


def summarize_hit_rates(
    outcomes: Iterable[Dict[str, Any]],
    kind: str,
    min_scored: int = 5,
) -> List[Dict[str, Any]]:
    """Per-subreddit or per-author hit rates from durable outcome rows.

    `kind` selects which attribution list to fan out over ("subreddit" or
    "author"); a row contributes its grade to every key it names, because the
    row records that they all talked about that ticker that day.

    min_scored gates output: a 1-for-1 author is not a 100% forecaster, and
    publishing them as one is how a leaderboard becomes noise.
    """
    field = "subreddits" if kind == "subreddit" else "authors"
    buckets: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"scored": {h: 0 for h in HORIZONS}, "correct": {h: 0 for h in HORIZONS}, "rows": 0}
    )

    for row in outcomes:
        keys = row.get(field) or []
        if isinstance(keys, str):
            try:
                keys = json.loads(keys)
            except (TypeError, ValueError):
                keys = []
        mood = str(row.get("mood") or "")
        for key in {str(k) for k in keys if str(k).strip()}:
            bucket = buckets[key]
            bucket["rows"] += 1
            for horizon in HORIZONS:
                grade = grade_direction(mood, row.get(f"fwd_{horizon}d_pct"))
                if grade is None:
                    continue
                bucket["scored"][horizon] += 1
                if grade:
                    bucket["correct"][horizon] += 1

    out: List[Dict[str, Any]] = []
    for key, bucket in buckets.items():
        # Gate on the primary horizon: a key with plenty of 1d grades but few
        # 20d ones should still appear, with the thin horizon left null.
        if bucket["scored"][HORIZONS[0]] < min_scored:
            continue
        record: Dict[str, Any] = {"kind": kind, "key": key, "rows_total": bucket["rows"]}
        for horizon in HORIZONS:
            scored = bucket["scored"][horizon]
            correct = bucket["correct"][horizon]
            record[f"scored_{horizon}d"] = scored
            record[f"correct_{horizon}d"] = correct
            record[f"hit_rate_{horizon}d"] = round(correct / scored, 4) if scored else None
        out.append(record)

    out.sort(key=lambda r: (-(r.get(f"hit_rate_{HORIZONS[0]}d") or 0), -r["rows_total"], r["key"]))
    return out


def _needs_resolution(row: Dict[str, Any]) -> bool:
    return any(row.get(f"fwd_{h}d_pct") is None for h in HORIZONS)


def _has_elapsed(attention_day: date, horizon: int, today: date) -> bool:
    """Cheap calendar guard so the job does not fetch prices for a horizon
    that cannot possibly have resolved. Deliberately generous - the real
    check is the trading-day walk in forward_return_pct; this only avoids
    pointless network calls."""
    return (today - attention_day).days >= horizon + 2


def resolve_outcome_row(
    row: Dict[str, Any],
    closes_by_date: Dict[str, float],
    today: Optional[date] = None,
) -> Dict[str, Any]:
    """Fill whatever horizons have resolved. Pure: no I/O, no mutation of the
    input - the caller decides what to persist."""
    today = today or datetime.now(UTC).date()
    attention_day = row["attention_date"]
    if isinstance(attention_day, str):
        attention_day = date.fromisoformat(attention_day)

    updated = dict(row)
    for horizon in HORIZONS:
        key = f"fwd_{horizon}d_pct"
        if updated.get(key) is not None:
            continue
        if not _has_elapsed(attention_day, horizon, today):
            continue
        value = forward_return_pct(closes_by_date, attention_day, horizon)
        if value is not None:
            updated[key] = value
            updated[f"correct_{horizon}d"] = grade_direction(str(updated.get("mood") or ""), value)
    return updated


def _run(
    target_date: Optional[date],
    seed_days: int,
    dry_run: bool,
) -> Dict[str, Any]:
    today = datetime.now(UTC).date()
    summary: Dict[str, Any] = {
        "ok": True,
        "ran_at": datetime.now(UTC).isoformat(),
        "dry_run": dry_run,
        "seeded": 0,
        "resolved": 0,
        "price_fetch_failures": 0,
        "subreddit_stats": 0,
        "author_stats": 0,
        "errors": [],
    }

    seed_from = target_date or (today - timedelta(days=seed_days))
    seed_to = target_date or today

    try:
        pending = neon_feeds.get_attention_days_missing_outcomes(seed_from, seed_to)
    except Exception as exc:
        summary["ok"] = False
        summary["errors"].append(f"seed query failed: {exc}")
        return summary

    if pending and not dry_run:
        try:
            neon_feeds.upsert_attention_outcomes(pending)
        except Exception as exc:
            summary["ok"] = False
            summary["errors"].append(f"seed write failed: {exc}")
    summary["seeded"] = len(pending)

    try:
        unresolved = neon_feeds.get_unresolved_attention_outcomes(HORIZONS[-1])
    except Exception as exc:
        summary["ok"] = False
        summary["errors"].append(f"unresolved query failed: {exc}")
        return summary

    by_ticker: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in unresolved:
        if _needs_resolution(row):
            by_ticker[str(row["ticker"])].append(row)

    updates: List[Dict[str, Any]] = []
    for ticker, rows in by_ticker.items():
        closes = yahoo_market_data.fetch_close_series(ticker, "6mo")
        if not closes:
            summary["price_fetch_failures"] += 1
            continue
        for row in rows:
            resolved = resolve_outcome_row(row, closes, today)
            if any(resolved.get(f"fwd_{h}d_pct") != row.get(f"fwd_{h}d_pct") for h in HORIZONS):
                updates.append(resolved)

    if updates and not dry_run:
        try:
            neon_feeds.update_attention_outcome_returns(updates)
        except Exception as exc:
            summary["ok"] = False
            summary["errors"].append(f"resolve write failed: {exc}")
    summary["resolved"] = len(updates)

    try:
        resolved_rows = neon_feeds.get_resolved_attention_outcomes()
        subreddit_stats = summarize_hit_rates(resolved_rows, "subreddit")
        author_stats = summarize_hit_rates(resolved_rows, "author")
        if not dry_run:
            neon_feeds.replace_attention_source_stats(subreddit_stats + author_stats)
        summary["subreddit_stats"] = len(subreddit_stats)
        summary["author_stats"] = len(author_stats)
        summary["top_subreddits"] = subreddit_stats[:5]
    except Exception as exc:
        summary["ok"] = False
        summary["errors"].append(f"stats recompute failed: {exc}")

    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Score Reddit attention against forward returns.")
    parser.add_argument("--date", help="Seed only this attention date (YYYY-MM-DD).")
    parser.add_argument("--seed-days", type=int, default=DEFAULT_SEED_DAYS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-path")
    args = parser.parse_args(argv)

    if not os.environ.get("DATABASE_URL"):
        print(json.dumps({"ok": False, "error": "DATABASE_URL is not set."}))
        return 1

    target = date.fromisoformat(args.date) if args.date else None
    summary = _run(target, max(1, args.seed_days), args.dry_run)

    payload = json.dumps(summary, indent=2, default=str)
    print(payload)
    if args.summary_path:
        try:
            with open(args.summary_path, "w", encoding="utf-8") as fh:
                fh.write(payload)
        except OSError as exc:
            print(f"[attention_outcomes] could not write summary: {exc}", file=sys.stderr)
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

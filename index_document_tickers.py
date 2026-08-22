"""Index tracked tickers against the Neon `documents` corpus.

Why this exists: `intelligence_mentions` already carries ticker mentions, but
only for Reddit items (written by `reddit_attention_sweep.py`). Corpus
documents carry *entity* mentions instead, against an alias map covering ~40
mostly-regulator entities. So there is no way to ask "what does the corpus
know about NVDA" - this script builds that index.

    python index_document_tickers.py --backfill [--dry-run] [--limit N]
    python index_document_tickers.py                 # incremental, for cron

## A mention is not a subject

The resolver tells us a company was *named*, not that the document is *about*
it. A speech listing ten firms as examples would otherwise chip all ten as
loudly as an enforcement action filed against one. Since these chips render on
a public page beside a ticker, a wrong one reads as an accusation about a real
company - a different class of error from miscounting Reddit mentions.

Two defences, both here rather than in the UI:

1. Title and body are resolved separately, and where a match landed is encoded
   in the stored `confidence` (see TITLE_CONFIDENCE / BODY_CONFIDENCE). A
   company named in an enforcement action's title is almost certainly its
   subject; one named in paragraph 40 usually is not.
2. Body-only matches must be unambiguous - a cashtag or a gated bare symbol,
   never the 0.7 curated-company-name tier, which is the tier that fires on
   ordinary prose.

## What the first dry run changed

Run 32546850389 scanned 200 documents and produced 155 rows, and reading them
forced two corrections:

- The body tier was ~90% of rows and mostly junk. Financial and regulatory
  prose is dense with uppercase acronyms that are also real tickers: DAO
  (decentralized autonomous organization), RSI (relative strength index),
  ASIC (a chip type), WSE (Cerebras' wafer-scale engine), ASA (a Norwegian
  corporate suffix), ADV (Form ADV, matched inside an adviser enforcement
  action). It is now opt-in behind --include-body.
- The index covered all 9,304 symbols, but chips only ever render for tickers
  on the Movers or Industries boards. Everything outside that set was noise
  that could never be displayed and could only be wrong when it was. The
  index is now scoped to the tracked universe by default.

Even the title tier missed: "Hub Group, Inc. Investor Alert: Contact SBS by
August 28" resolved to SBS, a law firm's initials, for a document about HUBG.
Scoping to tracked tickers removes that particular one; the general case is
why accusatory source kinds stay title-only at render time.

Deliberately no new table and no new columns: reusing `intelligence_mentions`
keeps this clear of the deploy-order trap in CLAUDE.md, where a Vercel deploy
ships a reader before the Python migration that creates what it reads.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List

import neon_feeds
import ticker_resolver

# Where the match landed, encoded in the existing `confidence` column so this
# needs no schema change. The TS reader filters on these thresholds.
TITLE_CONFIDENCE = 1.0
BODY_CONFIDENCE = 0.6

# Body matches must come from the resolver's unambiguous tiers (cashtag, or a
# bare symbol that survived the ambiguous-word gate). The 0.7 company-name tier
# is what fires on regulatory prose, so it is title-only.
MIN_BODY_RESOLVER_CONFIDENCE = 1.0

# Long documents are truncated for the body pass. Testimony and transcripts run
# to hundreds of thousands of characters, and a company named once on page 90
# is not what these chips are for.
MAX_BODY_CHARS = 40_000

SOURCE_TYPE = "document"

# The boards a chip can render on. Anything outside this set is noise that can
# never be displayed, so indexing it only creates chances to be wrong.
INDUSTRY_CONFIG_PATH = os.path.join("apps", "web", "lib", "server", "industry-config.json")
MOVERS_ROUTE_PATH = os.path.join("apps", "web", "app", "api", "market", "movers", "route.ts")


def load_tracked_tickers() -> set:
    """Tickers that can appear on the Market page: the Industries universe plus
    the Movers watchlist.

    Fail-soft to an empty set, which the caller reads as "do not scope" and
    reports in the summary - a missing config should be visible, not silently
    turn the scope off or halt a run.
    """
    tracked: set = set()
    try:
        with io.open(INDUSTRY_CONFIG_PATH, encoding="utf-8") as handle:
            config = json.load(handle)
        for industry in config.get("industries", []):
            for member in industry.get("tickers", []):
                symbol = str(member.get("ticker", "") or "").strip().upper()
                if symbol:
                    tracked.add(symbol)
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not read {INDUSTRY_CONFIG_PATH}: {exc}", file=sys.stderr)

    try:
        with io.open(MOVERS_ROUTE_PATH, encoding="utf-8") as handle:
            tracked.update(re.findall(r'symbol: "([A-Z.\-]+)"', handle.read()))
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not read {MOVERS_ROUTE_PATH}: {exc}", file=sys.stderr)

    return tracked


def resolve_document(
    title: str,
    body: str,
    tracked: set | None = None,
    include_body: bool = False,
) -> Dict[str, float]:
    """Ticker mentions for one document, as {symbol: stored confidence}.

    A title match wins outright. The body tier is off unless asked for: the
    first dry run showed it was ~90% of rows and mostly acronyms that happen
    to be tickers.

    `tracked`, when given, restricts results to tickers that can actually
    appear on a board. An empty set means no scoping, so a failed config read
    degrades to the old behaviour rather than silently indexing nothing.
    """
    mentions: Dict[str, float] = {}

    for symbol in ticker_resolver.resolve_tickers(title or ""):
        mentions[symbol] = TITLE_CONFIDENCE

    if include_body:
        body_hits = ticker_resolver.resolve_tickers((body or "")[:MAX_BODY_CHARS])
        for symbol, resolver_confidence in body_hits.items():
            if symbol in mentions:
                continue
            if resolver_confidence >= MIN_BODY_RESOLVER_CONFIDENCE:
                mentions[symbol] = BODY_CONFIDENCE

    if tracked:
        mentions = {symbol: confidence for symbol, confidence in mentions.items() if symbol in tracked}

    return mentions


def build_mention_rows(
    document: Dict[str, Any],
    tracked: set | None = None,
    include_body: bool = False,
) -> List[Dict[str, Any]]:
    """Rows for `neon_feeds.insert_ticker_mentions`, one per resolved ticker."""
    document_id = str(document.get("document_id") or "").strip()
    if not document_id:
        return []
    resolved = resolve_document(
        str(document.get("title") or ""),
        str(document.get("full_text") or ""),
        tracked=tracked,
        include_body=include_body,
    )
    return [
        {
            "source_type": SOURCE_TYPE,
            "source_id": document_id,
            "mention_type": "ticker",
            "value": symbol,
            "normalized_value": symbol,
            "confidence": confidence,
        }
        for symbol, confidence in sorted(resolved.items())
    ]


def _apply_symbol_overrides() -> Dict[str, int]:
    """Reuse the admin-managed force-ambiguous list the Reddit sweep already
    honours, so a symbol that turns out to be a word is killed from the admin
    panel in one place rather than needing a deploy here.

    Fail-soft: an unreachable config leaves the resolver's built-in gating in
    place, which is the behaviour every run before this had anyway.
    """
    try:
        overrides = (neon_feeds.get_attention_sweep_config() or {}).get("symbol_overrides") or {}
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not load symbol overrides: {exc}", file=sys.stderr)
        return {"force_ambiguous": 0, "force_unambiguous": 0}

    force_ambiguous = overrides.get("force_ambiguous") or []
    force_unambiguous = overrides.get("force_unambiguous") or []
    ticker_resolver.set_runtime_overrides(
        force_ambiguous=force_ambiguous,
        force_unambiguous=force_unambiguous,
    )
    return {"force_ambiguous": len(force_ambiguous), "force_unambiguous": len(force_unambiguous)}


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    symbol_overrides = _apply_symbol_overrides()
    since = None
    if not args.backfill:
        since = (datetime.now(UTC) - timedelta(days=args.since_days)).isoformat()

    tracked = set() if args.no_scope else load_tracked_tickers()
    documents_scanned = 0
    documents_with_tickers = 0
    mention_rows = 0
    inserted = 0
    failed_batches: List[Dict[str, Any]] = []
    samples: List[Dict[str, Any]] = []

    for batch in neon_feeds.iter_documents_for_ticker_index(
        batch_size=args.batch_size, since=since, limit=args.limit
    ):
        rows: List[Dict[str, Any]] = []
        for document in batch:
            documents_scanned += 1
            document_rows = build_mention_rows(document, tracked=tracked, include_body=args.include_body)
            if not document_rows:
                continue
            documents_with_tickers += 1
            rows.extend(document_rows)
            if len(samples) < args.sample_size:
                samples.append({
                    "document_id": document["document_id"],
                    "title": str(document.get("title") or "")[:110],
                    "source_kind": document.get("source_kind"),
                    "tickers": {row["value"]: row["confidence"] for row in document_rows},
                })

        mention_rows += len(rows)
        if args.dry_run or not rows:
            continue
        try:
            inserted += neon_feeds.insert_ticker_mentions(rows)
        except Exception as exc:  # noqa: BLE001 - reported, never fatal mid-run
            failed_batches.append({"size": len(rows), "error": str(exc)})

    return {
        "ok": not failed_batches,
        "mode": "backfill" if args.backfill else "incremental",
        "dry_run": bool(args.dry_run),
        "since": since,
        "tracked_universe": len(tracked) or "unscoped",
        "symbol_overrides": symbol_overrides,
        "include_body": bool(args.include_body),
        "documents_scanned": documents_scanned,
        "documents_with_tickers": documents_with_tickers,
        "mention_rows": mention_rows,
        "inserted": inserted,
        "failed_batches": failed_batches,
        "samples": samples,
        "ran_at": datetime.now(UTC).isoformat(),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index tickers against the Neon documents corpus.")
    parser.add_argument("--backfill", action="store_true", help="Scan the whole corpus, not just recent documents.")
    parser.add_argument("--since-days", type=int, default=3, help="Incremental window in days (default 3).")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and report, write nothing.")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N documents (0 = no limit).")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument(
        "--include-body", action="store_true",
        help="Also index unambiguous body mentions. Off by default: the first dry run showed the "
             "body tier was ~90% of rows and mostly acronyms that are also tickers (DAO, RSI, ASIC).",
    )
    parser.add_argument(
        "--no-scope", action="store_true",
        help="Index every symbol rather than only tickers that can appear on a board.",
    )
    parser.add_argument(
        "--sample-size", type=int, default=15,
        help="How many resolved documents to echo in the summary, for eyeballing false positives.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        summary = _run(args)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(exc), "ran_at": datetime.now(UTC).isoformat()}, indent=2))
        return 1
    print(json.dumps(summary, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

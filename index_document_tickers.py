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

Deliberately no new table and no new columns: reusing `intelligence_mentions`
keeps this clear of the deploy-order trap in CLAUDE.md, where a Vercel deploy
ships a reader before the Python migration that creates what it reads.
"""

from __future__ import annotations

import argparse
import json
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


def resolve_document(title: str, body: str) -> Dict[str, float]:
    """Ticker mentions for one document, as {symbol: stored confidence}.

    A title match wins outright; a body-only match has to clear the
    unambiguous bar to be recorded at all.
    """
    mentions: Dict[str, float] = {}

    for symbol in ticker_resolver.resolve_tickers(title or ""):
        mentions[symbol] = TITLE_CONFIDENCE

    body_hits = ticker_resolver.resolve_tickers((body or "")[:MAX_BODY_CHARS])
    for symbol, resolver_confidence in body_hits.items():
        if symbol in mentions:
            continue
        if resolver_confidence >= MIN_BODY_RESOLVER_CONFIDENCE:
            mentions[symbol] = BODY_CONFIDENCE

    return mentions


def build_mention_rows(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Rows for `neon_feeds.insert_ticker_mentions`, one per resolved ticker."""
    document_id = str(document.get("document_id") or "").strip()
    if not document_id:
        return []
    resolved = resolve_document(str(document.get("title") or ""), str(document.get("full_text") or ""))
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


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    since = None
    if not args.backfill:
        since = (datetime.now(UTC) - timedelta(days=args.since_days)).isoformat()

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
            document_rows = build_mention_rows(document)
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

#!/usr/bin/env python3
"""Phase 2 of migrating off custom_documents.json (see CLAUDE.md): one-time
backfill of every existing document from the GCS blob into Neon's
`documents` table (Phase 1's dual-write mirror only covers documents
created/updated *after* it was wired in - this script catches up everything
that came before).

Read-only against GCS (never writes custom_documents.json back - the blob
stays the sole source of truth until the later reader-cutover phases) and
idempotent against Neon (every row is an upsert keyed on document_id), so
it's safe to re-run if it's interrupted partway through.

Usage:
    python backfill_neon_documents.py [--dry-run] [--limit N] [--batch-size N]
                                       [--verify-sample N] [--summary-path PATH]

Required env vars:
    DATABASE_URL          (Neon connection string - unlike the best-effort
                            mirror, this script's entire purpose is writing
                            to Neon, so a missing/bad value is a hard error)
    GCS_BUCKET_NAME
    GCS_CREDENTIALS_JSON
"""

from __future__ import annotations

import argparse
import random
import sys
from typing import Any, Dict, List

import neon_feeds
import run_financial_news_pipeline as core


def _utc_now_iso() -> str:
    return core._utc_now_iso()


def _corpus_documents(storage) -> List[Dict[str, Any]]:
    payload = core._load_custom_documents(storage)
    documents = payload.get("documents", [])
    return [doc for doc in documents if isinstance(doc, dict)]


def _batched(items: List[Any], size: int) -> List[List[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _verify(corpus_docs: List[Dict[str, Any]], sample_size: int) -> Dict[str, Any]:
    """Compares Neon's row count and a random sample of full_text lengths
    against the source corpus. Not exhaustive (a full content diff of tens of
    thousands of documents isn't worth the Neon read cost here) - this is a
    fast, cheap sanity check, not a substitute for a manual spot-check before
    trusting the table as authoritative."""
    neon_count = neon_feeds.count_documents()
    corpus_ids = {
        str((doc.get("metadata") or {}).get("document_id", "") or "").strip()
        for doc in corpus_docs
    }
    corpus_ids.discard("")

    sample_ids = random.sample(sorted(corpus_ids), min(sample_size, len(corpus_ids))) if corpus_ids else []
    mismatches: List[Dict[str, Any]] = []
    checked = 0
    docs_by_id = {
        str((doc.get("metadata") or {}).get("document_id", "") or "").strip(): doc
        for doc in corpus_docs
    }
    for doc_id in sample_ids:
        corpus_doc = docs_by_id.get(doc_id)
        if not corpus_doc:
            continue
        corpus_text = str((corpus_doc.get("content") or {}).get("full_text", "") or "")
        neon_row = neon_feeds.get_document(doc_id)
        checked += 1
        if neon_row is None:
            mismatches.append({"doc_id": doc_id, "issue": "missing_in_neon"})
            continue
        neon_text = str(neon_row.get("full_text", "") or "")
        if len(neon_text) != len(corpus_text):
            mismatches.append(
                {
                    "doc_id": doc_id,
                    "issue": "full_text_length_mismatch",
                    "corpus_length": len(corpus_text),
                    "neon_length": len(neon_text),
                }
            )

    return {
        "corpus_document_count": len(corpus_ids),
        "neon_row_count": neon_count,
        "row_count_matches": neon_count >= len(corpus_ids),
        "sample_checked": checked,
        "sample_mismatches": mismatches,
    }


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    secrets_payload = core._load_streamlit_secrets()
    storage, gcs_status = core._get_gcs_storage(secrets_payload)
    if storage is None:
        raise RuntimeError(f"GCS read access is required for this backfill: {gcs_status}")

    corpus_docs = _corpus_documents(storage)
    targets = corpus_docs[: args.limit] if args.limit else corpus_docs

    print(f"Loaded {len(corpus_docs)} documents from custom_documents.json; backfilling {len(targets)}.")
    if args.dry_run:
        return {
            "ok": True,
            "ran_at": _utc_now_iso(),
            "dry_run": True,
            "corpus_document_count": len(corpus_docs),
            "planned_backfill_count": len(targets),
        }

    batches = _batched(targets, args.batch_size)
    upserted_total = 0
    failed_batches: List[Dict[str, Any]] = []

    for index, batch in enumerate(batches, start=1):
        try:
            upserted = neon_feeds.mirror_documents_batch(batch)
            upserted_total += upserted
            print(f"  batch {index}/{len(batches)}: upserted {upserted}")
        except Exception as exc:
            print(f"  batch {index}/{len(batches)}: FAILED - {exc}", file=sys.stderr)
            failed_batches.append({"batch_index": index, "batch_size": len(batch), "error": str(exc)})

    verification = _verify(corpus_docs, args.verify_sample) if args.verify_sample > 0 else {}

    summary = {
        "ok": len(failed_batches) == 0,
        "ran_at": _utc_now_iso(),
        "dry_run": False,
        "corpus_document_count": len(corpus_docs),
        "targeted_count": len(targets),
        "batch_size": args.batch_size,
        "batch_count": len(batches),
        "upserted_total": upserted_total,
        "failed_batch_count": len(failed_batches),
        "failed_batches": failed_batches[:25],
        "verification": verification,
    }
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-time backfill of custom_documents.json into Neon's documents table.")
    parser.add_argument("--dry-run", action="store_true", help="Report counts only; no Neon writes.")
    parser.add_argument("--limit", type=int, default=0, help="Only backfill the first N documents (0 = all). Useful for a test run.")
    parser.add_argument("--batch-size", type=int, default=200, help="Documents per multi-row upsert statement (default: 200).")
    parser.add_argument("--verify-sample", type=int, default=10, help="Spot-check this many random documents after backfilling (0 to skip).")
    parser.add_argument("--summary-path", default="", help="Write JSON run summary to this path.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        summary = _run(args)
    except Exception as exc:
        error_payload = {"ok": False, "error": str(exc), "ran_at": _utc_now_iso()}
        core._write_summary(args.summary_path, error_payload)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    core._write_summary(args.summary_path, summary)
    print(summary)
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

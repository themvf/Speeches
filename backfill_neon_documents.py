#!/usr/bin/env python3
"""Backfill the legacy GCS document and enrichment snapshots into Neon.

The document path is Phase 2 of migrating off custom_documents.json (see
CLAUDE.md).  With ``--include-enrichment`` the same idempotent run also mirrors
document_enrichment_state.json into the row-level ``document_enrichments``
table used by SEC-20's bounded readers.

Read-only against GCS (never writes custom_documents.json back - the blob
stays the sole source of truth until the later reader-cutover phases) and
idempotent against Neon (every row is an upsert keyed on document_id), so
it's safe to re-run if it's interrupted partway through.

Usage:
    python backfill_neon_documents.py [--dry-run] [--limit N] [--batch-size N]
                                       [--verify-sample N] [--include-enrichment]
                                       [--include-speeches] [--summary-path PATH]

Required env vars:
    DATABASE_URL          (Neon connection string - unlike the best-effort
                            mirror, this script's entire purpose is writing
                            to Neon, so a missing/bad value is a hard error)
    GCS_BUCKET_NAME
    GCS_CREDENTIALS_JSON
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from typing import Any, Dict, List

import neon_feeds
import run_financial_news_pipeline as core

ID_COVERAGE_BATCH_SIZE = 1000


def _utc_now_iso() -> str:
    return core._utc_now_iso()


def _normalize_legacy_speech(record: Dict[str, Any]) -> Dict[str, Any]:
    """Match the stable document shape historically produced by web readers."""
    metadata = dict(record.get("metadata", {})) if isinstance(record.get("metadata"), dict) else {}
    content = dict(record.get("content", {})) if isinstance(record.get("content"), dict) else {}
    full_text = str(content.get("full_text", "") or "").strip()
    organization = core._normalize_org_label(metadata.get("organization") or metadata.get("org") or "SEC")
    organization_key = core._org_key_from_label(organization)
    stable = "|".join(
        [
            organization_key,
            str(metadata.get("url", "") or "").strip(),
            str(metadata.get("title", "") or "").strip(),
            str(metadata.get("speaker", "") or "").strip(),
            str(metadata.get("date", "") or "").strip(),
        ]
    )
    if not stable.replace("|", "").strip():
        stable = full_text[:1000]
    document_id = str(metadata.get("document_id", "") or "").strip() or hashlib.sha256(
        stable.encode("utf-8")
    ).hexdigest()[:24]
    source_kind = str(metadata.get("source_kind", "") or "").strip() or "sec_speech"
    published_date = str(metadata.get("published_date") or metadata.get("date") or "").strip()
    updated_date = str(metadata.get("updated_date") or metadata.get("extraction_date") or "").strip()
    metadata.update(
        {
            "document_id": document_id,
            "organization": organization,
            "source_kind": source_kind,
            "source_family": str(metadata.get("source_family", "") or source_kind),
            "doc_type": str(metadata.get("doc_type", "") or "Speech"),
            "published_date": published_date,
            "updated_date": updated_date,
            "last_reviewed_or_updated": str(
                metadata.get("last_reviewed_or_updated") or updated_date or published_date
            ),
            "source_format": str(metadata.get("source_format", "") or "html"),
            "word_count": int(metadata.get("word_count", 0) or len(full_text.split())),
        }
    )
    normalized = dict(record)
    normalized["metadata"] = metadata
    normalized["content"] = content
    return normalized


def _ensure_custom_document_identity(record: Dict[str, Any]) -> Dict[str, Any]:
    """Preserve a custom record while matching the web corpus's fallback ID."""
    metadata = dict(record.get("metadata", {})) if isinstance(record.get("metadata"), dict) else {}
    if str(metadata.get("document_id", "") or "").strip():
        return record
    content = dict(record.get("content", {})) if isinstance(record.get("content"), dict) else {}
    full_text = str(content.get("full_text", "") or "").strip()
    organization = core._normalize_org_label(metadata.get("organization") or metadata.get("org") or "SEC")
    stable = "|".join(
        [
            core._org_key_from_label(organization),
            str(metadata.get("url", "") or "").strip(),
            str(metadata.get("title", "") or "").strip(),
            str(metadata.get("speaker", "") or "").strip(),
            str(metadata.get("date", "") or "").strip(),
        ]
    )
    identity_seed = stable if stable.replace("|", "").strip() else full_text[:1000]
    metadata["document_id"] = hashlib.sha256(identity_seed.encode("utf-8")).hexdigest()[:24]
    normalized = dict(record)
    normalized["metadata"] = metadata
    return normalized


def _corpus_documents(storage, include_speeches: bool = False) -> List[Dict[str, Any]]:
    payload = core._load_custom_documents(storage)
    documents = payload.get("documents", [])
    custom_documents = [
        _ensure_custom_document_identity(doc)
        for doc in documents
        if isinstance(doc, dict)
    ]
    if not include_speeches:
        return custom_documents

    speech_payload = storage.load_speeches()
    speeches = [
        _normalize_legacy_speech(item)
        for item in speech_payload.get("speeches", [])
        if isinstance(item, dict)
    ]
    # Legacy speeches establish the base corpus; custom documents win on a
    # stable ID collision, matching loadCorpusDocuments() in the web tier.
    merged: Dict[str, Dict[str, Any]] = {}
    for record in [*speeches, *custom_documents]:
        metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}
        document_id = str(metadata.get("document_id", "") or "").strip()
        if document_id:
            merged[document_id] = record
    return list(merged.values())


def _corpus_enrichment_entries(storage) -> Dict[str, Dict[str, Any]]:
    payload = core._load_enrichment_state(storage)
    entries = payload.get("entries", {}) if isinstance(payload, dict) else {}
    if not isinstance(entries, dict):
        return {}
    return {
        str(document_id).strip(): entry
        for document_id, entry in entries.items()
        if str(document_id).strip() and isinstance(entry, dict)
    }


def _batched(items: List[Any], size: int) -> List[List[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _existing_ids_in_batches(document_ids: List[str], reader) -> set[str]:
    existing: set[str] = set()
    for batch in _batched(document_ids, ID_COVERAGE_BATCH_SIZE):
        existing.update(reader(batch))
    return existing


def _verify(corpus_docs: List[Dict[str, Any]], sample_size: int) -> Dict[str, Any]:
    """Compares Neon's row count and a random sample of full_text lengths
    against the source corpus. Not exhaustive (a full content diff of tens of
    thousands of documents isn't worth the Neon read cost here) - this is a
    fast, cheap sanity check, not a substitute for a manual spot-check before
    trusting the table as authoritative.

    Deliberately never raises: a verification-step failure (e.g. Neon
    unreachable) must not blow away the batch-upload summary that already
    ran successfully - it should show up as a self-contained "error" key in
    the returned dict instead, exactly like a failed batch does.
    """
    try:
        neon_count = neon_feeds.count_documents()
    except Exception as exc:
        return {"error": f"could not read Neon documents table: {exc}"}

    corpus_ids = {
        str((doc.get("metadata") or {}).get("document_id", "") or "").strip()
        for doc in corpus_docs
    }
    corpus_ids.discard("")
    sorted_corpus_ids = sorted(corpus_ids)
    try:
        existing_ids = _existing_ids_in_batches(
            sorted_corpus_ids,
            neon_feeds.get_existing_document_ids,
        )
    except Exception as exc:
        return {"error": f"could not verify Neon document ID coverage: {exc}"}
    missing_ids = [document_id for document_id in sorted_corpus_ids if document_id not in existing_ids]

    sample_ids = random.sample(sorted_corpus_ids, min(sample_size, len(corpus_ids))) if corpus_ids else []
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
        try:
            neon_row = neon_feeds.get_document(doc_id)
        except Exception as exc:
            mismatches.append({"doc_id": doc_id, "issue": "verify_read_failed", "error": str(exc)})
            continue
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
        "covered_document_count": len(existing_ids),
        "coverage_matches": not missing_ids,
        "missing_document_ids": missing_ids[:25],
        "sample_checked": checked,
        "sample_mismatches": mismatches,
    }


def _verify_enrichments(
    corpus_entries: Dict[str, Dict[str, Any]], sample_size: int
) -> Dict[str, Any]:
    try:
        neon_count = neon_feeds.count_enrichment_entries()
    except Exception as exc:
        return {"error": f"could not read Neon enrichment table: {exc}"}

    sorted_entry_ids = sorted(corpus_entries)
    try:
        existing_ids = _existing_ids_in_batches(
            sorted_entry_ids,
            neon_feeds.get_existing_enrichment_ids,
        )
    except Exception as exc:
        return {"error": f"could not verify Neon enrichment ID coverage: {exc}"}
    missing_ids = [document_id for document_id in sorted_entry_ids if document_id not in existing_ids]

    sample_ids = random.sample(
        sorted_entry_ids, min(sample_size, len(corpus_entries))
    ) if corpus_entries else []
    try:
        mirrored = neon_feeds.get_enrichment_entries(sample_ids)
    except Exception as exc:
        return {"error": f"could not sample Neon enrichment entries: {exc}"}

    mismatches: List[Dict[str, Any]] = []
    for document_id in sample_ids:
        if mirrored.get(document_id) != corpus_entries[document_id]:
            mismatches.append({"doc_id": document_id, "issue": "entry_mismatch_or_missing"})

    return {
        "corpus_enrichment_count": len(corpus_entries),
        "neon_row_count": neon_count,
        "row_count_matches": neon_count >= len(corpus_entries),
        "covered_enrichment_count": len(existing_ids),
        "coverage_matches": not missing_ids,
        "missing_enrichment_ids": missing_ids[:25],
        "sample_checked": len(sample_ids),
        "sample_mismatches": mismatches,
    }


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.dry_run and not bool(getattr(args, "force", False)):
        existing_checkpoint = neon_feeds.get_migration_checkpoint(
            neon_feeds.NEON_FULL_BACKFILL_CHECKPOINT
        )
        if (
            existing_checkpoint
            and str(existing_checkpoint.get("status", "") or "").strip().lower() == "verified"
        ):
            raise RuntimeError(
                "A verified full backfill already exists. Refusing to overwrite newer Neon rows; "
                "use --force only during a deliberate writer freeze."
            )

    secrets_payload = core._load_streamlit_secrets()
    storage, gcs_status = core._get_gcs_storage(secrets_payload)
    if storage is None:
        raise RuntimeError(f"GCS read access is required for this backfill: {gcs_status}")

    include_speeches = bool(getattr(args, "include_speeches", False))
    corpus_docs = _corpus_documents(storage, include_speeches=include_speeches)
    targets = corpus_docs[: args.limit] if args.limit else corpus_docs
    include_enrichment = bool(getattr(args, "include_enrichment", False))
    enrichment_entries = _corpus_enrichment_entries(storage) if include_enrichment else {}
    if args.limit and include_enrichment:
        target_document_ids = {
            str((document.get("metadata") or {}).get("document_id", "") or "").strip()
            for document in targets
            if isinstance(document, dict)
        }
        target_document_ids.discard("")
        enrichment_entries = {
            document_id: entry
            for document_id, entry in enrichment_entries.items()
            if document_id in target_document_ids
        }

    print(f"Loaded {len(corpus_docs)} documents from custom_documents.json; backfilling {len(targets)}.")
    if args.dry_run:
        return {
            "ok": True,
            "ran_at": _utc_now_iso(),
            "dry_run": True,
            "corpus_document_count": len(corpus_docs),
            "planned_backfill_count": len(targets),
            "planned_enrichment_backfill_count": len(enrichment_entries),
            "include_speeches": include_speeches,
        }

    batches = _batched(targets, args.batch_size)
    upserted_total = 0
    failed_batches: List[Dict[str, Any]] = []

    for index, batch in enumerate(batches, start=1):
        try:
            upserted = neon_feeds.mirror_documents_batch(batch)
            if upserted != len(batch):
                raise RuntimeError(
                    f"Neon accepted {upserted} of {len(batch)} document rows."
                )
            upserted_total += upserted
            print(f"  batch {index}/{len(batches)}: upserted {upserted}")
        except Exception as exc:
            print(f"  batch {index}/{len(batches)}: FAILED - {exc}", file=sys.stderr)
            failed_batches.append({"batch_index": index, "batch_size": len(batch), "error": str(exc)})

    enrichment_upserted_total = 0
    failed_enrichment_batches: List[Dict[str, Any]] = []
    enrichment_items = list(enrichment_entries.items())
    enrichment_batches = _batched(enrichment_items, args.batch_size)
    for index, batch in enumerate(enrichment_batches, start=1):
        batch_entries = dict(batch)
        try:
            upserted = neon_feeds.upsert_enrichment_entries(batch_entries)
            if upserted != len(batch_entries):
                raise RuntimeError(
                    f"Neon accepted {upserted} of {len(batch_entries)} enrichment rows."
                )
            enrichment_upserted_total += upserted
            print(f"  enrichment batch {index}/{len(enrichment_batches)}: submitted {upserted}")
        except Exception as exc:
            print(f"  enrichment batch {index}/{len(enrichment_batches)}: FAILED - {exc}", file=sys.stderr)
            failed_enrichment_batches.append(
                {"batch_index": index, "batch_size": len(batch), "error": str(exc)}
            )

    verification = _verify(targets, args.verify_sample) if args.verify_sample > 0 else {}
    enrichment_verification = (
        _verify_enrichments(enrichment_entries, args.verify_sample)
        if include_enrichment and args.verify_sample > 0
        else {}
    )

    document_verification_ok = (
        not verification
        or (
            "error" not in verification
            and verification.get("row_count_matches") is True
            and verification.get("coverage_matches") is True
            and not verification.get("sample_mismatches")
        )
    )
    enrichment_verification_ok = (
        not enrichment_verification
        or (
            "error" not in enrichment_verification
            and enrichment_verification.get("row_count_matches") is True
            and enrichment_verification.get("coverage_matches") is True
            and not enrichment_verification.get("sample_mismatches")
        )
    )

    base_ok = (
            len(failed_batches) == 0
            and len(failed_enrichment_batches) == 0
            and document_verification_ok
            and enrichment_verification_ok
    )
    checkpoint_eligible = bool(
        base_ok
        and not args.limit
        and include_speeches
        and include_enrichment
        and args.verify_sample > 0
    )
    checkpoint_recorded = False
    checkpoint_error = ""
    if checkpoint_eligible:
        try:
            neon_feeds.set_migration_checkpoint(
                neon_feeds.NEON_FULL_BACKFILL_CHECKPOINT,
                "verified",
                {
                    "document_count": len(targets),
                    "enrichment_count": len(enrichment_entries),
                    "verified_at": _utc_now_iso(),
                },
            )
            checkpoint_recorded = True
        except Exception as exc:
            checkpoint_error = str(exc)

    summary = {
        "ok": base_ok and (not checkpoint_eligible or checkpoint_recorded),
        "ran_at": _utc_now_iso(),
        "dry_run": False,
        "corpus_document_count": len(corpus_docs),
        "include_speeches": include_speeches,
        "targeted_count": len(targets),
        "batch_size": args.batch_size,
        "batch_count": len(batches),
        "upserted_total": upserted_total,
        "failed_batch_count": len(failed_batches),
        "failed_batches": failed_batches[:25],
        "verification": verification,
        "include_enrichment": include_enrichment,
        "corpus_enrichment_count": len(enrichment_entries),
        "enrichment_batch_count": len(enrichment_batches),
        "enrichment_upserted_total": enrichment_upserted_total,
        "failed_enrichment_batch_count": len(failed_enrichment_batches),
        "failed_enrichment_batches": failed_enrichment_batches[:25],
        "enrichment_verification": enrichment_verification,
        "activation_checkpoint": {
            "key": neon_feeds.NEON_FULL_BACKFILL_CHECKPOINT,
            "eligible": checkpoint_eligible,
            "recorded": checkpoint_recorded,
            "error": checkpoint_error,
        },
    }
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-time backfill of custom_documents.json into Neon's documents table.")
    parser.add_argument("--dry-run", action="store_true", help="Report counts only; no Neon writes.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow a non-dry rerun after activation (requires an external writer freeze).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only backfill the first N documents (0 = all). Useful for a test run.")
    parser.add_argument("--batch-size", type=int, default=200, help="Documents per multi-row upsert statement (default: 200).")
    parser.add_argument("--verify-sample", type=int, default=10, help="Spot-check this many random documents after backfilling (0 to skip).")
    parser.add_argument(
        "--include-enrichment",
        action="store_true",
        help="Also backfill document_enrichment_state.json into document_enrichments.",
    )
    parser.add_argument(
        "--include-speeches",
        action="store_true",
        help="Also normalize and backfill legacy all_speeches.json records.",
    )
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

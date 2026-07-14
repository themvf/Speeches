#!/usr/bin/env python3
"""Additively sync pilot Neon rows into the legacy GCS snapshots.

The incremental ingestion paths use Neon for row-level reads and writes.  A
small number of legacy readers still need ``custom_documents.json`` and
``document_enrichment_state.json`` while that migration finishes.  This
compatibility job updates those snapshots once per day without putting the
hourly ingestion jobs back on the full-blob read/write path.

Only the three SEC-20 pilot source kinds are included.  The merge never
deletes a legacy document or enrichment entry, preserves fields that are not
present in the Neon record, and skips both uploads when there is no material
difference.  The existing GCS generation-match helpers remain the final guard
against concurrent writers.
"""

from __future__ import annotations

import argparse
import copy
import re
import sys
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import neon_feeds
import run_financial_news_pipeline as core


PILOT_SOURCE_KINDS: Tuple[str, ...] = (
    "bloomberg_public_article",
    "substack_public_article",
    "newsapi_article",
)
DEFAULT_ENRICHMENT_BATCH_SIZE = 500
PRE_CUTOVER_BACKUP_PREFIX = "sec20-backups/pre-neon-cutover"


def _batched(items: Sequence[str], size: int) -> Iterable[List[str]]:
    if size <= 0:
        raise ValueError("batch size must be greater than zero")
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def _normalize_custom_snapshot(payload: Any) -> Dict[str, Any]:
    """Validate the required shape without discarding unknown top-level keys."""
    if not isinstance(payload, dict):
        raise ValueError("custom_documents.json must contain a JSON object")
    if not isinstance(payload.get("documents"), list):
        raise ValueError("custom_documents.json has an invalid documents collection")
    normalized = copy.deepcopy(payload)
    normalized["updated_at"] = str(normalized.get("updated_at", "") or "")
    return normalized


def _normalize_enrichment_snapshot(payload: Any) -> Dict[str, Any]:
    """Validate the required shape without discarding rollback metadata."""
    if not isinstance(payload, dict):
        raise ValueError("document_enrichment_state.json must contain a JSON object")
    if not isinstance(payload.get("entries"), dict):
        raise ValueError("document_enrichment_state.json has an invalid entries collection")
    normalized = copy.deepcopy(payload)
    normalized["updated_at"] = str(normalized.get("updated_at", "") or "")
    return normalized


def _load_custom_snapshot(storage: Any) -> Dict[str, Any]:
    payload = core._load_json_store(
        storage=storage,
        blob_name=core.CUSTOM_DOCS_BLOB_NAME,
        local_path=core.CUSTOM_DOCS_LOCAL_PATH,
        default_factory=lambda: {"updated_at": "", "documents": []},
        normalize_fn=_normalize_custom_snapshot,
    )
    if core.CUSTOM_DOCS_BLOB_NAME in core._REMOTE_LOAD_ERRORED_BLOBS:
        raise RuntimeError("Could not safely load custom_documents.json from GCS.")
    return payload


def _load_enrichment_snapshot(storage: Any) -> Dict[str, Any]:
    payload = core._load_json_store(
        storage=storage,
        blob_name=core.ENRICHMENT_STATE_BLOB_NAME,
        local_path=core.ENRICHMENT_STATE_LOCAL_PATH,
        default_factory=core._empty_enrichment_state,
        normalize_fn=_normalize_enrichment_snapshot,
    )
    if core.ENRICHMENT_STATE_BLOB_NAME in core._REMOTE_LOAD_ERRORED_BLOBS:
        raise RuntimeError("Could not safely load document_enrichment_state.json from GCS.")
    return payload


def _save_custom_snapshot(storage: Any, payload: Dict[str, Any]) -> None:
    core._save_json_store(
        storage=storage,
        blob_name=core.CUSTOM_DOCS_BLOB_NAME,
        local_path=core.CUSTOM_DOCS_LOCAL_PATH,
        payload=payload,
        normalize_fn=_normalize_custom_snapshot,
        require_remote=True,
    )


def _save_enrichment_snapshot(storage: Any, payload: Dict[str, Any]) -> None:
    core._save_json_store(
        storage=storage,
        blob_name=core.ENRICHMENT_STATE_BLOB_NAME,
        local_path=core.ENRICHMENT_STATE_LOCAL_PATH,
        payload=payload,
        normalize_fn=_normalize_enrichment_snapshot,
        require_remote=True,
    )


def _metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    value = record.get("metadata", {}) if isinstance(record, dict) else {}
    return value if isinstance(value, dict) else {}


def _content(record: Dict[str, Any]) -> Dict[str, Any]:
    value = record.get("content", {}) if isinstance(record, dict) else {}
    return value if isinstance(value, dict) else {}


def _document_id(record: Dict[str, Any]) -> str:
    metadata = _metadata(record)
    return str(metadata.get("document_id", record.get("document_id", "")) or "").strip()


def _document_url(record: Dict[str, Any]) -> str:
    metadata = _metadata(record)
    raw_url = metadata.get("url", record.get("url", ""))
    return core._url_match_key(str(raw_url or "").strip())


def _deep_additive_merge(existing: Any, incoming: Any) -> Any:
    """Overlay incoming values without removing keys absent from incoming."""
    if not isinstance(existing, dict) or not isinstance(incoming, dict):
        return copy.deepcopy(incoming)
    merged = copy.deepcopy(existing)
    for key, value in incoming.items():
        current = merged.get(key)
        # This is a compatibility export, not a destructive replica. Empty
        # projection values must never erase useful legacy/admin data.
        if value in (None, "") and current not in (None, ""):
            continue
        if isinstance(value, (list, dict)) and not value and current:
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_additive_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _derive_text_arrays(full_text: str) -> Tuple[List[str], List[str]]:
    """Build deterministic legacy text arrays for a new or changed body."""
    clean_text = str(full_text or "").strip()
    if not clean_text:
        return [], []

    paragraphs = [
        block.strip()
        for block in re.split(r"\n\s*\n", clean_text)
        if block.strip()
    ]
    # Extractors sometimes store one paragraph per line rather than blank-line
    # separated blocks.  Preserve that useful structure when it is present.
    if len(paragraphs) == 1 and "\n" in clean_text:
        paragraphs = [line.strip() for line in clean_text.splitlines() if line.strip()]

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", clean_text.replace("\n", " "))
        if sentence.strip()
    ]
    return paragraphs, sentences


def _merge_document(
    existing: Dict[str, Any] | None,
    incoming: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge one Neon record while retaining legacy-only record fields."""
    old_record = existing if isinstance(existing, dict) else {}
    new_record = incoming if isinstance(incoming, dict) else {}
    merged = _deep_additive_merge(old_record, new_record)

    old_content = _content(old_record)
    incoming_content = _content(new_record)
    merged_content = _deep_additive_merge(old_content, incoming_content)

    old_text = str(old_content.get("full_text", "") or "")
    incoming_text = str(incoming_content.get("full_text", "") or "")
    # An empty mirror value must not erase a body already retained by the
    # legacy snapshot.  Non-empty Neon text is authoritative for the pilot.
    if not incoming_text and old_text:
        merged_content["full_text"] = old_text
        text_changed = False
    else:
        text_changed = incoming_text != old_text

    is_new = existing is None
    if (is_new or text_changed) and incoming_text:
        paragraphs, sentences = _derive_text_arrays(incoming_text)
        if not isinstance(incoming_content.get("paragraphs"), list):
            merged_content["paragraphs"] = paragraphs
        if not isinstance(incoming_content.get("sentences"), list):
            merged_content["sentences"] = sentences

    merged["content"] = merged_content
    return merged


def merge_documents(
    legacy_payload: Dict[str, Any],
    neon_records: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Add or update Neon records by document id first, then canonical URL."""
    output = copy.deepcopy(legacy_payload) if isinstance(legacy_payload, dict) else {}
    raw_documents = output.get("documents", [])
    documents = list(raw_documents) if isinstance(raw_documents, list) else []
    output["documents"] = documents

    id_index: Dict[str, int] = {}
    url_index: Dict[str, int] = {}
    for index, record in enumerate(documents):
        if not isinstance(record, dict):
            continue
        document_id = _document_id(record)
        url = _document_url(record)
        if document_id and document_id not in id_index:
            id_index[document_id] = index
        if url and url not in url_index:
            url_index[url] = index

    stats = {"seen": 0, "added": 0, "updated": 0, "unchanged": 0, "skipped": 0}
    for incoming in neon_records:
        stats["seen"] += 1
        if not isinstance(incoming, dict):
            stats["skipped"] += 1
            continue
        document_id = _document_id(incoming)
        url = _document_url(incoming)
        if not document_id and not url:
            stats["skipped"] += 1
            continue

        match_index = id_index.get(document_id) if document_id else None
        if match_index is None and url:
            match_index = url_index.get(url)

        if match_index is None:
            merged = _merge_document(None, incoming)
            documents.append(merged)
            match_index = len(documents) - 1
            stats["added"] += 1
        else:
            existing = documents[match_index]
            merged = _merge_document(existing if isinstance(existing, dict) else None, incoming)
            if merged == existing:
                stats["unchanged"] += 1
            else:
                documents[match_index] = merged
                stats["updated"] += 1

        merged_id = _document_id(documents[match_index])
        merged_url = _document_url(documents[match_index])
        if merged_id:
            id_index[merged_id] = match_index
        if merged_url:
            url_index[merged_url] = match_index

    return output, stats


def merge_enrichments(
    legacy_state: Dict[str, Any],
    neon_entries: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Additively merge bounded Neon enrichment entries by document id."""
    output = copy.deepcopy(legacy_state) if isinstance(legacy_state, dict) else {}
    raw_entries = output.get("entries", {})
    entries = copy.deepcopy(raw_entries) if isinstance(raw_entries, dict) else {}
    output["entries"] = entries

    stats = {"seen": 0, "added": 0, "updated": 0, "unchanged": 0, "skipped": 0}
    for raw_document_id, incoming in neon_entries.items():
        stats["seen"] += 1
        document_id = str(raw_document_id or "").strip()
        if not document_id or not isinstance(incoming, dict):
            stats["skipped"] += 1
            continue
        existing = entries.get(document_id)
        merged = _deep_additive_merge(existing, incoming) if isinstance(existing, dict) else copy.deepcopy(incoming)
        if isinstance(existing, dict):
            existing_review = existing.get("review")
            if isinstance(existing_review, dict):
                decision = str(existing_review.get("decision", "") or "").strip().lower()
                notes = str(existing_review.get("notes", "") or "").strip()
                reviewed_at = str(existing_review.get("reviewed_at", "") or "").strip()
                if decision not in {"", "pending"} or notes or reviewed_at:
                    merged["review"] = copy.deepcopy(existing_review)
                    if str(existing.get("status", "") or "").strip().lower() == "reviewed":
                        merged["status"] = "reviewed"
        if existing is None:
            entries[document_id] = merged
            stats["added"] += 1
        elif merged == existing:
            stats["unchanged"] += 1
        else:
            entries[document_id] = merged
            stats["updated"] += 1

    return output, stats


def _fetch_neon_records(batch_size: int) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], int]:
    records = neon_feeds.get_document_records_by_source_kinds(
        list(PILOT_SOURCE_KINDS),
        include_full_text=True,
    )
    clean_records = [record for record in records if isinstance(record, dict)]
    document_ids = list(
        dict.fromkeys(document_id for document_id in map(_document_id, clean_records) if document_id)
    )
    entries: Dict[str, Dict[str, Any]] = {}
    batch_count = 0
    for batch in _batched(document_ids, batch_size):
        batch_count += 1
        result = neon_feeds.get_enrichment_entries(batch)
        if isinstance(result, dict):
            entries.update(
                {
                    str(document_id): entry
                    for document_id, entry in result.items()
                    if str(document_id).strip() and isinstance(entry, dict)
                }
            )
    return clean_records, entries, batch_count


def _ensure_pre_cutover_backup(storage: Any, source_blob_name: str) -> str:
    """Server-side copy the first pre-sync generation to an immutable name."""
    expected_generation = core._BLOB_GENERATIONS.get(source_blob_name)
    if expected_generation in (None, 0):
        raise RuntimeError(
            f"Cannot preserve {source_blob_name}: no existing source generation was loaded."
        )
    backup_name = f"{PRE_CUTOVER_BACKUP_PREFIX}/{source_blob_name}"
    destination = storage.bucket.blob(backup_name)
    if destination.exists():
        return backup_name
    source = storage.bucket.blob(source_blob_name, generation=expected_generation)
    storage.bucket.copy_blob(
        source,
        storage.bucket,
        new_name=backup_name,
        source_generation=expected_generation,
        if_source_generation_match=expected_generation,
        if_generation_match=0,
    )
    return backup_name


def _ensure_pre_cutover_backups(storage: Any) -> List[str]:
    return [
        _ensure_pre_cutover_backup(storage, core.CUSTOM_DOCS_BLOB_NAME),
        _ensure_pre_cutover_backup(storage, core.ENRICHMENT_STATE_BLOB_NAME),
    ]


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")

    # Fail before either large GCS read when Neon is unavailable or the exact
    # migration checkpoint was never recorded.
    core._require_neon_authoritative_ready()

    # Read the bounded Neon rows before opening the GCS snapshots.  This keeps
    # the generation baseline-to-save window as short as practical.
    neon_records, neon_entries, enrichment_batch_count = _fetch_neon_records(batch_size)

    storage, gcs_status = core._get_gcs_storage(core._load_streamlit_secrets())
    if storage is None:
        raise RuntimeError(f"GCS read/write access is required: {gcs_status}")

    # Each legacy object is loaded exactly once.  Besides controlling egress,
    # these reads capture the generation preconditions used by the saves.
    legacy_documents = _load_custom_snapshot(storage)
    legacy_enrichments = _load_enrichment_snapshot(storage)

    merged_documents, document_stats = merge_documents(legacy_documents, neon_records)
    merged_enrichments, enrichment_stats = merge_enrichments(legacy_enrichments, neon_entries)
    documents_changed = merged_documents != legacy_documents
    enrichments_changed = merged_enrichments != legacy_enrichments

    wrote_documents = False
    wrote_enrichments = False
    backup_blobs: List[str] = []
    if not args.dry_run:
        if documents_changed or enrichments_changed:
            backup_blobs = _ensure_pre_cutover_backups(storage)
        # Save documents first.  If its generation changed, the exception
        # stops the run before enrichment can get ahead of the document set.
        if documents_changed:
            _save_custom_snapshot(storage, merged_documents)
            wrote_documents = True
        if enrichments_changed:
            _save_enrichment_snapshot(storage, merged_enrichments)
            wrote_enrichments = True

    return {
        "ok": True,
        "ran_at": core._utc_now_iso(),
        "dry_run": bool(args.dry_run),
        "source_kinds": list(PILOT_SOURCE_KINDS),
        "neon_document_count": len(neon_records),
        "neon_enrichment_count": len(neon_entries),
        "enrichment_batch_size": batch_size,
        "enrichment_batch_count": enrichment_batch_count,
        "document_merge": document_stats,
        "enrichment_merge": enrichment_stats,
        "planned_document_write": documents_changed,
        "planned_enrichment_write": enrichments_changed,
        "wrote_documents": wrote_documents,
        "wrote_enrichments": wrote_enrichments,
        "pre_cutover_backups": backup_blobs,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Additively sync SEC-20 pilot Neon rows to legacy GCS snapshots."
    )
    parser.add_argument("--dry-run", action="store_true", help="Report changes without uploading snapshots.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_ENRICHMENT_BATCH_SIZE,
        help=f"Document IDs per bounded enrichment query (default: {DEFAULT_ENRICHMENT_BATCH_SIZE}).",
    )
    parser.add_argument("--summary-path", default="", help="Write the JSON run summary to this path.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        summary = _run(args)
    except Exception as exc:
        summary = {"ok": False, "ran_at": core._utc_now_iso(), "error": str(exc)}
        core._write_summary(args.summary_path, summary)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    core._write_summary(args.summary_path, summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

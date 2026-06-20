#!/usr/bin/env python3
"""Sync corpus documents to OpenAI vector stores.

Reads the corpus from GCS, computes a content hash per document, diffs against
the stored index manifest, then uploads new/changed docs and removes stale ones.

Usage:
    python sync_knowledge_index.py [--org ORG_KEY] [--force-rebuild] [--dry-run]

Required env vars:
    OPENAI_API_KEY
    GCS_BUCKET_NAME
    GCS_CREDENTIALS_JSON  (raw JSON string or base64-encoded JSON)
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

from google.cloud import storage as gcs_lib
from google.oauth2 import service_account
from openai import OpenAI

# ─── GCS blob names ────────────────────────────────────────────────────────────
SPEECHES_BLOB = "all_speeches.json"
CUSTOM_DOCS_BLOB = "custom_documents.json"
ENRICHMENT_BLOB = "document_enrichment_state.json"
VECTOR_STATE_BLOB = "openai_vector_store_state.json"

# ─── org routing ───────────────────────────────────────────────────────────────
_SOURCE_KIND_TO_ORG: dict[str, str] = {
    "sec_speech": "sec",
    "sec_tm_faq": "sec",
    "sec_enforcement_litigation": "sec",
    "finra_regulatory_notice": "finra",
    "finra_key_topic": "finra",
    "finra_comment_letter": "finra",
    "finra_awc": "finra",
    "doj_usao_press_release": "doj",
    "federal_reserve_speech_testimony": "federal_reserve",
    "cftc_press_release": "cftc",
    "cftc_public_statement_remark": "cftc",
    "treasury_featured_story": "treasury",
    "treasury_press_release": "treasury",
    "treasury_statement_remark": "treasury",
    "sifma_news_item": "sifma",
    "jdsupra_article": "trade_media",
    "investmentnews_article": "trade_media",
    "citywire_article": "trade_media",
    "congress_crs_product": "congress",
}

_ORG_LABELS: dict[str, str] = {
    "sec": "SEC",
    "finra": "FINRA",
    "doj": "DOJ",
    "federal_reserve": "Federal Reserve",
    "cftc": "CFTC",
    "treasury": "Treasury",
    "sifma": "SIFMA",
    "trade_media": "Trade Media",
    "congress": "Congress",
}


def _org_key(source_kind: str) -> str:
    return _SOURCE_KIND_TO_ORG.get(source_kind, "other")


def _org_label(org_key: str) -> str:
    return _ORG_LABELS.get(org_key, org_key.upper())


# ─── helpers ───────────────────────────────────────────────────────────────────

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_gcs_credentials(raw: str) -> dict:
    text = raw.strip().strip("'\"")
    for candidate in [raw.strip(), text]:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    try:
        decoded = base64.b64decode(raw.strip()).decode("utf-8")
        parsed = json.loads(decoded)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    raise ValueError("GCS_CREDENTIALS_JSON could not be parsed as JSON or base64 JSON")


def _get_bucket(bucket_name: str, credentials_raw: str):
    creds_info = _parse_gcs_credentials(credentials_raw)
    creds = service_account.Credentials.from_service_account_info(creds_info)
    client = gcs_lib.Client(credentials=creds, project=creds_info.get("project_id"))
    return client.bucket(bucket_name)


def _download(bucket, blob_name: str) -> Any:
    blob = bucket.blob(blob_name)
    if not blob.exists():
        return None
    raw = blob.download_as_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON from GCS blob '{blob_name}': {exc}") from exc


def _upload_json(bucket, blob_name: str, payload: Any) -> None:
    blob = bucket.blob(blob_name)
    blob.upload_from_string(
        json.dumps(payload, indent=2, ensure_ascii=False),
        content_type="application/json",
    )


# ─── corpus loading ─────────────────────────────────────────────────────────

def _load_corpus(bucket) -> list[dict]:
    docs: list[dict] = []
    for blob_name in [SPEECHES_BLOB, CUSTOM_DOCS_BLOB]:
        raw = _download(bucket, blob_name)
        if not raw:
            continue
        for key in ("speeches", "documents"):
            for entry in raw.get(key, []):
                docs.append(entry)
    return docs


# ─── document rendering ─────────────────────────────────────────────────────

def _render_doc(doc: dict, enrichment_entries: dict) -> str:
    meta = doc.get("metadata", {}) or {}
    doc_id = str(meta.get("document_id", "") or "").strip()
    title = str(meta.get("title", "") or "").strip()
    speaker = str(meta.get("speaker", "") or "").strip()
    date = str(meta.get("date", "") or "").strip()
    url = str(meta.get("url", "") or "").strip()
    org = str(meta.get("organization", "") or "").strip()
    source_kind = str(meta.get("source_kind", "") or "").strip()
    tags_raw = str(meta.get("tags", "") or "").strip()

    enrich = (enrichment_entries.get(doc_id) or {})
    enrich_data = enrich.get("enrichment", {}) or {}
    enrich_tags = ", ".join(enrich_data.get("tags", []) or [])
    enrich_keywords = ", ".join(enrich_data.get("keywords", []) or [])
    enrich_summary = str(enrich_data.get("summary", "") or "").strip()
    enrich_stance = str(enrich_data.get("regulatory_stance", "") or "").strip()

    header_parts = [
        f"TITLE: {title}",
        f"SPEAKER: {speaker}" if speaker else None,
        f"DATE: {date}" if date else None,
        f"ORGANIZATION: {org}" if org else None,
        f"SOURCE: {source_kind}" if source_kind else None,
        f"URL: {url}" if url else None,
        f"TAGS: {tags_raw}" if tags_raw else None,
        f"ENRICHED_TAGS: {enrich_tags}" if enrich_tags else None,
        f"KEYWORDS: {enrich_keywords}" if enrich_keywords else None,
        f"REGULATORY_STANCE: {enrich_stance}" if enrich_stance else None,
        (f"\nSUMMARY:\n{enrich_summary}") if enrich_summary else None,
    ]
    header = "\n".join(p for p in header_parts if p)
    content = str(doc.get("content", "") or doc.get("text", "") or "").strip()
    return f"{header}\n\n---\n\n{content}" if content else header


def _build_org_manifest(all_docs: list[dict], target_org_key: str, enrichment_entries: dict) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for doc in all_docs:
        meta = doc.get("metadata", {}) or {}
        source_kind = str(meta.get("source_kind", "") or "").strip()
        if _org_key(source_kind) != target_org_key:
            continue
        doc_id = str(meta.get("document_id", "") or "").strip()
        if not doc_id:
            continue
        rendered = _render_doc(doc, enrichment_entries)
        result[doc_id] = {
            "doc_id": doc_id,
            "title": str(meta.get("title", "") or "").strip(),
            "speaker": str(meta.get("speaker", "") or "").strip(),
            "date": str(meta.get("date", "") or "").strip(),
            "url": str(meta.get("url", "") or "").strip(),
            "filename": f"{doc_id}.txt",
            "content_hash": _sha256(rendered),
            "_rendered": rendered,
        }
    return result


# ─── OpenAI operations ───────────────────────────────────────────────────────

def _ensure_vector_store(client: OpenAI, existing_id: str, label: str, force_rebuild: bool) -> tuple[str, bool, str]:
    if existing_id and not force_rebuild:
        try:
            client.vector_stores.retrieve(existing_id)
            return existing_id, False, ""
        except Exception:
            pass
    store = client.vector_stores.create(name=f"{label} Knowledge Index ({_utc_now()})")
    return store.id, True, existing_id


UPLOAD_BATCH_SIZE = 500
UPLOAD_WORKERS = 12


def _upload_file(client: OpenAI, doc: dict) -> str:
    content = doc["_rendered"].encode("utf-8")
    filename = doc["filename"]
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            uploaded = client.files.create(
                file=(filename, content, "text/plain"),
                purpose="assistants",
            )
            return str(getattr(uploaded, "id", "") or "")
        except Exception as exc:
            msg = str(exc).lower()
            retryable = any(t in msg for t in ["rate limit", "timeout", "temporar", "502", "503", "504", "connection"])
            if attempt >= max_attempts or not retryable:
                raise
            time.sleep(min(8, 2**attempt))
    raise RuntimeError("Upload failed after retries")


def _upload_doc_batch(client: OpenAI, vector_store_id: str, targets: list[tuple[str, dict]]) -> tuple[dict[str, str], list[dict]]:
    """Upload files concurrently, then attach and poll once for the whole batch."""
    uploaded: dict[str, str] = {}
    failed: list[dict] = []

    with ThreadPoolExecutor(max_workers=min(UPLOAD_WORKERS, len(targets))) as pool:
        future_map = {pool.submit(_upload_file, client, doc): doc_id for doc_id, doc in targets}
        for future in as_completed(future_map):
            doc_id = future_map[future]
            try:
                file_id = future.result()
                if not file_id:
                    raise RuntimeError("OpenAI returned an empty file ID")
                uploaded[doc_id] = file_id
            except Exception as exc:
                failed.append({"doc_id": doc_id, "stage": "upload", "error": str(exc)})

    if not uploaded:
        return {}, failed

    try:
        batch = client.vector_stores.file_batches.create_and_poll(
            vector_store_id=vector_store_id,
            file_ids=list(uploaded.values()),
        )
    except Exception as exc:
        failed.extend({"doc_id": doc_id, "stage": "attach", "error": str(exc)} for doc_id in uploaded)
        return {}, failed

    failed_count = int(getattr(getattr(batch, "file_counts", None), "failed", 0) or 0)
    if failed_count == 0:
        return uploaded, failed

    attached: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(UPLOAD_WORKERS, len(uploaded))) as pool:
        future_map = {
            pool.submit(client.vector_stores.files.retrieve, file_id, vector_store_id=vector_store_id): (doc_id, file_id)
            for doc_id, file_id in uploaded.items()
        }
        for future in as_completed(future_map):
            doc_id, file_id = future_map[future]
            try:
                vector_file = future.result()
                status = str(getattr(vector_file, "status", "") or "")
                if status == "completed":
                    attached[doc_id] = file_id
                else:
                    error = getattr(vector_file, "last_error", None)
                    failed.append({"doc_id": doc_id, "stage": "attach", "error": str(error or status or "unknown status")})
            except Exception as exc:
                failed.append({"doc_id": doc_id, "stage": "attach", "error": str(exc)})
    return attached, failed


def _delete_doc(client: OpenAI, vector_store_id: str, entry: dict) -> None:
    fid = str(entry.get("file_id", "") or entry.get("vector_store_file_id", "") or "").strip()
    if fid:
        client.vector_stores.files.delete(fid, vector_store_id=vector_store_id)


# ─── per-org sync ───────────────────────────────────────────────────────────

def _sync_org(
    client: OpenAI,
    bucket,
    vector_state: dict,
    all_docs: list[dict],
    enrichment_entries: dict,
    org_key: str,
    force_rebuild: bool,
    dry_run: bool,
) -> dict:
    label = _org_label(org_key)
    stores = vector_state.setdefault("stores", {})
    org_state: dict = dict(stores.get(org_key, {}) or {})
    existing_id = str(org_state.get("vector_store_id", "") or "").strip()
    indexed_docs: dict = dict(org_state.get("docs", {}) or {})

    current_docs = _build_org_manifest(all_docs, org_key, enrichment_entries)
    if not current_docs:
        print(f"  [{org_key}] no corpus docs — skipping")
        return {}

    indexed_ids = set(indexed_docs)
    current_ids = set(current_docs)
    add_ids = sorted(current_ids - indexed_ids)
    remove_ids = sorted(indexed_ids - current_ids)
    update_ids = sorted(
        d for d in (current_ids & indexed_ids)
        if indexed_docs[d].get("content_hash") != current_docs[d]["content_hash"]
    )
    unchanged_ids = sorted((current_ids & indexed_ids) - set(update_ids))

    if force_rebuild:
        add_ids, update_ids, remove_ids, unchanged_ids = sorted(current_ids), [], [], []
        indexed_docs = {}

    print(f"  [{org_key}] corpus={len(current_docs)} add={len(add_ids)} update={len(update_ids)} remove={len(remove_ids)} unchanged={len(unchanged_ids)}")

    if dry_run:
        return {"org_key": org_key, "dry_run": True, "add": len(add_ids), "update": len(update_ids), "remove": len(remove_ids)}

    vector_store_id, created_new, replaced_id = _ensure_vector_store(client, existing_id, label, force_rebuild)
    if created_new:
        add_ids, update_ids, remove_ids, unchanged_ids = sorted(current_ids), [], [], []
        indexed_docs = {}

    next_docs: dict = {d: indexed_docs[d] for d in unchanged_ids if d in indexed_docs}
    failed: list[dict] = []
    deleted_count = uploaded_count = 0

    delete_targets = [] if created_new else (remove_ids + update_ids)
    for doc_id in delete_targets:
        try:
            _delete_doc(client, vector_store_id, indexed_docs.get(doc_id, {}))
            deleted_count += 1
            print(f"  [{org_key}] deleted {doc_id}")
        except Exception as exc:
            print(f"  [{org_key}] delete failed {doc_id}: {exc}", file=sys.stderr)
            failed.append({"doc_id": doc_id, "stage": "delete", "error": str(exc)})

    upload_targets = [(d, current_docs[d]) for d in (add_ids + update_ids) if d in current_docs]
    for offset in range(0, len(upload_targets), UPLOAD_BATCH_SIZE):
        batch_targets = upload_targets[offset:offset + UPLOAD_BATCH_SIZE]
        attached, batch_failures = _upload_doc_batch(client, vector_store_id, batch_targets)
        failed.extend(batch_failures)
        docs_by_id = dict(batch_targets)
        for doc_id, file_id in attached.items():
            doc = docs_by_id[doc_id]
            next_docs[doc_id] = {k: v for k, v in doc.items() if k != "_rendered"} | {
                "file_id": file_id,
                "vector_store_file_id": file_id,
                "indexed_at": _utc_now(),
            }
            uploaded_count += 1
        print(
            f"  [{org_key}] batch {offset // UPLOAD_BATCH_SIZE + 1} "
            f"attached={len(attached)} failed={len(batch_failures)}"
        )

    org_state.update({
        "org_label": label,
        "vector_store_id": vector_store_id,
        "docs": next_docs,
        "doc_count_indexed": len(next_docs),
        "updated_at": _utc_now(),
        "last_sync": {
            "planned_add": len(add_ids),
            "planned_update": len(update_ids),
            "planned_remove": len(remove_ids),
            "uploaded": uploaded_count,
            "deleted": deleted_count,
            "failed_count": len(failed),
            "sync_mode": "rebuild" if (force_rebuild or created_new) else "incremental",
            "status": "completed",
        },
    })
    stores[org_key] = org_state
    vector_state["version"] = 2
    vector_state["updated_at"] = _utc_now()

    _upload_json(bucket, VECTOR_STATE_BLOB, vector_state)
    print(f"  [{org_key}] state saved — indexed={len(next_docs)} uploaded={uploaded_count} deleted={deleted_count} failed={len(failed)}")

    if force_rebuild and replaced_id and replaced_id != vector_store_id:
        try:
            client.vector_stores.delete(replaced_id)
            print(f"  [{org_key}] deleted old store {replaced_id}")
        except Exception as e:
            print(f"  [{org_key}] WARNING: failed to delete old store {replaced_id} (resource leak): {e}", file=sys.stderr)

    return {"org_key": org_key, "uploaded": uploaded_count, "deleted": deleted_count, "failed": len(failed), "total": len(next_docs)}


# ─── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Sync corpus documents to OpenAI vector stores.")
    parser.add_argument("--org", default="", help="Sync a single org_key (e.g. sec, finra). Omit for all orgs.")
    parser.add_argument("--force-rebuild", action="store_true", help="Delete and recreate each vector store from scratch.")
    parser.add_argument("--dry-run", action="store_true", help="Plan only — no OpenAI or GCS writes.")
    args = parser.parse_args()

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    bucket_name = os.environ.get("GCS_BUCKET_NAME", "")
    credentials_raw = os.environ.get("GCS_CREDENTIALS_JSON", "")

    missing = [k for k, v in [("OPENAI_API_KEY", openai_key), ("GCS_BUCKET_NAME", bucket_name), ("GCS_CREDENTIALS_JSON", credentials_raw)] if not v]
    if missing:
        print(f"ERROR: missing env vars: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=openai_key)
    bucket = _get_bucket(bucket_name, credentials_raw)

    print("Loading corpus from GCS…")
    all_docs = _load_corpus(bucket)
    print(f"  {len(all_docs)} documents")

    enrichment_raw = _download(bucket, ENRICHMENT_BLOB) or {}
    enrichment_entries = enrichment_raw.get("entries", {}) or {}
    print(f"  {len(enrichment_entries)} enrichment entries")

    vector_state: dict = _download(bucket, VECTOR_STATE_BLOB) or {"version": 2, "updated_at": "", "stores": {}}
    vector_state.setdefault("stores", {})

    # Discover orgs from corpus + existing state
    corpus_orgs = {_org_key(str((d.get("metadata") or {}).get("source_kind", "") or "")) for d in all_docs} - {"other"}
    state_orgs = set(vector_state["stores"].keys())
    all_orgs = sorted(corpus_orgs | state_orgs)

    if args.org:
        if args.org not in all_orgs:
            print(f"ERROR: org '{args.org}' not found. Known orgs: {all_orgs}", file=sys.stderr)
            sys.exit(1)
        target_orgs = [args.org]
    else:
        target_orgs = all_orgs

    print(f"\nTarget orgs: {target_orgs}")
    if args.dry_run:
        print("DRY RUN — no changes will be made\n")

    for org_key in target_orgs:
        print(f"\n── {org_key} ──")
        _sync_org(
            client=client,
            bucket=bucket,
            vector_state=vector_state,
            all_docs=all_docs,
            enrichment_entries=enrichment_entries,
            org_key=org_key,
            force_rebuild=args.force_rebuild,
            dry_run=args.dry_run,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

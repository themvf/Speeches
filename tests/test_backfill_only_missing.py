"""Additive-repair mode for the GCS -> Neon document backfill.

The two stores diverged rather than one lagging: Neon holds rows from
neon-authoritative connectors that no longer write to GCS, while GCS holds
documents whose workflow lacked DATABASE_URL when they were ingested (FINRA
notice 25-06's comment letters being the case that surfaced it). A full
re-upsert repairs the second set by overwriting the first with staler copies,
so the repair has to be additive.
"""

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import backfill_neon_documents as backfill  # noqa: E402


def _doc(document_id, title=""):
    return {"metadata": {"document_id": document_id, "title": title}, "content": {"full_text": "x"}}


def test_select_missing_keeps_only_documents_neon_has_never_seen(monkeypatch):
    monkeypatch.setattr(
        backfill.neon_feeds, "get_existing_document_ids", lambda ids: {"a", "c"}
    )

    missing, present = backfill._select_missing_documents(
        [_doc("a"), _doc("b"), _doc("c"), _doc("d")]
    )

    assert [d["metadata"]["document_id"] for d in missing] == ["b", "d"]
    assert present == 2


def test_select_missing_dedupes_and_ignores_idless_records(monkeypatch):
    monkeypatch.setattr(backfill.neon_feeds, "get_existing_document_ids", lambda ids: set())

    missing, present = backfill._select_missing_documents(
        [_doc("a", "first"), _doc("a", "second"), {"metadata": {}}, "not-a-dict", _doc("")]
    )

    assert [d["metadata"]["document_id"] for d in missing] == ["a"]
    # Last write wins, matching the corpus merge order elsewhere in this file.
    assert missing[0]["metadata"]["title"] == "second"
    assert present == 0


def test_select_missing_batches_large_id_sets(monkeypatch):
    seen_batches = []

    def reader(ids):
        seen_batches.append(len(ids))
        return set()

    monkeypatch.setattr(backfill.neon_feeds, "get_existing_document_ids", reader)

    backfill._select_missing_documents([_doc(f"id-{i}") for i in range(2500)])

    assert sum(seen_batches) == 2500
    assert max(seen_batches) <= backfill.ID_COVERAGE_BATCH_SIZE


def test_only_missing_skips_the_verified_backfill_freeze_guard(monkeypatch):
    """A full re-upsert must stay blocked; an additive repair must not be."""
    calls = {"checkpoint": 0}

    def checkpoint(_name):
        calls["checkpoint"] += 1
        return {"status": "verified"}

    monkeypatch.setattr(backfill.neon_feeds, "get_migration_checkpoint", checkpoint)
    monkeypatch.setattr(
        backfill.core, "_load_streamlit_secrets", lambda: {}
    )
    monkeypatch.setattr(
        backfill.core, "_get_gcs_storage", lambda secrets: (None, "no gcs in test")
    )

    full = types.SimpleNamespace(dry_run=False, only_missing=False, force=False)
    with pytest.raises(RuntimeError, match="verified full backfill already exists"):
        backfill._run(full)
    assert calls["checkpoint"] == 1

    additive = types.SimpleNamespace(dry_run=False, only_missing=True, force=False)
    # Gets past the freeze guard and fails later, on GCS access.
    with pytest.raises(RuntimeError, match="GCS read access is required"):
        backfill._run(additive)
    assert calls["checkpoint"] == 1


def test_only_missing_also_scopes_the_enrichment_upsert(monkeypatch):
    """A 30-document repair must not re-upsert 20k enrichment rows."""
    monkeypatch.setattr(
        backfill.neon_feeds, "get_existing_document_ids", lambda ids: {"present"}
    )
    monkeypatch.setattr(backfill.core, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setattr(backfill.core, "_get_gcs_storage", lambda secrets: (object(), ""))
    monkeypatch.setattr(
        backfill, "_corpus_documents", lambda storage, include_speeches=False: [_doc("present"), _doc("missing")]
    )
    monkeypatch.setattr(
        backfill,
        "_corpus_enrichment_entries",
        lambda storage: {"present": {"status": "enriched"}, "missing": {"status": "enriched"}},
    )

    args = types.SimpleNamespace(
        dry_run=True,
        only_missing=True,
        force=False,
        limit=0,
        include_enrichment=True,
        include_speeches=False,
        batch_size=200,
        verify_sample=0,
    )
    summary = backfill._run(args)

    assert summary["planned_backfill_count"] == 1
    assert summary["already_present_in_neon"] == 1
    # Only the missing document's enrichment travels with it.
    assert summary["planned_enrichment_backfill_count"] == 1

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

import sync_neon_to_legacy_snapshots as sync


def _doc(
    document_id: str,
    url: str,
    full_text: str,
    *,
    title: str = "Title",
    source_kind: str = "newsapi_article",
):
    return {
        "metadata": {
            "document_id": document_id,
            "url": url,
            "title": title,
            "source_kind": source_kind,
        },
        "content": {"full_text": full_text},
    }


def _args(**overrides):
    values = {"dry_run": False, "batch_size": 2, "summary_path": ""}
    values.update(overrides)
    return SimpleNamespace(**values)


def _configure_run(monkeypatch, documents, entries, legacy_documents, legacy_entries):
    storage = object()
    monkeypatch.setattr(
        sync.neon_feeds,
        "get_document_records_by_source_kinds",
        lambda source_kinds, include_full_text: documents,
    )
    monkeypatch.setattr(sync.neon_feeds, "get_enrichment_entries", lambda ids: entries)
    monkeypatch.setattr(sync.core, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setattr(sync.core, "_require_neon_authoritative_ready", lambda: None)
    monkeypatch.setattr(sync.core, "_get_gcs_storage", lambda secrets: (storage, "ok"))
    monkeypatch.setattr(sync, "_load_custom_snapshot", lambda value: legacy_documents)
    monkeypatch.setattr(sync, "_load_enrichment_snapshot", lambda value: legacy_entries)
    monkeypatch.setattr(sync, "_ensure_pre_cutover_backups", lambda value: ["docs-backup", "enrichment-backup"])
    return storage


def test_document_merge_is_additive_and_never_deletes_unrelated_data():
    unrelated = _doc("keep-me", "https://example.com/keep", "Keep this body.")
    legacy_match = {
        "metadata": {
            "url": "https://example.com/story?utm_source=old",
            "title": "Old title",
            "legacy_flag": "retain-me",
        },
        "content": {
            "full_text": "Old body.",
            "html": "<p>legacy rendering</p>",
            "paragraphs": ["Old body."],
            "sentences": ["Old body."],
        },
        "legacy_analysis": {"score": 7},
    }
    payload = {
        "updated_at": "before",
        "documents": [unrelated, legacy_match],
        "legacy_top_level": "preserved",
    }
    incoming = _doc(
        "story-1",
        "https://example.com/story",
        "First paragraph.\n\nSecond paragraph is new!",
        title="New title",
    )

    merged, stats = sync.merge_documents(payload, [incoming])

    assert stats == {"seen": 1, "added": 0, "updated": 1, "unchanged": 0, "skipped": 0}
    assert merged["legacy_top_level"] == "preserved"
    assert merged["documents"][0] == unrelated
    updated = merged["documents"][1]
    assert updated["legacy_analysis"] == {"score": 7}
    assert updated["metadata"]["legacy_flag"] == "retain-me"
    assert updated["metadata"]["document_id"] == "story-1"
    assert updated["metadata"]["title"] == "New title"
    assert updated["content"]["html"] == "<p>legacy rendering</p>"
    assert updated["content"]["paragraphs"] == ["First paragraph.", "Second paragraph is new!"]
    assert updated["content"]["sentences"] == ["First paragraph.", "Second paragraph is new!"]
    # Inputs are not mutated while determining whether a write is needed.
    assert payload["documents"][1]["metadata"]["title"] == "Old title"


def test_empty_neon_body_does_not_erase_existing_full_text_or_arrays():
    existing = _doc("story-1", "https://example.com/story", "Existing body.")
    existing["content"].update(
        {"paragraphs": ["Existing body."], "sentences": ["Existing body."], "html": "keep"}
    )
    incoming = _doc("story-1", "https://example.com/story", "")

    merged, stats = sync.merge_documents({"documents": [existing]}, [incoming])

    assert stats["unchanged"] == 1
    assert merged["documents"][0] == existing


def test_empty_neon_metadata_does_not_erase_existing_values():
    existing = _doc("story-1", "https://example.com/story", "Existing body.", title="Edited title")
    existing["metadata"]["speaker"] = "Human Editor"
    incoming = _doc("story-1", "https://example.com/story", "Existing body.", title="")
    incoming["metadata"]["speaker"] = ""

    merged, stats = sync.merge_documents({"documents": [existing]}, [incoming])

    assert stats["unchanged"] == 1
    assert merged["documents"][0]["metadata"]["title"] == "Edited title"
    assert merged["documents"][0]["metadata"]["speaker"] == "Human Editor"


def test_identical_run_is_noop_and_enrichment_reads_are_batched(monkeypatch):
    documents = [
        _doc("d1", "https://example.com/1", "One."),
        _doc("d2", "https://example.com/2", "Two."),
        _doc("d3", "https://example.com/3", "Three."),
    ]
    # Existing arrays are retained rather than recomputed for unchanged text.
    for document in documents:
        body = document["content"]["full_text"]
        document["content"].update({"paragraphs": [body], "sentences": [body]})
    entries = {
        "d1": {"status": "enriched", "enrichment": {"summary": "One"}},
        "d2": {"status": "enriched", "enrichment": {"summary": "Two"}},
        "d3": {"status": "enriched", "enrichment": {"summary": "Three"}},
    }
    enrichment_reader = MagicMock(
        side_effect=lambda ids: {document_id: entries[document_id] for document_id in ids}
    )
    monkeypatch.setattr(
        sync.neon_feeds,
        "get_document_records_by_source_kinds",
        lambda source_kinds, include_full_text: documents,
    )
    monkeypatch.setattr(sync.neon_feeds, "get_enrichment_entries", enrichment_reader)
    monkeypatch.setattr(sync.core, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setattr(sync.core, "_require_neon_authoritative_ready", lambda: None)
    monkeypatch.setattr(sync.core, "_get_gcs_storage", lambda secrets: (object(), "ok"))
    load_docs = MagicMock(return_value={"updated_at": "old", "documents": documents})
    load_enrichments = MagicMock(
        return_value={"version": 1, "updated_at": "old", "entries": entries}
    )
    save_docs = MagicMock()
    save_enrichments = MagicMock()
    monkeypatch.setattr(sync, "_load_custom_snapshot", load_docs)
    monkeypatch.setattr(sync, "_load_enrichment_snapshot", load_enrichments)
    monkeypatch.setattr(sync, "_save_custom_snapshot", save_docs)
    monkeypatch.setattr(sync, "_save_enrichment_snapshot", save_enrichments)

    summary = sync._run(_args(batch_size=2))

    assert enrichment_reader.call_args_list == [call(["d1", "d2"]), call(["d3"])]
    load_docs.assert_called_once()
    load_enrichments.assert_called_once()
    save_docs.assert_not_called()
    save_enrichments.assert_not_called()
    assert summary["planned_document_write"] is False
    assert summary["planned_enrichment_write"] is False
    assert summary["wrote_documents"] is False
    assert summary["wrote_enrichments"] is False


def test_dry_run_reports_changes_without_uploading(monkeypatch):
    document = _doc("d1", "https://example.com/1", "New body.")
    _configure_run(
        monkeypatch,
        [document],
        {"d1": {"status": "enriched"}},
        {"documents": []},
        {"entries": {}},
    )
    save_docs = MagicMock()
    save_enrichments = MagicMock()
    monkeypatch.setattr(sync, "_save_custom_snapshot", save_docs)
    monkeypatch.setattr(sync, "_save_enrichment_snapshot", save_enrichments)

    summary = sync._run(_args(dry_run=True))

    assert summary["planned_document_write"] is True
    assert summary["planned_enrichment_write"] is True
    save_docs.assert_not_called()
    save_enrichments.assert_not_called()


def test_document_generation_conflict_stops_before_enrichment_save(monkeypatch):
    events = []
    document = _doc("d1", "https://example.com/1", "New body.")
    monkeypatch.setattr(
        sync.neon_feeds,
        "get_document_records_by_source_kinds",
        lambda source_kinds, include_full_text: events.append("read_neon_documents") or [document],
    )
    monkeypatch.setattr(
        sync.neon_feeds,
        "get_enrichment_entries",
        lambda ids: events.append("read_neon_enrichments") or {"d1": {"status": "enriched"}},
    )
    monkeypatch.setattr(sync.core, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setattr(sync.core, "_require_neon_authoritative_ready", lambda: None)
    monkeypatch.setattr(sync.core, "_get_gcs_storage", lambda secrets: (object(), "ok"))
    monkeypatch.setattr(
        sync,
        "_load_custom_snapshot",
        lambda storage: events.append("load_legacy_documents") or {"documents": []},
    )
    monkeypatch.setattr(
        sync,
        "_load_enrichment_snapshot",
        lambda storage: events.append("load_legacy_enrichments") or {"entries": {}},
    )
    monkeypatch.setattr(
        sync,
        "_ensure_pre_cutover_backups",
        lambda storage: events.append("backup_legacy_snapshots") or ["one", "two"],
    )

    def reject_document_save(storage, payload):
        events.append("save_legacy_documents")
        raise RuntimeError("another writer changed this blob")

    enrichment_save = MagicMock(side_effect=lambda *args, **kwargs: events.append("save_enrichment"))
    monkeypatch.setattr(sync, "_save_custom_snapshot", reject_document_save)
    monkeypatch.setattr(sync, "_save_enrichment_snapshot", enrichment_save)

    with pytest.raises(RuntimeError, match="another writer changed this blob"):
        sync._run(_args())

    assert events == [
        "read_neon_documents",
        "read_neon_enrichments",
        "load_legacy_documents",
        "load_legacy_enrichments",
        "backup_legacy_snapshots",
        "save_legacy_documents",
    ]
    enrichment_save.assert_not_called()


def test_enrichment_merge_preserves_unrelated_entries_and_legacy_fields():
    legacy = {
        "version": 1,
        "entries": {
            "unrelated": {"status": "enriched", "keep": True},
            "d1": {
                "status": "enriched",
                "legacy_review": {"reviewer": "human", "note": "keep"},
                "enrichment": {"summary": "Old", "legacy_detail": "keep"},
            },
        },
    }
    incoming = {
        "d1": {"status": "reviewed", "enrichment": {"summary": "New"}},
        "d2": {"status": "enriched", "enrichment": {"summary": "Added"}},
    }

    merged, stats = sync.merge_enrichments(legacy, incoming)

    assert stats == {"seen": 2, "added": 1, "updated": 1, "unchanged": 0, "skipped": 0}
    assert merged["entries"]["unrelated"] == legacy["entries"]["unrelated"]
    assert merged["entries"]["d1"]["legacy_review"]["note"] == "keep"
    assert merged["entries"]["d1"]["enrichment"] == {
        "summary": "New",
        "legacy_detail": "keep",
    }
    assert merged["entries"]["d2"]["status"] == "enriched"


def test_enrichment_merge_preserves_existing_human_review():
    legacy = {
        "entries": {
            "d1": {
                "status": "reviewed",
                "review": {
                    "decision": "accepted",
                    "notes": "human decision",
                    "reviewed_at": "2026-07-14T12:00:00Z",
                },
                "enrichment": {"summary": "Old"},
            }
        }
    }
    incoming = {
        "d1": {
            "status": "enriched",
            "review": {"decision": "pending", "notes": ""},
            "enrichment": {"summary": "New"},
        }
    }

    merged, stats = sync.merge_enrichments(legacy, incoming)

    assert stats["updated"] == 1
    assert merged["entries"]["d1"]["status"] == "reviewed"
    assert merged["entries"]["d1"]["review"]["decision"] == "accepted"
    assert merged["entries"]["d1"]["enrichment"]["summary"] == "New"


def test_pre_cutover_backup_is_server_side_and_generation_guarded(monkeypatch):
    source = object()
    destination = MagicMock()
    destination.exists.return_value = False
    bucket = MagicMock()
    bucket.blob.side_effect = lambda name, **kwargs: (
        destination if name.startswith(sync.PRE_CUTOVER_BACKUP_PREFIX) else source
    )
    storage = SimpleNamespace(bucket=bucket)
    monkeypatch.setitem(sync.core._BLOB_GENERATIONS, sync.core.CUSTOM_DOCS_BLOB_NAME, 123)

    backup_name = sync._ensure_pre_cutover_backup(storage, sync.core.CUSTOM_DOCS_BLOB_NAME)

    assert backup_name.endswith("/custom_documents.json")
    bucket.copy_blob.assert_called_once_with(
        source,
        bucket,
        new_name=backup_name,
        source_generation=123,
        if_source_generation_match=123,
        if_generation_match=0,
    )

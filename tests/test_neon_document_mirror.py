"""Tests for Phase 1 of migrating off custom_documents.json: the Neon
`documents` mirror table and its best-effort, non-blocking write path.

No real Postgres connection is used - psycopg2's connect/cursor are mocked,
matching this repo's existing pattern of testing storage-layer code against
fakes rather than live infrastructure (see test_json_store_concurrency.py).
"""

from unittest.mock import MagicMock, patch

import neon_feeds
import pytest
import run_financial_news_pipeline as core


def _doc_record(doc_id="abc123", **extra_metadata):
    metadata = {
        "document_id": doc_id,
        "title": "Test Document",
        "speaker": "Jane Regulator",
        "organization": "SEC",
        "doc_type": "Speech",
        "source_kind": "sec_speech",
        "url": "https://example.com/doc",
        "published_date": "July 9, 2026",
        "word_count": 42,
        **extra_metadata,
    }
    return {"metadata": metadata, "content": {"full_text": "Some substantive remarks."}}


def test_mirror_document_skips_records_without_document_id():
    with patch.object(neon_feeds, "_get_conn") as mock_conn:
        neon_feeds.mirror_document({"metadata": {}, "content": {}})
        mock_conn.assert_not_called()


def test_mirror_document_upserts_with_expected_shape(monkeypatch):
    monkeypatch.setattr(neon_feeds, "_DOCUMENTS_SCHEMA_ENSURED", True)  # skip schema DDL for this test
    cursor = MagicMock()
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.cursor.return_value.__enter__.return_value = cursor

    with patch.object(neon_feeds, "_get_conn", return_value=conn):
        neon_feeds.mirror_document(_doc_record())

    assert cursor.execute.called
    sql, params = cursor.execute.call_args[0]
    assert "INSERT INTO documents" in sql
    assert "ON CONFLICT (document_id) DO UPDATE" in sql
    assert params[0] == "abc123"  # document_id
    assert params[1] == "Test Document"  # title
    conn.commit.assert_called_once()


def test_ensure_documents_schema_runs_once(monkeypatch):
    monkeypatch.setattr(neon_feeds, "_DOCUMENTS_SCHEMA_ENSURED", False)
    cursor = MagicMock()
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.cursor.return_value.__enter__.return_value = cursor

    with patch.object(neon_feeds, "_get_conn", return_value=conn) as mock_get_conn:
        neon_feeds._ensure_documents_schema()
        neon_feeds._ensure_documents_schema()
        # Second call must be a no-op: schema DDL only runs once per process.
        assert mock_get_conn.call_count == 1


def test_pipeline_mirror_helper_is_noop_without_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with patch("neon_feeds.mirror_document") as mock_mirror:
        core._mirror_document_to_neon_best_effort(_doc_record())
        mock_mirror.assert_not_called()


def test_pipeline_mirror_helper_swallows_failure_and_never_raises(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://fake")
    with patch("neon_feeds.mirror_document", side_effect=RuntimeError("connection refused")):
        # Must not raise.
        core._mirror_document_to_neon_best_effort(_doc_record())


def test_upsert_custom_document_record_unaffected_by_mirror_outcome(monkeypatch):
    """The primary in-memory upsert (insert vs replace, returned bool) must
    behave identically regardless of whether the Neon mirror succeeds,
    fails, or is skipped entirely."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    payload = {"documents": []}

    replaced_first = core._upsert_custom_document_record(payload, _doc_record(doc_id="x1", title="First"))
    assert replaced_first is False
    assert len(payload["documents"]) == 1

    replaced_second = core._upsert_custom_document_record(payload, _doc_record(doc_id="x1", title="Updated"))
    assert replaced_second is True
    assert len(payload["documents"]) == 1
    assert payload["documents"][0]["metadata"]["title"] == "Updated"

    # Upserts only queue mirrors; a rejected authoritative save must never
    # write Neon ahead of GCS.
    monkeypatch.setenv("DATABASE_URL", "postgres://fake")
    with patch("neon_feeds.mirror_document") as mock_mirror:
        replaced_third = core._upsert_custom_document_record(
            payload, _doc_record(doc_id="x2", url="https://example.com/other-doc")
        )
        mock_mirror.assert_not_called()
    assert replaced_third is False
    assert len(payload["documents"]) == 2


def test_pipeline_mirrors_only_after_authoritative_save(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://fake")
    payload = {"documents": []}
    record = _doc_record(doc_id="committed")
    core._upsert_custom_document_record(payload, record)

    with (
        patch.object(core, "_save_json_store") as mock_save,
        patch.object(core, "_mirror_document_to_neon_best_effort") as mock_mirror,
    ):
        core._save_custom_documents(None, payload)

    mock_save.assert_called_once()
    mock_mirror.assert_called_once_with(record)
    assert payload[core.PENDING_NEON_MIRROR_FIELD] == []


def test_pipeline_does_not_mirror_when_authoritative_save_fails(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://fake")
    payload = {"documents": []}
    core._upsert_custom_document_record(payload, _doc_record(doc_id="rejected"))

    with (
        patch.object(core, "_save_json_store", side_effect=RuntimeError("generation mismatch")),
        patch.object(core, "_mirror_document_to_neon_best_effort") as mock_mirror,
    ):
        with pytest.raises(RuntimeError, match="generation mismatch"):
            core._save_custom_documents(None, payload)

    mock_mirror.assert_not_called()

"""Tests for Phase 2 of migrating off custom_documents.json: the bulk
mirror_documents_batch()/count_documents()/get_document() helpers in
neon_feeds.py, and the backfill_neon_documents.py driver script.

No real Postgres connection is used - psycopg2 is mocked, matching this
repo's existing pattern (see test_neon_document_mirror.py).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import backfill_neon_documents as backfill
import neon_feeds


def _doc(doc_id: str, full_text: str = "Some substantive remarks.", **extra_metadata):
    metadata = {
        "document_id": doc_id,
        "title": f"Title {doc_id}",
        "source_kind": "sec_speech",
        **extra_metadata,
    }
    return {"metadata": metadata, "content": {"full_text": full_text}}


def _mock_conn():
    cursor = MagicMock()
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.cursor.return_value.__enter__.return_value = cursor
    return conn, cursor


# ─── neon_feeds.mirror_documents_batch ──────────────────────────────────────

def test_mirror_documents_batch_skips_records_without_document_id():
    with patch.object(neon_feeds, "_get_conn") as mock_conn:
        result = neon_feeds.mirror_documents_batch([{"metadata": {}, "content": {}}])
        assert result == 0
        mock_conn.assert_not_called()


def test_mirror_documents_batch_empty_list_is_noop():
    with patch.object(neon_feeds, "_get_conn") as mock_conn:
        assert neon_feeds.mirror_documents_batch([]) == 0
        mock_conn.assert_not_called()


def test_mirror_documents_batch_upserts_all_valid_records(monkeypatch):
    monkeypatch.setattr(neon_feeds, "_DOCUMENTS_SCHEMA_ENSURED", True)
    conn, cursor = _mock_conn()
    records = [_doc("d1"), _doc("d2"), {"metadata": {}, "content": {}}, _doc("d3")]

    with patch.object(neon_feeds, "_get_conn", return_value=conn):
        with patch("psycopg2.extras.execute_values") as mock_execute_values:
            result = neon_feeds.mirror_documents_batch(records)

    assert result == 3  # the record with no document_id is dropped
    assert mock_execute_values.called
    args, kwargs = mock_execute_values.call_args
    passed_cursor, sql, rows = args[0], args[1], args[2]
    assert passed_cursor is cursor
    assert "INSERT INTO documents" in sql
    assert "ON CONFLICT (document_id) DO UPDATE" in sql
    assert [row[0] for row in rows] == ["d1", "d2", "d3"]
    conn.commit.assert_called_once()


def test_mirror_document_and_batch_use_the_same_column_mapping(monkeypatch):
    """The single-record and bulk upsert paths must stay in sync - this is
    exactly the kind of drift a shared row-builder is meant to prevent."""
    record = _doc("shared-doc", full_text="Body text", title="Shared Title")
    single_row = neon_feeds._document_record_to_row(record)
    batch_row = neon_feeds._document_record_to_row(record)
    # psycopg2.extras.Json wrappers don't compare equal by value even when
    # wrapping identical dicts, so compare everything else directly and the
    # JSONB payload's underlying dict separately.
    assert single_row[:-1] == batch_row[:-1]
    assert single_row[-1].adapted == batch_row[-1].adapted
    assert single_row[0] == "shared-doc"
    assert single_row[9] == "Body text"


# ─── neon_feeds.count_documents / get_document ──────────────────────────────

def test_count_documents_returns_int(monkeypatch):
    monkeypatch.setattr(neon_feeds, "_DOCUMENTS_SCHEMA_ENSURED", True)
    conn, cursor = _mock_conn()
    cursor.fetchone.return_value = {"count": 42}

    with patch.object(neon_feeds, "_get_conn", return_value=conn):
        assert neon_feeds.count_documents() == 42


def test_get_document_returns_row_or_none(monkeypatch):
    monkeypatch.setattr(neon_feeds, "_DOCUMENTS_SCHEMA_ENSURED", True)
    conn, cursor = _mock_conn()
    cursor.fetchone.return_value = {"document_id": "d1", "full_text": "hi"}

    with patch.object(neon_feeds, "_get_conn", return_value=conn):
        row = neon_feeds.get_document("d1")
    assert row["document_id"] == "d1"

    cursor.fetchone.return_value = None
    with patch.object(neon_feeds, "_get_conn", return_value=conn):
        assert neon_feeds.get_document("missing") is None


# ─── backfill_neon_documents.py ─────────────────────────────────────────────

def _args(**overrides):
    base = dict(dry_run=False, limit=0, batch_size=2, verify_sample=0, summary_path="")
    base.update(overrides)
    return SimpleNamespace(**base)


def test_batched_splits_into_expected_chunk_sizes():
    assert backfill._batched([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert backfill._batched([], 2) == []
    assert backfill._batched([1], 10) == [[1]]


def test_dry_run_reports_counts_without_writing(monkeypatch):
    docs = [_doc("d1"), _doc("d2"), _doc("d3")]
    monkeypatch.setattr(backfill.core, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setattr(backfill.core, "_get_gcs_storage", lambda secrets: (object(), "ok"))
    monkeypatch.setattr(backfill.core, "_load_custom_documents", lambda storage: {"documents": docs})

    with patch.object(neon_feeds, "mirror_documents_batch") as mock_batch:
        summary = backfill._run(_args(dry_run=True))

    mock_batch.assert_not_called()
    assert summary["dry_run"] is True
    assert summary["corpus_document_count"] == 3
    assert summary["planned_backfill_count"] == 3


def test_run_backfills_in_batches_and_reports_totals(monkeypatch):
    docs = [_doc("d1"), _doc("d2"), _doc("d3"), _doc("d4"), _doc("d5")]
    monkeypatch.setattr(backfill.core, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setattr(backfill.core, "_get_gcs_storage", lambda secrets: (object(), "ok"))
    monkeypatch.setattr(backfill.core, "_load_custom_documents", lambda storage: {"documents": docs})

    with patch.object(neon_feeds, "mirror_documents_batch", side_effect=lambda batch: len(batch)) as mock_batch:
        summary = backfill._run(_args(batch_size=2, verify_sample=0))

    assert mock_batch.call_count == 3  # batches of 2, 2, 1
    assert summary["upserted_total"] == 5
    assert summary["failed_batch_count"] == 0
    assert summary["ok"] is True


def test_run_continues_past_a_failed_batch_and_reports_it(monkeypatch):
    docs = [_doc("d1"), _doc("d2"), _doc("d3"), _doc("d4")]
    monkeypatch.setattr(backfill.core, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setattr(backfill.core, "_get_gcs_storage", lambda secrets: (object(), "ok"))
    monkeypatch.setattr(backfill.core, "_load_custom_documents", lambda storage: {"documents": docs})

    call_count = {"n": 0}

    def flaky_batch(batch):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("connection reset")
        return len(batch)

    with patch.object(neon_feeds, "mirror_documents_batch", side_effect=flaky_batch):
        summary = backfill._run(_args(batch_size=2, verify_sample=0))

    # First batch (2 docs) failed, second batch (2 docs) succeeded.
    assert summary["upserted_total"] == 2
    assert summary["failed_batch_count"] == 1
    assert summary["ok"] is False


def test_run_raises_when_gcs_storage_unavailable(monkeypatch):
    monkeypatch.setattr(backfill.core, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setattr(backfill.core, "_get_gcs_storage", lambda secrets: (None, "credentials missing"))

    try:
        backfill._run(_args())
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "credentials missing" in str(exc)


def test_verify_flags_row_count_and_sample_mismatches(monkeypatch):
    docs = [_doc("d1", full_text="12345"), _doc("d2", full_text="abcdefghij")]

    def fake_get_document(doc_id):
        if doc_id == "d1":
            return {"full_text": "12345"}  # matches
        return {"full_text": "short"}  # mismatched length for d2

    with patch.object(neon_feeds, "count_documents", return_value=2):
        with patch.object(neon_feeds, "get_document", side_effect=fake_get_document):
            result = backfill._verify(docs, sample_size=2)

    assert result["corpus_document_count"] == 2
    assert result["neon_row_count"] == 2
    assert result["row_count_matches"] is True
    assert result["sample_checked"] == 2
    assert len(result["sample_mismatches"]) == 1
    assert result["sample_mismatches"][0]["doc_id"] == "d2"


def test_verify_flags_missing_document_in_neon():
    docs = [_doc("only-doc")]

    with patch.object(neon_feeds, "count_documents", return_value=0):
        with patch.object(neon_feeds, "get_document", return_value=None):
            result = backfill._verify(docs, sample_size=1)

    assert result["row_count_matches"] is False
    assert result["sample_mismatches"] == [{"doc_id": "only-doc", "issue": "missing_in_neon"}]

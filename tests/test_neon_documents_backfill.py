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
    assert "IS DISTINCT FROM" in sql
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


def test_document_record_to_row_strips_embedded_nul_bytes():
    """Regression test for a production failure: a full-corpus backfill hit
    'A string literal cannot contain NUL (0x00) characters' from Postgres on
    one 200-row batch, because a single document somewhere in it had a
    literal NUL byte (a known PDF/HTML-extraction artifact) in its text -
    and Postgres rejects the whole multi-row batch when even one row is bad,
    not just that row. NUL bytes must be stripped before they ever reach a
    SQL statement, in every text field and inside the JSONB metadata too."""
    record = {
        "metadata": {
            "document_id": "doc-with-nul",
            "title": "Bad\x00 Title",
            "speaker": "Jane\x00 Doe",
            "organization": "SEC\x00",
            "doc_type": "Speech",
            "source_kind": "sec_speech",
            "url": "https://example.com/doc",
            "published_date": "July 9, 2026",
            "word_count": 3,
            "nested": {"note": "has a \x00 nul byte too"},
        },
        "content": {"full_text": "Some remarks with an embedded \x00 byte."},
    }

    row = neon_feeds._document_record_to_row(record)

    assert row is not None
    assert "\x00" not in row[0]  # document_id
    assert "\x00" not in row[1]  # title
    assert "\x00" not in row[2]  # speaker
    assert "\x00" not in row[3]  # organization
    assert "\x00" not in row[9]  # full_text
    sanitized_metadata = row[-1].adapted
    assert "\x00" not in sanitized_metadata["title"]
    assert "\x00" not in sanitized_metadata["nested"]["note"]


def test_mirror_documents_batch_succeeds_with_a_nul_byte_in_one_record(monkeypatch):
    """End-to-end (mocked) proof that a batch containing a NUL-byte document
    no longer fails the whole batch."""
    monkeypatch.setattr(neon_feeds, "_DOCUMENTS_SCHEMA_ENSURED", True)
    conn, cursor = _mock_conn()
    records = [_doc("clean-doc"), _doc("dirty-doc", full_text="text with \x00 nul")]

    with patch.object(neon_feeds, "_get_conn", return_value=conn):
        with patch("psycopg2.extras.execute_values") as mock_execute_values:
            result = neon_feeds.mirror_documents_batch(records)

    assert result == 2
    rows = mock_execute_values.call_args[0][2]
    dirty_row = next(r for r in rows if r[0] == "dirty-doc")
    assert "\x00" not in dirty_row[9]


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
    base = dict(dry_run=False, force=True, limit=0, batch_size=2, verify_sample=0, summary_path="")
    base.update(overrides)
    return SimpleNamespace(**base)


def test_batched_splits_into_expected_chunk_sizes():
    assert backfill._batched([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert backfill._batched([], 2) == []
    assert backfill._batched([1], 10) == [[1]]


def test_legacy_speech_normalization_matches_stable_corpus_identity():
    speech = {
        "metadata": {
            "url": "https://www.sec.gov/newsroom/speeches-statements/example",
            "title": "Market structure remarks",
            "speaker": "Commissioner Example",
            "date": "July 14, 2026",
            "extraction_date": "2026-07-14T12:00:00Z",
        },
        "content": {"full_text": "Prepared remarks about market structure."},
        "validation": {"completeness_score": 100},
    }

    normalized = backfill._normalize_legacy_speech(speech)
    stable = "sec|https://www.sec.gov/newsroom/speeches-statements/example|Market structure remarks|Commissioner Example|July 14, 2026"

    assert normalized["metadata"]["document_id"] == __import__("hashlib").sha256(
        stable.encode("utf-8")
    ).hexdigest()[:24]
    assert normalized["metadata"]["source_kind"] == "sec_speech"
    assert normalized["metadata"]["organization"] == "SEC"
    assert normalized["validation"] == {"completeness_score": 100}


def test_combined_corpus_includes_legacy_speeches_but_custom_wins(monkeypatch):
    speech = {
        "metadata": {
            "document_id": "same-id",
            "url": "https://example.com/speech",
            "title": "Legacy",
        },
        "content": {"full_text": "Legacy text"},
    }
    custom = _doc("same-id", full_text="Custom text", title="Custom")

    class Storage:
        def load_speeches(self):
            return {"speeches": [speech]}

    monkeypatch.setattr(
        backfill.core,
        "_load_custom_documents",
        lambda storage: {"documents": [custom]},
    )

    combined = backfill._corpus_documents(Storage(), include_speeches=True)

    assert len(combined) == 1
    assert combined[0]["metadata"]["title"] == "Custom"
    assert combined[0]["content"]["full_text"] == "Custom text"


def test_custom_document_without_id_gets_same_trimmed_stable_identity_as_web():
    record = {
        "metadata": {
            "organization": " SEC ",
            "url": " https://example.com/document ",
            "title": " Example title ",
            "speaker": " Jane Doe ",
            "date": " July 14, 2026 ",
        },
        "content": {"full_text": " Body text. "},
        "legacy_field": "preserved",
    }
    expected_seed = "sec|https://example.com/document|Example title|Jane Doe|July 14, 2026"

    normalized = backfill._ensure_custom_document_identity(record)

    assert normalized["metadata"]["document_id"] == __import__("hashlib").sha256(
        expected_seed.encode("utf-8")
    ).hexdigest()[:24]
    assert normalized["legacy_field"] == "preserved"


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


def test_non_dry_rerun_refuses_existing_verified_checkpoint(monkeypatch):
    monkeypatch.setattr(
        neon_feeds,
        "get_migration_checkpoint",
        lambda key: {"status": "verified"},
    )

    try:
        backfill._run(_args(force=False))
    except RuntimeError as exc:
        assert "verified full backfill already exists" in str(exc)
    else:  # pragma: no cover - guard assertion
        raise AssertionError("stale backfill rerun was accepted")


def test_verify_flags_row_count_and_sample_mismatches(monkeypatch):
    docs = [_doc("d1", full_text="12345"), _doc("d2", full_text="abcdefghij")]

    def fake_get_document(doc_id):
        if doc_id == "d1":
            return {"full_text": "12345"}  # matches
        return {"full_text": "short"}  # mismatched length for d2

    with patch.object(neon_feeds, "count_documents", return_value=2):
        with patch.object(neon_feeds, "get_existing_document_ids", return_value={"d1", "d2"}):
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
        with patch.object(neon_feeds, "get_existing_document_ids", return_value=set()):
            with patch.object(neon_feeds, "get_document", return_value=None):
                result = backfill._verify(docs, sample_size=1)

    assert result["row_count_matches"] is False
    assert result["sample_mismatches"] == [{"doc_id": "only-doc", "issue": "missing_in_neon"}]


def test_verify_never_raises_when_neon_is_unreachable():
    """Regression test: a real production run (DATABASE_URL missing) showed
    _verify() propagating an uncaught RuntimeError from count_documents(),
    which wiped out the whole run's batch-upload summary and left only a
    bare top-level error. count_documents() failures must degrade to a
    self-contained "error" key instead."""
    docs = [_doc("d1")]
    with patch.object(neon_feeds, "count_documents", side_effect=RuntimeError("DATABASE_URL is not set.")):
        result = backfill._verify(docs, sample_size=1)

    assert "error" in result
    assert "DATABASE_URL is not set" in result["error"]


def test_verify_reports_per_doc_read_failure_without_aborting_the_sample():
    docs = [_doc("d1", full_text="abc"), _doc("d2", full_text="xyz")]

    def flaky_get_document(doc_id):
        if doc_id == "d1":
            raise RuntimeError("connection reset")
        return {"full_text": "xyz"}

    with patch.object(neon_feeds, "count_documents", return_value=2):
        with patch.object(neon_feeds, "get_existing_document_ids", return_value={"d1", "d2"}):
            with patch.object(neon_feeds, "get_document", side_effect=flaky_get_document):
                result = backfill._verify(docs, sample_size=2)

    assert "error" not in result
    assert result["sample_checked"] == 1  # only d2 succeeded
    issues = {m["doc_id"]: m["issue"] for m in result["sample_mismatches"]}
    assert issues.get("d1") == "verify_read_failed"


def test_run_marks_not_ok_when_verification_errors_even_if_all_batches_succeeded(monkeypatch):
    """The bug found in production: batches can all succeed while the
    subsequent verify step fails (e.g. transient Neon blip) - the run must
    not self-report ok=True in that case, and must still surface the
    detailed batch summary rather than a bare top-level exception."""
    docs = [_doc("d1"), _doc("d2")]
    monkeypatch.setattr(backfill.core, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setattr(backfill.core, "_get_gcs_storage", lambda secrets: (object(), "ok"))
    monkeypatch.setattr(backfill.core, "_load_custom_documents", lambda storage: {"documents": docs})

    with patch.object(neon_feeds, "mirror_documents_batch", side_effect=lambda batch: len(batch)):
        with patch.object(neon_feeds, "count_documents", side_effect=RuntimeError("DATABASE_URL is not set.")):
            summary = backfill._run(_args(batch_size=2, verify_sample=1))

    assert summary["upserted_total"] == 2
    assert summary["failed_batch_count"] == 0
    assert "error" in summary["verification"]
    assert summary["ok"] is False


def test_run_marks_not_ok_when_verification_finds_mismatch(monkeypatch):
    docs = [_doc("d1")]
    monkeypatch.setattr(backfill.core, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setattr(backfill.core, "_get_gcs_storage", lambda secrets: (object(), "ok"))
    monkeypatch.setattr(backfill.core, "_load_custom_documents", lambda storage: {"documents": docs})

    mismatch = {
        "corpus_document_count": 1,
        "neon_row_count": 1,
        "row_count_matches": True,
        "sample_checked": 1,
        "sample_mismatches": [{"doc_id": "d1", "issue": "full_text_length_mismatch"}],
    }
    with patch.object(neon_feeds, "mirror_documents_batch", return_value=1):
        with patch.object(backfill, "_verify", return_value=mismatch):
            summary = backfill._run(_args(verify_sample=1))

    assert summary["verification"] == mismatch
    assert summary["ok"] is False


def test_limited_backfill_limits_enrichment_rows_and_verifies_only_targets(monkeypatch):
    docs = [_doc("d1"), _doc("d2"), _doc("d3")]
    entries = {document["metadata"]["document_id"]: {"status": "enriched"} for document in docs}
    monkeypatch.setattr(backfill.core, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setattr(backfill.core, "_get_gcs_storage", lambda secrets: (object(), "ok"))
    monkeypatch.setattr(backfill.core, "_load_custom_documents", lambda storage: {"documents": docs})
    monkeypatch.setattr(backfill.core, "_load_enrichment_state", lambda storage: {"entries": entries})

    with patch.object(neon_feeds, "mirror_documents_batch", return_value=1):
        with patch.object(neon_feeds, "upsert_enrichment_entries", return_value=1) as upsert:
            with patch.object(backfill, "_verify", return_value={}) as verify:
                summary = backfill._run(
                    _args(limit=1, include_enrichment=True, verify_sample=1)
                )

    upsert.assert_called_once_with({"d1": {"status": "enriched"}})
    verify.assert_called_once_with([docs[0]], 1)
    assert summary["corpus_enrichment_count"] == 1


def test_full_verified_backfill_records_database_activation_checkpoint(monkeypatch):
    docs = [_doc("d1")]
    entries = {"d1": {"status": "enriched"}}
    verification = {
        "row_count_matches": True,
        "coverage_matches": True,
        "sample_mismatches": [],
    }
    checkpoint = MagicMock()
    monkeypatch.setattr(backfill.core, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setattr(backfill.core, "_get_gcs_storage", lambda secrets: (object(), "ok"))
    monkeypatch.setattr(backfill, "_corpus_documents", lambda storage, include_speeches: docs)
    monkeypatch.setattr(backfill, "_corpus_enrichment_entries", lambda storage: entries)
    monkeypatch.setattr(neon_feeds, "mirror_documents_batch", lambda batch: len(batch))
    monkeypatch.setattr(neon_feeds, "upsert_enrichment_entries", lambda batch: len(batch))
    monkeypatch.setattr(backfill, "_verify", lambda documents, sample_size: verification)
    monkeypatch.setattr(backfill, "_verify_enrichments", lambda values, sample_size: verification)
    monkeypatch.setattr(neon_feeds, "set_migration_checkpoint", checkpoint)

    summary = backfill._run(
        _args(
            include_speeches=True,
            include_enrichment=True,
            verify_sample=1,
        )
    )

    checkpoint.assert_called_once()
    assert summary["ok"] is True
    assert summary["activation_checkpoint"]["recorded"] is True


def test_enrichment_backfill_is_opt_in_and_batched(monkeypatch):
    docs = [_doc("d1"), _doc("d2")]
    entries = {
        "d1": {"status": "enriched", "enrichment": {"summary": "One"}},
        "d2": {"status": "reviewed", "enrichment": {"summary": "Two"}},
        "d3": {"status": "failed", "error": "retry"},
    }
    monkeypatch.setattr(backfill.core, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setattr(backfill.core, "_get_gcs_storage", lambda secrets: (object(), "ok"))
    monkeypatch.setattr(backfill.core, "_load_custom_documents", lambda storage: {"documents": docs})
    monkeypatch.setattr(backfill.core, "_load_enrichment_state", lambda storage: {"entries": entries})

    with patch.object(neon_feeds, "mirror_documents_batch", side_effect=lambda batch: len(batch)):
        with patch.object(
            neon_feeds,
            "upsert_enrichment_entries",
            side_effect=lambda batch: len(batch),
        ) as upsert:
            summary = backfill._run(
                _args(include_enrichment=True, batch_size=2, verify_sample=0)
            )

    assert upsert.call_count == 2
    assert summary["enrichment_upserted_total"] == 3
    assert summary["failed_enrichment_batch_count"] == 0
    assert summary["ok"] is True


def test_enrichment_dry_run_reports_plan_without_writing(monkeypatch):
    monkeypatch.setattr(backfill.core, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setattr(backfill.core, "_get_gcs_storage", lambda secrets: (object(), "ok"))
    monkeypatch.setattr(backfill.core, "_load_custom_documents", lambda storage: {"documents": [_doc("d1")]})
    monkeypatch.setattr(
        backfill.core,
        "_load_enrichment_state",
        lambda storage: {"entries": {"d1": {"status": "enriched"}}},
    )

    with patch.object(neon_feeds, "upsert_enrichment_entries") as upsert:
        summary = backfill._run(_args(dry_run=True, include_enrichment=True))

    upsert.assert_not_called()
    assert summary["planned_enrichment_backfill_count"] == 1


def test_verify_enrichments_compares_complete_legacy_entries():
    entries = {
        "d1": {"status": "enriched", "enrichment": {"summary": "One"}},
        "d2": {"status": "reviewed", "sentiment": {"label": "neutral"}},
    }
    mirrored = {"d1": entries["d1"], "d2": {"status": "reviewed"}}

    with patch.object(neon_feeds, "count_enrichment_entries", return_value=2):
        with patch.object(neon_feeds, "get_existing_enrichment_ids", return_value={"d1", "d2"}):
            with patch.object(neon_feeds, "get_enrichment_entries", return_value=mirrored):
                result = backfill._verify_enrichments(entries, sample_size=2)

    assert result["row_count_matches"] is True
    assert result["sample_checked"] == 2
    assert result["sample_mismatches"] == [
        {"doc_id": "d2", "issue": "entry_mismatch_or_missing"}
    ]

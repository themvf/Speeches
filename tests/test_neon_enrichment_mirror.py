"""Focused tests for bounded Neon document reads and enrichment row storage."""

from unittest.mock import MagicMock, patch

import neon_feeds


def _mock_conn():
    cursor = MagicMock()
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.cursor.return_value.__enter__.return_value = cursor
    return conn, cursor


def _entry(doc_id: str = "doc-1", **overrides):
    return {
        "doc_id": doc_id,
        "status": "enriched",
        "pipeline_version": "v1",
        "updated_at": "2026-07-14T12:00:00+00:00",
        "enrichment": {"summary": "A useful summary", "tags": ["Markets"]},
        **overrides,
    }


def _document_row(doc_id: str = "doc-1", **overrides):
    return {
        "document_id": doc_id,
        "title": "Database title",
        "speaker": "",
        "organization": "SEC",
        "doc_type": "Article",
        "source_kind": "financial_news",
        "url": f"https://example.com/{doc_id}",
        "published_date": "2026-07-14",
        "word_count": 2,
        "full_text": "full body",
        "metadata": {"document_id": doc_id, "title": "Metadata title"},
        **overrides,
    }


def test_ensure_enrichment_schema_is_idempotent(monkeypatch):
    monkeypatch.setattr(neon_feeds, "_ENRICHMENTS_SCHEMA_ENSURED", False)
    conn, cursor = _mock_conn()

    with patch.object(neon_feeds, "_get_conn", return_value=conn) as get_conn:
        neon_feeds._ensure_enrichments_schema()
        neon_feeds._ensure_enrichments_schema()

    assert get_conn.call_count == 1
    assert "CREATE TABLE IF NOT EXISTS document_enrichments" in cursor.execute.call_args_list[0].args[0]
    conn.commit.assert_called_once()


def test_upsert_enrichment_entries_is_bulk_additive_and_skips_identical_writes(monkeypatch):
    monkeypatch.setattr(neon_feeds, "_ENRICHMENTS_SCHEMA_ENSURED", True)
    monkeypatch.setattr(neon_feeds, "_database_url_is_configured", lambda: True)
    conn, cursor = _mock_conn()
    entries = {"doc-1": _entry("doc-1"), "doc-2": _entry("doc-2"), "bad": "not-a-dict"}

    with (
        patch.object(neon_feeds, "_get_conn", return_value=conn),
        patch("psycopg2.extras.execute_values") as execute_values,
    ):
        written = neon_feeds.upsert_enrichment_entries(entries)

    assert written == 2
    passed_cursor, sql, rows = execute_values.call_args.args[:3]
    assert passed_cursor is cursor
    assert "INSERT INTO document_enrichments" in sql
    assert "ON CONFLICT (document_id) DO UPDATE" in sql
    assert "IS DISTINCT FROM" in sql
    assert [row[0] for row in rows] == ["doc-1", "doc-2"]
    assert rows[0][-1].adapted == entries["doc-1"]
    conn.commit.assert_called_once()


def test_upsert_enrichment_entries_is_noop_without_database_url(monkeypatch):
    monkeypatch.setattr(neon_feeds, "_database_url_is_configured", lambda: False)
    with patch.object(neon_feeds, "_get_conn") as get_conn:
        assert neon_feeds.upsert_enrichment_entries({"doc-1": _entry()}) == 0
        assert neon_feeds.upsert_enrichment_entries({}) == 0
    get_conn.assert_not_called()


def test_enrichment_entry_mapping_preserves_payload_and_strips_nul_bytes():
    entry = _entry(status="enriched\x00", enrichment={"summary": "bad\x00text"})
    row = neon_feeds._enrichment_entry_to_row("doc\x00-1", entry)

    assert row is not None
    assert row[0] == "doc-1"
    assert row[1] == "enriched"
    assert row[-1].adapted["enrichment"]["summary"] == "badtext"


def test_get_enrichment_entry_returns_legacy_payload(monkeypatch):
    monkeypatch.setattr(neon_feeds, "_ENRICHMENTS_SCHEMA_ENSURED", True)
    monkeypatch.setattr(neon_feeds, "_database_url_is_configured", lambda: True)
    conn, cursor = _mock_conn()
    cursor.fetchone.return_value = {"entry": _entry()}

    with patch.object(neon_feeds, "_get_conn", return_value=conn):
        assert neon_feeds.get_enrichment_entry("doc-1") == _entry()

    cursor.fetchone.return_value = None
    with patch.object(neon_feeds, "_get_conn", return_value=conn):
        assert neon_feeds.get_enrichment_entry("missing") is None


def test_get_enrichment_entries_is_bounded_and_keyed_by_document_id(monkeypatch):
    monkeypatch.setattr(neon_feeds, "_ENRICHMENTS_SCHEMA_ENSURED", True)
    monkeypatch.setattr(neon_feeds, "_database_url_is_configured", lambda: True)
    conn, cursor = _mock_conn()
    cursor.fetchall.return_value = [
        {"document_id": "doc-1", "entry": _entry("doc-1")},
        {"document_id": "doc-2", "entry": _entry("doc-2")},
    ]

    with patch.object(neon_feeds, "_get_conn", return_value=conn):
        result = neon_feeds.get_enrichment_entries(["doc-1", "doc-1", "doc-2", ""])

    assert list(result) == ["doc-1", "doc-2"]
    sql, params = cursor.execute.call_args.args
    assert "document_id = ANY(%s)" in sql
    assert params == (["doc-1", "doc-2"],)


def test_enrichment_reads_gracefully_fallback_without_database_url(monkeypatch):
    monkeypatch.setattr(neon_feeds, "_database_url_is_configured", lambda: False)
    with patch.object(neon_feeds, "_get_conn") as get_conn:
        assert neon_feeds.get_enrichment_entry("doc-1") is None
        assert neon_feeds.get_enrichment_entries(["doc-1"]) == {}
        assert neon_feeds.get_enrichment_entries([]) == {}
    get_conn.assert_not_called()


def test_count_enrichment_entries_returns_int(monkeypatch):
    monkeypatch.setattr(neon_feeds, "_ENRICHMENTS_SCHEMA_ENSURED", True)
    conn, cursor = _mock_conn()
    cursor.fetchone.return_value = {"count": 123}

    with patch.object(neon_feeds, "_get_conn", return_value=conn):
        assert neon_feeds.count_enrichment_entries() == 123

    cursor.execute.assert_called_once_with(
        "SELECT COUNT(*) AS count FROM document_enrichments"
    )


def test_enrichment_catchup_query_is_source_scoped_bounded_and_stale_aware(monkeypatch):
    monkeypatch.setattr(neon_feeds, "_DOCUMENTS_SCHEMA_ENSURED", True)
    monkeypatch.setattr(neon_feeds, "_ENRICHMENTS_SCHEMA_ENSURED", True)
    conn, cursor = _mock_conn()
    cursor.fetchall.return_value = [{"document_id": "doc-1"}, {"document_id": "doc-2"}]

    with patch.object(neon_feeds, "_get_conn", return_value=conn):
        result = neon_feeds.get_document_ids_needing_enrichment(
            ["substack_public_article", "substack_public_article"],
            limit=10,
        )

    sql, params = cursor.execute.call_args.args
    assert result == ["doc-1", "doc-2"]
    assert "documents.source_kind = ANY(%s)" in sql
    assert "documents.updated_at > enrichment.updated_at" in sql
    assert "attempt_count" in sql
    assert "LIMIT %s" in sql
    assert params == (["substack_public_article"], 10)


def test_get_documents_returns_legacy_records_in_one_query(monkeypatch):
    monkeypatch.setattr(neon_feeds, "_DOCUMENTS_SCHEMA_ENSURED", True)
    monkeypatch.setattr(neon_feeds, "_database_url_is_configured", lambda: True)
    conn, cursor = _mock_conn()
    cursor.fetchall.return_value = [_document_row("doc-1"), _document_row("doc-2")]

    with patch.object(neon_feeds, "_get_conn", return_value=conn):
        result = neon_feeds.get_documents(["doc-1", "doc-1", "doc-2"])

    assert result["doc-1"]["metadata"]["title"] == "Metadata title"
    assert result["doc-1"]["metadata"]["organization"] == "SEC"
    assert result["doc-1"]["metadata"]["date"] == "2026-07-14"
    assert result["doc-1"]["content"] == {"full_text": "full body"}
    assert cursor.execute.call_args.args[1] == (["doc-1", "doc-2"],)


def test_get_documents_by_urls_keeps_newest_exact_url_row(monkeypatch):
    monkeypatch.setattr(neon_feeds, "_DOCUMENTS_SCHEMA_ENSURED", True)
    monkeypatch.setattr(neon_feeds, "_database_url_is_configured", lambda: True)
    conn, cursor = _mock_conn()
    url = "https://example.com/shared"
    cursor.fetchall.return_value = [
        _document_row("new", url=url, metadata={"document_id": "new"}),
        _document_row("old", url=url, metadata={"document_id": "old"}),
    ]

    with patch.object(neon_feeds, "_get_conn", return_value=conn):
        result = neon_feeds.get_documents_by_urls([url, url])

    assert result[url]["metadata"]["document_id"] == "new"
    sql, params = cursor.execute.call_args.args
    assert "full_text" not in sql.split("FROM documents", 1)[0]
    assert "url = ANY(%s)" in sql
    assert "ORDER BY updated_at DESC" in sql
    assert params == ([url],)
    assert result[url]["content"] == {}


def test_source_kind_lookup_omits_full_text_by_default(monkeypatch):
    monkeypatch.setattr(neon_feeds, "_DOCUMENTS_SCHEMA_ENSURED", True)
    monkeypatch.setattr(neon_feeds, "_database_url_is_configured", lambda: True)
    conn, cursor = _mock_conn()
    cursor.fetchall.return_value = [_document_row()]

    with patch.object(neon_feeds, "_get_conn", return_value=conn):
        records = neon_feeds.get_document_records_by_source_kinds(
            ["financial_news", "financial_news"]
        )

    sql, params = cursor.execute.call_args.args
    selected_columns = sql.split("FROM documents", 1)[0]
    assert "full_text" not in selected_columns
    assert "source_kind = ANY(%s)" in sql
    assert params == (["financial_news"],)
    assert records[0]["content"] == {}
    assert records[0]["metadata"]["source_kind"] == "financial_news"


def test_source_kind_lookup_includes_full_text_only_when_requested(monkeypatch):
    monkeypatch.setattr(neon_feeds, "_DOCUMENTS_SCHEMA_ENSURED", True)
    monkeypatch.setattr(neon_feeds, "_database_url_is_configured", lambda: True)
    conn, cursor = _mock_conn()
    cursor.fetchall.return_value = [_document_row()]

    with patch.object(neon_feeds, "_get_conn", return_value=conn):
        records = neon_feeds.get_document_records_by_source_kinds(
            ["financial_news"], include_full_text=True
        )

    sql = cursor.execute.call_args.args[0]
    assert "full_text" in sql.split("FROM documents", 1)[0]
    assert records[0]["content"] == {"full_text": "full body"}


def test_bounded_document_reads_gracefully_fallback_without_database_url(monkeypatch):
    monkeypatch.setattr(neon_feeds, "_database_url_is_configured", lambda: False)
    with patch.object(neon_feeds, "_get_conn") as get_conn:
        assert neon_feeds.get_documents(["doc-1"]) == {}
        assert neon_feeds.get_documents_by_urls(["https://example.com/doc-1"]) == {}
        assert neon_feeds.get_document_records_by_source_kinds(["financial_news"]) == []
    get_conn.assert_not_called()

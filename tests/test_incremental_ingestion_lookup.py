from types import SimpleNamespace

import neon_feeds
import run_connector_extraction_pipeline as connector_pipeline
import run_financial_news_pipeline as financial_pipeline


def _record(url: str, *, published: str = "July 14, 2026", source_kind: str = "sec_speech") -> dict:
    return {
        "metadata": {
            "document_id": "doc-existing",
            "url": url,
            "title": "Existing item",
            "date": published,
            "published_date": published,
            "published_at": published,
            "source_kind": source_kind,
        },
        "content": {},
    }


def _connector_args(connector: str = "sec_speech") -> SimpleNamespace:
    return SimpleNamespace(
        connector=connector,
        require_remote_persistence=False,
        base_url="https://example.com/listing",
        max_pages=1,
        include_pdfs=False,
        include_rss=False,
        keywords="",
        exclude_terms="",
        selection="new_or_updated",
        limit=25,
        relevance_provider="deepseek",
        relevance_model="deepseek-v4-flash",
        persistence_mode="gcs_authoritative",
        dry_run=False,
        summary_path="",
    )


def test_complete_neon_url_lookup_returns_bounded_records(monkeypatch):
    url = "https://example.com/item"
    calls = []
    monkeypatch.setenv("DATABASE_URL", "postgresql://configured")
    monkeypatch.setattr(
        neon_feeds,
        "get_documents_by_urls",
        lambda urls: calls.append(urls) or {url: _record(url)},
    )

    payload = financial_pipeline._load_complete_neon_documents_for_urls([url])

    assert calls == [[url]]
    assert payload == {"updated_at": "", "documents": [_record(url)]}


def test_incomplete_neon_url_lookup_requires_authoritative_fallback(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://configured")
    monkeypatch.setattr(neon_feeds, "get_documents_by_urls", lambda urls: {})

    assert (
        financial_pipeline._load_complete_neon_documents_for_urls(
            ["https://example.com/not-yet-mirrored"]
        )
        is None
    )


def test_neon_source_kind_metadata_fallback_handles_tracking_url_variants(monkeypatch):
    canonical_url = "https://example.com/item"
    monkeypatch.setenv("DATABASE_URL", "postgresql://configured")
    monkeypatch.setattr(neon_feeds, "get_documents_by_urls", lambda urls: {})
    monkeypatch.setattr(
        neon_feeds,
        "get_document_records_by_source_kinds",
        lambda kinds, include_full_text=False: [_record(canonical_url, source_kind="newsapi_article")],
    )

    payload = financial_pipeline._load_complete_neon_documents_for_urls(
        [f"{canonical_url}?utm_source=rss"],
        source_kinds=["newsapi_article"],
    )

    assert payload is not None
    assert payload["documents"][0]["metadata"]["url"] == canonical_url


def test_connector_no_change_uses_neon_without_loading_gcs_snapshots(monkeypatch):
    url = "https://example.com/existing"
    monkeypatch.setattr(connector_pipeline.core, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setattr(connector_pipeline.core, "_get_gcs_storage", lambda secrets: (object(), ""))
    monkeypatch.setattr(
        connector_pipeline.core,
        "_load_complete_neon_documents_for_urls",
        lambda urls, **kwargs: {"documents": [_record(url)]},
    )
    monkeypatch.setattr(
        connector_pipeline.core,
        "_load_custom_documents",
        lambda storage: (_ for _ in ()).throw(AssertionError("custom_documents.json was loaded")),
    )
    monkeypatch.setattr(
        connector_pipeline,
        "_load_existing_speech_url_keys",
        lambda storage: (_ for _ in ()).throw(AssertionError("all_speeches.json was loaded")),
    )
    monkeypatch.setattr(
        connector_pipeline,
        "_discover_connector",
        lambda **kwargs: (
            object(),
            [{"url": url, "title": "Existing item", "date": "July 14, 2026"}],
            {},
        ),
    )
    monkeypatch.setattr(connector_pipeline.core, "_write_summary", lambda *args, **kwargs: None)

    summary = connector_pipeline._run_connector_extraction(_connector_args())

    assert summary["candidate_count"] == 0
    assert summary["processed_count"] == 0
    assert summary["existing_lookup"] == "neon"
    assert summary["gcs_snapshot_loaded"] is False
    assert summary["status_counts"]["existing"] == 1


def test_connector_rechecks_authoritative_gcs_before_a_detected_update(monkeypatch):
    url = "https://example.com/existing"
    gcs_payload = {"documents": [_record(url, published="July 14, 2026")]}
    gcs_loads = []
    monkeypatch.setattr(connector_pipeline.core, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setattr(connector_pipeline.core, "_get_gcs_storage", lambda secrets: (object(), ""))
    monkeypatch.setattr(
        connector_pipeline.core,
        "_load_complete_neon_documents_for_urls",
        lambda urls, **kwargs: {"documents": [_record(url, published="July 13, 2026")]},
    )
    monkeypatch.setattr(
        connector_pipeline.core,
        "_load_custom_documents",
        lambda storage: gcs_loads.append(True) or gcs_payload,
    )
    monkeypatch.setattr(connector_pipeline, "_load_existing_speech_url_keys", lambda storage: set())
    monkeypatch.setattr(
        connector_pipeline,
        "_discover_connector",
        lambda **kwargs: (
            object(),
            [{"url": url, "title": "Existing item", "date": "July 14, 2026"}],
            {},
        ),
    )
    monkeypatch.setattr(
        connector_pipeline,
        "_extract_record",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("stale mirror caused extraction")),
    )
    monkeypatch.setattr(connector_pipeline.core, "_write_summary", lambda *args, **kwargs: None)

    summary = connector_pipeline._run_connector_extraction(_connector_args())

    assert gcs_loads == [True]
    assert summary["candidate_count"] == 0
    assert summary["existing_lookup"] == "neon_then_gcs"
    assert summary["gcs_snapshot_loaded"] is True


def test_substack_skips_deepseek_relevance_for_unchanged_posts(monkeypatch):
    url = "https://example.substack.com/p/existing"
    relevance_batches = []

    class Scraper:
        def filter_institutional_finance(self, entries, **kwargs):
            relevance_batches.append(list(entries))
            return list(entries), []

    monkeypatch.setattr(connector_pipeline.core, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setattr(connector_pipeline.core, "_get_gcs_storage", lambda secrets: (object(), ""))
    monkeypatch.setattr(connector_pipeline, "_load_current_topic_rules", lambda: [])
    monkeypatch.setattr(connector_pipeline.core, "_get_model_client", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        connector_pipeline.core,
        "_load_complete_neon_documents_for_urls",
        lambda urls, **kwargs: {"documents": [_record(url, source_kind="substack_public_article")]},
    )
    monkeypatch.setattr(
        connector_pipeline.core,
        "_load_custom_documents",
        lambda storage: (_ for _ in ()).throw(AssertionError("custom_documents.json was loaded")),
    )
    monkeypatch.setattr(
        connector_pipeline,
        "_load_existing_speech_url_keys",
        lambda storage: (_ for _ in ()).throw(AssertionError("all_speeches.json was loaded")),
    )
    monkeypatch.setattr(
        connector_pipeline,
        "_discover_connector",
        lambda **kwargs: (
            Scraper(),
            [{"url": url, "title": "Existing item", "date": "July 14, 2026"}],
            {},
        ),
    )
    monkeypatch.setattr(connector_pipeline.core, "_write_summary", lambda *args, **kwargs: None)

    summary = connector_pipeline._run_connector_extraction(
        _connector_args("substack_public_article")
    )

    assert relevance_batches == [[]]
    assert summary["candidate_count"] == 0
    assert summary["discovery_debug"]["relevance_checked_count"] == 0
    assert summary["gcs_snapshot_loaded"] is False


def test_newsapi_no_change_uses_neon_without_loading_custom_documents(monkeypatch):
    url = "https://example.com/news/existing"

    class Scraper:
        def __init__(self, api_key):
            self.last_discovery_debug = {}

        def discover_documents(self, **kwargs):
            return [
                {
                    "url": url,
                    "title": "Existing item",
                    "date": "2026-07-14T12:00:00Z",
                    "published_at": "2026-07-14T12:00:00Z",
                }
            ]

    existing = _record(
        url,
        published="2026-07-14T12:00:00Z",
        source_kind="newsapi_article",
    )
    monkeypatch.setattr(financial_pipeline, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setattr(financial_pipeline, "_get_gcs_storage", lambda secrets: (object(), ""))
    monkeypatch.setattr(financial_pipeline, "_get_newsapi_api_key", lambda secrets: "test-key")
    monkeypatch.setattr(financial_pipeline, "_get_newsapi_api_key_source", lambda secrets: "test")
    monkeypatch.setattr(financial_pipeline, "_load_news_connector_settings", lambda storage: financial_pipeline._empty_news_connector_settings())
    monkeypatch.setattr(financial_pipeline, "NewsAPIFinancialScraper", Scraper)
    monkeypatch.setattr(
        financial_pipeline,
        "_load_complete_neon_documents_for_urls",
        lambda urls, **kwargs: {"documents": [existing]},
    )
    monkeypatch.setattr(
        financial_pipeline,
        "_load_custom_documents",
        lambda storage: (_ for _ in ()).throw(AssertionError("custom_documents.json was loaded")),
    )
    monkeypatch.setattr(financial_pipeline, "_write_summary", lambda *args, **kwargs: None)

    args = SimpleNamespace(
        require_remote_persistence=False,
        query=None,
        lookback_days=None,
        max_pages=None,
        page_size=None,
        target_count=None,
        sort_by=None,
        organization_label=None,
        domains=None,
        exclude_domains=None,
        tags_csv=None,
        selection="new_or_updated",
        limit=None,
        dry_run=False,
        summary_path="",
    )

    summary = financial_pipeline._run_news_ingest(args)

    assert summary["candidate_count"] == 0
    assert summary["processed_count"] == 0
    assert summary["existing_lookup"] == "neon"
    assert summary["gcs_snapshot_loaded"] is False


def test_neon_authoritative_requires_explicit_verified_backfill(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://configured")
    monkeypatch.delenv(financial_pipeline.NEON_BACKFILL_VERIFIED_ENV, raising=False)

    try:
        financial_pipeline._require_neon_authoritative_ready()
    except RuntimeError as exc:
        assert financial_pipeline.NEON_BACKFILL_VERIFIED_ENV in str(exc)
    else:  # pragma: no cover - guard assertion
        raise AssertionError("unverified backfill was accepted")

    monkeypatch.setenv(financial_pipeline.NEON_BACKFILL_VERIFIED_ENV, "true")
    monkeypatch.setattr(neon_feeds, "count_documents", lambda: 0)
    monkeypatch.setattr(
        neon_feeds,
        "get_migration_checkpoint",
        lambda key: {"status": "verified"},
    )
    try:
        financial_pipeline._require_neon_authoritative_ready()
    except RuntimeError as exc:
        assert "mirror is empty" in str(exc)
    else:  # pragma: no cover - guard assertion
        raise AssertionError("empty mirror was accepted")

    monkeypatch.setattr(neon_feeds, "count_documents", lambda: 10)
    monkeypatch.setattr(neon_feeds, "get_migration_checkpoint", lambda key: None)
    try:
        financial_pipeline._require_neon_authoritative_ready()
    except RuntimeError as exc:
        assert "checkpoint is not verified" in str(exc)
    else:  # pragma: no cover - guard assertion
        raise AssertionError("missing database checkpoint was accepted")


def test_required_neon_document_batch_is_single_transaction_and_deduplicated(monkeypatch):
    first = {
        "metadata": {"document_id": "doc-1", "title": "first"},
        "content": {"full_text": "old"},
    }
    latest = {
        "metadata": {"document_id": "doc-1", "title": "latest"},
        "content": {"full_text": "new"},
    }
    second = {
        "metadata": {"document_id": "doc-2", "title": "second"},
        "content": {"full_text": "body"},
    }
    submitted = []

    def persist(records):
        submitted.append(records)
        return len(records)

    monkeypatch.setattr(neon_feeds, "mirror_documents_batch", persist)

    financial_pipeline._persist_documents_to_neon([first, latest, second])

    assert submitted == [[latest, second]]


def test_required_neon_document_batch_rejects_partial_submission(monkeypatch):
    record = {
        "metadata": {"document_id": "doc-1"},
        "content": {"full_text": "body"},
    }
    monkeypatch.setattr(neon_feeds, "mirror_documents_batch", lambda records: 0)

    try:
        financial_pipeline._persist_documents_to_neon([record])
    except financial_pipeline.NeonPersistenceError as exc:
        assert "submitted 0 of 1" in str(exc)
    else:  # pragma: no cover - guard assertion
        raise AssertionError("partial Neon document batch was accepted")


def test_bloomberg_neon_authoritative_persists_changed_row_without_gcs(monkeypatch):
    url = "https://www.bloomberg.com/news/articles/2026-07-14/example"
    persisted = []
    args = _connector_args("bloomberg_public_latest")
    args.persistence_mode = "neon_authoritative"
    monkeypatch.setattr(connector_pipeline.core, "_require_neon_authoritative_ready", lambda: None)
    monkeypatch.setattr(connector_pipeline.core, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setattr(
        connector_pipeline.core,
        "_get_gcs_storage",
        lambda secrets: (_ for _ in ()).throw(AssertionError("GCS was initialized")),
    )
    monkeypatch.setattr(
        connector_pipeline.core,
        "_load_complete_neon_documents_for_urls",
        lambda urls, **kwargs: {"documents": []},
    )
    monkeypatch.setattr(
        connector_pipeline.core,
        "_load_custom_documents",
        lambda storage: (_ for _ in ()).throw(AssertionError("custom_documents.json was loaded")),
    )
    monkeypatch.setattr(
        connector_pipeline,
        "_load_existing_speech_url_keys",
        lambda storage: (_ for _ in ()).throw(AssertionError("all_speeches.json was loaded")),
    )
    monkeypatch.setattr(
        connector_pipeline,
        "_discover_connector",
        lambda **kwargs: (
            object(),
            [{"url": url, "title": "New Bloomberg item", "date": "July 14, 2026"}],
            {},
        ),
    )
    record = {
        "metadata": {
            "document_id": "bloomberg-doc-1",
            "url": url,
            "source_kind": "bloomberg_public_article",
        },
        "content": {"full_text": "Complete Bloomberg article body."},
    }
    monkeypatch.setattr(connector_pipeline, "_extract_record", lambda *args, **kwargs: record)
    monkeypatch.setattr(
        connector_pipeline.core,
        "_persist_documents_to_neon",
        lambda items: persisted.extend(items),
    )
    monkeypatch.setattr(connector_pipeline.core, "_write_summary", lambda *args, **kwargs: None)

    summary = connector_pipeline._run_connector_extraction(args)

    assert persisted == [record]
    assert summary["saved_new"] == 1
    assert summary["persistence_mode"] == "neon_authoritative"
    assert summary["gcs_snapshot_loaded"] is False
    assert summary["rule_summaries_rebuilt"] is False


def test_neon_document_write_failure_is_fatal_and_never_falls_back_to_gcs(monkeypatch):
    url = "https://www.bloomberg.com/news/articles/2026-07-14/example"
    args = _connector_args("bloomberg_public_latest")
    args.persistence_mode = "neon_authoritative"
    monkeypatch.setattr(connector_pipeline.core, "_require_neon_authoritative_ready", lambda: None)
    monkeypatch.setattr(connector_pipeline.core, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setattr(
        connector_pipeline.core,
        "_load_complete_neon_documents_for_urls",
        lambda urls, **kwargs: {"documents": []},
    )
    monkeypatch.setattr(
        connector_pipeline,
        "_discover_connector",
        lambda **kwargs: (object(), [{"url": url, "title": "New item"}], {}),
    )
    monkeypatch.setattr(
        connector_pipeline,
        "_extract_record",
        lambda *args, **kwargs: {
            "metadata": {"document_id": "doc-1", "url": url},
            "content": {"full_text": "Full body"},
        },
    )
    monkeypatch.setattr(
        connector_pipeline.core,
        "_persist_documents_to_neon",
        lambda record: (_ for _ in ()).throw(
            financial_pipeline.NeonPersistenceError("database unavailable")
        ),
    )
    monkeypatch.setattr(
        connector_pipeline.core,
        "_load_custom_documents",
        lambda storage: (_ for _ in ()).throw(AssertionError("GCS fallback was attempted")),
    )

    try:
        connector_pipeline._run_connector_extraction(args)
    except financial_pipeline.NeonPersistenceError as exc:
        assert "database unavailable" in str(exc)
    else:  # pragma: no cover - guard assertion
        raise AssertionError("required Neon failure was swallowed")


def test_unsupported_connector_rejects_neon_authoritative_mode():
    args = _connector_args("sec_speech")
    args.persistence_mode = "neon_authoritative"

    try:
        connector_pipeline._run_connector_extraction(args)
    except RuntimeError as exc:
        assert "not enabled" in str(exc)
    else:  # pragma: no cover - guard assertion
        raise AssertionError("unsupported connector was accepted")


def test_newsapi_neon_authoritative_writes_full_document_without_snapshot_reads(monkeypatch):
    url = "https://example.com/news/new-item"
    persisted = []

    class Scraper:
        def __init__(self, api_key):
            self.last_discovery_debug = {}

        def discover_documents(self, **kwargs):
            return [
                {
                    "url": url,
                    "title": "New item",
                    "date": "2026-07-14T12:00:00Z",
                    "published_at": "2026-07-14T12:00:00Z",
                }
            ]

        def extract_document(self, *args, **kwargs):
            return {
                "success": True,
                "data": {
                    "url": url,
                    "title": "New item",
                    "date": "2026-07-14T12:00:00Z",
                    "full_text": " ".join(["substantive"] * 50),
                    "source_name": "Example News",
                    "source_format": "html",
                },
            }

    monkeypatch.setattr(financial_pipeline, "_require_neon_authoritative_ready", lambda: None)
    monkeypatch.setattr(financial_pipeline, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setattr(financial_pipeline, "_get_gcs_storage", lambda secrets: (None, "not configured"))
    monkeypatch.setattr(financial_pipeline, "_get_newsapi_api_key", lambda secrets: "test-key")
    monkeypatch.setattr(financial_pipeline, "_get_newsapi_api_key_source", lambda secrets: "test")
    monkeypatch.setattr(
        financial_pipeline,
        "_load_news_connector_settings",
        lambda storage: financial_pipeline._empty_news_connector_settings(),
    )
    monkeypatch.setattr(financial_pipeline, "NewsAPIFinancialScraper", Scraper)
    monkeypatch.setattr(
        financial_pipeline,
        "_load_complete_neon_documents_for_urls",
        lambda urls, **kwargs: {"documents": []},
    )
    monkeypatch.setattr(
        financial_pipeline,
        "_load_custom_documents",
        lambda storage: (_ for _ in ()).throw(AssertionError("custom_documents.json was loaded")),
    )
    monkeypatch.setattr(
        financial_pipeline,
        "_persist_documents_to_neon",
        lambda records: persisted.extend(records),
    )
    monkeypatch.setattr(financial_pipeline, "_write_summary", lambda *args, **kwargs: None)

    summary = financial_pipeline._run_news_ingest(
        SimpleNamespace(
            persistence_mode="neon_authoritative",
            require_remote_persistence=True,
            query=None,
            lookback_days=None,
            max_pages=None,
            page_size=None,
            target_count=None,
            sort_by=None,
            organization_label=None,
            domains=None,
            exclude_domains=None,
            tags_csv=None,
            selection="new_or_updated",
            limit=None,
            dry_run=False,
            summary_path="",
        )
    )

    assert len(persisted) == 1
    assert persisted[0]["content"]["full_text"].startswith("substantive substantive")
    assert summary["saved_new"] == 1
    assert summary["persistence_mode"] == "neon_authoritative"
    assert summary["gcs_snapshot_loaded"] is False

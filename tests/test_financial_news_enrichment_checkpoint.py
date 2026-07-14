from types import SimpleNamespace

import pytest

import run_financial_news_pipeline as pipeline


def _doc(doc_id: str) -> dict:
    return {
        "metadata": {
            "document_id": doc_id,
            "source_kind": "substack_public_article",
            "organization": "Example Substack",
            "title": f"Document {doc_id}",
            "date": "2026-07-03T12:00:00Z",
            "url": f"https://example.com/{doc_id}",
            "doc_type": "Article",
        },
        "content": {"full_text": "This is a substantive policy and market text for enrichment."},
    }


def test_uploaded_document_id_is_stable_for_source_url():
    first = pipeline._create_uploaded_document_record(
        text=" ".join(["first extraction text"] * 40),
        organization="Decrypt",
        title="Fake Mac Clipboard App Delivers New Password-Stealing Malware",
        speaker="Decrypt",
        doc_date="July 4, 2026",
        doc_type="Article",
        source_url="https://decrypt.co/330205/fake-mac-clipboard-app-password-stealing-malware",
        source_filename="fake-mac-clipboard.html",
        source_ext=".html",
        source_local_path="",
        source_gcs_path="",
        tags_csv="decrypt,crypto",
        source_kind="decrypt_article",
    )
    second = pipeline._create_uploaded_document_record(
        text=" ".join(["cleaned extraction with different length"] * 80),
        organization="Decrypt",
        title="Fake Mac Clipboard App Delivers New Password-Stealing Malware - Updated",
        speaker="Different Author",
        doc_date="July 5, 2026",
        doc_type="Article",
        source_url="https://decrypt.co/330205/fake-mac-clipboard-app-password-stealing-malware?utm_source=rss",
        source_filename="different.html",
        source_ext=".html",
        source_local_path="",
        source_gcs_path="",
        tags_csv="decrypt,crypto",
        source_kind="decrypt_article",
    )

    assert first["metadata"]["document_id"] == second["metadata"]["document_id"]


def test_enrichment_entries_migrate_to_stable_document_id():
    state = {"entries": {"old-doc": {"doc_id": "old-doc", "model": "deepseek-v4-flash"}}}

    migrated = pipeline._migrate_enrichment_entry_ids(state, {"old-doc": "new-doc"})

    assert migrated == 1
    assert "old-doc" not in state["entries"]
    assert state["entries"]["new-doc"]["doc_id"] == "new-doc"
    assert state["entries"]["new-doc"]["model"] == "deepseek-v4-flash"


def test_enrichment_candidates_skip_metadata_fallback_stub_by_extraction_mode():
    doc = _doc("doc-stub-1")
    doc["metadata"]["extraction_mode"] = "metadata_fallback"

    candidates = pipeline._build_news_enrichment_candidates(
        {"documents": [doc]}, source_kind="substack_public_article"
    )

    assert candidates == []


def test_enrichment_candidates_skip_metadata_fallback_stub_by_text_marker():
    doc = _doc("doc-stub-2")
    doc["content"]["full_text"] = (
        "Some Title\nSource: Example\nOrganization: Example\nDate: \nURL: https://example.com\n"
        "Note: The source page was discovered successfully, but the article body extraction "
        f"returned a short result. {pipeline.METADATA_FALLBACK_TEXT_MARKER}"
    )

    candidates = pipeline._build_news_enrichment_candidates(
        {"documents": [doc]}, source_kind="substack_public_article"
    )

    assert candidates == []


def test_enrichment_candidates_include_real_body_text():
    candidates = pipeline._build_news_enrichment_candidates(
        {"documents": [_doc("doc-real-1")]}, source_kind="substack_public_article"
    )

    assert [item["doc_id"] for item in candidates] == ["doc-real-1"]


def test_news_enrichment_checkpoints_progress(monkeypatch):
    payload = {"documents": [_doc("doc-1"), _doc("doc-2"), _doc("doc-3")]}
    saved_doc_ids = []
    mirrored_doc_ids = []

    monkeypatch.setattr(pipeline, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setattr(pipeline, "_get_gcs_storage", lambda secrets: (None, "local"))
    monkeypatch.setattr(pipeline, "_load_custom_documents", lambda storage: payload)
    monkeypatch.setattr(pipeline, "_load_enrichment_state", lambda storage: {"entries": {}})
    monkeypatch.setattr(
        pipeline,
        "_save_enrichment_state",
        lambda storage, state, require_remote=False: saved_doc_ids.append(tuple(sorted(state["entries"].keys()))),
    )
    monkeypatch.setattr(
        pipeline,
        "_mirror_enrichment_entries_to_neon_best_effort",
        lambda entries: mirrored_doc_ids.append(tuple(sorted(entries.keys()))),
    )
    monkeypatch.setattr(pipeline, "_rebuild_rule_summaries", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_write_summary", lambda *args, **kwargs: None)

    args = SimpleNamespace(
        require_remote_persistence=False,
        source_kind="substack_public_article",
        doc_ids_from_summary="",
        doc_id=[],
        mode="all",
        order="stored",
        limit=None,
        provider="deepseek",
        model="deepseek-v4-flash",
        heuristic_only=True,
        dry_run=False,
        checkpoint_every=2,
        summary_path="",
    )

    summary = pipeline._run_news_enrichment(args)

    assert summary["processed_count"] == 3
    assert summary["fallback_enriched_count"] == 3
    assert saved_doc_ids == [("doc-1", "doc-2"), ("doc-1", "doc-2", "doc-3")]
    assert mirrored_doc_ids == [("doc-1", "doc-2"), ("doc-3",)]


def test_failed_authoritative_enrichment_save_never_mirrors_to_neon(monkeypatch):
    payload = {"documents": [_doc("doc-1")]}
    mirrored_doc_ids = []

    monkeypatch.setattr(pipeline, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setattr(pipeline, "_get_gcs_storage", lambda secrets: (object(), ""))
    monkeypatch.setattr(pipeline, "_load_custom_documents", lambda storage: payload)
    monkeypatch.setattr(pipeline, "_load_enrichment_state", lambda storage: {"entries": {}})
    monkeypatch.setattr(
        pipeline,
        "_save_enrichment_state",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("GCS generation conflict")),
    )
    monkeypatch.setattr(
        pipeline,
        "_mirror_enrichment_entries_to_neon_best_effort",
        lambda entries: mirrored_doc_ids.append(tuple(sorted(entries.keys()))),
    )
    monkeypatch.setattr(pipeline, "_write_summary", lambda *args, **kwargs: None)

    args = SimpleNamespace(
        require_remote_persistence=True,
        source_kind="substack_public_article",
        doc_ids_from_summary="",
        doc_id=[],
        mode="all",
        order="stored",
        limit=None,
        provider="deepseek",
        model="deepseek-v4-flash",
        heuristic_only=True,
        dry_run=False,
        checkpoint_every=0,
        summary_path="",
    )

    with pytest.raises(RuntimeError, match="GCS generation conflict"):
        pipeline._run_news_enrichment(args)

    assert mirrored_doc_ids == []


def test_neon_authoritative_targeted_enrichment_uses_only_bounded_rows(monkeypatch):
    persisted_batches = []
    monkeypatch.setattr(pipeline, "_require_neon_authoritative_ready", lambda: None)
    monkeypatch.setattr(pipeline, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setattr(
        pipeline,
        "_get_gcs_storage",
        lambda secrets: (_ for _ in ()).throw(AssertionError("GCS was initialized")),
    )
    monkeypatch.setattr(pipeline, "_load_neon_documents_by_ids", lambda ids: {"doc-1": _doc("doc-1")})
    monkeypatch.setattr(pipeline, "_load_neon_enrichment_entries", lambda ids: {})
    monkeypatch.setattr(
        pipeline,
        "_load_custom_documents",
        lambda storage: (_ for _ in ()).throw(AssertionError("custom snapshot was loaded")),
    )
    monkeypatch.setattr(
        pipeline,
        "_load_enrichment_state",
        lambda storage: (_ for _ in ()).throw(AssertionError("enrichment snapshot was loaded")),
    )
    monkeypatch.setattr(
        pipeline,
        "_persist_enrichment_entries_to_neon",
        lambda entries: persisted_batches.append(dict(entries)),
    )
    monkeypatch.setattr(
        pipeline,
        "_rebuild_rule_summaries",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy summaries rebuilt")),
    )
    monkeypatch.setattr(pipeline, "_write_summary", lambda *args, **kwargs: None)

    summary = pipeline._run_news_enrichment(
        SimpleNamespace(
            persistence_mode="neon_authoritative",
            require_remote_persistence=True,
            source_kind="substack_public_article",
            doc_ids_from_summary="",
            doc_id=["doc-1"],
            mode="all",
            order="stored",
            limit=None,
            provider="deepseek",
            model="deepseek-v4-flash",
            heuristic_only=True,
            dry_run=False,
            checkpoint_every=0,
            summary_path="",
        )
    )

    assert [sorted(batch) for batch in persisted_batches] == [["doc-1"]]
    assert persisted_batches[0]["doc-1"]["status"] == "fallback_enriched"
    assert summary["processed_count"] == 1
    assert summary["persistence_mode"] == "neon_authoritative"
    assert summary["gcs_snapshot_loaded"] is False
    assert summary["rule_summaries_rebuilt"] is False


def test_neon_authoritative_catchup_enriches_bounded_missing_rows(monkeypatch):
    persisted_batches = []
    monkeypatch.setattr(pipeline, "_require_neon_authoritative_ready", lambda: None)
    monkeypatch.setattr(pipeline, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setattr(
        pipeline,
        "_load_neon_enrichment_catchup_ids",
        lambda source_kind, limit: ["doc-1"],
    )
    monkeypatch.setattr(pipeline, "_load_neon_documents_by_ids", lambda ids: {"doc-1": _doc("doc-1")})
    monkeypatch.setattr(pipeline, "_load_neon_enrichment_entries", lambda ids: {})
    monkeypatch.setattr(
        pipeline,
        "_persist_enrichment_entries_to_neon",
        lambda entries: persisted_batches.append(dict(entries)),
    )
    monkeypatch.setattr(pipeline, "_write_summary", lambda *args, **kwargs: None)

    summary = pipeline._run_news_enrichment(
        SimpleNamespace(
            persistence_mode="neon_authoritative",
            require_remote_persistence=True,
            source_kind="substack_public_article",
            doc_ids_from_summary="",
            doc_id=[],
            mode="only_missing_or_failed",
            order="stored",
            limit=10,
            provider="deepseek",
            model="deepseek-v4-flash",
            heuristic_only=True,
            dry_run=False,
            checkpoint_every=0,
            summary_path="",
        )
    )

    assert summary["catchup_mode"] is True
    assert summary["requested_doc_ids"] == ["doc-1"]
    assert summary["processed_count"] == 1
    assert list(persisted_batches[0]) == ["doc-1"]


def test_neon_authoritative_empty_catchup_exits_before_model_or_gcs(monkeypatch):
    monkeypatch.setattr(pipeline, "_require_neon_authoritative_ready", lambda: None)
    monkeypatch.setattr(pipeline, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setattr(pipeline, "_load_neon_enrichment_catchup_ids", lambda source_kind, limit: [])
    monkeypatch.setattr(
        pipeline,
        "_get_model_client",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("model initialized")),
    )
    monkeypatch.setattr(
        pipeline,
        "_get_gcs_storage",
        lambda secrets: (_ for _ in ()).throw(AssertionError("GCS initialized")),
    )
    monkeypatch.setattr(pipeline, "_write_summary", lambda *args, **kwargs: None)

    summary = pipeline._run_news_enrichment(
        SimpleNamespace(
            persistence_mode="neon_authoritative",
            require_remote_persistence=True,
            source_kind="substack_public_article",
            doc_ids_from_summary="",
            doc_id=[],
            mode="only_missing_or_failed",
            order="stored",
            limit=10,
            provider="deepseek",
            model="deepseek-v4-flash",
            heuristic_only=False,
            dry_run=False,
            checkpoint_every=0,
            summary_path="",
        )
    )

    assert summary["catchup_mode"] is True
    assert summary["processed_count"] == 0
    assert summary["gcs_snapshot_loaded"] is False


def test_neon_authoritative_enrichment_write_failure_is_fatal(monkeypatch):
    monkeypatch.setattr(pipeline, "_require_neon_authoritative_ready", lambda: None)
    monkeypatch.setattr(pipeline, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setattr(pipeline, "_load_neon_documents_by_ids", lambda ids: {"doc-1": _doc("doc-1")})
    monkeypatch.setattr(pipeline, "_load_neon_enrichment_entries", lambda ids: {})
    monkeypatch.setattr(
        pipeline,
        "_persist_enrichment_entries_to_neon",
        lambda entries: (_ for _ in ()).throw(
            pipeline.NeonPersistenceError("required row write failed")
        ),
    )
    monkeypatch.setattr(pipeline, "_write_summary", lambda *args, **kwargs: None)

    with pytest.raises(pipeline.NeonPersistenceError, match="required row write failed"):
        pipeline._run_news_enrichment(
            SimpleNamespace(
                persistence_mode="neon_authoritative",
                require_remote_persistence=True,
                source_kind="substack_public_article",
                doc_ids_from_summary="",
                doc_id=["doc-1"],
                mode="all",
                order="stored",
                limit=None,
                provider="deepseek",
                model="deepseek-v4-flash",
                heuristic_only=True,
                dry_run=False,
                checkpoint_every=0,
                summary_path="",
            )
        )


@pytest.mark.parametrize("target_kind", ["summary", "doc_id"])
def test_explicit_empty_enrichment_target_never_expands_to_all_documents(monkeypatch, tmp_path, target_kind):
    saved_states = []

    monkeypatch.setattr(pipeline, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setattr(pipeline, "_get_gcs_storage", lambda secrets: (None, "local"))
    monkeypatch.setattr(
        pipeline,
        "_load_custom_documents",
        lambda storage: (_ for _ in ()).throw(AssertionError("custom_documents.json was loaded")),
    )
    monkeypatch.setattr(
        pipeline,
        "_load_enrichment_state",
        lambda storage: (_ for _ in ()).throw(AssertionError("enrichment snapshot was loaded")),
    )
    monkeypatch.setattr(
        pipeline,
        "_save_enrichment_state",
        lambda storage, state, require_remote=False: saved_states.append(state),
    )
    monkeypatch.setattr(pipeline, "_write_summary", lambda *args, **kwargs: None)

    summary_path = ""
    doc_ids = []
    if target_kind == "summary":
        path = tmp_path / "empty-extraction-summary.json"
        path.write_text('{"processed_doc_ids": []}', encoding="utf-8")
        summary_path = str(path)
    else:
        doc_ids = [""]

    summary = pipeline._run_news_enrichment(
        SimpleNamespace(
            require_remote_persistence=False,
            source_kind="substack_public_article",
            doc_ids_from_summary=summary_path,
            doc_id=doc_ids,
            mode="all",
            order="stored",
            limit=None,
            provider="deepseek",
            model="deepseek-v4-flash",
            heuristic_only=True,
            dry_run=False,
            checkpoint_every=0,
            summary_path="",
        )
    )

    assert summary["requested_doc_ids"] == []
    assert summary["candidate_count"] == 0
    assert summary["selected_count"] == 0
    assert summary["processed_count"] == 0
    assert summary["gcs_snapshot_loaded"] is False
    assert saved_states == []

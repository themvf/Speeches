from types import SimpleNamespace

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


def test_news_enrichment_checkpoints_progress(monkeypatch):
    payload = {"documents": [_doc("doc-1"), _doc("doc-2"), _doc("doc-3")]}
    saved_doc_ids = []

    monkeypatch.setattr(pipeline, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setattr(pipeline, "_get_gcs_storage", lambda secrets: (None, "local"))
    monkeypatch.setattr(pipeline, "_load_custom_documents", lambda storage: payload)
    monkeypatch.setattr(pipeline, "_load_enrichment_state", lambda storage: {"entries": {}})
    monkeypatch.setattr(
        pipeline,
        "_save_enrichment_state",
        lambda storage, state, require_remote=False: saved_doc_ids.append(tuple(sorted(state["entries"].keys()))),
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

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


def _args(**overrides):
    base = dict(
        require_remote_persistence=False,
        source_kind="substack_public_article",
        doc_ids_from_summary="",
        doc_id=[],
        mode="only_missing_or_failed",
        order="stored",
        limit=None,
        provider="deepseek",
        model="deepseek-v4-flash",
        heuristic_only=True,
        dry_run=False,
        checkpoint_every=0,
        summary_path="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_candidate_deepseek_models_excludes_deprecated_chat():
    models = pipeline._candidate_deepseek_models()
    assert "deepseek-chat" not in models
    assert "deepseek-v4-pro" in models
    assert "deepseek-v4-flash" in models


def test_normalize_sentiment_fields_clamps_and_validates_label():
    normalized = pipeline._normalize_sentiment_fields(
        {"score": 5.0, "label": "VERY_POSITIVE", "rationale": "x" * 400}
    )
    assert normalized["score"] == 1.0
    assert normalized["label"] == "neutral"  # invalid label falls back to neutral
    assert len(normalized["rationale"]) == 300

    normalized_valid = pipeline._normalize_sentiment_fields({"score": -0.4, "label": "negative", "rationale": "ok"})
    assert normalized_valid == {"score": -0.4, "label": "negative", "rationale": "ok"}


def test_normalize_enrichment_payload_includes_sentiment():
    result = pipeline._normalize_enrichment_payload(
        {"summary": "s", "sentiment": {"score": 0.6, "label": "positive", "rationale": "celebratory framing"}}
    )
    assert result["sentiment"] == {"score": 0.6, "label": "positive", "rationale": "celebratory framing"}


def test_heuristic_enrichment_defaults_neutral_sentiment():
    doc = _doc("doc-heuristic")
    result = pipeline._heuristic_enrichment(doc)
    assert result["sentiment"] == {"score": 0.0, "label": "neutral", "rationale": ""}


def test_extract_usage_normalizes_chat_completion_shape():
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=120, completion_tokens=45, total_tokens=165))
    usage = pipeline._extract_usage(response)
    assert usage == {"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165}


def test_extract_usage_normalizes_responses_api_shape():
    response = SimpleNamespace(usage={"input_tokens": 300, "output_tokens": 80})
    usage = pipeline._extract_usage(response)
    assert usage == {"prompt_tokens": 300, "completion_tokens": 80, "total_tokens": 380}


def test_extract_usage_handles_missing_usage():
    assert pipeline._extract_usage(SimpleNamespace()) == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def _run_with_shared_state(monkeypatch, payload, state, times=1, **arg_overrides):
    monkeypatch.setattr(pipeline, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setattr(pipeline, "_get_gcs_storage", lambda secrets: (None, "local"))
    monkeypatch.setattr(pipeline, "_load_custom_documents", lambda storage: payload)
    monkeypatch.setattr(pipeline, "_load_enrichment_state", lambda storage: state)
    monkeypatch.setattr(pipeline, "_save_enrichment_state", lambda storage, st, require_remote=False: state.update(st))
    monkeypatch.setattr(pipeline, "_rebuild_rule_summaries", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_write_summary", lambda *args, **kwargs: None)

    summaries = []
    for _ in range(times):
        summaries.append(pipeline._run_news_enrichment(_args(**arg_overrides)))
    return summaries


def test_run_news_enrichment_mirrors_sentiment_into_entry(monkeypatch):
    payload = {"documents": [_doc("doc-1")]}
    state = {"entries": {}}

    _run_with_shared_state(monkeypatch, payload, state)

    entry = state["entries"]["doc-1"]
    assert entry["status"] == "fallback_enriched"
    assert entry["sentiment"]["status"] == "fallback_scored"
    assert entry["sentiment"]["label"] == "neutral"
    assert entry["sentiment"]["score"] == 0.0
    assert entry["attempt_count"] == 1


def test_run_news_enrichment_caps_retries_after_max_attempts(monkeypatch):
    payload = {"documents": [_doc("doc-1")]}
    state = {"entries": {}}

    # heuristic_only=True always yields status="fallback_enriched", so every
    # run increments attempt_count with no forward progress - exactly the
    # scenario the cap exists for.
    summaries = _run_with_shared_state(monkeypatch, payload, state, times=pipeline.MAX_ENRICHMENT_ATTEMPTS + 2)

    assert state["entries"]["doc-1"]["attempt_count"] == pipeline.MAX_ENRICHMENT_ATTEMPTS
    # Once attempt_count reaches the cap, the doc stops being selected at all.
    capped_runs = summaries[pipeline.MAX_ENRICHMENT_ATTEMPTS:]
    assert all(s["candidate_count"] == 0 for s in capped_runs)
    assert all(s["selected_count"] == 0 for s in capped_runs)


def test_run_news_enrichment_resets_attempt_count_on_success(monkeypatch):
    payload = {"documents": [_doc("doc-1")]}
    state = {"entries": {"doc-1": {"status": "fallback_enriched", "attempt_count": 2}}}

    def fake_agent(client, doc, model_name, provider="deepseek"):
        return (
            pipeline._normalize_enrichment_payload({"summary": "ok", "sentiment": {"score": 0.1, "label": "positive"}}),
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    monkeypatch.setattr(pipeline, "_get_model_client", lambda secrets, provider: object())
    monkeypatch.setattr(pipeline, "_run_enrichment_agent", fake_agent)
    _run_with_shared_state(monkeypatch, payload, state, heuristic_only=False)

    entry = state["entries"]["doc-1"]
    assert entry["status"] == "enriched"
    assert entry["attempt_count"] == 0
    assert entry["sentiment"]["status"] == "scored"
    assert entry["sentiment"]["label"] == "positive"

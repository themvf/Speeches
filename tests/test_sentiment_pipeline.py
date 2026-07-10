from types import SimpleNamespace

import run_sentiment_pipeline as sentiment


def test_needs_scoring_retries_fallback_and_failed():
    assert sentiment._needs_scoring(None) is True
    assert sentiment._needs_scoring({"doc_id": "x"}) is True
    assert sentiment._needs_scoring({"sentiment": {"status": "fallback_scored"}}) is True
    assert sentiment._needs_scoring({"sentiment": {"status": "failed"}}) is True
    assert sentiment._needs_scoring({"sentiment": {"status": ""}}) is True
    # Completed statuses are skipped.
    assert sentiment._needs_scoring({"sentiment": {"status": "scored"}}) is False
    assert sentiment._needs_scoring({"sentiment": {"status": "reviewed"}}) is False


def test_heuristic_sentiment_defaults_neutral_for_institutional_text():
    assert sentiment._heuristic_sentiment("The SEC announced an enforcement action.")["label"] == "neutral"
    assert sentiment._heuristic_sentiment("This reckless, dangerously overreaching rule is a fiasco.")["label"] == "negative"
    assert sentiment._heuristic_sentiment("A landmark victory, praised and hailed by all.")["label"] == "positive"


def _doc(doc_id, source_kind, text="Some editorial body text here."):
    return {"metadata": {"document_id": doc_id, "source_kind": source_kind, "title": f"T {doc_id}"}, "content": {"full_text": text}}


def test_only_missing_skips_already_scored(monkeypatch):
    payload = {"documents": [_doc("a", "newsapi_article"), _doc("b", "newsapi_article")]}
    state = {"entries": {"a": {"sentiment": {"status": "scored"}}}}
    saved = {}

    monkeypatch.setattr(sentiment.core, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setattr(sentiment.core, "_get_gcs_storage", lambda secrets: (None, "local"))
    monkeypatch.setattr(sentiment.core, "_load_custom_documents", lambda storage: payload)
    monkeypatch.setattr(sentiment.core, "_load_enrichment_state", lambda storage: state)
    monkeypatch.setattr(sentiment.core, "_save_enrichment_state", lambda storage, st, require_remote=False: saved.update(st))
    monkeypatch.setattr(sentiment.core, "_write_summary", lambda *a, **k: None)

    args = SimpleNamespace(
        source_kind="newsapi_article",
        mode="only_missing",
        doc_id=[],
        provider="deepseek",
        model="",
        heuristic_only=True,
        limit=None,
        dry_run=False,
        require_remote_persistence=False,
        summary_path="",
    )
    summary = sentiment._run_score(args)

    # Only "b" should be scored; "a" was already scored.
    assert summary["candidate_count"] == 1
    assert summary["selected_count"] == 1
    assert "sentiment" in state["entries"]["b"]
    # "a" untouched.
    assert state["entries"]["a"]["sentiment"]["status"] == "scored"

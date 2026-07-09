import ast
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse

import run_financial_news_pipeline as pipeline

# app.py is a Streamlit script that executes page logic (including real GCS/
# local-file reads and writes) at module scope rather than behind
# `if __name__ == "__main__":`, so `import app` is not safe to run in tests
# (it has been observed to overwrite data/custom_documents.json as a side
# effect of Streamlit's page-render code path executing on import). Instead,
# extract just the `_url_match_key` function (and the constant it depends on)
# from app.py's source and exec it in an isolated namespace, so this test can
# verify the two duplicate implementations stay in sync without importing
# the rest of the module.
_APP_PY_PATH = Path(__file__).resolve().parent.parent / "app.py"


def _load_app_url_match_key():
    source = _APP_PY_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    namespace = {"urlparse": urlparse, "parse_qsl": parse_qsl, "urlencode": urlencode}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_URL_MATCH_KEY_TRACKING_PARAMS"
            for target in node.targets
        ):
            exec(compile(ast.Module(body=[node], type_ignores=[]), filename=str(_APP_PY_PATH), mode="exec"), namespace)
        if isinstance(node, ast.FunctionDef) and node.name == "_url_match_key":
            exec(compile(ast.Module(body=[node], type_ignores=[]), filename=str(_APP_PY_PATH), mode="exec"), namespace)
    assert "_url_match_key" in namespace, "Could not find _url_match_key in app.py"
    return namespace["_url_match_key"]


def test_youtube_watch_urls_with_different_video_ids_produce_distinct_keys():
    key_a = pipeline._url_match_key("https://www.youtube.com/watch?v=abc123")
    key_b = pipeline._url_match_key("https://www.youtube.com/watch?v=zzz999")
    assert key_a != key_b
    assert key_a == "https://www.youtube.com/watch?v=abc123"
    assert key_b == "https://www.youtube.com/watch?v=zzz999"


def test_http_and_https_variants_of_same_url_match():
    assert pipeline._url_match_key("http://example.com/a") == pipeline._url_match_key(
        "https://example.com/a"
    )


def test_trailing_slash_is_ignored():
    assert pipeline._url_match_key("https://example.com/a/") == pipeline._url_match_key(
        "https://example.com/a"
    )


def test_query_param_order_does_not_affect_key():
    key_a = pipeline._url_match_key("https://example.com/a?b=1&a=2")
    key_b = pipeline._url_match_key("https://example.com/a?a=2&b=1")
    assert key_a == key_b


def test_tracking_params_are_stripped_so_rss_refetches_still_dedupe():
    key_a = pipeline._url_match_key("https://decrypt.co/330205/some-article")
    key_b = pipeline._url_match_key(
        "https://decrypt.co/330205/some-article?utm_source=rss&utm_medium=feed&fbclid=abc"
    )
    assert key_a == key_b


def test_non_tracking_query_params_are_not_stripped():
    key_a = pipeline._url_match_key("https://example.com/a?id=1")
    key_b = pipeline._url_match_key("https://example.com/a?id=2")
    assert key_a != key_b


def test_empty_url_returns_empty_key():
    assert pipeline._url_match_key("") == ""
    assert pipeline._url_match_key(None) == ""


def test_streamlit_app_and_pipeline_implementations_stay_in_sync():
    app_url_match_key = _load_app_url_match_key()
    urls = [
        "https://www.youtube.com/watch?v=abc123",
        "https://www.youtube.com/watch?v=zzz999",
        "http://example.com/a",
        "https://example.com/a/",
        "https://example.com/a?b=1&a=2",
        "https://decrypt.co/330205/some-article?utm_source=rss&utm_medium=feed&fbclid=abc",
        "",
    ]
    for url in urls:
        assert app_url_match_key(url) == pipeline._url_match_key(url)

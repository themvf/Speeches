"""Retiring news_connector_settings.json in favor of a single-row Neon table.

The blob had exactly one writer (the TS admin PUT route) and one Python
reader (run_financial_news_pipeline.py's CLI-override merge), so this is a
straight cutover - same single-JSONB-row shape and same "None means use
in-code defaults, never an error" contract as attention_sweep_config
(neon_feeds.get_attention_sweep_config), which this mirrors.

No real Postgres connection is used - psycopg2 is mocked, matching
test_neon_document_mirror.py's existing pattern.
"""

from unittest.mock import MagicMock, patch

import neon_feeds
import run_financial_news_pipeline as core


def test_get_news_connector_settings_returns_none_when_table_missing(monkeypatch):
    monkeypatch.setattr(neon_feeds, "_NEWS_CONNECTOR_SETTINGS_SCHEMA_ENSURED", False)
    with patch.object(neon_feeds, "_get_conn", side_effect=RuntimeError("relation does not exist")):
        assert neon_feeds.get_news_connector_settings() is None


def test_get_news_connector_settings_returns_none_when_no_row_saved(monkeypatch):
    monkeypatch.setattr(neon_feeds, "_NEWS_CONNECTOR_SETTINGS_SCHEMA_ENSURED", True)
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.cursor.return_value.__enter__.return_value = cursor

    with patch.object(neon_feeds, "_get_conn", return_value=conn):
        assert neon_feeds.get_news_connector_settings() is None


def test_get_news_connector_settings_returns_the_latest_row(monkeypatch):
    monkeypatch.setattr(neon_feeds, "_NEWS_CONNECTOR_SETTINGS_SCHEMA_ENSURED", True)
    saved_config = {"query": "custom query", "lookback_days": 5}
    cursor = MagicMock()
    cursor.fetchone.return_value = {"config": saved_config}
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.cursor.return_value.__enter__.return_value = cursor

    with patch.object(neon_feeds, "_get_conn", return_value=conn):
        result = neon_feeds.get_news_connector_settings()

    assert result == saved_config
    sql = cursor.execute.call_args[0][0]
    assert "ORDER BY id DESC LIMIT 1" in sql


def test_get_news_connector_settings_treats_a_non_dict_row_as_absent(monkeypatch):
    """Defensive: a malformed row must degrade to defaults, not crash callers."""
    monkeypatch.setattr(neon_feeds, "_NEWS_CONNECTOR_SETTINGS_SCHEMA_ENSURED", True)
    cursor = MagicMock()
    cursor.fetchone.return_value = {"config": "not-a-dict"}
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.cursor.return_value.__enter__.return_value = cursor

    with patch.object(neon_feeds, "_get_conn", return_value=conn):
        assert neon_feeds.get_news_connector_settings() is None


def test_pipeline_falls_back_to_defaults_when_neon_config_unavailable():
    """No storage/GCS fallback: a missing config means in-code defaults."""
    with patch.object(neon_feeds, "get_news_connector_settings", return_value=None):
        settings = core._load_news_connector_settings(storage=None)

    assert settings["query"] == core.NEWSAPI_DEFAULT_QUERY
    assert settings["lookback_days"] == 3


def test_pipeline_uses_the_saved_neon_config_when_present():
    saved_config = {
        "query": "stablecoin enforcement",
        "lookback_days": 5,
        "max_pages": 2,
        "sort_by": "relevancy",
    }
    with patch.object(neon_feeds, "get_news_connector_settings", return_value=saved_config):
        settings = core._load_news_connector_settings(storage=None)

    assert settings["query"] == "stablecoin enforcement"
    assert settings["lookback_days"] == 5
    assert settings["sort_by"] == "relevancy"


def test_pipeline_ignores_the_storage_argument_entirely():
    """The blob path is retired; a GCS storage object must never be touched."""
    unused_storage = MagicMock()
    with patch.object(neon_feeds, "get_news_connector_settings", return_value=None):
        core._load_news_connector_settings(storage=unused_storage)
    unused_storage.assert_not_called()


def test_pipeline_degrades_to_defaults_if_the_neon_read_itself_raises():
    with patch.object(neon_feeds, "get_news_connector_settings", side_effect=RuntimeError("connection refused")):
        settings = core._load_news_connector_settings(storage=None)
    assert settings["query"] == core.NEWSAPI_DEFAULT_QUERY

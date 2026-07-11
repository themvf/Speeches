"""Tests for the daily stock-attention rollup
(docs/stock-attention-spec.md §6). Aggregation math is a pure function
tested directly; DB interaction uses mocked psycopg2 per repo pattern."""

from datetime import UTC, date, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import aggregate_stock_attention as agg
import neon_feeds


def _row(ticker, source_id, author, subreddit="wallstreetbets", score=0, mood="neutral"):
    return {"ticker": ticker, "source_id": source_id, "author": author,
            "subreddit": subreddit, "score": score, "mood": mood}


# ─── pure aggregation ───────────────────────────────────────────────────────

def test_mention_count_dedupes_by_author():
    # One user posting the same ticker 3 times = 1 "real mention".
    rows = [
        _row("GME", "t3_a", "spammer"),
        _row("GME", "t3_b", "spammer"),
        _row("GME", "t1_c", "spammer"),
        _row("GME", "t3_d", "human2"),
    ]
    result = agg.aggregate_rows(rows)
    assert len(result) == 1
    assert result[0]["mention_count"] == 2      # spammer + human2
    assert result[0]["source_count"] == 4       # all four items still count as sources


def test_subreddit_spread_amplifies_score():
    one_board = agg.aggregate_rows([
        _row("AAA", f"t3_{i}", f"user{i}", subreddit="wallstreetbets") for i in range(5)
    ])[0]
    spread = agg.aggregate_rows([
        _row("AAA", f"t3_{i}", f"user{i}", subreddit=f"sub{i}") for i in range(5)
    ])[0]
    assert spread["weighted_score"] > one_board["weighted_score"]
    assert spread["subreddit_count"] == 5


def test_weighted_score_formula():
    assert agg.compute_weighted_score(10, 1, 1) == round(10 * 1.15 * 1.05, 4)
    # caps: subreddit_count at 6, source_count at 10
    assert agg.compute_weighted_score(10, 12, 50) == round(10 * (1 + 0.15 * 6) * (1 + 0.05 * 10), 4)


def test_mood_plurality_and_mixed():
    bullish = agg.aggregate_rows([
        _row("AAA", "t3_a", "u1", mood="bullish"),
        _row("AAA", "t3_b", "u2", mood="bullish"),
        _row("AAA", "t3_c", "u3", mood="bearish"),
    ])[0]
    assert bullish["mood"] == "bullish"

    mixed = agg.aggregate_rows([
        _row("BBB", "t3_a", "u1", mood="bullish"),
        _row("BBB", "t3_b", "u2", mood="bearish"),
    ])[0]
    assert mixed["mood"] == "mixed"

    neutral = agg.aggregate_rows([_row("CCC", "t3_a", "u1", mood="neutral")])[0]
    assert neutral["mood"] == "neutral"


def test_top_sources_ranked_by_score_capped_at_ten():
    rows = [_row("AAA", f"t3_{i:02d}", f"u{i}", score=i) for i in range(15)]
    result = agg.aggregate_rows(rows)[0]
    import json
    top = json.loads(result["top_source_ids"])
    assert len(top) == 10
    assert top[0] == "t3_14"  # highest score first


def test_rollups_sorted_by_score_desc():
    rows = [_row("SMALL", "t3_a", "u1")] + [
        _row("BIG", f"t3_{i}", f"user{i}", subreddit=f"sub{i}") for i in range(4)
    ]
    result = agg.aggregate_rows(rows)
    assert [r["ticker"] for r in result] == ["BIG", "SMALL"]


def test_day_bounds_are_utc_midnights():
    start, end = agg.day_bounds(date(2026, 7, 10))
    assert start == datetime(2026, 7, 10, tzinfo=UTC)
    assert end == datetime(2026, 7, 11, tzinfo=UTC)


# ─── _run against mocked DB ─────────────────────────────────────────────────

def _mock_conn(fetch_rows):
    cursor = MagicMock()
    cursor.fetchall.return_value = fetch_rows
    cursor.rowcount = 0
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.cursor.return_value.__enter__.return_value = cursor
    return conn, cursor


def test_run_replaces_date_wholesale_and_prunes(monkeypatch):
    monkeypatch.setattr(neon_feeds, "_STOCK_ATTENTION_SCHEMA_ENSURED", True)
    conn, cursor = _mock_conn([_row("GME", "t3_a", "u1")])
    with patch.object(neon_feeds, "_get_conn", return_value=conn):
        with patch("psycopg2.extras.execute_values") as mock_ev:
            summary = agg._run(date(2026, 7, 10), dry_run=False, retention_days=90)

    assert summary["ok"] is True
    assert summary["tickers"] == 1
    assert summary["rows_written"] == 1
    executed_sql = " ".join(str(call.args[0]) for call in cursor.execute.call_args_list)
    assert "DELETE FROM daily_stock_attention WHERE attention_date" in executed_sql
    assert "DELETE FROM reddit_attention_items WHERE created_utc" in executed_sql
    assert "source_type IN ('reddit_post', 'reddit_comment')" in executed_sql
    assert mock_ev.called
    # day-boundary params: created_utc window is [day, day+1)
    fetch_params = cursor.execute.call_args_list[0].args[1]
    assert fetch_params["day_start"] == datetime(2026, 7, 10, tzinfo=UTC)
    assert fetch_params["day_end"] == datetime(2026, 7, 11, tzinfo=UTC)
    assert summary["retention"]["skipped"] is False


def test_dry_run_writes_nothing_and_rolls_back(monkeypatch):
    monkeypatch.setattr(neon_feeds, "_STOCK_ATTENTION_SCHEMA_ENSURED", True)
    conn, cursor = _mock_conn([_row("GME", "t3_a", "u1")])
    with patch.object(neon_feeds, "_get_conn", return_value=conn):
        with patch("psycopg2.extras.execute_values") as mock_ev:
            summary = agg._run(date(2026, 7, 10), dry_run=True, retention_days=90)

    assert summary["tickers"] == 1
    assert summary["rows_written"] == 0
    assert summary["retention"] == {"skipped": True}
    assert not mock_ev.called
    executed_sql = " ".join(str(call.args[0]) for call in cursor.execute.call_args_list)
    assert "DELETE" not in executed_sql
    conn.rollback.assert_called_once()


def test_main_rejects_dangerous_retention():
    exit_code = agg.main(["--retention-days", "1", "--dry-run"])
    assert exit_code == 1


def test_main_reports_failure_as_json():
    with patch.object(agg.neon_feeds, "_ensure_stock_attention_schema", side_effect=RuntimeError("DATABASE_URL is not set")):
        exit_code = agg.main(["--date", "2026-07-10"])
    assert exit_code == 1

"""Tests for the daily stock-attention rollup
(docs/stock-attention-spec.md §6). Aggregation math is a pure function
tested directly; DB interaction uses mocked psycopg2 per repo pattern."""

import json
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
            summary = agg._run(date(2026, 7, 10), dry_run=False, retention_days=90, skip_news=True, skip_market=True)

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
            summary = agg._run(date(2026, 7, 10), dry_run=True, retention_days=90, skip_news=True, skip_market=True)

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


# ─── item 1: news-channel attention ─────────────────────────────────────────

def test_compute_news_ticker_counts_counts_distinct_articles():
    articles = [
        {"id": 1, "title": "NVDA earnings beat", "description": ""},
        {"id": 2, "title": "Chipmakers rally", "description": "NVDA and AMD both up"},
        {"id": 3, "title": "Weather report", "description": "sunny today"},
    ]
    counts = agg.compute_news_ticker_counts(articles)
    assert counts == {"NVDA": 2, "AMD": 1}


def test_compute_news_ticker_counts_one_article_one_count_per_ticker():
    # NVDA mentioned twice in the same article's text - still 1 article.
    articles = [{"id": 1, "title": "NVDA up, NVDA up again", "description": ""}]
    assert agg.compute_news_ticker_counts(articles) == {"NVDA": 1}


def test_compute_news_ticker_counts_respects_ambiguous_gating():
    # Same false-positive gate as Reddit - "ALL" bare must not count.
    articles = [{"id": 1, "title": "I lost all my savings today", "description": ""}]
    assert agg.compute_news_ticker_counts(articles) == {}


def test_merge_news_counts_adds_to_existing_reddit_ticker():
    reddit_rollups = agg.aggregate_rows([_row("NVDA", "t3_a", "u1")])
    merged = agg.merge_news_counts(reddit_rollups, {"NVDA": 5})
    row = merged[0]
    assert row["reddit_count"] == 1
    assert row["news_count"] == 5
    assert row["total_mention_count"] == 6
    assert row["mention_count"] == 1  # unchanged meaning


def test_merge_news_counts_appends_news_only_ticker():
    reddit_rollups = agg.aggregate_rows([_row("NVDA", "t3_a", "u1")])
    merged = agg.merge_news_counts(reddit_rollups, {"MSFT": 3})
    tickers = {row["ticker"]: row for row in merged}
    assert "MSFT" in tickers
    msft = tickers["MSFT"]
    assert msft["reddit_count"] == 0
    assert msft["news_count"] == 3
    assert msft["total_mention_count"] == 3
    assert msft["weighted_score"] == 0.0
    assert msft["mood"] == "neutral"
    assert json.loads(msft["top_source_ids"]) == []


def test_merge_news_counts_with_empty_news_is_noop():
    reddit_rollups = agg.aggregate_rows([_row("NVDA", "t3_a", "u1")])
    merged = agg.merge_news_counts(reddit_rollups, {})
    assert len(merged) == 1
    assert merged[0]["news_count"] == 0
    assert merged[0]["total_mention_count"] == merged[0]["reddit_count"]


# ─── item 2: market context + divergence ────────────────────────────────────

def test_divergence_attention_spike_no_price_move():
    assert agg.compute_divergence(rank=5, price_pct=0.3) == "attention_spike_no_price_move"
    assert agg.compute_divergence(rank=20, price_pct=-1.0) == "attention_spike_no_price_move"


def test_divergence_price_move_no_attention():
    assert agg.compute_divergence(rank=21, price_pct=6.0) == "price_move_no_attention"
    assert agg.compute_divergence(rank=100, price_pct=-8.0) == "price_move_no_attention"


def test_divergence_none_when_unremarkable_or_no_price():
    assert agg.compute_divergence(rank=5, price_pct=3.0) == ""       # top-ranked but moderate move
    assert agg.compute_divergence(rank=50, price_pct=0.5) == ""      # low-ranked, small move
    assert agg.compute_divergence(rank=5, price_pct=None) == ""      # no price data


def test_merge_market_context_attaches_by_rank_position():
    rollups = agg.aggregate_rows([
        _row("BIG", f"t3_{i}", f"u{i}", subreddit=f"sub{i}") for i in range(5)
    ]) + agg.aggregate_rows([_row("SMALL", "t3_x", "ux")])
    market = {
        "BIG": {"price_close": 100.0, "price_pct": 0.5, "volume": 1000, "volume_vs_20d": 1.1},
        "SMALL": {"price_close": 5.0, "price_pct": 7.0, "volume": 50, "volume_vs_20d": 3.0},
    }
    result = agg.merge_market_context(rollups, market)
    big = next(r for r in result if r["ticker"] == "BIG")
    small = next(r for r in result if r["ticker"] == "SMALL")
    assert big["price_close"] == 100.0
    assert big["divergence"] == "attention_spike_no_price_move"  # rank 1, small move
    assert small["price_pct"] == 7.0


def test_merge_market_context_missing_ticker_gets_nulls():
    rollups = agg.aggregate_rows([_row("NVDA", "t3_a", "u1")])
    result = agg.merge_market_context(rollups, {})
    assert result[0]["price_close"] is None
    assert result[0]["divergence"] == ""


# ─── integrated _run with news/market (mocked DB + mocked fetchers) ────────

def test_run_integrates_news_counts(monkeypatch):
    monkeypatch.setattr(neon_feeds, "_STOCK_ATTENTION_SCHEMA_ENSURED", True)
    conn, cursor = _mock_conn([])
    cursor.fetchall.side_effect = [
        [_row("NVDA", "t3_a", "u1")],
        [{"id": 1, "title": "MSFT news today", "description": ""}],
    ]
    with patch.object(neon_feeds, "_get_conn", return_value=conn):
        with patch("psycopg2.extras.execute_values") as mock_ev:
            summary = agg._run(date(2026, 7, 10), dry_run=False, retention_days=90, skip_market=True)

    assert summary["news_articles_scanned"] == 1
    assert summary["news_only_tickers"] == 1  # MSFT added, NVDA already present
    assert summary["tickers"] == 2
    rows_written = mock_ev.call_args[0][2]
    by_ticker = {row[1]: row for row in rows_written}
    assert by_ticker["MSFT"][9] == 0    # reddit_count
    assert by_ticker["MSFT"][10] == 1   # news_count
    assert by_ticker["MSFT"][11] == 1   # total_mention_count


def test_run_integrates_market_context(monkeypatch):
    monkeypatch.setattr(neon_feeds, "_STOCK_ATTENTION_SCHEMA_ENSURED", True)
    conn, cursor = _mock_conn([_row("NVDA", "t3_a", "u1")])
    with patch.object(neon_feeds, "_get_conn", return_value=conn):
        with patch("psycopg2.extras.execute_values") as mock_ev:
            with patch.object(
                agg.yahoo_market_data, "fetch_market_context_batch",
                return_value={"NVDA": {"price_close": 120.0, "price_pct": 2.0, "volume": 500, "volume_vs_20d": 1.5}},
            ) as mock_fetch:
                summary = agg._run(date(2026, 7, 10), dry_run=False, retention_days=90, skip_news=True)

    mock_fetch.assert_called_once_with(["NVDA"])
    assert summary["market_context_fetched"] == 1
    assert summary["market_context_failed"] == 0
    rows_written = mock_ev.call_args[0][2]
    row = rows_written[0]
    assert row[12] == 120.0  # price_close
    assert row[13] == 2.0    # price_pct


def test_run_market_fetch_failure_leaves_nulls_not_crash(monkeypatch):
    monkeypatch.setattr(neon_feeds, "_STOCK_ATTENTION_SCHEMA_ENSURED", True)
    conn, cursor = _mock_conn([_row("NVDA", "t3_a", "u1")])
    with patch.object(neon_feeds, "_get_conn", return_value=conn):
        with patch("psycopg2.extras.execute_values") as mock_ev:
            with patch.object(agg.yahoo_market_data, "fetch_market_context_batch", return_value={}):
                summary = agg._run(date(2026, 7, 10), dry_run=False, retention_days=90, skip_news=True)

    assert summary["ok"] is True
    assert summary["market_context_failed"] == 1
    row = mock_ev.call_args[0][2][0]
    assert row[12] is None  # price_close

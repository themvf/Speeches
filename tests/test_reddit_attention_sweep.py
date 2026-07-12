"""Tests for the Reddit attention sweep (docs/stock-attention-spec.md §4)
and its neon_feeds storage writers.

PRAW is never imported - the reddit client is a fake built from
SimpleNamespace objects, patched in via _build_reddit. psycopg2 is mocked
for the writer tests, matching the repo's existing pattern."""

import argparse
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import neon_feeds
import reddit_attention_sweep as sweep


# ─── fakes ──────────────────────────────────────────────────────────────────

def _submission(sid, title, selftext="", author="user1", score=10, created=1720656000.0):
    comments = MagicMock()
    comments.replace_more = MagicMock()
    comments.__iter__ = lambda self: iter([])
    return SimpleNamespace(
        fullname=f"t3_{sid}",
        title=title,
        selftext=selftext,
        author=author,
        score=score,
        created_utc=created,
        permalink=f"/r/test/comments/{sid}/slug/",
        comments=comments,
    )


def _comment(cid, body, author="user2", score=5, created=1720656100.0):
    return SimpleNamespace(
        fullname=f"t1_{cid}",
        body=body,
        author=author,
        score=score,
        created_utc=created,
        permalink=f"/r/test/comments/parent/slug/{cid}/",
    )


def _reddit_with(new_posts, hot_posts):
    subreddit = MagicMock()
    subreddit.new.return_value = list(new_posts)
    subreddit.hot.return_value = list(hot_posts)
    reddit = MagicMock()
    reddit.subreddit.return_value = subreddit
    return reddit


def _args(**overrides):
    base = dict(subreddits="wallstreetbets", include_tier2=False,
                limit_new=100, hot_threads=20, dry_run=True, summary_path="")
    base.update(overrides)
    return argparse.Namespace(**base)


# ─── mood heuristic ─────────────────────────────────────────────────────────

def test_mood_bullish_bearish_neutral():
    assert sweep.infer_reddit_mood("loading up on calls, this will moon") == "bullish"
    assert sweep.infer_reddit_mood("buying puts, this is going to crash") == "bearish"
    assert sweep.infer_reddit_mood("earnings are on Thursday") == "neutral"
    assert sweep.infer_reddit_mood("calls or puts, could go either way") == "neutral"


# ─── sweep behavior ─────────────────────────────────────────────────────────

def test_sweep_collects_post_and_comment_mentions():
    post = _submission("p1", "NVDA earnings play", selftext="going long $MU too")
    comment = _comment("c1", "$GME never dies")
    hot = _submission("p2", "Daily Discussion Thread")
    hot.comments.__iter__ = lambda self: iter([comment])

    with patch.object(sweep, "_build_reddit", return_value=_reddit_with([post], [hot])):
        summary = sweep._run(_args())

    assert summary["ok"] is True
    assert summary["items_with_tickers"] == 2
    assert summary["unique_tickers"] == 3  # NVDA, MU, GME
    assert summary["posts_scanned"] == 1
    assert summary["comments_scanned"] == 1
    # dry run: nothing written
    assert summary["item_rows_written"] == 0
    assert summary["mention_rows_written"] == 0


def test_sweep_skips_bot_authors_and_no_ticker_items():
    bot_post = _submission("p1", "NVDA to the moon", author="AutoModerator")
    plain_post = _submission("p2", "what should I have for lunch")
    with patch.object(sweep, "_build_reddit", return_value=_reddit_with([bot_post, plain_post], [])):
        summary = sweep._run(_args())
    assert summary["items_with_tickers"] == 0
    assert summary["unique_tickers"] == 0


def test_sweep_writes_rows_when_not_dry_run():
    post = _submission("p1", "yolo $TSLA calls")
    with patch.object(sweep, "_build_reddit", return_value=_reddit_with([post], [])):
        with patch.object(sweep.neon_feeds, "upsert_reddit_attention_items", return_value=1) as mock_items:
            with patch.object(sweep.neon_feeds, "insert_ticker_mentions", return_value=1) as mock_mentions:
                summary = sweep._run(_args(dry_run=False))

    assert summary["item_rows_written"] == 1
    assert summary["mention_rows_written"] == 1
    item_row = mock_items.call_args[0][0][0]
    assert item_row["source_id"] == "t3_p1"
    assert item_row["kind"] == "post"
    assert item_row["mood"] == "bullish"
    assert item_row["created_utc"] == datetime.fromtimestamp(1720656000.0, tz=UTC)
    assert item_row["permalink"].startswith("https://www.reddit.com/r/")
    mention_row = mock_mentions.call_args[0][0][0]
    assert mention_row == {
        "source_type": "reddit_post",
        "source_id": "t3_p1",
        "mention_type": "ticker",
        "value": "TSLA",
        "normalized_value": "tsla",
        "confidence": 1.0,
    }


def test_comment_mentions_carry_parent_title_and_comment_source_type():
    comment = _comment("c9", "adding to my NVDA position")
    hot = _submission("p5", "Tech megathread")
    hot.comments.__iter__ = lambda self: iter([comment])
    with patch.object(sweep, "_build_reddit", return_value=_reddit_with([], [hot])):
        with patch.object(sweep.neon_feeds, "upsert_reddit_attention_items", return_value=1) as mock_items:
            with patch.object(sweep.neon_feeds, "insert_ticker_mentions", return_value=1) as mock_mentions:
                sweep._run(_args(dry_run=False))

    item_row = mock_items.call_args[0][0][0]
    assert item_row["kind"] == "comment"
    assert item_row["title"] == "Tech megathread"
    assert mock_mentions.call_args[0][0][0]["source_type"] == "reddit_comment"
    hot.comments.replace_more.assert_called_once_with(limit=0)


def test_ambiguous_bare_symbols_do_not_produce_mentions():
    # The caps-rant false positive the resolver gating exists for.
    post = _submission("p1", "I LOST ALL MY MONEY AND NOW I HAVE NOTHING")
    with patch.object(sweep, "_build_reddit", return_value=_reddit_with([post], [])):
        summary = sweep._run(_args())
    assert summary["items_with_tickers"] == 0


def test_per_subreddit_errors_are_collected_not_fatal():
    ok_sub = MagicMock()
    ok_post = _submission("p1", "NVDA calls")
    ok_sub.new.return_value = [ok_post]
    ok_sub.hot.return_value = []
    bad_sub = MagicMock()
    bad_sub.new.side_effect = RuntimeError("403 blocked")

    reddit = MagicMock()
    reddit.subreddit.side_effect = lambda name: bad_sub if name == "badsub" else ok_sub

    with patch.object(sweep, "_build_reddit", return_value=reddit):
        summary = sweep._run(_args(subreddits="wallstreetbets,badsub"))

    assert summary["ok"] is True  # one of two failed - partial success
    assert summary["failed_count"] == 1
    assert "badsub" in summary["errors"][0]
    assert summary["items_with_tickers"] == 1


def test_all_subreddits_failing_marks_run_not_ok():
    bad_sub = MagicMock()
    bad_sub.new.side_effect = RuntimeError("403 blocked")
    reddit = MagicMock()
    reddit.subreddit.return_value = bad_sub
    with patch.object(sweep, "_build_reddit", return_value=reddit):
        summary = sweep._run(_args())
    assert summary["ok"] is False


def test_main_records_source_health_even_on_total_failure():
    with patch.object(sweep, "_build_reddit", side_effect=RuntimeError("REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET are not set")):
        with patch.object(sweep, "record_source_health") as mock_health:
            exit_code = sweep.main(["--dry-run"])
    assert exit_code == 1
    recorded = mock_health.call_args[0][0]
    assert recorded["source_key"] == "reddit_attention_sweep"
    assert recorded["ok"] is False


# ─── neon_feeds writers (mocked psycopg2) ───────────────────────────────────

def _mock_conn():
    cursor = MagicMock()
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.cursor.return_value.__enter__.return_value = cursor
    return conn, cursor


def test_upsert_items_strips_nul_bytes_and_skips_missing_ids(monkeypatch):
    monkeypatch.setattr(neon_feeds, "_REDDIT_ATTENTION_SCHEMA_ENSURED", True)
    conn, cursor = _mock_conn()
    items = [
        {"source_id": "t3_a", "kind": "post", "subreddit": "stocks", "author": "u1",
         "title": "bad\x00title", "permalink": "https://x", "created_utc": datetime.now(UTC), "score": 3, "mood": "neutral"},
        {"source_id": "", "kind": "post"},
    ]
    with patch.object(neon_feeds, "_get_conn", return_value=conn):
        with patch("psycopg2.extras.execute_values") as mock_ev:
            written = neon_feeds.upsert_reddit_attention_items(items)
    assert written == 1
    rows = mock_ev.call_args[0][2]
    assert rows[0][4] == "badtitle"  # NUL stripped from title
    sql = mock_ev.call_args[0][1]
    assert "ON CONFLICT (source_id) DO UPDATE" in sql


def test_insert_mentions_uses_do_nothing_conflict(monkeypatch):
    monkeypatch.setattr(neon_feeds, "_REDDIT_ATTENTION_SCHEMA_ENSURED", True)
    conn, cursor = _mock_conn()
    mentions = [{
        "source_type": "reddit_post", "source_id": "t3_a", "mention_type": "ticker",
        "value": "GME", "normalized_value": "gme", "confidence": 1.0,
    }]
    with patch.object(neon_feeds, "_get_conn", return_value=conn):
        with patch("psycopg2.extras.execute_values") as mock_ev:
            written = neon_feeds.insert_ticker_mentions(mentions)
    assert written == 1
    sql = mock_ev.call_args[0][1]
    assert "ON CONFLICT (source_type, source_id, mention_type, normalized_value) DO NOTHING" in sql


def test_writers_are_noops_for_empty_input():
    with patch.object(neon_feeds, "_get_conn") as mock_conn:
        assert neon_feeds.upsert_reddit_attention_items([]) == 0
        assert neon_feeds.insert_ticker_mentions([]) == 0
        assert neon_feeds.upsert_author_stats_batch([]) == 0
        assert neon_feeds.upsert_author_account_info([]) == 0
        mock_conn.assert_not_called()


# ─── item 4: admin-managed sweep config ─────────────────────────────────────

def _config(**overrides):
    base = {
        "subreddits": [
            {"name": "wallstreetbets", "tier": 1, "weight": 1.0, "active": True},
            {"name": "pennystocks", "tier": 2, "weight": 0.7, "active": True},
            {"name": "dividends", "tier": 2, "weight": 0.9, "active": False},
        ],
        "bot_blocklist": ["customspambot"],
        "symbol_overrides": {"force_ambiguous": [], "force_unambiguous": []},
        "author_weighting": {},
    }
    base.update(overrides)
    return base


def test_sweep_uses_config_subreddits_and_skips_inactive():
    post = _submission("p1", "NVDA calls")
    with patch.object(sweep.neon_feeds, "get_attention_sweep_config", return_value=_config()):
        with patch.object(sweep, "_build_reddit", return_value=_reddit_with([post], [])):
            summary = sweep._run(_args(subreddits=""))
    assert summary["config_source"] == "db"
    assert summary["subreddits"] == ["wallstreetbets", "pennystocks"]  # dividends inactive


def test_sweep_explicit_subreddits_override_config():
    post = _submission("p1", "NVDA calls")
    with patch.object(sweep.neon_feeds, "get_attention_sweep_config", return_value=_config()):
        with patch.object(sweep, "_build_reddit", return_value=_reddit_with([post], [])):
            summary = sweep._run(_args(subreddits="stocks"))
    assert summary["subreddits"] == ["stocks"]


def test_sweep_falls_back_to_tier_defaults_when_config_unavailable():
    post = _submission("p1", "NVDA calls")
    with patch.object(sweep.neon_feeds, "get_attention_sweep_config", return_value=None):
        with patch.object(sweep, "_build_reddit", return_value=_reddit_with([post], [])):
            summary = sweep._run(_args(subreddits=""))
    assert summary["config_source"] == "defaults"
    assert summary["subreddits"] == sweep.TIER1_SUBREDDITS


def test_sweep_config_bot_blocklist_extends_defaults():
    bot_post = _submission("p1", "NVDA to the moon", author="CustomSpamBot")
    with patch.object(sweep.neon_feeds, "get_attention_sweep_config", return_value=_config()):
        with patch.object(sweep, "_build_reddit", return_value=_reddit_with([bot_post], [])):
            summary = sweep._run(_args(subreddits=""))
    assert summary["items_with_tickers"] == 0


def test_sweep_applies_symbol_overrides_to_resolver():
    import ticker_resolver
    # ALL is normally gated; the admin force-unambiguous override allows it.
    post = _submission("p1", "ALL is undervalued here")
    config = _config(symbol_overrides={"force_ambiguous": [], "force_unambiguous": ["ALL"]})
    try:
        with patch.object(sweep.neon_feeds, "get_attention_sweep_config", return_value=config):
            with patch.object(sweep, "_build_reddit", return_value=_reddit_with([post], [])):
                # explicit subreddit isolates the assertion to the override
                # (config subreddit list is exercised by the tests above)
                summary = sweep._run(_args(subreddits="stocks"))
        assert summary["items_with_tickers"] == 1
        assert summary["unique_tickers"] == 1
    finally:
        ticker_resolver.clear_runtime_overrides()


# ─── item 5: account-info enrichment ────────────────────────────────────────

def test_account_info_enrichment_respects_budget_cap():
    post = _submission("p1", "yolo $TSLA calls")
    reddit = _reddit_with([post], [])
    missing = [f"user{i}" for i in range(40)]
    redditor = SimpleNamespace(created_utc=1600000000.0, link_karma=42)
    reddit.redditor.return_value = redditor

    with patch.object(sweep.neon_feeds, "get_attention_sweep_config", return_value=None):
        with patch.object(sweep, "_build_reddit", return_value=reddit):
            with patch.object(sweep.neon_feeds, "upsert_reddit_attention_items", return_value=1):
                with patch.object(sweep.neon_feeds, "insert_ticker_mentions", return_value=1):
                    with patch.object(sweep.neon_feeds, "get_authors_needing_account_info", return_value=missing):
                        with patch.object(sweep.neon_feeds, "upsert_author_account_info", return_value=25) as mock_upsert:
                            summary = sweep._run(_args(dry_run=False))

    assert reddit.redditor.call_count == sweep.ACCOUNT_INFO_LOOKUPS_PER_SWEEP
    rows = mock_upsert.call_args[0][0]
    assert len(rows) == sweep.ACCOUNT_INFO_LOOKUPS_PER_SWEEP
    assert rows[0]["link_karma"] == 42
    assert summary["author_info_enriched"] == 25


def test_get_authors_needing_account_info_prioritizes_board_then_recent(monkeypatch):
    monkeypatch.setattr(neon_feeds, "_STOCK_ATTENTION_SCHEMA_ENSURED", True)
    cursor = MagicMock()
    # First execute → board query (top-by-items_total missing); second →
    # which recent authors already have account info.
    cursor.fetchall.side_effect = [
        [{"author": "topA"}, {"author": "topB"}],  # board authors
        [{"author": "known_recent"}],               # recent already enriched
    ]
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.cursor.return_value.__enter__.return_value = cursor

    with patch.object(neon_feeds, "_get_conn", return_value=conn):
        result = neon_feeds.get_authors_needing_account_info(
            ["known_recent", "fresh1", "fresh2", "topA"],  # topA already in board; known_recent skipped
            board_budget=17,
            recent_budget=8,
        )

    # board authors first, then fresh recent authors (dedup topA, skip known_recent)
    assert result[:2] == ["topA", "topB"]
    assert "fresh1" in result and "fresh2" in result
    assert "known_recent" not in result
    assert result.count("topA") == 1
    board_limit = cursor.execute.call_args_list[0].args[1]
    assert board_limit == (17,)  # board budget passed through to the LIMIT


def test_get_authors_needing_account_info_respects_recent_reserve(monkeypatch):
    monkeypatch.setattr(neon_feeds, "_STOCK_ATTENTION_SCHEMA_ENSURED", True)
    cursor = MagicMock()
    cursor.fetchall.side_effect = [
        [{"author": f"b{i}"} for i in range(17)],  # board fills its budget
        [],                                          # no recent known
    ]
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.cursor.return_value.__enter__.return_value = cursor

    with patch.object(neon_feeds, "_get_conn", return_value=conn):
        result = neon_feeds.get_authors_needing_account_info(
            [f"fresh{i}" for i in range(20)], board_budget=17, recent_budget=8
        )

    # 17 board + at most 8 recent = 25 total, recent reserve honored
    assert len(result) == 25
    assert len([a for a in result if a.startswith("fresh")]) == 8


def test_account_info_enrichment_failure_is_nonfatal():
    post = _submission("p1", "yolo $TSLA calls")
    with patch.object(sweep.neon_feeds, "get_attention_sweep_config", return_value=None):
        with patch.object(sweep, "_build_reddit", return_value=_reddit_with([post], [])):
            with patch.object(sweep.neon_feeds, "upsert_reddit_attention_items", return_value=1):
                with patch.object(sweep.neon_feeds, "insert_ticker_mentions", return_value=1):
                    with patch.object(sweep.neon_feeds, "get_authors_needing_account_info", side_effect=RuntimeError("db down")):
                        summary = sweep._run(_args(dry_run=False))
    assert summary["ok"] is True
    assert "db down" in summary["author_info_error"]


# ─── neon_feeds config + author writers (mocked psycopg2) ───────────────────

def test_get_attention_sweep_config_fail_soft():
    with patch.object(neon_feeds, "_get_conn", side_effect=RuntimeError("DATABASE_URL is not set")):
        assert neon_feeds.get_attention_sweep_config() is None


def test_upsert_author_stats_preserves_account_columns(monkeypatch):
    monkeypatch.setattr(neon_feeds, "_STOCK_ATTENTION_SCHEMA_ENSURED", True)
    conn, cursor = _mock_conn()
    rows = [{"author": "u1", "items_total": 3, "tickers_distinct": 2, "subreddits_distinct": 1, "top_ticker_share": 0.5}]
    with patch.object(neon_feeds, "_get_conn", return_value=conn):
        with patch("psycopg2.extras.execute_values") as mock_ev:
            written = neon_feeds.upsert_author_stats_batch(rows)
    assert written == 1
    sql = " ".join(mock_ev.call_args[0][1].split())
    assert "ON CONFLICT (author) DO UPDATE" in sql
    # the daily recompute must never clobber the sweep-owned columns
    assert "account_created = EXCLUDED" not in sql
    assert "link_karma = EXCLUDED" not in sql


def test_upsert_author_account_info_touches_only_account_columns(monkeypatch):
    monkeypatch.setattr(neon_feeds, "_STOCK_ATTENTION_SCHEMA_ENSURED", True)
    conn, cursor = _mock_conn()
    rows = [{"author": "u1", "account_created": datetime.now(UTC), "link_karma": 10}]
    with patch.object(neon_feeds, "_get_conn", return_value=conn):
        with patch("psycopg2.extras.execute_values") as mock_ev:
            written = neon_feeds.upsert_author_account_info(rows)
    assert written == 1
    sql = " ".join(mock_ev.call_args[0][1].split())
    assert "account_created = EXCLUDED.account_created" in sql
    assert "items_total" not in sql

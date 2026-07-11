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
        mock_conn.assert_not_called()

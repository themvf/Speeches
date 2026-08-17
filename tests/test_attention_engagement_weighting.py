"""Enhancement 1: upvote engagement as a bounded amplifier on weighted_score.

Before this, reddit_attention_items.score was stored on every item and used
only to order the ten permalinks shown in the drawer - a ticker named once in
a 4,000-upvote post ranked exactly level with one named in a downvoted comment.
"""

import aggregate_stock_attention as agg


def test_engagement_dedupes_by_source():
    """Input has one row per item+ticker pair, so a post naming three tickers
    must not have its score counted three times."""
    scored = [(500, "post_a"), (500, "post_a"), (500, "post_a"), (10, "post_b")]
    assert agg.compute_engagement_score(scored) == 510


def test_downvoted_items_clamp_to_zero_and_never_subtract():
    assert agg.compute_engagement_score([(-40, "post_a")]) == 0
    # A downvoted post must not drag down a genuinely popular one.
    assert agg.compute_engagement_score([(-40, "post_a"), (100, "post_b")]) == 100


def test_engagement_factor_is_bounded_and_log_scaled():
    assert agg.compute_engagement_factor(0) == 1.0
    assert agg.compute_engagement_factor(-5) == 1.0
    # Order-of-magnitude steps produce equal increments, not linear ones.
    step_10_to_100 = agg.compute_engagement_factor(100) - agg.compute_engagement_factor(10)
    step_1k_to_10k = agg.compute_engagement_factor(10_000) - agg.compute_engagement_factor(1_000)
    assert abs(step_10_to_100 - step_1k_to_10k) < 0.005
    # Capped, so one viral thread cannot dominate the board.
    assert agg.compute_engagement_factor(10**9) == agg.compute_engagement_factor(10**5)
    assert agg.compute_engagement_factor(10**9) <= 1.5


def test_weighted_score_unchanged_when_engagement_omitted():
    """Every existing caller and every historical recompute must score exactly
    as before - the amplifier defaults to a 1.0x no-op."""
    assert agg.compute_weighted_score(10, 3, 5) == agg.compute_weighted_score(10, 3, 5, 0)


def test_engagement_reorders_equal_mention_counts():
    """The actual point of the change: same mentions, same spread, different
    reach - the one people actually read ranks higher."""
    quiet = agg.compute_weighted_score(4, 2, 4, engagement_score=2)
    loud = agg.compute_weighted_score(4, 2, 4, engagement_score=4000)
    assert loud > quiet


def test_engagement_cannot_outrank_a_much_broader_ticker():
    """Bounded amplification: breadth still beats a single viral thread."""
    one_viral_thread = agg.compute_weighted_score(3, 1, 1, engagement_score=500_000)
    broad_conversation = agg.compute_weighted_score(12, 5, 9, engagement_score=0)
    assert broad_conversation > one_viral_thread


def test_aggregate_rows_emits_engagement_and_uses_it():
    rows = [
        {"ticker": "AAA", "source_id": "p1", "author": "u1", "subreddit": "stocks", "score": 900, "mood": "bullish"},
        {"ticker": "AAA", "source_id": "p2", "author": "u2", "subreddit": "stocks", "score": 100, "mood": "bullish"},
        {"ticker": "BBB", "source_id": "p3", "author": "u3", "subreddit": "stocks", "score": 0, "mood": "neutral"},
        {"ticker": "BBB", "source_id": "p4", "author": "u4", "subreddit": "stocks", "score": 0, "mood": "neutral"},
    ]
    out = {r["ticker"]: r for r in agg.aggregate_rows(rows)}
    assert out["AAA"]["engagement_score"] == 1000
    assert out["BBB"]["engagement_score"] == 0
    # Identical mention/source/subreddit counts; engagement is the only
    # difference, and it decides the ranking.
    assert out["AAA"]["mention_count"] == out["BBB"]["mention_count"]
    assert out["AAA"]["weighted_score"] > out["BBB"]["weighted_score"]


def test_news_only_rows_carry_zero_engagement():
    merged = agg.merge_news_counts([], {"CCC": 4}, {"CCC": [1, 2]})
    assert merged[0]["engagement_score"] == 0

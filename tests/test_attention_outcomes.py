"""Enhancement 2: forward-return scoring for Reddit attention.

Turns the attention tracker from descriptive ("what is loud") into evaluative
("was loud right"). The pure functions carry all the correctness; the DB layer
is thin.
"""

from datetime import date

import attention_outcomes as ao


# A short synthetic close series with a deliberate weekend gap between
# 2026-08-07 (Fri) and 2026-08-10 (Mon), so trading-day walking is exercised.
CLOSES = {
    "2026-08-05": 100.0,
    "2026-08-06": 101.0,
    "2026-08-07": 102.0,
    "2026-08-10": 110.0,
    "2026-08-11": 112.0,
    "2026-08-12": 90.0,
}


def test_horizon_walks_trading_days_not_calendar_days():
    """A Friday attention day with a 1-day horizon must resolve to Monday, not
    to a Saturday bar that does not exist."""
    got = ao.forward_return_pct(CLOSES, date(2026, 8, 7), 1)
    assert got == round(((110.0 - 102.0) / 102.0) * 100, 4)


def test_weekend_attention_grades_against_the_last_actionable_close():
    """Chatter on a Saturday is graded from Friday's close - the last price
    anyone reading it could have acted on."""
    saturday = ao.forward_return_pct(CLOSES, date(2026, 8, 8), 1)
    friday = ao.forward_return_pct(CLOSES, date(2026, 8, 7), 1)
    assert saturday == friday


def test_unelapsed_horizon_returns_none_rather_than_a_partial_number():
    assert ao.forward_return_pct(CLOSES, date(2026, 8, 12), 1) is None
    assert ao.forward_return_pct(CLOSES, date(2026, 8, 5), 20) is None


def test_attention_before_the_series_starts_is_unscoreable():
    assert ao.forward_return_pct(CLOSES, date(2026, 7, 1), 1) is None


def test_grading_rewards_direction_not_magnitude():
    assert ao.grade_direction("bullish", 6.0) is True
    assert ao.grade_direction("bullish", -6.0) is False
    assert ao.grade_direction("bearish", -6.0) is True
    assert ao.grade_direction("bearish", 6.0) is False


def test_flat_moves_and_non_directional_moods_are_not_scored():
    """None means 'no call to grade', and must never be silently counted as a
    loss - otherwise every neutral day would drag a source's hit rate down."""
    assert ao.grade_direction("bullish", 0.2) is None
    assert ao.grade_direction("neutral", 9.0) is None
    assert ao.grade_direction("mixed", 9.0) is None
    assert ao.grade_direction("bullish", None) is None


def test_hit_rates_fan_out_to_every_named_source():
    rows = [
        {"mood": "bullish", "fwd_1d_pct": 5.0, "subreddits": ["stocks", "investing"]},
        {"mood": "bullish", "fwd_1d_pct": -5.0, "subreddits": ["stocks"]},
    ]
    stats = {r["key"]: r for r in ao.summarize_hit_rates(rows, "subreddit", min_scored=1)}
    assert stats["stocks"]["scored_1d"] == 2
    assert stats["stocks"]["correct_1d"] == 1
    assert stats["stocks"]["hit_rate_1d"] == 0.5
    assert stats["investing"]["hit_rate_1d"] == 1.0


def test_thin_samples_are_withheld():
    """A 1-for-1 author is not a 100% forecaster; publishing them as one is how
    a leaderboard becomes noise."""
    rows = [{"mood": "bullish", "fwd_1d_pct": 5.0, "authors": ["u_lucky"]}]
    assert ao.summarize_hit_rates(rows, "author", min_scored=5) == []
    assert len(ao.summarize_hit_rates(rows, "author", min_scored=1)) == 1


def test_attribution_accepts_json_strings_from_the_database():
    """The durable table stores JSON text, so the summarizer must handle both
    parsed lists and raw strings."""
    rows = [{"mood": "bearish", "fwd_1d_pct": -4.0, "subreddits": '["pennystocks"]'}] * 5
    stats = ao.summarize_hit_rates(rows, "subreddit", min_scored=5)
    assert stats[0]["key"] == "pennystocks"
    assert stats[0]["hit_rate_1d"] == 1.0


def test_malformed_attribution_does_not_crash_the_rollup():
    rows = [{"mood": "bullish", "fwd_1d_pct": 5.0, "subreddits": "not json"}]
    assert ao.summarize_hit_rates(rows, "subreddit", min_scored=1) == []


def test_resolve_fills_only_elapsed_horizons_and_never_overwrites():
    row = {
        "attention_date": date(2026, 8, 5),
        "ticker": "AAA",
        "mood": "bullish",
        "fwd_1d_pct": None,
        "fwd_5d_pct": None,
        "fwd_20d_pct": None,
    }
    out = ao.resolve_outcome_row(row, CLOSES, today=date(2026, 8, 13))
    assert out["fwd_1d_pct"] is not None
    assert out["correct_1d"] is True          # 100 -> 101 is +1.0%, at the flat threshold
    assert out["fwd_20d_pct"] is None         # 20 trading days have not elapsed
    assert row["fwd_1d_pct"] is None          # input not mutated

    # An already-filled horizon is left alone even if prices later change.
    already = dict(row, fwd_1d_pct=42.0)
    assert ao.resolve_outcome_row(already, CLOSES, today=date(2026, 8, 13))["fwd_1d_pct"] == 42.0


def test_resolve_accepts_an_iso_string_date():
    row = {"attention_date": "2026-08-05", "ticker": "AAA", "mood": "bullish"}
    out = ao.resolve_outcome_row(row, CLOSES, today=date(2026, 8, 13))
    assert out["fwd_1d_pct"] is not None

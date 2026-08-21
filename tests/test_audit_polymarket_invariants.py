"""These tests assert the audit CATCHES the bugs that actually shipped.

Each case is a real defect that reached production while 601 unit tests were
green, so the point is not that the checks are reasonable - it is that they
would have failed on the exact rows the pipeline really produced.
"""

import audit_polymarket_invariants as audit

KEYS = ("wallet", "events", "wins", "entry_avg")


def _run(row):
    report = audit.run_checks([row], audit.STATS_CHECKS, "t", KEYS)
    return {v["check"] for v in report["violations"]}


def _ok(**over):
    base = {"wallet": "0xa", "events": 10, "wins": 6, "pnl": 50.0, "cost": 100.0,
            "buy_size": 250.0, "entry_avg": 0.4, "win_entry_avg": 0.4,
            "recent_events": 5, "recent_wins": 3}
    base.update(over)
    return base


def test_a_clean_row_raises_nothing():
    assert _run(_ok()) == set()


def test_catches_the_entry_price_above_one_that_shipped():
    # Observed live: $3.40 on a contract that cannot trade above $1, caused by
    # dividing full cost by partial buy_size.
    assert "entry_avg_is_a_probability" in _run(_ok(entry_avg=3.40))


def test_catches_more_wins_than_events():
    # The shape the clamp bug trended toward: crediting wins a wallet never had.
    assert "wins_not_above_events" in _run(_ok(events=10, wins=11))


def test_catches_a_recent_window_larger_than_the_history():
    # Would fire if the trailing-window logic ever read across wallets.
    assert "recent_window_within_history" in _run(_ok(events=5, recent_events=9))


def test_catches_losing_more_than_was_staked():
    assert "roi_above_total_loss" in _run(_ok(cost=100.0, pnl=-250.0))


def test_catches_negative_sizes():
    assert "buy_size_not_negative" in _run(_ok(buy_size=-1))
    assert "cost_not_negative" in _run(_ok(cost=-5))


def test_missing_values_are_not_treated_as_violations():
    # Columns are absent for one deploy cycle before the Python ALTERs run;
    # that is expected, not a bug, and must not produce noise.
    assert _run(_ok(entry_avg=None, win_entry_avg=None)) == set()


def test_report_counts_every_violation_not_just_the_first():
    report = audit.run_checks([_ok(entry_avg=3.4, buy_size=-1)], audit.STATS_CHECKS, "t", KEYS)
    assert report["violation_count"] == 2
    assert report["rows_checked"] == 1

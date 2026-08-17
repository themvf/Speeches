"""Enhancement 3: attention alerts.

The board is a pull product - a ticker can spike, diverge from its price, and
get manipulation-flagged, and nobody finds out unless they open the tab.
Detection is pure, so all the behaviour is testable without a database.
"""

from datetime import date

import attention_alerts as aa

DAY = date(2026, 8, 16)


def _row(ticker, mentions=20, **kw):
    row = {"ticker": ticker, "total_mention_count": mentions, "quality_flags": "[]", "divergence": ""}
    row.update(kw)
    return row


def test_first_appearance_fires_for_a_genuinely_new_top_ticker():
    alerts = aa.detect_alerts(DAY, [_row("NEWCO")], known_tickers=["AAPL", "TSLA"])
    assert [a["alert_type"] for a in alerts] == [aa.ALERT_FIRST_APPEARANCE]
    assert alerts[0]["ticker"] == "NEWCO"
    assert alerts[0]["rank"] == 1


def test_first_appearance_is_suppressed_when_there_is_no_history():
    """Day one must not fire an alert for the entire board."""
    assert aa.detect_alerts(DAY, [_row("AAA"), _row("BBB")], known_tickers=[]) == []


def test_long_tail_tickers_below_the_rank_cut_are_ignored():
    rows = [_row(f"T{i}") for i in range(40)]
    alerts = aa.detect_alerts(DAY, rows, known_tickers=["SEED"], top_n=5)
    assert len(alerts) == 5
    assert all(a["rank"] <= 5 for a in alerts)


def test_surge_requires_both_a_multiple_and_a_floor():
    known = ["AAA", "BBB"]
    # 2 -> 6 is a 3x rise and means nothing.
    tiny = aa.detect_alerts(
        DAY, [_row("AAA", mentions=6)], prior_rows=[_row("AAA", mentions=2)], known_tickers=known
    )
    assert [a["alert_type"] for a in tiny] == []
    # 5 -> 30 clears both the multiple and the floor.
    real = aa.detect_alerts(
        DAY, [_row("AAA", mentions=30)], prior_rows=[_row("AAA", mentions=5)], known_tickers=known
    )
    assert [a["alert_type"] for a in real] == [aa.ALERT_MENTION_SURGE]


def test_a_ticker_with_no_yesterday_is_a_first_appearance_not_a_surge():
    """Emitting both would double-report the same event."""
    alerts = aa.detect_alerts(DAY, [_row("NEWCO", mentions=50)], prior_rows=[], known_tickers=["AAPL"])
    assert [a["alert_type"] for a in alerts] == [aa.ALERT_FIRST_APPEARANCE]


def test_divergence_and_quality_flags_each_fire():
    alerts = aa.detect_alerts(
        DAY,
        [_row("AAA", divergence="attention_spike_no_price_move", price_pct=0.2)],
        known_tickers=["AAA"],
    )
    assert [a["alert_type"] for a in alerts] == [aa.ALERT_DIVERGENCE]

    flagged = aa.detect_alerts(
        DAY, [_row("AAA", quality_flags='["single_thread_concentration"]')], known_tickers=["AAA"]
    )
    assert [a["alert_type"] for a in flagged] == [aa.ALERT_QUALITY_FLAG]


def test_malformed_quality_flags_do_not_crash_the_rollup():
    assert aa.detect_alerts(DAY, [_row("AAA", quality_flags="not json")], known_tickers=["AAA"]) == []
    assert aa.detect_alerts(DAY, [_row("AAA", quality_flags=None)], known_tickers=["AAA"]) == []


def test_one_ticker_can_raise_several_distinct_alerts():
    alerts = aa.detect_alerts(
        DAY,
        [_row("AAA", mentions=40, divergence="price_move_no_attention", quality_flags='["same_author_crew"]')],
        prior_rows=[_row("AAA", mentions=5)],
        known_tickers=["AAA"],
    )
    assert {a["alert_type"] for a in alerts} == {
        aa.ALERT_MENTION_SURGE, aa.ALERT_DIVERGENCE, aa.ALERT_QUALITY_FLAG
    }
    # Keys stay distinct, so all three survive the ON CONFLICT dedup.
    assert len({a["alert_key"] for a in alerts}) == 3


def test_alert_keys_are_stable_so_reruns_do_not_duplicate():
    args = ([_row("AAA", mentions=40)], [_row("AAA", mentions=5)], ["AAA"])
    first = aa.detect_alerts(DAY, args[0], prior_rows=args[1], known_tickers=args[2])
    second = aa.detect_alerts(DAY, args[0], prior_rows=args[1], known_tickers=args[2])
    assert [a["alert_key"] for a in first] == [a["alert_key"] for a in second]


def test_output_is_ordered_by_rarity_then_rank():
    rows = [
        _row("AAA", quality_flags='["same_author_crew"]'),
        _row("NEWCO"),
    ]
    alerts = aa.detect_alerts(DAY, rows, known_tickers=["AAA"])
    assert [a["alert_type"] for a in alerts] == [aa.ALERT_FIRST_APPEARANCE, aa.ALERT_QUALITY_FLAG]

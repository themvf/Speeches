"""Tests for the Polymarket pilot's pure scoring/classification (SEC-25/28).

Network-touching functions (fetch_*) are not exercised here; these cover the
settlement math and archetype classification, which are the parts that decide
who ranks and what badge they get.
"""

from __future__ import annotations

import polymarket_pilot as pilot


def _fill(wallet, outcome, side, size, price, name=""):
    return {"proxyWallet": wallet, "outcome": outcome, "side": side,
            "size": size, "price": price, "name": name}


def test_settle_market_pnl_for_winning_and_losing_buys():
    # Winner = "Yes". A bought 10 Yes @0.40 (wins -> +10 payout, -4 cash = +6).
    # B bought 10 No @0.30 (loses -> 0 payout, -3 cash = -3).
    fills = [
        _fill("A", "Yes", "BUY", 10, 0.40, "alice"),
        _fill("B", "No", "BUY", 10, 0.30, "bob"),
    ]
    out = pilot.settle_market(fills, "Yes")
    assert round(out["A"]["pnl"], 4) == 6.0
    assert round(out["B"]["pnl"], 4) == -3.0
    assert out["A"]["win_entry_avg"] == 0.40
    assert out["B"]["win_entry_avg"] is None  # B never bought the winning side


def test_settle_market_sell_realizes_cash_and_clamps_negative_net():
    # A buys 10 Yes @0.50 then sells 10 Yes @0.70: net 0 Yes, cash = -5 + 7 = +2.
    fills = [
        _fill("A", "Yes", "BUY", 10, 0.50),
        _fill("A", "Yes", "SELL", 10, 0.70),
    ]
    out = pilot.settle_market(fills, "Yes")
    assert round(out["A"]["pnl"], 4) == 2.0  # flat position, pure trading cash


def _row(markets=10, win_rate=0.6, pnl=100.0, roi=0.5, entry=0.5):
    return {"markets": markets, "win_rate": win_rate, "pnl_usd": pnl,
            "roi": roi, "avg_winner_entry_price": entry}


def test_classify_early_sharp():
    assert pilot.classify_archetype(_row(entry=0.55, win_rate=0.66, pnl=20000)) == "early_sharp"


def test_classify_news_scalper():
    assert pilot.classify_archetype(_row(entry=0.90, win_rate=1.0, pnl=14000)) == "news_scalper"


def test_classify_longshot():
    assert pilot.classify_archetype(_row(entry=0.13, win_rate=0.25, roi=2.6, pnl=3700)) == "longshot"


def test_classify_sample_size_gate():
    # Same shape as an early sharp but only 5 markets -> not badged.
    assert pilot.classify_archetype(_row(markets=5, entry=0.55, win_rate=0.66)) == "unclassified"


def test_classify_none_entry_is_unclassified():
    assert pilot.classify_archetype(_row(entry=None)) == "unclassified"


def test_classify_midband_is_unclassified():
    # entry 0.70, win 0.50: falls in the gap between every band.
    assert pilot.classify_archetype(_row(entry=0.70, win_rate=0.50)) == "unclassified"


def test_classify_bands_are_mutually_exclusive():
    # A wallet can never satisfy two archetypes: exercise the boundaries.
    early = _row(entry=0.60, win_rate=0.55, pnl=1)
    scalp = _row(entry=0.80, win_rate=0.90)
    assert pilot.classify_archetype(early) == "early_sharp"
    assert pilot.classify_archetype(scalp) == "news_scalper"
    # early's win-rate floor (0.55) excludes the longshot band (< 0.40).
    assert pilot.classify_archetype(_row(entry=0.30, win_rate=0.55, roi=2.0)) != "longshot"

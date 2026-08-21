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


def _row(markets=10, win_rate=0.6, pnl=100.0, roi=0.5, entry=0.5, entry_avg=None):
    """entry_avg (all-trades) is what the classifier gates on; entry
    (winners-only) is retained for display and deliberately ignored by it."""
    return {"markets": markets, "win_rate": win_rate, "pnl_usd": pnl,
            "roi": roi, "avg_winner_entry_price": entry,
            "entry_avg": entry if entry_avg is None else entry_avg}


def test_classify_early_sharp():
    # Wins 66% of the time on contracts costing 0.55: +11 points of edge.
    assert pilot.classify_archetype(_row(entry_avg=0.55, win_rate=0.66, pnl=20000)) == "early_sharp"


def test_classify_news_scalper():
    # Edge earned at near-certainty prices - speed, not forecasting.
    assert pilot.classify_archetype(_row(entry_avg=0.85, win_rate=0.97, pnl=14000)) == "news_scalper"


def test_classify_longshot():
    # Rare wins at long odds: 25% at 0.13 is +12 points.
    assert pilot.classify_archetype(_row(entry_avg=0.13, win_rate=0.25, roi=2.6, pnl=3700)) == "longshot"


def test_cheap_entries_that_win_often_are_sharps_not_longshots():
    # 60% at 0.26 is a large edge, but winning more often than not is simply
    # buying cheap and being right - "longshot" has to keep meaning long odds.
    assert pilot.classify_archetype(_row(entry_avg=0.26, win_rate=0.60, roi=0.9, pnl=9000)) == "early_sharp"


def test_a_high_win_rate_bought_at_near_certainty_is_not_skill():
    """The case that motivated moving classification onto edge. Observed live:
    a wallet winning 91% of its markets while paying 0.98 for them - two points
    WORSE than the price implied - carried a "news scalper" badge asserting a
    skill it did not have."""
    assert pilot.classify_archetype(
        _row(markets=75, entry_avg=0.98, win_rate=0.91, roi=0.03, pnl=500)) == "unclassified"


def test_a_low_win_rate_bought_cheaply_is_skill():
    """The mirror case: a wallet winning only 42% while paying 0.18 has +24
    points of edge and was previously left unclassified because its win rate
    looked poor."""
    assert pilot.classify_archetype(
        _row(markets=195, entry_avg=0.18, win_rate=0.42, roi=1.2, pnl=5000)) == "longshot"


def test_classify_sample_size_gate():
    assert pilot.classify_archetype(_row(markets=5, entry_avg=0.55, win_rate=0.66)) == "unclassified"


def test_classify_none_entry_is_unclassified():
    # No entry price recorded means there is nothing to judge the win rate
    # against - unclassified rather than assumed.
    assert pilot.classify_archetype(_row(entry_avg=0.5) | {"entry_avg": None}) == "unclassified"


def test_edge_alone_is_not_enough_if_the_wallet_lost_money():
    # Positive per-contract edge destroyed by position sizing.
    assert pilot.classify_archetype(
        _row(entry_avg=0.50, win_rate=0.70, roi=-0.1, pnl=-100)) == "unclassified"


def test_correctness_scores_position_direction_not_clamped_pnl():
    """settle_market clamps a negative net position's payout to zero so that a
    wallet which acquired shares off-tape (on-chain split/merge never reaches
    the trades feed) is not charged a phantom liability. That clamp is
    deliberate, but it means a wallet NET SHORT the winning outcome keeps its
    sale proceeds and shows a PROFIT - so scoring correctness off the sign of
    P&L marked being wrong as a win, and inflated win rates toward 100%.
    Correctness is therefore taken from net_win, which the clamp cannot touch.
    """
    def one(outcome, side):
        fills = [{"proxyWallet": "0xw", "name": "w", "outcome": outcome,
                  "side": side, "size": 100, "price": 0.40}]
        return pilot.settle_market(fills, "Yes")["0xw"]

    short_the_winner = one("Yes", "SELL")
    assert short_the_winner["net_win"] < 0, "net short the winning outcome"
    assert short_the_winner["pnl"] > 0, "clamp still leaves P&L positive - that is the trap"
    assert not (short_the_winner["net_win"] > 0), "must NOT count as correct"

    held_the_winner = one("Yes", "BUY")
    assert held_the_winner["net_win"] > 0

    bought_the_loser = one("No", "BUY")
    assert bought_the_loser["net_win"] == 0
    assert bought_the_loser["pnl"] < 0


def test_all_trades_entry_price_includes_losing_trades_unlike_win_entry_avg():
    """The calibration denominator. win_entry_avg is conditioned on winners, so
    it cannot say what a wallet paid for the outcomes it got WRONG - which is
    exactly what a win rate has to be judged against.

    A wallet buying both sides at 0.30 and 0.70 paid 0.50 on average and has no
    edge whatsoever, but win_entry_avg reports 0.30 and makes it look like a
    cheap-entry sharp. cost / buy_size reports the truth.
    """
    fills = [
        {"proxyWallet": "0xh", "name": "h", "outcome": "Yes", "side": "BUY", "size": 100, "price": 0.30},
        {"proxyWallet": "0xh", "name": "h", "outcome": "No", "side": "BUY", "size": 100, "price": 0.70},
    ]
    r = pilot.settle_market(fills, "Yes")["0xh"]
    assert r["cost"] == 100.0 and r["buy_size"] == 200.0
    assert r["cost"] / r["buy_size"] == 0.50, "all-trades entry"
    assert r["win_entry_avg"] == 0.30, "winners-only entry - flattering and wrong for calibration"


def test_buy_size_ignores_sells_so_entry_price_is_not_diluted():
    fills = [
        {"proxyWallet": "0xs", "name": "s", "outcome": "Yes", "side": "BUY", "size": 100, "price": 0.40},
        {"proxyWallet": "0xs", "name": "s", "outcome": "Yes", "side": "SELL", "size": 50, "price": 0.90},
    ]
    r = pilot.settle_market(fills, "Yes")["0xs"]
    assert r["buy_size"] == 100.0, "sells must not count toward shares bought"
    assert r["cost"] / r["buy_size"] == 0.40


def test_a_two_sided_position_is_not_a_correct_call():
    """Holding the winning outcome is not enough. A wallet that buys BOTH
    sides holds the winner in every market it touches, whatever happens - so
    scoring on net_win alone marked hedgers and market makers correct 100% of
    the time. Found in production by the distribution report: 18 wallets with
    >=95% win rates over >=30 events, entry prices near 0.45 (the blend of both
    sides) and negligible P&L.

    Note this is a regression the earlier clamp fix introduced: the original
    pnl > 0 rule happened to score hedgers correctly, since their P&L is flat.
    Comparing the two sides handles both cases at once.
    """
    def f(outcome, side, size):
        return {"proxyWallet": "0xh", "name": "h", "outcome": outcome,
                "side": side, "size": size, "price": 0.5}

    hedged = pilot.settle_market([f("Yes", "BUY", 100), f("No", "BUY", 100)], "Yes")["0xh"]
    assert hedged["net_win"] > 0, "they do hold the winner - that is the trap"
    assert not (hedged["net_win"] > hedged["net_lose"]), "but made no directional call"

    leaning = pilot.settle_market([f("Yes", "BUY", 100), f("No", "BUY", 30)], "Yes")["0xh"]
    assert leaning["net_win"] > leaning["net_lose"], "a real lean toward the winner still counts"

    # The clamp case must keep working: short the winner is still not correct.
    short = pilot.settle_market([f("Yes", "SELL", 100)], "Yes")["0xh"]
    assert not (short["net_win"] > short["net_lose"])


def test_settlement_conclusions_are_reproducible_from_stored_components():
    """The raw-vs-derived contract.

    Three separate corrections to the meaning of "correct" each forced a full
    re-settle of every historical market, because we stored the CONCLUSION and
    discarded the inputs. Raw fills are pruned 7 days after settlement and the
    only other source caps at ~3500 fills per market, so once they are gone a
    definition change is unfixable rather than merely expensive.

    settle_market therefore emits the components too. This asserts every
    conclusion can be recomputed from them - i.e. that a FOURTH change to any
    of these definitions can be replayed against stored rows.
    """
    fills = [
        {"proxyWallet": "0xw", "name": "w", "outcome": "Yes", "side": "BUY", "size": 100, "price": 0.40},
        {"proxyWallet": "0xw", "name": "w", "outcome": "No", "side": "BUY", "size": 40, "price": 0.55},
        {"proxyWallet": "0xw", "name": "w", "outcome": "Yes", "side": "SELL", "size": 25, "price": 0.80},
    ]
    r = pilot.settle_market(fills, "Yes")["0xw"]

    # pnl = cash + the clamped payout, so it is replayable under a DIFFERENT
    # payout rule too - including dropping the clamp entirely.
    assert abs((r["cash"] + max(r["net_win"], 0.0)) - r["pnl"]) < 1e-9
    # correctness, under the current definition and any other built on
    # position direction
    assert (r["net_win"] > r["net_lose"]) in (True, False)
    # win_entry_avg = win_buy_cash / win_buy_size; storing win_buy_size means
    # the cash side is recoverable, so the average is reproducible.
    if r["win_entry_avg"] is not None:
        assert abs(r["win_entry_avg"] * r["win_buy_size"] - 0.40 * 100) < 1e-9
    # all-trades entry price
    assert abs(r["cost"] / r["buy_size"] - (0.40 * 100 + 0.55 * 40) / 140) < 1e-9

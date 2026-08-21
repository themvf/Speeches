from datetime import date

import wallet_trajectory as traj


def r(day, correct, pnl=0.0, cost=100.0):
    return {"resolved_date": date(2026, 1, day), "correct": correct, "pnl": pnl, "cost": cost}


def test_recent_form_reads_the_tail_not_the_lifetime_average():
    # 25 losses then 25 wins: the lifetime record is 50/50 and hides the fact
    # that everything recent is a win. This is the exact case that made every
    # order-blind aggregate insufficient.
    results = [r(1, False, pnl=-100) for _ in range(25)] + [r(2, True, pnl=100) for _ in range(25)]
    form = traj.recent_form(results, window=10)
    assert form["recent_events"] == 10
    assert form["recent_win_rate"] == 1.0
    assert form["recent_roi"] == 1.0


def test_recent_form_is_order_sensitive_not_just_a_subset():
    wins_last = [r(1, False, pnl=-100), r(2, False, pnl=-100), r(3, True, pnl=100)]
    wins_first = [r(1, True, pnl=100), r(2, False, pnl=-100), r(3, False, pnl=-100)]
    assert traj.recent_form(wins_last, window=1)["recent_win_rate"] == 1.0
    assert traj.recent_form(wins_first, window=1)["recent_win_rate"] == 0.0


def test_recent_form_handles_an_empty_history_without_dividing_by_zero():
    form = traj.recent_form([])
    assert form["recent_events"] == 0
    assert form["recent_roi"] is None and form["recent_win_rate"] is None


def test_undated_rows_are_kept_rather_than_silently_dropped():
    results = [r(1, True, pnl=50), {"resolved_date": None, "correct": True, "pnl": 50, "cost": 100}]
    assert traj.recent_form(results)["recent_events"] == 2


def test_loss_chasing_flags_bigger_bets_after_losses():
    # Alternating loss->big, win->small: the martingale shape.
    results = []
    for day in range(1, 11):
        loss = day % 2 == 1
        results.append(r(day, not loss, pnl=0.0, cost=1000.0 if not loss else 100.0))
    chase = traj.loss_chasing(results)
    assert chase["chases_losses"] is True
    assert chase["chase_ratio"] and chase["chase_ratio"] >= traj.CHASE_SIZE_RATIO


def test_steady_sizing_is_not_flagged_as_chasing():
    results = [r(day, day % 2 == 0, cost=500.0) for day in range(1, 11)]
    assert traj.loss_chasing(results)["chases_losses"] is False


def test_growing_size_after_WINS_is_not_flagged_as_chasing():
    # The opposite pattern - scaling up a run - must not trip the loss-chasing
    # flag, since the two shapes are only separable by what precedes them.
    results = []
    for day in range(1, 11):
        results.append(r(day, True, cost=100.0 * day))
    assert traj.loss_chasing(results)["chases_losses"] is False


def test_loss_chasing_needs_enough_post_loss_observations():
    results = [r(1, False, cost=100), r(2, True, cost=1000)]
    assert traj.loss_chasing(results)["chases_losses"] is False
    assert traj.loss_chasing(results)["chase_ratio"] is None


def test_watchlist_never_overrides_a_proven_verdict():
    form = {"recent_events": 10, "recent_roi": 0.9}
    assert traj.watchlist_status(True, 0.5, form) == "proven"


def test_developing_requires_beating_its_own_lifetime_record():
    strong = {"recent_events": 10, "recent_roi": 0.60}
    # Recent better than lifetime -> the case a lifetime average would bury.
    assert traj.watchlist_status(False, 0.10, strong) == "developing"
    # Recent good but no better than lifetime -> merely consistent, not developing.
    assert traj.watchlist_status(False, 0.80, strong) == "watching"


def test_thin_or_unprofitable_recent_windows_do_not_reach_the_watchlist():
    assert traj.watchlist_status(False, 0.0, {"recent_events": 2, "recent_roi": 5.0}) == "none"
    assert traj.watchlist_status(False, 0.0, {"recent_events": 10, "recent_roi": 0.01}) == "none"


def test_summarize_combines_form_chasing_and_status():
    results = [r(day, True, pnl=100, cost=100) for day in range(1, 11)]
    out = traj.summarize(results, qualified=False, lifetime_roi=0.1)
    assert out["watchlist_status"] == "developing"
    assert out["recent_events"] == 10
    assert "chases_losses" in out

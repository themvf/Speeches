#!/usr/bin/env python3
"""Order-aware wallet metrics: recent form, and loss-chasing detection.

Every other statistic in this pipeline is a lifetime aggregate, which is blind
to ORDER - a 50/50 record could be alternating results, or 25 losses followed
by 25 wins. Those are completely different traders and nothing downstream could
tell them apart. resolved_date was already stored on both durable results
tables; it simply was never read back.

Two deliberate constraints, from the design review that produced this module:

1. Nothing here decides whether a wallet is SKILLED. Recent form feeds a
   permissive watchlist ("worth noticing"), never the conservative verdict
   ("proven"). The two carry opposite error costs - missing a developing
   trader forfeits the whole opportunity, wrongly watching one costs a row on
   a list - so they must not share a threshold.

2. Nothing here reports a DIRECTION. At these sample sizes, regression to the
   mean (anyone who started badly is expected to look better next, with no
   learning at all), market-wide regime shifts, and survivorship among wallets
   that kept trading all counterfeit "improvement". Recent form is reported as
   a measurement over a narrower window, never as a trend or a trajectory.

Pure functions only - no database, no network.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

# Trailing window, in events rather than days: many wallets trade rarely, and a
# fixed time window would empty out for exactly the patient traders worth
# watching.
RECENT_WINDOW_EVENTS = 10

# Watchlist gates, deliberately permissive (see constraint 1 above).
WATCHLIST_MIN_RECENT_EVENTS = 5
WATCHLIST_MIN_RECENT_ROI = 0.10

# Loss-chasing: how much bigger the average position after a loss has to be
# than after a win before it is worth flagging, and how many post-loss
# observations are needed before the ratio means anything.
CHASE_SIZE_RATIO = 1.5
CHASE_MIN_OBSERVATIONS = 4


def _ordered(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Oldest first. Rows without a resolved_date sort last rather than being
    dropped: they are real observations, just unusable for ordering."""
    dated = [r for r in results if r.get("resolved_date") is not None]
    undated = [r for r in results if r.get("resolved_date") is None]
    return sorted(dated, key=lambda r: r["resolved_date"]) + undated


def recent_form(results: Sequence[Dict[str, Any]], window: int = RECENT_WINDOW_EVENTS) -> Dict[str, Any]:
    """Same arithmetic as the lifetime rollup, over the last `window` events.

    Reported alongside the lifetime figure, never instead of it: shortening the
    window trades bias for variance, so a recent-form number carries a wider
    error bar than the lifetime one it sits next to.
    """
    ordered = _ordered(results)[-window:]
    events = len(ordered)
    if not events:
        return {"recent_events": 0, "recent_wins": 0, "recent_win_rate": None,
                "recent_pnl": 0.0, "recent_cost": 0.0, "recent_roi": None}
    wins = sum(1 for r in ordered if r.get("correct"))
    pnl = sum(float(r.get("pnl") or 0) for r in ordered)
    cost = sum(float(r.get("cost") or 0) for r in ordered)
    return {
        "recent_events": events,
        "recent_wins": wins,
        "recent_win_rate": round(wins / events, 4),
        "recent_pnl": round(pnl, 4),
        "recent_cost": round(cost, 4),
        "recent_roi": round(pnl / cost, 4) if cost > 0 else None,
    }


def loss_chasing(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Does this wallet bet BIGGER after losing?

    A rising position-size curve has two opposite readings: a trader scaling up
    a validated edge, and a trader chasing losses. Identical shape. They are
    separable only by conditioning on what PRECEDED the increase, which is why
    raw size growth is not reported as a confidence signal anywhere.

    The loss-chasing side is the more defensible output of the two: a
    doubling-down trader looks indistinguishable from a skilled one right up
    until the run ends, so it flags something a reader cannot infer unaided.
    """
    ordered = _ordered(results)
    after_loss: List[float] = []
    after_win: List[float] = []
    for previous, current in zip(ordered, ordered[1:]):
        cost = float(current.get("cost") or 0)
        if cost <= 0:
            continue
        (after_win if previous.get("correct") else after_loss).append(cost)
    if len(after_loss) < CHASE_MIN_OBSERVATIONS or not after_win:
        return {"chases_losses": False, "chase_ratio": None,
                "after_loss_avg": None, "after_win_avg": None}
    loss_avg = sum(after_loss) / len(after_loss)
    win_avg = sum(after_win) / len(after_win)
    ratio = (loss_avg / win_avg) if win_avg > 0 else None
    return {
        "chases_losses": bool(ratio is not None and ratio >= CHASE_SIZE_RATIO),
        "chase_ratio": round(ratio, 3) if ratio is not None else None,
        "after_loss_avg": round(loss_avg, 2),
        "after_win_avg": round(win_avg, 2),
    }


def watchlist_status(qualified: bool, lifetime_roi: Optional[float], form: Dict[str, Any]) -> str:
    """'proven' | 'developing' | 'watching' | 'none'.

    'developing' is explicitly NOT a skill claim and must never be presented as
    one - it means "the recent window looks better than the lifetime record,
    on a sample too small to conclude anything". It exists so a wallet whose
    early history drags its lifetime average cannot be permanently buried by
    data that no longer describes how it trades.
    """
    if qualified:
        return "proven"
    recent_roi = form.get("recent_roi")
    if form.get("recent_events", 0) < WATCHLIST_MIN_RECENT_EVENTS or recent_roi is None:
        return "none"
    if recent_roi < WATCHLIST_MIN_RECENT_ROI:
        return "none"
    # Better than its own lifetime record is what makes it "developing" rather
    # than merely "doing fine" - the case the lifetime average would hide.
    if lifetime_roi is not None and recent_roi > lifetime_roi:
        return "developing"
    return "watching"


def summarize(results: Sequence[Dict[str, Any]], qualified: bool,
              lifetime_roi: Optional[float]) -> Dict[str, Any]:
    """Everything order-aware for one wallet (or wallet-cohort pair)."""
    form = recent_form(results)
    return {**form, **loss_chasing(results),
            "watchlist_status": watchlist_status(qualified, lifetime_roi, form)}

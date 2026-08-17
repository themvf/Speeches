"""Attention alerts (enhancement 3) - pure detection.

The attention board is a pull product: a ticker can spike, diverge from its
price, and get flagged for manipulation, and nobody finds out unless they
happen to open the tab that day. This turns the interesting transitions into
events.

Detection is pure and deterministic - no model calls, no I/O. It runs inside
the daily rollup, where today's rows, yesterday's rows, and the set of tickers
ever seen are all already in hand, so it costs one extra pass over data the
job has already fetched.

Shape follows polymarket_sharp_alerts (SEC-29), which solved the same problem:
a content-hash dedup key with ON CONFLICT DO NOTHING, age-pruning rather than
settle-then-prune, and a lazily-fetched read route. Alerts are a display
concern, not a durable results source - attention_outcomes is where the
durable record lives - so losing old ones costs nothing.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

# A ticker must clear this rank to be worth alerting on at all. Without it,
# every long-tail ticker with two mentions generates a first-appearance alert
# every single day and the feed is unreadable.
DEFAULT_TOP_N = 25

# Surge needs both a multiple and a floor: 2 mentions -> 6 is a 3x rise and
# means nothing, so the floor is what stops noise from dominating.
DEFAULT_SURGE_MULTIPLE = 3.0
DEFAULT_SURGE_MIN_MENTIONS = 8

ALERT_FIRST_APPEARANCE = "first_appearance"
ALERT_MENTION_SURGE = "mention_surge"
ALERT_DIVERGENCE = "divergence"
ALERT_QUALITY_FLAG = "quality_flag"

# Ordering for display and for deterministic test assertions: rarity first.
ALERT_PRIORITY = {
    ALERT_FIRST_APPEARANCE: 0,
    ALERT_MENTION_SURGE: 1,
    ALERT_DIVERGENCE: 2,
    ALERT_QUALITY_FLAG: 3,
}


def alert_key(attention_date: Any, ticker: str, alert_type: str) -> str:
    """Dedup key. One alert per (day, ticker, type): a rollup re-run with
    --date must not duplicate what it already emitted, and the daily job is
    idempotent by design."""
    return f"{attention_date}:{ticker}:{alert_type}"


def _mentions(row: Dict[str, Any]) -> int:
    return int(row.get("total_mention_count") or row.get("mention_count") or 0)


def _parse_flags(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return []
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
    return []


def detect_alerts(
    attention_date: Any,
    today_rows: Sequence[Dict[str, Any]],
    prior_rows: Optional[Sequence[Dict[str, Any]]] = None,
    known_tickers: Optional[Iterable[str]] = None,
    top_n: int = DEFAULT_TOP_N,
    surge_multiple: float = DEFAULT_SURGE_MULTIPLE,
    surge_min_mentions: int = DEFAULT_SURGE_MIN_MENTIONS,
) -> List[Dict[str, Any]]:
    """Alert rows for one day's rollup.

    `today_rows` is expected pre-sorted by rank the way the rollup emits it;
    rank is taken from position rather than recomputed, so an alert's rank
    always matches what the board shows.

    `known_tickers` is every ticker that has appeared in any prior rollup. An
    empty set means no history, in which case first-appearance is suppressed
    entirely rather than firing for the whole board on day one.
    """
    prior_by_ticker = {str(r["ticker"]): r for r in (prior_rows or [])}
    known: Set[str] = {str(t) for t in (known_tickers or [])}
    have_history = bool(known)

    out: List[Dict[str, Any]] = []
    for index, row in enumerate(today_rows):
        ticker = str(row["ticker"])
        rank = index + 1
        in_top = rank <= top_n
        mentions = _mentions(row)

        if in_top and have_history and ticker not in known:
            out.append({
                "alert_key": alert_key(attention_date, ticker, ALERT_FIRST_APPEARANCE),
                "attention_date": attention_date,
                "ticker": ticker,
                "alert_type": ALERT_FIRST_APPEARANCE,
                "rank": rank,
                "detail": json.dumps({"mentions": mentions, "rank": rank}),
            })

        prior = prior_by_ticker.get(ticker)
        prior_mentions = _mentions(prior) if prior else 0
        # Requires prior_mentions > 0: a ticker with no yesterday is a
        # first appearance, not a surge, and emitting both would double-report
        # the same event.
        if (
            in_top
            and prior_mentions > 0
            and mentions >= surge_min_mentions
            and mentions >= prior_mentions * surge_multiple
        ):
            out.append({
                "alert_key": alert_key(attention_date, ticker, ALERT_MENTION_SURGE),
                "attention_date": attention_date,
                "ticker": ticker,
                "alert_type": ALERT_MENTION_SURGE,
                "rank": rank,
                "detail": json.dumps({
                    "mentions": mentions,
                    "prior_mentions": prior_mentions,
                    "multiple": round(mentions / prior_mentions, 2),
                }),
            })

        divergence = str(row.get("divergence") or "").strip()
        if in_top and divergence:
            out.append({
                "alert_key": alert_key(attention_date, ticker, ALERT_DIVERGENCE),
                "attention_date": attention_date,
                "ticker": ticker,
                "alert_type": ALERT_DIVERGENCE,
                "rank": rank,
                "detail": json.dumps({
                    "divergence": divergence,
                    "price_pct": row.get("price_pct"),
                    "mentions": mentions,
                }),
            })

        flags = _parse_flags(row.get("quality_flags"))
        if in_top and flags:
            out.append({
                "alert_key": alert_key(attention_date, ticker, ALERT_QUALITY_FLAG),
                "attention_date": attention_date,
                "ticker": ticker,
                "alert_type": ALERT_QUALITY_FLAG,
                "rank": rank,
                "detail": json.dumps({"flags": flags, "mentions": mentions}),
            })

    out.sort(key=lambda a: (ALERT_PRIORITY.get(a["alert_type"], 9), a["rank"], a["ticker"]))
    return out

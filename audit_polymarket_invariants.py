#!/usr/bin/env python3
"""Bounds check on every derived Polymarket figure we store.

601 unit tests passed while the site served win rates of 100% that were
arithmetically impossible, and an entry price of $3.40 on a contract that
cannot trade above $1. Unit tests could not catch either: each asserts that
the code does what its fixture says, and the fixture was written alongside the
code holding the same wrong assumption.

These assertions are different in kind. They do not encode intent - they encode
facts about the quantities themselves. A probability is in [0,1]. Wins cannot
exceed events. A recent window cannot be longer than the whole history. A
violation is never a legitimate edge case, so it is always a bug, and it is
catchable the moment it is written rather than when someone notices it on a
dashboard weeks later.

Read-only. Exits non-zero when anything is violated so a scheduled run fails
loudly rather than reporting into a log nobody reads.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import neon_feeds

SOURCE_KEY = "polymarket_invariants"

# (label, predicate, description). Predicate returns True when the row is FINE.
Check = Tuple[str, Callable[[Dict[str, Any]], bool], str]


def _num(row: Dict[str, Any], key: str) -> Optional[float]:
    value = row.get(key)
    return None if value is None else float(value)


def _prob_in_range(key: str) -> Callable[[Dict[str, Any]], bool]:
    def check(row: Dict[str, Any]) -> bool:
        value = _num(row, key)
        return value is None or 0.0 <= value <= 1.0
    return check


STATS_CHECKS: List[Check] = [
    ("entry_avg_is_a_probability", _prob_in_range("entry_avg"),
     "average entry price must be in [0,1] - contracts settle at 0 or 1"),
    ("win_entry_avg_is_a_probability", _prob_in_range("win_entry_avg"),
     "winners-only entry price must be in [0,1]"),
    ("wins_not_above_events",
     lambda r: (r.get("wins") or 0) <= (r.get("events") or r.get("markets") or 0),
     "a wallet cannot win more markets than it settled"),
    ("counts_not_negative",
     lambda r: (r.get("wins") or 0) >= 0 and (r.get("events") or r.get("markets") or 0) >= 0,
     "event and win counts must be non-negative"),
    ("recent_window_within_history",
     lambda r: (r.get("recent_events") or 0) <= (r.get("events") or r.get("markets") or 0),
     "the recent window cannot contain more events than the full history"),
    ("recent_wins_within_recent_events",
     lambda r: (r.get("recent_wins") or 0) <= (r.get("recent_events") or 0),
     "recent wins cannot exceed recent events"),
    ("roi_above_total_loss",
     lambda r: (_num(r, "cost") or 0) <= 0 or (_num(r, "pnl") or 0) / (_num(r, "cost") or 1) >= -1.0001,
     "you cannot lose more than you staked"),
    ("buy_size_not_negative",
     lambda r: (_num(r, "buy_size") or 0) >= 0, "shares bought must be non-negative"),
    ("cost_not_negative",
     lambda r: (_num(r, "cost") or 0) >= 0, "cost must be non-negative"),
]


def run_checks(rows: List[Dict[str, Any]], checks: List[Check], label: str,
               key_fields: Tuple[str, ...]) -> Dict[str, Any]:
    violations: List[Dict[str, Any]] = []
    for row in rows:
        for name, predicate, description in checks:
            try:
                ok = predicate(row)
            except Exception as exc:
                ok = False
                description = f"{description} (check raised {exc})"
            if not ok:
                violations.append({
                    "check": name, "why": description,
                    "row": {k: row.get(k) for k in key_fields},
                })
    return {"table": label, "rows_checked": len(rows),
            "violations": violations[:50], "violation_count": len(violations)}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    summary: Dict[str, Any] = {"source_key": SOURCE_KEY, "ran_at": datetime.now(UTC).isoformat(),
                               "errors": [], "reports": []}
    try:
        macro_rows = neon_feeds.get_polymarket_macro_wallet_stats_rows()
        summary["reports"].append(run_checks(
            macro_rows, STATS_CHECKS, "polymarket_macro_wallet_stats",
            ("wallet", "cohort", "events", "wins", "entry_avg", "buy_size", "cost", "pnl")))
        earnings_rows = neon_feeds.get_polymarket_wallet_stats_rows()
        summary["reports"].append(run_checks(
            earnings_rows, STATS_CHECKS, "polymarket_wallet_stats",
            ("wallet", "markets", "wins", "entry_avg", "buy_size", "cost", "pnl")))
        total = sum(r["violation_count"] for r in summary["reports"])
        summary["total_violations"] = total
        summary["ok"] = total == 0
    except Exception as exc:
        summary["errors"].append(str(exc))
        summary["ok"] = False
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Plausibility report on the wallet population.

The bounds audit catches values that are IMPOSSIBLE. It cannot catch values
that are merely absurd: a 100% win rate over 159 markets sits comfortably
inside [0,1] and shipped for weeks. Catching that class needs a view of the
whole distribution rather than one row at a time.

The strongest single check here is the population's mean edge. Polymarket is a
liquid market with fees; across thousands of wallets the average edge should
sit at or slightly below zero, because the average trader cannot beat the price
they pay. If OUR measured population shows a large positive mean, the finding
is not that we discovered thousands of skilled traders - it is that our
measurement is biased. The known candidate is tape truncation: the trades
endpoint stops paging at ~3500 fills, so on busy markets we see only the most
recent ones, and late trades cluster at prices where the outcome is already
decided.

Reports rather than fails. Unlike a bound, a shifted distribution needs
judgement - it can move for real reasons - so this opens the question instead
of answering it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

import neon_feeds

SOURCE_KEY = "polymarket_distributions"

# A high win rate is NOT by itself suspicious: buying at 0.999 and winning
# every time is a real strategy earning a tenth of a cent, and several wallets
# do exactly that. What cannot be explained is a large EDGE sustained over a
# large sample - beating the price you paid, repeatedly, in a liquid market.
# The first version of this check flagged win rate alone and buried the one
# genuine case among a dozen legitimate chalk buyers.
IMPLAUSIBLE_EDGE = 0.20
IMPLAUSIBLE_MIN_EVENTS = 30
# Mean edge beyond this is treated as a measurement-bias signal rather than
# a population of unusually good traders.
POPULATION_EDGE_TOLERANCE = 0.05


def _histogram(values: List[float], width: float = 0.1) -> Dict[str, int]:
    buckets: Counter = Counter()
    for value in values:
        low = min(int(value / width) * width, 1.0 - width) if value >= 0 else -width
        buckets[f"{low:.1f}-{low + width:.1f}"] += 1
    return dict(sorted(buckets.items()))


def _describe(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"n": 0, "mean": None, "median": None}
    return {"n": len(values), "mean": round(statistics.fmean(values), 4),
            "median": round(statistics.median(values), 4)}


def analyze(rows: List[Dict[str, Any]], events_key: str) -> Dict[str, Any]:
    win_rates: List[float] = []
    edges: List[float] = []
    entries: List[float] = []
    implausible: List[Dict[str, Any]] = []
    weighted_edge_num = 0.0
    weighted_edge_den = 0.0

    for row in rows:
        events = int(row.get(events_key) or 0)
        if events <= 0:
            continue
        win_rate = float(row.get("wins") or 0) / events
        win_rates.append(win_rate)
        entry = row.get("entry_avg")
        if entry is not None:
            entry = float(entry)
            entries.append(entry)
            edge = win_rate - entry
            edges.append(edge)
            # Weighted by sample so one lucky single-market wallet cannot move
            # the population figure.
            weighted_edge_num += edge * events
            weighted_edge_den += events
        if (entry is not None and events >= IMPLAUSIBLE_MIN_EVENTS
                and (win_rate - entry) >= IMPLAUSIBLE_EDGE):
            implausible.append({"wallet": row.get("wallet"), "cohort": row.get("cohort"),
                                "events": events, "win_rate": round(win_rate, 4),
                                "entry_avg": round(entry, 4),
                                "edge": round(win_rate - entry, 4)})

    weighted_edge = (weighted_edge_num / weighted_edge_den) if weighted_edge_den else None
    return {
        "wallets": len(win_rates),
        "win_rate": {**_describe(win_rates), "histogram": _histogram(win_rates)},
        "entry_avg": _describe(entries),
        "edge": {**_describe(edges), "sample_weighted_mean": round(weighted_edge, 4) if weighted_edge is not None else None},
        "implausible_edges": {
            "threshold": f"edge >=+{IMPLAUSIBLE_EDGE:.0%} over >={IMPLAUSIBLE_MIN_EVENTS} events",
            "count": len(implausible),
            "sample": sorted(implausible, key=lambda r: -r["edge"])[:10],
        },
        "population_edge_suspicious": (
            weighted_edge is not None and weighted_edge > POPULATION_EDGE_TOLERANCE),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    summary: Dict[str, Any] = {"source_key": SOURCE_KEY, "ran_at": datetime.now(UTC).isoformat(),
                               "errors": []}
    try:
        summary["earnings"] = analyze(neon_feeds.get_polymarket_wallet_stats_rows(), "markets")
        summary["macro"] = analyze(neon_feeds.get_polymarket_macro_wallet_stats_rows(), "events")
        summary["ok"] = True
    except Exception as exc:
        summary["errors"].append(str(exc))
        summary["ok"] = False
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

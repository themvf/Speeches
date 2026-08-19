#!/usr/bin/env python3
"""Read-only diagnostic: is classify_wallet's threshold scheme actually
separating the real macro wallet population, or is a chunk of it falling
into "unclassified" for a reason that isn't "not enough sample"?

classify_wallet's named archetypes only cover three narrow bands of
predictive_share (< 0.25 for release_scalper, >= 0.60 for early_sharp, plus
longshot's separate entry/win-rate/ROI band). Everything from 0.25-0.60
predictive_share falls through to "unclassified" unless it happens to also
match longshot - indistinguishable in the UI from a wallet that genuinely
doesn't have enough history yet. This script buckets every wallet-cohort
pair (regardless of whether it meets the cohort's event-count bar) into a
named "band" so that dead zone is visible and measurable instead of assumed.

Never writes to the database. DATABASE_URL is the sole secret (same as
polymarket_macro_sync.py).
"""

from __future__ import annotations

import json
import statistics
import sys
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

import neon_feeds
import polymarket_macro_sync as macro

SOURCE_KEY = "analyze_macro_archetype_bands"

# Bands, in the order classify_wallet itself checks them, plus two diagnostic
# bands classify_wallet collapses into a single "unclassified" verdict.
BAND_INSUFFICIENT_EVENTS = "insufficient_events"
BAND_LOW_TIMING_COVERAGE = "low_timing_coverage"
BAND_RELEASE_SCALPER = "release_scalper"
BAND_EARLY_SHARP = "early_sharp"
BAND_LONGSHOT = "longshot"
BAND_DEAD_ZONE = "dead_zone"  # enough sample + timing coverage, matches no named archetype


def band_for(events: int, wins: int, pnl: float, cost: float, predictive_cost: float,
             timing_cost: float, entry: Optional[float], cohort: str) -> str:
    """Same decision tree as classify_wallet, but with the two implicit
    "unclassified" reasons (not enough sample; not enough pre-release timing
    coverage) split out from the "sample is fine, just doesn't match a named
    archetype" case, so the three can be told apart."""
    if events < macro.cohort_min_events(cohort):
        return BAND_INSUFFICIENT_EVENTS
    if cost <= 0 or timing_cost / cost < 0.5:
        return BAND_LOW_TIMING_COVERAGE
    win_rate = wins / events
    roi = pnl / cost
    predictive_share = predictive_cost / timing_cost if timing_cost > 0 else 0.0
    if predictive_share < 0.25 and win_rate >= 0.60:
        return BAND_RELEASE_SCALPER
    if predictive_share >= 0.60 and win_rate >= 0.55 and pnl > 0:
        return BAND_EARLY_SHARP
    if entry is not None and entry <= 0.35 and win_rate < 0.40 and roi > 1:
        return BAND_LONGSHOT
    return BAND_DEAD_ZONE


def _stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p25": None, "p75": None}
    ordered = sorted(values)
    quantiles = statistics.quantiles(ordered, n=4) if len(ordered) >= 2 else [ordered[0], ordered[0], ordered[0]]
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "p25": round(quantiles[0], 4),
        "p75": round(quantiles[-1], 4),
    }


def analyze(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """rows: the same shape group_macro_wallet_results() produces (one row per
    wallet-cohort pair, entry already averaged). Excludes macro_generalist
    rows - the dead-zone question is about per-cohort classification, and
    generalist rows are a derived cross-cohort rollup, not a cohort archetype."""
    per_cohort: Dict[str, Dict[str, Any]] = {}
    dead_zone_win_rates: List[float] = []
    dead_zone_rois: List[float] = []
    dead_zone_predictive_shares: List[float] = []
    dead_zone_pnls: List[float] = []
    overall_bands: Dict[str, int] = {}
    total_pairs = 0

    for row in rows:
        cohort = row["cohort"]
        if cohort == "macro_generalist":
            continue
        total_pairs += 1
        events, wins, pnl, cost = row["events"], row["wins"], row["pnl"], row["cost"]
        predictive_cost, timing_cost = row["predictive_cost"], row["timing_cost"]
        entry = row.get("win_entry_avg")
        band = band_for(events, wins, pnl, cost, predictive_cost, timing_cost, entry, cohort)

        overall_bands[band] = overall_bands.get(band, 0) + 1
        cohort_bucket = per_cohort.setdefault(cohort, {})
        cohort_bucket[band] = cohort_bucket.get(band, 0) + 1

        if band == BAND_DEAD_ZONE:
            dead_zone_win_rates.append(wins / events if events else 0.0)
            dead_zone_rois.append(pnl / cost if cost > 0 else 0.0)
            if timing_cost > 0:
                dead_zone_predictive_shares.append(predictive_cost / timing_cost)
            dead_zone_pnls.append(pnl)

    return {
        "total_wallet_cohort_pairs": total_pairs,
        "overall_bands": overall_bands,
        "per_cohort_bands": per_cohort,
        "dead_zone_profile": {
            "win_rate": _stats(dead_zone_win_rates),
            "roi": _stats(dead_zone_rois),
            "predictive_share": _stats(dead_zone_predictive_shares),
            "pnl": _stats(dead_zone_pnls),
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    summary: Dict[str, Any] = {"source_key": SOURCE_KEY, "ran_at": datetime.now(UTC).isoformat(), "errors": []}
    try:
        results = neon_feeds.get_polymarket_macro_wallet_results()
        rows = macro.group_macro_wallet_results(results, include_unqualified_generalist=True)
        summary["report"] = analyze(rows)
        summary["ok"] = True
    except Exception as exc:  # pragma: no cover - defensive, same pattern as polymarket_macro_sync.main
        summary["errors"].append(str(exc))
        summary["ok"] = False
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

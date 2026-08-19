#!/usr/bin/env python3
"""Recurring US macro Polymarket ingestion and cohort-specific wallet scoring.

Tracks repeatable release families: FOMC decisions, nonfarm payrolls,
unemployment, headline CPI, core CPI, US GDP, core PCE, ISM Manufacturing
PMI, ISM Services PMI, and PPI. All brackets belonging to one release are
collapsed to one wallet observation. Entry cost is bucketed relative to the
scheduled release so post-print scalpers never become macro sharps. Public
Polymarket APIs only; DATABASE_URL is the sole secret.

Note: weekly initial jobless claims was evaluated and deliberately excluded
(SEC wallet-intelligence OKR, 2026-08-18) - Polymarket ran a "How many
jobless claims during the week ending X?" bracket series only from Feb-Mar
2026 before discontinuing it (verified live against gamma-api.polymarket.com;
zero events of any jobless-claims title format exist after 2026-03-05), so
there is currently nothing live to track for that cohort.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import neon_feeds
import polymarket_earnings_sync as earnings
import polymarket_pilot as pilot
from source_health import record_source_health

SOURCE_KEY = "polymarket_macro_sync"
# Polymarket has drifted its tag taxonomy since this was first built (2026-07-16):
# "Unemployment Rate" events dropped the "nfp" tag entirely around May 2026 (now tagged
# "unemployment" instead), and "How many jobs added" events inconsistently carry "nfp"
# month to month (now tagged "jobs-report"). "nfp" is kept for older/back-compat events;
# "jobs-report"/"unemployment" are the tags actually carried by current live events
# (verified live against gamma-api.polymarket.com 2026-08-17).
DISCOVERY_TAGS = ("fed", "cpi-release", "nfp", "jobs-report", "unemployment", "gdp", "pce", "ism", "ppi")
# Default assumes ~monthly-or-better cadence: 10 events seasons in <=15 months
# even for fed_decision's ~8/year rate. Quarterly cohorts get their own,
# lower bar (see COHORT_MIN_EVENTS_OVERRIDES) so they season on a comparable
# timeline instead of needing 2.5 years to hit the shared default.
COHORT_MIN_EVENTS = 10
COHORT_MIN_EVENTS_OVERRIDES = {
    "us_gdp": 5,  # quarterly: 5 releases = 15 months, matching every other cohort's bar
}
GENERALIST_MIN_EVENTS = 20
GENERALIST_MIN_COHORTS = 3


def cohort_min_events(cohort: str) -> int:
    return COHORT_MIN_EVENTS_OVERRIDES.get(cohort, COHORT_MIN_EVENTS)
MONTHS = {name.lower(): i for i, name in enumerate(
    ("January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"), 1)}

COHORT_META = {
    "fed_decision": ("Fed Sharp", "~8 releases/year"),
    "nonfarm_payrolls": ("Payrolls Sharp", "monthly"),
    "unemployment": ("Unemployment Sharp", "monthly"),
    "headline_cpi": ("Headline CPI Sharp", "monthly"),
    "core_cpi": ("Core CPI Sharp", "monthly"),
    "us_gdp": ("GDP Sharp", "quarterly"),
    "core_pce": ("Core PCE Sharp", "monthly"),
    "ism_manufacturing": ("ISM Manufacturing Sharp", "monthly"),
    "ism_services": ("ISM Services Sharp", "monthly"),
    "ppi": ("PPI Sharp", "monthly"),
    "macro_generalist": ("Macro Generalist", "cross-cohort"),
}
EASTERN = ZoneInfo("America/New_York")


def parse_dt(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value or "").strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except ValueError:
        return None


def _period_key(cohort: str, title: str, release_at: Optional[datetime]) -> Optional[str]:
    year_match = re.search(r"\b(20\d{2})\b", title)
    year = int(year_match.group(1)) if year_match else (release_at.year if release_at else None)
    month_match = re.search(r"\b(" + "|".join(m.title() for m in MONTHS) + r")\b", title, re.I)
    if month_match and year:
        month = MONTHS[month_match.group(1).lower()]
        if not year_match and release_at and month > release_at.month + 6:
            year -= 1
        return f"{cohort}:{year:04d}-{month:02d}"
    quarter = re.search(r"\bQ([1-4])\s+(20\d{2})\b", title, re.I)
    if quarter:
        return f"{cohort}:{quarter.group(2)}-Q{quarter.group(1)}"
    return None


def classify_macro_event(title: str, release_at: Optional[datetime]) -> Optional[Tuple[str, str]]:
    normalized = " ".join(title.split())
    cohort: Optional[str] = None
    if re.match(r"^Fed Decision in .+\?$", normalized, re.I):
        cohort = "fed_decision"
    elif re.match(r"^How many jobs added in .+\?$", normalized, re.I):
        cohort = "nonfarm_payrolls"
    elif re.match(r"^(?:January|February|March|April|May|June|July|August|September|October|November|December) Unemployment Rate$", normalized, re.I):
        cohort = "unemployment"
    elif re.match(r"^(?:January|February|March|April|May|June|July|August|September|October|November|December) Inflation(?: US)? - (?:Monthly|Annual)$", normalized, re.I):
        cohort = "headline_cpi"
    elif re.match(r"^Core CPI (?:MoM|YoY) - .+$", normalized, re.I):
        cohort = "core_cpi"
    elif re.match(r"^US GDP growth in Q[1-4] 20\d{2}\??$", normalized, re.I):
        cohort = "us_gdp"
    elif re.match(r"^Core PCE (?:MoM|YoY) - .+$", normalized, re.I):
        cohort = "core_pce"
    elif re.match(r"^ISM Manufacturing PMI - .+$", normalized, re.I):
        cohort = "ism_manufacturing"
    elif re.match(r"^ISM Services PMI - .+$", normalized, re.I):
        cohort = "ism_services"
    elif re.match(r"^(?:Producer Price Index \(PPI\)|PPI) YoY - .+$", normalized, re.I):
        cohort = "ppi"
    if not cohort:
        return None
    event_key = _period_key(cohort, normalized, release_at)
    return (cohort, event_key) if event_key else None


_RELEASE_TIME_ET = {
    "fed_decision": (14, 0),  # FOMC statement
    "ism_manufacturing": (10, 0),  # ISM Report on Business
    "ism_services": (10, 0),  # ISM Report on Business
}


def scheduled_release_at(cohort: str, value: Optional[datetime]) -> Optional[datetime]:
    """Gamma endDate is generally date-only midnight. Normalize it to the
    official US release time so entry buckets do not label same-day,
    pre-release trading as post-release. Default (8:30am ET) covers the BLS/BEA
    releases (payrolls, unemployment, CPI, PCE, PPI); ISM's own Report on
    Business goes out at 10:00am ET, not 8:30, so it needs its own entry."""
    if not value:
        return None
    hour, minute = _RELEASE_TIME_ET.get(cohort, (8, 30))
    local = datetime(value.year, value.month, value.day, hour, minute, tzinfo=EASTERN)
    return local.astimezone(UTC)


def _winning_outcome(market: Dict[str, Any]) -> Optional[str]:
    try:
        outcomes = json.loads(market.get("outcomes") or "[]")
        prices = [float(value) for value in json.loads(market.get("outcomePrices") or "[]")]
    except (TypeError, ValueError):
        return None
    winners = [outcome for outcome, price in zip(outcomes, prices) if price > 0.99]
    return str(winners[0]) if len(winners) == 1 else None


def fetch_events(closed: bool, pages: int) -> List[Dict[str, Any]]:
    unique: Dict[str, Dict[str, Any]] = {}
    for tag in DISCOVERY_TAGS:
        for page in range(pages):
            events = pilot._get(f"{pilot.GAMMA}/events", {
                "tag_slug": tag, "closed": str(closed).lower(), "order": "volume",
                "ascending": "false", "limit": 100, "offset": page * 100,
            })
            for event in events or []:
                release_at = parse_dt(event.get("endDate"))
                classified = classify_macro_event(str(event.get("title") or ""), release_at)
                if classified:
                    unique[str(event.get("id") or event.get("slug"))] = event
            if len(events or []) < 100:
                break
    return list(unique.values())


def event_market_rows(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for event in events:
        release_at = parse_dt(event.get("endDate"))
        classified = classify_macro_event(str(event.get("title") or ""), release_at)
        if not classified:
            continue
        cohort, event_key = classified
        release_at = scheduled_release_at(cohort, release_at)
        for market in event.get("markets") or []:
            condition_id = str(market.get("conditionId") or "")
            if not condition_id:
                continue
            try:
                outcomes = json.loads(market.get("outcomes") or "[]")
                prices = [float(value) for value in json.loads(market.get("outcomePrices") or "[]")]
                yes_index = next((i for i, value in enumerate(outcomes) if str(value).lower() == "yes"), None)
                yes_price = prices[yes_index] if yes_index is not None else None
            except (TypeError, ValueError):
                yes_price = None
            rows.append({
                "condition_id": condition_id, "question": str(market.get("question") or event.get("title") or ""),
                "slug": str(event.get("slug") or ""), "end_date": release_at,
                "volume": float(market.get("volume") or 0), "implied_prob_yes": yes_price,
                "market_type": "macro", "cohort": cohort, "event_key": event_key,
                "event_title": str(event.get("title") or ""), "release_at": release_at,
                "winner": _winning_outcome(market),
            })
    return rows


def _fill_time(fill: Dict[str, Any]) -> Optional[datetime]:
    if fill.get("filled_at") is not None:
        return parse_dt(fill["filled_at"])
    try:
        return datetime.fromtimestamp(float(fill.get("timestamp") or 0), tz=UTC)
    except (TypeError, ValueError, OSError):
        return None


def aggregate_release(markets: List[Dict[str, Any]], fills_by_market: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """Combine every bracket/market in a release into one wallet result."""
    aggregate: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "pnl": 0.0, "cost": 0.0, "early_cost": 0.0, "pre_release_cost": 0.0,
        "late_cost": 0.0, "post_release_cost": 0.0, "winner_cash": 0.0,
        "winner_size": 0.0, "name": "",
    })
    for market in markets:
        fills = fills_by_market.get(market["condition_id"], [])
        settled = pilot.settle_market(fills, str(market["winner"]))
        release_at = parse_dt(market.get("release_at"))
        for wallet, result in settled.items():
            row = aggregate[wallet]
            row["pnl"] += float(result.get("pnl") or 0)
            row["cost"] += float(result.get("cost") or 0)
            row["name"] = result.get("name") or row["name"]
        for fill in fills:
            if str(fill.get("side") or "").upper() != "BUY" or not release_at:
                continue
            wallet = str(fill.get("proxyWallet") or "")
            if not wallet:
                continue
            try:
                cost = float(fill.get("size") or 0) * float(fill.get("price") or 0)
            except (TypeError, ValueError):
                continue
            filled_at = _fill_time(fill)
            if not filled_at:
                continue
            hours = (release_at - filled_at).total_seconds() / 3600
            bucket = "early_cost" if hours > 24 else "pre_release_cost" if hours > 1 else "late_cost" if hours >= 0 else "post_release_cost"
            aggregate[wallet][bucket] += cost
            if str(fill.get("outcome") or "") == str(market["winner"]):
                aggregate[wallet]["winner_cash"] += cost
                aggregate[wallet]["winner_size"] += float(fill.get("size") or 0)
    for row in aggregate.values():
        size = row.pop("winner_size")
        cash = row.pop("winner_cash")
        row["win_entry_avg"] = cash / size if size > 0 else None
    return dict(aggregate)


def classify_wallet(events: int, wins: int, pnl: float, cost: float, predictive_cost: float, timing_cost: float, entry: Optional[float], cohort: str = "") -> str:
    if events < cohort_min_events(cohort) or cost <= 0 or timing_cost / cost < 0.5:
        return "unclassified"
    win_rate, roi, predictive_share = wins / events, pnl / cost, predictive_cost / timing_cost
    if predictive_share < 0.25 and win_rate >= 0.60:
        return "release_scalper"
    if predictive_share >= 0.60 and win_rate >= 0.55 and pnl > 0:
        return "early_sharp"
    if entry is not None and entry <= 0.35 and win_rate < 0.40 and roi > 1:
        return "longshot"
    return "unclassified"


def group_macro_wallet_results(results: List[Dict[str, Any]], include_unqualified_generalist: bool = False) -> List[Dict[str, Any]]:
    """Aggregate raw per-event settlement rows into one row per (wallet, cohort),
    plus a synthetic "macro_generalist" row per wallet across cohorts. Shared by
    the write path (recompute_wallet_stats) and read-only diagnostics (e.g.
    analyze_macro_archetype_bands.py) so the two can never drift apart on what
    "a wallet's stats for a cohort" means.

    include_unqualified_generalist=True skips the GENERALIST_MIN_EVENTS/
    GENERALIST_MIN_COHORTS gate on generalist rows - diagnostics want to see
    the full population including wallets that don't (yet) qualify as
    generalists, not just the ones that already do.
    """
    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    cohorts_by_wallet: Dict[str, set] = defaultdict(set)
    for result in results:
        wallet, cohort = str(result["wallet"]), str(result["cohort"])
        cohorts_by_wallet[wallet].add(cohort)
        for key in ((wallet, cohort), (wallet, "macro_generalist")):
            row = groups.setdefault(key, {"events": 0, "wins": 0, "pnl": 0.0, "cost": 0.0,
                "predictive_cost": 0.0, "timing_cost": 0.0, "entries": [], "name": ""})
            row["events"] += 1; row["wins"] += int(bool(result["correct"]))
            row["pnl"] += float(result["pnl"] or 0); row["cost"] += float(result["cost"] or 0)
            predictive = float(result["early_cost"] or 0) + float(result["pre_release_cost"] or 0)
            timing = predictive + float(result["late_cost"] or 0) + float(result["post_release_cost"] or 0)
            row["predictive_cost"] += predictive; row["timing_cost"] += timing
            if result["win_entry_avg"] is not None: row["entries"].append(float(result["win_entry_avg"]))
            if result["name"]: row["name"] = str(result["name"])
    rows = []
    for (wallet, cohort), value in groups.items():
        if (cohort == "macro_generalist" and not include_unqualified_generalist
                and (value["events"] < GENERALIST_MIN_EVENTS or len(cohorts_by_wallet[wallet]) < GENERALIST_MIN_COHORTS)):
            continue
        entry = sum(value["entries"]) / len(value["entries"]) if value["entries"] else None
        archetype = classify_wallet(value["events"], value["wins"], value["pnl"], value["cost"], value["predictive_cost"], value["timing_cost"], entry, cohort)
        rows.append({"wallet": wallet, "cohort": cohort, "name": value["name"], **{k: value[k] for k in ("events", "wins", "pnl", "cost", "predictive_cost", "timing_cost")}, "win_entry_avg": entry, "archetype": archetype})
    return rows


def recompute_wallet_stats() -> Dict[str, int]:
    results = neon_feeds.get_polymarket_macro_wallet_results()
    rows = group_macro_wallet_results(results)
    return {"wallets": neon_feeds.upsert_polymarket_macro_wallet_stats(rows), "results": len(results)}


def settle_groups(markets: List[Dict[str, Any]], fetch_stored: bool) -> int:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for market in markets:
        grouped[str(market["event_key"])].append(market)
    settled_count = 0
    for event_key, event_markets in grouped.items():
        if not event_key or not all(m.get("winner") for m in event_markets):
            continue
        fills = {}
        for market in event_markets:
            fills[market["condition_id"]] = (neon_feeds.get_polymarket_market_fills(market["condition_id"])
                if fetch_stored else pilot.fetch_market_fills(market["condition_id"]))
        combined = aggregate_release(event_markets, fills)
        release_at = parse_dt(event_markets[0].get("release_at"))
        neon_feeds.save_polymarket_macro_event_settlement(event_key, event_markets[0]["cohort"], release_at.date() if release_at else None, combined)
        settled_count += 1
    return settled_count


def run_sync(summary: Dict[str, Any]) -> None:
    active_rows = event_market_rows(fetch_events(False, 1))
    closed_rows = event_market_rows(fetch_events(True, 2))
    summary["active_markets"] = neon_feeds.upsert_polymarket_markets(active_rows)
    neon_feeds.upsert_polymarket_markets(closed_rows)
    for row in closed_rows:
        if row["winner"]:
            neon_feeds.mark_polymarket_resolved(row["condition_id"], row["winner"])
    tracked = neon_feeds.get_polymarket_tracked_markets("macro")
    fills_written = 0
    for market in tracked:
        if market["settled_at"] is None:
            fills_written += neon_feeds.insert_polymarket_fills(earnings.fetch_new_fills(market["condition_id"], market["fill_cursor"]))
    summary["fills_written"] = fills_written
    unsettled_resolved = [market for market in neon_feeds.get_polymarket_tracked_markets("macro") if market["status"] == "resolved" and market["settled_at"] is None]
    summary["settled_releases"] = settle_groups(unsettled_resolved, True)
    summary["stats"] = recompute_wallet_stats()
    summary["fills_pruned"] = neon_feeds.prune_settled_polymarket_fills(earnings.PRUNE_DAYS_AFTER_SETTLEMENT)


def run_backfill(max_releases: int, summary: Dict[str, Any]) -> None:
    rows = event_market_rows(fetch_events(True, 5))
    grouped = defaultdict(list)
    for row in rows: grouped[row["event_key"]].append(row)
    selected_keys = sorted(grouped, key=lambda key: -sum(m["volume"] for m in grouped[key]))[:max_releases]
    selected = [market for key in selected_keys for market in grouped[key]]
    neon_feeds.upsert_polymarket_markets(selected)
    for row in selected:
        if row["winner"]: neon_feeds.mark_polymarket_resolved(row["condition_id"], row["winner"])
    summary["backfill_releases"] = settle_groups(selected, False)
    summary["stats"] = recompute_wallet_stats()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--backfill-releases", type=int, default=120)
    args = parser.parse_args(argv)
    summary: Dict[str, Any] = {"source_key": SOURCE_KEY, "connector": SOURCE_KEY, "mode": "backfill" if args.backfill else "sync", "errors": [], "ran_at": datetime.now(UTC).isoformat()}
    try:
        run_backfill(args.backfill_releases, summary) if args.backfill else run_sync(summary)
        summary["ok"] = True
    except Exception as exc:
        summary["errors"].append(str(exc)); summary["ok"] = False
    summary["processed_count"] = summary.get("fills_written", 0) + summary.get("settled_releases", 0) + summary.get("backfill_releases", 0)
    summary["discovered_count"] = summary.get("active_markets", 0)
    summary["failed_count"] = len(summary["errors"])
    record_source_health(summary)
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

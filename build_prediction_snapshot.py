#!/usr/bin/env python3
"""SEC-28 (static first cut): bake the Prediction Markets tab dataset from the
Polymarket earnings pilot analysis, mirroring the CBOE tab's
static-snapshot-ahead-of-live-pipeline pattern (SEC-19). Writes
apps/web/lib/server/prediction-markets-data.json.

The live Neon-backed pipeline (SEC-26 ingestion + SEC-27 scoring) will later
replace this file as the API route's data source without changing the
response contract.

Two views are assembled:
  - wallets:  the earnings-market leaderboard with archetype badges.
  - calendar: currently-open earnings markets paired with Polymarket's implied
              beat probability and a "sharp money" consensus that counts only
              early_sharp / longshot wallets (news_scalper positions are shown
              on the wallet but deliberately excluded from consensus - they're
              post-print reactions, not predictions).

Read-only against public no-auth endpoints. Research context only.

Usage:
    python build_prediction_snapshot.py [--max-markets 120] [--min-markets 8]
        [--leaderboard 60] [--consensus-pool 40]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import polymarket_pilot as pilot

OUT_PATH = os.path.join("apps", "web", "lib", "server", "prediction-markets-data.json")

# Parse the EPS threshold + report date baked into the event slug, e.g.
# "usb-quarterly-earnings-gaap-eps-07-16-2026-1pt27" -> 1.27 on 2026-07-16.
_SLUG_EPS_RE = re.compile(r"-(\d{2})-(\d{2})-(\d{4})-(\d+)pt(\d+)$")


def _parse_slug(slug: str) -> Dict[str, Optional[str]]:
    match = _SLUG_EPS_RE.search(str(slug or ""))
    if not match:
        return {"report_date": None, "eps": None}
    mm, dd, yyyy, whole, frac = match.groups()
    return {"report_date": f"{yyyy}-{mm}-{dd}", "eps": f"{whole}.{frac}"}


def fetch_open_earnings_detailed() -> Dict[str, Dict[str, Any]]:
    """condition_id -> {question, ticker, implied_prob_yes, report_date, eps,
    end_date, volume}. Implied prob for an open market is just the current
    Yes price."""
    out: Dict[str, Dict[str, Any]] = {}
    events = pilot._get(f"{pilot.GAMMA}/events", {
        "tag_slug": "earnings", "closed": "false", "limit": 100,
        "order": "volume", "ascending": "false",
    })
    for event in events or []:
        for market in event.get("markets") or []:
            condition_id = market.get("conditionId")
            if not condition_id:
                continue
            try:
                outcomes = json.loads(market.get("outcomes") or "[]")
                prices = [float(p) for p in json.loads(market.get("outcomePrices") or "[]")]
            except (ValueError, TypeError):
                outcomes, prices = [], []
            implied_yes = None
            for outcome, price in zip(outcomes, prices):
                if str(outcome).lower() == "yes":
                    implied_yes = round(price, 3)
            question = str(market.get("question") or event.get("title") or "")
            ticker_match = pilot._TICKER_RE.search(question)
            slug_info = _parse_slug(market.get("slug") or event.get("slug") or "")
            out[condition_id] = {
                "question": question,
                "ticker": ticker_match.group(1) if ticker_match else "",
                "implied_prob_yes": implied_yes,
                "report_date": slug_info["report_date"] or (str(event.get("endDate") or "")[:10] or None),
                "eps": slug_info["eps"],
                "end_date": str(event.get("endDate") or ""),
                "volume": round(float(market.get("volume") or 0), 2),
            }
    return out


def _net_side(net_shares: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """Reduce a wallet's per-outcome net holdings to a single dominant side."""
    yes = net_shares.get("Yes", 0.0)
    no = net_shares.get("No", 0.0)
    if yes <= 0.5 and no <= 0.5:
        return None
    if yes >= no:
        return {"side": "Yes", "shares": round(yes, 1)}
    return {"side": "No", "shares": round(no, 1)}


def build(max_markets: int, min_markets: int, leaderboard_n: int, consensus_pool: int) -> Dict[str, Any]:
    markets = pilot.fetch_resolved_earnings_markets(max_markets)
    print(f"resolved earnings markets: {len(markets)}", file=sys.stderr)
    wallet_stats, total_fills = pilot.analyze_resolved_markets(markets)
    ranked = pilot.rank_wallets(wallet_stats, min_markets)
    print(f"qualified wallets (>= {min_markets} markets): {len(ranked)}", file=sys.stderr)

    open_detailed = fetch_open_earnings_detailed()
    # fetch_wallet_open_positions only needs {question, ticker} per market.
    open_lite = {cid: {"question": m["question"], "ticker": m["ticker"]} for cid, m in open_detailed.items()}
    print(f"open earnings markets: {len(open_detailed)}", file=sys.stderr)

    # Consensus pool: top non-scalper wallets by PnL. Their live positions form
    # each market's sharp-money consensus. Scalpers are excluded here by
    # construction (their fills are post-news reactions, not predictions).
    pool = [r for r in ranked if r["archetype"] in ("early_sharp", "longshot")][:consensus_pool]
    print(f"fetching open positions for {len(pool)} consensus wallets...", file=sys.stderr)

    consensus: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"yes": 0, "no": 0, "wallets": []})
    positions_by_wallet: Dict[str, List[Dict[str, Any]]] = {}
    # Also grab positions for any leaderboard wallet not already in the pool,
    # so the wallet drawer can show holdings for scalpers too (shown, not
    # counted). Bounded to the leaderboard we actually render.
    leaderboard = ranked[:leaderboard_n]
    fetch_targets = {r["wallet"]: r for r in leaderboard}
    for r in pool:
        fetch_targets.setdefault(r["wallet"], r)

    for wallet, row in fetch_targets.items():
        raw = _fetch_wallet_positions_detailed(wallet, open_lite)
        positions_by_wallet[wallet] = raw
        # Only pool wallets contribute to consensus.
        if row["archetype"] not in ("early_sharp", "longshot"):
            continue
        for pos in raw:
            side = pos.get("side")
            if not side:
                continue
            bucket = consensus[pos["condition_id"]]
            if side == "Yes":
                bucket["yes"] += 1
            else:
                bucket["no"] += 1
            bucket["wallets"].append({
                "name": row["name"] or (wallet[:8] + "…"),
                "wallet": wallet,
                "archetype": row["archetype"],
                "side": side,
                "shares": pos["shares"],
            })

    calendar = []
    for condition_id, market in sorted(open_detailed.items(), key=lambda kv: -(kv[1]["volume"] or 0)):
        c = consensus.get(condition_id)
        calendar.append({
            "conditionId": condition_id,
            "ticker": market["ticker"],
            "question": market["question"],
            "reportDate": market["report_date"],
            "eps": market["eps"],
            "impliedProbYes": market["implied_prob_yes"],
            "volume": market["volume"],
            "consensus": {
                "yes": c["yes"] if c else 0,
                "no": c["no"] if c else 0,
                "wallets": sorted(c["wallets"], key=lambda w: -w["shares"]) if c else [],
            },
        })
    # Markets with tracked sharp positions first, then by volume.
    calendar.sort(key=lambda row: (-(row["consensus"]["yes"] + row["consensus"]["no"]), -(row["volume"] or 0)))

    wallets_out = []
    for r in leaderboard:
        positions = positions_by_wallet.get(r["wallet"], [])
        wallets_out.append({
            "wallet": r["wallet"],
            "name": r["name"] or (r["wallet"][:8] + "…"),
            "archetype": r["archetype"],
            "markets": r["markets"],
            "wins": r["wins"],
            "winRate": r["win_rate"],
            "pnlUsd": r["pnl_usd"],
            "roi": r["roi"],
            "avgWinnerEntry": r["avg_winner_entry_price"],
            "openPositions": [
                {"ticker": p["ticker"], "question": p["question"], "side": p["side"], "shares": p["shares"]}
                for p in positions if p.get("side")
            ],
        })

    return {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "Polymarket public data API (earnings markets) — SEC-25 pilot analysis",
        "marketsAnalyzed": len(markets),
        "fillsProcessed": total_fills,
        "walletsQualified": len(ranked),
        "openMarketCount": len(open_detailed),
        "archMinMarkets": pilot.ARCH_MIN_MARKETS,
        "consensusPoolSize": len(pool),
        "calendar": calendar,
        "wallets": wallets_out,
    }


def _fetch_wallet_positions_detailed(wallet: str, open_lite: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
    """Like pilot.fetch_wallet_open_positions but keeps condition_id and a
    reduced dominant side for consensus assembly."""
    try:
        fills = pilot._get(f"{pilot.DATA}/trades", {"user": wallet, "limit": 500, "takerOnly": "false"})
    except Exception:
        return []
    stance: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for fill in fills or []:
        condition_id = str(fill.get("conditionId") or "")
        if condition_id not in open_lite:
            continue
        outcome = str(fill.get("outcome") or "")
        side = str(fill.get("side") or "").upper()
        try:
            size = float(fill.get("size") or 0)
        except (TypeError, ValueError):
            continue
        stance[condition_id][outcome] += size if side == "BUY" else -size
    out = []
    for condition_id, outcomes in stance.items():
        reduced = _net_side(dict(outcomes))
        if reduced:
            out.append({
                "condition_id": condition_id,
                "ticker": open_lite[condition_id]["ticker"],
                "question": open_lite[condition_id]["question"],
                "side": reduced["side"],
                "shares": reduced["shares"],
            })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-markets", type=int, default=120)
    parser.add_argument("--min-markets", type=int, default=pilot.ARCH_MIN_MARKETS)
    parser.add_argument("--leaderboard", type=int, default=60)
    parser.add_argument("--consensus-pool", type=int, default=40)
    args = parser.parse_args()

    snapshot = build(args.max_markets, args.min_markets, args.leaderboard, args.consensus_pool)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=1)

    counts = defaultdict(int)
    for w in snapshot["wallets"]:
        counts[w["archetype"]] += 1
    print(f"\nwrote {OUT_PATH}")
    print(f"  leaderboard: {len(snapshot['wallets'])} wallets {dict(counts)}")
    print(f"  calendar: {len(snapshot['calendar'])} open markets, "
          f"{sum(1 for c in snapshot['calendar'] if c['consensus']['wallets'])} with sharp positions")
    return 0


if __name__ == "__main__":
    sys.exit(main())

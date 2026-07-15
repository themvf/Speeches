#!/usr/bin/env python3
"""SEC-25 pilot: are certain traders persistently better at Polymarket's
per-company earnings markets ("Will X beat quarterly earnings?")?

Backfills the highest-volume RESOLVED earnings events (gamma-api), pulls each
market's full fill tape (data-api, takerOnly=false so maker sides are
included), reconstructs per-wallet per-market P&L against the resolved
outcome, and ranks wallets with a minimum-markets guard. Then reports what
the top wallets currently hold in OPEN earnings markets (the actionable
part: this week's earnings calendar).

Read-only against public, no-auth endpoints. No DB, no UI - kpi_pilot.py
pattern. Research context only - not investment advice.

Usage:
    python polymarket_pilot.py [--max-markets 120] [--min-markets 5]
        [--out polymarket_pilot_output.json]

P&L method (per wallet per market): net position and cash flow per outcome
from fills (BUY: +size, -size*price cash; SELL: -size, +size*price cash);
at resolution the winning outcome's tokens redeem at $1, losers at $0, so
pnl = cash + max(net_winner, 0). Negative net positions (possible via
on-chain split/merge, which the fill tape doesn't show) are clamped to 0
payout - a small, documented approximation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import requests

GAMMA = "https://gamma-api.polymarket.com"
DATA = "https://data-api.polymarket.com"
THROTTLE_S = 0.15
TRADES_PAGE = 500
MAX_TRADES_PER_MARKET = 6000
EVENT_PAGES = 3  # x100 events/page

_TICKER_RE = re.compile(r"\(([A-Z][A-Z0-9.]{0,6})\)")

session = requests.Session()
session.headers["User-Agent"] = "PolicyResearchHub-pilot/1.0 (research)"


def _get(url: str, params: Dict[str, Any]) -> Any:
    time.sleep(THROTTLE_S)
    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_resolved_earnings_markets(max_markets: int) -> List[Dict[str, Any]]:
    """Top-volume resolved earnings markets: conditionId + winning outcome."""
    markets: List[Dict[str, Any]] = []
    for page in range(EVENT_PAGES):
        events = _get(f"{GAMMA}/events", {
            "tag_slug": "earnings", "closed": "true", "order": "volume",
            "ascending": "false", "limit": 100, "offset": page * 100,
        })
        if not events:
            break
        for event in events:
            for market in event.get("markets") or []:
                condition_id = market.get("conditionId")
                try:
                    outcomes = json.loads(market.get("outcomes") or "[]")
                    prices = [float(p) for p in json.loads(market.get("outcomePrices") or "[]")]
                except (ValueError, TypeError):
                    continue
                if not condition_id or len(outcomes) != len(prices) or not prices:
                    continue
                # Resolved = one outcome priced at (effectively) $1.
                winners = [o for o, p in zip(outcomes, prices) if p > 0.99]
                if len(winners) != 1:
                    continue
                question = str(market.get("question") or event.get("title") or "")
                ticker_match = _TICKER_RE.search(question)
                markets.append({
                    "condition_id": condition_id,
                    "question": question,
                    "ticker": ticker_match.group(1) if ticker_match else "",
                    "winner": winners[0],
                    "volume": float(market.get("volume") or 0),
                    "end_date": str(event.get("endDate") or ""),
                })
    markets.sort(key=lambda m: -m["volume"])
    return markets[:max_markets]


def fetch_market_fills(condition_id: str) -> List[Dict[str, Any]]:
    """Newest-first fill tape. The data API 400s past offset ~3500, so the
    very largest markets yield only their most recent ~3.5k fills - keep the
    partial tape rather than discarding the market (documented caveat: for
    mega-markets the earliest fills are unreachable via this endpoint)."""
    fills: List[Dict[str, Any]] = []
    offset = 0
    while offset < MAX_TRADES_PER_MARKET:
        try:
            page = _get(f"{DATA}/trades", {
                "market": condition_id, "limit": TRADES_PAGE, "offset": offset,
                "takerOnly": "false",
            })
        except requests.HTTPError:
            break  # offset ceiling reached - return what we have
        if not isinstance(page, list) or not page:
            break
        fills.extend(page)
        if len(page) < TRADES_PAGE:
            break
        offset += TRADES_PAGE
    return fills


def settle_market(fills: List[Dict[str, Any]], winner: str) -> Dict[str, Dict[str, float]]:
    """Per-wallet P&L for one resolved market. Returns wallet -> stats."""
    positions: Dict[str, Dict[str, float]] = defaultdict(lambda: {"cash": 0.0, "cost": 0.0, "net_win": 0.0, "win_buy_size": 0.0, "win_buy_cash": 0.0})
    names: Dict[str, str] = {}
    for fill in fills:
        wallet = str(fill.get("proxyWallet") or "")
        outcome = str(fill.get("outcome") or "")
        side = str(fill.get("side") or "").upper()
        try:
            size = float(fill.get("size") or 0)
            price = float(fill.get("price") or 0)
        except (TypeError, ValueError):
            continue
        if not wallet or size <= 0 or side not in ("BUY", "SELL"):
            continue
        pos = positions[wallet]
        name = str(fill.get("name") or fill.get("pseudonym") or "")
        if name:
            names[wallet] = name
        signed = size if side == "BUY" else -size
        cash = -size * price if side == "BUY" else size * price
        pos["cash"] += cash
        if side == "BUY":
            pos["cost"] += size * price
        if outcome == winner:
            pos["net_win"] += signed
            if side == "BUY":
                pos["win_buy_size"] += size
                pos["win_buy_cash"] += size * price
    out: Dict[str, Dict[str, float]] = {}
    for wallet, pos in positions.items():
        payout = max(pos["net_win"], 0.0)
        out[wallet] = {
            "pnl": pos["cash"] + payout,
            "cost": pos["cost"],
            "win_entry_avg": (pos["win_buy_cash"] / pos["win_buy_size"]) if pos["win_buy_size"] > 0 else None,
            "name": names.get(wallet, ""),
        }
    return out


def fetch_open_earnings_markets() -> Dict[str, Dict[str, str]]:
    """condition_id -> {question, ticker} for currently-open earnings markets."""
    open_markets: Dict[str, Dict[str, str]] = {}
    events = _get(f"{GAMMA}/events", {
        "tag_slug": "earnings", "closed": "false", "limit": 100,
        "order": "volume", "ascending": "false",
    })
    for event in events or []:
        for market in event.get("markets") or []:
            condition_id = market.get("conditionId")
            if not condition_id:
                continue
            question = str(market.get("question") or event.get("title") or "")
            ticker_match = _TICKER_RE.search(question)
            open_markets[condition_id] = {
                "question": question,
                "ticker": ticker_match.group(1) if ticker_match else "",
            }
    return open_markets


def fetch_wallet_open_positions(wallet: str, open_markets: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
    """Net stance per open earnings market from the wallet's recent fills."""
    try:
        fills = _get(f"{DATA}/trades", {"user": wallet, "limit": 500, "takerOnly": "false"})
    except Exception:
        return []
    stance: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for fill in fills or []:
        condition_id = str(fill.get("conditionId") or "")
        if condition_id not in open_markets:
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
        held = {o: round(v, 2) for o, v in outcomes.items() if abs(v) > 0.5}
        if held:
            out.append({
                "question": open_markets[condition_id]["question"],
                "ticker": open_markets[condition_id]["ticker"],
                "net_shares": held,
            })
    return out


def run(max_markets: int, min_markets: int) -> Dict[str, Any]:
    markets = fetch_resolved_earnings_markets(max_markets)
    print(f"resolved earnings markets selected: {len(markets)}", file=sys.stderr)

    wallet_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "markets": 0, "wins": 0, "pnl": 0.0, "cost": 0.0,
        "win_entries": [], "name": "",
    })
    total_fills = 0
    for i, market in enumerate(markets, 1):
        try:
            fills = fetch_market_fills(market["condition_id"])
        except Exception as exc:
            print(f"  ! {market['ticker'] or market['condition_id'][:10]}: {exc}", file=sys.stderr)
            continue
        total_fills += len(fills)
        settled = settle_market(fills, market["winner"])
        for wallet, stats in settled.items():
            agg = wallet_stats[wallet]
            agg["markets"] += 1
            agg["pnl"] += stats["pnl"]
            agg["cost"] += stats["cost"]
            if stats["pnl"] > 0:
                agg["wins"] += 1
            if stats["win_entry_avg"] is not None:
                agg["win_entries"].append(stats["win_entry_avg"])
            if stats["name"]:
                agg["name"] = stats["name"]
        if i % 20 == 0:
            print(f"  processed {i}/{len(markets)} markets ({total_fills} fills)", file=sys.stderr)

    qualified = []
    for wallet, agg in wallet_stats.items():
        if agg["markets"] < min_markets:
            continue
        qualified.append({
            "wallet": wallet,
            "name": agg["name"],
            "markets": agg["markets"],
            "wins": agg["wins"],
            "win_rate": round(agg["wins"] / agg["markets"], 3),
            "pnl_usd": round(agg["pnl"], 2),
            "cost_usd": round(agg["cost"], 2),
            "roi": round(agg["pnl"] / agg["cost"], 3) if agg["cost"] > 0 else None,
            "avg_winner_entry_price": round(sum(agg["win_entries"]) / len(agg["win_entries"]), 3) if agg["win_entries"] else None,
        })
    qualified.sort(key=lambda w: -w["pnl_usd"])

    print("fetching open earnings markets + top-wallet positioning...", file=sys.stderr)
    open_markets = fetch_open_earnings_markets()
    for row in qualified[:10]:
        row["open_earnings_positions"] = fetch_wallet_open_positions(row["wallet"], open_markets)

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "markets_analyzed": len(markets),
        "fills_processed": total_fills,
        "wallets_seen": len(wallet_stats),
        "wallets_qualified": len(qualified),
        "min_markets": min_markets,
        "open_earnings_markets": len(open_markets),
        "leaderboard": qualified[:50],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-markets", type=int, default=120)
    parser.add_argument("--min-markets", type=int, default=5)
    parser.add_argument("--out", default="polymarket_pilot_output.json")
    args = parser.parse_args()

    output = run(args.max_markets, args.min_markets)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=1)

    print(f"\n=== Earnings-market wallet leaderboard (min {args.min_markets} markets) ===")
    print(f"{'wallet/name':<28} {'mkts':>4} {'win%':>5} {'pnl $':>10} {'roi':>6} {'entry':>6}")
    for row in output["leaderboard"][:20]:
        label = (row["name"] or row["wallet"][:10] + "...")[:27]
        roi = f"{row['roi']:.2f}" if row["roi"] is not None else "-"
        entry = f"{row['avg_winner_entry_price']:.2f}" if row["avg_winner_entry_price"] is not None else "-"
        print(f"{label:<28} {row['markets']:>4} {row['win_rate']*100:>4.0f}% {row['pnl_usd']:>10.2f} {roi:>6} {entry:>6}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

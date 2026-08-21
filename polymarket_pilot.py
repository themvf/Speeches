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
    positions: Dict[str, Dict[str, float]] = defaultdict(lambda: {"cash": 0.0, "cost": 0.0, "net_win": 0.0, "net_lose": 0.0, "win_buy_size": 0.0, "win_buy_cash": 0.0, "buy_size": 0.0})
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
            # Shares bought across BOTH outcomes. Paired with cost (which is
            # already total buy CASH), this gives the all-trades average entry
            # price - cost / buy_size - and both terms sum across markets, so
            # the wallet-level average stays correctly cost-weighted.
            pos["buy_size"] += size
        if outcome == winner:
            pos["net_win"] += signed
            if side == "BUY":
                pos["win_buy_size"] += size
                pos["win_buy_cash"] += size * price
        else:
            # Net position across every LOSING outcome. Needed because holding
            # the winner is not by itself a correct call: a wallet that buys
            # both sides holds the winning outcome in every market it touches,
            # whatever the result.
            pos["net_lose"] += signed
    out: Dict[str, Dict[str, float]] = {}
    for wallet, pos in positions.items():
        # The clamp keeps a wallet that acquired shares off-tape (on-chain
        # split/merge never appears in the trades feed) from being charged a
        # phantom settlement liability. It stays - but it must not be allowed
        # to decide CORRECTNESS, because it silently turns a net short of the
        # winning outcome into a positive number. net_win is therefore exported
        # so callers score direction from the position itself rather than from
        # the sign of a deliberately-conservative P&L figure.
        payout = max(pos["net_win"], 0.0)
        out[wallet] = {
            "pnl": pos["cash"] + payout,
            "cost": pos["cost"],
            # The settlement COMPONENTS, persisted alongside the conclusions
            # they produce. cash + net positions reconstruct pnl under any
            # payout rule (clamped or not); net_win/net_lose reconstruct
            # correctness under any definition of it; win_buy_size recovers
            # win_entry_avg. Three separate corrections to the meaning of
            # "correct" each forced a full re-settle purely because these were
            # thrown away - and raw fills are pruned 7 days after settlement,
            # so once they are gone the only source is a third-party API that
            # caps at ~3500 fills per market.
            "cash": pos["cash"],
            "net_win": pos["net_win"],
            "net_lose": pos["net_lose"],
            "win_buy_size": pos["win_buy_size"],
            "buy_size": pos["buy_size"],
            # Winners-only: what they paid on the trades that happened to win.
            # Cannot answer "did they beat the price they paid" - that needs
            # entry across ALL trades, which is cost / buy_size above.
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


# Archetype classification (shared with the SEC-28 snapshot builder). Code
# constants, deliberately not config - a silently-moved threshold would make
# historical badges incomparable (same stance as the Reddit quality flags).
# ARCH_MIN_MARKETS is a sample-size guard: one lucky call must not mint a
# "sharp" (mirrors the Reddit author-weighting min_items guard).
ARCH_MIN_MARKETS = 8


MIN_EDGE = 0.05


def classify_archetype(row: Dict[str, Any]) -> str:
    """Map a ranked-wallet row to early_sharp / news_scalper / longshot /
    unclassified.

    Gated on EDGE - win rate minus the all-trades average entry price - rather
    than on win rate and entry price as separate conditions. In a prediction
    market the price IS a probability: buy at 0.90 and win 90% of the time and
    you have added nothing. Testing the two separately let a wallet paying 0.98
    and winning 91% (edge -7 points) qualify as a "news scalper", implying a
    skill it did not have, while a wallet paying 0.18 and winning 42% (edge +22)
    stayed unclassified because its win rate looked poor. Both were real,
    observed live before this change.

    So: first decide whether there is edge at all, then describe WHERE it comes
    from. Style is read off entry price, which is the same signal as before -
    only now it labels a kind of edge rather than standing in for edge itself.
    """
    if row["markets"] < ARCH_MIN_MARKETS:
        return "unclassified"
    # All-trades average entry, NOT avg_winner_entry_price: the latter is
    # conditioned on winners and cannot say what was paid for the losses.
    entry = row.get("entry_avg")
    if entry is None:
        return "unclassified"
    win = row["win_rate"]
    roi = row.get("roi")
    edge = win - entry
    # Beating the prices paid AND actually making money. ROI is the same claim
    # denominated in dollars, so requiring both guards against a wallet whose
    # edge is real per-contract but destroyed by position sizing.
    if edge < MIN_EDGE or (roi is not None and roi <= 0):
        return "unclassified"
    # "Longshot" must keep meaning what it says: rare wins at long odds. Cheap
    # entries that win MORE often than not are not longshots - that is simply
    # buying cheap and being right, which is what early_sharp describes.
    if entry <= 0.35 and win < 0.50:
        return "longshot"
    # Edge earned at near-certainty prices is a speed edge, not a forecasting
    # one: the outcome was public, they were faster to it.
    if entry >= 0.80:
        return "news_scalper"
    return "early_sharp"


def analyze_resolved_markets(markets: List[Dict[str, Any]], capture_per_market: bool = False) -> tuple:
    """Fetch each resolved market's tape and fold every fill into per-wallet
    aggregates. Returns (wallet_stats, total_fills), or
    (wallet_stats, total_fills, per_market) when capture_per_market is set -
    per_market[condition_id] is that market's settle_market result, retained
    so the closed-markets retrospective can show per-market sharp performance
    without a second fetch pass."""
    wallet_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "markets": 0, "wins": 0, "pnl": 0.0, "cost": 0.0,
        "win_entries": [], "name": "",
    })
    per_market: Dict[str, Dict[str, Any]] = {}
    total_fills = 0
    for i, market in enumerate(markets, 1):
        try:
            fills = fetch_market_fills(market["condition_id"])
        except Exception as exc:
            print(f"  ! {market['ticker'] or market['condition_id'][:10]}: {exc}", file=sys.stderr)
            continue
        total_fills += len(fills)
        settled = settle_market(fills, market["winner"])
        if capture_per_market:
            per_market[market["condition_id"]] = settled
        for wallet, stats in settled.items():
            agg = wallet_stats[wallet]
            agg["markets"] += 1
            agg["pnl"] += stats["pnl"]
            agg["cost"] += stats["cost"]
            if stats.get("net_win", 0) > stats.get("net_lose", 0):
                agg["wins"] += 1
            if stats["win_entry_avg"] is not None:
                agg["win_entries"].append(stats["win_entry_avg"])
            if stats["name"]:
                agg["name"] = stats["name"]
        if i % 20 == 0:
            print(f"  processed {i}/{len(markets)} markets ({total_fills} fills)", file=sys.stderr)
    if capture_per_market:
        return wallet_stats, total_fills, per_market
    return wallet_stats, total_fills


def rank_wallets(wallet_stats: Dict[str, Dict[str, Any]], min_markets: int) -> List[Dict[str, Any]]:
    """Qualified (>=min_markets) wallets as ranked rows, PnL desc, each tagged
    with its archetype."""
    qualified = []
    for wallet, agg in wallet_stats.items():
        if agg["markets"] < min_markets:
            continue
        row = {
            "wallet": wallet,
            "name": agg["name"],
            "markets": agg["markets"],
            "wins": agg["wins"],
            "win_rate": round(agg["wins"] / agg["markets"], 3),
            "pnl_usd": round(agg["pnl"], 2),
            "cost_usd": round(agg["cost"], 2),
            "roi": round(agg["pnl"] / agg["cost"], 3) if agg["cost"] > 0 else None,
            "avg_winner_entry_price": round(sum(agg["win_entries"]) / len(agg["win_entries"]), 3) if agg["win_entries"] else None,
        }
        row["archetype"] = classify_archetype(row)
        qualified.append(row)
    qualified.sort(key=lambda w: -w["pnl_usd"])
    return qualified


def run(max_markets: int, min_markets: int) -> Dict[str, Any]:
    markets = fetch_resolved_earnings_markets(max_markets)
    print(f"resolved earnings markets selected: {len(markets)}", file=sys.stderr)

    wallet_stats, total_fills = analyze_resolved_markets(markets)
    qualified = rank_wallets(wallet_stats, min_markets)

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

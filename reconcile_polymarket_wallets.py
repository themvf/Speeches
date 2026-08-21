#!/usr/bin/env python3
"""Reconcile our reconstructed wallet numbers against Polymarket's own.

We rebuild every wallet statistic from the raw trade tape - P&L, entry price,
shares bought, which side they held - and until now never checked any of it
against the source that can say whether it is right. Polymarket publishes
per-position ground truth on a public endpoint: avgPrice, totalBought, cashPnl
and the outcome held. This compares the two.

Reads only. Takes our figures from the live /api/market/predictions payload
rather than the database, so it runs anywhere without credentials.

Known and expected sources of disagreement, which is the point of measuring
rather than assuming:
  - Our tape is truncated: the trades endpoint 400s past offset ~3500, so on
    high-volume markets we only ever saw the most recent fills. Positions do
    not have that limit, so a systematic gap here MEASURES that bias.
  - We score only earnings/macro markets; a wallet's positions include
    everything it trades (sports, politics). Comparison is therefore scoped to
    markets we actually track, where the payload allows it.

First run (2026-08-21) validated ENTRY PRICE: ours agreed with their avgPrice
to within ~3 points on every wallet with real overlap, which independently
confirms the buy_size/entry_avg reconstruction the archetype classifier now
depends on.

P&L is NOT yet a usable comparison and its disagreements should not be read as
bugs on our side:
  - Their tracked-position counts run far below our market counts (31 against
    389 on one wallet), so redeemed positions appear to drop out of the
    endpoint entirely - their sum covers fewer markets than ours.
  - cashPnl is unrealized for open positions while realizedPnl carries closed
    ones, so summing cashPnl alone mixes the two inconsistently.
Making P&L meaningful needs a per-conditionId join against our own settlement
rows (database, not this payload) rather than a wallet-level total.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from typing import Any, Dict, List, Optional

DATA_API = "https://data-api.polymarket.com"
DEFAULT_APP = "https://speeches-zeta.vercel.app"


def _get(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "sec-speeches-reconcile/1.0"})
    with urllib.request.urlopen(req, timeout=45) as response:
        return json.load(response)


def fetch_positions(wallet: str, page_size: int = 500, max_pages: int = 6) -> List[Dict[str, Any]]:
    """Every position we can see, closed ones included."""
    out: List[Dict[str, Any]] = []
    for page in range(max_pages):
        batch = _get(f"{DATA_API}/positions?user={wallet}&limit={page_size}"
                     f"&offset={page * page_size}&closed=true")
        if not batch:
            break
        out.extend(batch)
        if len(batch) < page_size:
            break
    return out


# Our earnings universe is Polymarket's per-company "Will X beat quarterly
# earnings?" series. Their positions payload covers everything a wallet trades,
# so without this filter the comparison is meaningless: one wallet showed 869
# positions against the 195 markets we score, and its overall P&L was dominated
# by sports. Scoping both sides to the same markets is what makes the numbers
# comparable at all.
EARNINGS_TITLE = re.compile(r"beat quarterly earnings", re.I)


def is_tracked(position: Dict[str, Any]) -> bool:
    return bool(EARNINGS_TITLE.search(str(position.get("title") or "")))


def summarize_positions(positions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Polymarket's own view, aggregated the same way we aggregate ours."""
    bought = 0.0
    cost = 0.0
    pnl = 0.0
    for position in positions:
        try:
            size = float(position.get("totalBought") or 0)
            price = float(position.get("avgPrice") or 0)
            pnl += float(position.get("cashPnl") or 0)
        except (TypeError, ValueError):
            continue
        bought += size
        cost += size * price
    return {
        "positions": len(positions),
        "total_bought": round(bought, 2),
        "entry_avg": round(cost / bought, 4) if bought > 0 else None,
        "pnl": round(pnl, 2),
    }


def our_wallets(app_url: str) -> Dict[str, Dict[str, Any]]:
    payload = _get(f"{app_url}/api/market/predictions")
    data = payload.get("data", payload)
    return {w["wallet"].lower(): w for w in data.get("wallets", [])}


def compare(ours: Dict[str, Any], theirs: Dict[str, Any]) -> Dict[str, Any]:
    trajectory = ours.get("trajectory") or {}
    our_entry = trajectory.get("entryAvg")
    their_entry = theirs["entry_avg"]
    entry_gap = (round(our_entry - their_entry, 4)
                 if our_entry is not None and their_entry is not None else None)
    return {
        "name": ours.get("name"),
        "our_markets": ours.get("markets"),
        "their_positions": theirs["positions"],
        "our_entry_avg": our_entry,
        "their_entry_avg": their_entry,
        "entry_gap": entry_gap,
        "our_pnl": ours.get("pnlUsd"),
        "their_pnl": theirs["pnl"],
        "entry_gap_material": entry_gap is not None and abs(entry_gap) > 0.05,
        # Their position count on tracked markets should be in the same
        # ballpark as our market count. A large shortfall on THEIR side is not
        # our bug; a large shortfall on OURS is the truncated-tape bias, since
        # the trades endpoint stops paging at ~3500 fills while positions do
        # not.
        "coverage_ratio": (round(theirs["positions"] / ours["markets"], 2)
                           if ours.get("markets") else None),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-url", default=DEFAULT_APP)
    parser.add_argument("--limit", type=int, default=10, help="wallets to reconcile")
    parser.add_argument("--wallets", nargs="*", help="specific wallet names")
    args = parser.parse_args(argv)

    summary: Dict[str, Any] = {"source_key": "reconcile_polymarket_wallets", "errors": []}
    try:
        ours = our_wallets(args.app_url)
        # Prefer wallets with an entry price to compare and real history.
        ranked = sorted(
            [w for w in ours.values() if (w.get("trajectory") or {}).get("entryAvg") is not None],
            key=lambda w: -(w.get("markets") or 0))
        if args.wallets:
            wanted = {n.lower() for n in args.wallets}
            ranked = [w for w in ours.values() if str(w.get("name", "")).lower() in wanted]
        rows = []
        for wallet in ranked[: args.limit]:
            try:
                positions = fetch_positions(wallet["wallet"])
            except Exception as exc:
                summary["errors"].append(f"{wallet.get('name')}: {exc}")
                continue
            tracked = [p for p in positions if is_tracked(p)]
            row = compare(wallet, summarize_positions(tracked))
            row["their_untracked_positions"] = len(positions) - len(tracked)
            rows.append(row)
        summary["reconciled"] = rows
        summary["entry_gaps_material"] = sum(1 for r in rows if r["entry_gap_material"])
        summary["ok"] = True
    except Exception as exc:
        summary["errors"].append(str(exc))
        summary["ok"] = False
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

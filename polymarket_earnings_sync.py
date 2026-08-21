#!/usr/bin/env python3
"""SEC-26/27: Polymarket earnings live pipeline - 3x-daily sync.

Each run (cron 13:00 / 20:25 / 01:00 UTC = 9:00am / 4:25pm / 9:00pm ET
during EDT; +1h drift in winter, harmless - ingestion is cursor-based so
cadence affects freshness only, never completeness):

1. Refresh open earnings markets from gamma-api (metadata + implied prob).
2. Detect resolutions (recently-closed earnings events, endDate-desc).
3. Ingest new fills for every unsettled tracked market - client-side cursor
   (newest-first paging, stop below the stored max fill timestamp; the data
   API has no since-param and no trade ids, so dedup is a content-hash
   fill_key with ON CONFLICT DO NOTHING). SEC-29: any new fill in a still-OPEN
   market whose wallet is on the sharp watchlist (polymarket_wallet_stats
   archetype early_sharp/longshot) is written to polymarket_sharp_alerts in
   the same pass - no second scan of the fills.
4. Settle newly-resolved markets into the DURABLE
   polymarket_wallet_market_results table (one compact row per
   wallet-market), then recompute polymarket_wallet_stats + archetypes from
   that table - never from raw fills.
5. Prune raw fills of markets settled >= 7 days ago (settle-then-prune;
   open markets keep their full tape regardless of age), and age out
   polymarket_sharp_alerts older than 30 days.

--backfill (one-time cold start): settles the top-volume RESOLVED earnings
markets in memory straight into the durable results table - their raw fills
are never stored.

Requires: DATABASE_URL. Network: gamma-api/data-api.polymarket.com (public,
no auth). Research context only - not investment advice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

import requests

import neon_feeds
import polymarket_pilot as pilot
import wallet_trajectory
from build_prediction_snapshot import fetch_open_earnings_detailed
from source_health import record_source_health

SOURCE_KEY = "polymarket_earnings_sync"
TRADES_PAGE = 500
MAX_INCREMENTAL_FILLS = 6000
PRUNE_DAYS_AFTER_SETTLEMENT = 7
PRUNE_ALERT_DAYS = 30
RESOLUTION_PAGES = 2  # x100 recently-closed events checked per run


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def fill_key(condition_id: str, fill: Dict[str, Any]) -> str:
    """Content-hash identity for a fill - the data API exposes no trade id.
    Two genuinely identical distinct fills colliding is possible but rare and
    immaterial for aggregate stats."""
    raw = "|".join([
        condition_id,
        str(fill.get("transactionHash") or ""),
        str(fill.get("proxyWallet") or ""),
        str(fill.get("outcome") or ""),
        str(fill.get("side") or ""),
        str(fill.get("size") or ""),
        str(fill.get("price") or ""),
        str(fill.get("timestamp") or ""),
    ])
    return hashlib.md5(raw.encode()).hexdigest()


def fill_row(condition_id: str, fill: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        ts = float(fill.get("timestamp") or 0)
        size = float(fill.get("size") or 0)
        price = float(fill.get("price") or 0)
    except (TypeError, ValueError):
        return None
    wallet = str(fill.get("proxyWallet") or "")
    side = str(fill.get("side") or "").upper()
    if not wallet or ts <= 0 or size <= 0 or side not in ("BUY", "SELL"):
        return None
    return {
        "fill_key": fill_key(condition_id, fill),
        "condition_id": condition_id,
        "wallet": wallet,
        "name": str(fill.get("name") or fill.get("pseudonym") or ""),
        "outcome": str(fill.get("outcome") or ""),
        "side": side,
        "size": size,
        "price": price,
        "filled_at": datetime.fromtimestamp(ts, tz=UTC),
    }


def fetch_new_fills(condition_id: str, cursor: Optional[datetime]) -> List[Dict[str, Any]]:
    """Newest-first incremental fetch: stop as soon as a fill is strictly
    older than the cursor (ties re-fetched and deduped by fill_key). With no
    cursor (newly-discovered market) this pulls the full tape, subject to the
    API's ~3500 offset ceiling (partial-mega-market caveat, same as pilot)."""
    fills: List[Dict[str, Any]] = []
    offset = 0
    while offset < MAX_INCREMENTAL_FILLS:
        try:
            page = pilot._get(f"{pilot.DATA}/trades", {
                "market": condition_id, "limit": TRADES_PAGE, "offset": offset,
                "takerOnly": "false",
            })
        except requests.HTTPError:
            break  # offset ceiling - keep what we have
        if not isinstance(page, list) or not page:
            break
        for fill in page:
            row = fill_row(condition_id, fill)
            if row is None:
                continue
            if cursor is not None and row["filled_at"] < cursor:
                return fills
            fills.append(row)
        if len(page) < TRADES_PAGE:
            break
        offset += TRADES_PAGE
    return fills


def fetch_recent_resolutions() -> Dict[str, str]:
    """condition_id -> winning outcome for recently-closed earnings events,
    newest resolutions first (bounded pages - a 3x-daily run only needs to
    catch what resolved since the last run, with generous margin)."""
    winners: Dict[str, str] = {}
    for page in range(RESOLUTION_PAGES):
        events = pilot._get(f"{pilot.GAMMA}/events", {
            "tag_slug": "earnings", "closed": "true", "order": "endDate",
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
                if not condition_id or len(outcomes) != len(prices):
                    continue
                resolved = [o for o, p in zip(outcomes, prices) if p > 0.99]
                if len(resolved) == 1:
                    winners[condition_id] = resolved[0]
    return winners


def recompute_wallet_stats() -> Dict[str, int]:
    """SEC-27: rebuild polymarket_wallet_stats from the DURABLE results table
    (never raw fills), reusing the pilot's archetype classifier."""
    results = neon_feeds.get_polymarket_wallet_results()
    agg: Dict[str, Dict[str, Any]] = {}
    # Raw per-market rows retained per wallet so wallet_trajectory can read the
    # tail of the history in order; the aggregate above is order-blind.
    events_by_wallet: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        a = agg.setdefault(str(r["wallet"]), {
            "markets": 0, "wins": 0, "pnl": 0.0, "cost": 0.0, "buy_size": 0.0, "entries": [], "name": "",
        })
        events_by_wallet.setdefault(str(r["wallet"]), []).append(r)
        a["markets"] += 1
        a["pnl"] += float(r["pnl"] or 0)
        a["cost"] += float(r["cost"] or 0)
        a["buy_size"] += float(r.get("buy_size") or 0)
        if r["correct"]:
            a["wins"] += 1
        if r["win_entry_avg"] is not None:
            a["entries"].append(float(r["win_entry_avg"]))
        if r["name"]:
            a["name"] = str(r["name"])

    rows: List[Dict[str, Any]] = []
    archetype_counts: Dict[str, int] = {}
    for wallet, a in agg.items():
        entry_avg = round(sum(a["entries"]) / len(a["entries"]), 4) if a["entries"] else None
        classify_row = {
            "markets": a["markets"],
            "win_rate": a["wins"] / a["markets"],
            "pnl_usd": a["pnl"],
            "roi": (a["pnl"] / a["cost"]) if a["cost"] > 0 else None,
            "avg_winner_entry_price": entry_avg,
        }
        archetype = pilot.classify_archetype(classify_row)
        archetype_counts[archetype] = archetype_counts.get(archetype, 0) + 1
        trajectory = wallet_trajectory.summarize(
            events_by_wallet.get(wallet, []),
            qualified=archetype != "unclassified", lifetime_roi=classify_row["roi"])
        rows.append({
            "wallet": wallet, "name": a["name"], "markets": a["markets"],
            "wins": a["wins"], "pnl": a["pnl"], "cost": a["cost"],
            "win_entry_avg": entry_avg, "archetype": archetype,
            "buy_size": a["buy_size"],
            # All-trades average entry: cost is total buy cash, buy_size total
            # shares. Unlike win_entry_avg this includes losing trades, so it
            # is what a win rate must beat to count as edge.
            "entry_avg": (a["cost"] / a["buy_size"]) if a["buy_size"] > 0 else None,
            **trajectory,
        })
    written = neon_feeds.upsert_polymarket_wallet_stats(rows)
    return {"wallets": written, **{f"arch_{k}": v for k, v in archetype_counts.items()}}


def run_backfill(max_markets: int, summary: Dict[str, Any]) -> None:
    """Cold start: settle already-resolved markets straight into the durable
    results table. Raw fills are held in memory only - never stored."""
    markets = pilot.fetch_resolved_earnings_markets(max_markets)
    summary["backfill_markets"] = len(markets)
    neon_feeds.upsert_polymarket_markets([
        {
            "condition_id": m["condition_id"], "ticker": m["ticker"],
            "question": m["question"], "volume": m["volume"],
            "end_date": m["end_date"] or None,
        }
        for m in markets
    ])
    settled_count = 0
    for i, market in enumerate(markets, 1):
        try:
            fills = pilot.fetch_market_fills(market["condition_id"])
            settled = pilot.settle_market(fills, market["winner"])
            neon_feeds.mark_polymarket_resolved(market["condition_id"], market["winner"])
            neon_feeds.save_polymarket_settlement(
                market["condition_id"], market["ticker"],
                (market["end_date"] or "")[:10] or None, settled,
            )
            settled_count += 1
        except Exception as exc:
            summary["errors"].append(f"backfill {market['ticker'] or market['condition_id'][:10]}: {exc}")
        if i % 20 == 0:
            print(f"  backfill {i}/{len(markets)}", file=sys.stderr)
    summary["backfill_settled"] = settled_count
    summary["stats"] = recompute_wallet_stats()


def run_sync(summary: Dict[str, Any]) -> None:
    # 1. Refresh open markets (metadata, implied prob, newly-listed markets).
    open_detailed = fetch_open_earnings_detailed()
    market_rows = []
    for condition_id, m in open_detailed.items():
        market_rows.append({
            "condition_id": condition_id, "ticker": m["ticker"],
            "question": m["question"], "eps": m["eps"],
            "report_date": m["report_date"], "end_date": m["end_date"] or None,
            "volume": m["volume"], "implied_prob_yes": m["implied_prob_yes"],
        })
    summary["open_markets"] = neon_feeds.upsert_polymarket_markets(market_rows)

    # 2. Resolution detection.
    tracked = neon_feeds.get_polymarket_tracked_markets()
    winners = fetch_recent_resolutions()
    resolved_now: List[Dict[str, Any]] = []
    for market in tracked:
        if market["status"] == "open" and market["condition_id"] in winners:
            neon_feeds.mark_polymarket_resolved(market["condition_id"], winners[market["condition_id"]])
            market["status"] = "resolved"
            market["winner"] = winners[market["condition_id"]]
        if market["status"] == "resolved" and market["settled_at"] is None:
            resolved_now.append(market)
    summary["newly_resolved"] = len(resolved_now)

    # 3. Incremental fill ingestion for every unsettled tracked market, plus
    #    SEC-29 sharp-wallet alerting on new fills into still-open markets -
    #    piggybacked on this same fetch so alerting never needs its own pass.
    sharp_wallets = neon_feeds.get_polymarket_sharp_wallet_set()
    fills_written = 0
    alerts_written = 0
    for market in tracked:
        if market["settled_at"] is not None:
            continue
        try:
            new_fills = fetch_new_fills(market["condition_id"], market["fill_cursor"])
            if new_fills:
                fills_written += neon_feeds.insert_polymarket_fills(new_fills)
                if market["status"] == "open" and sharp_wallets:
                    alerts = [
                        {
                            "fill_key": f["fill_key"], "condition_id": f["condition_id"],
                            "ticker": market["ticker"] or "", "wallet": f["wallet"],
                            "name": sharp_wallets[f["wallet"]]["name"] or f["name"],
                            "archetype": sharp_wallets[f["wallet"]]["archetype"],
                            "side": f["side"], "outcome": f["outcome"],
                            "size": f["size"], "price": f["price"], "filled_at": f["filled_at"],
                        }
                        for f in new_fills if f["wallet"] in sharp_wallets
                    ]
                    if alerts:
                        alerts_written += neon_feeds.insert_polymarket_sharp_alerts(alerts)
        except Exception as exc:
            summary["errors"].append(f"fills {market['ticker'] or market['condition_id'][:10]}: {exc}")
    summary["fills_written"] = fills_written
    summary["alerts_written"] = alerts_written

    # 4. Settle newly-resolved markets from their (now-complete) stored tapes,
    #    then recompute durable wallet stats once.
    settled_count = 0
    for market in resolved_now:
        try:
            fills = neon_feeds.get_polymarket_market_fills(market["condition_id"])
            settled = pilot.settle_market(fills, market["winner"])
            end_date = market.get("end_date")
            resolved_date = end_date.date() if hasattr(end_date, "date") else None
            neon_feeds.save_polymarket_settlement(market["condition_id"], market["ticker"] or "", resolved_date, settled)
            settled_count += 1
        except Exception as exc:
            summary["errors"].append(f"settle {market['ticker'] or market['condition_id'][:10]}: {exc}")
    summary["settled"] = settled_count
    if settled_count:
        summary["stats"] = recompute_wallet_stats()

    # 5. Settle-then-prune raw-fill retention, plus age-based alert retention.
    summary["fills_pruned"] = neon_feeds.prune_settled_polymarket_fills(PRUNE_DAYS_AFTER_SETTLEMENT)
    summary["alerts_pruned"] = neon_feeds.prune_old_polymarket_sharp_alerts(PRUNE_ALERT_DAYS)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backfill", action="store_true",
                        help="One-time cold start: settle resolved markets into the durable results table")
    parser.add_argument("--backfill-markets", type=int, default=120)
    args = parser.parse_args(argv)

    summary: Dict[str, Any] = {
        "source_key": SOURCE_KEY,
        "connector": SOURCE_KEY,
        "mode": "backfill" if args.backfill else "sync",
        "errors": [],
        "ran_at": _utc_now_iso(),
    }
    try:
        if args.backfill:
            run_backfill(args.backfill_markets, summary)
        else:
            run_sync(summary)
        summary["ok"] = not summary["errors"] or (
            # partial failures don't fail the run; a run with zero successes does
            summary.get("fills_written", 0) > 0 or summary.get("open_markets", 0) > 0
            or summary.get("backfill_settled", 0) > 0
        )
    except Exception as exc:
        summary["errors"].append(str(exc))
        summary["ok"] = False

    summary["failed_count"] = len(summary["errors"])
    summary["processed_count"] = (
        summary.get("fills_written", 0) + summary.get("settled", 0) + summary.get("backfill_settled", 0)
    )
    summary["discovered_count"] = summary.get("open_markets", 0) + summary.get("backfill_markets", 0)
    record_source_health(summary)
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())

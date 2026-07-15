"""Tests for polymarket_earnings_sync (SEC-26/27): cursor logic, fill
normalization, durable stats recompute, and the settle-then-prune ordering
guarantee. Network and DB are mocked throughout."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import polymarket_earnings_sync as sync


def _api_fill(ts: int, wallet="0xabc", size=10.0, price=0.5, side="BUY", outcome="Yes", tx="0xt1"):
    return {"proxyWallet": wallet, "outcome": outcome, "side": side, "size": size,
            "price": price, "timestamp": ts, "transactionHash": tx, "name": "tester"}


NOW_TS = 1_784_000_000


def test_fill_row_normalizes_and_rejects_junk():
    row = sync.fill_row("0xc1", _api_fill(NOW_TS))
    assert row is not None
    assert row["wallet"] == "0xabc" and row["side"] == "BUY" and row["size"] == 10.0
    assert row["filled_at"] == datetime.fromtimestamp(NOW_TS, tz=UTC)
    assert sync.fill_row("0xc1", _api_fill(NOW_TS, wallet="")) is None
    assert sync.fill_row("0xc1", _api_fill(NOW_TS, size=0)) is None
    assert sync.fill_row("0xc1", {"proxyWallet": "0xabc", "side": "HOLD", "size": 1, "price": 0.5, "timestamp": NOW_TS}) is None


def test_fill_key_is_stable_and_distinguishes_fills():
    a1 = sync.fill_key("0xc1", _api_fill(NOW_TS))
    a2 = sync.fill_key("0xc1", _api_fill(NOW_TS))
    b = sync.fill_key("0xc1", _api_fill(NOW_TS, size=11.0))
    assert a1 == a2  # deterministic -> ON CONFLICT dedup works across runs
    assert a1 != b


def test_fetch_new_fills_stops_at_cursor():
    # Newest-first page: fills at t+300, t+200, t+100 (cursor), t-100.
    cursor = datetime.fromtimestamp(NOW_TS, tz=UTC)
    page = [
        _api_fill(NOW_TS + 300, tx="0xa"),
        _api_fill(NOW_TS + 200, tx="0xb"),
        _api_fill(NOW_TS, tx="0xc"),        # == cursor: re-fetched, deduped by key
        _api_fill(NOW_TS - 100, tx="0xd"),  # strictly older: stop before this
    ]
    with patch.object(sync.pilot, "_get", return_value=page):
        fills = sync.fetch_new_fills("0xc1", cursor)
    txs = [f["fill_key"] for f in fills]
    assert len(fills) == 3  # the boundary-equal fill included, the older one not
    assert sync.fill_key("0xc1", page[3]) not in txs


def test_fetch_new_fills_no_cursor_pages_until_short_page():
    pages = [
        [_api_fill(NOW_TS + i, tx=f"0x{i}") for i in range(500)],
        [_api_fill(NOW_TS - 1, tx="0xlast")],
    ]
    calls = []

    def fake_get(url, params):
        calls.append(params["offset"])
        return pages[params["offset"] // 500]

    with patch.object(sync.pilot, "_get", side_effect=fake_get):
        fills = sync.fetch_new_fills("0xc1", None)
    assert len(fills) == 501
    assert calls == [0, 500]


def test_recompute_wallet_stats_aggregates_durable_results_and_classifies():
    results = []
    # 10 markets for a sharp: 7 wins, cheap entries.
    for i in range(10):
        results.append({"wallet": "0xsharp", "name": "sharpie", "pnl": 50 if i < 7 else -20,
                        "cost": 100, "win_entry_avg": 0.5 if i < 7 else None, "correct": i < 7})
    # 2 markets only: below the archetype gate.
    for i in range(2):
        results.append({"wallet": "0xnew", "name": "newbie", "pnl": 10, "cost": 10,
                        "win_entry_avg": 0.3, "correct": True})
    with patch.object(sync.neon_feeds, "get_polymarket_wallet_results", return_value=results):
        with patch.object(sync.neon_feeds, "upsert_polymarket_wallet_stats", return_value=2) as mock_upsert:
            summary = sync.recompute_wallet_stats()
    rows = {r["wallet"]: r for r in mock_upsert.call_args.args[0]}
    assert rows["0xsharp"]["markets"] == 10 and rows["0xsharp"]["wins"] == 7
    assert rows["0xsharp"]["archetype"] == "early_sharp"
    assert rows["0xnew"]["archetype"] == "unclassified"  # sample-size gate
    assert summary["wallets"] == 2


def test_run_sync_settles_before_pruning_and_only_after_resolution():
    """The retention guarantee: settlement of a newly-resolved market happens
    BEFORE any prune call, and open markets are never settled."""
    manager = MagicMock()
    tracked = [
        {"condition_id": "0xopen", "ticker": "AAA", "status": "open", "winner": None,
         "settled_at": None, "fills_pruned_at": None, "end_date": None, "fill_cursor": None},
        {"condition_id": "0xdone", "ticker": "BBB", "status": "open", "winner": None,
         "settled_at": None, "fills_pruned_at": None,
         "end_date": datetime.now(UTC) - timedelta(days=1), "fill_cursor": None},
    ]
    with patch.object(sync, "fetch_open_earnings_detailed", return_value={}), \
         patch.object(sync, "fetch_recent_resolutions", return_value={"0xdone": "No"}), \
         patch.object(sync, "fetch_new_fills", return_value=[]), \
         patch.object(sync, "recompute_wallet_stats", return_value={"wallets": 0}), \
         patch.object(sync.pilot, "settle_market", return_value={}) as mock_settle, \
         patch.object(sync.neon_feeds, "upsert_polymarket_markets", return_value=0), \
         patch.object(sync.neon_feeds, "get_polymarket_tracked_markets", return_value=tracked), \
         patch.object(sync.neon_feeds, "mark_polymarket_resolved", manager.mark), \
         patch.object(sync.neon_feeds, "get_polymarket_market_fills", return_value=[]), \
         patch.object(sync.neon_feeds, "save_polymarket_settlement", manager.settle), \
         patch.object(sync.neon_feeds, "prune_settled_polymarket_fills", manager.prune):
        manager.settle.return_value = 0
        manager.prune.return_value = 0
        summary = {"errors": []}
        sync.run_sync(summary)

    # Only the resolved market was settled; the open one wasn't.
    manager.mark.assert_called_once_with("0xdone", "No")
    manager.settle.assert_called_once()
    assert manager.settle.call_args.args[0] == "0xdone"
    # Ordering: settle strictly before prune.
    ops = [c[0] for c in manager.mock_calls if c[0] in ("settle", "prune")]
    assert ops == ["settle", "prune"]
    assert summary["newly_resolved"] == 1 and summary["settled"] == 1

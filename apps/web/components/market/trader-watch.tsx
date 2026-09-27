"use client";

import { useCallback, useEffect, useState } from "react";
import { PUMP_ACTIONS, TRADER_WATCH_PROFILES, type PumpAction, type PumpActivity, type PumpActivityPage, type TraderWatchData } from "@/lib/trader-watch";

const PUMP_ACTION_LABELS: Record<PumpAction, string> = { BUY: "Buys", RECEIVE: "Receipts", SELL: "Sells", SEND: "Sends" };

function timeLabel(value: string | null): string {
  if (!value) return "Time unavailable";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Time unavailable" : date.toLocaleString();
}

export function TraderWatch() {
  const [trader, setTrader] = useState(TRADER_WATCH_PROFILES[0].id);
  const [data, setData] = useState<TraderWatchData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pumpAction, setPumpAction] = useState<PumpAction>("RECEIVE");
  const [pumpPages, setPumpPages] = useState<PumpActivity | null>(null);
  const [olderLoading, setOlderLoading] = useState(false);
  const [olderError, setOlderError] = useState<string | null>(null);

  const load = useCallback(async (signal?: AbortSignal) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`/api/market/crypto/trader-watch?trader=${encodeURIComponent(trader)}`, { signal });
      const result = await response.json();
      if (!response.ok || !result.ok) throw new Error(result.error || "Trader watch failed to load");
      const next = result.data as TraderWatchData;
      setData(next);
      setPumpPages(next.pumpActivity);
    } catch (cause) {
      if (signal?.aborted) return;
      setError(cause instanceof Error ? cause.message : "Trader watch failed to load");
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }, [trader]);

  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    const timer = setInterval(() => void load(controller.signal), 120_000);
    return () => { clearInterval(timer); controller.abort(); };
  }, [load]);

  const loadOlder = useCallback(async () => {
    const cursor = pumpPages?.[pumpAction].nextCursor;
    if (!cursor || olderLoading) return;
    setOlderLoading(true);
    setOlderError(null);
    try {
      const query = new URLSearchParams({ trader, action: pumpAction, cursor });
      const response = await fetch(`/api/market/crypto/trader-watch/pump-activity?${query}`);
      const result = await response.json();
      if (!response.ok || !result.ok || result.page?.status !== "available") throw new Error("Older activity could not be loaded");
      const page = result.page as PumpActivityPage;
      setPumpPages((current) => {
        if (!current || current[pumpAction].nextCursor !== cursor) return current;
        const seen = new Set(current[pumpAction].items.map((item) => item.signature));
        return { ...current, [pumpAction]: {
          ...page,
          items: [...current[pumpAction].items, ...page.items.filter((item) => !seen.has(item.signature))],
        } };
      });
    } catch (cause) {
      setOlderError(cause instanceof Error ? cause.message : "Older activity could not be loaded");
    } finally { setOlderLoading(false); }
  }, [pumpPages, pumpAction, olderLoading, trader]);

  const profile = data?.profile ?? TRADER_WATCH_PROFILES.find((item) => item.id === trader)!;
  const wallet = profile.wallet;
  const selectedPumpPage = pumpPages?.[pumpAction];
  const seenMints = new Set<string>();
  const recentIncreases = (data?.walletActivity.items ?? [])
    .flatMap((activity) => activity.tokenChanges.filter((change) => change.delta.startsWith("+")).map((change) => ({ change, activity })))
    .filter(({ change }) => {
      if (seenMints.has(change.mint)) return false;
      seenMints.add(change.mint);
      return true;
    });

  return (
    <section className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)] p-4 sm:p-5" aria-labelledby="wallet-watch-heading">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 id="wallet-watch-heading" className="text-base font-semibold text-[color:var(--ink)]">Wallet Watch</h2>
          <p className="mt-1 text-sm text-[color:var(--ink-faint)]">Current balances and recent transactions for this address. Reloads every two minutes while open.</p>
        </div>
        <button type="button" onClick={() => void load()} disabled={loading} className="min-h-11 rounded-lg border border-[color:var(--line-strong)] px-3 text-sm text-[color:var(--ink)] disabled:opacity-50">
          {loading ? "Reloading…" : "Reload data"}
        </button>
      </div>

      {TRADER_WATCH_PROFILES.length > 1 && (
        <label className="mt-4 block text-xs text-[color:var(--ink-faint)]">
          Wallet profile
          <select value={trader} onChange={(event) => { setTrader(event.target.value); setData(null); setPumpPages(null); }} className="ml-2 min-h-11 rounded-lg border border-[color:var(--line)] bg-[color:var(--surface)] px-3 text-base text-[color:var(--ink)]">
            {TRADER_WATCH_PROFILES.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
          </select>
        </label>
      )}

      <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
        <span className="font-semibold text-[color:var(--ink)]">{profile.name}</span>
        <span className="text-[color:var(--ink-faint)]">{profile.description}</span>
        {wallet && <a href={wallet.profileUrl} target="_blank" rel="noopener noreferrer" className="text-[color:var(--accent)] underline underline-offset-2">Open Pump.fun profile ↗</a>}
      </div>
      {wallet && (
        <div className="mt-3 rounded-lg border border-amber-400/30 bg-amber-400/5 p-3 text-sm text-[color:var(--ink)]">
          <p className="font-medium">Tracked Solana wallet</p>
          <p className="mt-1 text-[color:var(--ink-faint)]">Following this address directly. The Pump.fun profile name is a label, not a claim about who controls the wallet.</p>
          <a href={`https://explorer.solana.com/address/${wallet.address}`} target="_blank" rel="noopener noreferrer" className="mt-2 inline-block break-all font-mono text-xs text-[color:var(--accent)] underline underline-offset-2">{wallet.address} ↗</a>
        </div>
      )}

      <p className="mt-3 text-xs text-[color:var(--ink-faint)]" role="status">
        {error ? `Update failed: ${error}. Showing the last available result.` : loading && !data ? "Loading wallet activity…" : data ? `Checked ${timeLabel(data.generatedAt)}. Balances and transactions have separate source times.` : ""}
      </p>

      <div className="mt-4 grid gap-4">
        <div className="min-w-0 rounded-lg border border-[color:var(--line)] p-3">
          <h3 className="text-sm font-semibold text-[color:var(--ink)]">Current wallet balances</h3>
          <p className="mt-1 text-xs text-[color:var(--ink-faint)]">Positive balances in this address’s SPL Token and Token-2022 accounts, plus native SOL. Snapshot checked {timeLabel(data?.walletHoldings.observedAt ?? null)}.</p>
          {data?.walletHoldings.note && <p className="mt-3 text-sm text-amber-300">{data.walletHoldings.note}</p>}
          {data && data.walletHoldings.status !== "unavailable" && <>
            <p className="mt-3 text-xs text-[color:var(--ink-faint)]">{data?.walletHoldings.items.length ?? 0} token mints with a positive balance{data?.walletHoldings.status === "partial" ? " · partial RPC result" : ""}</p>
            {data?.walletHoldings.sol !== null && data?.walletHoldings.sol !== undefined && <p className="mt-2 text-sm text-[color:var(--ink)]"><span className="font-semibold">{data.walletHoldings.sol} SOL</span> · native balance</p>}
            {data?.walletHoldings.items.length ? <ul className="mt-2 max-h-[420px] divide-y divide-[color:var(--line)] overflow-y-auto">
              {data.walletHoldings.items.map((holding) => <li key={holding.mint} className="py-2 text-sm text-[color:var(--ink)]">
                <span className="font-semibold">{holding.amount} {holding.symbol ?? "tokens"}</span>
                <span className="ml-2 text-xs text-[color:var(--ink-faint)]">{holding.name ?? "Unlabeled mint"}{holding.labelSource ? " · DEX Screener label" : ""}</span>
                <a href={`https://explorer.solana.com/address/${holding.mint}`} target="_blank" rel="noopener noreferrer" className="mt-1 block break-all font-mono text-xs text-[color:var(--accent)] underline underline-offset-2">{holding.mint} ↗</a>
              </li>)}
            </ul> : <p className="mt-2 text-xs text-[color:var(--ink-faint)]">No positive SPL token balances were returned.</p>}
          </>}
          <p className="mt-2 text-xs text-[color:var(--ink-faint)]">A token label comes from a trading pair and is not identity verification. Zero-balance accounts, other wallets, exchange balances, and off-chain positions are outside this view.</p>
        </div>
        <div className="min-w-0 rounded-lg border border-[color:var(--line)] p-3">
          <h3 className="text-sm font-semibold text-[color:var(--ink)]">Named wallet activity from Pump.fun</h3>
          <p className="mt-1 text-xs text-[color:var(--ink-faint)]">Pump.fun groups this address’s history into buys, receipts, sells, and sends. A receipt is a transfer into the wallet, not proof of a purchase. Missing token names are filled from DEX Screener when available.</p>
          {pumpPages?.BUY.status === "available" && pumpPages.BUY.items.length === 0 && <p className="mt-2 text-xs text-amber-300">Pump.fun currently reports no Buy records for this address. Check Receipts for tokens transferred in.</p>}
          <div className="mt-3 flex flex-wrap gap-2" role="group" aria-label="Pump.fun activity category">
            {PUMP_ACTIONS.map((action) => <button key={action} type="button" onClick={() => { setPumpAction(action); setOlderError(null); }} aria-pressed={pumpAction === action} className={`min-h-11 rounded-lg border px-3 text-sm ${pumpAction === action ? "border-[color:var(--accent)] text-[color:var(--ink)]" : "border-[color:var(--line)] text-[color:var(--ink-faint)]"}`}>
              {PUMP_ACTION_LABELS[action]}{pumpPages?.[action].status === "available" ? ` (${pumpPages[action].items.length}${pumpPages[action].nextCursor ? "+" : ""})` : ""}
            </button>)}
          </div>
          {selectedPumpPage?.status === "unavailable" && <p className="mt-3 text-sm text-amber-300">{selectedPumpPage.note}</p>}
          {selectedPumpPage?.status === "available" && selectedPumpPage.items.length === 0 && <p className="mt-3 text-sm text-[color:var(--ink-faint)]">{pumpAction === "BUY" ? "Pump.fun currently returns no Buy records for this address. Tokens may still have arrived by transfer; see Receipts." : `No ${PUMP_ACTION_LABELS[pumpAction].toLowerCase()} returned by Pump.fun.`}</p>}
          {selectedPumpPage?.status === "available" && selectedPumpPage.items.length > 0 && <ul className="mt-3 max-h-[420px] divide-y divide-[color:var(--line)] overflow-y-auto">
            {selectedPumpPage.items.map((item) => <li key={item.signature} className="py-2 text-sm text-[color:var(--ink)]">
              <span className="font-semibold">{item.name ?? item.symbol ?? "Unlabeled token"}</span>
              {item.name && item.symbol && <span className="ml-1 text-xs text-[color:var(--ink-faint)]">({item.symbol})</span>}
              {item.labelSource === "dexscreener" && <span className="ml-1 text-xs text-[color:var(--ink-faint)]">· DEX Screener label</span>}
              <span className="ml-2 text-xs text-[color:var(--ink-faint)]">{item.amount} · {timeLabel(item.timestamp)}</span>
              <a href={item.url} target="_blank" rel="noopener noreferrer" className="ml-2 text-xs text-[color:var(--accent)] underline underline-offset-2">transaction ↗</a>
              <a href={`https://explorer.solana.com/address/${item.mint}`} target="_blank" rel="noopener noreferrer" className="mt-1 block break-all font-mono text-xs text-[color:var(--accent)] underline underline-offset-2">{item.mint} ↗</a>
            </li>)}
          </ul>}
          {selectedPumpPage?.nextCursor && <button type="button" onClick={() => void loadOlder()} disabled={olderLoading} className="mt-3 min-h-11 rounded-lg border border-[color:var(--line-strong)] px-3 text-sm text-[color:var(--ink)] disabled:opacity-50">{olderLoading ? "Loading older activity…" : `Load older ${PUMP_ACTION_LABELS[pumpAction].toLowerCase()}`}</button>}
          {olderError && <p className="mt-2 text-xs text-amber-300" role="status">{olderError}</p>}
          {wallet && <p className="mt-3 text-xs text-[color:var(--ink-faint)]">This is Pump.fun’s indexed history, which may be incomplete or differ from raw chain data. <a href={wallet.profileUrl} target="_blank" rel="noopener noreferrer" className="text-[color:var(--accent)] underline underline-offset-2">Open source profile ↗</a></p>}
        </div>
        <div className="min-w-0 rounded-lg border border-[color:var(--line)] p-3">
          <h3 className="text-sm font-semibold text-[color:var(--ink)]">Recent wallet transactions</h3>
          <p className="mt-1 text-xs text-[color:var(--ink-faint)]">Latest 8 returned signatures for this address. Token balance changes are not classified as buys, sells, or profit.</p>
          {wallet && <div className="mt-3 rounded-lg border border-[color:var(--line)] p-3 text-sm text-[color:var(--ink)]">
            <p className="font-semibold">Want coin names from a wider trade history?</p>
            <a href={wallet.profileUrl} target="_blank" rel="noopener noreferrer" className="mt-1 inline-block text-[color:var(--accent)] underline underline-offset-2">Open this wallet on Pump.fun → Activity, Open, or Closed ↗</a>
            <p className="mt-1 text-xs text-[color:var(--ink-faint)]">That profile shows Pump.fun activity; trades through other venues may not appear there.</p>
          </div>}
          {data?.walletActivity.note && <p className="mt-3 text-sm text-amber-300">{data.walletActivity.note}</p>}
          {recentIncreases.length > 0 && <div className="mt-3 rounded-lg border border-[color:var(--line)] p-3">
            <h4 className="text-xs font-semibold uppercase tracking-wide text-[color:var(--ink)]">Recent token increases</h4>
            <p className="mt-1 text-xs text-[color:var(--ink-faint)]">These may be purchases, transfers, or other receipts. Open the transaction to check.</p>
            <ul className="mt-2 space-y-2">
              {recentIncreases.map(({ change, activity }) => <li key={change.mint} className="text-sm text-[color:var(--ink)]">
                <span className="font-semibold">{change.name ?? change.symbol ?? "Unlabeled token"}</span>
                {change.symbol && change.name && <span className="ml-1 text-xs text-[color:var(--ink-faint)]">({change.symbol})</span>}
                <span className="ml-2 text-xs text-[color:var(--ink-faint)]">{change.delta}</span>
                <a href={activity.url} target="_blank" rel="noopener noreferrer" className="ml-2 text-xs text-[color:var(--accent)] underline underline-offset-2">transaction ↗</a>
              </li>)}
            </ul>
          </div>}
          {data && data.walletActivity.status !== "unavailable" && recentIncreases.length === 0 && <p className="mt-3 text-xs text-[color:var(--ink-faint)]">No token balance increases appear in these latest {data.walletActivity.items.length} transactions. This does not rule out earlier purchases.</p>}
          {data?.walletActivity.items.length ? <ol className="mt-3 divide-y divide-[color:var(--line)]">
            {data.walletActivity.items.map((activity) => <li key={activity.signature} className="py-3 first:pt-0">
              <p className="text-xs text-[color:var(--ink-faint)]">{timeLabel(activity.timestamp)} · {activity.status === "confirmed" ? "Confirmed" : activity.status === "failed" ? "Failed transaction" : "Details unavailable"}</p>
              {activity.tokenChanges.length ? <ul className="mt-1 space-y-1">
                {activity.tokenChanges.map((change) => <li key={change.mint} className="break-all text-xs text-[color:var(--ink)]"><span className="font-semibold">{change.delta} {change.symbol ?? "tokens"}</span>{change.name && <span className="ml-1 text-[color:var(--ink-faint)]">· {change.name}</span>} · <a href={`https://explorer.solana.com/address/${change.mint}`} target="_blank" rel="noopener noreferrer" className="text-[color:var(--accent)] underline underline-offset-2">mint {change.mint} ↗</a></li>)}
              </ul> : <p className="mt-1 text-xs text-[color:var(--ink-faint)]">No owner-attributed SPL token balance change in the available transaction metadata.</p>}
              <a href={activity.url} target="_blank" rel="noopener noreferrer" className="mt-1 inline-block text-xs text-[color:var(--accent)] underline underline-offset-2">View transaction {activity.signature.slice(0, 8)}… ↗</a>
            </li>)}
          </ol> : null}
        </div>
      </div>
      <p className="mt-4 text-xs text-[color:var(--ink-faint)]">A single wallet may omit other holdings or off-chain trades. Transfers and deposits can change balances without a purchase or sale.</p>
    </section>
  );
}

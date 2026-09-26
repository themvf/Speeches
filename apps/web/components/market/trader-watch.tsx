"use client";

import { useCallback, useEffect, useState } from "react";
import { TRADER_WATCH_PROFILES, type TraderWatchData } from "@/lib/trader-watch";

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

  const load = useCallback(async (signal?: AbortSignal) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`/api/market/crypto/trader-watch?trader=${encodeURIComponent(trader)}`, { signal });
      const result = await response.json();
      if (!response.ok || !result.ok) throw new Error(result.error || "Trader watch failed to load");
      setData(result.data as TraderWatchData);
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
    return () => controller.abort();
  }, [load]);

  const profile = data?.profile ?? TRADER_WATCH_PROFILES.find((item) => item.id === trader)!;
  const wallet = profile.wallet;

  return (
    <section className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)] p-4 sm:p-5" aria-labelledby="trader-watch-heading">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 id="trader-watch-heading" className="text-base font-semibold text-[color:var(--ink)]">Trader Watch</h2>
          <p className="mt-1 text-sm text-[color:var(--ink-faint)]">Public posts, current wallet holdings, and recent transactions, with separate source records.</p>
        </div>
        <button type="button" onClick={() => void load()} disabled={loading} className="min-h-11 rounded-lg border border-[color:var(--line-strong)] px-3 text-sm text-[color:var(--ink)] disabled:opacity-50">
          {loading ? "Reloading…" : "Reload data"}
        </button>
      </div>

      {TRADER_WATCH_PROFILES.length > 1 && (
        <label className="mt-4 block text-xs text-[color:var(--ink-faint)]">
          Trader
          <select value={trader} onChange={(event) => { setTrader(event.target.value); setData(null); }} className="ml-2 min-h-11 rounded-lg border border-[color:var(--line)] bg-[color:var(--surface)] px-3 text-base text-[color:var(--ink)]">
            {TRADER_WATCH_PROFILES.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
          </select>
        </label>
      )}

      <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
        <span className="font-semibold text-[color:var(--ink)]">{profile.name}</span>
        <span className="text-[color:var(--ink-faint)]">{profile.description}</span>
        <a href={`https://x.com/${profile.xHandle}`} target="_blank" rel="noopener noreferrer" className="text-[color:var(--accent)] underline underline-offset-2">Follow @{profile.xHandle} on X ↗</a>
      </div>
      <p className="mt-1 text-xs text-[color:var(--ink-faint)]">For immediate post alerts, use the account’s notification bell on X and select all posts.</p>
      {wallet && (
        <div className="mt-3 rounded-lg border border-amber-400/30 bg-amber-400/5 p-3 text-sm text-[color:var(--ink)]">
          <p className="font-medium">{wallet.attribution === "verified" ? "Verified wallet" : "Candidate wallet — ownership unverified"}</p>
          <p className="mt-1 text-[color:var(--ink-faint)]">The matching name and challenge description do not prove this address belongs to {profile.name}. Treat its activity as a separate observation.</p>
          <a href={`https://explorer.solana.com/address/${wallet.address}`} target="_blank" rel="noopener noreferrer" className="mt-2 inline-block break-all font-mono text-xs text-[color:var(--accent)] underline underline-offset-2">{wallet.address} ↗</a>
          {wallet.evidenceUrl && <a href={wallet.evidenceUrl} target="_blank" rel="noopener noreferrer" className="ml-3 text-xs text-[color:var(--accent)] underline">Attribution evidence ↗</a>}
        </div>
      )}

      <p className="mt-3 text-xs text-[color:var(--ink-faint)]" role="status">
        {error ? `Update failed: ${error}. Showing the last available result.` : loading && !data ? "Loading posts and wallet activity…" : data ? `Checked ${timeLabel(data.generatedAt)}. Each source has its own availability and timestamp.` : ""}
      </p>

      <div className="mt-4 grid gap-4">
        <div className="min-w-0 rounded-lg border border-[color:var(--line)] p-3">
          <h3 className="text-sm font-semibold text-[color:var(--ink)]">Current holdings in candidate wallet</h3>
          <p className="mt-1 text-xs text-[color:var(--ink-faint)]">Positive balances in this address’s SPL Token and Token-2022 accounts, plus native SOL. Snapshot checked {timeLabel(data?.walletHoldings.observedAt ?? null)}. These holdings do not establish {profile.name}&rsquo;s complete portfolio.</p>
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
          <h3 className="text-sm font-semibold text-[color:var(--ink)]">Public X posts</h3>
          <p className="mt-1 text-xs text-[color:var(--ink-faint)]">Saved posts from the tracked account. Statements are the author’s claims.</p>
          {data?.posts.note && <p className="mt-3 text-sm text-amber-300">{data.posts.note}</p>}
          {data?.posts.items.length ? <ol className="mt-3 divide-y divide-[color:var(--line)]">
            {data.posts.items.map((post) => <li key={post.id} className="py-3 first:pt-0">
              <p className="whitespace-pre-wrap break-words text-sm text-[color:var(--ink)]">{post.text}</p>
              <p className="mt-1 text-xs text-[color:var(--ink-faint)]">Posted {timeLabel(post.publishedAt)} · Saved {timeLabel(post.observedAt)}</p>
              <a href={post.url} target="_blank" rel="noopener noreferrer" className="mt-1 inline-block text-xs text-[color:var(--accent)] underline underline-offset-2">View post ↗</a>
            </li>)}
          </ol> : null}
        </div>

        <div className="min-w-0 rounded-lg border border-[color:var(--line)] p-3">
          <h3 className="text-sm font-semibold text-[color:var(--ink)]">Solana transaction observations</h3>
          <p className="mt-1 text-xs text-[color:var(--ink-faint)]">Latest 8 returned signatures for the candidate address. Token balance changes are not classified as buys, sells, or profit.</p>
          {data?.walletActivity.note && <p className="mt-3 text-sm text-amber-300">{data.walletActivity.note}</p>}
          {data?.walletActivity.items.length ? <ol className="mt-3 divide-y divide-[color:var(--line)]">
            {data.walletActivity.items.map((activity) => <li key={activity.signature} className="py-3 first:pt-0">
              <p className="text-xs text-[color:var(--ink-faint)]">{timeLabel(activity.timestamp)} · {activity.status === "confirmed" ? "Confirmed" : activity.status === "failed" ? "Failed transaction" : "Details unavailable"}</p>
              {activity.tokenChanges.length ? <ul className="mt-1 space-y-1">
                {activity.tokenChanges.map((change) => <li key={change.mint} className="break-all text-xs text-[color:var(--ink)]"><span className="font-semibold">{change.delta}</span> tokens · mint {change.mint}</li>)}
              </ul> : <p className="mt-1 text-xs text-[color:var(--ink-faint)]">No owner-attributed SPL token balance change in the available transaction metadata.</p>}
              <a href={activity.url} target="_blank" rel="noopener noreferrer" className="mt-1 inline-block text-xs text-[color:var(--accent)] underline underline-offset-2">View transaction {activity.signature.slice(0, 8)}… ↗</a>
            </li>)}
          </ol> : null}
        </div>
      </div>
      <p className="mt-4 text-xs text-[color:var(--ink-faint)]">A single wallet may omit other holdings or off-chain trades. Transfers and deposits can change balances without a trade. Compare source timestamps with the price available when a post became public.</p>
    </section>
  );
}

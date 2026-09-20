"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { COINS, coinConfig } from "@/lib/crypto-coins";
import type { BriefEvidence, CryptoBrief, CryptoBriefWindow } from "@/lib/crypto-brief-types";

const WINDOWS: { id: CryptoBriefWindow; label: string }[] = [{ id: "24h", label: "24 hours" }, { id: "7d", label: "7 days" }, { id: "30d", label: "30 days" }];
const button = "min-h-11 rounded-lg border border-[color:var(--line)] px-3 py-2 text-sm font-semibold transition hover:bg-white/5 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-300";
const fmt = (value: string | null) => value ? new Intl.DateTimeFormat("en", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }).format(new Date(value)) : "Unavailable";
const shortAddress = (value: string | null) => value ? `${value.slice(0, 6)}…${value.slice(-4)}` : "Native asset · no contract";

function Source({ evidence }: { evidence: BriefEvidence }) {
  const href = evidence.url && /^https:\/\/(?:www\.|mobile\.)?(?:x|twitter)\.com\//i.test(evidence.url) ? evidence.url : null;
  return <article className="rounded-xl border border-[color:var(--line)] bg-[color:var(--bg)] p-4">
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-[color:var(--ink-faint)]"><strong className="text-[color:var(--ink)]">{evidence.handle ? `@${evidence.handle}` : "Saved account"}</strong><span>{fmt(evidence.publishedAt)}</span><span>{evidence.kind}</span></div>
    <p className="mt-3 whitespace-pre-line break-words text-sm leading-relaxed text-[color:var(--ink-soft)]">{evidence.text}</p>
    <p className="mt-3 text-xs leading-relaxed text-amber-200">Identity: {evidence.match.reason}{evidence.match.provisional ? " · provisional" : ""}</p>
    {href ? <a href={href} target="_blank" rel="noopener noreferrer" className="mt-2 inline-flex min-h-11 items-center text-sm font-semibold text-cyan-300 hover:underline">Read on X ↗</a> : <p className="mt-2 text-xs text-[color:var(--ink-faint)]">Source link unavailable</p>}
  </article>;
}

export function CryptoBriefPanel({ coin, window, onCoin, onWindow }: { coin: string; window: CryptoBriefWindow; onCoin: (coin: string) => void; onWindow: (window: CryptoBriefWindow) => void }) {
  const [brief, setBrief] = useState<CryptoBrief | null>(null), [loading, setLoading] = useState(true), [error, setError] = useState("");
  const [query, setQuery] = useState(() => coinConfig(coin).label), [openEvidence, setOpenEvidence] = useState<string[] | null>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  useEffect(() => setQuery(coinConfig(coin).label), [coin]);
  const load = async (signal?: AbortSignal) => {
    setLoading(true); setError("");
    try {
      const response = await fetch(`/api/market/crypto/brief?coin=${encodeURIComponent(coin)}&window=${window}`, { signal });
      const body = await response.json();
      if (!response.ok || !body.ok) throw new Error(body.error ?? "The brief could not be loaded.");
      setBrief(body.data); setOpenEvidence(null);
    } catch (cause) { if (!signal?.aborted) setError(cause instanceof Error ? cause.message : "The brief could not be loaded."); }
    finally { if (!signal?.aborted) setLoading(false); }
  };
  useEffect(() => { const controller = new AbortController(); void load(controller.signal); return () => controller.abort(); }, [coin, window]);
  const suggestions = useMemo(() => {
    const value = query.trim().toLowerCase();
    if (!value || value === coinConfig(coin).label.toLowerCase()) return [];
    return COINS.filter((item) => [item.symbol, item.name, item.networkLabel, item.address ?? ""].some((field) => field.toLowerCase().includes(value))).slice(0, 6);
  }, [query, coin]);
  const evidenceById = useMemo(() => new Map((brief?.evidence ?? []).map((item) => [item.id, item])), [brief]);
  const activeEvidence = (openEvidence ?? []).map((id) => evidenceById.get(id)).filter((item): item is BriefEvidence => !!item);
  const identity = coinConfig(coin);
  return <div className="min-w-0 overflow-y-auto p-3 md:p-5" aria-busy={loading}>
    <div className="rounded-2xl border border-[color:var(--line)] bg-[color:var(--bg-elev)] p-4">
      <div className="flex flex-wrap items-start justify-between gap-3"><div><p className="text-xs font-semibold uppercase tracking-widest text-purple-300">Quick brief</p><h2 className="mt-1 text-xl font-semibold">Saved crypto conversation</h2></div><button className={button} disabled={loading} onClick={() => void load()}>{loading ? "Loading…" : "Refresh"}</button></div>
      <label htmlFor="crypto-brief-search" className="mt-4 block text-sm font-medium">Tracked asset</label>
      <input ref={searchRef} id="crypto-brief-search" type="search" value={query} onChange={(event) => setQuery(event.target.value)} className="mt-2 min-h-12 w-full rounded-xl border border-[color:var(--line)] bg-[color:var(--bg)] px-3 text-base" placeholder="Name, symbol, network, or contract" autoComplete="off" />
      {suggestions.length ? <div className="mt-1 rounded-xl border border-[color:var(--line)] bg-[color:var(--bg)] p-1">{suggestions.map((item) => <button key={item.symbol} className="block min-h-11 w-full rounded-lg px-3 py-2 text-left text-sm hover:bg-white/5" onClick={() => { onCoin(item.symbol); setQuery(item.label); }}>{item.label} · {item.networkLabel} · {shortAddress(item.address)}</button>)}</div> : null}
      <div className="mt-3 flex flex-wrap gap-2" role="group" aria-label="Brief publication window">{WINDOWS.map((item) => <button key={item.id} className={`${button} ${window === item.id ? "border-purple-400 bg-purple-400/10 text-purple-200" : "text-[color:var(--ink-faint)]"}`} aria-pressed={window === item.id} onClick={() => onWindow(item.id)}>{item.label}</button>)}</div>
      <p className="mt-3 break-words text-xs leading-relaxed text-[color:var(--ink-faint)]"><strong className="text-[color:var(--ink)]">{identity.name} · {identity.symbol}</strong> · {identity.networkLabel} · {shortAddress(identity.address)}<br />{identity.identityNote}</p>
    </div>
    {error ? <div role="alert" className="mt-4 rounded-xl border border-amber-400/30 bg-amber-400/5 p-4 text-sm text-amber-200">{error}{brief ? " Showing the last successful brief." : ""}</div> : null}
    <div className="mt-4" aria-live="polite">{loading && !brief ? <div role="status" className="rounded-xl border border-[color:var(--line)] p-5 text-sm">Reading saved posts…</div> : brief ? <>
      <section className="rounded-2xl border border-purple-400/25 bg-gradient-to-br from-purple-400/10 to-[color:var(--bg-elev)] p-4">
        <div className="flex flex-wrap items-center justify-between gap-2"><h3 className="text-lg font-semibold">{brief.asset.name} at a glance</h3><span className="rounded-full bg-purple-300/10 px-2.5 py-1 text-xs text-purple-200">{WINDOWS.find((item) => item.id === window)?.label}</span></div>
        <p className="mt-3 text-sm leading-relaxed text-[color:var(--ink-soft)]">{brief.overview}</p>
        <p className="mt-2 text-xs text-[color:var(--ink-faint)]">Latest evidence: {fmt(brief.coverage.latestPublishedAt)} · Coverage: {brief.coverage.status}</p>
        {brief.coverage.reasons.length ? <ul className="mt-2 list-disc pl-5 text-xs leading-relaxed text-amber-200">{brief.coverage.reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul> : null}
        <ol className="mt-4 divide-y divide-purple-300/15">{brief.takeaways.map((takeaway, index) => <li key={takeaway.id} className="py-4 first:pt-0 last:pb-0">
          <p className="text-xs font-semibold uppercase tracking-wider text-purple-200">{index + 1} · {takeaway.category.replace("_", " ")} · post excerpt</p><p className="mt-2 break-words text-base font-medium leading-relaxed">{takeaway.text}</p>
          <button className={`${button} mt-3`} onClick={() => setOpenEvidence(takeaway.evidenceIds)}>{takeaway.evidenceIds.length} supporting post{takeaway.evidenceIds.length === 1 ? "" : "s"}</button>
        </li>)}</ol>
        {!brief.takeaways.length ? <p className="mt-4 text-sm text-[color:var(--ink-soft)]">No clear development in the eligible saved posts. Try a wider window; the archive is never widened silently.</p> : null}
      </section>
      {openEvidence ? <section className="mt-4 rounded-2xl border border-[color:var(--line)] bg-[color:var(--bg-elev)] p-4" aria-labelledby="brief-evidence-title"><div className="flex items-center justify-between gap-3"><h3 id="brief-evidence-title" className="font-semibold">Supporting evidence</h3><button className={button} onClick={() => { setOpenEvidence(null); searchRef.current?.focus(); }}>Close</button></div><div className="mt-3 space-y-3">{activeEvidence.map((item) => <Source key={item.id} evidence={item} />)}</div></section> : null}
      <details className="mt-4 rounded-2xl border border-[color:var(--line)] bg-[color:var(--bg-elev)]"><summary className="min-h-12 cursor-pointer px-4 py-4 text-sm font-semibold">All included evidence ({brief.evidence.length}{brief.coverage.truncated ? "+" : ""})</summary><div className="space-y-3 px-4 pb-4">{brief.evidence.map((item) => <Source key={item.id} evidence={item} />)}</div></details>
      <p className="mt-3 text-xs leading-relaxed text-[color:var(--ink-faint)]">Based on saved posts, not all of X. Identity reasons are shown per source. This is evidence context, not a trade recommendation or a claim that posts caused a market move.</p>
    </> : null}</div>
  </div>;
}

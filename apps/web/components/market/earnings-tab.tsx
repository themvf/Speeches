"use client";

import { useEffect, useState } from "react";
import type { EarningsWeekCompany, MarketEarningsWeekData } from "@/lib/server/types";
import { TickerEventChart } from "./ticker-event-chart";

interface Props {
  data: MarketEarningsWeekData | null;
  loading: boolean;
  error: string | null;
}

const ARCHETYPE_COLORS: Record<string, string> = {
  early_sharp: "#41d39d",
  longshot: "#a78bfa",
};

function dayLabel(iso: string): string {
  const d = new Date(`${iso}T00:00:00Z`);
  return d.toLocaleDateString("en-US", { weekday: "long", month: "short", day: "numeric", timeZone: "UTC" });
}

function Sparkline({ series }: { series: { end: string; value: number }[] }) {
  if (series.length < 2) return <span className="text-[10px] text-[color:var(--ink-faint)]">—</span>;
  const w = 90;
  const h = 26;
  const values = series.map((p) => p.value);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo || 1;
  const points = values
    .map((v, i) => `${((i / (values.length - 1)) * w).toFixed(1)},${(h - ((v - lo) / span) * h).toFixed(1)}`)
    .join(" ");
  const up = values[values.length - 1]! >= values[0]!;
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} className="inline-block overflow-visible">
      <polyline points={points} fill="none" stroke={up ? "#41d39d" : "#f87171"} strokeWidth="1.5" opacity="0.9" />
    </svg>
  );
}

// Lazy headlines via the existing per-symbol company-news endpoint; not
// every reporting ticker is in the sector-company catalog, so 4xx/errors
// just hide the section.
function Headlines({ ticker }: { ticker: string }) {
  const [articles, setArticles] = useState<{ title: string; url: string; publisher?: string }[] | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch(`/api/market/company-news?symbol=${encodeURIComponent(ticker)}`)
      .then((r) => r.json())
      .then((env) => {
        if (cancelled) return;
        if (env.ok && env.data?.articles) setArticles(env.data.articles.slice(0, 5));
        else setFailed(true);
      })
      .catch(() => { if (!cancelled) setFailed(true); });
    return () => { cancelled = true; };
  }, [ticker]);

  if (failed) return null;
  if (!articles) return <p className="px-4 pb-2 text-[10px] text-[color:var(--ink-faint)]">Loading headlines…</p>;
  if (articles.length === 0) return null;
  return (
    <div className="px-4 pb-3">
      <p className="mb-1 text-[10px] font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">Headlines</p>
      <ul className="space-y-1">
        {articles.map((a, i) => (
          <li key={i} className="truncate text-xs">
            <a href={a.url} target="_blank" rel="noopener noreferrer" className="text-[color:var(--ink-soft)] hover:text-[color:var(--accent)] hover:underline">
              {a.title}
            </a>
            {a.publisher && <span className="ml-1.5 text-[10px] text-[color:var(--ink-faint)]">{a.publisher}</span>}
          </li>
        ))}
      </ul>
    </div>
  );
}

function CompanyCard({ company }: { company: EarningsWeekCompany }) {
  const [open, setOpen] = useState(false);
  const { yes, no } = company.consensus;
  const total = yes + no;
  const consensusColor = no > yes ? "#f87171" : yes > no ? "#41d39d" : "var(--ink-faint)";
  const delta = company.mentions != null && company.mentionsPrev != null && company.mentionsPrev > 0
    ? ((company.mentions - company.mentionsPrev) / company.mentionsPrev) * 100
    : null;

  return (
    <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)]">
      <button type="button" onClick={() => setOpen(!open)} className="block w-full px-4 py-3 text-left hover:bg-[color:rgba(79,213,255,0.04)]">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <span className="w-14 text-sm font-bold text-[color:var(--accent)]">{company.ticker}</span>
          <span className="text-xs text-[color:var(--ink-faint)]">EPS line {company.eps ? `$${company.eps}` : "—"}</span>
          <span className="text-xs tabular-nums">
            <span className="text-[color:var(--ink-faint)]">implied beat </span>
            <span className="font-semibold text-[color:var(--ink)]">
              {company.impliedProbYes == null ? "—" : `${Math.round(company.impliedProbYes * 100)}%`}
            </span>
          </span>
          <span className="text-xs">
            {total > 0 ? (
              <span
                className="rounded px-1.5 py-0.5 text-[10px] font-semibold tabular-nums"
                style={{ color: consensusColor, backgroundColor: `color-mix(in srgb, ${consensusColor} 14%, transparent)` }}
                title="Sharp-money consensus (early sharps + longshots only)"
              >
                {no} No / {yes} Yes
              </span>
            ) : (
              <span className="text-[10px] text-[color:var(--ink-faint)]">no sharp positions</span>
            )}
          </span>
          <span className="flex items-center gap-1.5 text-xs" title={company.kpiLabel ? `${company.kpiLabel} - trailing quarters (CBOE KPI snapshot)` : "Not covered by the CBOE KPI set"}>
            <span className="text-[10px] text-[color:var(--ink-faint)]">{company.kpiLabel ?? "KPI"}</span>
            <Sparkline series={company.kpiSeries} />
          </span>
          <span className="text-xs tabular-nums" title="Reddit mentions, latest attention day (Δ vs the day before)">
            <span className="text-[color:var(--ink-faint)]">Reddit </span>
            {company.mentions == null ? "—" : company.mentions}
            {delta != null && (
              <span className="ml-1 font-semibold" style={{ color: delta >= 0 ? "#41d39d" : "#f87171" }}>
                {delta >= 0 ? "▲" : "▼"}{Math.abs(delta).toFixed(0)}%
              </span>
            )}
          </span>
          <span className="ml-auto text-[10px] text-[color:var(--ink-faint)]">{open ? "▾ hide" : "▸ chart + news"}</span>
        </div>
        {company.consensus.wallets.length > 0 && (
          <p className="mt-1.5 text-[10px] text-[color:var(--ink-faint)]">
            {company.consensus.wallets.slice(0, 4).map((w, i) => (
              <span key={w.wallet}>
                {i > 0 && " · "}
                <span style={{ color: ARCHETYPE_COLORS[w.archetype] ?? "var(--ink-soft)" }}>{w.name}</span>{" "}
                <span style={{ color: w.side === "No" ? "#f87171" : "#41d39d" }}>{w.side}</span> {w.shares.toLocaleString()}
              </span>
            ))}
          </p>
        )}
      </button>
      {open && (
        <div className="border-t border-[color:var(--line)]">
          <TickerEventChart ticker={company.ticker} />
          <Headlines ticker={company.ticker} />
        </div>
      )}
    </div>
  );
}

export function EarningsTab({ data, loading, error }: Props) {
  if (loading && !data) {
    return <div className="flex items-center justify-center py-16 text-sm text-[color:var(--ink-faint)]">Loading earnings week…</div>;
  }
  if (error && !data) {
    return <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-4 text-sm text-red-400">{error}</div>;
  }
  if (!data) return null;

  const byDay = new Map<string, EarningsWeekCompany[]>();
  for (const company of data.companies) {
    const list = byDay.get(company.reportDate) ?? [];
    list.push(company);
    byDay.set(company.reportDate, list);
  }

  return (
    <div className="space-y-5">
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
          Earnings Week · implied odds, sharp money, KPIs, attention
        </p>
        <p className="mt-0.5 text-[10px] text-[color:var(--ink-faint)]">
          Research context only — not investment advice.
        </p>
      </div>

      {data.warning && (
        <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-3 text-xs text-amber-300">{data.warning}</div>
      )}

      <p className="max-w-3xl text-xs text-[color:var(--ink-faint)]">
        Every company with an open &quot;beat quarterly earnings?&quot; market reporting between {data.windowStart} and{" "}
        {data.windowEnd}: Polymarket&apos;s implied beat probability, the sharp-money consensus (early sharps +
        longshots), the CBOE KPI trend where covered, and Reddit attention with a day-over-day delta. Expand a card for
        the event-annotated price chart and latest headlines.
      </p>

      {data.companies.length === 0 && (
        <p className="py-10 text-center text-sm text-[color:var(--ink-faint)]">No earnings markets in the window.</p>
      )}

      {[...byDay.entries()].map(([date, companies]) => (
        <section key={date}>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-soft)]">
            {dayLabel(date)}
          </h3>
          <div className="space-y-2">
            {companies.map((company) => (
              <CompanyCard key={company.ticker + company.reportDate} company={company} />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

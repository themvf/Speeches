"use client";

import { Fragment, useEffect, useState } from "react";
import type { IndustrySummary, MarketIndustriesData } from "@/lib/server/types";

interface Props {
  data: MarketIndustriesData | null;
  loading: boolean;
  error: string | null;
}

function reportLabel(iso: string | null): string {
  if (!iso) return "";
  const d = new Date(`${iso}T00:00:00Z`);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
}

// Peer table for one expanded industry - lazy-fetched so quotes only load
// for the industry actually being looked at (SEC-53 budget rule).
function PeerTable({ label }: { label: string }) {
  const [data, setData] = useState<MarketIndustriesData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetch(`/api/market/industries?industry=${encodeURIComponent(label)}`)
      .then((r) => r.json())
      .then((env) => {
        if (!cancelled && env.ok && env.data) setData(env.data);
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [label]);

  if (loading) return <p className="px-4 py-3 text-xs text-[color:var(--ink-faint)]">Loading peers…</p>;
  const rows = data?.peers?.rows ?? [];
  if (rows.length === 0) return <p className="px-4 py-3 text-xs text-[color:var(--ink-faint)]">No peer data.</p>;

  return (
    <div className="px-4 py-3">
      <table className="w-full">
        <thead>
          <tr className="border-b border-[color:var(--line)] text-[10px] uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
            <th className="py-1.5 pr-2 text-left font-semibold">Ticker</th>
            <th className="hidden px-2 py-1.5 text-left font-semibold sm:table-cell">Company</th>
            <th className="px-2 py-1.5 text-right font-semibold">Price</th>
            <th className="px-2 py-1.5 text-right font-semibold">Δ Today</th>
            <th className="px-2 py-1.5 text-right font-semibold">Mentions</th>
            <th className="py-1.5 pl-2 text-right font-semibold">Reports</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const priceColor = (row.pricePct ?? 0) >= 0 ? "#41d39d" : "#f87171";
            return (
              <tr key={row.ticker} className="border-b border-[color:var(--line)] text-xs last:border-0">
                <td className="py-2 pr-2 font-bold text-[color:var(--accent)]">{row.ticker}</td>
                <td className="hidden max-w-[220px] truncate px-2 py-2 text-[color:var(--ink-faint)] sm:table-cell">{row.name}</td>
                <td className="px-2 py-2 text-right tabular-nums text-[color:var(--ink)]">
                  {row.price == null ? "—" : `$${row.price.toLocaleString(undefined, { maximumFractionDigits: 2 })}`}
                </td>
                <td className="px-2 py-2 text-right tabular-nums">
                  {row.pricePct == null ? (
                    <span className="text-[color:var(--ink-faint)]">—</span>
                  ) : (
                    <span className="font-semibold" style={{ color: priceColor }}>
                      {row.pricePct >= 0 ? "+" : ""}{row.pricePct.toFixed(2)}%
                    </span>
                  )}
                </td>
                <td className="px-2 py-2 text-right tabular-nums text-[color:var(--ink-faint)]">
                  {row.mentions > 0 ? row.mentions : "—"}
                </td>
                <td className="py-2 pl-2 text-right text-[color:var(--ink-soft)]">{reportLabel(row.reportDate) || "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function IndustryRow({ industry, open, onToggle }: { industry: IndustrySummary; open: boolean; onToggle: () => void }) {
  const preview = industry.tickers.slice(0, 7);
  return (
    <>
      <tr
        onClick={onToggle}
        className="cursor-pointer border-b border-[color:var(--line)] last:border-0 hover:bg-[color:rgba(79,213,255,0.04)]"
      >
        <td className="max-w-[240px] py-2.5 pl-4 pr-2 text-xs font-medium text-[color:var(--ink)]">
          <span className="mr-1.5 inline-block w-3 text-[color:var(--ink-faint)]">{open ? "▾" : "▸"}</span>
          {industry.label}
        </td>
        <td className="px-2 py-2.5 text-right text-xs tabular-nums text-[color:var(--ink-faint)]">{industry.tickers.length}</td>
        <td className="hidden px-2 py-2.5 text-xs text-[color:var(--ink-faint)] md:table-cell">
          {preview.join(" · ")}{industry.tickers.length > preview.length ? " …" : ""}
        </td>
        <td className="px-2 py-2.5 text-right text-xs tabular-nums text-[color:var(--ink)]">
          {industry.attentionTotal > 0 ? industry.attentionTotal : "—"}
        </td>
        <td className="py-2.5 pl-2 pr-4 text-right text-xs">
          {industry.reportingSoon.length > 0 ? (
            <span
              className="rounded bg-[rgba(79,213,255,0.1)] px-1.5 py-0.5 text-[10px] font-semibold text-[color:var(--accent)]"
              title={industry.reportingSoon.map((r) => `${r.ticker} ${r.reportDate}`).join(", ")}
            >
              {industry.reportingSoon.length} reporting
            </span>
          ) : (
            <span className="text-[color:var(--ink-faint)]">—</span>
          )}
        </td>
      </tr>
      {open && (
        <tr className="border-b border-[color:var(--line)] bg-[color:rgba(9,21,34,0.5)] last:border-0">
          <td colSpan={5}>
            <PeerTable label={industry.label} />
          </td>
        </tr>
      )}
    </>
  );
}

export function IndustriesTab({ data, loading, error }: Props) {
  const [expanded, setExpanded] = useState<string | null>(null);

  if (loading && !data) {
    return <div className="flex items-center justify-center py-16 text-sm text-[color:var(--ink-faint)]">Loading industries…</div>;
  }
  if (error && !data) {
    return <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-4 text-sm text-red-400">{error}</div>;
  }
  if (!data) return null;

  return (
    <div className="space-y-4">
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
          Industry Groups · SEC SIC classification
        </p>
        <p className="mt-0.5 text-[10px] text-[color:var(--ink-faint)]">
          Research context only — not investment advice.
        </p>
      </div>

      {data.warning && (
        <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-3 text-xs text-amber-300">
          {data.warning}
        </div>
      )}

      <p className="max-w-3xl text-xs text-[color:var(--ink-faint)]">
        A curated universe grouped into industries by each company&apos;s SEC-registered SIC classification.
        Mentions come from the Reddit attention rollup; the reporting chip marks members with an open earnings
        market this cycle. Click an industry to load its live peer table.
      </p>

      <div className="overflow-hidden rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)]">
        <table className="w-full">
          <thead>
            <tr className="border-b border-[color:var(--line)] text-[10px] uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
              <th className="py-2 pl-4 pr-2 text-left font-semibold">Industry</th>
              <th className="px-2 py-2 text-right font-semibold">Members</th>
              <th className="hidden px-2 py-2 text-left font-semibold md:table-cell">Tickers</th>
              <th className="px-2 py-2 text-right font-semibold">Mentions</th>
              <th className="py-2 pl-2 pr-4 text-right font-semibold">Earnings</th>
            </tr>
          </thead>
          <tbody>
            {data.industries.map((industry) => (
              <Fragment key={industry.label}>
                <IndustryRow
                  industry={industry}
                  open={expanded === industry.label}
                  onToggle={() => setExpanded(expanded === industry.label ? null : industry.label)}
                />
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

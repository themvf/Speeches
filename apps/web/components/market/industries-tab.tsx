"use client";

import { Fragment, useEffect, useState } from "react";
import type { IndustrySummary, MarketIndustriesData } from "@/lib/server/types";
import { TickerEventChart } from "./ticker-event-chart";

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

// Compact USD for the financial columns; negatives (loss-making peers) keep
// their sign so a red -$1.2B reads correctly next to profitable peers.
function usdCompact(value: number | null): string {
  if (value == null) return "—";
  const abs = Math.abs(value);
  const sign = value < 0 ? "-" : "";
  if (abs >= 1e12) return `${sign}$${(abs / 1e12).toFixed(2)}T`;
  if (abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(0)}M`;
  return `${sign}$${abs.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

// Peer table for one expanded industry - lazy-fetched so quotes only load
// for the industry actually being looked at (SEC-53 budget rule).
function PeerTable({ label }: { label: string }) {
  const [data, setData] = useState<MarketIndustriesData | null>(null);
  const [loading, setLoading] = useState(true);
  // SEC-51: clicking a peer's ticker toggles its event-annotated chart.
  const [chartTicker, setChartTicker] = useState<string | null>(null);

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
  const period = rows.find((r) => r.periodEnd)?.periodEnd ?? null;

  return (
    <div className="px-4 py-3">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[760px]">
          <thead>
            <tr className="border-b border-[color:var(--line)] text-[10px] uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
              <th className="py-1.5 pr-2 text-left font-semibold">Ticker</th>
              <th className="hidden px-2 py-1.5 text-left font-semibold lg:table-cell">Company</th>
              <th className="px-2 py-1.5 text-right font-semibold">Price</th>
              <th className="px-2 py-1.5 text-right font-semibold">Δ Today</th>
              <th className="px-2 py-1.5 text-right font-semibold">Market cap</th>
              <th className="px-2 py-1.5 text-right font-semibold">Revenue</th>
              <th className="px-2 py-1.5 text-right font-semibold">Expenses</th>
              <th className="px-2 py-1.5 text-right font-semibold">Profit</th>
              <th className="hidden px-2 py-1.5 text-right font-semibold sm:table-cell">Mentions</th>
              <th className="py-1.5 pl-2 text-right font-semibold">Reports</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const priceColor = (row.pricePct ?? 0) >= 0 ? "#41d39d" : "#f87171";
              return (
                <Fragment key={row.ticker}>
                <tr className="border-b border-[color:var(--line)] text-xs last:border-0">
                  <td className="py-2 pr-2">
                    <button
                      type="button"
                      onClick={() => setChartTicker(chartTicker === row.ticker ? null : row.ticker)}
                      className="font-bold text-[color:var(--accent)] hover:underline"
                      title={`${chartTicker === row.ticker ? "Hide" : "Show"} ${row.ticker} price chart with filing/earnings/attention events`}
                    >
                      {row.ticker}
                    </button>
                  </td>
                  <td className="hidden max-w-[190px] truncate px-2 py-2 text-[color:var(--ink-faint)] lg:table-cell">{row.name}</td>
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
                  <td className="px-2 py-2 text-right tabular-nums text-[color:var(--ink)]">{usdCompact(row.marketCap)}</td>
                  <td className="px-2 py-2 text-right tabular-nums text-[color:var(--ink-soft)]">{usdCompact(row.revenue)}</td>
                  <td className="px-2 py-2 text-right tabular-nums text-[color:var(--ink-faint)]">{usdCompact(row.expenses)}</td>
                  <td
                    className="px-2 py-2 text-right font-semibold tabular-nums"
                    style={{ color: row.profit == null ? undefined : row.profit >= 0 ? "#41d39d" : "#f87171" }}
                  >
                    {usdCompact(row.profit)}
                  </td>
                  <td className="hidden px-2 py-2 text-right tabular-nums text-[color:var(--ink-faint)] sm:table-cell">
                    {row.mentions > 0 ? row.mentions : "—"}
                  </td>
                  <td className="py-2 pl-2 text-right text-[color:var(--ink-soft)]">{reportLabel(row.reportDate) || "—"}</td>
                </tr>
                {chartTicker === row.ticker && (
                  <tr className="border-b border-[color:var(--line)] bg-[color:rgba(9,21,34,0.6)] last:border-0">
                    <td colSpan={10}>
                      <TickerEventChart ticker={row.ticker} />
                    </td>
                  </tr>
                )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-[10px] leading-relaxed text-[color:var(--ink-faint)]">
        Market cap = shares outstanding × live price. Revenue and profit are the latest quarter reported to SEC XBRL
        {period ? ` (most recent here ends ${period})` : ""}; fiscal calendars differ, so quarters aren&apos;t always
        aligned across peers. Expenses = revenue − profit (total cost including tax) — derived rather than taken from a
        filed cost line, because filers tag those inconsistently and a peer column needs one shared definition.
      </p>
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

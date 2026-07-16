"use client";

import { Fragment, useMemo, useState } from "react";
import type {
  MarketMacroPredictionsData,
  MarketPredictionsData,
  PredictionArchetype,
  PredictionCalendarRow,
  PredictionClosedMarket,
  PredictionWallet,
} from "@/lib/server/types";
import { MacroPredictionMarketsView } from "./macro-prediction-panel";

interface Props {
  data: MarketPredictionsData | null;
  loading: boolean;
  error: string | null;
  macro: {
    data: MarketMacroPredictionsData | null;
    loading: boolean;
    error: string | null;
  };
}

// Reuses this page's existing attention-tab hues so the archetype vocabulary
// stays consistent: green = the trustworthy signal, amber = caution/discount,
// purple = high-variance, faint = unhighlighted.
const ARCHETYPE_STYLES: Record<PredictionArchetype, { label: string; color: string; blurb: (w: PredictionWallet) => string }> = {
  early_sharp: {
    label: "Early sharp",
    color: "#41d39d",
    blurb: (w) =>
      `Buys eventual winners while they're still cheap (avg entry ${w.avgWinnerEntry != null ? Math.round(w.avgWinnerEntry * 100) + "¢" : "—"}) — informed conviction, not hindsight. These are the follow targets.`,
  },
  news_scalper: {
    label: "News scalper",
    color: "#fbbf24",
    blurb: (w) =>
      `${Math.round(w.winRate * 100)}% win rate but buys late (avg entry ${w.avgWinnerEntry != null ? Math.round(w.avgWinnerEntry * 100) + "¢" : "—"}) — trading the print after it's public, not predicting it. Excluded from the sharp-money consensus.`,
  },
  longshot: {
    label: "Longshot",
    color: "#a78bfa",
    blurb: (w) =>
      `Wrong more often than not (${Math.round(w.winRate * 100)}% win) but paid off big when right (ROI ${w.roi != null ? w.roi.toFixed(2) : "—"}) — directional, high-variance.`,
  },
  unclassified: {
    label: "—",
    color: "var(--ink-faint)",
    blurb: () => "Qualifies on volume but doesn't fit a defined pattern.",
  },
};

const FILTERS: { id: PredictionArchetype | "all"; label: string }[] = [
  { id: "all", label: "All" },
  { id: "early_sharp", label: "Early sharps" },
  { id: "news_scalper", label: "News scalpers" },
  { id: "longshot", label: "Longshots" },
];

function ArchetypeBadge({ archetype }: { archetype: PredictionArchetype }) {
  const style = ARCHETYPE_STYLES[archetype];
  if (archetype === "unclassified") {
    return <span className="text-[10px] text-[color:var(--ink-faint)]">—</span>;
  }
  return (
    <span
      className="rounded px-1.5 py-0.5 text-[10px] font-semibold"
      style={{ color: style.color, backgroundColor: `color-mix(in srgb, ${style.color} 14%, transparent)` }}
    >
      {style.label}
    </span>
  );
}

function usd(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1000) return `${value < 0 ? "-" : ""}$${(abs / 1000).toFixed(1)}k`;
  return `${value < 0 ? "-" : ""}$${abs.toFixed(0)}`;
}

function reportLabel(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(`${iso}T00:00:00Z`);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
}

function ConsensusCell({ row }: { row: PredictionCalendarRow }) {
  const { yes, no } = row.consensus;
  const total = yes + no;
  if (total === 0) {
    return <span className="text-[10px] text-[color:var(--ink-faint)]">no tracked positions</span>;
  }
  // Lean No = sharps expect a miss (danger tint); lean Yes = expect a beat (ok
  // tint); split = neutral.
  const color = no > yes ? "#f87171" : yes > no ? "#41d39d" : "var(--ink-faint)";
  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        className="rounded px-1.5 py-0.5 text-[10px] font-semibold tabular-nums"
        style={{ color, backgroundColor: `color-mix(in srgb, ${color} 14%, transparent)` }}
      >
        {no}&nbsp;No&nbsp;/&nbsp;{yes}&nbsp;Yes
      </span>
      <span className="text-[10px] text-[color:var(--ink-faint)]">{total} sharp{total === 1 ? "" : "s"}</span>
    </span>
  );
}

function ImpliedBeatExplainer() {
  return (
    <details className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.35)] px-3 py-2">
      <summary className="cursor-pointer text-[11px] font-medium text-[color:var(--ink-soft)]">
        What does &quot;implied beat&quot; mean?
      </summary>
      <div className="mt-2 space-y-1.5 text-[11px] leading-relaxed text-[color:var(--ink-faint)]">
        <p>
          Each market trades a <span className="text-[color:var(--ink-soft)]">Yes</span> share that pays out{" "}
          <span className="text-[color:var(--ink-soft)]">$1</span> if the company beats consensus EPS and{" "}
          <span className="text-[color:var(--ink-soft)]">$0</span> if it doesn&apos;t. So a Yes share trading at 71¢
          means the market is collectively pricing a <span className="text-[color:var(--ink-soft)]">71% chance of a
          beat</span> — buyers are willing to risk 71¢ to make 29¢, sellers the reverse, and the price is where that
          balances. <span className="text-[color:var(--ink-soft)]">Implied beat = the current Yes price</span>, shown
          as a percent.
        </p>
        <p>
          It moves in real time as traders take positions, so it&apos;s a live, money-weighted probability — not a
          survey or a model. The sharp-money column is the useful contrast: when tracked sharps lean the opposite way
          from the implied probability, that&apos;s informed money disagreeing with the crowd. (Not a forecast, and
          it carries no view on whether a beat is priced in vs. the stock reaction — research context only.)
        </p>
      </div>
    </details>
  );
}

function CalendarView({ rows }: { rows: PredictionCalendarRow[] }) {
  const [expanded, setExpanded] = useState<string | null>(null);
  // Chronological by report date, soonest first; undated markets sink to the
  // bottom. Volume breaks ties so the busiest market leads a given day.
  const sorted = useMemo(
    () =>
      [...rows].sort((a, b) => {
        const da = a.reportDate ?? "9999-12-31";
        const db = b.reportDate ?? "9999-12-31";
        if (da !== db) return da < db ? -1 : 1;
        return b.volume - a.volume;
      }),
    [rows]
  );
  if (rows.length === 0) {
    return <p className="py-8 text-center text-sm text-[color:var(--ink-faint)]">No open earnings markets in the snapshot.</p>;
  }
  return (
    <div className="space-y-3">
      <ImpliedBeatExplainer />
      <div className="overflow-hidden rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)]">
        <table className="w-full">
          <thead>
            <tr className="border-b border-[color:var(--line)] text-[10px] uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
              <th className="py-2 pl-4 pr-2 text-left font-semibold">Ticker</th>
              <th className="px-2 py-2 text-left font-semibold">Reports</th>
              <th className="hidden px-2 py-2 text-right font-semibold sm:table-cell">EPS line</th>
              <th className="px-2 py-2 text-right font-semibold">Implied beat</th>
              <th className="hidden px-2 py-2 text-right font-semibold md:table-cell">Volume</th>
              <th className="py-2 pl-2 pr-4 text-left font-semibold">Sharp money</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((row) => {
            const open = expanded === row.conditionId;
            const hasWallets = row.consensus.wallets.length > 0;
            return (
              <FragmentRow
                key={row.conditionId}
                row={row}
                open={open}
                hasWallets={hasWallets}
                onToggle={() => setExpanded(open ? null : hasWallets ? row.conditionId : null)}
              />
            );
          })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function FragmentRow({ row, open, hasWallets, onToggle }: { row: PredictionCalendarRow; open: boolean; hasWallets: boolean; onToggle: () => void }) {
  const implied = row.impliedProbYes;
  return (
    <>
      <tr
        onClick={onToggle}
        className={`border-b border-[color:var(--line)] last:border-0 ${hasWallets ? "cursor-pointer hover:bg-[color:rgba(79,213,255,0.04)]" : ""}`}
      >
        <td className="py-2.5 pl-4 pr-2 text-xs font-bold text-[color:var(--accent)]">{row.ticker || "—"}</td>
        <td className="px-2 py-2.5 text-xs text-[color:var(--ink-soft)]">{reportLabel(row.reportDate)}</td>
        <td className="hidden px-2 py-2.5 text-right text-xs tabular-nums text-[color:var(--ink-faint)] sm:table-cell">
          {row.eps ? `$${row.eps}` : "—"}
        </td>
        <td className="px-2 py-2.5 text-right text-xs tabular-nums text-[color:var(--ink)]">
          {implied == null ? "—" : `${Math.round(implied * 100)}%`}
        </td>
        <td className="hidden px-2 py-2.5 text-right text-xs tabular-nums text-[color:var(--ink-faint)] md:table-cell">
          {usd(row.volume)}
        </td>
        <td className="py-2.5 pl-2 pr-4 text-left"><ConsensusCell row={row} /></td>
      </tr>
      {open && hasWallets && (
        <tr className="border-b border-[color:var(--line)] bg-[color:rgba(9,21,34,0.5)] last:border-0">
          <td colSpan={6} className="px-4 py-3">
            <p className="mb-2 text-[10px] uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
              Tracked sharp positions · {row.question}
            </p>
            <ul className="space-y-1.5">
              {row.consensus.wallets.map((w) => (
                <li key={w.wallet} className="flex items-center gap-2 text-xs">
                  <a
                    href={`https://polymarket.com/profile/${w.wallet}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-medium text-[color:var(--ink-soft)] hover:text-[color:var(--accent)] hover:underline"
                  >
                    {w.name}
                  </a>
                  <ArchetypeBadge archetype={w.archetype} />
                  <span
                    className="font-semibold tabular-nums"
                    style={{ color: w.side === "No" ? "#f87171" : "#41d39d" }}
                  >
                    {w.side}
                  </span>
                  <span className="tabular-nums text-[color:var(--ink-faint)]">{w.shares.toLocaleString()} shares</span>
                </li>
              ))}
            </ul>
          </td>
        </tr>
      )}
    </>
  );
}

function WalletsView({ wallets }: { wallets: PredictionWallet[] }) {
  const [filter, setFilter] = useState<PredictionArchetype | "all">("all");
  const [expanded, setExpanded] = useState<string | null>(null);
  const filtered = useMemo(
    () => (filter === "all" ? wallets : wallets.filter((w) => w.archetype === filter)),
    [wallets, filter]
  );

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-1">
        {FILTERS.map((f) => (
          <button
            key={f.id}
            type="button"
            onClick={() => setFilter(f.id)}
            className={`rounded-lg px-2.5 py-1 text-[11px] font-medium ${filter === f.id ? "bg-[rgba(79,213,255,0.12)] text-[color:var(--ink)]" : "text-[color:var(--ink-faint)] hover:text-[color:var(--ink-soft)]"}`}
          >
            {f.label}
          </button>
        ))}
      </div>

      <div className="overflow-hidden rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)]">
        <table className="w-full">
          <thead>
            <tr className="border-b border-[color:var(--line)] text-[10px] uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
              <th className="py-2 pl-4 pr-2 text-left font-semibold">Wallet</th>
              <th className="px-2 py-2 text-left font-semibold">Archetype</th>
              <th className="px-2 py-2 text-right font-semibold">Mkts</th>
              <th className="px-2 py-2 text-right font-semibold">Win</th>
              <th className="px-2 py-2 text-right font-semibold">PnL</th>
              <th className="hidden py-2 pl-2 pr-4 text-right font-semibold sm:table-cell">Entry</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((w) => {
              const open = expanded === w.wallet;
              return (
                <Fragment key={w.wallet}>
                  <tr
                    onClick={() => setExpanded(open ? null : w.wallet)}
                    className="cursor-pointer border-b border-[color:var(--line)] last:border-0 hover:bg-[color:rgba(79,213,255,0.04)]"
                  >
                    <td className="py-2 pl-4 pr-2 text-xs font-medium text-[color:var(--ink-soft)]">{w.name}</td>
                    <td className="px-2 py-2"><ArchetypeBadge archetype={w.archetype} /></td>
                    <td className="px-2 py-2 text-right text-xs tabular-nums text-[color:var(--ink-faint)]">{w.markets}</td>
                    <td className="px-2 py-2 text-right text-xs tabular-nums text-[color:var(--ink)]">{Math.round(w.winRate * 100)}%</td>
                    <td className="px-2 py-2 text-right text-xs font-semibold tabular-nums" style={{ color: w.pnlUsd >= 0 ? "#41d39d" : "#f87171" }}>
                      {usd(w.pnlUsd)}
                    </td>
                    <td className="hidden py-2 pl-2 pr-4 text-right text-xs tabular-nums text-[color:var(--ink-faint)] sm:table-cell">
                      {w.avgWinnerEntry != null ? w.avgWinnerEntry.toFixed(2) : "—"}
                    </td>
                  </tr>
                  {open && (
                    <tr className="border-b border-[color:var(--line)] bg-[color:rgba(9,21,34,0.5)] last:border-0">
                      <td colSpan={6} className="px-4 py-3">
                        <p className="text-xs text-[color:var(--ink-soft)]">{ARCHETYPE_STYLES[w.archetype].blurb(w)}</p>
                        <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[10px] text-[color:var(--ink-faint)]">
                          <span>{w.wins}/{w.markets} markets won</span>
                          {w.roi != null && <span>ROI {w.roi.toFixed(2)}</span>}
                          <a
                            href={`https://polymarket.com/profile/${w.wallet}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-[color:var(--ink-faint)] underline hover:text-[color:var(--accent)]"
                          >
                            Polymarket profile ↗
                          </a>
                        </div>
                        {w.openPositions.length > 0 && (
                          <div className="mt-2">
                            <p className="mb-1 text-[10px] uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">Open earnings positions</p>
                            <ul className="flex flex-wrap gap-x-4 gap-y-1">
                              {w.openPositions.map((p, i) => (
                                <li key={i} className="text-xs">
                                  <span className="font-bold text-[color:var(--accent)]">{p.ticker || "?"}</span>{" "}
                                  <span className="font-semibold" style={{ color: p.side === "No" ? "#f87171" : "#41d39d" }}>{p.side}</span>{" "}
                                  <span className="tabular-nums text-[color:var(--ink-faint)]">{p.shares.toLocaleString()}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
      {filtered.length === 0 && (
        <p className="py-6 text-center text-xs text-[color:var(--ink-faint)]">No wallets in this archetype.</p>
      )}
    </div>
  );
}

function ClosedView({ rows }: { rows: PredictionClosedMarket[] }) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const sorted = useMemo(
    () => [...rows].sort((a, b) => (b.resolvedDate ?? "").localeCompare(a.resolvedDate ?? "")),
    [rows]
  );
  if (rows.length === 0) {
    return <p className="py-8 text-center text-sm text-[color:var(--ink-faint)]">No resolved markets in the snapshot.</p>;
  }
  return (
    <div className="overflow-hidden rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)]">
      <table className="w-full">
        <thead>
          <tr className="border-b border-[color:var(--line)] text-[10px] uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
            <th className="py-2 pl-4 pr-2 text-left font-semibold">Ticker</th>
            <th className="px-2 py-2 text-left font-semibold">Resolved</th>
            <th className="px-2 py-2 text-left font-semibold">Outcome</th>
            <th className="hidden px-2 py-2 text-right font-semibold sm:table-cell">Volume</th>
            <th className="py-2 pl-2 pr-4 text-left font-semibold">Sharps correct</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => {
            const open = expanded === row.conditionId;
            const hasCohort = row.sharpCohort.total > 0;
            const beat = row.outcome === "beat";
            return (
              <Fragment key={row.conditionId}>
                <tr
                  onClick={() => setExpanded(open ? null : hasCohort ? row.conditionId : null)}
                  className={`border-b border-[color:var(--line)] last:border-0 ${hasCohort ? "cursor-pointer hover:bg-[color:rgba(79,213,255,0.04)]" : ""}`}
                >
                  <td className="py-2.5 pl-4 pr-2 text-xs font-bold text-[color:var(--accent)]">{row.ticker || "—"}</td>
                  <td className="px-2 py-2.5 text-xs text-[color:var(--ink-soft)]">{reportLabel(row.resolvedDate)}</td>
                  <td className="px-2 py-2.5">
                    <span
                      className="rounded px-1.5 py-0.5 text-[10px] font-semibold"
                      style={{ color: beat ? "#41d39d" : "#f87171", backgroundColor: `color-mix(in srgb, ${beat ? "#41d39d" : "#f87171"} 14%, transparent)` }}
                    >
                      {beat ? "Beat" : "Miss"}
                    </span>
                  </td>
                  <td className="hidden px-2 py-2.5 text-right text-xs tabular-nums text-[color:var(--ink-faint)] sm:table-cell">
                    {usd(row.volume)}
                  </td>
                  <td className="py-2.5 pl-2 pr-4 text-left text-xs tabular-nums">
                    {hasCohort ? (
                      <span className="text-[color:var(--ink-soft)]">
                        <span className="font-semibold text-[color:var(--ink)]">{row.sharpCohort.correct}</span>
                        <span className="text-[color:var(--ink-faint)]">/{row.sharpCohort.total}</span>
                      </span>
                    ) : (
                      <span className="text-[10px] text-[color:var(--ink-faint)]">no tracked sharps</span>
                    )}
                  </td>
                </tr>
                {open && hasCohort && (
                  <tr className="border-b border-[color:var(--line)] bg-[color:rgba(9,21,34,0.5)] last:border-0">
                    <td colSpan={5} className="px-4 py-3">
                      <p className="mb-2 text-[10px] uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
                        Tracked sharp results · {row.question} · P&amp;L is what each wallet made on this one market
                        (these markets are small — see Volume); cumulative records live in Sharp wallets
                      </p>
                      <ul className="space-y-1.5">
                        {row.sharpCohort.wallets.map((w) => (
                          <li key={w.wallet} className="flex items-center gap-2 text-xs">
                            <span style={{ color: w.correct ? "#41d39d" : "#f87171" }}>{w.correct ? "✓" : "✕"}</span>
                            <a
                              href={`https://polymarket.com/profile/${w.wallet}`}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="font-medium text-[color:var(--ink-soft)] hover:text-[color:var(--accent)] hover:underline"
                            >
                              {w.name}
                            </a>
                            <ArchetypeBadge archetype={w.archetype} />
                            <span className="ml-auto font-semibold tabular-nums" style={{ color: w.pnlUsd >= 0 ? "#41d39d" : "#f87171" }}>
                              {w.pnlUsd >= 0 ? "+" : ""}{usd(w.pnlUsd)}
                            </span>
                          </li>
                        ))}
                      </ul>
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

type View = "calendar" | "closed" | "wallets" | "macro";

export function PredictionMarketsTab({ data, loading, error, macro }: Props) {
  const [view, setView] = useState<View>("calendar");

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
            Prediction Markets · Polymarket
          </p>
          <p className="mt-0.5 text-[10px] text-[color:var(--ink-faint)]">
            Research context only — not investment advice.
          </p>
        </div>
        <div className="flex overflow-hidden rounded-lg border border-[color:var(--line)]">
          {(["calendar", "closed", "wallets", "macro"] as const).map((id) => (
            <button
              key={id}
              type="button"
              onClick={() => setView(id)}
              className={`whitespace-nowrap px-3 py-1 text-xs font-medium ${view === id ? "bg-[rgba(79,213,255,0.12)] text-[color:var(--ink)]" : "text-[color:var(--ink-faint)]"}`}
            >
              {id === "wallets" ? "Sharp wallets" : id === "closed" ? "Closed" : id === "macro" ? "Macro" : "Upcoming"}
            </button>
          ))}
        </div>
      </div>

      {view !== "macro" && data?.warning && (
        <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-3 text-xs text-amber-300">
          {data.warning} · snapshot {data.snapshotDate}
        </div>
      )}

      <p className="max-w-3xl text-xs text-[color:var(--ink-faint)]">
        {view === "macro" ? (
          <>
            Live contract distributions for Fed policy, growth, inflation, labor, recession risk, and housing,
            mapped to the related FRED indicators. Exact-series matches and related signals are labeled explicitly.
          </>
        ) : view === "calendar" ? (
          <>
            Upcoming &quot;Will X beat quarterly earnings?&quot; markets in report-date order, paired with
            Polymarket&apos;s implied beat probability and what tracked <span className="text-[color:var(--ink-soft)]">sharp</span> wallets
            currently hold. The consensus counts early sharps and longshots only — news scalpers (post-print traders)
            are shown on each wallet but never aggregated here. Click a row with tracked positions to expand.
          </>
        ) : view === "closed" ? (
          <>
            Resolved earnings markets, most recent first — each showing how it settled (
            <span className="text-[color:var(--ink-soft)]">beat</span> or{" "}
            <span className="text-[color:var(--ink-soft)]">miss</span>) and how the tracked sharp cohort (early sharps
            + longshots) actually did on it. Click a market to see each wallet&apos;s call and P&amp;L — the track
            record behind the badges.
          </>
        ) : (
          <>
            Wallets ranked by realized P&amp;L across {data?.archMinMarkets ?? 8}+ resolved earnings markets. Archetype is
            set by win rate and average entry price on eventual winners — an
            <span className="text-[color:var(--ink-soft)]"> early sharp</span> buys winners cheap (real edge), a
            <span className="text-[color:var(--ink-soft)]"> news scalper</span> buys them near-certain after the print
            (no predictive value), a <span className="text-[color:var(--ink-soft)]">longshot</span> is wrong often but
            paid when right. Click a wallet for the reasoning and its open positions.
          </>
        )}
      </p>

      {view !== "macro" && loading && !data && <div className="flex items-center justify-center py-16 text-sm text-[color:var(--ink-faint)]">Loading prediction markets…</div>}
      {view !== "macro" && error && !data && <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-4 text-sm text-red-400">{error}</div>}
      {view === "calendar" && data && <CalendarView rows={data.calendar} />}
      {view === "closed" && data && <ClosedView rows={data.closed} />}
      {view === "wallets" && data && <WalletsView wallets={data.wallets} />}
      {view === "macro" && <MacroPredictionMarketsView {...macro} />}
    </div>
  );
}

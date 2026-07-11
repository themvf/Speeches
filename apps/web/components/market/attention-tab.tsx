"use client";

import { useState } from "react";
import type { AttentionRow, MarketAttentionData } from "@/lib/server/types";

interface Props {
  data: MarketAttentionData | null;
  loading: boolean;
  error: string | null;
}

const MOOD_STYLES: Record<string, { label: string; color: string; symbol: string }> = {
  bullish: { label: "Bullish", color: "#41d39d", symbol: "▲" },
  bearish: { label: "Bearish", color: "#f87171", symbol: "▼" },
  mixed:   { label: "Mixed",   color: "#fbbf24", symbol: "◆" },
  neutral: { label: "Neutral", color: "var(--ink-faint)", symbol: "◆" },
};

function MoodChip({ mood, deemphasized = false }: { mood: string; deemphasized?: boolean }) {
  const style = MOOD_STYLES[mood] ?? MOOD_STYLES.neutral;
  return (
    // Keyword-heuristic mood is directional at best on sarcasm-heavy Reddit
    // text (spec §6.2) - rendered small and muted on purpose.
    <span
      className="text-[10px] font-medium"
      style={{ color: style.color, opacity: deemphasized ? 0.75 : 1 }}
      title="Keyword-based tone estimate - directional only"
    >
      {style.symbol} {style.label}
    </span>
  );
}

function MentionDelta({ current, prev }: { current: number; prev: number | null }) {
  if (prev == null || prev === 0) {
    return <span className="text-xs text-[color:var(--ink-faint)]">new</span>;
  }
  const pct = ((current - prev) / prev) * 100;
  const color = pct >= 0 ? "#41d39d" : "#f87171";
  const sign = pct >= 0 ? "▲" : "▼";
  return (
    <span className="text-xs font-semibold tabular-nums" style={{ color }}>
      {sign} {Math.abs(pct).toFixed(0)}%
    </span>
  );
}

function SourcesDrawer({ row }: { row: AttentionRow }) {
  if (row.topSources.length === 0) {
    return <p className="px-4 py-3 text-xs text-[color:var(--ink-faint)]">No source threads stored for this ticker.</p>;
  }
  return (
    <ul className="space-y-1.5 px-4 py-3">
      {row.topSources.map((source, i) => (
        <li key={i} className="flex items-baseline gap-2 text-xs">
          <span className="shrink-0 rounded bg-[rgba(79,213,255,0.08)] px-1.5 py-0.5 font-mono text-[10px] text-[color:var(--ink-faint)]">
            r/{source.subreddit}
          </span>
          <a
            href={source.permalink}
            target="_blank"
            rel="noopener noreferrer"
            className="truncate text-[color:var(--ink-soft)] hover:text-[color:var(--accent)] hover:underline"
          >
            {source.title || source.permalink}
          </a>
          <span className="shrink-0 text-[10px] text-[color:var(--ink-faint)]">u/{source.author}</span>
          <MoodChip mood={source.mood} deemphasized />
        </li>
      ))}
    </ul>
  );
}

function AttentionTableRow({ row, expanded, onToggle }: { row: AttentionRow; expanded: boolean; onToggle: () => void }) {
  const priceColor = (row.pricePct ?? 0) >= 0 ? "#41d39d" : "#f87171";
  return (
    <>
      <tr
        onClick={onToggle}
        className="cursor-pointer border-b border-[color:var(--line)] last:border-0 hover:bg-[color:rgba(79,213,255,0.04)]"
      >
        <td className="w-8 py-2.5 pl-4 pr-2 text-xs tabular-nums text-[color:var(--ink-faint)]">{row.rank}</td>
        <td className="w-16 px-2 py-2.5">
          <span className="text-xs font-bold text-[color:var(--accent)]">{row.ticker}</span>
        </td>
        <td className="max-w-[160px] truncate px-2 py-2.5 text-xs text-[color:var(--ink-faint)]">{row.company}</td>
        <td className="px-2 py-2.5 text-right text-xs tabular-nums text-[color:var(--ink)]">{row.mentionCount}</td>
        <td className="px-2 py-2.5 text-right">
          <MentionDelta current={row.mentionCount} prev={row.prevMentionCount} />
        </td>
        <td className="hidden px-2 py-2.5 text-right text-xs tabular-nums text-[color:var(--ink-faint)] sm:table-cell">
          {row.sourceCount}
        </td>
        <td className="hidden px-2 py-2.5 text-right text-xs tabular-nums text-[color:var(--ink-faint)] sm:table-cell">
          {row.subredditCount}
        </td>
        <td className="px-2 py-2.5 text-right">
          <MoodChip mood={row.mood} deemphasized />
        </td>
        <td className="py-2.5 pl-2 pr-4 text-right text-xs tabular-nums">
          {row.pricePct == null ? (
            <span className="text-[color:var(--ink-faint)]">—</span>
          ) : (
            <span className="font-semibold" style={{ color: priceColor }}>
              {row.pricePct >= 0 ? "+" : ""}
              {row.pricePct.toFixed(2)}%
            </span>
          )}
        </td>
      </tr>
      {expanded && (
        <tr className="border-b border-[color:var(--line)] bg-[color:rgba(9,21,34,0.5)] last:border-0">
          <td colSpan={9}>
            <SourcesDrawer row={row} />
          </td>
        </tr>
      )}
    </>
  );
}

export function AttentionTab({ data, loading, error }: Props) {
  const [expandedTicker, setExpandedTicker] = useState<string | null>(null);

  if (loading && !data) {
    return (
      <div className="flex items-center justify-center py-16 text-sm text-[color:var(--ink-faint)]">
        Loading attention data…
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-4 text-sm text-red-400">
        {error}
      </div>
    );
  }

  if (!data) return null;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
            Stocks Getting Attention · Reddit
          </p>
          <p className="mt-0.5 text-[10px] text-[color:var(--ink-faint)]">
            Research context only — not investment advice.
          </p>
        </div>
        {data.date && (
          <span className="text-xs text-[color:var(--ink-faint)]">
            {data.date} <span className="opacity-70">(UTC day)</span>
          </span>
        )}
      </div>

      {data.warning && (
        <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-3 text-xs text-amber-300">
          {data.warning}
        </div>
      )}

      {data.rows.length > 0 && (
        <div className="overflow-hidden rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)]">
          <table className="w-full">
            <thead>
              <tr className="border-b border-[color:var(--line)] text-[10px] uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
                <th className="py-2 pl-4 pr-2 text-left font-semibold">#</th>
                <th className="px-2 py-2 text-left font-semibold">Ticker</th>
                <th className="px-2 py-2 text-left font-semibold">Company</th>
                <th className="px-2 py-2 text-right font-semibold">Mentions</th>
                <th className="px-2 py-2 text-right font-semibold">Δ 24h</th>
                <th className="hidden px-2 py-2 text-right font-semibold sm:table-cell">Threads</th>
                <th className="hidden px-2 py-2 text-right font-semibold sm:table-cell">Subs</th>
                <th className="px-2 py-2 text-right font-semibold">Mood</th>
                <th className="py-2 pl-2 pr-4 text-right font-semibold">Price Δ</th>
              </tr>
            </thead>
            <tbody>
              {data.rows.map((row) => (
                <AttentionTableRow
                  key={row.ticker}
                  row={row}
                  expanded={expandedTicker === row.ticker}
                  onToggle={() => setExpandedTicker(expandedTicker === row.ticker ? null : row.ticker)}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}

      {data.rows.length === 0 && !data.warning && (
        <p className="py-8 text-center text-sm text-[color:var(--ink-faint)]">
          No attention data for this day.
        </p>
      )}

      <p className="text-[10px] text-[color:var(--ink-faint)]">
        Mentions are deduplicated per author per day across swept subreddits. Click a row for source threads.
      </p>
    </div>
  );
}

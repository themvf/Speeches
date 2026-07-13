"use client";

import { useEffect, useState } from "react";
import type {
  AttentionActivityItem,
  AttentionAuthorRow,
  AttentionHistoryPoint,
  AttentionRow,
  IntradayAttentionRow,
  MarketAttentionActivityData,
  MarketAttentionAuthorsData,
  MarketAttentionData,
  MarketAttentionHistoryData,
  MarketAttentionIntradayData,
} from "@/lib/server/types";

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

const DIVERGENCE_STYLES: Record<string, { label: string; color: string }> = {
  attention_spike_no_price_move: { label: "Chatter w/o price move", color: "#fbbf24" },
  price_move_no_attention: { label: "Price move w/o chatter", color: "#a78bfa" },
};

const QUALITY_FLAG_LABELS: Record<string, string> = {
  same_author_crew: "Same accounts as yesterday",
  young_account_concentration: "Mostly young accounts",
  single_thread_concentration: "Single-thread driven",
};

// Posts and comments both flow through the sweep; comments display their
// parent submission's title, so without this chip three commenters on one
// thread look like three identical posts.
function KindChip({ kind }: { kind: string }) {
  const isPost = kind === "post";
  return (
    <span
      className={`shrink-0 rounded px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-[0.08em] ${
        isPost
          ? "border border-[rgba(79,213,255,0.35)] bg-[rgba(79,213,255,0.1)] text-[color:var(--accent)]"
          : "border border-[color:var(--line)] text-[color:var(--ink-faint)]"
      }`}
      title={isPost ? "Original submission by this author" : "Comment by this author - the title shown is the parent post's"}
    >
      {isPost ? "post" : "↳ comment"}
    </span>
  );
}

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

function DivergenceBadge({ divergence }: { divergence: string }) {
  const style = DIVERGENCE_STYLES[divergence];
  if (!style) return null;
  return (
    <span
      title={`${style.label} (provisional heuristic — see docs/stock-attention-enhancements-spec.md item 2)`}
      className="ml-1 inline-block h-1.5 w-1.5 shrink-0 rounded-full align-middle"
      style={{ backgroundColor: style.color }}
    />
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

// Minimal hand-rolled sparkline - no chart lib, matches repo convention
// (see MoversTab's proportional bars). Renders total_mention_count over
// the trailing window (item 3a).
function Sparkline({ values, color }: { values: number[]; color: string }) {
  if (values.length < 2) {
    return <span className="text-[10px] text-[color:var(--ink-faint)]">—</span>;
  }
  const width = 64;
  const height = 20;
  const max = Math.max(...values, 1);
  const min = Math.min(...values, 0);
  const range = max - min || 1;
  const points = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * width;
      const y = height - ((v - min) / range) * height;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="inline-block overflow-visible">
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" opacity="0.85" />
    </svg>
  );
}

function HistoryChart({ points }: { points: AttentionHistoryPoint[] }) {
  if (points.length === 0) {
    return <p className="text-xs text-[color:var(--ink-faint)]">No stored history yet.</p>;
  }
  const width = 280;
  const height = 60;
  const maxMentions = Math.max(...points.map((p) => p.mentionCount), 1);
  const barWidth = width / points.length;
  return (
    <div className="space-y-2">
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="block">
        {points.map((p, i) => {
          const barHeight = (p.mentionCount / maxMentions) * height;
          return (
            <rect
              key={p.date}
              x={i * barWidth + 1}
              y={height - barHeight}
              width={Math.max(1, barWidth - 2)}
              height={barHeight}
              fill="rgba(79,213,255,0.55)"
            />
          );
        })}
      </svg>
      <div className="flex justify-between text-[9px] text-[color:var(--ink-faint)]">
        <span>{points[0]?.date}</span>
        <span>{points[points.length - 1]?.date}</span>
      </div>
    </div>
  );
}

function TickerHistory({ ticker }: { ticker: string }) {
  const [history, setHistory] = useState<MarketAttentionHistoryData | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetch(`/api/market/attention/history?ticker=${encodeURIComponent(ticker)}&days=30`)
      .then((r) => r.json())
      .then((env) => {
        if (!cancelled && env.ok && env.data) setHistory(env.data);
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [ticker]);

  if (loading) return <p className="text-xs text-[color:var(--ink-faint)]">Loading 30-day history…</p>;
  if (!history || history.warning) {
    return <p className="text-xs text-[color:var(--ink-faint)]">{history?.warning ?? "History unavailable."}</p>;
  }
  return <HistoryChart points={history.points} />;
}

function SourcesDrawer({ row }: { row: AttentionRow }) {
  return (
    <div className="space-y-3 px-4 py-3">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[10px] text-[color:var(--ink-faint)]">
        <span>Reddit: {row.redditCount}</span>
        <span>News: {row.newsCount}</span>
        {row.weightedMentionCount !== row.redditCount && (
          <span title="Credibility-weighted mention count: repeat single-ticker accounts and lower-trust subreddits count less">
            Weighted: {row.weightedMentionCount.toFixed(1)}
          </span>
        )}
        {row.volumeVs20d != null && <span>Volume: {row.volumeVs20d.toFixed(1)}x 20d avg</span>}
        {row.storedPricePct != null && (
          <span>
            As-of rollup: {row.storedPricePct >= 0 ? "+" : ""}
            {row.storedPricePct.toFixed(2)}%
          </span>
        )}
        {row.divergence && DIVERGENCE_STYLES[row.divergence] && (
          <span style={{ color: DIVERGENCE_STYLES[row.divergence]!.color }}>
            {DIVERGENCE_STYLES[row.divergence]!.label}
          </span>
        )}
        {row.qualityFlags.map((flag) => (
          <span
            key={flag}
            className="rounded border border-amber-500/30 bg-amber-500/10 px-1.5 py-0.5 text-amber-300"
            title="Data-quality annotation (see stock-attention enhancement spec item 6) - the row still ranks normally"
          >
            ⚠ {QUALITY_FLAG_LABELS[flag] ?? flag}
          </span>
        ))}
      </div>

      <div>
        <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
          30-Day Trend
        </p>
        <TickerHistory ticker={row.ticker} />
      </div>

      {row.topNews.length > 0 && (
        <div>
          <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
            News Articles
          </p>
          <ul className="space-y-1.5">
            {row.topNews.map((article, i) => (
              <li key={i} className="flex items-baseline gap-2 text-xs">
                <span className="shrink-0 rounded bg-[rgba(79,213,255,0.08)] px-1.5 py-0.5 font-mono text-[10px] text-[color:var(--ink-faint)]">
                  news
                </span>
                <a
                  href={article.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="truncate text-[color:var(--ink-soft)] hover:text-[color:var(--accent)] hover:underline"
                >
                  {article.title || article.url}
                </a>
              </li>
            ))}
          </ul>
        </div>
      )}

      {row.topSources.length > 0 && (
        <div>
          <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
            Source Threads
          </p>
          <ul className="space-y-1.5">
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
        </div>
      )}
    </div>
  );
}

function AttentionTableRow({ row, expanded, onToggle }: { row: AttentionRow; expanded: boolean; onToggle: () => void }) {
  const priceColor = (row.pricePct ?? 0) >= 0 ? "#41d39d" : "#f87171";
  const sparklineColor = row.sparkline.length > 1 && row.sparkline[row.sparkline.length - 1]! >= row.sparkline[0]!
    ? "#41d39d"
    : "#f87171";
  return (
    <>
      <tr
        onClick={onToggle}
        className="cursor-pointer border-b border-[color:var(--line)] last:border-0 hover:bg-[color:rgba(79,213,255,0.04)]"
      >
        <td className="w-8 py-2.5 pl-4 pr-2 text-xs tabular-nums text-[color:var(--ink-faint)]">{row.rank}</td>
        <td className="w-16 px-2 py-2.5">
          <span className="text-xs font-bold text-[color:var(--accent)]">{row.ticker}</span>
          <DivergenceBadge divergence={row.divergence} />
          {row.qualityFlags.length > 0 && (
            <span
              className="ml-1 align-middle text-[10px] text-amber-400"
              title={row.qualityFlags.map((flag) => QUALITY_FLAG_LABELS[flag] ?? flag).join("; ")}
            >
              ⚠
            </span>
          )}
        </td>
        <td className="max-w-[160px] truncate px-2 py-2.5 text-xs text-[color:var(--ink-faint)]">{row.company}</td>
        <td className="px-2 py-2.5 text-right text-xs tabular-nums text-[color:var(--ink)]">{row.mentionCount}</td>
        <td className="px-2 py-2.5 text-right">
          <MentionDelta current={row.mentionCount} prev={row.prevMentionCount} />
        </td>
        <td className="hidden px-2 py-2.5 text-right md:table-cell">
          <Sparkline values={row.sparkline} color={sparklineColor} />
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
          <td colSpan={8}>
            <SourcesDrawer row={row} />
          </td>
        </tr>
      )}
    </>
  );
}

function timeAgo(iso: string): string {
  const ms = Date.now() - new Date(iso).getTime();
  const minutes = Math.max(0, Math.floor(ms / 60_000));
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  if (hours < 48) return `${hours}h`;
  return `${Math.floor(hours / 24)}d`;
}

function accountAge(iso: string | null): string {
  if (!iso) return "—";
  const days = Math.floor((Date.now() - new Date(iso).getTime()) / 86_400_000);
  if (days < 60) return `${days}d`;
  if (days < 730) return `${Math.floor(days / 30)}mo`;
  return `${Math.floor(days / 365)}y`;
}

// SEC-5: sub-view polling cadence while the view stays open.
const SUBVIEW_POLL_MS = 120_000;
// Debounce for the free-text ticker/author filters so typing doesn't fire
// a request per keystroke.
const FILTER_DEBOUNCE_MS = 300;

function useDebounced<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(timer);
  }, [value, delayMs]);
  return debounced;
}

function ActivityBoard({
  authorFilter,
  onAuthorFilterChange,
}: {
  authorFilter: string;
  onAuthorFilterChange: (author: string) => void;
}) {
  const [data, setData] = useState<MarketAttentionActivityData | null>(null);
  const [loading, setLoading] = useState(true);
  const [tickerFilter, setTickerFilter] = useState("");
  const [subredditFilter, setSubredditFilter] = useState("");
  const [kindFilter, setKindFilter] = useState<"all" | "post" | "comment">("all");

  // SEC-2 follow-up: filtering moved server-side (see neon.ts's
  // getRecentAttentionActivity comment) - a single sweep can produce far
  // more than the 150-row cap in a day, so filtering client-side over an
  // already-truncated list could show "0 items" for a real, recent post
  // just because enough other items were more recent. Text inputs are
  // debounced; discrete controls (subreddit/kind) refetch immediately.
  const debouncedTicker = useDebounced(tickerFilter.trim().toUpperCase(), FILTER_DEBOUNCE_MS);
  const debouncedAuthor = useDebounced(authorFilter.trim(), FILTER_DEBOUNCE_MS);

  useEffect(() => {
    let cancelled = false;
    const params = new URLSearchParams({ hours: "24" });
    if (debouncedTicker) params.set("ticker", debouncedTicker);
    if (debouncedAuthor) params.set("author", debouncedAuthor);
    if (subredditFilter) params.set("subreddit", subredditFilter);
    if (kindFilter !== "all") params.set("kind", kindFilter);
    const load = () => {
      fetch(`/api/market/attention/activity?${params.toString()}`)
        .then((r) => r.json())
        .then((env) => {
          if (!cancelled && env.ok && env.data) setData(env.data);
        })
        .catch(() => {})
        .finally(() => {
          if (!cancelled) setLoading(false);
        });
    };
    load();
    // SEC-5: silent refresh while the view is open (loading spinner only
    // covers the initial fetch), re-applying whatever filters are active.
    const timer = setInterval(load, SUBVIEW_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [debouncedTicker, debouncedAuthor, subredditFilter, kindFilter]);

  // SEC-7: configured subreddits come from the API (union of admin config
  // + everything observed in the window, independent of the current
  // filter - see the route), so the list never shrinks while filtering.
  const subreddits = data?.subreddits ?? [];
  // Filtering already happened server-side; items are the result.
  const filtered = data?.items ?? [];

  if (loading && !data) {
    return <p className="py-8 text-center text-sm text-[color:var(--ink-faint)]">Loading activity…</p>;
  }
  if (!data || (data.items.length === 0 && data.warning)) {
    return <p className="py-8 text-center text-sm text-[color:var(--ink-faint)]">{data?.warning ?? "No activity data."}</p>;
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <input
          value={tickerFilter}
          onChange={(e) => setTickerFilter(e.target.value)}
          placeholder="Filter by ticker…"
          className="w-36 rounded-lg border border-[color:var(--line)] bg-transparent px-2 py-1 text-xs text-[color:var(--ink)]"
        />
        <input
          value={authorFilter}
          onChange={(e) => onAuthorFilterChange(e.target.value)}
          placeholder="Filter by author…"
          className="w-36 rounded-lg border border-[color:var(--line)] bg-transparent px-2 py-1 text-xs text-[color:var(--ink)]"
        />
        <select
          value={subredditFilter}
          onChange={(e) => setSubredditFilter(e.target.value)}
          className="rounded-lg border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.95)] px-2 py-1 text-xs text-[color:var(--ink-soft)]"
        >
          <option value="">All subreddits</option>
          {subreddits.map((name) => (
            <option key={name} value={name}>r/{name}</option>
          ))}
        </select>
        <div className="flex overflow-hidden rounded-lg border border-[color:var(--line)]">
          {(["all", "post", "comment"] as const).map((kind) => (
            <button
              key={kind}
              type="button"
              onClick={() => setKindFilter(kind)}
              className={`px-2 py-1 text-[10px] font-medium capitalize ${kindFilter === kind ? "bg-[rgba(79,213,255,0.12)] text-[color:var(--ink)]" : "text-[color:var(--ink-faint)]"}`}
            >
              {kind === "all" ? "All" : `${kind}s`}
            </button>
          ))}
        </div>
        {(tickerFilter || authorFilter || subredditFilter || kindFilter !== "all") && (
          <button
            type="button"
            onClick={() => { setTickerFilter(""); onAuthorFilterChange(""); setSubredditFilter(""); setKindFilter("all"); }}
            className="text-[10px] text-[color:var(--ink-faint)] underline hover:text-[color:var(--accent)]"
          >
            clear filters
          </button>
        )}
        <span className="text-[10px] text-[color:var(--ink-faint)]">
          {filtered.length} items · past {data.hoursBack}h
        </span>
      </div>

      <div className="overflow-hidden rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)]">
        <ul className="divide-y divide-[color:var(--line)]">
          {filtered.map((item: AttentionActivityItem) => (
            <li key={item.sourceId} className="flex items-baseline gap-2 px-3 py-2 text-xs">
              <span className="w-8 shrink-0 text-right text-[10px] tabular-nums text-[color:var(--ink-faint)]">
                {timeAgo(item.createdUtc)}
              </span>
              <span className="shrink-0 rounded bg-[rgba(79,213,255,0.08)] px-1.5 py-0.5 font-mono text-[10px] text-[color:var(--ink-faint)]">
                r/{item.subreddit}
              </span>
              <KindChip kind={item.kind} />
              <span className="flex shrink-0 gap-1">
                {item.tickers.slice(0, 4).map((symbol) => (
                  <button
                    key={symbol}
                    type="button"
                    onClick={() => setTickerFilter(symbol)}
                    className="font-bold text-[color:var(--accent)] hover:underline"
                    title={`Filter to ${symbol}`}
                  >
                    {symbol}
                  </button>
                ))}
                {item.tickers.length > 4 && (
                  <span className="text-[10px] text-[color:var(--ink-faint)]">+{item.tickers.length - 4}</span>
                )}
              </span>
              <a
                href={item.permalink}
                target="_blank"
                rel="noopener noreferrer"
                className="min-w-0 flex-1 truncate text-[color:var(--ink-soft)] hover:text-[color:var(--accent)] hover:underline"
                title={item.title}
              >
                {item.title || item.permalink}
              </a>
              <button
                type="button"
                onClick={() => onAuthorFilterChange(item.author)}
                className="shrink-0 text-[10px] text-[color:var(--ink-faint)] hover:text-[color:var(--accent)] hover:underline"
                title={`Filter to u/${item.author}`}
              >
                u/{item.author}
              </button>
              <span className="hidden shrink-0 sm:inline">
                <MoodChip mood={item.mood} deemphasized />
              </span>
            </li>
          ))}
          {filtered.length === 0 && (
            <li className="px-3 py-6 text-center text-xs text-[color:var(--ink-faint)]">No items match the filter.</li>
          )}
        </ul>
      </div>
      <p className="text-[10px] text-[color:var(--ink-faint)]">
        Every swept post/comment that resolved to at least one ticker, newest first. The <span className="rounded border border-[color:var(--line)] px-1 text-[9px] uppercase text-[color:var(--ink-faint)]">↳ comment</span> chip
        marks a reply — the title shown is its parent submission&apos;s, and the author is the commenter, not the
        original poster.
      </p>
    </div>
  );
}

function AuthorsBoard({ onSelectAuthor }: { onSelectAuthor: (author: string) => void }) {
  const [data, setData] = useState<MarketAttentionAuthorsData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetch("/api/market/attention/authors")
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
  }, []);

  if (loading && !data) {
    return <p className="py-8 text-center text-sm text-[color:var(--ink-faint)]">Loading author stats…</p>;
  }
  if (!data || data.rows.length === 0) {
    return <p className="py-8 text-center text-sm text-[color:var(--ink-faint)]">{data?.warning ?? "No author stats yet."}</p>;
  }

  return (
    <div className="space-y-3">
      <div className="overflow-hidden rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)]">
        <table className="w-full">
          <thead>
            <tr className="border-b border-[color:var(--line)] text-[10px] uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
              <th className="py-2 pl-4 pr-2 text-left font-semibold">#</th>
              <th className="px-2 py-2 text-left font-semibold">Author</th>
              <th className="px-2 py-2 text-right font-semibold">Items</th>
              <th className="px-2 py-2 text-right font-semibold">Tickers</th>
              <th className="hidden px-2 py-2 text-right font-semibold sm:table-cell">Subs</th>
              <th className="px-2 py-2 text-left font-semibold">Top Ticker</th>
              <th className="hidden px-2 py-2 text-right font-semibold sm:table-cell">Account Age</th>
              <th className="hidden py-2 pl-2 pr-4 text-right font-semibold md:table-cell">Karma</th>
            </tr>
          </thead>
          <tbody>
            {data.rows.map((row: AttentionAuthorRow) => (
              <tr key={row.author} className="border-b border-[color:var(--line)] last:border-0">
                <td className="w-8 py-2 pl-4 pr-2 text-xs tabular-nums text-[color:var(--ink-faint)]">{row.rank}</td>
                <td className="px-2 py-2 text-xs">
                  <button
                    type="button"
                    onClick={() => onSelectAuthor(row.author)}
                    className="text-[color:var(--ink-soft)] hover:text-[color:var(--accent)] hover:underline"
                    title={`Show u/${row.author}'s posts/comments in the Activity view`}
                  >
                    u/{row.author}
                  </button>
                  <a
                    href={`https://www.reddit.com/user/${encodeURIComponent(row.author)}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="ml-1 text-[10px] text-[color:var(--ink-faint)] hover:text-[color:var(--accent)]"
                    title="Reddit profile"
                  >
                    ↗
                  </a>
                  {row.discounted && (
                    <span
                      className="ml-1.5 rounded border border-amber-500/30 bg-amber-500/10 px-1 py-0.5 text-[9px] text-amber-300"
                      title="Currently discounted to 0.25 weight in the leaderboard scoring: repeat activity concentrated on one or two tickers"
                    >
                      discounted
                    </span>
                  )}
                </td>
                <td className="px-2 py-2 text-right text-xs tabular-nums text-[color:var(--ink)]">{row.itemsTotal}</td>
                <td className="px-2 py-2 text-right text-xs tabular-nums text-[color:var(--ink-faint)]">{row.tickersDistinct}</td>
                <td className="hidden px-2 py-2 text-right text-xs tabular-nums text-[color:var(--ink-faint)] sm:table-cell">
                  {row.subredditsDistinct}
                </td>
                <td className="px-2 py-2 text-xs">
                  {row.topTicker ? (
                    <>
                      <span className="font-bold text-[color:var(--accent)]">{row.topTicker}</span>{" "}
                      <span className="text-[10px] text-[color:var(--ink-faint)]">
                        {(row.topTickerShare * 100).toFixed(0)}%
                      </span>
                    </>
                  ) : (
                    <span className="text-[color:var(--ink-faint)]">—</span>
                  )}
                </td>
                <td className="hidden px-2 py-2 text-right text-xs tabular-nums text-[color:var(--ink-faint)] sm:table-cell">
                  {accountAge(row.accountCreated)}
                </td>
                <td className="hidden py-2 pl-2 pr-4 text-right text-xs tabular-nums text-[color:var(--ink-faint)] md:table-cell">
                  {row.linkKarma ?? "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-[10px] text-[color:var(--ink-faint)]">
        Aggregated from public Reddit activity in the swept subreddits over the retention window (90 days). Account
        age/karma appear as they&apos;re looked up (a small budget per sweep). These stats feed the credibility
        weighting on the Daily board.
      </p>
    </div>
  );
}

function IntradayBoard() {
  const [data, setData] = useState<MarketAttentionIntradayData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      fetch("/api/market/attention/intraday?hours=24")
        .then((r) => r.json())
        .then((env) => {
          if (!cancelled && env.ok && env.data) setData(env.data);
        })
        .catch(() => {})
        .finally(() => {
          if (!cancelled) setLoading(false);
        });
    };
    load();
    // SEC-5: the "right now" board refreshes itself while open.
    const timer = setInterval(load, SUBVIEW_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  if (loading && !data) {
    return <p className="py-8 text-center text-sm text-[color:var(--ink-faint)]">Loading intraday board…</p>;
  }
  if (!data || data.rows.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-[color:var(--ink-faint)]">
        {data?.warning ?? "No intraday data available."}
      </p>
    );
  }

  return (
    <div className="overflow-hidden rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)]">
      <table className="w-full">
        <thead>
          <tr className="border-b border-[color:var(--line)] text-[10px] uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
            <th className="py-2 pl-4 pr-2 text-left font-semibold">#</th>
            <th className="px-2 py-2 text-left font-semibold">Ticker</th>
            <th className="px-2 py-2 text-right font-semibold">Freshness-Weighted</th>
            <th className="py-2 pl-2 pr-4 text-right font-semibold">Raw Mentions</th>
          </tr>
        </thead>
        <tbody>
          {data.rows.map((row: IntradayAttentionRow) => (
            <tr key={row.ticker} className="border-b border-[color:var(--line)] last:border-0">
              <td className="py-2 pl-4 pr-2 text-xs tabular-nums text-[color:var(--ink-faint)]">{row.rank}</td>
              <td className="px-2 py-2 text-xs font-bold text-[color:var(--accent)]">{row.ticker}</td>
              <td className="px-2 py-2 text-right text-xs tabular-nums text-[color:var(--ink)]">
                {row.decayedMentionCount.toFixed(1)}
              </td>
              <td className="py-2 pl-2 pr-4 text-right text-xs tabular-nums text-[color:var(--ink-faint)]">
                {row.rawMentionCount}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

type AttentionView = "daily" | "intraday" | "activity" | "authors";

const VIEW_LABELS: { id: AttentionView; label: string }[] = [
  { id: "daily", label: "Daily" },
  { id: "intraday", label: "Hot Right Now" },
  { id: "activity", label: "Activity" },
  { id: "authors", label: "Authors" },
];

export function AttentionTab({ data, loading, error }: Props) {
  const [expandedTicker, setExpandedTicker] = useState<string | null>(null);
  const [view, setView] = useState<AttentionView>("daily");
  // SEC-2: author filter lives here so the Authors board can drive the
  // Activity view.
  const [activityAuthor, setActivityAuthor] = useState("");
  // SEC-1: past-day override for the Daily board. null = latest (the
  // polled prop data); a date string switches to a one-off fetch.
  const [selectedDate, setSelectedDate] = useState<string | null>(null);
  const [dateData, setDateData] = useState<MarketAttentionData | null>(null);
  const [dateLoading, setDateLoading] = useState(false);

  useEffect(() => {
    if (!selectedDate) {
      setDateData(null);
      return;
    }
    let cancelled = false;
    setDateLoading(true);
    fetch(`/api/market/attention?date=${encodeURIComponent(selectedDate)}`)
      .then((r) => r.json())
      .then((env) => {
        if (!cancelled && env.ok && env.data) setDateData(env.data);
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setDateLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedDate]);

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

  // The Daily board renders the selected past day when one is chosen,
  // otherwise the live (polled) latest-day payload.
  const board = selectedDate ? dateData : data;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
            Stocks Getting Attention · Reddit + News
          </p>
          <p className="mt-0.5 text-[10px] text-[color:var(--ink-faint)]">
            Research context only — not investment advice.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex overflow-hidden rounded-lg border border-[color:var(--line)]">
            {VIEW_LABELS.map(({ id, label }) => (
              <button
                key={id}
                type="button"
                onClick={() => setView(id)}
                className={`whitespace-nowrap px-3 py-1 text-xs font-medium ${view === id ? "bg-[rgba(79,213,255,0.12)] text-[color:var(--ink)]" : "text-[color:var(--ink-faint)]"}`}
              >
                {label}
              </button>
            ))}
          </div>
          {view === "daily" && data.date && (
            <span className="flex items-center gap-1 text-xs text-[color:var(--ink-faint)]">
              <input
                type="date"
                value={selectedDate ?? data.date}
                max={data.date}
                onChange={(e) => {
                  const value = e.target.value;
                  // Picking the latest day returns to live/polled mode.
                  setSelectedDate(value && value !== data.date ? value : null);
                }}
                className="rounded-lg border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.95)] px-2 py-0.5 text-xs text-[color:var(--ink-soft)] [color-scheme:dark]"
              />
              <span className="opacity-70">(UTC day)</span>
              {selectedDate && (
                <button
                  type="button"
                  onClick={() => setSelectedDate(null)}
                  className="underline hover:text-[color:var(--accent)]"
                >
                  latest
                </button>
              )}
            </span>
          )}
        </div>
      </div>

      {view === "intraday" && <IntradayBoard />}
      {view === "activity" && (
        <ActivityBoard authorFilter={activityAuthor} onAuthorFilterChange={setActivityAuthor} />
      )}
      {view === "authors" && (
        <AuthorsBoard
          onSelectAuthor={(author) => {
            setActivityAuthor(author);
            setView("activity");
          }}
        />
      )}
      {view === "daily" && (dateLoading || !board) && (
        <p className="py-8 text-center text-sm text-[color:var(--ink-faint)]">
          {dateLoading ? "Loading selected day…" : "No data for the selected day."}
        </p>
      )}
      {view === "daily" && !dateLoading && board && (
        <>
          {board.warning && (
            <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-3 text-xs text-amber-300">
              {board.warning}
            </div>
          )}

          {board.rows.length > 0 && (
            <div className="overflow-hidden rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)]">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-[color:var(--line)] text-[10px] uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
                    <th className="py-2 pl-4 pr-2 text-left font-semibold">#</th>
                    <th className="px-2 py-2 text-left font-semibold">Ticker</th>
                    <th className="px-2 py-2 text-left font-semibold">Company</th>
                    <th className="px-2 py-2 text-right font-semibold">Mentions</th>
                    <th className="px-2 py-2 text-right font-semibold">Δ 24h</th>
                    <th className="hidden px-2 py-2 text-right font-semibold md:table-cell">14d Trend</th>
                    <th className="hidden px-2 py-2 text-right font-semibold sm:table-cell">Subs</th>
                    <th className="px-2 py-2 text-right font-semibold">Mood</th>
                    <th className="py-2 pl-2 pr-4 text-right font-semibold">Price Δ</th>
                  </tr>
                </thead>
                <tbody>
                  {board.rows.map((row) => (
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

          {board.rows.length === 0 && !board.warning && (
            <p className="py-8 text-center text-sm text-[color:var(--ink-faint)]">
              No attention data for this day.
            </p>
          )}

          <p className="text-[10px] text-[color:var(--ink-faint)]">
            Mentions are deduplicated per author per day across swept subreddits and counted per article for news
            coverage. Click a row for source threads, trend, and channel breakdown. The colored dot next to a ticker
            flags a provisional divergence signal (chatter without a price move, or a price move without much
            chatter).
          </p>
        </>
      )}
    </div>
  );
}

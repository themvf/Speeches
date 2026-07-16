"use client";

import { useCallback, useEffect, useState } from "react";
import {
  dominantCatalyst,
  filterCompanyNewsArticles,
  isPossiblePriceCatalyst,
  isSessionNewsCacheFresh,
  sortAndFilterSectorStocks,
  summarizeLoadedSectorNews,
  type SourceTierFilter,
  type SectorPriceFilter,
  type SectorStockSort,
} from "@/lib/market-news-signals";
import type {
  CompanyNewsCatalyst,
  MarketCompanyNewsData,
  MarketSectorsData,
  SectorData,
  SectorStock,
} from "@/lib/server/types";
import { InlineChart } from "./price-chart";

const SECTOR_ETF: Record<string, string> = {
  "Technology":              "XLK",
  "Communication Services":  "XLC",
  "Consumer Discretionary":  "XLY",
  "Consumer Staples":        "XLP",
  "Energy":                  "XLE",
  "Financials":              "XLF",
  "Healthcare":              "XLV",
  "Industrials":             "XLI",
  "Materials":               "XLB",
  "Real Estate":             "XLRE",
  "Utilities":               "XLU",
};

interface Props {
  data: MarketSectorsData | null;
  loading: boolean;
  error: string | null;
}

type RangeId = "d1" | "w1" | "m1" | "m3" | "ytd";

const RANGES: { id: RangeId; label: string }[] = [
  { id: "d1",  label: "1D" },
  { id: "w1",  label: "1W" },
  { id: "m1",  label: "1M" },
  { id: "m3",  label: "3M" },
  { id: "ytd", label: "YTD" },
];

const SESSION_NEWS_CACHE_KEY = "market-sector-company-news:v1";
const CATALYSTS: CompanyNewsCatalyst[] = [
  "Earnings", "M&A", "Product", "Regulation", "Litigation", "Analyst Rating", "Management",
];

type SessionNewsCache = Record<string, { savedAt: number; data: MarketCompanyNewsData }>;

function fmtPct(n: number): string {
  const sign = n >= 0 ? "+" : "";
  return `${sign}${n.toFixed(2)}%`;
}

function PctBar({ pct, maxAbs }: { pct: number; maxAbs: number }) {
  const w = Math.round((Math.abs(pct) / maxAbs) * 80);
  const color = pct >= 0 ? "#41d39d" : "#f87171";
  return (
    <div className="flex items-center justify-end gap-1.5">
      <span className="tabular-nums text-xs font-semibold" style={{ color }}>{fmtPct(pct)}</span>
      <div className="h-3 rounded-sm shrink-0" style={{ width: w, backgroundColor: color, opacity: 0.7 }} />
    </div>
  );
}

async function fetchCompanyNews(
  symbol: string,
  limit: 5 | 10,
  options: { refresh?: boolean; signal?: AbortSignal } = {},
): Promise<MarketCompanyNewsData> {
  const params = new URLSearchParams({ symbol, limit: String(limit) });
  if (options.refresh) params.set("refresh", "1");
  const response = await fetch(`/api/market/company-news?${params}`, { signal: options.signal });
  const envelope = await response.json();
  if (!response.ok || !envelope.ok || !envelope.data) {
    throw new Error(envelope.error ?? "Recent company news is unavailable.");
  }
  return envelope.data as MarketCompanyNewsData;
}

function StockRow({
  stock,
  maxAbs,
  news,
  onNewsUpdate,
  hidePressReleases,
  minSourceTier,
}: {
  stock: SectorStock;
  maxAbs: number;
  news: MarketCompanyNewsData | null;
  onNewsUpdate: (data: MarketCompanyNewsData) => void;
  hidePressReleases: boolean;
  minSourceTier: SourceTierFilter;
}) {
  const color = stock.up ? "#41d39d" : "#f87171";
  const sign = stock.pct >= 0 ? "+" : "";
  const barW = Math.round((Math.abs(stock.pct) / maxAbs) * 60);
  const [expanded, setExpanded] = useState(false);
  const [loadingNews, setLoadingNews] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [newsError, setNewsError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const [articleLimit, setArticleLimit] = useState<5 | 10>(5);

  useEffect(() => {
    if (news && news.articles.length > 5) setArticleLimit(10);
  }, [news]);

  useEffect(() => {
    if (!expanded || news) return;
    const controller = new AbortController();
    let active = true;
    setLoadingNews(true);
    setNewsError(null);

    fetchCompanyNews(stock.symbol, 5, { signal: controller.signal })
      .then((data) => { if (active) onNewsUpdate(data); })
      .catch((error) => {
        if (active && error instanceof Error && error.name !== "AbortError") {
          setNewsError(error.message);
        }
      })
      .finally(() => { if (active) setLoadingNews(false); });

    return () => {
      active = false;
      controller.abort();
    };
  }, [expanded, news, onNewsUpdate, retryCount, stock.symbol]);

  const toggle = () => setExpanded((current) => !current);
  const retry = () => {
    setNewsError(null);
    setRetryCount((current) => current + 1);
  };
  const viewMore = async () => {
    setLoadingMore(true);
    setNewsError(null);
    try {
      const data = await fetchCompanyNews(stock.symbol, 10);
      setArticleLimit(10);
      onNewsUpdate(data);
    } catch (error) {
      setNewsError(error instanceof Error ? error.message : "More company news is unavailable.");
    } finally {
      setLoadingMore(false);
    }
  };
  const refresh = async () => {
    setRefreshing(true);
    setNewsError(null);
    try {
      onNewsUpdate(await fetchCompanyNews(stock.symbol, articleLimit, { refresh: true }));
    } catch (error) {
      setNewsError(error instanceof Error ? error.message : "Company news refresh failed.");
    } finally {
      setRefreshing(false);
    }
  };
  const visibleArticles = news
    ? filterCompanyNewsArticles(news.articles, { hidePressReleases, minSourceTier })
    : [];
  const rowCatalyst = dominantCatalyst(news);

  return (
    <>
      <tr
        className="border-b border-[color:var(--line)] cursor-pointer hover:bg-[color:rgba(79,213,255,0.03)]"
        onClick={toggle}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            toggle();
          }
        }}
        role="button"
        tabIndex={0}
        aria-expanded={expanded}
        aria-label={`${expanded ? "Hide" : "Show"} recent news for ${stock.name}`}
      >
        <td className="pl-10 pr-2 py-2 w-20">
          <span className="mr-1.5 text-[10px] text-[color:var(--ink-faint)]">{expanded ? "−" : "+"}</span>
          <span className="text-xs font-bold text-[color:var(--accent)]">{stock.symbol}</span>
        </td>
        <td className="px-2 py-2 text-xs text-[color:var(--ink-faint)]">
          <div className="flex flex-wrap items-center gap-1.5">
            <span>{stock.name}</span>
            {!expanded && news && (
              <>
                <span className="rounded-full border border-[color:var(--line)] px-1.5 py-0.5 text-[9px] font-semibold text-[color:var(--accent)]">
                  {news.availableArticleCount} news
                </span>
                {rowCatalyst && (
                  <span className="rounded-full bg-[color:rgba(79,213,255,0.08)] px-1.5 py-0.5 text-[9px] text-[color:var(--ink-faint)]">
                    {rowCatalyst}
                  </span>
                )}
              </>
            )}
          </div>
        </td>
        <td className="px-2 py-2 tabular-nums text-xs text-right text-[color:var(--ink)]">
          ${stock.price.toFixed(2)}
        </td>
        <td className="px-2 py-2 tabular-nums text-xs text-right font-semibold" style={{ color }}>
          {sign}{stock.pct.toFixed(2)}%
        </td>
        <td className="pl-2 pr-4 py-2 w-20">
          <div className="flex justify-end">
            <div className="h-2.5 rounded-sm" style={{ width: barW, backgroundColor: color, opacity: 0.7 }} />
          </div>
        </td>
      </tr>
      {expanded && (
        <tr className="border-b border-[color:var(--line)]">
          <td colSpan={5} className="px-5 py-4 bg-[color:rgba(5,15,25,0.45)]">
            <div className="mx-auto max-w-4xl space-y-3">
              <div className="flex items-center justify-between gap-3">
                <p className="text-[11px] font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
                  Recent news · {stock.name}
                </p>
                {news && (
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-[color:var(--ink-faint)]">
                      {news.provider} · last {news.searchedDays} days
                    </span>
                    <button
                      type="button"
                      onClick={(event) => { event.stopPropagation(); void refresh(); }}
                      disabled={refreshing}
                      className="rounded-md border border-[color:var(--line)] px-2 py-1 text-[10px] font-semibold text-[color:var(--accent)] transition-colors hover:bg-[color:rgba(79,213,255,0.08)] disabled:cursor-wait disabled:opacity-50"
                    >
                      {refreshing ? "Refreshing…" : "Refresh"}
                    </button>
                  </div>
                )}
              </div>

              {loadingNews && (
                <div className="space-y-2" aria-label={`Loading recent news for ${stock.name}`}>
                  {[0, 1, 2].map((item) => (
                    <div key={item} className="h-12 animate-pulse rounded-lg bg-[color:rgba(79,213,255,0.06)]" />
                  ))}
                </div>
              )}

              {newsError && !loadingNews && (
                <div className="flex items-center justify-between gap-3 rounded-lg border border-red-500/20 bg-red-500/5 px-3 py-2 text-xs text-red-300">
                  <span>{newsError}</span>
                  <button type="button" onClick={(event) => { event.stopPropagation(); retry(); }} className="font-semibold text-red-200 hover:text-white">
                    Retry
                  </button>
                </div>
              )}

              {news && news.articles.length === 0 && (
                <p className="rounded-lg border border-[color:var(--line)] px-3 py-4 text-center text-xs text-[color:var(--ink-faint)]">
                  No relevant English-language U.S. news was found in the last {news.searchedDays} days.
                </p>
              )}

              {news && news.articles.length > 0 && visibleArticles.length === 0 && (
                <p className="rounded-lg border border-[color:var(--line)] px-3 py-4 text-center text-xs text-[color:var(--ink-faint)]">
                  Loaded articles are hidden by the current source-quality filters.
                </p>
              )}

              {news && visibleArticles.length > 0 && (
                <div className="divide-y divide-[color:var(--line)] overflow-hidden rounded-lg border border-[color:var(--line)]">
                  {visibleArticles.map((article) => (
                    <a
                      key={`${article.url}:${article.publishedAt}`}
                      href={article.url}
                      target="_blank"
                      rel="noreferrer"
                      onClick={(event) => event.stopPropagation()}
                      className="block bg-[color:rgba(9,21,34,0.45)] px-3 py-3 transition-colors hover:bg-[color:rgba(79,213,255,0.06)]"
                    >
                      <div className="flex items-start justify-between gap-4">
                        <p className="text-xs font-semibold leading-5 text-[color:var(--ink)]">{article.title}</p>
                        <span className="shrink-0 text-[10px] text-[color:var(--ink-faint)]">
                          {new Date(article.publishedAt).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
                        </span>
                      </div>
                      <div className="mt-0.5 flex items-center gap-2">
                        <p className="text-[10px] font-medium text-[color:var(--accent)]">{article.publisher}</p>
                        <span className="rounded-full border border-[color:var(--line)] px-1.5 py-0.5 text-[9px] text-[color:var(--ink-faint)]">
                          {article.sourceTier}
                        </span>
                        {article.isLikelyPaywalled && <span className="text-[9px] text-amber-300/80">Paywall likely</span>}
                        {article.isPressRelease && <span className="text-[9px] text-violet-300/80">Press release</span>}
                        {article.clusterSize > 1 && <span className="text-[9px] text-[color:var(--ink-faint)]">{article.clusterSize} similar</span>}
                        {article.catalyst && (
                          <span
                            title={`Relevance score: ${article.relevanceScore}`}
                            className="rounded-full border border-[color:rgba(79,213,255,0.24)] bg-[color:rgba(79,213,255,0.08)] px-1.5 py-0.5 text-[9px] font-semibold text-[color:var(--ink-faint)]"
                          >
                            {article.catalyst}
                          </span>
                        )}
                        {isPossiblePriceCatalyst(article, stock.pct) && (
                          <span
                            title="Temporal association only: a categorized story appeared within 36 hours of a 1%+ price move. This does not establish causation."
                            className="rounded-full border border-amber-400/25 bg-amber-400/10 px-1.5 py-0.5 text-[9px] font-semibold text-amber-200"
                          >
                            Possible catalyst
                          </span>
                        )}
                      </div>
                      {article.snippet && <p className="mt-1 line-clamp-2 text-[11px] leading-4 text-[color:var(--ink-faint)]">{article.snippet}</p>}
                    </a>
                  ))}
                </div>
              )}

              {news?.hasMore && (
                <div className="flex justify-center">
                  <button
                    type="button"
                    onClick={(event) => { event.stopPropagation(); void viewMore(); }}
                    disabled={loadingMore}
                    className="rounded-lg border border-[color:var(--line)] px-3 py-1.5 text-[10px] font-semibold text-[color:var(--accent)] transition-colors hover:bg-[color:rgba(79,213,255,0.08)] disabled:cursor-wait disabled:opacity-50"
                  >
                    {loadingMore ? "Loading…" : `View more (${news.availableArticleCount - news.articles.length})`}
                  </button>
                </div>
              )}

              {news?.warning && <p className="text-[10px] text-amber-300/80">{news.warning}</p>}
              {news?.refreshStatus === "refreshed" && !news.warning && (
                <p className="text-[10px] text-[color:var(--ink-faint)]">Fresh results loaded. Refresh is available again in 60 seconds.</p>
              )}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

function SectorRow({
  sector,
  range,
  expanded,
  onToggle,
  maxAbs,
  newsBySymbol,
  onNewsUpdate,
}: {
  sector: SectorData;
  range: RangeId;
  expanded: boolean;
  onToggle: () => void;
  maxAbs: number;
  newsBySymbol: Record<string, MarketCompanyNewsData>;
  onNewsUpdate: (data: MarketCompanyNewsData) => void;
}) {
  const [sort, setSort] = useState<SectorStockSort>("move");
  const [price, setPrice] = useState<SectorPriceFilter>("all");
  const [catalyst, setCatalyst] = useState<"all" | CompanyNewsCatalyst>("all");
  const [loadedOnly, setLoadedOnly] = useState(false);
  const [hidePressReleases, setHidePressReleases] = useState(false);
  const [minSourceTier, setMinSourceTier] = useState<SourceTierFilter>("All");
  const pct = sector.pcts[range];
  const stockMax = sector.stocks.length > 0
    ? Math.max(...sector.stocks.map((s) => Math.abs(s.pct)), 1)
    : 1;
  const summary = summarizeLoadedSectorNews(sector.stocks, newsBySymbol);
  const visibleStocks = sortAndFilterSectorStocks(sector.stocks, newsBySymbol, {
    sort, price, catalyst, loadedOnly,
  });
  const controlClass = "rounded-md border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.85)] px-2 py-1 text-[10px] text-[color:var(--ink)]";

  return (
    <>
      <tr
        className="border-b border-[color:var(--line)] cursor-pointer hover:bg-[color:rgba(79,213,255,0.04)] transition-colors"
        onClick={onToggle}
      >
        <td className="px-4 py-3 text-xs font-semibold text-[color:var(--ink)]">
          <span className="mr-2 text-[color:var(--ink-faint)]">{expanded ? "[-]" : "[+]"}</span>
          {sector.name}
        </td>
        <td className="px-4 py-3 text-right">
          <PctBar pct={pct} maxAbs={maxAbs} />
        </td>
      </tr>
      {expanded && (
        <tr>
          <td colSpan={2} className="p-0 bg-[color:rgba(9,21,34,0.3)]">
            {/* ETF price chart */}
            {SECTOR_ETF[sector.name] && (
              <div className="px-4 pt-4 pb-3 border-b border-[color:var(--line)]">
                <InlineChart
                  symbol={SECTOR_ETF[sector.name]}
                  type="yahoo"
                  name={sector.name}
                  up={sector.pcts[range] >= 0}
                  label={`${SECTOR_ETF[sector.name]} ETF`}
                />
              </div>
            )}
            <div className="space-y-3 border-b border-[color:var(--line)] px-4 py-3">
              <div className="flex flex-wrap items-center justify-between gap-3">
                {summary.loadedCompanies === 0 ? (
                  <p className="text-[11px] text-[color:var(--ink-faint)]">
                    Open company rows to build a sector-news summary. News is never prefetched.
                  </p>
                ) : (
                  <div className="flex flex-wrap items-center gap-2 text-[10px] text-[color:var(--ink-faint)]">
                    <span className="font-semibold text-[color:var(--ink)]">Loaded coverage {summary.loadedCompanies}/{sector.stocks.length}</span>
                    <span>{summary.totalArticles} available articles</span>
                    <span>Dominant catalyst: {summary.dominantCatalyst ?? "None"}</span>
                    <span>Price reaction: {summary.priceReaction} ({summary.gainers} up / {summary.losers} down)</span>
                  </div>
                )}
                <span className="text-[9px] uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Session cache · 15 min</span>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <label className="flex items-center gap-1 text-[10px] text-[color:var(--ink-faint)]">
                  Sort
                  <select value={sort} onChange={(event) => setSort(event.target.value as SectorStockSort)} className={controlClass}>
                    <option value="move">Largest move</option>
                    <option value="latest">Latest loaded news</option>
                    <option value="relevance">Highest relevance</option>
                  </select>
                </label>
                <label className="flex items-center gap-1 text-[10px] text-[color:var(--ink-faint)]">
                  Price
                  <select value={price} onChange={(event) => setPrice(event.target.value as SectorPriceFilter)} className={controlClass}>
                    <option value="all">All</option>
                    <option value="gainers">Gainers</option>
                    <option value="losers">Losers</option>
                  </select>
                </label>
                <label className="flex items-center gap-1 text-[10px] text-[color:var(--ink-faint)]">
                  Catalyst
                  <select value={catalyst} onChange={(event) => setCatalyst(event.target.value as "all" | CompanyNewsCatalyst)} className={controlClass}>
                    <option value="all">All</option>
                    {CATALYSTS.map((item) => <option key={item} value={item}>{item}</option>)}
                  </select>
                </label>
                <label className="flex items-center gap-1 text-[10px] text-[color:var(--ink-faint)]">
                  Source
                  <select value={minSourceTier} onChange={(event) => setMinSourceTier(event.target.value as SourceTierFilter)} className={controlClass}>
                    <option value="All">All tiers</option>
                    <option value="Established">Established+</option>
                    <option value="Premium">Premium only</option>
                  </select>
                </label>
                <label className="flex items-center gap-1.5 text-[10px] text-[color:var(--ink-faint)]">
                  <input type="checkbox" checked={loadedOnly} onChange={(event) => setLoadedOnly(event.target.checked)} />
                  Loaded news only
                </label>
                <label className="flex items-center gap-1.5 text-[10px] text-[color:var(--ink-faint)]">
                  <input type="checkbox" checked={hidePressReleases} onChange={(event) => setHidePressReleases(event.target.checked)} />
                  Hide press releases
                </label>
              </div>
            </div>
            {/* Top stocks */}
            {sector.stocks.length > 0 && (
              <table className="w-full">
                <tbody>
                  {visibleStocks.map((s) => (
                    <StockRow
                      key={s.symbol}
                      stock={s}
                      maxAbs={stockMax}
                      news={newsBySymbol[s.symbol] ?? null}
                      onNewsUpdate={onNewsUpdate}
                      hidePressReleases={hidePressReleases}
                      minSourceTier={minSourceTier}
                    />
                  ))}
                  {visibleStocks.length === 0 && (
                    <tr><td colSpan={5} className="px-4 py-6 text-center text-xs text-[color:var(--ink-faint)]">No companies match these filters.</td></tr>
                  )}
                </tbody>
              </table>
            )}
          </td>
        </tr>
      )}
    </>
  );
}

export function SectorsTab({ data, loading, error }: Props) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [range, setRange] = useState<RangeId>("d1");
  const [newsBySymbol, setNewsBySymbol] = useState<Record<string, MarketCompanyNewsData>>({});

  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(SESSION_NEWS_CACHE_KEY);
      if (!raw) return;
      const cached = JSON.parse(raw) as SessionNewsCache;
      const freshEntries = Object.entries(cached).filter(([, entry]) =>
        entry?.data?.symbol && isSessionNewsCacheFresh(entry.savedAt)
      );
      setNewsBySymbol(Object.fromEntries(freshEntries.map(([symbol, entry]) => [symbol, entry.data])));
      sessionStorage.setItem(SESSION_NEWS_CACHE_KEY, JSON.stringify(Object.fromEntries(freshEntries)));
    } catch {
      sessionStorage.removeItem(SESSION_NEWS_CACHE_KEY);
    }
  }, []);

  const handleNewsUpdate = useCallback((news: MarketCompanyNewsData) => {
    setNewsBySymbol((current) => ({ ...current, [news.symbol]: news }));
    try {
      const raw = sessionStorage.getItem(SESSION_NEWS_CACHE_KEY);
      const cached = raw ? JSON.parse(raw) as SessionNewsCache : {};
      cached[news.symbol] = { savedAt: Date.now(), data: news };
      sessionStorage.setItem(SESSION_NEWS_CACHE_KEY, JSON.stringify(cached));
    } catch {
      sessionStorage.setItem(SESSION_NEWS_CACHE_KEY, JSON.stringify({
        [news.symbol]: { savedAt: Date.now(), data: news },
      }));
    }
  }, []);

  if (loading && !data) {
    return (
      <div className="flex items-center justify-center py-16 text-sm text-[color:var(--ink-faint)]">
        Loading sectors…
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

  const sorted = [...data.sectors].sort((a, b) => b.pcts[range] - a.pcts[range]);
  const maxAbs = Math.max(...sorted.map((s) => Math.abs(s.pcts[range])), 1);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-3">
        <p className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
          Market Sectors
        </p>

        <div className="flex items-center gap-2">
          {/* Range selector */}
          <div className="flex items-center gap-0.5 rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.6)] p-1">
            {RANGES.map(({ id, label }) => (
              <button
                key={id}
                type="button"
                onClick={() => setRange(id)}
                className={`rounded-lg px-3 py-1 text-xs font-medium transition-colors ${
                  range === id
                    ? "bg-[color:rgba(79,213,255,0.18)] text-[color:var(--ink)]"
                    : "text-[color:var(--ink-faint)] hover:text-[color:var(--ink)]"
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          <span className="text-xs text-[color:var(--ink-faint)]">
            {new Date(data.generatedAt).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
          </span>
        </div>
      </div>

      <div className="overflow-hidden rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)]">
        <table className="w-full">
          <tbody>
            {sorted.map((sector) => (
              <SectorRow
                key={sector.name}
                sector={sector}
                range={range}
                expanded={expandedId === sector.name}
                onToggle={() => setExpandedId((prev) => prev === sector.name ? null : sector.name)}
                maxAbs={maxAbs}
                newsBySymbol={newsBySymbol}
                onNewsUpdate={handleNewsUpdate}
              />
            ))}
          </tbody>
        </table>
      </div>

      {range !== "d1" && (
        <p className="text-right text-[11px] text-[color:var(--ink-faint)]">
          Sector % shown for {RANGES.find((r) => r.id === range)?.label}. Individual stock % is always 1D.
        </p>
      )}
    </div>
  );
}

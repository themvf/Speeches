import { type NextRequest } from "next/server";
import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import type { AttentionRow, AttentionSource, MarketAttentionData } from "@/lib/server/types";
import {
  getDailyAttentionForSubreddit,
  getDailyStockAttention,
  getDistinctAttentionSubredditsForDay,
  getLatestStockAttentionDate,
  getRedditAttentionItems,
  getRssArticlesByIds,
  getStockAttentionSparklines,
  type DailyStockAttentionRow,
  type RedditAttentionItemRow,
  type RssArticleRef,
} from "@/lib/server/neon";
import { fetchYahooQuote } from "@/lib/server/yahoo";
import { loadFilingChips } from "@/lib/server/filing-chips";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_ROWS = 50;
const QUOTE_PAIR_LIMIT = 25; // live-quote pairing only for the top of the board; below that the columns render as —
const SPARKLINE_DAYS = 14;

function parseJsonStringArray(raw: string): string[] {
  try {
    const parsed = JSON.parse(raw || "[]");
    return Array.isArray(parsed) ? parsed.map((entry) => String(entry)) : [];
  } catch {
    return [];
  }
}

function parseTopSourceIds(row: DailyStockAttentionRow): string[] {
  return parseJsonStringArray(row.top_source_ids);
}

function moodFromCounts(bullish: number, bearish: number): string {
  if (bullish > 0 && bearish > 0 && bullish === bearish) return "mixed";
  if (bullish > bearish) return "bullish";
  if (bearish > bullish) return "bearish";
  return "neutral";
}

// Pair live Yahoo quotes for the top N tickers of a board, same policy as the
// unfiltered path (below that limit, price columns render as —).
async function pairQuotes(tickers: string[]): Promise<Map<string, { price: number | null; pct: number | null } | null>> {
  const quoteTickers = tickers.slice(0, QUOTE_PAIR_LIMIT);
  const quotes = await Promise.allSettled(quoteTickers.map((ticker) => fetchYahooQuote(ticker, 300)));
  return new Map(
    quoteTickers.map((ticker, i) => {
      const settled = quotes[i];
      return [ticker, settled.status === "fulfilled" ? settled.value : null] as const;
    })
  );
}

export async function GET(req: NextRequest) {
  const requestId = createRequestId();
  const rawDate = req.nextUrl.searchParams.get("date") ?? "";
  const subredditFilter = (req.nextUrl.searchParams.get("subreddit") ?? "").trim();
  if (rawDate && !/^\d{4}-\d{2}-\d{2}$/.test(rawDate)) {
    return fail("Invalid date format - use YYYY-MM-DD", "BAD_DATE", 400, requestId);
  }

  // The attention tables are Python-owned and only exist after the first
  // sweep/rollup has run - a missing table (or any read failure) degrades
  // to an empty payload with a visible warning, matching the recap GET's
  // store-unavailable behavior, instead of a 500 that breaks the tab.
  try {
    const date = rawDate || (await getLatestStockAttentionDate());
    if (!date) {
      const empty: MarketAttentionData = {
        date: null,
        rows: [],
        subreddits: [],
        subredditFilter: null,
        warning: "No attention data aggregated yet - the daily rollup has not produced any days.",
        generatedAt: new Date().toISOString(),
      };
      return ok(empty, requestId);
    }

    // The Daily-view subreddit dropdown is populated regardless of whether a
    // filter is applied. Filing chips are fail-soft (SEC-50).
    const [subreddits, filingChips] = await Promise.all([
      getDistinctAttentionSubredditsForDay(date),
      loadFilingChips(72),
    ]);

    // ── Single-subreddit view: recompute the day's board from raw items ──
    if (subredditFilter) {
      const [filtered, dayRollup] = await Promise.all([
        getDailyAttentionForSubreddit(date, subredditFilter, MAX_ROWS),
        // Only for company names - the filtered aggregation doesn't carry them.
        getDailyStockAttention(date, 500),
      ]);
      const companyByTicker = new Map(dayRollup.map((row) => [row.ticker, row.company]));
      const allSourceIds = filtered.flatMap((row) => row.top_source_ids);
      const items = await getRedditAttentionItems([...new Set(allSourceIds)]);
      const itemsById = new Map<string, RedditAttentionItemRow>(items.map((item) => [item.source_id, item]));
      const quoteByTicker = await pairQuotes(filtered.map((row) => row.ticker));

      const data: MarketAttentionData = {
        date,
        subreddits,
        subredditFilter,
        rows: filtered.map((row, i): AttentionRow => {
          const quote = quoteByTicker.get(row.ticker) ?? null;
          const topSources: AttentionSource[] = row.top_source_ids
            .map((id) => itemsById.get(id))
            .filter((item): item is RedditAttentionItemRow => Boolean(item))
            .map((item) => ({
              title: item.title,
              permalink: item.permalink,
              subreddit: item.subreddit,
              author: item.author,
              kind: item.kind,
              mood: item.mood,
            }));
          return {
            rank: i + 1,
            ticker: row.ticker,
            company: companyByTicker.get(row.ticker) ?? "",
            mentionCount: row.mention_count,
            redditCount: row.mention_count,
            newsCount: 0,
            prevMentionCount: null, // no per-subreddit prior-day rollup to compare against
            sourceCount: row.source_count,
            subredditCount: 1,
            weightedScore: row.mention_count,
            mood: moodFromCounts(row.bullish, row.bearish),
            price: quote?.price ?? null,
            pricePct: quote?.pct ?? null,
            storedPriceClose: null,
            storedPricePct: null,
            volume: null,
            volumeVs20d: null,
            divergence: "",
            weightedMentionCount: row.mention_count,
            // Recomputed per-subreddit from raw items; the rollup's engagement
            // figure is blended across all subreddits and cannot be sliced.
            engagementScore: 0,
            qualityFlags: [],
            sparkline: [], // 14d trend is a blended-rollup signal; not meaningful for one subreddit
            topSources,
            topNews: [],
            ...(filingChips.has(row.ticker) ? { filings: filingChips.get(row.ticker) } : {}),
          };
        }),
        generatedAt: new Date().toISOString(),
      };
      return ok(data, requestId);
    }

    // ── Default blended view: the pre-aggregated rollup ──
    const prevDate = new Date(date + "T00:00:00Z");
    prevDate.setUTCDate(prevDate.getUTCDate() - 1);
    const prevDateIso = prevDate.toISOString().split("T")[0] as string;

    const [rows, prevRows] = await Promise.all([
      getDailyStockAttention(date, MAX_ROWS),
      getDailyStockAttention(prevDateIso, 500),
    ]);
    // Compared on total_mention_count (reddit + news, item 1) since that's
    // what the leaderboard now displays and ranks news-only tickers by.
    const prevByTicker = new Map(prevRows.map((row) => [row.ticker, row.total_mention_count]));

    const allSourceIds = rows.flatMap((row) => parseTopSourceIds(row));
    const tickers = rows.map((row) => row.ticker);
    const allNewsIds = [...new Set(
      rows.flatMap((row) => parseJsonStringArray(row.top_news_ids).map((id) => Number(id)).filter(Number.isFinite))
    )];

    const [items, sparklines, newsArticles] = await Promise.all([
      getRedditAttentionItems([...new Set(allSourceIds)]),
      getStockAttentionSparklines(tickers, SPARKLINE_DAYS),
      getRssArticlesByIds(allNewsIds),
    ]);
    const itemsById = new Map<string, RedditAttentionItemRow>(items.map((item) => [item.source_id, item]));
    const newsById = new Map<number, RssArticleRef>(newsArticles.map((article) => [article.id, article]));

    const quoteByTicker = await pairQuotes(tickers);

    const data: MarketAttentionData = {
      date,
      subreddits,
      subredditFilter: null,
      rows: rows.map((row, i): AttentionRow => {
        const quote = quoteByTicker.get(row.ticker) ?? null;
        const topSources: AttentionSource[] = parseTopSourceIds(row)
          .map((id) => itemsById.get(id))
          .filter((item): item is RedditAttentionItemRow => Boolean(item))
          .map((item) => ({
            title: item.title,
            permalink: item.permalink,
            subreddit: item.subreddit,
            author: item.author,
            kind: item.kind,
            mood: item.mood,
          }));
        return {
          rank: i + 1,
          ticker: row.ticker,
          company: row.company,
          mentionCount: row.total_mention_count,
          redditCount: row.reddit_count,
          newsCount: row.news_count,
          prevMentionCount: prevByTicker.get(row.ticker) ?? null,
          sourceCount: row.source_count,
          subredditCount: row.subreddit_count,
          weightedScore: row.weighted_score,
          engagementScore: row.engagement_score ?? 0,
          mood: row.mood,
          price: quote?.price ?? null,
          pricePct: quote?.pct ?? null,
          storedPriceClose: row.price_close,
          storedPricePct: row.price_pct,
          volume: row.volume,
          volumeVs20d: row.volume_vs_20d,
          divergence: row.divergence,
          weightedMentionCount: row.weighted_mention_count,
          qualityFlags: parseJsonStringArray(row.quality_flags),
          sparkline: (sparklines.get(row.ticker) ?? []).map((point) => point.total_mention_count),
          topSources,
          topNews: parseJsonStringArray(row.top_news_ids)
            .map((id) => newsById.get(Number(id)))
            .filter((article): article is RssArticleRef => Boolean(article))
            .map((article) => ({ title: article.title, url: article.url })),
          ...(filingChips.has(row.ticker) ? { filings: filingChips.get(row.ticker) } : {}),
        };
      }),
      generatedAt: new Date().toISOString(),
    };
    return ok(data, requestId);
  } catch (err) {
    console.error("[market/attention]", err);
    const empty: MarketAttentionData = {
      date: null,
      rows: [],
      subreddits: [],
      subredditFilter: null,
      warning: `Attention data unavailable: ${err instanceof Error ? err.message : "unknown error"}`,
      generatedAt: new Date().toISOString(),
    };
    return ok(empty, requestId);
  }
}

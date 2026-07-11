import { type NextRequest } from "next/server";
import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import type { AttentionRow, AttentionSource, MarketAttentionData } from "@/lib/server/types";
import {
  getDailyStockAttention,
  getLatestStockAttentionDate,
  getRedditAttentionItems,
  getStockAttentionSparklines,
  type DailyStockAttentionRow,
  type RedditAttentionItemRow,
} from "@/lib/server/neon";
import { fetchYahooQuote } from "@/lib/server/yahoo";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_ROWS = 50;
const QUOTE_PAIR_LIMIT = 25; // live-quote pairing only for the top of the board; below that the columns render as —
const SPARKLINE_DAYS = 14;

function parseTopSourceIds(row: DailyStockAttentionRow): string[] {
  try {
    const parsed = JSON.parse(row.top_source_ids || "[]");
    return Array.isArray(parsed) ? parsed.map((id) => String(id)) : [];
  } catch {
    return [];
  }
}

export async function GET(req: NextRequest) {
  const requestId = createRequestId();
  const rawDate = req.nextUrl.searchParams.get("date") ?? "";
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
        warning: "No attention data aggregated yet - the daily rollup has not produced any days.",
        generatedAt: new Date().toISOString(),
      };
      return ok(empty, requestId);
    }

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

    const [items, sparklines] = await Promise.all([
      getRedditAttentionItems([...new Set(allSourceIds)]),
      getStockAttentionSparklines(tickers, SPARKLINE_DAYS),
    ]);
    const itemsById = new Map<string, RedditAttentionItemRow>(items.map((item) => [item.source_id, item]));

    const quoteTickers = rows.slice(0, QUOTE_PAIR_LIMIT).map((row) => row.ticker);
    const quotes = await Promise.allSettled(quoteTickers.map((ticker) => fetchYahooQuote(ticker, 300)));
    const quoteByTicker = new Map(
      quoteTickers.map((ticker, i) => {
        const settled = quotes[i];
        return [ticker, settled.status === "fulfilled" ? settled.value : null] as const;
      })
    );

    const data: MarketAttentionData = {
      date,
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
          mood: row.mood,
          price: quote?.price ?? null,
          pricePct: quote?.pct ?? null,
          storedPriceClose: row.price_close,
          storedPricePct: row.price_pct,
          volume: row.volume,
          volumeVs20d: row.volume_vs_20d,
          divergence: row.divergence,
          sparkline: (sparklines.get(row.ticker) ?? []).map((point) => point.total_mention_count),
          topSources,
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
      warning: `Attention data unavailable: ${err instanceof Error ? err.message : "unknown error"}`,
      generatedAt: new Date().toISOString(),
    };
    return ok(empty, requestId);
  }
}

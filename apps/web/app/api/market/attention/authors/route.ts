import { createRequestId, ok } from "@/lib/server/api-utils";
import type {
  AttentionAuthorRow,
  AttentionSubredditCount,
  AttentionTickerCount,
  MarketAttentionAuthorsData,
} from "@/lib/server/types";
import { getAttentionSweepConfig, getRedditAuthorStats } from "@/lib/server/neon";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const DEFAULT_WEIGHTING = { low_diversity_share: 0.8, low_diversity_max_tickers: 2, min_items: 5 };
const AUTHOR_ROW_LIMIT = 150;

// The Python rollup stores top_tickers/top_subreddits as JSON strings; parse
// them defensively (old rows / the missing-column fallback yield "[]" or "").
function parseTickerCounts(raw: string): AttentionTickerCount[] {
  try {
    const parsed = JSON.parse(raw || "[]");
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((e) => e && typeof e.ticker === "string")
      .map((e) => ({ ticker: String(e.ticker), count: Number(e.count) || 0 }));
  } catch {
    return [];
  }
}

function parseSubredditCounts(raw: string): AttentionSubredditCount[] {
  try {
    const parsed = JSON.parse(raw || "[]");
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((e) => e && typeof e.subreddit === "string")
      .map((e) => ({ subreddit: String(e.subreddit), count: Number(e.count) || 0 }));
  } catch {
    return [];
  }
}

export async function GET() {
  const requestId = createRequestId();

  try {
    const [rows, config] = await Promise.all([
      getRedditAuthorStats(AUTHOR_ROW_LIMIT),
      // Fail-soft: the discount badge falls back to default thresholds if
      // the config read fails - the leaderboard itself must not depend on it.
      getAttentionSweepConfig().catch(() => null),
    ]);
    const weighting = { ...DEFAULT_WEIGHTING, ...(config?.author_weighting ?? {}) };

    const data: MarketAttentionAuthorsData = {
      rows: rows.map((row, i): AttentionAuthorRow => ({
        rank: i + 1,
        author: row.author,
        itemsTotal: row.items_total,
        tickersDistinct: row.tickers_distinct,
        subredditsDistinct: row.subreddits_distinct,
        topTicker: row.top_ticker,
        topTickerShare: row.top_ticker_share,
        topTickers: parseTickerCounts(row.top_tickers),
        topSubreddits: parseSubredditCounts(row.top_subreddits),
        accountCreated: row.account_created,
        linkKarma: row.link_karma,
        firstSeen: row.first_seen,
        lastSeen: row.last_seen,
        discounted:
          row.items_total >= (weighting.min_items ?? 5)
          && row.tickers_distinct <= weighting.low_diversity_max_tickers
          && row.top_ticker_share > weighting.low_diversity_share,
      })),
      generatedAt: new Date().toISOString(),
      ...(rows.length === 0 ? { warning: "No author stats yet - they populate with the daily rollup." } : {}),
    };
    return ok(data, requestId);
  } catch (err) {
    console.error("[market/attention/authors]", err);
    const data: MarketAttentionAuthorsData = {
      rows: [],
      warning: `Author stats unavailable: ${err instanceof Error ? err.message : "unknown error"}`,
      generatedAt: new Date().toISOString(),
    };
    return ok(data, requestId);
  }
}

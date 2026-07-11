import { type NextRequest } from "next/server";
import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import type { AttentionHistoryPoint, MarketAttentionHistoryData } from "@/lib/server/types";
import { getStockAttentionHistory } from "@/lib/server/neon";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_DAYS = 90;
const DEFAULT_DAYS = 30;

export async function GET(req: NextRequest) {
  const requestId = createRequestId();
  const ticker = (req.nextUrl.searchParams.get("ticker") ?? "").trim().toUpperCase();
  if (!ticker || !/^[A-Z.\-]{1,10}$/.test(ticker)) {
    return fail("A valid ticker query param is required", "BAD_TICKER", 400, requestId);
  }
  const rawDays = Number.parseInt(req.nextUrl.searchParams.get("days") ?? "", 10);
  const days = Number.isFinite(rawDays) ? Math.min(MAX_DAYS, Math.max(1, rawDays)) : DEFAULT_DAYS;

  try {
    const rows = await getStockAttentionHistory(ticker, days);
    const points: AttentionHistoryPoint[] = rows.map((row) => ({
      date: row.attention_date,
      mentionCount: row.total_mention_count,
      redditCount: row.reddit_count,
      newsCount: row.news_count,
      priceClose: row.price_close,
      pricePct: row.price_pct,
    }));
    const data: MarketAttentionHistoryData = {
      ticker,
      company: rows[0]?.company ?? "",
      points,
      ...(points.length === 0 ? { warning: `No attention history found for ${ticker}.` } : {}),
    };
    return ok(data, requestId);
  } catch (err) {
    console.error("[market/attention/history]", err);
    const data: MarketAttentionHistoryData = {
      ticker,
      company: "",
      points: [],
      warning: `Attention history unavailable: ${err instanceof Error ? err.message : "unknown error"}`,
    };
    return ok(data, requestId);
  }
}

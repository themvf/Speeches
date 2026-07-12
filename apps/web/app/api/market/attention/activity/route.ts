import { type NextRequest } from "next/server";
import { createRequestId, ok } from "@/lib/server/api-utils";
import type { AttentionActivityItem, MarketAttentionActivityData } from "@/lib/server/types";
import { getRecentAttentionActivity } from "@/lib/server/neon";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const DEFAULT_HOURS_BACK = 24;
const MAX_HOURS_BACK = 72;

export async function GET(req: NextRequest) {
  const requestId = createRequestId();
  const rawHours = Number.parseInt(req.nextUrl.searchParams.get("hours") ?? "", 10);
  const hoursBack = Number.isFinite(rawHours) ? Math.min(MAX_HOURS_BACK, Math.max(1, rawHours)) : DEFAULT_HOURS_BACK;

  try {
    const rows = await getRecentAttentionActivity(hoursBack);
    const items: AttentionActivityItem[] = rows.map((row) => ({
      sourceId: row.source_id,
      kind: row.kind,
      subreddit: row.subreddit,
      author: row.author,
      title: row.title,
      permalink: row.permalink,
      createdUtc: row.created_utc,
      score: row.score,
      mood: row.mood,
      tickers: Array.isArray(row.tickers) ? row.tickers : [],
    }));
    const data: MarketAttentionActivityData = {
      hoursBack,
      items,
      generatedAt: new Date().toISOString(),
      ...(items.length === 0 ? { warning: "No swept Reddit activity in this window yet." } : {}),
    };
    return ok(data, requestId);
  } catch (err) {
    console.error("[market/attention/activity]", err);
    const data: MarketAttentionActivityData = {
      hoursBack,
      items: [],
      warning: `Activity feed unavailable: ${err instanceof Error ? err.message : "unknown error"}`,
      generatedAt: new Date().toISOString(),
    };
    return ok(data, requestId);
  }
}

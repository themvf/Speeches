import { type NextRequest } from "next/server";
import { createRequestId, ok } from "@/lib/server/api-utils";
import type { AttentionActivityItem, MarketAttentionActivityData } from "@/lib/server/types";
import { getAttentionSweepConfig, getRecentAttentionActivity } from "@/lib/server/neon";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const DEFAULT_HOURS_BACK = 24;
const MAX_HOURS_BACK = 72;

export async function GET(req: NextRequest) {
  const requestId = createRequestId();
  const rawHours = Number.parseInt(req.nextUrl.searchParams.get("hours") ?? "", 10);
  const hoursBack = Number.isFinite(rawHours) ? Math.min(MAX_HOURS_BACK, Math.max(1, rawHours)) : DEFAULT_HOURS_BACK;

  try {
    // SEC-7: the filter dropdown must include admin-configured subreddits
    // even before they have swept items - config read is fail-soft so a
    // missing config table can't take down the feed.
    const [rows, config] = await Promise.all([
      getRecentAttentionActivity(hoursBack),
      getAttentionSweepConfig().catch(() => null),
    ]);
    const configured = (config?.subreddits ?? [])
      .filter((sub) => sub.active)
      .map((sub) => sub.name);
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
      subreddits: [...new Set([...configured, ...items.map((item) => item.subreddit)])].sort((a, b) =>
        a.toLowerCase().localeCompare(b.toLowerCase())
      ),
      generatedAt: new Date().toISOString(),
      ...(items.length === 0 ? { warning: "No swept Reddit activity in this window yet." } : {}),
    };
    return ok(data, requestId);
  } catch (err) {
    console.error("[market/attention/activity]", err);
    const data: MarketAttentionActivityData = {
      hoursBack,
      items: [],
      subreddits: [],
      warning: `Activity feed unavailable: ${err instanceof Error ? err.message : "unknown error"}`,
      generatedAt: new Date().toISOString(),
    };
    return ok(data, requestId);
  }
}

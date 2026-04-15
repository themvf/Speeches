import { loadTrendsData } from "@/lib/server/data-store";
import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import type { TrendItem, TrendsPayload } from "@/lib/server/types";

export const runtime = "nodejs";

function parseOptionalFloat(value: string | null): number | null {
  if (!value) return null;
  const n = Number.parseFloat(value);
  return Number.isFinite(n) ? n : null;
}

function parseOptionalInt(value: string | null): number | null {
  if (!value) return null;
  const n = Number.parseInt(value, 10);
  return Number.isFinite(n) ? n : null;
}

export async function GET(request: Request) {
  const requestId = createRequestId();

  try {
    const url = new URL(request.url);

    // Filters
    const minGrowth = parseOptionalFloat(url.searchParams.get("growth"));
    const minSize = parseOptionalInt(url.searchParams.get("size"));
    const range = url.searchParams.get("range") || "";   // "7d" | "30d" | "90d" | ""
    const sourceFilter = (url.searchParams.get("source") || "").trim().toLowerCase();
    const limitParam = parseOptionalInt(url.searchParams.get("limit"));

    const payload = await loadTrendsData();

    let trends: TrendItem[] = payload.trends;

    // Filter by minimum growth percentage
    if (minGrowth !== null) {
      trends = trends.filter((t) => t.growth_pct >= minGrowth);
    }

    // Filter by minimum total mentions (size)
    if (minSize !== null) {
      trends = trends.filter((t) => t.total_mentions >= minSize);
    }

    // Filter by date range using last_seen
    if (range) {
      const days = range === "7d" ? 7 : range === "90d" ? 90 : 30;
      const cutoff = new Date();
      cutoff.setDate(cutoff.getDate() - days);
      const cutoffStr = cutoff.toISOString().slice(0, 10);
      trends = trends.filter((t) => t.last_seen >= cutoffStr);
    }

    // Filter by source kind
    if (sourceFilter) {
      trends = trends.filter((t) => t.sources.some((s) => s.toLowerCase().includes(sourceFilter)));
    }

    // Limit results
    if (limitParam && limitParam > 0) {
      trends = trends.slice(0, limitParam);
    }

    const filtered: TrendsPayload = {
      ...payload,
      trend_count: trends.length,
      trends
    };

    return ok<TrendsPayload>(filtered, requestId);
  } catch (error) {
    return fail(
      `Failed to load trends: ${error instanceof Error ? error.message : "Unknown error"}`,
      "TRENDS_LOAD_FAILED",
      500,
      requestId
    );
  }
}

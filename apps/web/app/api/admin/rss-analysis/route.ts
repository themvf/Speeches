import { type NextRequest } from "next/server";
import { createRequestId, fail, normalizeText, ok } from "@/lib/server/api-utils";
import { analyzeMissingRssArticles } from "@/lib/server/rss-analysis-runner";

export const dynamic = "force-dynamic";
export const maxDuration = 60;

export async function POST(req: NextRequest) {
  const requestId = createRequestId();
  try {
    const body = await req.json().catch(() => ({})) as Record<string, unknown>;
    const limit = Math.max(1, Math.min(50, Number.parseInt(normalizeText(body.limit) || "10", 10) || 10));
    const result = await analyzeMissingRssArticles(limit);
    return ok(result, requestId);
  } catch (error) {
    return fail(
      `Failed to analyze RSS feed items: ${error instanceof Error ? error.message : "Unknown error"}`,
      "RSS_ANALYSIS_FAILED",
      500,
      requestId
    );
  }
}

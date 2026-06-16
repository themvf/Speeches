import { type NextRequest } from "next/server";
import { createRequestId, fail, normalizeText, ok } from "@/lib/server/api-utils";
import { generateFeedAnalysis, type FeedAnalysisInput } from "@/lib/server/feed-analysis";
import { getClientIp, getGenerateGlobalLimiter, getGenerateIpLimiter, isRateLimited } from "@/lib/server/rate-limit";

export const dynamic = "force-dynamic";
export const maxDuration = 60;

function stringList(value: unknown, maxItems: number): string[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => normalizeText(item).slice(0, 120))
    .filter(Boolean)
    .slice(0, maxItems);
}

export async function POST(req: NextRequest) {
  const requestId = createRequestId();
  const ip = getClientIp(req.headers);
  if (await isRateLimited(getGenerateIpLimiter(), ip)) {
    return fail("Rate limit exceeded. Please slow down.", "RATE_LIMITED", 429, requestId);
  }
  if (await isRateLimited(getGenerateGlobalLimiter(), "global")) {
    return fail("Server is busy. Please try again shortly.", "GLOBAL_RATE_LIMITED", 429, requestId);
  }

  try {
    const body = await req.json().catch(() => ({})) as Record<string, unknown>;
    const input: FeedAnalysisInput = {
      title: normalizeText(body.title).slice(0, 400),
      description: normalizeText(body.description).slice(0, 8000),
      url: normalizeText(body.url).slice(0, 1000),
      source: normalizeText(body.source).slice(0, 200),
      author: normalizeText(body.author).slice(0, 200),
      published_at: normalizeText(body.published_at).slice(0, 80),
      tone_label: normalizeText(body.tone_label).slice(0, 40),
      topics: stringList(body.topics, 10),
      item_type: normalizeText(body.item_type).slice(0, 40),
    };

    if (!input.title && !input.description) {
      return fail("Title or description is required.", "MISSING_ANALYSIS_INPUT", 400, requestId);
    }

    const analysis = await generateFeedAnalysis(input, normalizeText(body.model));
    return ok({ analysis }, requestId);
  } catch (error) {
    return fail(
      `Failed to analyze feed item: ${error instanceof Error ? error.message : "Unknown error"}`,
      "FEED_ANALYSIS_FAILED",
      500,
      requestId
    );
  }
}

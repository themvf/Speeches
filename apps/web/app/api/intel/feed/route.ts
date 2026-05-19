import { NextRequest, NextResponse } from "next/server";
import { fetchRssFeed } from "@/lib/server/rss-fetcher";
import {
  ensureSchema,
  getFeeds,
  getRecentArticles,
  getTopicRules,
  upsertRssArticles,
} from "@/lib/server/neon";
import { getClientIp, getFeedLimiter, isRateLimited } from "@/lib/server/rate-limit";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest): Promise<NextResponse> {
  const ip = getClientIp(req.headers);
  if (await isRateLimited(getFeedLimiter(), ip)) {
    return NextResponse.json({ ok: false, error: "Rate limit exceeded. Please slow down." }, { status: 429 });
  }

  const { searchParams } = req.nextUrl;
  const limit = Math.min(Number(searchParams.get("limit") ?? "100"), 400);
  const feedKey = searchParams.get("feedKey") ?? undefined;
  const sinceParam = searchParams.get("since");
  const since = sinceParam ? new Date(sinceParam) : undefined;

  try {
    await ensureSchema();

    let [articles, topicRules] = await Promise.all([
      getRecentArticles({ limit, feedKey, since }),
      getTopicRules(true),
    ]);

    const latestFetchedAt = articles.reduce((max, a) => {
      const t = a.fetched_at ? new Date(a.fetched_at).getTime() : 0;
      return t > max ? t : max;
    }, 0);
    const ageMs = latestFetchedAt > 0 ? Date.now() - latestFetchedAt : Number.POSITIVE_INFINITY;
    const needsRefresh = !feedKey && !since && ageMs > 8 * 60_000;

    if (needsRefresh) {
      const activeFeeds = await getFeeds(true);
      const refreshResults = await Promise.allSettled(
        activeFeeds.map(async (feed) => {
          const feedArticles = await fetchRssFeed(feed.feed_url, 50);
          await upsertRssArticles(feedArticles, feed.feed_key);
        })
      );
      for (const result of refreshResults) {
        if (result.status === "rejected") {
          console.error("[intel/feed] feed refresh failed:", result.reason);
        }
      }
      articles = await getRecentArticles({ limit, feedKey, since });
    }

    return NextResponse.json(
      {
        ok: true,
        data: { articles, topicRules, generatedAt: new Date().toISOString() },
      },
      {
        headers: {
          "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate",
        },
      }
    );
  } catch (err) {
    return NextResponse.json(
      { ok: false, error: String(err) },
      { status: 500 }
    );
  }
}

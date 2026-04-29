import { NextRequest, NextResponse } from "next/server";
import { fetchRssFeed } from "@/lib/server/rss-fetcher";
import {
  ensureSchema,
  getFeeds,
  getRecentArticles,
  getTopicRules,
  upsertRssArticles,
} from "@/lib/server/neon";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest): Promise<NextResponse> {
  const { searchParams } = req.nextUrl;
  const limit = Math.min(Number(searchParams.get("limit") ?? "100"), 200);
  const feedKey = searchParams.get("feedKey") ?? undefined;
  const sinceParam = searchParams.get("since");
  const since = sinceParam ? new Date(sinceParam) : undefined;

  try {
    await ensureSchema();

    let [articles, topicRules] = await Promise.all([
      getRecentArticles({ limit, feedKey, since }),
      getTopicRules(true),
    ]);

    const latestFetchedAt = articles[0]?.fetched_at ? new Date(articles[0].fetched_at).getTime() : 0;
    const ageMs = latestFetchedAt > 0 ? Date.now() - latestFetchedAt : Number.POSITIVE_INFINITY;
    const needsRefresh = !feedKey && !since && ageMs > 8 * 60_000;

    if (needsRefresh) {
      const activeFeeds = await getFeeds(true);
      await Promise.allSettled(
        activeFeeds.map(async (feed) => {
          const feedArticles = await fetchRssFeed(feed.feed_url, 50);
          await upsertRssArticles(feedArticles, feed.feed_key);
        })
      );
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

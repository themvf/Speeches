import { NextRequest, NextResponse } from "next/server";
import { fetchRssFeed } from "@/lib/server/rss-fetcher";
import { upsertRssArticles, ensureSchema, getFeeds } from "@/lib/server/neon";
import { analyzeMissingRssArticles } from "@/lib/server/rss-analysis-runner";

export const dynamic = "force-dynamic";
export const maxDuration = 55;

export async function GET(req: NextRequest): Promise<NextResponse> {
  return handleRefresh(req);
}

export async function POST(req: NextRequest): Promise<NextResponse> {
  return handleRefresh(req);
}

async function handleRefresh(req: NextRequest): Promise<NextResponse> {
  const secret = process.env.CRON_SECRET ?? "";
  if (secret) {
    const authHeader = req.headers.get("authorization") ?? "";
    if (authHeader !== `Bearer ${secret}`) {
      return NextResponse.json({ ok: false, error: "Unauthorized" }, { status: 401 });
    }
  }

  try {
    await ensureSchema();
  } catch (err) {
    return NextResponse.json(
      { ok: false, error: `Schema init failed: ${String(err)}` },
      { status: 500 }
    );
  }

  const activeFeeds = await getFeeds(true);

  const feedResults = await Promise.all(
    activeFeeds.map(async (feed) => {
      try {
        const articles = await fetchRssFeed(feed.feed_url, 50);
        const inserted = await upsertRssArticles(articles, feed.feed_key);
        return { feedKey: feed.feed_key, label: feed.label, fetched: articles.length, inserted };
      } catch (err) {
        return {
          feedKey: feed.feed_key,
          label: feed.label,
          fetched: 0,
          inserted: 0,
          error: String(err),
        };
      }
    })
  );

  const feeds: Array<{ feedKey: string; label: string; fetched: number; inserted: number; error?: string }> = [];
  let totalInserted = 0;
  let failedCount = 0;

  for (const result of feedResults) {
    feeds.push(result);
    totalInserted += result.inserted;
    if (result.error) {
      failedCount++;
    }
  }

  const allFailed = failedCount > 0 && failedCount === feedResults.length;
  const analysisLimit = Math.max(0, Math.min(25, Number.parseInt(process.env.RSS_AUTO_ANALYSIS_LIMIT || "5", 10) || 5));
  const analysis = analysisLimit > 0 && totalInserted > 0
    ? await analyzeMissingRssArticles(analysisLimit)
    : { selected_count: 0, saved_count: 0, failed_count: 0, failed: [] };

  return NextResponse.json(
    { ok: !allFailed, data: { inserted: totalInserted, failed_count: failedCount, feeds, analysis } },
    { status: allFailed ? 500 : 200 }
  );
}

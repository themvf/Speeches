import { NextRequest, NextResponse } from "next/server";
import { fetchRssFeed } from "@/lib/server/rss-fetcher";
import { deleteInvalidCouponArticles, upsertRssArticles, ensureSchema, getFeeds, markFeedRefreshed } from "@/lib/server/neon";
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
  const maintenanceSecret = process.env.RSS_REENRICH_SECRET ?? "";
  const acceptedTokens = [secret, maintenanceSecret].filter(Boolean).map((token) => `Bearer ${token}`);
  if (acceptedTokens.length > 0) {
    const authHeader = req.headers.get("authorization") ?? "";
    if (!acceptedTokens.includes(authHeader)) {
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

  const { searchParams } = req.nextUrl;
  const shouldFetchFeeds = searchParams.get("fetch") !== "0";
  const forceRefresh = searchParams.get("force") === "1";
  const activeFeeds = shouldFetchFeeds ? await getFeeds(true, { dueOnly: !forceRefresh }) : [];

  const feedResults = await Promise.all(
    activeFeeds.map(async (feed) => {
      try {
        const articles = await fetchRssFeed(feed.feed_url, 50);
        const inserted = await upsertRssArticles(articles, feed.feed_key);
        return {
          feedKey: feed.feed_key,
          label: feed.label,
          refreshIntervalMinutes: feed.refresh_interval_minutes,
          fetched: articles.length,
          inserted,
        };
      } catch (err) {
        return {
          feedKey: feed.feed_key,
          label: feed.label,
          refreshIntervalMinutes: feed.refresh_interval_minutes,
          fetched: 0,
          inserted: 0,
          error: String(err),
        };
      } finally {
        await markFeedRefreshed(feed.feed_key).catch((error) => {
          console.error(`[intel/rss-refresh] failed to mark feed refreshed for ${feed.feed_key}:`, error);
        });
      }
    })
  );

  const feeds: Array<{ feedKey: string; label: string; refreshIntervalMinutes: number; fetched: number; inserted: number; error?: string }> = [];
  let totalInserted = 0;
  let failedCount = 0;

  for (const result of feedResults) {
    feeds.push(result);
    totalInserted += result.inserted;
    if (result.error) {
      failedCount++;
    }
  }

  const deletedCouponArticles = await deleteInvalidCouponArticles().catch((error) => {
    console.error("[intel/rss-refresh] coupon cleanup failed:", error);
    return 0;
  });

  const allFailed = failedCount > 0 && failedCount === feedResults.length;
  const analysisLimitParam = searchParams.get("analysisLimit") || process.env.RSS_AUTO_ANALYSIS_LIMIT || "0";
  const analysisLimit = Math.max(0, Math.min(50, Number.parseInt(analysisLimitParam, 10) || 0));
  const analysis = analysisLimit > 0
    ? await analyzeMissingRssArticles(analysisLimit)
    : { selected_count: 0, saved_count: 0, failed_count: 0, failed: [] };

  return NextResponse.json(
    { ok: !allFailed, data: { inserted: totalInserted, deleted_coupon_articles: deletedCouponArticles, failed_count: failedCount, feeds, analysis } },
    { status: allFailed ? 500 : 200 }
  );
}

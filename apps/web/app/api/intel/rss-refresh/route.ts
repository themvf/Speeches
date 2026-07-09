import { NextRequest, NextResponse } from "next/server";
import { fetchRssFeed } from "@/lib/server/rss-fetcher";
import { deleteBlockedRssArticles, deleteInvalidCouponArticles, deleteNonEnglishPrNewswireArticles, upsertRssArticles, ensureSchema, getFeeds, getTopicRules, markFeedRefreshed } from "@/lib/server/neon";
import { filterRssArticlesForIngestion, rssFetchLimitForFeed, shouldKeywordFilterFeed } from "@/lib/server/rss-ingestion-filter";
import { analyzeMissingRssArticles } from "@/lib/server/rss-analysis-runner";
import { FINRA_MEMBER_FIRM_NEWS_FEED_KEY, FINRA_MEMBER_FIRM_NEWS_LABEL, fetchFinraMemberFirmRssBatch } from "@/lib/server/finra-member-firm-rss";
import { checkCronAuth } from "@/lib/server/api-utils";

export const dynamic = "force-dynamic";
export const maxDuration = 55;

export async function GET(req: NextRequest): Promise<NextResponse> {
  return handleRefresh(req);
}

export async function POST(req: NextRequest): Promise<NextResponse> {
  return handleRefresh(req);
}

function parseAnalysisFeedKeys(searchParams: URLSearchParams): string[] {
  const raw = [
    ...searchParams.getAll("analysisFeedKey"),
    ...searchParams.getAll("analysisFeedKeys"),
  ].join(",");
  return Array.from(new Set(
    raw
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean)
  ));
}

async function handleRefresh(req: NextRequest): Promise<NextResponse> {
  const auth = checkCronAuth(req);
  if (!auth.ok) {
    return NextResponse.json({ ok: false, error: auth.error }, { status: auth.status });
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
  const topicRules = await getTopicRules(true);
  const ingestionTopicRules = activeFeeds.some((feed) => shouldKeywordFilterFeed(feed.feed_key))
    ? topicRules
    : [];

  const feedResults = await Promise.all(
    activeFeeds.map(async (feed) => {
      let errorMessage: string | undefined;
      let result: {
        feedKey: string;
        label: string;
        refreshIntervalMinutes: number;
        fetched: number;
        matched: number;
        filtered: number;
        inserted: number;
        error?: string;
      };
      try {
        const rawArticles = await fetchRssFeed(feed.feed_url, rssFetchLimitForFeed(feed.feed_key));
        const filtered = filterRssArticlesForIngestion(feed.feed_key, rawArticles, ingestionTopicRules);
        const inserted = await upsertRssArticles(filtered.articles, feed.feed_key);
        result = {
          feedKey: feed.feed_key,
          label: feed.label,
          refreshIntervalMinutes: feed.refresh_interval_minutes,
          fetched: filtered.fetched,
          matched: filtered.matched,
          filtered: filtered.filtered,
          inserted,
        };
      } catch (err) {
        errorMessage = String(err);
        result = {
          feedKey: feed.feed_key,
          label: feed.label,
          refreshIntervalMinutes: feed.refresh_interval_minutes,
          fetched: 0,
          matched: 0,
          filtered: 0,
          inserted: 0,
          error: errorMessage,
        };
      }
      // Still marks refreshed on failure (rate-limits retries against a
      // persistently broken feed), but now records the error/streak instead
      // of masking it - see last_error/consecutive_failures on rss_feeds.
      await markFeedRefreshed(feed.feed_key, errorMessage).catch((error) => {
        console.error(`[intel/rss-refresh] failed to mark feed refreshed for ${feed.feed_key}:`, error);
      });
      return result;
    })
  );

  const feeds: Array<{
    feedKey: string;
    label: string;
    refreshIntervalMinutes: number;
    fetched: number;
    matched: number;
    filtered: number;
    inserted: number;
    error?: string;
    firmCount?: number;
    batchSize?: number;
    offset?: number;
    firmFeedFailures?: number;
  }> = [];
  let totalInserted = 0;
  let failedCount = 0;

  for (const result of feedResults) {
    feeds.push(result);
    totalInserted += result.inserted;
    if (result.error) {
      failedCount++;
    }
  }

  if (shouldFetchFeeds && searchParams.get("finraFirmFeeds") !== "0") {
    try {
      const firmBatch = await fetchFinraMemberFirmRssBatch();
      const firmFiltered = filterRssArticlesForIngestion(FINRA_MEMBER_FIRM_NEWS_FEED_KEY, firmBatch.articles, topicRules);
      const inserted = await upsertRssArticles(firmFiltered.articles, FINRA_MEMBER_FIRM_NEWS_FEED_KEY);
      totalInserted += inserted;
      feeds.push({
        feedKey: FINRA_MEMBER_FIRM_NEWS_FEED_KEY,
        label: FINRA_MEMBER_FIRM_NEWS_LABEL,
        refreshIntervalMinutes: 0,
        fetched: firmBatch.fetched,
        matched: firmFiltered.matched,
        filtered: firmBatch.filtered + firmFiltered.filtered,
        inserted,
        firmCount: firmBatch.firmCount,
        batchSize: firmBatch.batchSize,
        offset: firmBatch.offset,
        firmFeedFailures: firmBatch.failed,
      });
    } catch (error) {
      failedCount++;
      feeds.push({
        feedKey: FINRA_MEMBER_FIRM_NEWS_FEED_KEY,
        label: FINRA_MEMBER_FIRM_NEWS_LABEL,
        refreshIntervalMinutes: 0,
        fetched: 0,
        matched: 0,
        filtered: 0,
        inserted: 0,
        error: String(error),
      });
    }
  }

  const deletedCouponArticles = await deleteInvalidCouponArticles().catch((error) => {
    console.error("[intel/rss-refresh] coupon cleanup failed:", error);
    return 0;
  });
  const deletedNonEnglishPrNewswireArticles = await deleteNonEnglishPrNewswireArticles().catch((error) => {
    console.error("[intel/rss-refresh] PR Newswire language cleanup failed:", error);
    return 0;
  });
  const deletedBlockedRssArticleCount = await deleteBlockedRssArticles(topicRules).catch((error) => {
    console.error("[intel/rss-refresh] RSS policy cleanup failed:", error);
    return 0;
  });

  const allFailed = feeds.length > 0 && failedCount > 0 && failedCount === feeds.length;
  const analysisLimitParam = searchParams.get("analysisLimit") || process.env.RSS_AUTO_ANALYSIS_LIMIT || "0";
  const analysisLimit = Math.max(0, Math.min(50, Number.parseInt(analysisLimitParam, 10) || 0));
  const analysisFeedKeys = parseAnalysisFeedKeys(searchParams);
  const analysis = analysisLimit > 0
    ? await analyzeMissingRssArticles(analysisLimit, { feedKeys: analysisFeedKeys })
    : { selected_count: 0, saved_count: 0, failed_count: 0, failed: [] };

  return NextResponse.json(
    {
      ok: !allFailed,
      data: {
        inserted: totalInserted,
        deleted_coupon_articles: deletedCouponArticles,
        deleted_non_english_prnewswire_articles: deletedNonEnglishPrNewswireArticles,
        deleted_blocked_rss_articles: deletedBlockedRssArticleCount,
        failed_count: failedCount,
        feeds,
        analysis_feed_keys: analysisFeedKeys,
        analysis,
      },
    },
    { status: allFailed ? 500 : 200 }
  );
}

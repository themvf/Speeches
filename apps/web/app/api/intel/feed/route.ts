import { NextRequest, NextResponse } from "next/server";
import { fetchRssFeed } from "@/lib/server/rss-fetcher";
import {
  buildDocumentListItems,
  loadCorpusDocuments,
  loadEnrichmentState,
  selectNewsFeedDocuments,
} from "@/lib/server/data-store";
import { compactFeedArticles } from "@/lib/server/feed-payload";
import {
  deleteInvalidCouponArticles,
  ensureSchema,
  getFeeds,
  getRecentArticles,
  markFeedRefreshed,
  getTopicRules,
  upsertRssArticles,
  type StoredRssArticle,
  type StoredRssTopicRule,
} from "@/lib/server/neon";
import { getClientIp, getFeedLimiter, isRateLimited } from "@/lib/server/rate-limit";
import { analyzeMissingRssArticles } from "@/lib/server/rss-analysis-runner";

export const dynamic = "force-dynamic";

const COUPON_SPAM_PATTERN = /\b(?:promo[\s-]*codes?|coupon(?:s|[\s-]*codes?)|discount[\s-]*(?:codes?|coupons?))\b/i;

function isInvalidCouponArticle(article: StoredRssArticle): boolean {
  return COUPON_SPAM_PATTERN.test(`${article.title || ""} ${article.url || ""} ${article.description || ""}`);
}

function isInvalidCouponDocument(doc: { source_kind?: string; title?: string; url?: string; tags?: string[]; keywords?: string[] }): boolean {
  if (String(doc.source_kind || "") !== "wired_article") {
    return false;
  }
  return COUPON_SPAM_PATTERN.test(
    [
      doc.title || "",
      doc.url || "",
      ...(Array.isArray(doc.tags) ? doc.tags : []),
      ...(Array.isArray(doc.keywords) ? doc.keywords : []),
    ].join(" ")
  );
}

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
  const refresh = searchParams.get("refresh") === "1";

  try {
    let articles: StoredRssArticle[] = [];
    let topicRules: StoredRssTopicRule[] = [];

    if (process.env.DATABASE_URL) {
      await ensureSchema();
      await deleteInvalidCouponArticles().catch((error) => {
        console.error("[intel/feed] coupon cleanup failed:", error);
        return 0;
      });

      [articles, topicRules] = await Promise.all([
        getRecentArticles({ limit, feedKey, since }),
        getTopicRules(true),
      ]);

      const latestFetchedAt = articles.reduce((max, a) => {
        const t = a.fetched_at ? new Date(a.fetched_at).getTime() : 0;
        return t > max ? t : max;
      }, 0);
      const ageMs = latestFetchedAt > 0 ? Date.now() - latestFetchedAt : Number.POSITIVE_INFINITY;
      const needsRefresh = refresh && !feedKey && !since && ageMs > 8 * 60_000;

      if (needsRefresh) {
        const activeFeeds = await getFeeds(true, { dueOnly: true });
        let insertedCount = 0;
        const refreshResults = await Promise.allSettled(
          activeFeeds.map(async (feed) => {
            try {
              const feedArticles = await fetchRssFeed(feed.feed_url, 50);
              return upsertRssArticles(feedArticles, feed.feed_key);
            } finally {
              await markFeedRefreshed(feed.feed_key).catch((error) => {
                console.error(`[intel/feed] failed to mark feed refreshed for ${feed.feed_key}:`, error);
              });
            }
          })
        );
        for (const result of refreshResults) {
          if (result.status === "rejected") {
            console.error("[intel/feed] feed refresh failed:", result.reason);
          } else {
            insertedCount += result.value;
          }
        }
        const analysisLimit = Math.max(0, Math.min(10, Number.parseInt(process.env.RSS_AUTO_ANALYSIS_LIMIT || "0", 10) || 0));
        if (insertedCount > 0 && analysisLimit > 0) {
          await analyzeMissingRssArticles(analysisLimit);
        }
        articles = await getRecentArticles({ limit, feedKey, since });
      }
    }

    const [corpusDocs, enrichment] = await Promise.all([
      loadCorpusDocuments(),
      loadEnrichmentState(),
    ]);
    articles = articles.filter((article) => !isInvalidCouponArticle(article));
    const documents = selectNewsFeedDocuments(buildDocumentListItems(corpusDocs, enrichment))
      .filter((doc) => !isInvalidCouponDocument(doc));

    return NextResponse.json(
      {
        ok: true,
        data: { articles: compactFeedArticles(articles), topicRules, documents, generatedAt: new Date().toISOString() },
      },
      {
        headers: {
          "Cache-Control": refresh
            ? "no-store, no-cache, must-revalidate, proxy-revalidate"
            : "public, s-maxage=3600, stale-while-revalidate=86400",
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

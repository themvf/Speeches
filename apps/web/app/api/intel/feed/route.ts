import { NextRequest, NextResponse } from "next/server";
import { fetchRssFeed } from "@/lib/server/rss-fetcher";
import {
  loadNewsFeedDocumentsFromNeon,
} from "@/lib/server/data-store";
import { compactFeedArticles } from "@/lib/server/feed-payload";
import { filterTopicMappedArticles, normalizeTopicRules } from "@/lib/intel-topic-matching";
import {
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
import {
  filterRssArticlesForIngestion,
  isAllowedRssArticleForIngestion,
  rssFetchLimitForFeed,
  shouldKeywordFilterFeed,
} from "@/lib/server/rss-ingestion-filter";
import { analyzeMissingRssArticles } from "@/lib/server/rss-analysis-runner";

export const dynamic = "force-dynamic";

const DEFAULT_FEED_LIMIT = 500;
const MAX_FEED_LIMIT = 1000;
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

function passesRssPolicy(article: StoredRssArticle, topicRules: StoredRssTopicRule[]): boolean {
  return isAllowedRssArticleForIngestion(article.feed_key, {
    guid: article.guid,
    title: article.title,
    url: article.url,
    description: article.description,
    author: article.author,
    publishedAt: article.published_at ? new Date(article.published_at) : null,
  }, topicRules);
}

function parseFeedLimit(value: string | null): number {
  const parsed = Number.parseInt(value ?? "", 10);
  if (!Number.isFinite(parsed)) {
    return DEFAULT_FEED_LIMIT;
  }
  return Math.max(0, Math.min(parsed, MAX_FEED_LIMIT));
}

export async function GET(req: NextRequest): Promise<NextResponse> {
  const ip = getClientIp(req.headers);
  if (await isRateLimited(getFeedLimiter(), ip)) {
    return NextResponse.json({ ok: false, error: "Rate limit exceeded. Please slow down." }, { status: 429 });
  }

  const { searchParams } = req.nextUrl;
  const limit = parseFeedLimit(searchParams.get("limit"));
  const feedKey = searchParams.get("feedKey") ?? undefined;
  const sinceParam = searchParams.get("since");
  const since = sinceParam ? new Date(sinceParam) : undefined;
  const refresh = searchParams.get("refresh") === "1";
  const documentsOnly = searchParams.get("documentsOnly") === "1";
  const includeDocuments = documentsOnly || searchParams.get("includeDocuments") === "1";

  try {
    let articles: StoredRssArticle[] = [];
    let topicRules: StoredRssTopicRule[] = [];
    let unmappedFilteredCount = 0;

    if (!documentsOnly && process.env.DATABASE_URL) {
      await ensureSchema();
      topicRules = await getTopicRules(true);

      articles = await getRecentArticles({ limit, feedKey, since });

      const latestFetchedAt = articles.reduce((max, a) => {
        const t = a.fetched_at ? new Date(a.fetched_at).getTime() : 0;
        return t > max ? t : max;
      }, 0);
      const ageMs = latestFetchedAt > 0 ? Date.now() - latestFetchedAt : Number.POSITIVE_INFINITY;
      const needsRefresh = refresh && !feedKey && !since && ageMs > 8 * 60_000;

      if (needsRefresh) {
        const activeFeeds = await getFeeds(true, { dueOnly: true });
        const refreshTopicRules = activeFeeds.some((feed) => shouldKeywordFilterFeed(feed.feed_key))
          ? topicRules
          : [];
        let changedCount = 0;
        const refreshResults = await Promise.allSettled(
          activeFeeds.map(async (feed) => {
            let errorMessage: string | undefined;
            try {
              const feedArticles = await fetchRssFeed(feed.feed_url, rssFetchLimitForFeed(feed.feed_key));
              const filteredArticles = filterRssArticlesForIngestion(feed.feed_key, feedArticles, refreshTopicRules);
              return upsertRssArticles(filteredArticles.articles, feed.feed_key);
            } catch (error) {
              errorMessage = String(error);
              throw error;
            } finally {
              await markFeedRefreshed(feed.feed_key, errorMessage).catch((error) => {
                console.error(`[intel/feed] failed to mark feed refreshed for ${feed.feed_key}:`, error);
              });
            }
          })
        );
        for (const result of refreshResults) {
          if (result.status === "rejected") {
            console.error("[intel/feed] feed refresh failed:", result.reason);
          } else {
            changedCount += result.value.inserted + result.value.updated;
          }
        }
        const analysisLimit = Math.max(0, Math.min(10, Number.parseInt(process.env.RSS_AUTO_ANALYSIS_LIMIT || "0", 10) || 0));
        if (changedCount > 0 && analysisLimit > 0) {
          await analyzeMissingRssArticles(analysisLimit);
        }
        articles = await getRecentArticles({ limit, feedKey, since });
      }
    }

    const activeTopicRules = normalizeTopicRules(topicRules);
    const policyEligibleArticles = articles.filter((article) => (
      !isInvalidCouponArticle(article) &&
      passesRssPolicy(article, topicRules)
    ));
    articles = filterTopicMappedArticles(policyEligibleArticles, activeTopicRules);
    unmappedFilteredCount = policyEligibleArticles.length - articles.length;
    let documents: Awaited<ReturnType<typeof loadNewsFeedDocumentsFromNeon>>["documents"] = [];
    let documentSource: "neon" | "unavailable" | undefined;
    let documentMetadataOnly: boolean | undefined;
    let documentWarning: string | undefined;
    if (includeDocuments) {
      const result = await loadNewsFeedDocumentsFromNeon({
        limit: 250,
        pinnedSourceKindLimit: 25,
      });
      documents = result.documents
        .filter((doc) => !isInvalidCouponDocument(doc));
      documentSource = result.source;
      documentMetadataOnly = result.metadata_only;
      documentWarning = result.warning;
    }

    return NextResponse.json(
      {
        ok: true,
        data: {
          articles: compactFeedArticles(articles),
          topicRules,
          unmappedFilteredCount,
          ...(includeDocuments ? { documents } : {}),
          ...(includeDocuments ? {
            documentSource,
            documentMetadataOnly,
            ...(documentWarning ? { documentWarning } : {}),
          } : {}),
          generatedAt: new Date().toISOString(),
        },
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

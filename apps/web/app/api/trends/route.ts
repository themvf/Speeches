import { loadTrendsData } from "@/lib/server/data-store";
import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import type { TrendItem, TrendsPayload } from "@/lib/server/types";
import { getRecentArticles, getTopicRules, type StoredRssArticle, type StoredRssTopicRule } from "@/lib/server/neon";
import { getMatchingTopics, normalizeTopicRules, type TopicRuleView } from "@/lib/intel-topic-matching";

export const runtime = "nodejs";

const LIVE_NEWS_ARTICLE_LIMIT = 400;
const LIVE_NEWS_SOURCE_KIND = "rss_news_feed";

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

function rangeDays(value: string): number {
  return value === "7d" ? 7 : value === "90d" ? 90 : 30;
}

function topicTrendId(topicKey: string): string {
  return `news-topic-${topicKey.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "")}`;
}

function dateOnly(value: Date): string {
  return value.toISOString().slice(0, 10);
}

function articleDate(article: StoredRssArticle): Date | null {
  const raw = article.published_at || article.fetched_at;
  if (!raw) return null;
  const parsed = new Date(raw);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function articleDescription(article: StoredRssArticle): string {
  return String(article.description || "").replace(/\s+/g, " ").trim();
}

function buildSparkline(dailyCounts: Map<string, number>, days = 30): Array<{ date: string; count: number }> {
  const out: Array<{ date: string; count: number }> = [];
  const today = new Date();
  today.setUTCHours(0, 0, 0, 0);
  for (let i = days - 1; i >= 0; i -= 1) {
    const d = new Date(today);
    d.setUTCDate(today.getUTCDate() - i);
    const key = dateOnly(d);
    out.push({ date: key, count: dailyCounts.get(key) ?? 0 });
  }
  return out;
}

function buildLiveNewsTopicTrends(
  articles: StoredRssArticle[],
  topicRules: StoredRssTopicRule[],
  days: number
): TrendItem[] {
  const rules = normalizeTopicRules(topicRules);
  if (articles.length === 0 || rules.length === 0) {
    return [];
  }

  const now = new Date();
  const currentStart = new Date(now);
  currentStart.setUTCDate(now.getUTCDate() - days);
  const baselineStart = new Date(currentStart);
  baselineStart.setUTCDate(currentStart.getUTCDate() - days);
  const sparklineStart = new Date(now);
  sparklineStart.setUTCDate(now.getUTCDate() - 29);
  sparklineStart.setUTCHours(0, 0, 0, 0);

  type Bucket = {
    rule: TopicRuleView;
    current: number;
    baseline: number;
    total: number;
    firstSeen: Date | null;
    lastSeen: Date | null;
    docs: Array<{ article: StoredRssArticle; date: Date }>;
    dailyCounts: Map<string, number>;
  };

  const buckets = new Map<string, Bucket>();
  for (const rule of rules) {
    buckets.set(rule.topic_key, {
      rule,
      current: 0,
      baseline: 0,
      total: 0,
      firstSeen: null,
      lastSeen: null,
      docs: [],
      dailyCounts: new Map<string, number>(),
    });
  }

  for (const article of articles) {
    const date = articleDate(article);
    if (!date) continue;

    const matches = getMatchingTopics(article, rules);
    if (matches.length === 0) continue;

    for (const rule of matches) {
      const bucket = buckets.get(rule.topic_key);
      if (!bucket) continue;

      bucket.total += 1;
      if (!bucket.firstSeen || date < bucket.firstSeen) bucket.firstSeen = date;
      if (!bucket.lastSeen || date > bucket.lastSeen) bucket.lastSeen = date;

      if (date >= currentStart) {
        bucket.current += 1;
        bucket.docs.push({ article, date });
      } else if (date >= baselineStart) {
        bucket.baseline += 1;
      }

      if (date >= sparklineStart) {
        const key = dateOnly(date);
        bucket.dailyCounts.set(key, (bucket.dailyCounts.get(key) ?? 0) + 1);
      }
    }
  }

  return [...buckets.values()]
    .filter((bucket) => bucket.current > 0 && bucket.lastSeen)
    .map((bucket) => {
      const growthPct = bucket.baseline === 0
        ? 100
        : Math.round(((bucket.current - bucket.baseline) / bucket.baseline) * 1000) / 10;
      const topDocs = bucket.docs
        .sort((a, b) => b.date.getTime() - a.date.getTime())
        .slice(0, 10)
        .map(({ article, date }) => ({
          id: `rss:${article.id}`,
          title: article.title,
          date: dateOnly(date),
          source_kind: LIVE_NEWS_SOURCE_KIND,
          url: article.url,
          summary: articleDescription(article).slice(0, 300),
        }));

      return {
        id: topicTrendId(bucket.rule.topic_key),
        label: `${bucket.rule.label} News Momentum`,
        canonical_tag: bucket.rule.topic_key.toLowerCase(),
        cluster_tags: [
          bucket.rule.label,
          bucket.rule.topic_key,
          "news feed",
          "rss",
          ...bucket.rule.keywords.slice(0, 12),
        ],
        description: `${bucket.current} live news article${bucket.current === 1 ? "" : "s"} matched ${bucket.rule.label} in the selected ${days}-day window using the News Feed topic rules.`,
        total_mentions: bucket.total,
        recent_mentions: bucket.current,
        growth_pct: growthPct,
        first_seen: bucket.firstSeen ? dateOnly(bucket.firstSeen) : "",
        last_seen: bucket.lastSeen ? dateOnly(bucket.lastSeen) : "",
        sparkline: buildSparkline(bucket.dailyCounts),
        top_doc_ids: topDocs.map((doc) => doc.id),
        top_docs: topDocs,
        sources: [LIVE_NEWS_SOURCE_KIND],
      };
    })
    .sort((a, b) => b.recent_mentions - a.recent_mentions || a.label.localeCompare(b.label));
}

async function loadLiveNewsTopicTrends(days: number): Promise<TrendItem[]> {
  try {
    const [articles, topicRules] = await Promise.all([
      getRecentArticles({ limit: LIVE_NEWS_ARTICLE_LIMIT }),
      getTopicRules(true),
    ]);
    return buildLiveNewsTopicTrends(articles, topicRules, days);
  } catch {
    return [];
  }
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
    const liveNewsTrends = await loadLiveNewsTopicTrends(rangeDays(range));

    let trends: TrendItem[] = [...liveNewsTrends, ...payload.trends];

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
      generated_at: liveNewsTrends.length > 0 ? new Date().toISOString() : payload.generated_at,
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

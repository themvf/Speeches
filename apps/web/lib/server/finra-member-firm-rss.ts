import registry from "@/lib/generated/finra-member-firms.json";
import { findFinraMemberFirmMatches, finraMemberFirmNewsSearchTerms } from "@/lib/server/finra-member-firm-matcher";
import { fetchRssFeed, type RssArticle } from "@/lib/server/rss-fetcher";

export const FINRA_MEMBER_FIRM_NEWS_FEED_KEY = "google_news_finra_member_firms";
export const FINRA_MEMBER_FIRM_NEWS_LABEL = "Google News: FINRA Member Firms";

type RegistryFirm = {
  name: string;
  rssUrl: string;
};

type FirmFetchResult = {
  firmName: string;
  fetched: number;
  articles: RssArticle[];
  error?: string;
};

export type FinraMemberFirmRssBatch = {
  feedKey: typeof FINRA_MEMBER_FIRM_NEWS_FEED_KEY;
  label: typeof FINRA_MEMBER_FIRM_NEWS_LABEL;
  firmCount: number;
  batchSize: number;
  offset: number;
  fetched: number;
  matched: number;
  filtered: number;
  failed: number;
  articles: RssArticle[];
};

const DEFAULT_BATCH_SIZE = 16;
const MAX_BATCH_SIZE = 32;
const RSS_ITEMS_PER_FIRM = 5;
const RSS_FETCH_TIMEOUT_MS = 4_500;
// Must track the Vercel dispatcher's cron interval in apps/web/vercel.json.
// The batch offset is derived from floor(now / BATCH_SLOT_MS), so a slot
// shorter than the cron interval advances the offset by more than one batch
// per invocation and skips the firms in between entirely - a silent coverage
// hole, not just a slower rotation. At 30 minutes and 16 firms per batch,
// 3,194 firms take ~100h to cycle, which stays inside the `when:7d` window
// each Google News query already uses, so coverage remains complete.
export const BATCH_SLOT_MS = 30 * 60_000;
const CONCURRENCY = 8;

function configuredBatchSize(): number {
  const raw = Number.parseInt(process.env.FINRA_MEMBER_FIRM_RSS_BATCH_SIZE || "", 10);
  if (!Number.isFinite(raw) || raw <= 0) return DEFAULT_BATCH_SIZE;
  return Math.max(1, Math.min(MAX_BATCH_SIZE, raw));
}

function firmBatch(firms: RegistryFirm[], now: Date): { selected: RegistryFirm[]; offset: number } {
  if (firms.length === 0) return { selected: [], offset: 0 };
  const batchSize = Math.min(configuredBatchSize(), firms.length);
  const slot = Math.floor(now.getTime() / BATCH_SLOT_MS);
  const offset = (slot * batchSize) % firms.length;
  const selected = Array.from({ length: batchSize }, (_, index) => firms[(offset + index) % firms.length]);
  return { selected, offset };
}

function articleMatchesFirm(article: RssArticle, firmName: string): boolean {
  return findFinraMemberFirmMatches(article, 8).some((match) => match.name === firmName);
}

function googleNewsRssUrlForFirm(firm: RegistryFirm): string {
  const searchTerms = finraMemberFirmNewsSearchTerms(firm.name);
  if (searchTerms.length <= 1) return firm.rssUrl;
  const query = `${searchTerms.map((term) => `"${term.replace(/"/g, "")}"`).join(" OR ")} when:7d`;
  return `https://news.google.com/rss/search?q=${encodeURIComponent(query)}&hl=en-US&gl=US&ceid=US:en`;
}

async function mapWithConcurrency<T, R>(items: T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const results: R[] = [];
  let nextIndex = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (nextIndex < items.length) {
      const index = nextIndex;
      nextIndex += 1;
      results[index] = await fn(items[index]);
    }
  });
  await Promise.all(workers);
  return results;
}

function dedupeArticles(articles: RssArticle[]): RssArticle[] {
  const seen = new Set<string>();
  const deduped: RssArticle[] = [];
  for (const article of articles) {
    const key = `${article.guid || ""}|${article.url || ""}`.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(article);
  }
  return deduped;
}

async function fetchFirm(firm: RegistryFirm): Promise<FirmFetchResult> {
  try {
    const articles = await fetchRssFeed(googleNewsRssUrlForFirm(firm), RSS_ITEMS_PER_FIRM, RSS_FETCH_TIMEOUT_MS);
    const firmMatchedArticles = articles.filter((article) => articleMatchesFirm(article, firm.name));
    return {
      firmName: firm.name,
      fetched: articles.length,
      articles: firmMatchedArticles,
    };
  } catch (error) {
    return {
      firmName: firm.name,
      fetched: 0,
      articles: [],
      error: String(error),
    };
  }
}

export async function fetchFinraMemberFirmRssBatch(now = new Date()): Promise<FinraMemberFirmRssBatch> {
  const firms = (registry.firms as RegistryFirm[])
    .filter((firm) => firm.name && firm.rssUrl);
  const { selected, offset } = firmBatch(firms, now);
  const results = await mapWithConcurrency(selected, CONCURRENCY, fetchFirm);
  const matchedArticles = dedupeArticles(results.flatMap((result) => result.articles));
  const fetched = results.reduce((sum, result) => sum + result.fetched, 0);
  const failed = results.filter((result) => result.error).length;

  return {
    feedKey: FINRA_MEMBER_FIRM_NEWS_FEED_KEY,
    label: FINRA_MEMBER_FIRM_NEWS_LABEL,
    firmCount: firms.length,
    batchSize: selected.length,
    offset,
    fetched,
    matched: matchedArticles.length,
    filtered: fetched - matchedArticles.length,
    failed,
    articles: matchedArticles,
  };
}

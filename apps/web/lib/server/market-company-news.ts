import type { RssArticle } from "./rss-fetcher.ts";
import { isEnglishRssArticle } from "./rss-language-filter.ts";
import type { MarketSectorCompany } from "./market-sector-companies.ts";
import type { CompanyNewsArticle, CompanyNewsCatalyst } from "./types.ts";

const GOOGLE_NEWS_BASE_URL = "https://news.google.com/rss/search";
const MAX_SNIPPET_LENGTH = 280;

function quoteSearchTerm(value: string): string {
  return `"${value.replace(/"/g, "").trim()}"`;
}

export function buildCompanyNewsRssUrl(company: MarketSectorCompany, days: 7 | 30): string {
  const companyTerms = [company.name, ...(company.aliases ?? [])]
    .filter(Boolean)
    .map(quoteSearchTerm)
    .join(" OR ");
  const query = `(${companyTerms}) (stock OR shares OR earnings OR company) when:${days}d`;
  const params = new URLSearchParams({
    q: query,
    hl: "en-US",
    gl: "US",
    ceid: "US:en",
  });
  return `${GOOGLE_NEWS_BASE_URL}?${params.toString()}`;
}

function normalizeText(value: string): string {
  return value
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function containsTerm(haystack: string, term: string): boolean {
  return ` ${haystack} `.includes(` ${term} `);
}

function titleAndPublisher(article: RssArticle): { title: string; publisher: string } {
  const separator = article.title.lastIndexOf(" - ");
  if (separator <= 0) {
    return { title: article.title.trim(), publisher: article.author.trim() || "Google News" };
  }
  const title = article.title.slice(0, separator).trim();
  const suffix = article.title.slice(separator + 3).trim();
  return { title, publisher: article.author.trim() || suffix || "Google News" };
}

function compactSnippet(value: string): string {
  const compact = value.replace(/\s+/g, " ").trim();
  if (compact.length <= MAX_SNIPPET_LENGTH) return compact;
  return `${compact.slice(0, MAX_SNIPPET_LENGTH - 1).trimEnd()}…`;
}

const CATALYST_RULES: { label: CompanyNewsCatalyst; pattern: RegExp }[] = [
  { label: "M&A", pattern: /\b(?:merger|merge[sd]?|acquisition|acquire[sd]?|takeover|buyout|deal talks?)\b/i },
  { label: "Analyst Rating", pattern: /\b(?:analyst|upgrade[sd]?|downgrade[sd]?|price target|rating|outperform|underperform)\b/i },
  { label: "Earnings", pattern: /\b(?:earnings|revenue|profit|quarterly|quarter|guidance|forecast|eps|sales beat|sales miss)\b/i },
  { label: "Litigation", pattern: /\b(?:lawsuit|litigation|sues?|sued|court|settlement|class action|jury|appeal)\b/i },
  { label: "Regulation", pattern: /\b(?:regulator|regulation|regulatory|antitrust|sec probe|ftc|doj|investigation|compliance|approval)\b/i },
  { label: "Management", pattern: /\b(?:ceo|cfo|chief executive|executive|appoint(?:s|ed|ment)?|resign(?:s|ed|ation)?|board of directors|leadership)\b/i },
  { label: "Product", pattern: /\b(?:product|launch(?:es|ed)?|unveil(?:s|ed)?|release(?:s|d)?|platform|service|device|drug|software|model)\b/i },
];

export function classifyCompanyNewsCatalyst(title: string, description = ""): CompanyNewsCatalyst | null {
  const text = `${title} ${description}`;
  return CATALYST_RULES.find((rule) => rule.pattern.test(text))?.label ?? null;
}

export function companyNewsResultWindow(articles: CompanyNewsArticle[], limit: 5 | 10) {
  const visibleArticles = articles.slice(0, limit);
  return {
    articles: visibleArticles,
    availableArticleCount: articles.length,
    hasMore: articles.length > visibleArticles.length,
  };
}

export function manualRefreshDecision(
  lastRefreshAt: number | undefined,
  now: number,
  cooldownSeconds = 60,
): { allowed: boolean; remainingSeconds: number } {
  const elapsedSeconds = (now - (lastRefreshAt ?? 0)) / 1000;
  if (elapsedSeconds >= cooldownSeconds) {
    return { allowed: true, remainingSeconds: cooldownSeconds };
  }
  return { allowed: false, remainingSeconds: Math.ceil(cooldownSeconds - elapsedSeconds) };
}

type CompanyMatchSignals = {
  headlineName: boolean;
  descriptionName: boolean;
  headlineTicker: boolean;
  descriptionTicker: boolean;
};

function companyMatchSignals(
  title: string,
  description: string,
  company: MarketSectorCompany,
): CompanyMatchSignals {
  const normalizedTitle = normalizeText(title);
  const normalizedDescription = normalizeText(description);
  const nameTerms = [company.name, ...(company.aliases ?? [])]
    .map(normalizeText)
    .filter((term) => term.length >= 3);
  const ticker = normalizeText(company.symbol);
  return {
    headlineName: nameTerms.some((term) => containsTerm(normalizedTitle, term)),
    descriptionName: nameTerms.some((term) => containsTerm(normalizedDescription, term)),
    headlineTicker: ticker.length >= 3 && containsTerm(normalizedTitle, ticker),
    descriptionTicker: ticker.length >= 3 && containsTerm(normalizedDescription, ticker),
  };
}

const TIER_ONE_PUBLISHERS = [
  "reuters", "associated press", "ap news", "bloomberg", "cnbc", "wall street journal",
  "financial times", "barron s", "new york times", "washington post",
];
const TIER_TWO_PUBLISHERS = [
  "yahoo finance", "marketwatch", "forbes", "business insider", "fortune", "investors business daily",
];

function publisherScore(publisher: string): number {
  const normalized = normalizeText(publisher);
  if (TIER_ONE_PUBLISHERS.some((name) => normalized.includes(name))) return 15;
  if (TIER_TWO_PUBLISHERS.some((name) => normalized.includes(name))) return 8;
  return 3;
}

function relevanceScore(
  signals: CompanyMatchSignals,
  publisher: string,
  catalyst: CompanyNewsCatalyst | null,
  publishedAt: Date,
  now: Date,
): number {
  const ageDays = Math.max(0, (now.getTime() - publishedAt.getTime()) / (24 * 60 * 60 * 1000));
  const recency = Math.max(0, Math.round(20 - ageDays * 0.65));
  const score =
    (signals.headlineName ? 45 : 0) +
    (signals.descriptionName ? 15 : 0) +
    (signals.headlineTicker ? 10 : 0) +
    (signals.descriptionTicker ? 5 : 0) +
    publisherScore(publisher) +
    recency +
    (catalyst ? 5 : 0);
  return Math.min(100, score);
}

export function normalizeCompanyNewsArticles(
  articles: RssArticle[],
  company: MarketSectorCompany,
  now = new Date(),
): CompanyNewsArticle[] {
  const oldestAllowed = now.getTime() - 31 * 24 * 60 * 60 * 1000;
  const newestAllowed = now.getTime() + 24 * 60 * 60 * 1000;
  const candidates: CompanyNewsArticle[] = [];

  for (const article of articles) {
    if (!article.publishedAt || !isEnglishRssArticle(article)) continue;
    const publishedAt = article.publishedAt.getTime();
    if (publishedAt < oldestAllowed || publishedAt > newestAllowed) continue;

    const { title, publisher } = titleAndPublisher(article);
    const signals = companyMatchSignals(title, article.description, company);
    if (!Object.values(signals).some(Boolean)) continue;
    const catalyst = classifyCompanyNewsCatalyst(title, article.description);

    candidates.push({
      title,
      publisher,
      url: article.url,
      snippet: compactSnippet(article.description),
      publishedAt: article.publishedAt.toISOString(),
      relevanceScore: relevanceScore(signals, publisher, catalyst, article.publishedAt, now),
      catalyst,
    });
  }

  candidates.sort((a, b) =>
    b.relevanceScore - a.relevanceScore || Date.parse(b.publishedAt) - Date.parse(a.publishedAt)
  );

  const seenUrls = new Set<string>();
  const seenTitles = new Set<string>();
  return candidates.filter((article) => {
    const urlKey = article.url.trim().toLowerCase();
    const titleKey = normalizeText(article.title);
    if (!urlKey || !titleKey || seenUrls.has(urlKey) || seenTitles.has(titleKey)) return false;
    seenUrls.add(urlKey);
    seenTitles.add(titleKey);
    return true;
  });
}

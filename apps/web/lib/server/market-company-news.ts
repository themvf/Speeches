import type { RssArticle } from "./rss-fetcher.ts";
import { isEnglishRssArticle } from "./rss-language-filter.ts";
import type { MarketSectorCompany } from "./market-sector-companies.ts";
import type { CompanyNewsArticle } from "./types.ts";

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

function articleMatchesCompany(article: RssArticle, company: MarketSectorCompany): boolean {
  const haystack = ` ${normalizeText(`${article.title} ${article.description}`)} `;
  const nameMatch = [company.name, ...(company.aliases ?? [])]
    .map(normalizeText)
    .filter((term) => term.length >= 3)
    .some((term) => haystack.includes(` ${term} `));
  if (nameMatch) return true;

  const ticker = normalizeText(company.symbol);
  return ticker.length >= 3 && haystack.includes(` ${ticker} `);
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

export function normalizeCompanyNewsArticles(
  articles: RssArticle[],
  company: MarketSectorCompany,
  now = new Date(),
): CompanyNewsArticle[] {
  const oldestAllowed = now.getTime() - 31 * 24 * 60 * 60 * 1000;
  const newestAllowed = now.getTime() + 24 * 60 * 60 * 1000;
  const seenUrls = new Set<string>();
  const seenTitles = new Set<string>();
  const normalized: CompanyNewsArticle[] = [];

  for (const article of articles) {
    if (!article.publishedAt || !isEnglishRssArticle(article) || !articleMatchesCompany(article, company)) continue;
    const publishedAt = article.publishedAt.getTime();
    if (publishedAt < oldestAllowed || publishedAt > newestAllowed) continue;

    const { title, publisher } = titleAndPublisher(article);
    const urlKey = article.url.trim().toLowerCase();
    const titleKey = normalizeText(title);
    if (!urlKey || !titleKey || seenUrls.has(urlKey) || seenTitles.has(titleKey)) continue;
    seenUrls.add(urlKey);
    seenTitles.add(titleKey);

    normalized.push({
      title,
      publisher,
      url: article.url,
      snippet: compactSnippet(article.description),
      publishedAt: article.publishedAt.toISOString(),
    });
  }

  return normalized.sort((a, b) => Date.parse(b.publishedAt) - Date.parse(a.publishedAt));
}

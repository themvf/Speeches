import { unstable_cache } from "next/cache";

import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import { buildCompanyNewsRssUrl, normalizeCompanyNewsArticles } from "@/lib/server/market-company-news";
import { findMarketSectorCompany } from "@/lib/server/market-sector-companies";
import { fetchRssFeed } from "@/lib/server/rss-fetcher";
import type { MarketCompanyNewsData } from "@/lib/server/types";

export const runtime = "nodejs";

const CACHE_SECONDS = 15 * 60;
const MAX_ARTICLES = 5;
const RSS_TIMEOUT_MS = 5_000;
const inflight = new Map<string, Promise<MarketCompanyNewsData>>();

async function loadCompanyNews(symbol: string): Promise<MarketCompanyNewsData> {
  const company = findMarketSectorCompany(symbol);
  if (!company) throw new Error(`Unknown market sector symbol: ${symbol}`);

  const sevenDayArticles = await fetchRssFeed(buildCompanyNewsRssUrl(company, 7), 20, RSS_TIMEOUT_MS);
  let articles = normalizeCompanyNewsArticles(sevenDayArticles, company);
  let searchedDays: 7 | 30 = 7;
  let warning: string | undefined;

  if (articles.length < MAX_ARTICLES) {
    try {
      const thirtyDayArticles = await fetchRssFeed(buildCompanyNewsRssUrl(company, 30), 30, RSS_TIMEOUT_MS);
      articles = normalizeCompanyNewsArticles([...sevenDayArticles, ...thirtyDayArticles], company);
      searchedDays = 30;
    } catch {
      warning = "The extended 30-day Google News search was unavailable.";
    }
  }

  return {
    symbol: company.symbol,
    companyName: company.name,
    articles: articles.slice(0, MAX_ARTICLES),
    provider: "Google News RSS",
    searchedDays,
    generatedAt: new Date().toISOString(),
    ...(warning ? { warning } : {}),
  };
}

const loadCachedCompanyNews = unstable_cache(
  loadCompanyNews,
  ["market-company-news-v1"],
  { revalidate: CACHE_SECONDS },
);

function getCompanyNews(symbol: string): Promise<MarketCompanyNewsData> {
  const normalizedSymbol = symbol.trim().toUpperCase();
  const existing = inflight.get(normalizedSymbol);
  if (existing) return existing;

  const pending = loadCachedCompanyNews(normalizedSymbol)
    .finally(() => inflight.delete(normalizedSymbol));
  inflight.set(normalizedSymbol, pending);
  return pending;
}

export async function GET(req: Request) {
  const requestId = createRequestId();
  const symbol = new URL(req.url).searchParams.get("symbol")?.trim().toUpperCase() ?? "";
  if (!symbol) return fail("A company symbol is required.", "MISSING_SYMBOL", 400, requestId);
  if (!findMarketSectorCompany(symbol)) {
    return fail("That symbol is not available on the Sectors page.", "UNKNOWN_SYMBOL", 404, requestId);
  }

  try {
    return ok(await getCompanyNews(symbol), requestId);
  } catch (error) {
    console.error("[market/company-news] Google News RSS lookup failed", { symbol, error });
    return fail("Recent company news is temporarily unavailable.", "COMPANY_NEWS_UNAVAILABLE", 502, requestId);
  }
}

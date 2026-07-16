import { revalidateTag, unstable_cache } from "next/cache";

import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import {
  buildCompanyNewsRssUrl,
  companyNewsResultWindow,
  manualRefreshDecision,
  normalizeCompanyNewsArticles,
} from "@/lib/server/market-company-news";
import { findMarketSectorCompany } from "@/lib/server/market-sector-companies";
import { fetchRssFeed } from "@/lib/server/rss-fetcher";
import type { MarketCompanyNewsData } from "@/lib/server/types";

export const runtime = "nodejs";

const CACHE_SECONDS = 15 * 60;
const DEFAULT_ARTICLES = 5;
const MAX_ARTICLES = 10;
const REFRESH_COOLDOWN_SECONDS = 60;
const RSS_TIMEOUT_MS = 5_000;
const inflight = new Map<string, Promise<MarketCompanyNewsData>>();
const lastManualRefresh = new Map<string, number>();

function cacheTag(symbol: string): string {
  return `market-company-news:${symbol}`;
}

async function loadCompanyNews(symbol: string): Promise<MarketCompanyNewsData> {
  const company = findMarketSectorCompany(symbol);
  if (!company) throw new Error(`Unknown market sector symbol: ${symbol}`);

  const sevenDayArticles = await fetchRssFeed(buildCompanyNewsRssUrl(company, 7), 20, RSS_TIMEOUT_MS);
  let articles = normalizeCompanyNewsArticles(sevenDayArticles, company);
  let searchedDays: 7 | 30 = 7;
  let warning: string | undefined;

  if (articles.length < DEFAULT_ARTICLES) {
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
    availableArticleCount: Math.min(articles.length, MAX_ARTICLES),
    hasMore: false,
    ...(warning ? { warning } : {}),
  };
}

function loadCachedCompanyNews(symbol: string): Promise<MarketCompanyNewsData> {
  return unstable_cache(
    () => loadCompanyNews(symbol),
    ["market-company-news-v2", symbol],
    { revalidate: CACHE_SECONDS, tags: [cacheTag(symbol)] },
  )();
}

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
  const params = new URL(req.url).searchParams;
  const symbol = params.get("symbol")?.trim().toUpperCase() ?? "";
  const limit = params.get("limit") === "10" ? MAX_ARTICLES : DEFAULT_ARTICLES;
  const refreshRequested = ["1", "true"].includes(params.get("refresh")?.toLowerCase() ?? "");
  if (!symbol) return fail("A company symbol is required.", "MISSING_SYMBOL", 400, requestId);
  if (!findMarketSectorCompany(symbol)) {
    return fail("That symbol is not available on the Sectors page.", "UNKNOWN_SYMBOL", 404, requestId);
  }

  let refreshStatus: "refreshed" | "throttled" | undefined;
  let refreshCooldownSeconds: number | undefined;
  let refreshGranted = false;
  try {
    if (refreshRequested) {
      const now = Date.now();
      const decision = manualRefreshDecision(lastManualRefresh.get(symbol), now, REFRESH_COOLDOWN_SECONDS);
      refreshCooldownSeconds = decision.remainingSeconds;
      if (decision.allowed) {
        refreshGranted = true;
        refreshStatus = "refreshed";
        lastManualRefresh.set(symbol, now);
        revalidateTag(cacheTag(symbol));
      } else {
        refreshStatus = "throttled";
      }
    }

    const data = await getCompanyNews(symbol);
    const resultWindow = companyNewsResultWindow(data.articles, limit);
    const throttleWarning = refreshStatus === "throttled"
      ? `Fresh results can be requested again in ${refreshCooldownSeconds} seconds.`
      : undefined;
    return ok({
      ...data,
      ...resultWindow,
      ...(refreshStatus ? { refreshStatus } : {}),
      ...(refreshCooldownSeconds ? { refreshCooldownSeconds } : {}),
      warning: [data.warning, throttleWarning].filter(Boolean).join(" ") || undefined,
    }, requestId);
  } catch (error) {
    if (refreshGranted) lastManualRefresh.delete(symbol);
    console.error("[market/company-news] Google News RSS lookup failed", { symbol, error });
    return fail("Recent company news is temporarily unavailable.", "COMPANY_NEWS_UNAVAILABLE", 502, requestId);
  }
}

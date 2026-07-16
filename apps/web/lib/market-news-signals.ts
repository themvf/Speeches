import type {
  CompanyNewsArticle,
  CompanyNewsCatalyst,
  CompanyNewsSourceTier,
  MarketCompanyNewsData,
  SectorStock,
} from "./server/types.ts";

export type SectorStockSort = "move" | "latest" | "relevance";
export type SectorPriceFilter = "all" | "gainers" | "losers";
export type SourceTierFilter = "All" | CompanyNewsSourceTier;

export interface SectorNewsControls {
  sort: SectorStockSort;
  price: SectorPriceFilter;
  catalyst: "all" | CompanyNewsCatalyst;
  loadedOnly: boolean;
}

export interface ArticleQualityControls {
  hidePressReleases: boolean;
  minSourceTier: SourceTierFilter;
}

const SOURCE_TIER_RANK: Record<CompanyNewsSourceTier, number> = {
  Other: 0,
  Established: 1,
  Premium: 2,
};

export function dominantCatalyst(news: MarketCompanyNewsData | null | undefined): CompanyNewsCatalyst | null {
  if (!news) return null;
  const counts = new Map<CompanyNewsCatalyst, number>();
  for (const article of news.articles) {
    if (article.catalyst) counts.set(article.catalyst, (counts.get(article.catalyst) ?? 0) + 1);
  }
  return [...counts.entries()].sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))[0]?.[0] ?? null;
}

export function summarizeLoadedSectorNews(
  stocks: SectorStock[],
  newsBySymbol: Record<string, MarketCompanyNewsData>,
) {
  const loadedStocks = stocks.filter((stock) => newsBySymbol[stock.symbol]);
  const catalystCounts = new Map<CompanyNewsCatalyst, number>();
  let totalArticles = 0;
  let gainers = 0;
  let losers = 0;

  for (const stock of loadedStocks) {
    const news = newsBySymbol[stock.symbol];
    totalArticles += news.availableArticleCount;
    if (stock.pct > 0.05) gainers += 1;
    else if (stock.pct < -0.05) losers += 1;
    for (const article of news.articles) {
      if (article.catalyst) catalystCounts.set(article.catalyst, (catalystCounts.get(article.catalyst) ?? 0) + 1);
    }
  }

  const topCatalyst = [...catalystCounts.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))[0]?.[0] ?? null;
  const priceReaction = gainers > 0 && losers > 0 ? "Mixed" : gainers > 0 ? "Up" : losers > 0 ? "Down" : "Flat";
  return { loadedCompanies: loadedStocks.length, totalArticles, dominantCatalyst: topCatalyst, priceReaction, gainers, losers };
}

export function isPossiblePriceCatalyst(
  article: CompanyNewsArticle,
  stockPct: number,
  now = Date.now(),
  windowHours = 36,
): boolean {
  const ageMs = now - Date.parse(article.publishedAt);
  return article.catalyst !== null
    && Math.abs(stockPct) >= 1
    && ageMs >= 0
    && ageMs <= windowHours * 60 * 60 * 1000;
}

export function filterCompanyNewsArticles(
  articles: CompanyNewsArticle[],
  controls: ArticleQualityControls,
): CompanyNewsArticle[] {
  const minimumRank = controls.minSourceTier === "All" ? 0 : SOURCE_TIER_RANK[controls.minSourceTier];
  return articles.filter((article) =>
    (!controls.hidePressReleases || !article.isPressRelease)
    && SOURCE_TIER_RANK[article.sourceTier] >= minimumRank
  );
}

export function sortAndFilterSectorStocks(
  stocks: SectorStock[],
  newsBySymbol: Record<string, MarketCompanyNewsData>,
  controls: SectorNewsControls,
): SectorStock[] {
  const filtered = stocks.filter((stock) => {
    const news = newsBySymbol[stock.symbol];
    if (controls.loadedOnly && !news) return false;
    if (controls.price === "gainers" && stock.pct <= 0) return false;
    if (controls.price === "losers" && stock.pct >= 0) return false;
    if (controls.catalyst !== "all" && !news?.articles.some((article) => article.catalyst === controls.catalyst)) return false;
    return true;
  });

  return filtered.sort((left, right) => {
    if (controls.sort === "move") return Math.abs(right.pct) - Math.abs(left.pct);
    const leftNews = newsBySymbol[left.symbol];
    const rightNews = newsBySymbol[right.symbol];
    if (controls.sort === "relevance") {
      const best = (news: MarketCompanyNewsData | undefined) => Math.max(0, ...(news?.articles.map((article) => article.relevanceScore) ?? []));
      return best(rightNews) - best(leftNews) || Math.abs(right.pct) - Math.abs(left.pct);
    }
    const latest = (news: MarketCompanyNewsData | undefined) => Math.max(0, ...(news?.articles.map((article) => Date.parse(article.publishedAt)) ?? []));
    return latest(rightNews) - latest(leftNews) || Math.abs(right.pct) - Math.abs(left.pct);
  });
}

export function isSessionNewsCacheFresh(savedAt: number, now = Date.now(), ttlMs = 15 * 60 * 1000): boolean {
  return savedAt <= now && now - savedAt < ttlMs;
}

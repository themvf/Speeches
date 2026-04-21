import type { IntelligenceEvidenceArticle } from "./intelligence-types.ts";

export function sourceSearchUrl(article: IntelligenceEvidenceArticle): string {
  const query = encodeURIComponent(`${article.source} ${article.headline}`);

  if (article.source === "Reuters") {
    return `https://www.reuters.com/site-search/?query=${query}`;
  }
  if (article.source === "Bloomberg") {
    return `https://www.bloomberg.com/search?query=${query}`;
  }
  if (article.source === "Financial Times") {
    return `https://www.ft.com/search?q=${query}`;
  }
  if (article.source === "CNBC") {
    return `https://www.cnbc.com/search/?query=${query}`;
  }
  if (article.source === "MarketWatch") {
    return `https://www.marketwatch.com/search?q=${query}`;
  }
  if (article.source === "Dow Jones" || article.source === "Wall Street Journal") {
    return `https://www.wsj.com/search?query=${query}`;
  }
  if (article.source === "Lloyd's List") {
    return `https://www.lloydslist.com/search?query=${query}`;
  }
  if (article.source === "CoinDesk") {
    return `https://www.coindesk.com/search?s=${query}`;
  }

  return `https://www.google.com/search?q=${query}`;
}

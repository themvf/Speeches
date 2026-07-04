import type { StoredRssArticle } from "@/lib/server/neon";
import { shouldRefreshFeedAnalysisForDeepSeek } from "@/lib/server/feed-analysis";
import { findFinraMemberFirmMatches } from "@/lib/server/finra-member-firm-matcher";

function isCurrentDeepSeekAnalysis(article: StoredRssArticle): boolean {
  return Boolean(article.analysis) && !shouldRefreshFeedAnalysisForDeepSeek(article.analysis);
}

function normalizeDedupeText(value: string | null | undefined): string {
  return String(value || "")
    .toLowerCase()
    .replace(/&amp;/g, "&")
    .replace(/['"\u2018\u2019\u201c\u201d]/g, "")
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function canonicalArticleUrl(value: string | null | undefined): string {
  const raw = String(value || "").trim();
  if (!raw) return "";
  try {
    const url = new URL(raw);
    url.hash = "";
    for (const key of [...url.searchParams.keys()]) {
      const lower = key.toLowerCase();
      if (
        lower.startsWith("utm_") ||
        ["fbclid", "gclid", "mc_cid", "mc_eid", "cmpid", "smid", "ref", "source"].includes(lower)
      ) {
        url.searchParams.delete(key);
      }
    }
    url.hostname = url.hostname.toLowerCase().replace(/^www\./, "");
    url.pathname = url.pathname.replace(/\/+$/, "");
    return `${url.hostname}${url.pathname}${url.searchParams.toString() ? `?${url.searchParams.toString()}` : ""}`.toLowerCase();
  } catch {
    return normalizeDedupeText(raw);
  }
}

function articleDedupeKeys(article: StoredRssArticle): string[] {
  const urlKey = canonicalArticleUrl(article.url);
  const titleKey = normalizeDedupeText(article.title);
  const descriptionKey = normalizeDedupeText(article.description).slice(0, 120);
  const authorKey = normalizeDedupeText(article.author || article.feed_label || article.feed_key);
  return [
    urlKey ? `url:${urlKey}` : "",
    titleKey && authorKey ? `title-source:${titleKey}:${authorKey}` : "",
    titleKey && descriptionKey ? `title-desc:${titleKey}:${descriptionKey}` : "",
  ].filter(Boolean);
}

function dedupeFeedArticles(articles: StoredRssArticle[]): StoredRssArticle[] {
  const seen = new Set<string>();
  const out: StoredRssArticle[] = [];
  for (const article of articles) {
    const keys = articleDedupeKeys(article);
    if (keys.some((key) => seen.has(key))) {
      continue;
    }
    for (const key of keys) {
      seen.add(key);
    }
    out.push(article);
  }
  return out;
}

export function compactFeedArticles(articles: StoredRssArticle[]): StoredRssArticle[] {
  return dedupeFeedArticles(articles).map((article) => {
    const matchedFinraFirms = findFinraMemberFirmMatches(article).map((match) => match.name);
    const firmAnnotatedArticle = matchedFinraFirms.length
      ? { ...article, matched_finra_firms: matchedFinraFirms }
      : article;

    if (!article.analysis) {
      return firmAnnotatedArticle;
    }

    if (!isCurrentDeepSeekAnalysis(article)) {
      return {
        ...firmAnnotatedArticle,
        analysis: null,
      };
    }

    return {
      ...firmAnnotatedArticle,
      analysis: {
        ...article.analysis,
        source_hash: "",
        analysis_text: "",
        error: article.analysis.error ? article.analysis.error.slice(0, 300) : "",
      },
    };
  });
}

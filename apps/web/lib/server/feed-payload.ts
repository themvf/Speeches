import type { StoredRssArticle } from "@/lib/server/neon";
import { shouldRefreshFeedAnalysisForDeepSeek } from "@/lib/server/feed-analysis";
import { findFinraMemberFirmMatches } from "@/lib/server/finra-member-firm-matcher";

function isCurrentDeepSeekAnalysis(article: StoredRssArticle): boolean {
  return Boolean(article.analysis) && !shouldRefreshFeedAnalysisForDeepSeek(article.analysis);
}

export function compactFeedArticles(articles: StoredRssArticle[]): StoredRssArticle[] {
  return articles.map((article) => {
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

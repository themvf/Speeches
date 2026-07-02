import type { StoredRssArticle } from "@/lib/server/neon";
import { shouldRefreshFeedAnalysisForDeepSeek } from "@/lib/server/feed-analysis";

function isCurrentDeepSeekAnalysis(article: StoredRssArticle): boolean {
  return Boolean(article.analysis) && !shouldRefreshFeedAnalysisForDeepSeek(article.analysis);
}

export function compactFeedArticles(articles: StoredRssArticle[]): StoredRssArticle[] {
  return articles.map((article) => {
    if (!article.analysis) {
      return article;
    }

    if (!isCurrentDeepSeekAnalysis(article)) {
      return {
        ...article,
        analysis: null,
      };
    }

    return {
      ...article,
      analysis: {
        ...article.analysis,
        source_hash: "",
        analysis_text: "",
        error: article.analysis.error ? article.analysis.error.slice(0, 300) : "",
      },
    };
  });
}

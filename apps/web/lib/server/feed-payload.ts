import type { StoredRssArticle } from "@/lib/server/neon";

function isCurrentDeepSeekAnalysis(article: StoredRssArticle): boolean {
  const model = String(article.analysis?.model || "").trim().toLowerCase();
  return Boolean(article.analysis) && !article.analysis?.fallback && model.startsWith("deepseek");
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

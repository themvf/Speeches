import type { StoredRssArticle } from "@/lib/server/neon";

export function compactFeedArticles(articles: StoredRssArticle[]): StoredRssArticle[] {
  return articles.map((article) => {
    if (!article.analysis) {
      return article;
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

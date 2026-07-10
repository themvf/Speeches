import { NextRequest, NextResponse } from "next/server";
import { compileKeywords, matchingKeywordsForArticle, parseKeywords } from "@/lib/intel-topic-matching";
import { getRecentRssArticlesForBacktest } from "@/lib/server/neon";

export const dynamic = "force-dynamic";

const BACKTEST_WINDOW_DAYS = 30;
const BACKTEST_ARTICLE_LIMIT = 1500;
const SAMPLE_LIMIT = 15;

export async function POST(req: NextRequest): Promise<NextResponse> {
  let body: { keywords?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ ok: false, error: "Invalid JSON" }, { status: 400 });
  }

  const keywords = parseKeywords(String(body.keywords ?? ""));
  if (keywords.length === 0) {
    return NextResponse.json({ ok: false, error: "keywords is required" }, { status: 400 });
  }

  try {
    const articles = await getRecentRssArticlesForBacktest(BACKTEST_WINDOW_DAYS, BACKTEST_ARTICLE_LIMIT);
    const matchers = compileKeywords(keywords);
    const perKeywordCounts: Record<string, number> = Object.fromEntries(keywords.map((k) => [k, 0]));
    const samples: Array<{ id: number; title: string; url: string; feed_key: string; matched_keywords: string[] }> = [];
    let matchedCount = 0;

    for (const article of articles) {
      const matched = matchingKeywordsForArticle(matchers, article);
      if (matched.length === 0) continue;
      matchedCount += 1;
      for (const keyword of matched) {
        perKeywordCounts[keyword] = (perKeywordCounts[keyword] ?? 0) + 1;
      }
      if (samples.length < SAMPLE_LIMIT) {
        samples.push({
          id: article.id,
          title: article.title,
          url: article.url,
          feed_key: article.feed_key,
          matched_keywords: matched,
        });
      }
    }

    return NextResponse.json({
      ok: true,
      data: {
        window_days: BACKTEST_WINDOW_DAYS,
        articles_scanned: articles.length,
        matched_count: matchedCount,
        keyword_counts: perKeywordCounts,
        samples,
      },
    });
  } catch (err) {
    return NextResponse.json({ ok: false, error: String(err) }, { status: 500 });
  }
}

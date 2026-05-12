import { type NextRequest, NextResponse } from "next/server";
import {
  getRecapSettings,
  getTopicRules,
  getRecentArticles,
  saveRecapRows,
  type StoredRssArticle,
} from "@/lib/server/neon";
import { getOpenAiConfig } from "@/lib/server/env";
import { getTopicMatches, normalizeTopicRules } from "@/lib/intel-topic-matching";

export const dynamic = "force-dynamic";
export const maxDuration = 60;

const MAX_ARTICLES_PER_TOPIC = 20;
const MIN_MATCH_SCORE = 100; // requires a title match; description-only matches are too noisy for recap

async function generateTopicSummary(
  topicLabel: string,
  articles: StoredRssArticle[],
  cfg: { apiKey: string; model: string; baseUrl: string }
): Promise<string> {
  const articleList = articles
    .slice(0, MAX_ARTICLES_PER_TOPIC)
    .map((a) => `- ${a.title}${a.description ? `: ${a.description}` : ""}`)
    .join("\n");

  const prompt = `You are a regulatory intelligence analyst.\n\nSummarize the following ${articles.length} news articles about "${topicLabel}" from the past 24 hours. Use exactly this format:\n\n**Executive Summary:** [2–3 sentence overview of the most important developments.]\n\n**Key Points:**\n- [First key point]\n- [Second key point]\n- [Third key point]\n- [Add 1–2 more if warranted]\n\nEach bullet must be on its own line starting with "- ". Be direct and analytical. Synthesize — do not quote or list articles individually. If an article is only tangentially related, incorporate any relevant angle rather than refusing to summarize.\n\nArticles:\n${articleList}`;

  const res = await fetch(`${cfg.baseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${cfg.apiKey}`,
    },
    body: JSON.stringify({
      model: cfg.model,
      messages: [{ role: "user", content: prompt }],
      max_tokens: 600,
      temperature: 0.4,
    }),
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`OpenAI error ${res.status}: ${body.slice(0, 300)}`);
  }
  const json = (await res.json()) as { choices: { message: { content: string } }[] };
  return json.choices[0]?.message?.content?.trim() ?? "";
}

export async function POST(req: NextRequest): Promise<NextResponse> {
  try {
    const cfg = getOpenAiConfig();
    if (!cfg.apiKey) {
      return NextResponse.json({ ok: false, error: "OpenAI not configured (OPENAI_API_KEY missing)" }, { status: 500 });
    }

    const body = await req.json().catch(() => ({})) as { date?: string };
    const todayIso = new Date().toISOString().split("T")[0] as string;
    const recapDate = body.date ?? todayIso;

    // For today: last 24h. For a past date: midnight-to-midnight of that day.
    let since: Date;
    let until: Date | undefined;
    if (recapDate === todayIso) {
      since = new Date(Date.now() - 24 * 60 * 60 * 1000);
    } else {
      since = new Date(recapDate + "T00:00:00Z");
      until = new Date(recapDate + "T23:59:59Z");
    }

    const [selectedTopicKeys, rawRules, articles] = await Promise.all([
      getRecapSettings(),
      getTopicRules(true),
      getRecentArticles({ limit: 400, since, until }),
    ]);

    if (selectedTopicKeys.length === 0) {
      return NextResponse.json({ ok: false, error: "No topics selected. Save topic settings first." }, { status: 400 });
    }

    const rules = normalizeTopicRules(rawRules);
    const selectedRules = rules.filter((r) => selectedTopicKeys.includes(r.topic_key));

    const topicArticleMap = new Map<string, StoredRssArticle[]>();
    for (const rule of selectedRules) {
      topicArticleMap.set(rule.topic_key, []);
    }
    for (const article of articles) {
      const matches = getTopicMatches(article, selectedRules);
      for (const { rule, score } of matches) {
        if (score >= MIN_MATCH_SCORE) {
          topicArticleMap.get(rule.topic_key)?.push(article);
        }
      }
    }

    const results: { topic_key: string; topic_label: string; article_count: number; summary: string }[] = [];

    for (const rule of selectedRules) {
      const topicArticles = topicArticleMap.get(rule.topic_key) ?? [];
      if (topicArticles.length === 0) continue;

      const summary = await generateTopicSummary(rule.label, topicArticles, cfg);

      const positive_count = topicArticles.filter((a) => a.tone_label === "positive").length;
      const negative_count = topicArticles.filter((a) => a.tone_label === "negative").length;
      const neutral_count = topicArticles.filter((a) => a.tone_label === "neutral").length;
      const sources = topicArticles.slice(0, MAX_ARTICLES_PER_TOPIC).map((a) => ({ title: a.title, url: a.url }));

      results.push({ topic_key: rule.topic_key, topic_label: rule.label, article_count: topicArticles.length, summary });

      await saveRecapRows([{
        recap_date: recapDate,
        topic_key: rule.topic_key,
        topic_label: rule.label,
        summary,
        article_count: topicArticles.length,
        positive_count,
        negative_count,
        neutral_count,
        sources,
      }]);
    }

    return NextResponse.json({ ok: true, data: { date: recapDate, topics: results } });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error("[recap/generate]", message);
    return NextResponse.json({ ok: false, error: message }, { status: 500 });
  }
}

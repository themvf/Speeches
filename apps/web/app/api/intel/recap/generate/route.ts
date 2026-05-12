import { NextResponse } from "next/server";
import {
  getRecapSettings,
  getTopicRules,
  getRecentArticles,
  saveRecapRows,
  type StoredRssArticle,
} from "@/lib/server/neon";
import { getOpenAiConfig } from "@/lib/server/env";
import { getMatchingTopics, normalizeTopicRules } from "@/lib/intel-topic-matching";

export const dynamic = "force-dynamic";
export const maxDuration = 60;

const MAX_ARTICLES_PER_TOPIC = 20;

async function generateTopicSummary(
  topicLabel: string,
  articles: StoredRssArticle[],
  cfg: { apiKey: string; model: string; baseUrl: string }
): Promise<string> {
  const articleList = articles
    .slice(0, MAX_ARTICLES_PER_TOPIC)
    .map((a) => `- ${a.title}${a.description ? `: ${a.description}` : ""}`)
    .join("\n");

  const prompt = `You are a regulatory intelligence analyst.\n\nSummarize the following ${articles.length} news articles about "${topicLabel}" from the past 24 hours in 2–3 concise paragraphs. Focus on key developments, regulatory signals, and notable trends. Synthesize — do not list or quote articles individually. Be direct and analytical.\n\nArticles:\n${articleList}`;

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

  if (!res.ok) throw new Error(`OpenAI error ${res.status}`);
  const json = (await res.json()) as { choices: { message: { content: string } }[] };
  return json.choices[0]?.message?.content?.trim() ?? "";
}

export async function POST(): Promise<NextResponse> {
  try {
    const cfg = getOpenAiConfig();
    if (!cfg.apiKey) {
      return NextResponse.json({ ok: false, error: "OpenAI not configured (OPENAI_API_KEY missing)" }, { status: 500 });
    }

    const since = new Date(Date.now() - 24 * 60 * 60 * 1000);
    const recapDate = new Date().toISOString().split("T")[0];

    const [selectedTopicKeys, rawRules, articles] = await Promise.all([
      getRecapSettings(),
      getTopicRules(true),
      getRecentArticles({ limit: 400, since }),
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
      const matched = getMatchingTopics(article, selectedRules);
      for (const rule of matched) {
        topicArticleMap.get(rule.topic_key)?.push(article);
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
      }]);
    }

    return NextResponse.json({ ok: true, data: { date: recapDate, topics: results } });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error("[recap/generate]", message);
    return NextResponse.json({ ok: false, error: message }, { status: 500 });
  }
}

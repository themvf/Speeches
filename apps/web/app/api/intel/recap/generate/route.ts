import { type NextRequest, NextResponse } from "next/server";
import {
  deleteBlockedRssArticles,
  getRecapSettings,
  getTopicRules,
  getRecentArticles,
  saveRecapRows,
  type RecapSource,
} from "@/lib/server/neon";
import { loadCorpusDocuments, loadEnrichmentState } from "@/lib/server/data-store";
import { getOpenAiConfig } from "@/lib/server/env";
import { getTopicMatches, normalizeTopicRules, type TopicRuleView } from "@/lib/intel-topic-matching";
import { getClientIp, getGenerateGlobalLimiter, getGenerateIpLimiter, isRateLimited } from "@/lib/server/rate-limit";
import { isAllowedRssArticleForIngestion } from "@/lib/server/rss-ingestion-filter";

export const dynamic = "force-dynamic";
export const maxDuration = 60;

const MAX_ITEMS_PER_TOPIC = 20;

type RecapProviderConfig = {
  provider: "deepseek" | "openai";
  apiKey: string;
  model: string;
  baseUrl: string;
};

function readEnv(name: string, fallback = ""): string {
  return String(process.env[name] ?? fallback).trim();
}

function providerLabel(provider: RecapProviderConfig["provider"]): string {
  return provider === "deepseek" ? "DeepSeek" : "OpenAI";
}

function getRecapProviderConfig(): RecapProviderConfig {
  const explicitProvider = readEnv("RECAP_ANALYSIS_PROVIDER").toLowerCase();
  if (explicitProvider === "openai") {
    const openai = getOpenAiConfig();
    return {
      provider: "openai",
      apiKey: openai.apiKey,
      model: readEnv("RECAP_ANALYSIS_MODEL") || openai.model,
      baseUrl: openai.baseUrl,
    };
  }

  return {
    provider: "deepseek",
    apiKey: readEnv("DEEPSEEK_API") || readEnv("DEEPSEEK_API_KEY"),
    model:
      readEnv("RECAP_ANALYSIS_MODEL") ||
      readEnv("DEEPSEEK_MODEL") ||
      readEnv("DEEPSEEK_CHAT_MODEL") ||
      "deepseek-v4-flash",
    baseUrl: readEnv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
  };
}

function normalizeDocDate(dateStr: string): string | null {
  if (!dateStr) return null;
  // Already YYYY-MM-DD
  if (/^\d{4}-\d{2}-\d{2}/.test(dateStr)) return dateStr.slice(0, 10);
  // Try JS Date parsing for "May 11, 2026", "11/05/2026", etc.
  const d = new Date(dateStr);
  if (isNaN(d.getTime())) return null;
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}
// Articles require a title match (100+) to avoid false positives from passing mentions.
// Corpus docs use enrichment tags/keywords for matching — LLM-curated, so a
// description-level match (50+) is reliable.
const MIN_ARTICLE_SCORE = 100;
const MIN_CORPUS_SCORE = 50;

type RecapItem = {
  title: string;
  description: string; // used in the LLM prompt
  matchText?: string;  // overrides description for topic matching; for corpus docs = enrichment tags + keywords
  url: string;
  source_type: "article" | "document";
  source_kind?: string; // e.g. "sec_speech", "custom" — used for UI labeling
  speaker?: string;
  tone_label?: "positive" | "neutral" | "negative" | null;
};

async function generateTopicSummary(
  topicLabel: string,
  items: RecapItem[],
  cfg: RecapProviderConfig
): Promise<string> {
  const itemList = items
    .slice(0, MAX_ITEMS_PER_TOPIC)
    .map((item) => {
      const prefix = item.source_type === "document"
        ? `[Regulatory Document${item.speaker ? ` — ${item.speaker}` : ""}]`
        : "[News]";
      return `- ${prefix} ${item.title}${item.description ? `: ${item.description}` : ""}`;
    })
    .join("\n");

  const prompt = `You are a regulatory intelligence analyst.\n\nSummarize the following ${items.length} sources about "${topicLabel}" from the past 24 hours. Sources are labeled [News] or [Regulatory Document]. Use exactly this format:\n\n**Executive Summary:** [2–3 sentence overview of the most important developments.]\n\n**Key Points:**\n- [First key point]\n- [Second key point]\n- [Third key point]\n- [Add 1–2 more if warranted]\n\nEach bullet must be on its own line starting with "- ". Prioritize regulatory documents over news when relevant. Be direct and analytical. Synthesize — do not quote or list sources individually.\n\nSources:\n${itemList}`;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 45_000);
  let res: Response;
  try {
    res = await fetch(`${cfg.baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${cfg.apiKey}`,
      },
      body: JSON.stringify({
        model: cfg.model,
        messages: [{ role: "user", content: prompt }],
        max_tokens: 700,
        temperature: 0.4,
      }),
      signal: controller.signal,
    });
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error(`${providerLabel(cfg.provider)} request timed out while generating ${topicLabel}`);
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${providerLabel(cfg.provider)} error ${res.status}: ${body.slice(0, 300)}`);
  }
  const json = (await res.json()) as { choices: { message: { content: string } }[] };
  return json.choices[0]?.message?.content?.trim() ?? "";
}

function matchItemsToTopics(
  items: RecapItem[],
  rules: TopicRuleView[],
  minScore: number
): Map<string, RecapItem[]> {
  const map = new Map<string, RecapItem[]>();
  for (const rule of rules) map.set(rule.topic_key, []);

  for (const item of items) {
    // Use matchText (enrichment tags/keywords) if provided, otherwise fall back to description
    const matchInput = {
      title: item.title,
      description: item.matchText ?? item.description,
    };
    const matches = getTopicMatches(matchInput, rules);
    for (const { rule, score } of matches) {
      if (score >= minScore) {
        map.get(rule.topic_key)?.push(item);
      }
    }
  }
  return map;
}

export async function POST(req: NextRequest): Promise<NextResponse> {
  const ip = getClientIp(req.headers);
  if (await isRateLimited(getGenerateIpLimiter(), ip)) {
    return NextResponse.json({ ok: false, error: "Rate limit exceeded. Please slow down." }, { status: 429 });
  }
  if (await isRateLimited(getGenerateGlobalLimiter(), "global")) {
    return NextResponse.json({ ok: false, error: "Server is busy. Please try again shortly." }, { status: 429 });
  }

  try {
    const cfg = getRecapProviderConfig();
    if (!cfg.apiKey) {
      const missingKey = cfg.provider === "deepseek" ? "DEEPSEEK_API" : "OPENAI_API_KEY";
      return NextResponse.json({ ok: false, error: `${providerLabel(cfg.provider)} not configured (${missingKey} missing)` }, { status: 500 });
    }

    const body = await req.json().catch(() => ({})) as { date?: string };
    const todayIso = new Date().toISOString().split("T")[0] as string;
    const recapDate = body.date ?? todayIso;

    let since: Date;
    let until: Date | undefined;
    if (recapDate === todayIso) {
      since = new Date(Date.now() - 24 * 60 * 60 * 1000);
    } else {
      since = new Date(recapDate + "T00:00:00Z");
      const nextDay = new Date(since); nextDay.setUTCDate(nextDay.getUTCDate() + 1);
      until = nextDay;
    }

    const [selectedTopicKeys, rawRules, articles, corpusDocs, enrichmentState] = await Promise.all([
      getRecapSettings(),
      getTopicRules(true),
      getRecentArticles({ limit: 400, since, until }),
      loadCorpusDocuments(),
      loadEnrichmentState(),
    ]);

    if (selectedTopicKeys.length === 0) {
      return NextResponse.json({ ok: false, error: "No topics selected. Save topic settings first." }, { status: 400 });
    }

    const rules = normalizeTopicRules(rawRules);
    const selectedRules = rules.filter((r) => selectedTopicKeys.includes(r.topic_key));
    if (selectedRules.length === 0) {
      return NextResponse.json(
        {
          ok: false,
          error: rawRules.length === 0
            ? "No active recap topic rules are available. Reload the page and save recap topics again."
            : "The saved recap topics no longer match active topic rules. Save recap settings again, then generate the recap.",
        },
        { status: 400 }
      );
    }
    await deleteBlockedRssArticles(rawRules).catch((error) => {
      console.error("[recap/generate] RSS policy cleanup failed:", error);
      return 0;
    });
    const articlesForRecap = articles.filter((article) => isAllowedRssArticleForIngestion(article.feed_key, {
      guid: article.guid,
      title: article.title,
      url: article.url,
      description: article.description,
      author: article.author,
      publishedAt: article.published_at ? new Date(article.published_at) : null,
    }, rawRules));

    // RSS articles — matched by title keyword (strict threshold)
    const articleItems: RecapItem[] = articlesForRecap.map((a) => ({
      title: a.title,
      description: a.description ?? "",
      url: a.url,
      source_type: "article",
      tone_label: a.tone_label,
    }));

    // Corpus docs — filter by date, match via enrichment tags + keywords (relaxed threshold)
    const enrichmentEntries = enrichmentState.entries ?? {};
    const corpusItems: RecapItem[] = corpusDocs
      .filter((doc) => doc.metadata.full_text_available !== false)
      .filter((doc) => {
        const dateStr = doc.metadata.published_date || doc.metadata.date || "";
        return normalizeDocDate(dateStr) === recapDate;
      })
      .map((doc) => {
        const enrichment = enrichmentEntries[doc.metadata.document_id];
        const tags: string[] = enrichment?.enrichment?.tags ?? [];
        const keywords: string[] = enrichment?.enrichment?.keywords ?? [];

        // matchText = enrichment tags + keywords joined; used for topic matching
        const matchText = [...tags, ...keywords].join(" ") || undefined;

        // description = enrichment summary for the LLM prompt
        const description =
          enrichment?.enrichment?.summary ||
          doc.metadata.summary ||
          doc.content.full_text.slice(0, 400);

        return {
          title: doc.metadata.title,
          description,
          matchText,
          url: doc.metadata.url,
          source_type: "document" as const,
          source_kind: doc.metadata.source_kind || undefined,
          speaker: doc.metadata.speaker || undefined,
        };
      });

    const articleMap = matchItemsToTopics(articleItems, selectedRules, MIN_ARTICLE_SCORE);
    const corpusMap = matchItemsToTopics(corpusItems, selectedRules, MIN_CORPUS_SCORE);

    const results: { topic_key: string; topic_label: string; article_count: number; summary: string }[] = [];
    const skipped: { topic_key: string; topic_label: string }[] = [];
    const failed: { topic_key: string; topic_label: string; error: string }[] = [];

    const settled = await Promise.allSettled(selectedRules.map(async (rule) => {
      const topicItems: RecapItem[] = [
        ...(articleMap.get(rule.topic_key) ?? []),
        ...(corpusMap.get(rule.topic_key) ?? []),
      ];

      if (topicItems.length === 0) {
        skipped.push({ topic_key: rule.topic_key, topic_label: rule.label });
        return;
      }

      const summary = await generateTopicSummary(rule.label, topicItems, cfg);

      const positive_count = topicItems.filter((i) => i.tone_label === "positive").length;
      const negative_count = topicItems.filter((i) => i.tone_label === "negative").length;
      const neutral_count = topicItems.filter((i) => i.tone_label === "neutral").length;

      const sources: RecapSource[] = topicItems.slice(0, MAX_ITEMS_PER_TOPIC).map((i) => ({
        title: i.title,
        url: i.url,
        source_type: i.source_type,
        source_kind: i.source_kind,
        speaker: i.speaker,
      }));

      await saveRecapRows([{
        recap_date: recapDate,
        topic_key: rule.topic_key,
        topic_label: rule.label,
        summary,
        article_count: topicItems.length,
        positive_count,
        negative_count,
        neutral_count,
        sources,
      }]);

      results.push({ topic_key: rule.topic_key, topic_label: rule.label, article_count: topicItems.length, summary });
    }));

    settled.forEach((outcome, index) => {
      if (outcome.status === "rejected") {
        const rule = selectedRules[index];
        failed.push({
          topic_key: rule?.topic_key ?? "",
          topic_label: rule?.label ?? "Unknown topic",
          error: outcome.reason instanceof Error ? outcome.reason.message : String(outcome.reason),
        });
        console.error("[recap/generate] topic failed:", outcome.reason);
      }
    });

    if (results.length === 0 && failed.length > 0) {
      return NextResponse.json(
        {
          ok: false,
          error: `Recap generation failed for all matching topics: ${failed.map((item) => `${item.topic_label}: ${item.error}`).join("; ")}`,
          data: { date: recapDate, topics: results, skipped, failed },
        },
        { status: 502 }
      );
    }

    return NextResponse.json({ ok: true, data: { date: recapDate, topics: results, skipped, failed } });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error("[recap/generate]", message);
    return NextResponse.json({ ok: false, error: message }, { status: 500 });
  }
}

import { type NextRequest, NextResponse } from "next/server";
import {
  getRecapSettings,
  getTopicRules,
  getRecentArticles,
  saveRecapRows,
  type StoredRssArticle,
  type RecapSource,
} from "@/lib/server/neon";
import { getOpenAiConfig } from "@/lib/server/env";
import { getTopicMatches, normalizeTopicRules, type TopicRuleView } from "@/lib/intel-topic-matching";
import { getClientIp, getGenerateGlobalLimiter, getGenerateIpLimiter, isRateLimited } from "@/lib/server/rate-limit";
import { isAllowedRssArticleForIngestion } from "@/lib/server/rss-ingestion-filter";

export const dynamic = "force-dynamic";
export const maxDuration = 60;

const MAX_ITEMS_PER_TOPIC = 20;
const MIN_RECAP_SUMMARY_CHARS = 320;
// One request now generates exactly one topic (see GenerateRecapBody), so up
// to MAX_MODEL_ATTEMPTS calls must fit inside the maxDuration=60s route
// budget alongside the DB reads/matching. 22s * 2 = 44s leaves headroom.
const MODEL_REQUEST_TIMEOUT_MS = 22_000;
const MAX_MODEL_ATTEMPTS = 2;
const MAX_RECAP_OUTPUT_TOKENS = 1600;
const RECENT_ARTICLES_LIMIT = 400;
const OFFICIAL_FEED_PREFIXES = [
  "sec_",
  "finra_",
  "cftc_",
  "fed_",
  "occ_",
  "cfpb_",
  "ftc_",
  "doj_",
  "treasury_",
  "congress_",
  "cisa_",
];

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
      "deepseek-v4-pro",
    baseUrl: readEnv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
  };
}

// Articles require a title match (100+) to avoid false positives from passing mentions.
const MIN_ARTICLE_SCORE = 100;

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

// One topic per request, keyed by topic_key rather than a positional index —
// a positional cursor recomputed against `selectedRules` on every request
// silently skips or double-generates topics if settings/active rules change
// mid-run. Keying by topic_key also makes a single request idempotent and
// safely retryable, and lets the client fan requests out however it likes
// instead of being locked into one fixed batch size.
type GenerateRecapBody = {
  date?: string;
  topicKey?: string;
};

function normalizeRecapKey(value: string): string {
  return String(value || "")
    .toLowerCase()
    .replace(/https?:\/\/(?:www\.)?/g, "")
    .replace(/[?&#].*$/g, "")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function recapItemIdentity(item: RecapItem): string {
  const urlKey = normalizeRecapKey(item.url);
  const titleKey = normalizeRecapKey(item.title)
    .replace(/\s+(?:the )?(?:[a-z0-9&.\- ]{2,40})$/i, "")
    .trim();
  return urlKey || titleKey;
}

function dedupeRecapItems(items: RecapItem[]): RecapItem[] {
  const seen = new Set<string>();
  const out: RecapItem[] = [];
  for (const item of items) {
    const identity = recapItemIdentity(item);
    if (identity && seen.has(identity)) continue;
    if (identity) seen.add(identity);
    out.push(item);
  }
  return out;
}

function isPonziInvestorFraudItem(item: RecapItem): boolean {
  const text = `${item.title} ${item.description} ${item.matchText || ""}`.toLowerCase();
  return /\b(?:ponzi|investment fraud|investor fraud|securities fraud|offering fraud|fraudulent securities offering|misappropriat(?:ed|ion) (?:investor )?funds?|crypto fraud|wire fraud|commodity pool fraud)\b/.test(text);
}

function filterTopicItems(rule: TopicRuleView, items: RecapItem[]): RecapItem[] {
  const deduped = dedupeRecapItems(items);
  if (rule.topic_key === "PONZI_INVESTOR_FRAUD") {
    return deduped.filter(isPonziInvestorFraudItem);
  }
  return deduped;
}

function isOfficialRegulatoryFeed(feedKey: string): boolean {
  const normalized = String(feedKey || "").toLowerCase();
  return OFFICIAL_FEED_PREFIXES.some((prefix) => normalized.startsWith(prefix));
}

function recapSourceKindForArticle(article: StoredRssArticle): string | undefined {
  const feedKey = String(article.feed_key || "").toLowerCase();
  if (feedKey === "sec_speeches_statements") return "sec_speech";
  if (feedKey === "cftc_speeches_testimony") return "cftc_public_statement_remark";
  if (isOfficialRegulatoryFeed(feedKey)) return feedKey;
  return undefined;
}

function compactSentence(value: string, maxLength = 220): string {
  const cleaned = String(value || "")
    .replace(/\s+/g, " ")
    .trim();
  if (cleaned.length <= maxLength) return cleaned;
  return `${cleaned.slice(0, maxLength - 1).replace(/\s+\S*$/, "")}.`;
}

function sourceTitleList(items: RecapItem[], count: number): string {
  const titles = items
    .slice(0, count)
    .map((item) => compactSentence(item.title, 120))
    .filter(Boolean);
  if (titles.length === 0) return "the matched source set";
  if (titles.length === 1) return titles[0];
  return `${titles.slice(0, -1).join("; ")}; and ${titles[titles.length - 1]}`;
}

function buildSourceFallbackSummary(topicLabel: string, items: RecapItem[], reason: string): string {
  const topItems = dedupeRecapItems(items).slice(0, MAX_ITEMS_PER_TOPIC);
  const regulatoryDocs = topItems.filter((item) => item.source_type === "document");
  const newsItems = topItems.filter((item) => item.source_type === "article");
  const leadItems = regulatoryDocs.length > 0 ? regulatoryDocs : topItems;
  const leadSource = sourceTitleList(leadItems, 2);
  const newsSource = sourceTitleList(newsItems, 2);
  const sourceMix = `${regulatoryDocs.length} regulatory document${regulatoryDocs.length === 1 ? "" : "s"} and ${newsItems.length} news item${newsItems.length === 1 ? "" : "s"}`;

  console.warn("[recap/generate] using source fallback summary", {
    topicLabel,
    reason,
    items: topItems.length,
  });

  return [
    `**Executive Summary:** ${topicLabel} had ${topItems.length} matched source${topItems.length === 1 ? "" : "s"} in the selected recap window, including ${sourceMix}. The most relevant regulatory signal comes from ${leadSource}, while the broader public-news backdrop includes ${newsSource}. For financial regulators, the practical takeaway is to triage whether these developments affect supervisory priorities, enforcement exposure, market integrity, investor protection, operational resilience, or disclosure obligations before relying on the feed as complete.`,
    "",
    "**Key Points:**",
    `- Primary regulatory source set: ${sourceTitleList(leadItems, 3)}.`,
    `- News and market context: ${sourceTitleList(newsItems.length > 0 ? newsItems : topItems, 3)}.`,
    "- Follow-up review should identify affected firms, products, jurisdictions, filing deadlines, enforcement posture, and whether any official source has superseded or narrowed the public reporting.",
    "- This topic should be reviewed against the linked source documents before making supervisory, enforcement, or market-risk judgments.",
  ].join("\n");
}

async function generateTopicSummary(
  topicLabel: string,
  items: RecapItem[],
  cfg: RecapProviderConfig
): Promise<{ summary: string; fallback: boolean }> {
  const promptItems = items.slice(0, MAX_ITEMS_PER_TOPIC);
  const itemList = promptItems
    .map((item) => {
      const prefix = item.source_type === "document"
        ? `[Regulatory Document${item.speaker ? ` — ${item.speaker}` : ""}]`
        : "[News]";
      return `- ${prefix} ${item.title}${item.description ? `: ${item.description}` : ""}`;
    })
    .join("\n");

  // promptItems.length (not items.length) so the count the model is told to
  // summarize always matches what it's actually shown — with >20 matches
  // the list is capped at MAX_ITEMS_PER_TOPIC but the prompt used to still
  // quote the uncapped total.
  const prompt = `You are a regulatory intelligence analyst.\n\nSummarize the following ${promptItems.length} sources about "${topicLabel}" from the past 24 hours. Sources are labeled [News] or [Regulatory Document]. Use exactly this format:\n\n**Executive Summary:** [2–3 sentence overview of the most important developments.]\n\n**Key Points:**\n- [First key point]\n- [Second key point]\n- [Third key point]\n- [Add 1–2 more if warranted]\n\nEach bullet must be on its own line starting with "- ". Prioritize regulatory documents over news when relevant. Be direct and analytical. Synthesize — do not quote or list sources individually.\n\nSources:\n${itemList}`;

  const fullPrompt = `${prompt}\n\nAdditional requirements: The Executive Summary must be 3 complete sentences and 90-150 words total. The response must include at least three complete Key Points. Do not return sentence fragments or stop after a partial phrase.`;

  const validateSummary = (value: string): string => {
    const cleaned = value.trim();
    const bulletCount = (cleaned.match(/(?:^|\n)-\s+\S/g) || []).length;
    if (cleaned.length < MIN_RECAP_SUMMARY_CHARS) {
      throw new Error(`${providerLabel(cfg.provider)} returned an incomplete recap for ${topicLabel}: response was only ${cleaned.length} characters.`);
    }
    if (!/\*\*Executive Summary:\*\*/i.test(cleaned)) {
      throw new Error(`${providerLabel(cfg.provider)} returned a recap without an Executive Summary section for ${topicLabel}.`);
    }
    if (!/\*\*Key Points:\*\*/i.test(cleaned) || bulletCount < 3) {
      throw new Error(`${providerLabel(cfg.provider)} returned a recap without at least three key points for ${topicLabel}.`);
    }
    if (/[A-Za-z0-9,'")\]]$/.test(cleaned)) {
      throw new Error(`${providerLabel(cfg.provider)} returned a recap that appears to end mid-sentence for ${topicLabel}.`);
    }
    return cleaned;
  };

  let lastError: Error | null = null;
  for (let attempt = 1; attempt <= MAX_MODEL_ATTEMPTS; attempt += 1) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), MODEL_REQUEST_TIMEOUT_MS);
    try {
      const requestBody = {
        model: cfg.model,
        messages: [
          {
            role: "user",
            content: attempt === 1
              ? fullPrompt
              : `${fullPrompt}\n\nPrevious response was incomplete or malformed. Regenerate the full recap now. Include the Executive Summary and at least three complete Key Points.`,
          },
        ],
        max_tokens: MAX_RECAP_OUTPUT_TOKENS,
        temperature: 0.25,
        ...(cfg.provider === "deepseek" ? { thinking: { type: "disabled" } } : {}),
      };

      const res = await fetch(`${cfg.baseUrl}/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${cfg.apiKey}`,
        },
        body: JSON.stringify(requestBody),
        signal: controller.signal,
      });

      if (!res.ok) {
        const body = await res.text().catch(() => "");
        lastError = new Error(`${providerLabel(cfg.provider)} error ${res.status}: ${body.slice(0, 300)}`);
        break;
      }

      let content = "";
      try {
        const json = (await res.json()) as {
          choices: {
            finish_reason?: string | null;
            message?: { content?: string | null; reasoning_content?: string | null };
          }[];
          model?: string;
          usage?: { completion_tokens?: number; total_tokens?: number };
        };
        const choice = json.choices[0];
        content = choice?.message?.content?.trim() ?? "";
        console.info("[recap/generate] model response", {
          provider: cfg.provider,
          model: json.model ?? cfg.model,
          finishReason: choice?.finish_reason ?? null,
          contentLength: content.length,
          reasoningLength: choice?.message?.reasoning_content?.length ?? 0,
          completionTokens: json.usage?.completion_tokens ?? null,
          totalTokens: json.usage?.total_tokens ?? null,
        });
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        break;
      }

      try {
        return { summary: validateSummary(content), fallback: false };
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
      }
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") {
        lastError = new Error(`${providerLabel(cfg.provider)} request timed out while generating ${topicLabel}`);
        break;
      }
      lastError = error instanceof Error ? error : new Error(String(error));
      break;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  return {
    summary: buildSourceFallbackSummary(
      topicLabel,
      items,
      lastError?.message ?? `${providerLabel(cfg.provider)} did not return a usable recap.`
    ),
    fallback: true,
  };
}

// Scores items against a single rule (the one requested topic) rather than
// building a match map for every selected rule — each request only needs
// one topic's matches, so matching the rest is wasted work.
function matchItemsToTopic(
  items: RecapItem[],
  rule: TopicRuleView,
  minScore: number
): RecapItem[] {
  const matched: RecapItem[] = [];
  for (const item of items) {
    const matchInput = {
      title: item.title,
      description: item.matchText ?? item.description,
    };
    const [best] = getTopicMatches(matchInput, [rule]);
    if (best && best.score >= minScore) {
      matched.push(item);
    }
  }
  return matched;
}

export async function POST(req: NextRequest): Promise<NextResponse> {
  const requestStartedAt = Date.now();
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

    const body = await req.json().catch(() => ({})) as GenerateRecapBody;
    const todayIso = new Date().toISOString().split("T")[0] as string;
    const recapDate = body.date ?? todayIso;
    const topicKey = String(body.topicKey || "").trim();
    if (!topicKey) {
      return NextResponse.json({ ok: false, error: "topicKey is required." }, { status: 400 });
    }
    console.info("[recap/generate] start", { recapDate, topicKey });

    let since: Date;
    let until: Date | undefined;
    if (recapDate === todayIso) {
      since = new Date(Date.now() - 24 * 60 * 60 * 1000);
    } else {
      since = new Date(recapDate + "T00:00:00Z");
      const nextDay = new Date(since); nextDay.setUTCDate(nextDay.getUTCDate() + 1);
      until = nextDay;
    }

    const [selectedTopicKeys, rawRules, articles] = await Promise.all([
      getRecapSettings(),
      getTopicRules(true),
      getRecentArticles({ limit: RECENT_ARTICLES_LIMIT, since, until }),
    ]);
    console.info("[recap/generate] loaded inputs", {
      recapDate,
      topicKey,
      selectedTopics: selectedTopicKeys.length,
      rules: rawRules.length,
      articles: articles.length,
      elapsedMs: Date.now() - requestStartedAt,
    });

    if (!selectedTopicKeys.includes(topicKey)) {
      return NextResponse.json(
        { ok: false, error: "This topic is not in the saved recap settings. Save recap settings again, then generate." },
        { status: 400 }
      );
    }

    const rules = normalizeTopicRules(rawRules);
    const rule = rules.find((r) => r.topic_key === topicKey);
    if (!rule) {
      return NextResponse.json(
        {
          ok: false,
          error: rawRules.length === 0
            ? "No active recap topic rules are available. Reload the page and save recap topics again."
            : "The saved recap topic no longer matches an active topic rule. Save recap settings again, then generate the recap.",
        },
        { status: 400 }
      );
    }

    const warnings: string[] = [];
    if (articles.length >= RECENT_ARTICLES_LIMIT) {
      warnings.push(`Recent-articles window hit the ${RECENT_ARTICLES_LIMIT}-article fetch limit; the recap may be missing older matches from a high-volume window.`);
    }

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
      source_type: isOfficialRegulatoryFeed(a.feed_key) ? "document" : "article",
      source_kind: recapSourceKindForArticle(a),
      tone_label: a.tone_label,
    }));

    const matchedItems = matchItemsToTopic(articleItems, rule, MIN_ARTICLE_SCORE);
    const topicItems = filterTopicItems(rule, matchedItems);

    console.info("[recap/generate] topic start", {
      topicKey: rule.topic_key,
      topicLabel: rule.label,
      matchedItems: topicItems.length,
      elapsedMs: Date.now() - requestStartedAt,
    });

    if (topicItems.length === 0) {
      return NextResponse.json({
        ok: true,
        data: {
          date: recapDate,
          topicKey: rule.topic_key,
          topicLabel: rule.label,
          status: "skipped",
          warnings,
        },
      });
    }

    try {
      const { summary, fallback } = await generateTopicSummary(rule.label, topicItems, cfg);

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
        fallback,
      }]);

      console.info("[recap/generate] topic saved", {
        topicKey: rule.topic_key,
        items: topicItems.length,
        fallback,
        elapsedMs: Date.now() - requestStartedAt,
      });

      return NextResponse.json({
        ok: true,
        data: {
          date: recapDate,
          topicKey: rule.topic_key,
          topicLabel: rule.label,
          status: "generated",
          articleCount: topicItems.length,
          fallback,
          warnings,
          elapsedMs: Date.now() - requestStartedAt,
        },
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.error("[recap/generate] topic failed:", error);
      return NextResponse.json({
        ok: true,
        data: {
          date: recapDate,
          topicKey: rule.topic_key,
          topicLabel: rule.label,
          status: "failed",
          error: message,
          warnings,
        },
      });
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error("[recap/generate]", message);
    return NextResponse.json({ ok: false, error: message }, { status: 500 });
  }
}

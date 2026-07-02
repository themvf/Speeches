import { normalizeText } from "@/lib/server/api-utils";
import { getOpenAiConfig } from "@/lib/server/env";

export interface FeedAnalysisInput {
  title: string;
  description: string;
  url: string;
  source: string;
  author: string;
  published_at: string;
  tone_label: string;
  topics: string[];
  item_type: string;
}

export interface FeedAnalysis {
  thesis: string;
  why_it_matters: string[];
  risk_signals: string[];
  follow_up_questions: string[];
  keywords: string[];
  individuals: string[];
  entities: string[];
  model: string;
  generated_at: string;
  fallback: boolean;
}

interface OpenAiTextContent {
  type?: string;
  text?: string;
}

interface OpenAiOutput {
  type?: string;
  content?: OpenAiTextContent[];
}

interface OpenAiPayload {
  output_text?: string;
  output?: OpenAiOutput[];
  error?: { message?: string };
}

interface ChatCompletionPayload {
  choices?: Array<{ message?: { content?: string } }>;
  error?: { message?: string };
}

interface FeedAnalysisProviderConfig {
  provider: "deepseek" | "openai";
  apiKey: string;
  model: string;
  baseUrl: string;
}

function readEnv(name: string, fallback = ""): string {
  return String(process.env[name] ?? fallback).trim();
}

function getFeedAnalysisConfig(modelOverride = ""): FeedAnalysisProviderConfig {
  const explicitProvider = readEnv("FEED_ANALYSIS_PROVIDER").toLowerCase();
  const deepseekApiKey = readEnv("DEEPSEEK_API") || readEnv("DEEPSEEK_API_KEY");
  const deepseekModel =
    normalizeText(modelOverride) ||
    readEnv("FEED_ANALYSIS_MODEL") ||
    readEnv("DEEPSEEK_MODEL") ||
    readEnv("DEEPSEEK_CHAT_MODEL") ||
    "deepseek-v4-flash";

  if (explicitProvider === "openai") {
    const openai = getOpenAiConfig();
    return {
      provider: "openai",
      apiKey: openai.apiKey,
      model: normalizeText(modelOverride) || readEnv("FEED_ANALYSIS_MODEL") || readEnv("OPENAI_CHAT_MODEL") || "gpt-5.1",
      baseUrl: openai.baseUrl,
    };
  }

  return {
    provider: "deepseek",
    apiKey: deepseekApiKey,
    model: deepseekModel,
    baseUrl: readEnv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
  };
}

export function isDeepSeekFeedAnalysisConfigured(): boolean {
  const explicitProvider = readEnv("FEED_ANALYSIS_PROVIDER").toLowerCase();
  return explicitProvider !== "openai";
}

function analysisList(value: unknown): string[] {
  return Array.isArray(value) ? value.map((item) => normalizeText(item)).filter(Boolean) : [];
}

function hasWeakFeedAnalysisFields(analysis: {
  thesis?: string;
  why_it_matters?: unknown;
  risk_signals?: unknown;
  follow_up_questions?: unknown;
}): boolean {
  return (
    normalizeText(analysis.thesis).length < 40 ||
    analysisList(analysis.why_it_matters).length < 2 ||
    analysisList(analysis.risk_signals).length < 2 ||
    analysisList(analysis.follow_up_questions).length < 2
  );
}

export function shouldRefreshFeedAnalysisForDeepSeek(analysis: {
  model?: string;
  fallback?: boolean;
  thesis?: string;
  why_it_matters?: unknown;
  risk_signals?: unknown;
  follow_up_questions?: unknown;
} | null | undefined): boolean {
  if (!isDeepSeekFeedAnalysisConfigured() || !analysis) {
    return false;
  }
  const model = normalizeText(analysis.model).toLowerCase();
  return Boolean(analysis.fallback) || !model.startsWith("deepseek") || hasWeakFeedAnalysisFields(analysis);
}

function stringList(value: unknown, maxItems: number, maxChars = 180): string[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => normalizeText(item).slice(0, maxChars))
    .filter(Boolean)
    .slice(0, maxItems);
}

function cleanJsonText(value: string): string {
  return value
    .replace(/^```(?:json)?/i, "")
    .replace(/```$/i, "")
    .trim();
}

function extractResponseText(payload: OpenAiPayload): string {
  const direct = normalizeText(payload.output_text);
  if (direct) return direct;
  const pieces: string[] = [];
  for (const item of payload.output || []) {
    if (item?.type !== "message" || !Array.isArray(item.content)) continue;
    for (const content of item.content) {
      const text = normalizeText(content?.text);
      if (text) pieces.push(text);
    }
  }
  return pieces.join("\n").trim();
}

function extractChatCompletionText(payload: ChatCompletionPayload): string {
  return normalizeText(payload.choices?.[0]?.message?.content);
}

function coerceAnalysis(raw: unknown, model: string, fallback: boolean): FeedAnalysis {
  const src = raw && typeof raw === "object" ? raw as Record<string, unknown> : {};
  return {
    thesis: normalizeText(src.thesis).slice(0, 360),
    why_it_matters: stringList(src.why_it_matters, 5),
    risk_signals: stringList(src.risk_signals, 5),
    follow_up_questions: stringList(src.follow_up_questions, 5),
    keywords: stringList(src.keywords, 12, 80),
    individuals: stringList(src.individuals, 10, 80),
    entities: stringList(src.entities, 14, 100),
    model,
    generated_at: new Date().toISOString(),
    fallback,
  };
}

function unique(values: string[], maxItems: number): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const value of values) {
    const cleaned = normalizeText(value).replace(/^[^A-Za-z0-9]+|[^A-Za-z0-9]+$/g, "");
    const key = cleaned.toLowerCase();
    if (!cleaned || cleaned.length < 2 || seen.has(key)) continue;
    seen.add(key);
    out.push(cleaned);
    if (out.length >= maxItems) break;
  }
  return out;
}

function heuristicKeywords(input: FeedAnalysisInput): string[] {
  const stop = new Set([
    "about", "after", "against", "also", "amid", "because", "been", "being", "between", "from", "have", "into",
    "more", "over", "said", "says", "than", "that", "their", "there", "this", "with", "will", "would", "could",
    "article", "bloomberg", "financial", "financial-news", "market", "markets", "news", "public-feed", "regulatory",
  ]);
  const words = `${input.title} ${input.description} ${input.topics.join(" ")}`
    .toLowerCase()
    .match(/[a-z][a-z0-9-]{2,}/g) || [];
  const authorWords = new Set((normalizeText(input.author).toLowerCase().match(/[a-z][a-z0-9-]{2,}/g) || []));
  const counts = new Map<string, number>();
  for (const word of words) {
    if (stop.has(word) || authorWords.has(word)) continue;
    counts.set(word, (counts.get(word) || 0) + 1);
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([word]) => word)
    .slice(0, 12);
}

function heuristicEntities(input: FeedAnalysisInput): { individuals: string[]; entities: string[] } {
  const text = `${input.title}. ${input.description}. ${input.author}. ${input.source}.`;
  const candidates = text.match(/\b(?:[A-Z][a-zA-Z&.'-]+|[A-Z]{2,})(?:\s+(?:of|and|for|the|[A-Z][a-zA-Z&.'-]+|[A-Z]{2,})){0,5}/g) || [];
  const people: string[] = [];
  const entities: string[] = [];
  for (const candidate of candidates) {
    const cleaned = normalizeText(candidate);
    if (!cleaned || /^(The|This|That|News|Feed|Source|Author)$/i.test(cleaned)) continue;
    if (/^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}$/.test(cleaned)) {
      people.push(cleaned);
    } else {
      entities.push(cleaned);
    }
  }
  return {
    individuals: unique(people, 10),
    entities: unique([...entities, input.source, ...input.topics], 14),
  };
}

export function fallbackFeedAnalysis(input: FeedAnalysisInput, model = "heuristic"): FeedAnalysis {
  const title = normalizeText(input.title) || "Untitled feed item";
  const summary = normalizeText(input.description);
  const topicText = input.topics.length ? ` It maps to ${input.topics.slice(0, 3).join(", ")}.` : "";
  const { individuals, entities } = heuristicEntities(input);
  return {
    thesis: summary ? `${title}: ${summary.slice(0, 260)}` : `${title}.${topicText}`,
    why_it_matters: [
      topicText ? `Topic relevance:${topicText}` : "This item entered the live regulatory intelligence feed and may warrant source review.",
      input.source ? `Source context: ${input.source}.` : "Source context is limited in the RSS metadata.",
      input.published_at ? `Published date: ${input.published_at}.` : "Published date was not supplied by the feed.",
    ],
    risk_signals: [
      input.tone_label ? `Feed tone classified as ${input.tone_label}.` : "No sentiment signal was supplied.",
      summary.length < 120 ? "RSS summary is short; review the source before relying on conclusions." : "Analysis is based on RSS metadata and summary text.",
    ],
    follow_up_questions: [
      "Does the source article identify a concrete regulatory action, investigation, rulemaking, market impact, or compliance obligation?",
      "Should this item be promoted into a briefing, saved research list, or deeper corpus ingestion?",
    ],
    keywords: heuristicKeywords(input),
    individuals,
    entities,
    model,
    generated_at: new Date().toISOString(),
    fallback: true,
  };
}

function pushUnique(target: string[], values: string[], maxItems: number): string[] {
  const seen = new Set(target.map((item) => item.toLowerCase()));
  for (const value of values) {
    const cleaned = normalizeText(value);
    const key = cleaned.toLowerCase();
    if (!cleaned || seen.has(key)) continue;
    seen.add(key);
    target.push(cleaned);
    if (target.length >= maxItems) break;
  }
  return target.slice(0, maxItems);
}

function uniqueText(values: string[], maxItems: number): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const value of values) {
    const cleaned = normalizeText(value);
    const key = cleaned.toLowerCase();
    if (!cleaned || seen.has(key)) continue;
    seen.add(key);
    out.push(cleaned);
    if (out.length >= maxItems) break;
  }
  return out;
}

function isSparseFeedInput(input: FeedAnalysisInput): boolean {
  return normalizeText(input.description).length < 180;
}

function contextualWhyItMatters(input: FeedAnalysisInput): string[] {
  const title = normalizeText(input.title);
  const description = normalizeText(input.description);
  const source = normalizeText(input.source);
  const text = `${title} ${description}`.toLowerCase();
  const topics = input.topics.filter(Boolean);
  const out: string[] = [];

  if (topics.length) {
    out.push(`Maps to ${topics.slice(0, 2).join(" and ")} because the item concerns ${title || "the reported development"}.`);
  }
  if (/\binsider[\s-]+trad/i.test(text)) {
    out.push("Insider-trading allegations are relevant to MNPI controls, restricted-list monitoring, employee trading policies, and broker-dealer surveillance.");
  }
  if (/\b(sec|securities and exchange commission)\b/i.test(text)) {
    out.push("SEC involvement makes this relevant for enforcement posture, investigation-stage risk, and securities-market compliance monitoring.");
  }
  if (/\b(probe|probes|investigat|inquir|exam|charge|sue|sues|settle|fine|penalt)/i.test(text)) {
    out.push("The procedural posture matters because an investigation or enforcement headline can move from allegation to charges, settlement, or litigation.");
  }
  if (/\b(loss|losses|cost|harm|damage|victim)\b/i.test(text)) {
    out.push("Reported losses or market harm increase the need to identify affected securities, counterparties, time period, and control failures.");
  }
  if (isSparseFeedInput(input)) {
    out.push(`The available feed text is sparse${source ? ` from ${source}` : ""}; open the source article before treating the analysis as complete.`);
  }

  return uniqueText(out, 5);
}

function contextualRiskSignals(input: FeedAnalysisInput): string[] {
  const title = normalizeText(input.title);
  const description = normalizeText(input.description);
  const text = `${title} ${description}`.toLowerCase();
  const out: string[] = [];

  if (/\binsider[\s-]+trad/i.test(text)) out.push("Alleged insider trading or material-nonpublic-information misuse");
  if (/\b(sec|securities and exchange commission)\b/i.test(text)) out.push("SEC inquiry, investigation, or enforcement posture");
  if (/\b(probe|probes|investigat|inquir|exam)\b/i.test(text)) out.push("Investigation-stage facts are incomplete");
  if (/\b(loss|losses|cost|harm|damage|victim)\b/i.test(text)) out.push("Reported financial loss or market harm");
  if (isSparseFeedInput(input)) {
    out.push("RSS excerpt omits key facts such as affected instruments, parties, dates, and source qualifications");
    out.push("Limited feed metadata should be checked against the full source before relying on the signal");
  }

  return uniqueText(out, 5);
}

function contextualFollowUps(input: FeedAnalysisInput): string[] {
  const title = normalizeText(input.title);
  const text = `${title} ${normalizeText(input.description)}`.toLowerCase();
  const out: string[] = [];

  if (/\binsider[\s-]+trad/i.test(text)) {
    out.push("Which securities, trades, dates, and accounts are tied to the alleged insider trading?");
    out.push("Who allegedly possessed or tipped material nonpublic information, and how was it obtained?");
  }
  if (/\b(probe|probes|investigat|inquir|exam)\b/i.test(text)) {
    out.push("Is this an informal inquiry, formal SEC investigation, filed enforcement action, or related private litigation?");
  }
  if (/\b(loss|losses|cost|harm|damage|victim)\b/i.test(text)) {
    out.push("Who suffered the reported loss and what transaction or market event caused it?");
  }
  if (!out.length) {
    out.push("What concrete regulatory action, market impact, compliance obligation, or affected entity does the full article identify?");
  }
  if (isSparseFeedInput(input)) {
    out.push("Does the source article add facts not present in the RSS excerpt that change the risk assessment?");
  }

  return uniqueText(out, 5);
}

function contextualThesis(input: FeedAnalysisInput): string {
  const title = normalizeText(input.title);
  const source = normalizeText(input.source);
  if (!title) return normalizeText(input.description).slice(0, 260);
  return source ? `${source} reports: ${title}.` : `${title}.`;
}

function authorNames(input: FeedAnalysisInput): string[] {
  return unique(
    normalizeText(input.author)
      .split(/\s*(?:,| and | & )\s*/i)
      .map((item) => item.replace(/^by\s+/i, "").trim()),
    10
  );
}

function contextualEntities(input: FeedAnalysisInput): { individuals: string[]; entities: string[] } {
  const title = normalizeText(input.title);
  const source = normalizeText(input.source);
  const topicEntities = input.topics.filter(Boolean);
  const stop = new Set([
    "alleged",
    "article",
    "cost",
    "insider",
    "probe",
    "probes",
    "trades",
    "that",
    "with",
    "from",
  ]);
  const entities: string[] = [];

  for (const acronym of title.match(/\b[A-Z]{2,}\b/g) || []) {
    entities.push(acronym);
  }
  for (const candidate of title.match(/\b[A-Z][a-zA-Z&.'-]{2,}\b/g) || []) {
    if (!stop.has(candidate.toLowerCase())) entities.push(candidate);
  }
  if (source) entities.push(source);
  entities.push(...topicEntities);

  return {
    individuals: authorNames(input),
    entities: unique(entities, 14),
  };
}

function strengthenFeedAnalysis(input: FeedAnalysisInput, analysis: FeedAnalysis): FeedAnalysis {
  const fallback = fallbackFeedAnalysis(input, analysis.model);
  const sparse = isSparseFeedInput(input);
  const names = sparse ? contextualEntities(input) : heuristicEntities(input);
  const why = pushUnique(sparse ? [] : [...analysis.why_it_matters], contextualWhyItMatters(input), 5);
  const risks = pushUnique(sparse ? [] : [...analysis.risk_signals], contextualRiskSignals(input), 5);
  const follow = pushUnique(sparse ? [] : [...analysis.follow_up_questions], contextualFollowUps(input), 5);
  const keywords = sparse ? fallback.keywords : pushUnique([...analysis.keywords], fallback.keywords, 12);
  const individuals = pushUnique(sparse ? [] : [...analysis.individuals], names.individuals, 10);
  const entities = pushUnique(sparse ? [] : [...analysis.entities], names.entities, 14);

  return {
    ...analysis,
    thesis: sparse ? contextualThesis(input) : analysis.thesis || fallback.thesis,
    why_it_matters: why.length ? why : fallback.why_it_matters,
    risk_signals: risks.length ? risks : fallback.risk_signals,
    follow_up_questions: follow.length ? follow : fallback.follow_up_questions,
    keywords,
    individuals,
    entities,
  };
}

export async function generateFeedAnalysis(input: FeedAnalysisInput, modelOverride = ""): Promise<FeedAnalysis> {
  const cfg = getFeedAnalysisConfig(modelOverride);
  const model = cfg.model;
  if (!cfg.apiKey) {
    return fallbackFeedAnalysis(input, model);
  }

  const prompt = [
    `Title: ${normalizeText(input.title)}`,
    `Source: ${normalizeText(input.source)}`,
    `Author: ${normalizeText(input.author)}`,
    `Published: ${normalizeText(input.published_at)}`,
    `URL: ${normalizeText(input.url)}`,
    `Feed tone: ${normalizeText(input.tone_label)}`,
    `Matched topics: ${input.topics.join(", ")}`,
    `Item type: ${normalizeText(input.item_type) || "article"}`,
    "",
    "RSS summary / excerpt:",
    normalizeText(input.description).slice(0, 6000),
  ].join("\n");

  const instructions = [
    "You are a regulatory intelligence analyst for financial services, securities, banking, fintech, and enforcement coverage.",
    "Analyze only the supplied RSS/feed metadata and excerpt. Do not invent facts beyond the supplied text.",
    "Return dense, specific JSON for a working analyst, not a generic news summary.",
    "thesis must state the concrete development in one sentence and include the named agency, company, person, market, product, or proceeding when supplied.",
    "why_it_matters must contain 3-5 substantive bullets connecting the supplied facts to enforcement posture, supervision, compliance controls, market structure, investor harm, litigation, or policy impact.",
    "risk_signals must contain concrete red flags from the text; if the feed excerpt is sparse, identify exactly which facts are missing instead of writing boilerplate.",
    "follow_up_questions must be specific to the item and ask for missing securities, parties, procedural posture, timing, losses, rules, or affected markets where relevant.",
    "Avoid vague phrases such as 'potential regulatory risk', 'may warrant review', or 'market impact' unless tied to a named fact from the input.",
    "Extract concrete keywords, named individuals, companies, agencies, courts, rules, statutes, products, and other entities from the title, author, source, topics, and excerpt.",
    "Use empty arrays only when the supplied text truly does not identify a category.",
  ].join("\n");

  if (cfg.provider === "deepseek") {
    const response = await fetch(`${cfg.baseUrl.replace(/\/$/, "")}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${cfg.apiKey}`,
      },
      body: JSON.stringify({
        model,
        messages: [
          { role: "system", content: `${instructions}\nReturn only valid JSON with keys: thesis, why_it_matters, risk_signals, follow_up_questions, keywords, individuals, entities.` },
          { role: "user", content: prompt },
        ],
        response_format: { type: "json_object" },
      }),
      cache: "no-store",
    });

    const text = await response.text();
    let json: ChatCompletionPayload | null = null;
    try {
      json = JSON.parse(text) as ChatCompletionPayload;
    } catch {
      json = null;
    }
    if (!response.ok || !json) {
      return fallbackFeedAnalysis(input, model);
    }

    try {
      const parsed = JSON.parse(cleanJsonText(extractChatCompletionText(json))) as unknown;
      const analysis = coerceAnalysis(parsed, model, false);
      return analysis.thesis ? strengthenFeedAnalysis(input, analysis) : fallbackFeedAnalysis(input, model);
    } catch {
      return fallbackFeedAnalysis(input, model);
    }
  }

  const response = await fetch(`${cfg.baseUrl.replace(/\/$/, "")}/responses`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${cfg.apiKey}`,
    },
    body: JSON.stringify({
      model,
      instructions,
      input: prompt,
      text: {
        format: {
          type: "json_schema",
          name: "feed_item_analysis",
          strict: true,
          schema: {
            type: "object",
            additionalProperties: false,
            properties: {
              thesis: { type: "string" },
              why_it_matters: { type: "array", items: { type: "string" } },
              risk_signals: { type: "array", items: { type: "string" } },
              follow_up_questions: { type: "array", items: { type: "string" } },
              keywords: { type: "array", items: { type: "string" } },
              individuals: { type: "array", items: { type: "string" } },
              entities: { type: "array", items: { type: "string" } },
            },
            required: ["thesis", "why_it_matters", "risk_signals", "follow_up_questions", "keywords", "individuals", "entities"],
          },
        },
      },
    }),
    cache: "no-store",
  });

  const text = await response.text();
  let json: OpenAiPayload | null = null;
  try {
    json = JSON.parse(text) as OpenAiPayload;
  } catch {
    json = null;
  }
  if (!response.ok || !json) {
    return fallbackFeedAnalysis(input, model);
  }

  try {
    const parsed = JSON.parse(cleanJsonText(extractResponseText(json))) as unknown;
    const analysis = coerceAnalysis(parsed, model, false);
    return analysis.thesis ? strengthenFeedAnalysis(input, analysis) : fallbackFeedAnalysis(input, model);
  } catch {
    return fallbackFeedAnalysis(input, model);
  }
}

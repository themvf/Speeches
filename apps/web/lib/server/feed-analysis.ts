import { normalizeText } from "@/lib/server/api-utils";
import { getOpenAiConfig } from "@/lib/server/env";
import { MAX_STRENGTHEN_ATTEMPTS } from "@/lib/server/neon";

export const FEED_ANALYSIS_VERSION = 2;

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
  source_kind?: string;
  doc_type?: string;
  document_text?: string;
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
  // True when a non-sparse input had to be padded with boilerplate to reach
  // the minimum list sizes, i.e. the model itself underdelivered on content it
  // had. Used to re-queue the item for analysis: the count-based weakness check
  // otherwise sees the padded lists as healthy and never regenerates them.
  strengthened: boolean;
  analysis_version: number;
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
  usage?: { input_tokens?: number; output_tokens?: number; total_tokens?: number };
}

interface ChatCompletionPayload {
  choices?: Array<{ message?: { content?: string } }>;
  error?: { message?: string };
  usage?: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number };
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

const PROVIDER_FETCH_TIMEOUT_MS = 25_000;

// A hung DeepSeek/OpenAI call used to be able to ride out this route's
// entire request budget (55-60s). Aborting after a fixed timeout makes it
// fail the same way any other network error already does (an uncaught
// fetch rejection) - callers already handle that as a "failed" analysis,
// so this doesn't change error-handling semantics, just bounds the wait.
function fetchWithTimeout(url: string, init: RequestInit): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), PROVIDER_FETCH_TIMEOUT_MS);
  return fetch(url, { ...init, signal: controller.signal }).finally(() => clearTimeout(timer));
}

function getFeedAnalysisConfig(modelOverride = ""): FeedAnalysisProviderConfig {
  const explicitProvider = readEnv("FEED_ANALYSIS_PROVIDER").toLowerCase();
  const deepseekApiKey = readEnv("DEEPSEEK_API") || readEnv("DEEPSEEK_API_KEY");
  const deepseekModel =
    normalizeText(modelOverride) ||
    readEnv("FEED_ANALYSIS_MODEL") ||
    readEnv("DEEPSEEK_MODEL") ||
    readEnv("DEEPSEEK_CHAT_MODEL") ||
    // RSS feed analysis works from title+description (not full documents),
    // a simpler task than full-document enrichment - v4-flash (~3x cheaper
    // than v4-pro) is a reasonable quality tradeoff here.
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
  analysis_version?: number;
  model?: string;
  fallback?: boolean;
  strengthened?: boolean;
  strengthen_attempts?: number;
  thesis?: string;
  why_it_matters?: unknown;
  risk_signals?: unknown;
  follow_up_questions?: unknown;
} | null | undefined): boolean {
  if (!isDeepSeekFeedAnalysisConfigured() || !analysis) {
    return false;
  }
  if (Number(analysis.analysis_version || 0) < FEED_ANALYSIS_VERSION) {
    return true;
  }
  const model = normalizeText(analysis.model).toLowerCase();
  // Capped so a chronically-underperforming article isn't re-queued (and
  // re-billed) forever - mirrors the SQL gate in neon.ts's
  // getRssArticlesNeedingAnalysis, which is what actually drives automatic
  // re-selection; this function only gates the two manual/on-demand paths.
  const strengthenedAndUncapped =
    Boolean(analysis.strengthened) && (analysis.strengthen_attempts ?? 0) < MAX_STRENGTHEN_ATTEMPTS;
  return (
    Boolean(analysis.fallback) ||
    strengthenedAndUncapped ||
    !model.startsWith("deepseek") ||
    hasWeakFeedAnalysisFields(analysis)
  );
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
    strengthened: false,
    analysis_version: FEED_ANALYSIS_VERSION,
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
  const summary = analysisText(input);
  const topicText = input.topics.length ? ` It maps to ${input.topics.slice(0, 3).join(", ")}.` : "";
  const { individuals, entities } = heuristicEntities(input);
  const regulatorDoc = isRegulatorDocument(input);
  const speechDoc = isSpeechTestimonyOrTranscript(input);
  return {
    thesis: speechDoc
      ? contextualThesis(input)
      : summary
        ? `${title}: ${summary.slice(0, 260)}`
        : `${title}.${topicText}`,
    why_it_matters: [
      regulatorDoc
        ? "Primary-source regulator material can signal agency priorities, supervision themes, rulemaking direction, and compliance expectations."
        : topicText
          ? `Topic relevance:${topicText}`
          : "This item entered the live regulatory intelligence feed and may warrant source review.",
      speechDoc
        ? "Speech, testimony, video, or transcript content should be checked for exact policy statements, timing signals, and rule references."
        : input.source
          ? `Source context: ${input.source}.`
          : "Source context is limited in the RSS metadata.",
      input.published_at ? `Published date: ${input.published_at}.` : "Published date was not supplied by the feed.",
    ],
    risk_signals: [
      speechDoc
        ? "Verify whether the remarks announce a new position, restate existing policy, or summarize outreach before briefing them as a policy shift."
        : input.tone_label
          ? `Feed tone classified as ${input.tone_label}.`
          : "No sentiment signal was supplied.",
      summary.length < 220 ? "Available text is short; review the stored source text before relying on conclusions." : "Analysis is based on supplied feed and stored document text.",
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
    strengthened: false,
    analysis_version: FEED_ANALYSIS_VERSION,
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

function sourceText(input: FeedAnalysisInput): string {
  return normalizeText([
    input.title,
    input.description,
    input.document_text,
    input.source,
    input.author,
    input.source_kind,
    input.doc_type,
    input.url,
    ...input.topics,
  ].join(" "));
}

function analysisText(input: FeedAnalysisInput): string {
  return normalizeText(`${input.description}\n\n${input.document_text || ""}`);
}

function isSparseFeedInput(input: FeedAnalysisInput): boolean {
  return analysisText(input).length < 220;
}

function isRegulatorDocument(input: FeedAnalysisInput): boolean {
  const text = sourceText(input).toLowerCase();
  return /\b(sec|securities and exchange commission|finra|cftc|federal reserve|treasury|occ|fdic|pcaob|msrb|doj)\b/.test(text);
}

function isSpeechTestimonyOrTranscript(input: FeedAnalysisInput): boolean {
  const text = sourceText(input).toLowerCase();
  return /\b(speech|remarks|statement|testimony|hearing|transcript|youtube|video|roundtable|chairman|commissioner|governor)\b/.test(text);
}

function isEnforcementItem(input: FeedAnalysisInput): boolean {
  const text = sourceText(input).toLowerCase();
  return (
    /\b(enforcement|litigation release|administrative proceeding|trading suspension|complaint|charges?|settlement|judgment|injunction|penalt|fraud|insider[\s-]+trading)\b/.test(text) ||
    /\bsec_enforcement|sec_administrative|sec_trading_suspension\b/.test(text)
  );
}

function contextualWhyItMatters(input: FeedAnalysisInput): string[] {
  const source = normalizeText(input.source);
  const text = sourceText(input).toLowerCase();
  const topics = input.topics.filter(Boolean);
  const out: string[] = [];
  const regulatorDoc = isRegulatorDocument(input);
  const speechDoc = isSpeechTestimonyOrTranscript(input);
  const enforcementItem = isEnforcementItem(input);

  if (topics.length) {
    out.push(`Maps to ${topics.slice(0, 2).join(" and ")} based on the supplied title, source, topics, and text.`);
  }
  if (regulatorDoc && speechDoc) {
    out.push("Regulator speeches, testimony, and transcripts are primary-source signals for agency priorities, rulemaking direction, supervisory emphasis, and market-structure policy.");
  }
  if (/\b(investor education|financial education|retail investor|main street|financial literacy)\b/i.test(text)) {
    out.push("Investor-education messaging can indicate where the agency sees retail confusion, disclosure gaps, or conduct risks that may shape exams, guidance, or outreach.");
  }
  if (/\b(roundtable|hearing|testimony)\b/i.test(text)) {
    out.push("The forum matters because public testimony or roundtable remarks can preview policy arguments, stakeholder concerns, and questions Congress or the agency may revisit.");
  }
  if (/\b(crypto|digital asset|stablecoin|token|market structure|equity market|treasury market|clearing|settlement|disclosure|private fund|investment adviser|broker-dealer)\b/i.test(text)) {
    out.push("The item touches a regulated market, product, or intermediary that may affect compliance monitoring, disclosure controls, supervision, or market-structure planning.");
  }
  if (/\binsider[\s-]+trad/i.test(text)) {
    out.push("Insider-trading allegations are relevant to MNPI controls, restricted-list monitoring, employee trading policies, and broker-dealer surveillance.");
  }
  if (enforcementItem && /\b(sec|securities and exchange commission)\b/i.test(text)) {
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
  const text = sourceText(input).toLowerCase();
  const out: string[] = [];
  const speechDoc = isSpeechTestimonyOrTranscript(input);
  const enforcementItem = isEnforcementItem(input);

  if (/\binsider[\s-]+trad/i.test(text)) out.push("Alleged insider trading or material-nonpublic-information misuse");
  if (enforcementItem && /\b(sec|securities and exchange commission)\b/i.test(text)) out.push("SEC inquiry, investigation, or enforcement posture");
  if (/\b(probe|probes|investigat|inquir|exam)\b/i.test(text)) out.push("Investigation-stage facts are incomplete");
  if (/\b(loss|losses|cost|harm|damage|victim)\b/i.test(text)) out.push("Reported financial loss or market harm");
  if (speechDoc) {
    out.push("Review the stored transcript or testimony for exact quoted commitments, rule references, timing signals, and limiting language.");
    out.push("The feed view may not identify whether the remarks announce policy, restate existing priorities, or merely summarize outreach.");
  }
  if (isSparseFeedInput(input)) {
    out.push("The available text is short; verify affected instruments, parties, dates, legal authority, and source qualifications before relying on the signal.");
    out.push("Limited metadata should be checked against the full source before using the item in a briefing.");
  }

  return uniqueText(out, 5);
}

function contextualFollowUps(input: FeedAnalysisInput): string[] {
  const text = sourceText(input).toLowerCase();
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
  if (isSpeechTestimonyOrTranscript(input)) {
    out.push("What exact policy position, rule reference, market practice, or compliance expectation does the speaker identify?");
    out.push("Does the testimony or transcript include timing, next steps, dissenting views, or limits on the agency position?");
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
  const docType = normalizeText(input.doc_type || "");
  if (isSpeechTestimonyOrTranscript(input)) {
    const label = docType || (sourceText(input).toLowerCase().includes("transcript") ? "transcript" : "remarks");
    return source ? `${source} ${label.toLowerCase()} concerns ${title}.` : `${label} concerns ${title}.`;
  }
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

function ensureMinimumList(values: string[], fallback: string[], minItems: number, maxItems: number): string[] {
  return values.length >= minItems ? values.slice(0, maxItems) : pushUnique([...values], fallback, maxItems);
}

function strengthenFeedAnalysis(input: FeedAnalysisInput, analysis: FeedAnalysis): FeedAnalysis {
  const fallback = fallbackFeedAnalysis(input, analysis.model);
  const sparse = isSparseFeedInput(input);
  // Count the model's own output before any padding. On a non-sparse input,
  // if the model itself produced fewer than the minimum in any bucket, the
  // strengthened lists below are propped up by boilerplate — flag it so the
  // retry logic re-queues it instead of treating the padded result as healthy.
  const modelUnderdelivered =
    analysisList(analysis.why_it_matters).length < 2 ||
    analysisList(analysis.risk_signals).length < 2 ||
    analysisList(analysis.follow_up_questions).length < 2;
  const strengthened = !sparse && modelUnderdelivered;
  const names = sparse ? contextualEntities(input) : heuristicEntities(input);
  const why = pushUnique(sparse ? [] : [...analysis.why_it_matters], contextualWhyItMatters(input), 5);
  const risks = pushUnique(sparse ? [] : [...analysis.risk_signals], contextualRiskSignals(input), 5);
  const follow = pushUnique(sparse ? [] : [...analysis.follow_up_questions], contextualFollowUps(input), 5);
  const keywords = sparse ? fallback.keywords : pushUnique([...analysis.keywords], fallback.keywords, 12);
  const individuals = pushUnique(sparse ? [] : [...analysis.individuals], names.individuals, 10);
  const entities = pushUnique(sparse ? [] : [...analysis.entities], names.entities, 14);
  const thesis = sparse ? contextualThesis(input) : analysis.thesis || fallback.thesis;

  return {
    ...analysis,
    thesis: normalizeText(thesis).length >= 40 ? thesis : fallback.thesis,
    why_it_matters: ensureMinimumList(why, fallback.why_it_matters, 2, 5),
    risk_signals: ensureMinimumList(risks, fallback.risk_signals, 2, 5),
    follow_up_questions: ensureMinimumList(follow, fallback.follow_up_questions, 2, 5),
    keywords,
    individuals,
    entities,
    strengthened,
    analysis_version: FEED_ANALYSIS_VERSION,
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
    `Source kind: ${normalizeText(input.source_kind || "")}`,
    `Document type: ${normalizeText(input.doc_type || "")}`,
    "",
    "RSS summary / excerpt:",
    normalizeText(input.description).slice(0, 6000),
    "",
    "Stored document text / transcript / testimony excerpt:",
    normalizeText(input.document_text || "").slice(0, 14000),
  ].join("\n");

  const instructions = [
    "You are a senior financial-regulatory intelligence analyst for a website used to monitor regulators, enforcement, markets, supervision, compliance, and policy posture.",
    "Decide whether this item should be monitored, saved, briefed, escalated, or ignored.",
    "Use only the supplied title, source, author, date, URL, matched topics, tone label, source kind, document type, RSS/feed excerpt, and stored document text. Do not add outside facts, background, or assumptions.",
    "If a fact needed for assessment is missing, state the exact missing fact in risk_signals or follow_up_questions instead of inferring it.",
    "Return dense, source-bounded JSON for a working analyst. Do not write a generic news summary, media recap, or basic explanation of common terms.",
    "For speeches, testimony, roundtables, videos, and transcripts, prioritize policy posture, rulemaking direction, supervisory emphasis, market-structure implications, investor-protection themes, affected registrants/intermediaries, timing signals, and exact statements that need source review.",
    "Do not call a speech, testimony, roundtable, education item, or video an enforcement risk unless the supplied text explicitly says investigation, enforcement, charges, litigation, settlement, penalty, fraud, suspension, or similar.",
    "thesis: one concrete sentence naming the event, agency, speaker, market/product, policy topic, proceeding, or transaction supplied in the input. Avoid repeating the headline verbatim unless the input has no other substance.",
    "why_it_matters: 3-5 bullets tied to the supplied facts and to at least one of policy, enforcement posture, supervision, compliance controls, market structure, investor/customer harm, litigation, capital markets, financial-stability impact, or regulatory agenda.",
    "risk_signals: 2-5 concrete red flags, uncertainties, missing facts, or source-review needs. For testimony/transcripts, include exact-quote/timing/rule-reference checks rather than generic risk labels.",
    "follow_up_questions: 2-5 item-specific questions about policy position, rule references, parties, securities/instruments, dates, procedural posture, jurisdiction, losses, obligations, next steps, or market impact where relevant.",
    "keywords, individuals, and entities must be extracted only from the title, source, author, topics, URL, and excerpt; avoid author fragments and generic media words.",
    "Use empty arrays only when the supplied text truly does not identify a category.",
  ].join("\n");

  if (cfg.provider === "deepseek") {
    const response = await fetchWithTimeout(`${cfg.baseUrl.replace(/\/$/, "")}/chat/completions`, {
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
    console.info("[feed-analysis] deepseek usage", {
      model,
      promptTokens: json.usage?.prompt_tokens ?? null,
      completionTokens: json.usage?.completion_tokens ?? null,
      totalTokens: json.usage?.total_tokens ?? null,
    });

    try {
      const parsed = JSON.parse(cleanJsonText(extractChatCompletionText(json))) as unknown;
      const analysis = coerceAnalysis(parsed, model, false);
      return analysis.thesis ? strengthenFeedAnalysis(input, analysis) : fallbackFeedAnalysis(input, model);
    } catch {
      return fallbackFeedAnalysis(input, model);
    }
  }

  const response = await fetchWithTimeout(`${cfg.baseUrl.replace(/\/$/, "")}/responses`, {
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
  console.info("[feed-analysis] openai usage", {
    model,
    promptTokens: json.usage?.input_tokens ?? null,
    completionTokens: json.usage?.output_tokens ?? null,
    totalTokens: json.usage?.total_tokens ?? null,
  });

  try {
    const parsed = JSON.parse(cleanJsonText(extractResponseText(json))) as unknown;
    const analysis = coerceAnalysis(parsed, model, false);
    return analysis.thesis ? strengthenFeedAnalysis(input, analysis) : fallbackFeedAnalysis(input, model);
  } catch {
    return fallbackFeedAnalysis(input, model);
  }
}

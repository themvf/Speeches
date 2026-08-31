import { createHash, randomUUID } from "node:crypto";
import { neon } from "@neondatabase/serverless";
import { getRecentArticles, type StoredRssArticle } from "@/lib/server/neon";

export type EditorialProvider = "openai" | "deepseek";
export type ScheduledEditorialSettings = {
  enabled: boolean;
  timezone: string;
  hour: number;
  minute: number;
  lookback_hours: number;
  openai_enabled: boolean;
  openai_model: string;
  deepseek_enabled: boolean;
  deepseek_model: string;
  blind_comparison: boolean;
  rough_draft: boolean;
};

export type EditorialSource = {
  source_id: string;
  title: string;
  description: string;
  url: string;
  publisher: string;
  published_at: string | null;
};

export type EditorialCandidateDraft = {
  id: number;
  output_id: number;
  candidate_id: string;
  provider: EditorialProvider;
  model: string;
  status: "completed" | "failed";
  article: string;
  latency_ms: number;
  usage: Record<string, unknown>;
  error: string;
  created_at: string;
  updated_at: string;
};

export type EditorialOutput = {
  id: number;
  run_id: number;
  provider: EditorialProvider;
  model: string;
  status: "completed" | "failed";
  latency_ms: number;
  usage: Record<string, unknown>;
  package: Record<string, unknown> | null;
  error: string;
  created_at: string;
  candidate_drafts: EditorialCandidateDraft[];
};

export type EditorialRun = {
  id: number;
  run_date: string;
  trigger: "manual" | "scheduled";
  status: "running" | "completed" | "partial" | "failed";
  snapshot_hash: string;
  source_count: number;
  source_snapshot: EditorialSource[];
  settings_snapshot: ScheduledEditorialSettings;
  error: string;
  started_at: string;
  finished_at: string | null;
  outputs: EditorialOutput[];
};

const DEFAULT_SETTINGS: ScheduledEditorialSettings = {
  enabled: false,
  timezone: "America/New_York",
  hour: 21,
  minute: 0,
  lookback_hours: 24,
  openai_enabled: true,
  openai_model: "gpt-5.6-luna",
  deepseek_enabled: true,
  deepseek_model: "deepseek-v4-pro",
  blind_comparison: true,
  rough_draft: true,
};

const OUTPUT_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["editorial_recommendation", "candidates", "selected_package", "draft", "quality_warnings"],
  properties: {
    editorial_recommendation: {
      type: "object",
      additionalProperties: false,
      required: ["decision", "selected_candidate_id", "rationale"],
      properties: {
        decision: { type: "string", enum: ["publish", "no_publish"] },
        selected_candidate_id: { type: ["string", "null"] },
        rationale: { type: "string" },
      },
    },
    candidates: {
      type: "array",
      maxItems: 3,
      items: {
        type: "object",
        additionalProperties: false,
        required: ["candidate_id", "working_title", "subtitle", "thesis", "reader_promise", "why_now", "original_contribution", "counterargument", "support_score", "originality_score", "recap_risk", "supporting_source_ids"],
        properties: {
          candidate_id: { type: "string" },
          working_title: { type: "string" },
          subtitle: { type: "string" },
          thesis: { type: "string" },
          reader_promise: { type: "string" },
          why_now: { type: "string" },
          original_contribution: { type: "string" },
          counterargument: { type: "string" },
          support_score: { type: "integer", minimum: 1, maximum: 5 },
          originality_score: { type: "integer", minimum: 1, maximum: 5 },
          recap_risk: { type: "string", enum: ["low", "medium", "high"] },
          supporting_source_ids: { type: "array", items: { type: "string" } },
        },
      },
    },
    selected_package: {
      anyOf: [
        { type: "null" },
        {
          type: "object",
          additionalProperties: false,
          required: ["opening_hooks", "outline", "claim_ledger", "author_questions"],
          properties: {
            opening_hooks: { type: "array", items: { type: "string" } },
            outline: {
              type: "array",
              items: {
                type: "object",
                additionalProperties: false,
                required: ["heading", "purpose", "source_ids"],
                properties: {
                  heading: { type: "string" },
                  purpose: { type: "string" },
                  source_ids: { type: "array", items: { type: "string" } },
                },
              },
            },
            claim_ledger: {
              type: "array",
              items: {
                type: "object",
                additionalProperties: false,
                required: ["claim", "claim_type", "support_level", "supporting_source_ids"],
                properties: {
                  claim: { type: "string" },
                  claim_type: { type: "string", enum: ["fact", "inference", "prediction", "recommendation"] },
                  support_level: { type: "string", enum: ["supported", "partial", "author_judgment"] },
                  supporting_source_ids: { type: "array", items: { type: "string" } },
                },
              },
            },
            author_questions: { type: "array", items: { type: "string" } },
          },
        },
      ],
    },
    draft: { type: ["string", "null"] },
    quality_warnings: { type: "array", items: { type: "string" } },
  },
} as const;

let sqlClient: ReturnType<typeof neon> | null = null;
let schemaPromise: Promise<void> | null = null;

function getSql() {
  if (!sqlClient) {
    const databaseUrl = process.env.DATABASE_URL?.trim();
    if (!databaseUrl) throw new Error("DATABASE_URL is not configured.");
    sqlClient = neon(databaseUrl);
  }
  return sqlClient;
}

async function ensureScheduledEditorialSchema(): Promise<void> {
  if (!schemaPromise) {
    schemaPromise = (async () => {
      const sql = getSql();
      await sql`
        CREATE TABLE IF NOT EXISTS scheduled_editorial_settings (
          id                 INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
          enabled            BOOLEAN NOT NULL DEFAULT false,
          timezone           TEXT NOT NULL DEFAULT 'America/New_York',
          run_hour           INTEGER NOT NULL DEFAULT 21,
          run_minute         INTEGER NOT NULL DEFAULT 0,
          lookback_hours     INTEGER NOT NULL DEFAULT 24,
          openai_enabled     BOOLEAN NOT NULL DEFAULT true,
          openai_model       TEXT NOT NULL DEFAULT 'gpt-5.6-luna',
          deepseek_enabled   BOOLEAN NOT NULL DEFAULT true,
          deepseek_model     TEXT NOT NULL DEFAULT 'deepseek-v4-pro',
          blind_comparison   BOOLEAN NOT NULL DEFAULT true,
          rough_draft        BOOLEAN NOT NULL DEFAULT true,
          updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
        )
      `;
      await sql`
        CREATE TABLE IF NOT EXISTS scheduled_editorial_runs (
          id                 BIGSERIAL PRIMARY KEY,
          dedupe_key         TEXT UNIQUE NOT NULL,
          run_date           DATE NOT NULL,
          trigger            TEXT NOT NULL CHECK (trigger IN ('manual', 'scheduled')),
          status             TEXT NOT NULL CHECK (status IN ('running', 'completed', 'partial', 'failed')),
          snapshot_hash      TEXT NOT NULL DEFAULT '',
          source_count       INTEGER NOT NULL DEFAULT 0,
          source_snapshot    JSONB NOT NULL DEFAULT '[]'::jsonb,
          settings_snapshot  JSONB NOT NULL DEFAULT '{}'::jsonb,
          error              TEXT NOT NULL DEFAULT '',
          started_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
          finished_at        TIMESTAMPTZ
        )
      `;
      await sql`CREATE INDEX IF NOT EXISTS scheduled_editorial_runs_started_at ON scheduled_editorial_runs (started_at DESC)`;
      await sql`
        CREATE TABLE IF NOT EXISTS scheduled_editorial_outputs (
          id          BIGSERIAL PRIMARY KEY,
          run_id      BIGINT NOT NULL REFERENCES scheduled_editorial_runs(id) ON DELETE CASCADE,
          provider    TEXT NOT NULL CHECK (provider IN ('openai', 'deepseek')),
          model       TEXT NOT NULL,
          status      TEXT NOT NULL CHECK (status IN ('completed', 'failed')),
          latency_ms  INTEGER NOT NULL DEFAULT 0,
          usage       JSONB NOT NULL DEFAULT '{}'::jsonb,
          package     JSONB,
          error       TEXT NOT NULL DEFAULT '',
          created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
          UNIQUE (run_id, provider)
        )
      `;
      await sql`
        CREATE TABLE IF NOT EXISTS scheduled_editorial_candidate_drafts (
          id            BIGSERIAL PRIMARY KEY,
          output_id     BIGINT NOT NULL REFERENCES scheduled_editorial_outputs(id) ON DELETE CASCADE,
          candidate_id  TEXT NOT NULL,
          provider      TEXT NOT NULL CHECK (provider IN ('openai', 'deepseek')),
          model         TEXT NOT NULL,
          status        TEXT NOT NULL CHECK (status IN ('completed', 'failed')),
          article       TEXT NOT NULL DEFAULT '',
          latency_ms    INTEGER NOT NULL DEFAULT 0,
          usage         JSONB NOT NULL DEFAULT '{}'::jsonb,
          error         TEXT NOT NULL DEFAULT '',
          created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
          UNIQUE (output_id, candidate_id)
        )
      `;
      await sql`CREATE INDEX IF NOT EXISTS scheduled_editorial_candidate_drafts_output_id ON scheduled_editorial_candidate_drafts (output_id)`;
    })().catch((error) => {
      schemaPromise = null;
      throw error;
    });
  }
  return schemaPromise;
}

function bool(value: unknown, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function boundedInt(value: unknown, fallback: number, min: number, max: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.max(min, Math.min(max, Math.round(parsed))) : fallback;
}

function safeModel(value: unknown, fallback: string): string {
  const model = String(value ?? "").trim();
  return /^[a-zA-Z0-9._:-]{2,80}$/.test(model) ? model : fallback;
}

export function sanitizeScheduledEditorialSettings(value: unknown): ScheduledEditorialSettings {
  const raw = value && typeof value === "object" ? value as Record<string, unknown> : {};
  return {
    enabled: bool(raw.enabled, DEFAULT_SETTINGS.enabled),
    timezone: raw.timezone === "America/New_York" ? raw.timezone : DEFAULT_SETTINGS.timezone,
    hour: boundedInt(raw.hour, DEFAULT_SETTINGS.hour, 0, 23),
    minute: boundedInt(raw.minute, DEFAULT_SETTINGS.minute, 0, 59),
    lookback_hours: boundedInt(raw.lookback_hours, DEFAULT_SETTINGS.lookback_hours, 6, 72),
    openai_enabled: bool(raw.openai_enabled, DEFAULT_SETTINGS.openai_enabled),
    openai_model: safeModel(raw.openai_model, DEFAULT_SETTINGS.openai_model),
    deepseek_enabled: bool(raw.deepseek_enabled, DEFAULT_SETTINGS.deepseek_enabled),
    deepseek_model: safeModel(raw.deepseek_model, DEFAULT_SETTINGS.deepseek_model),
    blind_comparison: bool(raw.blind_comparison, DEFAULT_SETTINGS.blind_comparison),
    rough_draft: bool(raw.rough_draft, DEFAULT_SETTINGS.rough_draft),
  };
}

function rowSettings(row: Record<string, unknown> | undefined): ScheduledEditorialSettings {
  if (!row) return DEFAULT_SETTINGS;
  return sanitizeScheduledEditorialSettings({
    enabled: row.enabled,
    timezone: row.timezone,
    hour: row.run_hour,
    minute: row.run_minute,
    lookback_hours: row.lookback_hours,
    openai_enabled: row.openai_enabled,
    openai_model: row.openai_model,
    deepseek_enabled: row.deepseek_enabled,
    deepseek_model: row.deepseek_model,
    blind_comparison: row.blind_comparison,
    rough_draft: row.rough_draft,
  });
}

export async function getScheduledEditorialSettings(): Promise<ScheduledEditorialSettings> {
  await ensureScheduledEditorialSchema();
  const rows = await getSql()`SELECT * FROM scheduled_editorial_settings WHERE id = 1` as unknown as Record<string, unknown>[];
  return rowSettings(rows[0]);
}

export async function saveScheduledEditorialSettings(settings: ScheduledEditorialSettings): Promise<ScheduledEditorialSettings> {
  await ensureScheduledEditorialSchema();
  const clean = sanitizeScheduledEditorialSettings(settings);
  await getSql()`
    INSERT INTO scheduled_editorial_settings (
      id, enabled, timezone, run_hour, run_minute, lookback_hours,
      openai_enabled, openai_model, deepseek_enabled, deepseek_model,
      blind_comparison, rough_draft, updated_at
    ) VALUES (
      1, ${clean.enabled}, ${clean.timezone}, ${clean.hour}, ${clean.minute}, ${clean.lookback_hours},
      ${clean.openai_enabled}, ${clean.openai_model}, ${clean.deepseek_enabled}, ${clean.deepseek_model},
      ${clean.blind_comparison}, ${clean.rough_draft}, now()
    )
    ON CONFLICT (id) DO UPDATE SET
      enabled = EXCLUDED.enabled,
      timezone = EXCLUDED.timezone,
      run_hour = EXCLUDED.run_hour,
      run_minute = EXCLUDED.run_minute,
      lookback_hours = EXCLUDED.lookback_hours,
      openai_enabled = EXCLUDED.openai_enabled,
      openai_model = EXCLUDED.openai_model,
      deepseek_enabled = EXCLUDED.deepseek_enabled,
      deepseek_model = EXCLUDED.deepseek_model,
      blind_comparison = EXCLUDED.blind_comparison,
      rough_draft = EXCLUDED.rough_draft,
      updated_at = now()
  `;
  return clean;
}

function canonicalUrl(value: string): string {
  try {
    const url = new URL(value);
    for (const key of [...url.searchParams.keys()]) {
      if (key.toLowerCase().startsWith("utm_") || ["ref", "source", "campaign"].includes(key.toLowerCase())) {
        url.searchParams.delete(key);
      }
    }
    url.hash = "";
    return url.toString().replace(/\/$/, "").toLowerCase();
  } catch {
    return value.trim().toLowerCase();
  }
}

function isEligibleAiArticle(article: StoredRssArticle): boolean {
  const title = article.title || "";
  const description = article.description || "";
  const explicitAi = /(?:\bAI\b|A\.I\.|artificial intelligence|OpenAI|Anthropic|Claude|ChatGPT|DeepSeek|machine learning|Nvidia|AI agent)/i;
  const promotional = /(?:press release|sponsored|coupon|discount|webinar|register now|ETF launch)/i;
  return explicitAi.test(`${title} ${description}`) && !promotional.test(title);
}

async function captureSources(lookbackHours: number): Promise<EditorialSource[]> {
  const since = new Date(Date.now() - lookbackHours * 60 * 60 * 1000);
  const articles = await getRecentArticles({ since, limit: 500 });
  const seen = new Set<string>();
  const sources: EditorialSource[] = [];
  for (const article of articles) {
    if (!isEligibleAiArticle(article)) continue;
    const identity = canonicalUrl(article.url) || article.title.toLowerCase();
    if (seen.has(identity)) continue;
    seen.add(identity);
    sources.push({
      source_id: `rss:${article.id}`,
      title: article.title,
      description: String(article.description || "").replace(/\s+/g, " ").trim().slice(0, 1200),
      url: article.url,
      publisher: article.feed_label || article.feed_key,
      published_at: article.published_at,
    });
    if (sources.length >= 16) break;
  }
  return sources;
}

function promptMessages(sources: EditorialSource[], settings: ScheduledEditorialSettings) {
  const developer = [
    "You are the Daily AI Column Editor for a financial and regulatory intelligence publication.",
    "Create an original editorial decision package from only the supplied captured headlines and descriptions.",
    "Never imply that you read the full articles. Never invent facts, quotations, first-person experience, or the author's opinion.",
    "Every factual claim must cite one or more source_id values in square brackets. Distinguish fact, inference, prediction, and recommendation.",
    "Prefer a coherent, non-obvious thesis for professionals following AI, markets, regulation, governance, and enterprise risk.",
    settings.rough_draft
      ? "Also produce a 900-1,400 word Medium-style rough draft. It must remain source-bounded, use inline [source_id] citations, and leave personal judgments to the human editor."
      : "Set draft to null. The human author has disabled rough-draft generation.",
    "A no_publish decision is valid when evidence is weak. Return JSON only and conform exactly to the schema.",
  ].join(" ");
  const user = [
    "Generate tonight's editorial package. Propose up to three genuinely different angles and select the strongest.",
    `Output schema: ${JSON.stringify(OUTPUT_SCHEMA)}`,
    `Frozen sources: ${JSON.stringify(sources, null, 2)}`,
  ].join("\n\n");
  return [{ role: "developer", content: developer }, { role: "user", content: user }];
}

function parseJsonObject(value: string): Record<string, unknown> {
  const cleaned = value.trim().replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "");
  const parsed = JSON.parse(cleaned) as unknown;
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("Provider response was not a JSON object.");
  return parsed as Record<string, unknown>;
}

async function fetchWithTimeout(url: string, init: RequestInit, timeoutMs = 180_000): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function runOpenAi(messages: Array<{ role: string; content: string }>, model: string) {
  const apiKey = process.env.OPENAI_API_KEY?.trim();
  if (!apiKey) throw new Error("OPENAI_API_KEY is not configured.");
  const started = Date.now();
  const response = await fetchWithTimeout("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({
      model,
      input: messages,
      reasoning: { effort: "medium" },
      text: { format: { type: "json_schema", name: "editorial_package", strict: true, schema: OUTPUT_SCHEMA } },
      max_output_tokens: 14_000,
      store: false,
    }),
  });
  const raw = await response.json() as Record<string, unknown>;
  if (!response.ok) {
    const error = raw.error as Record<string, unknown> | undefined;
    throw new Error(`OpenAI ${response.status}: ${String(error?.code || error?.message || "request failed")}`);
  }
  let content = typeof raw.output_text === "string" ? raw.output_text : "";
  if (!content) {
    const output = Array.isArray(raw.output) ? raw.output as Array<Record<string, unknown>> : [];
    content = output.flatMap((item) => Array.isArray(item.content) ? item.content as Array<Record<string, unknown>> : [])
      .map((item) => String(item.text || ""))
      .filter(Boolean)
      .join("\n");
  }
  return { model: String(raw.model || model), latency_ms: Date.now() - started, usage: (raw.usage || {}) as Record<string, unknown>, package: parseJsonObject(content) };
}

async function runDeepSeek(messages: Array<{ role: string; content: string }>, model: string) {
  const apiKey = (process.env.DEEPSEEK_API || process.env.DEEPSEEK_API_KEY || "").trim();
  if (!apiKey) throw new Error("DEEPSEEK_API is not configured.");
  const started = Date.now();
  const compatible = messages.map((message) => ({ ...message, role: message.role === "developer" ? "system" : message.role }));
  const invoke = async (requestMessages: Array<{ role: string; content: string }>, thinking: boolean) => {
    const response = await fetchWithTimeout(`${(process.env.DEEPSEEK_BASE_URL || "https://api.deepseek.com").replace(/\/$/, "")}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
      body: JSON.stringify({
        model,
        messages: requestMessages,
        thinking: { type: thinking ? "enabled" : "disabled" },
        ...(thinking ? { reasoning_effort: "high" } : {}),
        response_format: { type: "json_object" },
        max_tokens: 14_000,
      }),
    });
    const raw = await response.json() as Record<string, unknown>;
    if (!response.ok) {
      const error = raw.error as Record<string, unknown> | undefined;
      throw new Error(`DeepSeek ${response.status}: ${String(error?.code || error?.message || "request failed")}`);
    }
    const choices = Array.isArray(raw.choices) ? raw.choices as Array<Record<string, unknown>> : [];
    const message = choices[0]?.message as Record<string, unknown> | undefined;
    return { raw, content: String(message?.content || "") };
  };
  let response = await invoke(compatible, true);
  let packageValue: Record<string, unknown>;
  try {
    packageValue = parseJsonObject(response.content);
  } catch {
    response = await invoke([
      ...compatible,
      { role: "assistant", content: response.content },
      { role: "user", content: "Repair the preceding response into one valid JSON object matching the supplied schema. Return JSON only." },
    ], false);
    packageValue = parseJsonObject(response.content);
  }
  return { model: String(response.raw.model || model), latency_ms: Date.now() - started, usage: (response.raw.usage || {}) as Record<string, unknown>, package: packageValue };
}

function articlePrompt(candidate: Record<string, unknown>, sources: EditorialSource[]) {
  const supportedIds = Array.isArray(candidate.supporting_source_ids)
    ? candidate.supporting_source_ids.map(String)
    : [];
  const candidateSources = sources.filter((source) => supportedIds.includes(source.source_id));
  const frozenSources = candidateSources.length ? candidateSources : sources;
  return [
    {
      role: "developer",
      content: [
        "You are the Daily AI Column Editor for a financial and regulatory intelligence publication.",
        "Write a polished but explicitly editable Medium-style article of 900-1,400 words for the supplied candidate angle.",
        "Use only the supplied frozen headlines and descriptions. Never imply that you read the full articles.",
        "Never invent facts, quotations, statistics, first-person experience, or the human author's opinion.",
        "Every factual claim must include one or more supplied source_id values in square brackets.",
        "Clearly signal inference, prediction, and recommendation as such. Do not include a bibliography or cite IDs that were not supplied.",
        "Use a strong headline, optional subtitle, short paragraphs, useful section headings, and an analytical conclusion.",
        "Return only the article as clean plain text, with headings on their own lines and no Markdown symbols. Do not add notes about these instructions.",
      ].join(" "),
    },
    {
      role: "user",
      content: `Expand this candidate angle into the article.\n\nCandidate: ${JSON.stringify(candidate, null, 2)}\n\nFrozen sources: ${JSON.stringify(frozenSources, null, 2)}`,
    },
  ];
}

function openAiText(raw: Record<string, unknown>): string {
  if (typeof raw.output_text === "string") return raw.output_text.trim();
  const output = Array.isArray(raw.output) ? raw.output as Array<Record<string, unknown>> : [];
  return output.flatMap((item) => Array.isArray(item.content) ? item.content as Array<Record<string, unknown>> : [])
    .map((item) => String(item.text || ""))
    .filter(Boolean)
    .join("\n")
    .trim();
}

async function runOpenAiArticle(messages: Array<{ role: string; content: string }>, model: string) {
  const apiKey = process.env.OPENAI_API_KEY?.trim();
  if (!apiKey) throw new Error("OPENAI_API_KEY is not configured.");
  const started = Date.now();
  const response = await fetchWithTimeout("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({ model, input: messages, reasoning: { effort: "medium" }, max_output_tokens: 7_000, store: false }),
  });
  const raw = await response.json() as Record<string, unknown>;
  if (!response.ok) {
    const error = raw.error as Record<string, unknown> | undefined;
    throw new Error(`OpenAI ${response.status}: ${String(error?.code || error?.message || "request failed")}`);
  }
  const article = openAiText(raw);
  if (!article) throw new Error("OpenAI returned an empty article.");
  return { model: String(raw.model || model), latency_ms: Date.now() - started, usage: (raw.usage || {}) as Record<string, unknown>, article };
}

async function runDeepSeekArticle(messages: Array<{ role: string; content: string }>, model: string) {
  const apiKey = (process.env.DEEPSEEK_API || process.env.DEEPSEEK_API_KEY || "").trim();
  if (!apiKey) throw new Error("DEEPSEEK_API is not configured.");
  const started = Date.now();
  const compatible = messages.map((message) => ({ ...message, role: message.role === "developer" ? "system" : message.role }));
  const response = await fetchWithTimeout(`${(process.env.DEEPSEEK_BASE_URL || "https://api.deepseek.com").replace(/\/$/, "")}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({ model, messages: compatible, thinking: { type: "enabled" }, reasoning_effort: "high", max_tokens: 7_000 }),
  });
  const raw = await response.json() as Record<string, unknown>;
  if (!response.ok) {
    const error = raw.error as Record<string, unknown> | undefined;
    throw new Error(`DeepSeek ${response.status}: ${String(error?.code || error?.message || "request failed")}`);
  }
  const choices = Array.isArray(raw.choices) ? raw.choices as Array<Record<string, unknown>> : [];
  const message = choices[0]?.message as Record<string, unknown> | undefined;
  const article = String(message?.content || "").trim();
  if (!article) throw new Error("DeepSeek returned an empty article.");
  return { model: String(raw.model || model), latency_ms: Date.now() - started, usage: (raw.usage || {}) as Record<string, unknown>, article };
}

function localDateParts(date: Date, timezone: string): { date: string; hour: number; minute: number } {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: timezone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  }).formatToParts(date);
  const value = (type: Intl.DateTimeFormatPartTypes) => parts.find((part) => part.type === type)?.value || "00";
  return { date: `${value("year")}-${value("month")}-${value("day")}`, hour: Number(value("hour")), minute: Number(value("minute")) };
}

function normalizeJson<T>(value: unknown, fallback: T): T {
  if (typeof value === "string") {
    try { return JSON.parse(value) as T; } catch { return fallback; }
  }
  return (value ?? fallback) as T;
}

function normalizeCandidateDraft(row: Record<string, unknown>): EditorialCandidateDraft {
  return {
    id: Number(row.id),
    output_id: Number(row.output_id),
    candidate_id: String(row.candidate_id || ""),
    provider: String(row.provider) as EditorialProvider,
    model: String(row.model || ""),
    status: String(row.status) as EditorialCandidateDraft["status"],
    article: String(row.article || ""),
    latency_ms: Number(row.latency_ms || 0),
    usage: normalizeJson(row.usage, {}),
    error: String(row.error || ""),
    created_at: String(row.created_at),
    updated_at: String(row.updated_at),
  };
}

function normalizeOutput(row: Record<string, unknown>, candidateDrafts: EditorialCandidateDraft[] = []): EditorialOutput {
  return {
    id: Number(row.id),
    run_id: Number(row.run_id),
    provider: String(row.provider) as EditorialProvider,
    model: String(row.model || ""),
    status: String(row.status) as EditorialOutput["status"],
    latency_ms: Number(row.latency_ms || 0),
    usage: normalizeJson(row.usage, {}),
    package: row.package == null ? null : normalizeJson(row.package, {}),
    error: String(row.error || ""),
    created_at: String(row.created_at),
    candidate_drafts: candidateDrafts.filter((draft) => draft.output_id === Number(row.id)),
  };
}

function normalizeRun(row: Record<string, unknown>, outputs: EditorialOutput[]): EditorialRun {
  return {
    id: Number(row.id),
    run_date: String(row.run_date).slice(0, 10),
    trigger: String(row.trigger) as EditorialRun["trigger"],
    status: String(row.status) as EditorialRun["status"],
    snapshot_hash: String(row.snapshot_hash || ""),
    source_count: Number(row.source_count || 0),
    source_snapshot: normalizeJson(row.source_snapshot, []),
    settings_snapshot: sanitizeScheduledEditorialSettings(normalizeJson(row.settings_snapshot, {})),
    error: String(row.error || ""),
    started_at: String(row.started_at),
    finished_at: row.finished_at ? String(row.finished_at) : null,
    outputs,
  };
}

export async function listEditorialRuns(limit = 20): Promise<EditorialRun[]> {
  await ensureScheduledEditorialSchema();
  const safeLimit = Math.max(1, Math.min(50, Math.round(limit)));
  const sql = getSql();
  const runRows = await sql`SELECT * FROM scheduled_editorial_runs ORDER BY started_at DESC LIMIT ${safeLimit}` as unknown as Record<string, unknown>[];
  if (!runRows.length) return [];
  const ids = runRows.map((row) => Number(row.id));
  const outputRows = await sql`SELECT * FROM scheduled_editorial_outputs WHERE run_id = ANY(${ids}::bigint[]) ORDER BY provider ASC` as unknown as Record<string, unknown>[];
  const outputIds = outputRows.map((row) => Number(row.id));
  const draftRows = outputIds.length
    ? await sql`SELECT * FROM scheduled_editorial_candidate_drafts WHERE output_id = ANY(${outputIds}::bigint[]) ORDER BY created_at ASC` as unknown as Record<string, unknown>[]
    : [];
  const candidateDrafts = draftRows.map(normalizeCandidateDraft);
  const outputs = outputRows.map((row) => normalizeOutput(row, candidateDrafts));
  return runRows.map((row) => normalizeRun(row, outputs.filter((output) => output.run_id === Number(row.id))));
}

export async function generateCandidateArticle(input: {
  runId: number;
  outputId: number;
  candidateId: string;
  regenerate?: boolean;
}): Promise<EditorialCandidateDraft> {
  await ensureScheduledEditorialSchema();
  const runId = Math.round(Number(input.runId));
  const outputId = Math.round(Number(input.outputId));
  const candidateId = String(input.candidateId || "").trim();
  if (!Number.isSafeInteger(runId) || runId < 1 || !Number.isSafeInteger(outputId) || outputId < 1 || !/^[a-zA-Z0-9._:-]{1,100}$/.test(candidateId)) {
    throw new Error("A valid run, output, and candidate are required.");
  }

  const sql = getSql();
  if (!input.regenerate) {
    const existingRows = await sql`
      SELECT * FROM scheduled_editorial_candidate_drafts
      WHERE output_id = ${outputId} AND candidate_id = ${candidateId} AND status = 'completed'
      LIMIT 1
    ` as unknown as Record<string, unknown>[];
    if (existingRows[0]) return normalizeCandidateDraft(existingRows[0]);
  }

  const rows = await sql`
    SELECT o.*, r.source_snapshot
    FROM scheduled_editorial_outputs o
    JOIN scheduled_editorial_runs r ON r.id = o.run_id
    WHERE o.id = ${outputId} AND o.run_id = ${runId} AND o.status = 'completed'
    LIMIT 1
  ` as unknown as Record<string, unknown>[];
  const row = rows[0];
  if (!row) throw new Error("The requested completed editorial output was not found.");
  const packageValue = normalizeJson<Record<string, unknown> | null>(row.package, null);
  if (!packageValue) throw new Error("The editorial output does not contain a candidate package.");
  const candidates = Array.isArray(packageValue.candidates)
    ? packageValue.candidates.filter((value): value is Record<string, unknown> => Boolean(value) && typeof value === "object" && !Array.isArray(value))
    : [];
  const candidate = candidates.find((value) => String(value.candidate_id || "") === candidateId);
  if (!candidate) throw new Error("That candidate angle is not part of this editorial output.");

  const provider = String(row.provider) as EditorialProvider;
  if (provider !== "openai" && provider !== "deepseek") throw new Error("The editorial provider is not supported.");
  const model = String(row.model || (provider === "openai" ? DEFAULT_SETTINGS.openai_model : DEFAULT_SETTINGS.deepseek_model));
  const sources = normalizeJson<EditorialSource[]>(row.source_snapshot, []);
  const messages = articlePrompt(candidate, sources);

  try {
    const result = provider === "openai"
      ? await runOpenAiArticle(messages, model)
      : await runDeepSeekArticle(messages, model);
    const usage = JSON.stringify(result.usage);
    const saved = await sql`
      INSERT INTO scheduled_editorial_candidate_drafts (output_id, candidate_id, provider, model, status, article, latency_ms, usage, error)
      VALUES (${outputId}, ${candidateId}, ${provider}, ${result.model}, 'completed', ${result.article}, ${result.latency_ms}, ${usage}::jsonb, '')
      ON CONFLICT (output_id, candidate_id) DO UPDATE SET
        provider = EXCLUDED.provider,
        model = EXCLUDED.model,
        status = EXCLUDED.status,
        article = EXCLUDED.article,
        latency_ms = EXCLUDED.latency_ms,
        usage = EXCLUDED.usage,
        error = '',
        updated_at = now()
      RETURNING *
    ` as unknown as Record<string, unknown>[];
    return normalizeCandidateDraft(saved[0]);
  } catch (error) {
    const message = (error instanceof Error ? error.message : String(error)).slice(0, 1000);
    const usage = JSON.stringify({});
    await sql`
      INSERT INTO scheduled_editorial_candidate_drafts (output_id, candidate_id, provider, model, status, article, latency_ms, usage, error)
      VALUES (${outputId}, ${candidateId}, ${provider}, ${model}, 'failed', '', 0, ${usage}::jsonb, ${message})
      ON CONFLICT (output_id, candidate_id) DO UPDATE SET
        provider = EXCLUDED.provider,
        model = EXCLUDED.model,
        status = 'failed',
        article = '',
        latency_ms = 0,
        usage = EXCLUDED.usage,
        error = EXCLUDED.error,
        updated_at = now()
    `;
    throw error;
  }
}

async function saveOutput(runId: number, provider: EditorialProvider, result: Awaited<ReturnType<typeof runOpenAi>> | null, error: string) {
  const sql = getSql();
  const model = result?.model || (provider === "openai" ? DEFAULT_SETTINGS.openai_model : DEFAULT_SETTINGS.deepseek_model);
  const usage = JSON.stringify(result?.usage || {});
  const packageJson = result ? JSON.stringify(result.package) : null;
  await sql`
    INSERT INTO scheduled_editorial_outputs (run_id, provider, model, status, latency_ms, usage, package, error)
    VALUES (${runId}, ${provider}, ${model}, ${result ? "completed" : "failed"}, ${result?.latency_ms || 0}, ${usage}::jsonb, ${packageJson}::jsonb, ${error})
    ON CONFLICT (run_id, provider) DO UPDATE SET
      model = EXCLUDED.model,
      status = EXCLUDED.status,
      latency_ms = EXCLUDED.latency_ms,
      usage = EXCLUDED.usage,
      package = EXCLUDED.package,
      error = EXCLUDED.error,
      created_at = now()
  `;
}

export async function runScheduledEditorial(trigger: "manual" | "scheduled", now = new Date()): Promise<{ skipped?: string; run?: EditorialRun }> {
  await ensureScheduledEditorialSchema();
  const settings = await getScheduledEditorialSettings();
  const local = localDateParts(now, settings.timezone);
  if (trigger === "scheduled") {
    if (!settings.enabled) return { skipped: "Scheduled editorial briefing is disabled." };
    if (local.hour !== settings.hour || local.minute < settings.minute || local.minute >= settings.minute + 15) {
      return { skipped: `Not inside the configured ${String(settings.hour).padStart(2, "0")}:${String(settings.minute).padStart(2, "0")} ${settings.timezone} run window.` };
    }
  }
  const providers: EditorialProvider[] = [];
  if (settings.openai_enabled) providers.push("openai");
  if (settings.deepseek_enabled) providers.push("deepseek");
  if (!providers.length) throw new Error("Enable at least one provider before running the editorial briefing.");

  const sql = getSql();
  const dedupeKey = trigger === "scheduled" ? `scheduled:${local.date}` : `manual:${Date.now()}:${randomUUID()}`;
  if (trigger === "scheduled") {
    const existing = await sql`SELECT id FROM scheduled_editorial_runs WHERE dedupe_key = ${dedupeKey} LIMIT 1` as unknown as Array<{ id: number }>;
    if (existing.length) return { skipped: `The ${local.date} scheduled briefing already exists.` };
  }

  const sources = await captureSources(settings.lookback_hours);
  const snapshotHash = createHash("sha256").update(JSON.stringify(sources)).digest("hex");
  const sourceJson = JSON.stringify(sources);
  const settingsJson = JSON.stringify(settings);
  const inserted = await sql`
    INSERT INTO scheduled_editorial_runs (dedupe_key, run_date, trigger, status, snapshot_hash, source_count, source_snapshot, settings_snapshot)
    VALUES (${dedupeKey}, ${local.date}, ${trigger}, 'running', ${snapshotHash}, ${sources.length}, ${sourceJson}::jsonb, ${settingsJson}::jsonb)
    RETURNING id
  ` as unknown as Array<{ id: number }>;
  const runId = Number(inserted[0]?.id);

  if (sources.length < 3) {
    const message = `Only ${sources.length} eligible AI sources were found; at least 3 are required.`;
    await sql`UPDATE scheduled_editorial_runs SET status = 'failed', error = ${message}, finished_at = now() WHERE id = ${runId}`;
    const [run] = await listEditorialRuns(1);
    return { run };
  }

  const messages = promptMessages(sources, settings);
  const outcomes = await Promise.all(providers.map(async (provider) => {
    try {
      const result = provider === "openai"
        ? await runOpenAi(messages, settings.openai_model)
        : await runDeepSeek(messages, settings.deepseek_model);
      await saveOutput(runId, provider, result, "");
      return { provider, ok: true as const };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      await saveOutput(runId, provider, null, message.slice(0, 1000));
      return { provider, ok: false as const, error: message };
    }
  }));
  const succeeded = outcomes.filter((outcome) => outcome.ok).length;
  const status: EditorialRun["status"] = succeeded === outcomes.length ? "completed" : succeeded > 0 ? "partial" : "failed";
  const errors = outcomes.filter((outcome) => !outcome.ok).map((outcome) => `${outcome.provider}: ${"error" in outcome ? outcome.error : "failed"}`).join("; ");
  await sql`UPDATE scheduled_editorial_runs SET status = ${status}, error = ${errors.slice(0, 2000)}, finished_at = now() WHERE id = ${runId}`;
  const rows = await listEditorialRuns(20);
  return { run: rows.find((run) => run.id === runId) };
}

export function scheduledEditorialRuntimeStatus() {
  return {
    openai_configured: Boolean(process.env.OPENAI_API_KEY?.trim()),
    deepseek_configured: Boolean((process.env.DEEPSEEK_API || process.env.DEEPSEEK_API_KEY || "").trim()),
  };
}

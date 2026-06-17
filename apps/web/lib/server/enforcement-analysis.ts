import { normalizeText } from "@/lib/server/api-utils";
import { getOpenAiConfig } from "@/lib/server/env";
import type { CustomDocumentRecord, JsonValue } from "@/lib/server/types";

export interface EnforcementAiAnalysis {
  thesis: string;
  entities: string[];
  why_it_matters: string[];
  legal_theory: string[];
  risk_signals: string[];
  follow_up_questions: string[];
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

export function isEnforcementAiAnalysis(value: unknown): value is EnforcementAiAnalysis {
  if (!value || typeof value !== "object") {
    return false;
  }
  const raw = value as Record<string, unknown>;
  return Boolean(
    normalizeText(raw.thesis) &&
    Array.isArray(raw.why_it_matters) &&
    Array.isArray(raw.legal_theory) &&
    Array.isArray(raw.risk_signals) &&
    Array.isArray(raw.follow_up_questions)
  );
}

export function analysisToJsonValue(analysis: EnforcementAiAnalysis, model: string): Record<string, JsonValue> {
  return {
    ...analysis,
    model,
    generated_at: new Date().toISOString(),
  } as Record<string, JsonValue>;
}

export function jsonValueToAnalysis(value: unknown): EnforcementAiAnalysis | null {
  if (!isEnforcementAiAnalysis(value)) {
    return null;
  }
  return {
    thesis: normalizeText(value.thesis),
    entities: stringList(value.entities, 8),
    why_it_matters: stringList(value.why_it_matters, 4),
    legal_theory: stringList(value.legal_theory, 4),
    risk_signals: stringList(value.risk_signals, 4),
    follow_up_questions: stringList(value.follow_up_questions, 4),
  };
}

function extractResponseText(payload: OpenAiPayload): string {
  const direct = normalizeText(payload.output_text);
  if (direct) {
    return direct;
  }
  const pieces: string[] = [];
  for (const item of payload.output || []) {
    if (item?.type !== "message" || !Array.isArray(item.content)) {
      continue;
    }
    for (const content of item.content) {
      const text = normalizeText(content?.text);
      if (text) {
        pieces.push(text);
      }
    }
  }
  return pieces.join("\n").trim();
}

function cleanJsonText(value: string): string {
  return value
    .replace(/^```(?:json)?/i, "")
    .replace(/```$/i, "")
    .trim();
}

function stringList(value: unknown, maxItems: number): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item) => normalizeText(item).slice(0, 220))
    .filter(Boolean)
    .slice(0, maxItems);
}

function coerceAnalysis(raw: unknown): EnforcementAiAnalysis {
  const src = raw && typeof raw === "object" ? raw as Record<string, unknown> : {};
  return {
    thesis: normalizeText(src.thesis).slice(0, 280),
    entities: stringList(src.entities, 8),
    why_it_matters: stringList(src.why_it_matters, 4),
    legal_theory: stringList(src.legal_theory, 4),
    risk_signals: stringList(src.risk_signals, 4),
    follow_up_questions: stringList(src.follow_up_questions, 4),
  };
}

function trimText(value: string, maxChars: number): string {
  const text = normalizeText(value);
  return text.length > maxChars ? `${text.slice(0, maxChars)}...` : text;
}

function buildInput(doc: CustomDocumentRecord): string {
  const metadata = doc.metadata || {};
  const fullText = normalizeText(doc.content?.full_text);
  return [
    `Title: ${normalizeText(metadata.title) || "Untitled enforcement action"}`,
    `Date: ${normalizeText(metadata.published_date) || normalizeText(metadata.date)}`,
    `Release No: ${normalizeText(metadata.release_no)}`,
    `Source Kind: ${normalizeText(metadata.source_kind)}`,
    `Doc Type: ${normalizeText(metadata.doc_type)}`,
    `URL: ${normalizeText(metadata.url)}`,
    `Action Type: ${normalizeText(metadata.action_type)}`,
    `Forum: ${normalizeText(metadata.forum)}`,
    `Outcome: ${normalizeText(metadata.outcome_status)}`,
    `Alleged Violations: ${Array.isArray(metadata.alleged_violations) ? metadata.alleged_violations.join(", ") : normalizeText(metadata.alleged_violations)}`,
    `Entities: ${Array.isArray(metadata.entities) ? metadata.entities.join(", ") : normalizeText(metadata.entities)}`,
    `Sanctions: ${Array.isArray(metadata.sanctions) ? metadata.sanctions.join(", ") : normalizeText(metadata.sanctions)}`,
    "",
    "Release text:",
    trimText(fullText, 9000),
  ].join("\n");
}

async function callOpenAi(input: string, model: string): Promise<string> {
  const cfg = getOpenAiConfig();
  const response = await fetch(`${cfg.baseUrl.replace(/\/$/, "")}/responses`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${cfg.apiKey}`,
    },
    body: JSON.stringify({
      model,
      instructions: [
        "You are an enforcement analyst writing for securities compliance, legal, and risk teams.",
        "Analyze only the supplied enforcement release text and metadata. Do not invent facts.",
        "Return strict JSON with keys: thesis, entities, why_it_matters, legal_theory, risk_signals, follow_up_questions.",
        "entities must list named defendants/respondents, issuers, companies, funds, offering vehicles, broker-dealers, securities, or other market participants involved in the alleged conduct. Do not include SEC staff, courts, or assisting agencies unless they are defendants/respondents.",
        "why_it_matters must explain concrete market, investor-protection, compliance, or supervision significance from the release. Do not use generic enforcement boilerplate.",
        "risk_signals must identify specific red flags from the release, such as account usage, timing, compensation, disclosure gaps, controls failures, harmed investor class, or transaction structure.",
        "Each list should contain concise, specific bullets tied to facts in the release.",
      ].join("\n"),
      input,
      text: {
        format: {
          type: "json_schema",
          name: "enforcement_action_analysis",
          strict: true,
          schema: {
            type: "object",
            additionalProperties: false,
            properties: {
              thesis: { type: "string" },
              entities: { type: "array", items: { type: "string" } },
              why_it_matters: { type: "array", items: { type: "string" } },
              legal_theory: { type: "array", items: { type: "string" } },
              risk_signals: { type: "array", items: { type: "string" } },
              follow_up_questions: { type: "array", items: { type: "string" } },
            },
            required: ["thesis", "entities", "why_it_matters", "legal_theory", "risk_signals", "follow_up_questions"],
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
  if (!response.ok) {
    throw new Error(normalizeText(json?.error?.message) || normalizeText(text) || `OpenAI request failed: ${response.status}`);
  }
  if (!json) {
    throw new Error("OpenAI returned a non-JSON response.");
  }
  return extractResponseText(json);
}

export async function generateEnforcementAnalysis(doc: CustomDocumentRecord, modelOverride = ""): Promise<{
  model: string;
  analysis: EnforcementAiAnalysis;
}> {
  const cfg = getOpenAiConfig();
  if (!cfg.apiKey) {
    throw new Error("OPENAI_API_KEY is not configured.");
  }
  const model = normalizeText(modelOverride) || normalizeText(cfg.model) || "gpt-4.1-mini";
  const responseText = await callOpenAi(buildInput(doc), model);
  const parsed = JSON.parse(cleanJsonText(responseText)) as unknown;
  const analysis = coerceAnalysis(parsed);
  if (!analysis.thesis) {
    throw new Error("OpenAI returned analysis without a thesis.");
  }
  return { model, analysis };
}

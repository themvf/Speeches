import { createRequestId, fail, normalizeText, ok } from "@/lib/server/api-utils";
import { loadCorpusDocuments } from "@/lib/server/data-store";
import { getOpenAiConfig } from "@/lib/server/env";
import { getClientIp, getGenerateGlobalLimiter, getGenerateIpLimiter, isRateLimited } from "@/lib/server/rate-limit";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

interface EnforcementAnalysisPayload {
  document_id: string;
  generated_at: string;
  model: string;
  analysis: {
    thesis: string;
    why_it_matters: string[];
    legal_theory: string[];
    risk_signals: string[];
    follow_up_questions: string[];
  };
}

type OpenAiTextContent = { type?: string; text?: string };
type OpenAiOutput = { type?: string; content?: OpenAiTextContent[] };
type OpenAiPayload = {
  output_text?: string;
  output?: OpenAiOutput[];
  error?: { message?: string };
};

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

function coerceAnalysis(raw: unknown): EnforcementAnalysisPayload["analysis"] {
  const src = raw && typeof raw === "object" ? raw as Record<string, unknown> : {};
  return {
    thesis: normalizeText(src.thesis).slice(0, 280),
    why_it_matters: stringList(src.why_it_matters, 4),
    legal_theory: stringList(src.legal_theory, 4),
    risk_signals: stringList(src.risk_signals, 4),
    follow_up_questions: stringList(src.follow_up_questions, 4),
  };
}

function buildFallbackAnalysis(title: string, summary: string): EnforcementAnalysisPayload["analysis"] {
  return {
    thesis: summary || `${title} appears to be an enforcement action requiring review of the alleged conduct, posture, and cited provisions.`,
    why_it_matters: ["The matter may indicate an active enforcement priority or repeat pattern in the current action set."],
    legal_theory: ["Review the release text and extracted citations to confirm the SEC or FINRA theory."],
    risk_signals: ["The extracted metadata is incomplete, so analyst review is needed before drawing stronger conclusions."],
    follow_up_questions: ["What conduct triggered the action?", "Is this a filed case, settlement, judgment, or administrative disposition?"],
  };
}

function trimText(value: string, maxChars: number): string {
  const text = normalizeText(value);
  return text.length > maxChars ? `${text.slice(0, maxChars)}...` : text;
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
        "Return strict JSON with keys: thesis, why_it_matters, legal_theory, risk_signals, follow_up_questions.",
        "Each list should contain concise, specific bullets. Avoid generic statements.",
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
              why_it_matters: { type: "array", items: { type: "string" } },
              legal_theory: { type: "array", items: { type: "string" } },
              risk_signals: { type: "array", items: { type: "string" } },
              follow_up_questions: { type: "array", items: { type: "string" } },
            },
            required: ["thesis", "why_it_matters", "legal_theory", "risk_signals", "follow_up_questions"],
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

export async function POST(request: Request) {
  const requestId = createRequestId();
  const ip = getClientIp(request.headers);
  if (await isRateLimited(getGenerateIpLimiter(), ip) || await isRateLimited(getGenerateGlobalLimiter(), "global")) {
    return fail("Rate limit exceeded. Please slow down.", "RATE_LIMITED", 429, requestId);
  }

  try {
    const cfg = getOpenAiConfig();
    if (!cfg.apiKey) {
      return fail("OPENAI_API_KEY is not configured.", "OPENAI_NOT_CONFIGURED", 503, requestId);
    }

    const body = await request.json().catch(() => ({})) as Record<string, unknown>;
    const documentId = normalizeText(body.document_id);
    if (!documentId) {
      return fail("document_id is required.", "DOCUMENT_ID_REQUIRED", 400, requestId);
    }

    const corpus = await loadCorpusDocuments();
    const doc = corpus.find((item) => normalizeText(item.metadata?.document_id) === documentId);
    if (!doc) {
      return fail("Document not found.", "DOCUMENT_NOT_FOUND", 404, requestId);
    }

    const metadata = doc.metadata || {};
    const title = normalizeText(metadata.title) || "Untitled enforcement action";
    const date = normalizeText(metadata.published_date) || normalizeText(metadata.date);
    const releaseNo = normalizeText(metadata.release_no);
    const sourceKind = normalizeText(metadata.source_kind);
    const fullText = normalizeText(doc.content?.full_text);
    const summary = normalizeText(metadata.summary) || trimText(fullText, 380);

    const input = [
      `Title: ${title}`,
      `Date: ${date}`,
      `Release No: ${releaseNo}`,
      `Source Kind: ${sourceKind}`,
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
      trimText(fullText || summary, 9000),
    ].join("\n");

    const model = normalizeText(cfg.model) || "gpt-4.1-mini";
    const responseText = await callOpenAi(input, model);
    let parsed: unknown = null;
    try {
      parsed = JSON.parse(cleanJsonText(responseText));
    } catch {
      parsed = null;
    }
    const analysis = parsed ? coerceAnalysis(parsed) : buildFallbackAnalysis(title, summary);
    if (!analysis.thesis) {
      analysis.thesis = buildFallbackAnalysis(title, summary).thesis;
    }

    return ok<EnforcementAnalysisPayload>({
      document_id: documentId,
      generated_at: new Date().toISOString(),
      model,
      analysis,
    }, requestId);
  } catch (error) {
    console.error("[enforcement/analyze]", error);
    return fail(
      `Failed to generate enforcement analysis: ${error instanceof Error ? error.message : "Unknown error"}`,
      "ENFORCEMENT_ANALYSIS_FAILED",
      500,
      requestId
    );
  }
}

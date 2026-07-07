import { type NextRequest } from "next/server";
import { askVectorStoreChat, type ChatHistoryMessage } from "@/lib/server/openai-chat";
import { parseComparableDate } from "@/lib/server/data-store";
import { createRequestId, fail, normalizeText, ok } from "@/lib/server/api-utils";
import { getOpenAiConfig } from "@/lib/server/env";
import { getClientIp, getGenerateGlobalLimiter, getGenerateIpLimiter, isRateLimited } from "@/lib/server/rate-limit";
import type { DocumentListItem } from "@/lib/server/types";
import { listActiveVectorStores, loadVectorStoreState, type VectorStoreStatePayload } from "@/lib/server/vector-state";

export const runtime = "nodejs";

function clampInt(value: unknown, fallback: number, minValue: number, maxValue: number): number {
  const parsed = Number.parseInt(String(value ?? ""), 10);
  const n = Number.isFinite(parsed) ? parsed : fallback;
  return Math.max(minValue, Math.min(maxValue, n));
}

function normalizeHistory(value: unknown): ChatHistoryMessage[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item) => {
      if (!item || typeof item !== "object") {
        return null;
      }
      const record = item as Record<string, unknown>;
      const role = String(record.role ?? "").trim().toLowerCase();
      if (role !== "user" && role !== "assistant") {
        return null;
      }
      const content = normalizeText(record.content);
      if (!content) {
        return null;
      }
      return {
        role,
        content
      } as ChatHistoryMessage;
    })
    .filter((item): item is ChatHistoryMessage => Boolean(item));
}

function latestIndexedDate(items: Array<{ published_at?: string; date?: string }>): string {
  let latestValue = 0;
  let latestText = "";
  for (const item of items) {
    const text = normalizeText(item.published_at || item.date || "");
    if (!text) {
      continue;
    }
    const comparable = parseComparableDate(text);
    if (comparable > latestValue) {
      latestValue = comparable;
      latestText = text;
    }
  }
  return latestText;
}

function buildIndexedDocumentItems(state: VectorStoreStatePayload): DocumentListItem[] {
  const items: DocumentListItem[] = [];
  for (const [orgKey, store] of Object.entries(state.stores || {})) {
    for (const [documentId, value] of Object.entries(store.docs || {})) {
      const doc = value && typeof value === "object" ? (value as Record<string, unknown>) : {};
      const date = normalizeText(doc.date);
      items.push({
        document_id: documentId,
        title: normalizeText(doc.title) || normalizeText(doc.filename) || "Unknown document",
        organization: store.org_label || orgKey.toUpperCase(),
        source_kind: orgKey,
        source_format: normalizeText(doc.source_format) || "text",
        extraction_quality: normalizeText(doc.extraction_quality) || "full_text",
        full_text_available: doc.full_text_available === false ? false : true,
        doc_type: "Document",
        speaker: normalizeText(doc.speaker),
        url: normalizeText(doc.url),
        date,
        published_at: date,
        word_count: 0,
        tags: [],
        keywords: [],
        topics: [],
        ingest_status: "existing",
        enrichment_status: "not_enriched",
        enrichment_summary: "",
        enrichment_model: "",
        enrichment_confidence: 0,
        review_decision: "pending",
        updated_at: normalizeText(doc.indexed_at),
        sentiment_label: "",
        sentiment_score: 0
      });
    }
  }
  return items;
}

export async function POST(request: NextRequest) {
  const requestId = createRequestId();
  const ip = getClientIp(request.headers);
  if (await isRateLimited(getGenerateIpLimiter(), ip)) {
    return fail("Rate limit exceeded. Please slow down.", "RATE_LIMITED", 429, requestId);
  }
  if (await isRateLimited(getGenerateGlobalLimiter(), "global")) {
    return fail("Server is busy. Please try again shortly.", "GLOBAL_RATE_LIMITED", 429, requestId);
  }

  try {
    let body: Record<string, unknown> = {};
    try {
      body = (await request.json()) as Record<string, unknown>;
    } catch {
      body = {};
    }

    const prompt = normalizeText(body.prompt);
    if (!prompt) {
      return fail("Prompt is required.", "CHAT_PROMPT_REQUIRED", 400, requestId);
    }
    if (prompt.length > 4000) {
      return fail("Prompt exceeds maximum length of 4000 characters.", "CHAT_PROMPT_TOO_LONG", 400, requestId);
    }

    const openAi = getOpenAiConfig();
    if (!openAi.apiKey) {
      return fail(
        "OPENAI_API_KEY is not configured for the web app. Add it to Vercel project environment variables.",
        "CHAT_OPENAI_NOT_CONFIGURED",
        500,
        requestId
      );
    }

    const topK = clampInt(body.top_k, 8, 1, 12);
    const history = normalizeHistory(body.history).slice(-6);

    const vectorState = await loadVectorStoreState();
    const vectorStores = listActiveVectorStores(vectorState);
    if (!vectorStores.length) {
      return fail(
        "No active vector stores were found. Build/Sync the Knowledge Index before using web chat.",
        "CHAT_VECTOR_STORE_MISSING",
        500,
        requestId
      );
    }

    const items = buildIndexedDocumentItems(vectorState);
    const result = await askVectorStoreChat({
      prompt,
      history,
      topK,
      vectorStoreIds: vectorStores.map((item) => item.vector_store_id),
      documents: items,
      latestIndexedDate: latestIndexedDate(items),
      model: openAi.model
    });

    return ok(result, requestId);
  } catch (error) {
    console.error("[chats/ask]", error);
    return fail("Failed to answer chat request.", "CHAT_ANSWER_FAILED", 500, requestId);
  }
}

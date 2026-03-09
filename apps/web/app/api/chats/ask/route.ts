import { askVectorStoreChat, type ChatHistoryMessage } from "@/lib/server/openai-chat";
import { buildDocumentListItems, loadCorpusDocuments, loadEnrichmentState, parseComparableDate } from "@/lib/server/data-store";
import { createRequestId, fail, normalizeText, ok } from "@/lib/server/api-utils";
import { getOpenAiConfig } from "@/lib/server/env";
import { listActiveVectorStores, loadVectorStoreState } from "@/lib/server/vector-state";

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

export async function POST(request: Request) {
  const requestId = createRequestId();

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
    const history = normalizeHistory(body.history);

    const [corpusDocs, enrichment, vectorState] = await Promise.all([
      loadCorpusDocuments(),
      loadEnrichmentState(),
      loadVectorStoreState()
    ]);

    const vectorStores = listActiveVectorStores(vectorState);
    if (!vectorStores.length) {
      return fail(
        "No active vector stores were found. Build/Sync the Knowledge Index before using web chat.",
        "CHAT_VECTOR_STORE_MISSING",
        500,
        requestId
      );
    }

    const items = buildDocumentListItems(corpusDocs, enrichment);
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
    return fail(
      `Failed to answer chat request: ${error instanceof Error ? error.message : "Unknown error"}`,
      "CHAT_ANSWER_FAILED",
      500,
      requestId
    );
  }
}

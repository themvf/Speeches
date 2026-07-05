import { createRequestId, fail, normalizeText, ok, toInt } from "@/lib/server/api-utils";
import { loadCorpusDocuments } from "@/lib/server/data-store";
import { fetchSemanticDocIds } from "@/lib/server/openai-chat";
import { getClientIp, getSearchLimiter, isRateLimited } from "@/lib/server/rate-limit";
import type { CustomDocumentRecord } from "@/lib/server/types";
import { listActiveVectorStores, loadVectorStoreState } from "@/lib/server/vector-state";

export const runtime = "nodejs";

const SEMANTIC_SEARCH_TIMEOUT_MS = 12_000;
const FALLBACK_LOAD_TIMEOUT_MS = 4_000;

const STOP_WORDS = new Set([
  "a",
  "an",
  "and",
  "are",
  "as",
  "at",
  "be",
  "by",
  "for",
  "from",
  "in",
  "into",
  "is",
  "it",
  "of",
  "on",
  "or",
  "the",
  "to",
  "with"
]);

type SearchResultPayload = {
  document_ids: string[];
  snippets: Record<string, string>;
  mode: "semantic" | "keyword_fallback";
  warning?: string;
};

function toMs(value: unknown): number {
  const ms = Date.parse(String(value || ""));
  return Number.isFinite(ms) ? ms : 0;
}

function termsForQuery(query: string): string[] {
  const terms = normalizeText(query)
    .toLowerCase()
    .split(/[^a-z0-9$.-]+/i)
    .map((term) => term.trim())
    .filter((term) => term.length >= 2 && !STOP_WORDS.has(term));
  return [...new Set(terms)].slice(0, 12);
}

function bestSnippet(text: string, terms: string[], maxLength = 280): string {
  const normalized = normalizeText(text);
  if (!normalized) {
    return "";
  }
  const lower = normalized.toLowerCase();
  const firstMatch = terms
    .map((term) => lower.indexOf(term.toLowerCase()))
    .filter((index) => index >= 0)
    .sort((a, b) => a - b)[0];
  const start = Math.max(0, (firstMatch ?? 0) - 80);
  return normalized.slice(start, start + maxLength).trim();
}

function scoreDocument(doc: CustomDocumentRecord, query: string, terms: string[]): { score: number; snippet: string; publishedMs: number } {
  const metadata = doc.metadata || {};
  const title = normalizeText(metadata.title);
  const summary = normalizeText(metadata.summary);
  const source = normalizeText(metadata.source_name || metadata.organization || metadata.source_kind);
  const fullText = normalizeText(doc.content?.full_text);
  const titleLower = title.toLowerCase();
  const summaryLower = summary.toLowerCase();
  const sourceLower = source.toLowerCase();
  const fullLower = fullText.toLowerCase();
  const queryLower = query.toLowerCase();

  let score = 0;
  if (titleLower.includes(queryLower)) score += 80;
  if (summaryLower.includes(queryLower)) score += 35;
  if (fullLower.includes(queryLower)) score += 20;

  for (const term of terms) {
    if (titleLower.includes(term)) score += 20;
    if (summaryLower.includes(term)) score += 8;
    if (sourceLower.includes(term)) score += 6;
    if (fullLower.includes(term)) score += 2;
  }

  const publishedMs = toMs(metadata.published_at || metadata.published_date || metadata.date);
  return {
    score,
    snippet: bestSnippet([title, summary, fullText].filter(Boolean).join(" "), terms),
    publishedMs
  };
}

function loadCorpusWithBudget(): Promise<CustomDocumentRecord[]> {
  const guarded = loadCorpusDocuments().catch((error) => {
    console.warn("[api/search] Keyword fallback corpus load failed:", error);
    return [];
  });
  return Promise.race([
    guarded,
    new Promise<CustomDocumentRecord[]>((resolve) => {
      setTimeout(() => resolve([]), FALLBACK_LOAD_TIMEOUT_MS);
    })
  ]);
}

function withSemanticBudget<T>(promise: Promise<T>): Promise<T> {
  const guarded = promise.catch((error) => {
    throw error;
  });
  return Promise.race([
    guarded,
    new Promise<T>((_, reject) => {
      setTimeout(() => reject(new Error(`Semantic search exceeded ${SEMANTIC_SEARCH_TIMEOUT_MS}ms budget.`)), SEMANTIC_SEARCH_TIMEOUT_MS);
    })
  ]);
}

async function fetchKeywordFallback(query: string, topK: number, warning: string): Promise<SearchResultPayload> {
  const terms = termsForQuery(query);
  if (!terms.length) {
    return { document_ids: [], snippets: {}, mode: "keyword_fallback", warning };
  }

  const documents = await loadCorpusWithBudget();
  const ranked = documents
    .map((doc) => {
      const documentId = normalizeText(doc.metadata?.document_id);
      if (!documentId) {
        return null;
      }
      const scored = scoreDocument(doc, query, terms);
      if (scored.score <= 0) {
        return null;
      }
      return { documentId, ...scored };
    })
    .filter((item): item is { documentId: string; score: number; snippet: string; publishedMs: number } => Boolean(item))
    .sort((a, b) => b.score - a.score || b.publishedMs - a.publishedMs)
    .slice(0, topK);

  const snippets: Record<string, string> = {};
  for (const row of ranked) {
    if (row.snippet) {
      snippets[row.documentId] = row.snippet;
    }
  }

  return {
    document_ids: ranked.map((row) => row.documentId),
    snippets,
    mode: "keyword_fallback",
    warning
  };
}

export async function GET(request: Request) {
  const requestId = createRequestId();

  const ip = getClientIp(request.headers);
  if (await isRateLimited(getSearchLimiter(), ip)) {
    return fail("Rate limit exceeded. Please slow down.", "RATE_LIMITED", 429, requestId);
  }

  const { searchParams } = new URL(request.url);
  const q = normalizeText(searchParams.get("q"));
  if (!q) return fail("Missing q", "MISSING_Q", 400, requestId);
  if (q.length > 2000) return fail("Query too long (max 2000 characters).", "QUERY_TOO_LONG", 400, requestId);

  const topK = toInt(searchParams.get("topK"), 20, 5, 50);

  try {
    const state = await loadVectorStoreState();
    const stores = listActiveVectorStores(state);
    const vectorStoreIds = stores.map((s) => s.vector_store_id);
    if (!vectorStoreIds.length) {
      const result = await fetchKeywordFallback(q, topK, "No vector stores are configured; used keyword fallback.");
      return ok(result, requestId);
    }

    const result = await withSemanticBudget(fetchSemanticDocIds(q, vectorStoreIds, topK));
    return ok({ ...result, mode: "semantic" } satisfies SearchResultPayload, requestId);
  } catch (error) {
    const warning = `Semantic search failed: ${error instanceof Error ? error.message : "Unknown error"}`;
    const result = await fetchKeywordFallback(q, topK, warning);
    return ok(result, requestId);
  }
}

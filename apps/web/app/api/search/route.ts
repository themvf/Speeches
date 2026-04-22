import { createRequestId, fail, normalizeText, ok, toInt } from "@/lib/server/api-utils";
import { fetchSemanticDocIds } from "@/lib/server/openai-chat";
import { listActiveVectorStores, loadVectorStoreState } from "@/lib/server/vector-state";

export const runtime = "nodejs";

export async function GET(request: Request) {
  const requestId = createRequestId();
  const { searchParams } = new URL(request.url);
  const q = normalizeText(searchParams.get("q"));
  if (!q) return fail("Missing q", "MISSING_Q", 400, requestId);

  const topK = toInt(searchParams.get("topK"), 20, 5, 50);

  try {
    const state = await loadVectorStoreState();
    const stores = listActiveVectorStores(state);
    const vectorStoreIds = stores.map((s) => s.vector_store_id);
    if (!vectorStoreIds.length) {
      return fail("No vector stores configured", "NO_STORES", 503, requestId);
    }

    const result = await fetchSemanticDocIds(q, vectorStoreIds, topK);
    return ok(result, requestId);
  } catch (error) {
    return fail(
      `Semantic search failed: ${error instanceof Error ? error.message : "Unknown error"}`,
      "SEARCH_FAILED",
      500,
      requestId
    );
  }
}

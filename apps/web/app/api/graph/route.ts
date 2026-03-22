import { createRequestId, fail, normalizeText, ok, parseDate, toInt } from "@/lib/server/api-utils";
import { buildKnowledgeGraphData } from "@/lib/server/knowledge-graph";
import { normalizeFacetToken } from "@/lib/server/document-query";
import type { GraphResponseData } from "@/lib/server/types";

export const runtime = "nodejs";

function parseBoolean(value: string | null): boolean {
  const normalized = String(value || "")
    .trim()
    .toLowerCase();
  return normalized === "1" || normalized === "true" || normalized === "yes" || normalized === "on";
}

export async function GET(request: Request) {
  const requestId = createRequestId();

  try {
    const url = new URL(request.url);
    const q = normalizeText(url.searchParams.get("q"));
    const org = normalizeText(url.searchParams.get("org"));
    const sourceKind = normalizeText(url.searchParams.get("source_kind") || url.searchParams.get("source"));
    const topic = normalizeFacetToken(url.searchParams.get("topic") || "");
    const keyword = normalizeFacetToken(url.searchParams.get("keyword") || "");
    const tag = normalizeFacetToken(url.searchParams.get("tag") || "");
    const status = normalizeText(url.searchParams.get("status"));
    const fromDate = parseDate(url.searchParams.get("date_from"));
    const toDate = parseDate(url.searchParams.get("date_to"));
    const includeDocuments = parseBoolean(url.searchParams.get("include_documents"));
    const maxNodes = toInt(url.searchParams.get("max_nodes"), 80, 1, 240);
    const maxEdges = toInt(url.searchParams.get("max_edges"), 160, 0, 480);

    const payload = await buildKnowledgeGraphData(
      {
        q,
        org,
        sourceKind,
        topic,
        keyword,
        tag,
        status,
        fromDate,
        toDate
      },
      {
        includeDocuments,
        maxNodes,
        maxEdges
      }
    );

    return ok<GraphResponseData>(payload, requestId);
  } catch (error) {
    return fail(
      `Failed to build graph data: ${error instanceof Error ? error.message : "Unknown error"}`,
      "GRAPH_BUILD_FAILED",
      500,
      requestId
    );
  }
}

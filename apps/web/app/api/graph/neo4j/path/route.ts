import { createRequestId, fail, normalizeText, ok, parseDate } from "@/lib/server/api-utils";
import { buildKnowledgeGraphData } from "@/lib/server/knowledge-graph";
import { normalizeFacetToken } from "@/lib/server/document-query";
import {
  buildNeo4jProjectionKey,
  findShortestPathInNeo4j,
  getNeo4jStatus,
  syncGraphProjectionToNeo4j
} from "@/lib/server/neo4j";
import type { Neo4jPathResponseData } from "@/lib/server/types";

export const runtime = "nodejs";

function parseBoolean(value: unknown): boolean {
  const normalized = String(value || "")
    .trim()
    .toLowerCase();
  return normalized === "1" || normalized === "true" || normalized === "yes" || normalized === "on";
}

function parseInteger(value: unknown, fallback: number, minValue: number, maxValue: number): number {
  const parsed = Number.parseInt(String(value ?? ""), 10);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.max(minValue, Math.min(maxValue, parsed));
}

export async function POST(request: Request) {
  const requestId = createRequestId();

  try {
    const status = getNeo4jStatus();
    if (!status.configured) {
      return fail(
        `Neo4j is not configured. Missing: ${status.missing_required_env.join(", ")}`,
        "NEO4J_NOT_CONFIGURED",
        503,
        requestId
      );
    }

    const body = (await request.json()) as Record<string, unknown>;
    const q = normalizeText(body.q);
    const org = normalizeText(body.org);
    const sourceKind = normalizeText(body.source_kind || body.source);
    const topic = normalizeFacetToken(String(body.topic || ""));
    const keyword = normalizeFacetToken(String(body.keyword || ""));
    const tag = normalizeFacetToken(String(body.tag || ""));
    const statusFilter = normalizeText(body.status);
    const fromDate = parseDate(typeof body.date_from === "string" ? body.date_from : null);
    const toDate = parseDate(typeof body.date_to === "string" ? body.date_to : null);
    const includeDocuments = parseBoolean(body.include_documents);
    const maxNodes = parseInteger(body.max_nodes, 80, 1, 240);
    const maxEdges = parseInteger(body.max_edges, 160, 0, 480);
    const maxHops = parseInteger(body.max_hops, 4, 1, 6);
    const sourceNodeId = normalizeText(body.source_node_id);
    const targetNodeId = normalizeText(body.target_node_id);

    if (!sourceNodeId || !targetNodeId) {
      return fail("Source node and target node are required.", "GRAPH_PATH_INPUT_INVALID", 400, requestId);
    }

    const graph = await buildKnowledgeGraphData(
      {
        q,
        org,
        sourceKind,
        topic,
        keyword,
        tag,
        status: statusFilter,
        fromDate,
        toDate
      },
      {
        includeDocuments,
        maxNodes,
        maxEdges
      }
    );

    const nodeIds = new Set(graph.nodes.map((node) => node.id));
    if (!nodeIds.has(sourceNodeId) || !nodeIds.has(targetNodeId)) {
      return fail(
        "Both selected nodes must exist in the current graph slice before running the Neo4j path query.",
        "GRAPH_PATH_NODE_NOT_FOUND",
        400,
        requestId
      );
    }

    const projectionKey = buildNeo4jProjectionKey({
      q,
      org,
      sourceKind,
      topic,
      keyword,
      tag,
      status: statusFilter,
      date_from: fromDate?.toISOString().slice(0, 10) || "",
      date_to: toDate?.toISOString().slice(0, 10) || "",
      includeDocuments,
      maxNodes,
      maxEdges
    });

    await syncGraphProjectionToNeo4j(projectionKey, graph.nodes, graph.edges);
    const path = await findShortestPathInNeo4j({
      projectionKey,
      sourceNodeId,
      targetNodeId,
      maxHops
    });

    return ok<Neo4jPathResponseData>(
      {
        ...path,
        synced_node_count: graph.nodes.length,
        synced_edge_count: graph.edges.length
      },
      requestId
    );
  } catch (error) {
    return fail(
      `Failed to run Neo4j path query: ${error instanceof Error ? error.message : "Unknown error"}`,
      "NEO4J_PATH_FAILED",
      500,
      requestId
    );
  }
}

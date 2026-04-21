import { createHash } from "node:crypto";

import { getNeo4jConfig } from "@/lib/server/env";
import type { GraphEdge, GraphNode, JsonValue, Neo4jPathResponseData, Neo4jStatusResponseData } from "@/lib/server/types";

interface Neo4jHttpRow {
  row: unknown[];
}

interface Neo4jHttpResult {
  columns: string[];
  data: Neo4jHttpRow[];
}

interface Neo4jHttpError {
  code: string;
  message: string;
}

interface Neo4jHttpResponse {
  results: Neo4jHttpResult[];
  errors: Neo4jHttpError[];
}

type SyncableNode = {
  id: string;
  kind: string;
  label: string;
  document_count: number;
  degree: number;
  metadata_json: string;
};

type SyncableEdge = {
  id: string;
  kind: string;
  source: string;
  target: string;
  weight: number;
  document_count: number;
  evidence_doc_ids: string[];
  metadata_json: string;
};

function neo4jEndpoint(url: string, database: string): string {
  return `${url.replace(/\/+$/, "")}/db/${encodeURIComponent(database)}/tx/commit`;
}

function parseMetadataJson(value: unknown): Record<string, JsonValue> {
  if (typeof value !== "string" || !value.trim()) {
    return {};
  }

  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? (parsed as Record<string, JsonValue>) : {};
  } catch {
    return {};
  }
}

function normalizeGraphNode(value: unknown): GraphNode | null {
  if (!value || typeof value !== "object") {
    return null;
  }

  const item = value as Record<string, unknown>;
  const id = String(item.id || "").trim();
  const kind = String(item.kind || "").trim();
  const label = String(item.label || "").trim();

  if (!id || !kind || !label) {
    return null;
  }

  return {
    id,
    kind: kind as GraphNode["kind"],
    label,
    document_count: Number(item.document_count || 0),
    degree: Number(item.degree || 0),
    metadata: parseMetadataJson(item.metadata_json)
  };
}

function normalizeGraphEdge(value: unknown): GraphEdge | null {
  if (!value || typeof value !== "object") {
    return null;
  }

  const item = value as Record<string, unknown>;
  const id = String(item.id || "").trim();
  const kind = String(item.kind || "").trim();
  const source = String(item.source || "").trim();
  const target = String(item.target || "").trim();

  if (!id || !kind || !source || !target) {
    return null;
  }

  return {
    id,
    kind: kind as GraphEdge["kind"],
    source,
    target,
    weight: Number(item.weight || 0),
    document_count: Number(item.document_count || 0),
    evidence_doc_ids: Array.isArray(item.evidence_doc_ids)
      ? item.evidence_doc_ids.map((entry) => String(entry || "").trim()).filter(Boolean)
      : [],
    metadata: parseMetadataJson(item.metadata_json)
  };
}

async function runCypher<T = Record<string, unknown>>(
  statement: string,
  parameters: Record<string, unknown>
): Promise<T[]> {
  const config = getNeo4jConfig();
  if (!config.configured) {
    throw new Error(`Neo4j is not configured. Missing: ${config.missingRequiredEnv.join(", ")}`);
  }

  const response = await fetch(neo4jEndpoint(config.url, config.database), {
    method: "POST",
    cache: "no-store",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Basic ${Buffer.from(`${config.username}:${config.password}`).toString("base64")}`
    },
    body: JSON.stringify({
      statements: [
        {
          statement,
          parameters
        }
      ]
    })
  });

  const payload = (await response.json()) as Neo4jHttpResponse;

  if (!response.ok) {
    const firstError = payload?.errors?.[0];
    throw new Error(firstError?.message || `Neo4j request failed (${response.status})`);
  }

  if (payload.errors?.length) {
    throw new Error(payload.errors[0].message || "Neo4j query failed.");
  }

  const firstResult = payload.results?.[0];
  if (!firstResult) {
    return [];
  }

  return firstResult.data.map((entry) => {
    const row = entry.row || [];
    const record: Record<string, unknown> = {};
    firstResult.columns.forEach((column, index) => {
      record[column] = row[index];
    });
    return record as T;
  });
}

function toSyncableNodes(nodes: GraphNode[]): SyncableNode[] {
  return nodes.map((node) => ({
    id: node.id,
    kind: node.kind,
    label: node.label,
    document_count: node.document_count,
    degree: node.degree,
    metadata_json: JSON.stringify(node.metadata || {})
  }));
}

function toSyncableEdges(edges: GraphEdge[]): SyncableEdge[] {
  return edges.map((edge) => ({
    id: edge.id,
    kind: edge.kind,
    source: edge.source,
    target: edge.target,
    weight: edge.weight,
    document_count: edge.document_count,
    evidence_doc_ids: edge.evidence_doc_ids || [],
    metadata_json: JSON.stringify(edge.metadata || {})
  }));
}

export function getNeo4jStatus(): Neo4jStatusResponseData {
  const config = getNeo4jConfig();

  return {
    configured: config.configured,
    database: config.database,
    url: config.url,
    missing_required_env: config.missingRequiredEnv
  };
}

export function buildNeo4jProjectionKey(payload: Record<string, unknown>): string {
  const hash = createHash("sha256").update(JSON.stringify(payload)).digest("hex").slice(0, 16);
  return `graph_${hash}`;
}

export async function syncGraphProjectionToNeo4j(
  projectionKey: string,
  nodes: GraphNode[],
  edges: GraphEdge[]
): Promise<void> {
  const syncableNodes = toSyncableNodes(nodes);
  const syncableEdges = toSyncableEdges(edges);

  await runCypher(
    `
      MATCH (n:GraphNode {projection_key: $projectionKey})
      DETACH DELETE n
    `,
    { projectionKey }
  );

  if (syncableNodes.length > 0) {
    await runCypher(
      `
        UNWIND $nodes AS node
        MERGE (n:GraphNode {projection_key: $projectionKey, id: node.id})
        SET n.kind = node.kind,
            n.label = node.label,
            n.document_count = toInteger(node.document_count),
            n.degree = toInteger(node.degree),
            n.metadata_json = node.metadata_json,
            n.synced_at = datetime(),
            n.projection_key = $projectionKey
      `,
      {
        projectionKey,
        nodes: syncableNodes
      }
    );
  }

  if (syncableEdges.length > 0) {
    await runCypher(
      `
        UNWIND $edges AS edge
        MATCH (source:GraphNode {projection_key: $projectionKey, id: edge.source})
        MATCH (target:GraphNode {projection_key: $projectionKey, id: edge.target})
        MERGE (source)-[r:GRAPH_EDGE {projection_key: $projectionKey, id: edge.id}]->(target)
        SET r.kind = edge.kind,
            r.source = edge.source,
            r.target = edge.target,
            r.weight = toInteger(edge.weight),
            r.document_count = toInteger(edge.document_count),
            r.evidence_doc_ids = edge.evidence_doc_ids,
            r.metadata_json = edge.metadata_json,
            r.synced_at = datetime(),
            r.projection_key = $projectionKey
      `,
      {
        projectionKey,
        edges: syncableEdges
      }
    );
  }
}

export async function findShortestPathInNeo4j(args: {
  projectionKey: string;
  sourceNodeId: string;
  targetNodeId: string;
  maxHops: number;
}): Promise<Neo4jPathResponseData> {
  const projectionKey = String(args.projectionKey || "").trim();
  const sourceNodeId = String(args.sourceNodeId || "").trim();
  const targetNodeId = String(args.targetNodeId || "").trim();
  const maxHops = Math.max(1, Math.min(args.maxHops || 4, 6));

  if (!projectionKey || !sourceNodeId || !targetNodeId) {
    throw new Error("Projection key, source node, and target node are required.");
  }

  if (sourceNodeId === targetNodeId) {
    const rows = await runCypher<{
      node: Record<string, unknown>;
    }>(
      `
        MATCH (n:GraphNode {projection_key: $projectionKey, id: $nodeId})
        RETURN {
          id: n.id,
          kind: n.kind,
          label: n.label,
          document_count: coalesce(n.document_count, 0),
          degree: coalesce(n.degree, 0),
          metadata_json: coalesce(n.metadata_json, "{}")
        } AS node
      `,
      {
        projectionKey,
        nodeId: sourceNodeId
      }
    );

    const node = normalizeGraphNode(rows[0]?.node);
    if (!node) {
      throw new Error("Selected node was not found in the Neo4j projection.");
    }

    return {
      projection_key: projectionKey,
      synced_node_count: 0,
      synced_edge_count: 0,
      path_found: true,
      hops: 0,
      nodes: [node],
      edges: []
    };
  }

  const rows = await runCypher<{
    path_found: boolean;
    hops: number;
    nodes: Record<string, unknown>[];
    edges: Record<string, unknown>[];
  }>(
    `
      MATCH (source:GraphNode {projection_key: $projectionKey, id: $sourceNodeId})
      MATCH (target:GraphNode {projection_key: $projectionKey, id: $targetNodeId})
      OPTIONAL MATCH p = shortestPath((source)-[:GRAPH_EDGE*..${maxHops}]-(target))
      RETURN
        CASE WHEN p IS NULL THEN false ELSE true END AS path_found,
        CASE WHEN p IS NULL THEN 0 ELSE length(p) END AS hops,
        CASE
          WHEN p IS NULL THEN []
          ELSE [n IN nodes(p) | {
            id: n.id,
            kind: n.kind,
            label: n.label,
            document_count: coalesce(n.document_count, 0),
            degree: coalesce(n.degree, 0),
            metadata_json: coalesce(n.metadata_json, "{}")
          }]
        END AS nodes,
        CASE
          WHEN p IS NULL THEN []
          ELSE [r IN relationships(p) | {
            id: r.id,
            kind: r.kind,
            source: startNode(r).id,
            target: endNode(r).id,
            weight: coalesce(r.weight, 0),
            document_count: coalesce(r.document_count, 0),
            evidence_doc_ids: coalesce(r.evidence_doc_ids, []),
            metadata_json: coalesce(r.metadata_json, "{}")
          }]
        END AS edges
    `,
    {
      projectionKey,
      sourceNodeId,
      targetNodeId
    }
  );

  const result = rows[0];
  const pathNodes = Array.isArray(result?.nodes) ? result.nodes.map(normalizeGraphNode).filter(Boolean) as GraphNode[] : [];
  const pathEdges = Array.isArray(result?.edges) ? result.edges.map(normalizeGraphEdge).filter(Boolean) as GraphEdge[] : [];

  return {
    projection_key: projectionKey,
    synced_node_count: 0,
    synced_edge_count: 0,
    path_found: Boolean(result?.path_found),
    hops: Number(result?.hops || 0),
    nodes: pathNodes,
    edges: pathEdges
  };
}

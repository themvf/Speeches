import {
  buildDocumentListItems,
  buildDocumentsFacets,
  loadCorpusDocuments,
  loadEnrichmentState,
  parseComparableDate
} from "@/lib/server/data-store";
import type { DocumentListFilters } from "@/lib/server/document-query";
import { buildFullTextById, filterDocumentListItems, normalizeFacetToken } from "@/lib/server/document-query";
import type {
  DocumentListItem,
  GraphEdge,
  GraphEdgeKind,
  GraphNode,
  GraphNodeKind,
  GraphResponseData,
  JsonValue
} from "@/lib/server/types";

const MAX_EVIDENCE_DOC_IDS = 12;

const NODE_KIND_PRIORITY: Record<GraphNodeKind, number> = {
  organization: 6,
  topic: 5,
  entity: 4,
  speaker: 3,
  keyword: 2,
  document: 1
};

const EDGE_KIND_PRIORITY: Record<GraphEdgeKind, number> = {
  org_topic: 9,
  topic_entity: 8,
  org_entity: 7,
  speaker_topic: 6,
  org_keyword: 5,
  published_by: 3,
  spoken_by: 3,
  has_topic: 2,
  has_keyword: 2,
  mentions_entity: 2
};

type GraphOptions = {
  includeDocuments?: boolean;
  maxNodes?: number;
  maxEdges?: number;
};

type NodeAccumulator = Omit<GraphNode, "degree">;
type EdgeAccumulator = GraphEdge;

function uniqueNormalized(values: string[]): Array<{ key: string; label: string }> {
  const dedup = new Map<string, string>();

  for (const value of values) {
    const label = String(value || "").trim();
    const key = normalizeFacetToken(label);
    if (!key || dedup.has(key)) {
      continue;
    }
    dedup.set(key, label);
  }

  return [...dedup.entries()].map(([key, label]) => ({ key, label }));
}

function buildNodeId(kind: GraphNodeKind, key: string): string {
  return `${kind}:${key}`;
}

function buildEdgeId(kind: GraphEdgeKind, source: string, target: string): string {
  return `${kind}:${source}:${target}`;
}

function mergeMetadata(
  base: Record<string, JsonValue>,
  incoming?: Record<string, JsonValue>
): Record<string, JsonValue> {
  if (!incoming) {
    return base;
  }

  for (const [key, value] of Object.entries(incoming)) {
    if (!(key in base)) {
      base[key] = value;
    }
  }

  return base;
}

function addNode(
  nodes: Map<string, NodeAccumulator>,
  kind: GraphNodeKind,
  key: string,
  label: string,
  docId: string,
  metadata?: Record<string, JsonValue>
): string | null {
  const normalizedKey = kind === "document" ? String(key || "").trim() : normalizeFacetToken(key || label);
  const normalizedLabel = String(label || "").trim();

  if (!normalizedKey || !normalizedLabel) {
    return null;
  }

  const id = buildNodeId(kind, normalizedKey);
  const existing = nodes.get(id);

  if (existing) {
    existing.document_count += 1;
    mergeMetadata(existing.metadata, metadata);
    return id;
  }

  nodes.set(id, {
    id,
    kind,
    label: normalizedLabel,
    document_count: docId ? 1 : 0,
    metadata: metadata ? { ...metadata } : {}
  });

  return id;
}

function addEdge(
  edges: Map<string, EdgeAccumulator>,
  kind: GraphEdgeKind,
  source: string | null,
  target: string | null,
  docId: string,
  metadata?: Record<string, JsonValue>
) {
  if (!source || !target || source === target) {
    return;
  }

  const id = buildEdgeId(kind, source, target);
  const existing = edges.get(id);

  if (existing) {
    existing.weight += 1;
    existing.document_count += 1;
    if (docId && existing.evidence_doc_ids.length < MAX_EVIDENCE_DOC_IDS && !existing.evidence_doc_ids.includes(docId)) {
      existing.evidence_doc_ids.push(docId);
    }
    mergeMetadata(existing.metadata, metadata);
    return;
  }

  edges.set(id, {
    id,
    kind,
    source,
    target,
    weight: 1,
    document_count: 1,
    evidence_doc_ids: docId ? [docId] : [],
    metadata: metadata ? { ...metadata } : {}
  });
}

function computeDegrees(edges: GraphEdge[]): Map<string, number> {
  const degrees = new Map<string, number>();

  for (const edge of edges) {
    degrees.set(edge.source, (degrees.get(edge.source) || 0) + 1);
    degrees.set(edge.target, (degrees.get(edge.target) || 0) + 1);
  }

  return degrees;
}

function nodeScore(node: GraphNode): number {
  return node.document_count * 100 + node.degree * 10 + NODE_KIND_PRIORITY[node.kind];
}

function edgeScore(edge: GraphEdge): number {
  return edge.weight * 100 + edge.document_count * 10 + EDGE_KIND_PRIORITY[edge.kind];
}

function sortNodes(nodes: GraphNode[]): GraphNode[] {
  return [...nodes].sort((a, b) => {
    const diff = nodeScore(b) - nodeScore(a);
    if (diff !== 0) {
      return diff;
    }
    return a.label.localeCompare(b.label);
  });
}

function sortEdges(edges: GraphEdge[]): GraphEdge[] {
  return [...edges].sort((a, b) => {
    const diff = edgeScore(b) - edgeScore(a);
    if (diff !== 0) {
      return diff;
    }
    return a.id.localeCompare(b.id);
  });
}

function summarizeKinds<T extends { kind: string }>(items: T[]): Record<string, number> {
  const counts: Record<string, number> = {};

  for (const item of items) {
    counts[item.kind] = (counts[item.kind] || 0) + 1;
  }

  return counts;
}

function pickGraphSlice(
  allNodes: GraphNode[],
  allEdges: GraphEdge[],
  maxNodes: number,
  maxEdges: number
): Pick<GraphResponseData, "nodes" | "edges"> {
  const sortedNodes = sortNodes(allNodes);
  const sortedEdges = sortEdges(allEdges);
  const selectedNodeIds = new Set<string>();
  const selectedEdges: GraphEdge[] = [];

  for (const edge of sortedEdges) {
    if (selectedEdges.length >= maxEdges) {
      break;
    }

    const missing =
      (selectedNodeIds.has(edge.source) ? 0 : 1) + (selectedNodeIds.has(edge.target) ? 0 : 1);

    if (selectedNodeIds.size + missing > maxNodes) {
      continue;
    }

    selectedEdges.push(edge);
    selectedNodeIds.add(edge.source);
    selectedNodeIds.add(edge.target);
  }

  for (const node of sortedNodes) {
    if (selectedNodeIds.size >= maxNodes) {
      break;
    }
    selectedNodeIds.add(node.id);
  }

  const returnedEdges = sortEdges(selectedEdges);
  const returnedDegrees = computeDegrees(returnedEdges);
  const returnedNodes = sortedNodes
    .filter((node) => selectedNodeIds.has(node.id))
    .map((node) => ({
      ...node,
      degree: returnedDegrees.get(node.id) || 0
    }));

  return {
    nodes: returnedNodes,
    edges: returnedEdges
  };
}

function minMaxDates(items: DocumentListItem[]): Pick<GraphResponseData["summary"], "start_date" | "end_date"> {
  let minDateMs = 0;
  let maxDateMs = 0;

  for (const item of items) {
    const comparable = parseComparableDate(item.published_at || item.date);
    if (!comparable) {
      continue;
    }
    minDateMs = minDateMs === 0 ? comparable : Math.min(minDateMs, comparable);
    maxDateMs = maxDateMs === 0 ? comparable : Math.max(maxDateMs, comparable);
  }

  return {
    start_date: minDateMs ? new Date(minDateMs).toISOString().slice(0, 10) : "",
    end_date: maxDateMs ? new Date(maxDateMs).toISOString().slice(0, 10) : ""
  };
}

export async function buildKnowledgeGraphData(
  filters: DocumentListFilters,
  options?: GraphOptions
): Promise<GraphResponseData> {
  const includeDocuments = Boolean(options?.includeDocuments);
  const maxNodes = Math.max(1, Math.min(options?.maxNodes || 80, 240));
  const maxEdges = Math.max(0, Math.min(options?.maxEdges || 160, 480));

  const [corpusDocs, enrichment] = await Promise.all([loadCorpusDocuments(), loadEnrichmentState()]);
  const items = buildDocumentListItems(corpusDocs, enrichment);
  const facets = buildDocumentsFacets(items);
  const fullTextById = buildFullTextById(corpusDocs);
  const filtered = filterDocumentListItems(items, fullTextById, filters);

  const nodeMap = new Map<string, NodeAccumulator>();
  const edgeMap = new Map<string, EdgeAccumulator>();

  for (const item of filtered) {
    const docId = String(item.document_id || "").trim();
    if (!docId) {
      continue;
    }

    const docNodeId = includeDocuments
      ? addNode(nodeMap, "document", docId, item.title || docId, docId, {
          document_id: docId,
          title: item.title,
          organization: item.organization,
          source_kind: item.source_kind,
          speaker: item.speaker,
          url: item.url,
          published_at: item.published_at || item.date,
          doc_type: item.doc_type
        })
      : null;

    const organizationId = addNode(nodeMap, "organization", item.organization, item.organization, docId);
    const speakerId = item.speaker ? addNode(nodeMap, "speaker", item.speaker, item.speaker, docId) : null;
    const topicNodes = uniqueNormalized(item.topics || []).map(({ key, label }) => ({
      id: addNode(nodeMap, "topic", key, label, docId),
      label
    }));
    const keywordNodes = uniqueNormalized(item.keywords || []).map(({ key, label }) => ({
      id: addNode(nodeMap, "keyword", key, label, docId),
      label
    }));

    const enrichmentEntry = enrichment.entries?.[docId];
    const entityNodes = uniqueNormalized(enrichmentEntry?.enrichment?.entities || []).map(({ key, label }) => ({
      id: addNode(nodeMap, "entity", key, label, docId),
      label
    }));

    if (includeDocuments) {
      addEdge(edgeMap, "published_by", docNodeId, organizationId, docId, { aggregate: false });
      addEdge(edgeMap, "spoken_by", docNodeId, speakerId, docId, { aggregate: false });
      for (const topic of topicNodes) {
        addEdge(edgeMap, "has_topic", docNodeId, topic.id, docId, { aggregate: false });
      }
      for (const keyword of keywordNodes) {
        addEdge(edgeMap, "has_keyword", docNodeId, keyword.id, docId, { aggregate: false });
      }
      for (const entity of entityNodes) {
        addEdge(edgeMap, "mentions_entity", docNodeId, entity.id, docId, { aggregate: false });
      }
    }

    for (const topic of topicNodes) {
      addEdge(edgeMap, "org_topic", organizationId, topic.id, docId, { aggregate: true });
      if (speakerId) {
        addEdge(edgeMap, "speaker_topic", speakerId, topic.id, docId, { aggregate: true });
      }
      for (const entity of entityNodes) {
        addEdge(edgeMap, "topic_entity", topic.id, entity.id, docId, { aggregate: true });
      }
    }

    for (const keyword of keywordNodes) {
      addEdge(edgeMap, "org_keyword", organizationId, keyword.id, docId, { aggregate: true });
    }

    for (const entity of entityNodes) {
      addEdge(edgeMap, "org_entity", organizationId, entity.id, docId, { aggregate: true });
    }
  }

  const allEdges = sortEdges([...edgeMap.values()]);
  const allDegrees = computeDegrees(allEdges);
  const allNodes = sortNodes(
    [...nodeMap.values()].map((node) => ({
      ...node,
      degree: allDegrees.get(node.id) || 0
    }))
  );
  const slice = pickGraphSlice(allNodes, allEdges, maxNodes, maxEdges);
  const dateRange = minMaxDates(filtered);

  return {
    nodes: slice.nodes,
    edges: slice.edges,
    summary: {
      matching_documents: filtered.length,
      node_count: allNodes.length,
      edge_count: allEdges.length,
      returned_nodes: slice.nodes.length,
      returned_edges: slice.edges.length,
      include_documents: includeDocuments,
      nodes_by_kind: summarizeKinds(allNodes),
      edges_by_kind: summarizeKinds(allEdges),
      start_date: dateRange.start_date,
      end_date: dateRange.end_date
    },
    facets
  };
}

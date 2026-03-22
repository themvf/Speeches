"use client";

import { useDeferredValue, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";

import type { DocumentsFacets, GraphEdge, GraphNode, GraphNodeKind, GraphResponseData } from "@/lib/server/types";

interface ApiEnvelope<T> {
  ok: boolean;
  data?: T;
  error?: string;
}

const EMPTY_FACETS: DocumentsFacets = {
  sources: [],
  organizations: [],
  topics: [],
  key_topics: [],
  keywords: [],
  statuses: []
};

const NODE_KIND_ORDER: GraphNodeKind[] = ["organization", "topic", "entity", "speaker", "keyword", "document"];

const SOURCE_KIND_LABELS: Record<string, string> = {
  sec_speech: "SEC Speeches & Statements",
  sec_tm_faq: "SEC Trading & Markets FAQ",
  sec_enforcement_litigation: "SEC Litigation Releases",
  finra_regulatory_notice: "FINRA Regulatory Notices",
  finra_comment_letter: "FINRA Comment Letters",
  finra_key_topic: "FINRA Key Topics",
  doj_usao_press_release: "DOJ USAO Press Releases",
  federal_reserve_speech_testimony: "Federal Reserve Speeches/Testimony",
  cftc_press_release: "CFTC Press Releases",
  cftc_public_statement_remark: "CFTC Public Statements & Remarks",
  jdsupra_article: "JD Supra",
  investmentnews_article: "InvestmentNews",
  citywire_article: "Citywire",
  congress_crs_product: "Congress CRS Products",
  newsapi_article: "News",
  uploaded: "Uploaded"
};

function fmt(value: number): string {
  return new Intl.NumberFormat("en-US").format(value || 0);
}

function fmtDateOnly(value: string): string {
  if (!value) {
    return "-";
  }
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function displaySourceKind(value: string): string {
  const raw = String(value || "").trim();
  if (!raw) {
    return "Unknown";
  }
  return SOURCE_KIND_LABELS[raw] || raw;
}

function displayNodeKind(value: GraphNodeKind): string {
  if (value === "organization") return "Organizations";
  if (value === "topic") return "Topics";
  if (value === "entity") return "Entities";
  if (value === "speaker") return "Speakers";
  if (value === "keyword") return "Keywords";
  return "Documents";
}

function displayEdgeKind(value: GraphEdge["kind"]): string {
  if (value === "org_topic") return "Org -> Topic";
  if (value === "org_entity") return "Org -> Entity";
  if (value === "org_keyword") return "Org -> Keyword";
  if (value === "speaker_topic") return "Speaker -> Topic";
  if (value === "topic_entity") return "Topic -> Entity";
  if (value === "published_by") return "Document -> Org";
  if (value === "spoken_by") return "Document -> Speaker";
  if (value === "has_topic") return "Document -> Topic";
  if (value === "has_keyword") return "Document -> Keyword";
  return "Document -> Entity";
}

function readString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function parseCap(value: string, fallback: number, min: number, max: number): number {
  const parsed = Number.parseInt(String(value || "").trim(), 10);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, parsed));
}

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url, {
    cache: "no-store",
    headers: { "Content-Type": "application/json" }
  });
  const payload = (await res.json()) as ApiEnvelope<T>;
  if (!res.ok || !payload?.ok || !payload.data) {
    throw new Error(payload?.error || `Request failed (${res.status})`);
  }
  return payload.data;
}

function buildGraphUrl(params: {
  q: string;
  org: string;
  source: string;
  topic: string;
  keyword: string;
  status: string;
  dateFrom: string;
  dateTo: string;
  includeDocuments: boolean;
  maxNodes: number;
  maxEdges: number;
}): string {
  const next = new URLSearchParams();
  if (params.q.trim()) next.set("q", params.q.trim());
  if (params.org.trim()) next.set("org", params.org.trim());
  if (params.source.trim()) next.set("source_kind", params.source.trim());
  if (params.topic.trim()) next.set("topic", params.topic.trim());
  if (params.keyword.trim()) next.set("keyword", params.keyword.trim());
  if (params.status.trim()) next.set("status", params.status.trim());
  if (params.dateFrom.trim()) next.set("date_from", params.dateFrom.trim());
  if (params.dateTo.trim()) next.set("date_to", params.dateTo.trim());
  if (params.includeDocuments) next.set("include_documents", "1");
  if (params.maxNodes !== 80) next.set("max_nodes", String(params.maxNodes));
  if (params.maxEdges !== 160) next.set("max_edges", String(params.maxEdges));
  const query = next.toString();
  return query ? `/graph?${query}` : "/graph";
}

function buildCorpusHref(params: {
  q: string;
  org: string;
  source: string;
  topic: string;
  keyword: string;
  status: string;
  dateFrom: string;
  dateTo: string;
}): string {
  const next = new URLSearchParams();
  if (params.q.trim()) next.set("q", params.q.trim());
  if (params.org.trim()) next.set("org", params.org.trim());
  if (params.source.trim()) next.set("source_kind", params.source.trim());
  if (params.topic.trim()) next.set("topic", params.topic.trim());
  if (params.keyword.trim()) next.set("keyword", params.keyword.trim());
  if (params.status.trim()) next.set("status", params.status.trim());
  if (params.dateFrom.trim()) next.set("date_from", params.dateFrom.trim());
  if (params.dateTo.trim()) next.set("date_to", params.dateTo.trim());
  const query = next.toString();
  return query ? `/?${query}` : "/";
}

function buildNodeCorpusHref(
  node: GraphNode,
  params: {
    q: string;
    org: string;
    source: string;
    topic: string;
    keyword: string;
    status: string;
    dateFrom: string;
    dateTo: string;
  }
): string {
  if (node.kind === "organization") {
    return buildCorpusHref({ ...params, org: node.label });
  }
  if (node.kind === "topic") {
    return buildCorpusHref({ ...params, topic: node.label });
  }
  if (node.kind === "keyword") {
    return buildCorpusHref({ ...params, keyword: node.label });
  }
  if (node.kind === "speaker" || node.kind === "entity") {
    return buildCorpusHref({ ...params, q: node.label });
  }
  return buildCorpusHref({ ...params, q: node.label });
}

function buildEdgeCorpusHref(
  edge: GraphEdge,
  nodeById: Map<string, GraphNode>,
  params: {
    q: string;
    org: string;
    source: string;
    topic: string;
    keyword: string;
    status: string;
    dateFrom: string;
    dateTo: string;
  }
): string {
  const sourceNode = nodeById.get(edge.source);
  const targetNode = nodeById.get(edge.target);
  if (!sourceNode || !targetNode) {
    return buildCorpusHref(params);
  }

  if (edge.kind === "org_topic") {
    return buildCorpusHref({ ...params, org: sourceNode.label, topic: targetNode.label });
  }
  if (edge.kind === "org_keyword") {
    return buildCorpusHref({ ...params, org: sourceNode.label, keyword: targetNode.label });
  }
  if (edge.kind === "org_entity") {
    return buildCorpusHref({ ...params, org: sourceNode.label, q: targetNode.label });
  }
  if (edge.kind === "speaker_topic") {
    return buildCorpusHref({ ...params, q: sourceNode.label, topic: targetNode.label });
  }
  if (edge.kind === "topic_entity") {
    return buildCorpusHref({ ...params, topic: sourceNode.label, q: targetNode.label });
  }
  if (edge.kind === "published_by") {
    return buildCorpusHref({ ...params, org: targetNode.label });
  }
  if (edge.kind === "spoken_by") {
    return buildCorpusHref({ ...params, q: targetNode.label });
  }
  if (edge.kind === "has_topic") {
    return buildCorpusHref({ ...params, topic: targetNode.label });
  }
  if (edge.kind === "has_keyword") {
    return buildCorpusHref({ ...params, keyword: targetNode.label });
  }
  return buildCorpusHref({ ...params, q: targetNode.label });
}

export function GraphView() {
  const searchParams = useSearchParams();

  const [q, setQ] = useState(searchParams.get("q") || "");
  const [org, setOrg] = useState(searchParams.get("org") || "");
  const [source, setSource] = useState(searchParams.get("source_kind") || searchParams.get("source") || "");
  const [topic, setTopic] = useState(searchParams.get("topic") || "");
  const [keyword, setKeyword] = useState(searchParams.get("keyword") || "");
  const [status, setStatus] = useState(searchParams.get("status") || "");
  const [dateFrom, setDateFrom] = useState(searchParams.get("date_from") || "");
  const [dateTo, setDateTo] = useState(searchParams.get("date_to") || "");
  const [includeDocuments, setIncludeDocuments] = useState(searchParams.get("include_documents") === "1");
  const [maxNodesInput, setMaxNodesInput] = useState(searchParams.get("max_nodes") || "80");
  const [maxEdgesInput, setMaxEdgesInput] = useState(searchParams.get("max_edges") || "160");

  const [data, setData] = useState<GraphResponseData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const deferredQ = useDeferredValue(q);
  const deferredKeyword = useDeferredValue(keyword);
  const maxNodes = useMemo(() => parseCap(maxNodesInput, 80, 1, 240), [maxNodesInput]);
  const maxEdges = useMemo(() => parseCap(maxEdgesInput, 160, 0, 480), [maxEdgesInput]);

  useEffect(() => {
    const href = buildGraphUrl({
      q,
      org,
      source,
      topic,
      keyword,
      status,
      dateFrom,
      dateTo,
      includeDocuments,
      maxNodes,
      maxEdges
    });
    window.history.replaceState(null, "", href);
  }, [dateFrom, dateTo, includeDocuments, keyword, maxEdges, maxNodes, org, q, source, status, topic]);

  useEffect(() => {
    let canceled = false;

    const load = async () => {
      setLoading(true);
      setError("");

      try {
        const params = new URLSearchParams();
        if (deferredQ.trim()) params.set("q", deferredQ.trim());
        if (org.trim()) params.set("org", org.trim());
        if (source.trim()) params.set("source_kind", source.trim());
        if (topic.trim()) params.set("topic", topic.trim());
        if (deferredKeyword.trim()) params.set("keyword", deferredKeyword.trim());
        if (status.trim()) params.set("status", status.trim());
        if (dateFrom.trim()) params.set("date_from", dateFrom.trim());
        if (dateTo.trim()) params.set("date_to", dateTo.trim());
        if (includeDocuments) params.set("include_documents", "1");
        if (maxNodes !== 80) params.set("max_nodes", String(maxNodes));
        if (maxEdges !== 160) params.set("max_edges", String(maxEdges));

        const payload = await fetchJson<GraphResponseData>(`/api/graph?${params.toString()}`);
        if (!canceled) {
          setData(payload);
        }
      } catch (err) {
        if (!canceled) {
          setData(null);
          setError(err instanceof Error ? err.message : "Failed to load graph.");
        }
      } finally {
        if (!canceled) {
          setLoading(false);
        }
      }
    };

    void load();

    return () => {
      canceled = true;
    };
  }, [dateFrom, dateTo, deferredKeyword, deferredQ, includeDocuments, maxEdges, maxNodes, org, source, status, topic]);

  const facets = data?.facets || EMPTY_FACETS;
  const sourceOptions = useMemo(
    () => [...facets.sources].sort((a, b) => displaySourceKind(a).localeCompare(displaySourceKind(b))),
    [facets.sources]
  );
  const topicOptions = useMemo(
    () => (facets.key_topics.length > 0 ? facets.key_topics : facets.topics.slice(0, 24)),
    [facets.key_topics, facets.topics]
  );
  const nodes = data?.nodes || [];
  const edges = data?.edges || [];
  const nodeById = useMemo(() => new Map(nodes.map((node) => [node.id, node])), [nodes]);
  const groupedNodes = useMemo(() => {
    const groups: Record<GraphNodeKind, GraphNode[]> = {
      organization: [],
      topic: [],
      entity: [],
      speaker: [],
      keyword: [],
      document: []
    };

    for (const node of nodes) {
      groups[node.kind].push(node);
    }

    return groups;
  }, [nodes]);
  const maxEdgeWeight = useMemo(
    () => edges.reduce((best, edge) => (edge.weight > best ? edge.weight : best), 0),
    [edges]
  );
  const baseCorpusFilters = useMemo(
    () => ({
      q,
      org,
      source,
      topic,
      keyword,
      status,
      dateFrom,
      dateTo
    }),
    [dateFrom, dateTo, keyword, org, q, source, status, topic]
  );

  const resetFilters = () => {
    setQ("");
    setOrg("");
    setSource("");
    setTopic("");
    setKeyword("");
    setStatus("");
    setDateFrom("");
    setDateTo("");
    setIncludeDocuments(false);
    setMaxNodesInput("80");
    setMaxEdgesInput("160");
  };

  return (
    <div className="mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-6 px-4 py-6 md:px-8 md:py-10">
      <header className="panel hero-panel reveal p-6 md:p-8">
        <span className="kicker">Knowledge Graph</span>
        <h1 className="mt-3 text-3xl font-bold leading-tight md:text-5xl">Corpus Relationship Map</h1>
        <p className="mt-3 max-w-3xl text-base text-[color:var(--ink-soft)] md:text-lg">
          Build a graph slice from the current corpus filters to see which organizations, topics, entities, speakers,
          and keywords cluster together. This first phase emphasizes auditable relationship signals over graph storage.
        </p>
      </header>
      <section className="grid gap-4 xl:grid-cols-[1.4fr_0.6fr]">
        <article className="panel reveal reveal-delay-1 p-5">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
                Graph Filters
              </p>
              <p className="mt-1 text-sm text-[color:var(--ink-soft)]">
                Use topic and keyword for structured enrichment, or search text for broad mention-based graph slices.
              </p>
            </div>
            <button type="button" className="btn-muted px-3 py-2 text-sm" onClick={resetFilters}>
              Reset
            </button>
          </div>

          <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            <input
              className="form-control px-3 py-2 text-sm"
              value={q}
              onChange={(event) => setQ(event.target.value)}
              placeholder="Search text / entity / speaker"
            />
            <select className="form-control px-3 py-2 text-sm" value={org} onChange={(event) => setOrg(event.target.value)}>
              <option value="">All orgs</option>
              {facets.organizations.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
            <select
              className="form-control px-3 py-2 text-sm"
              value={source}
              onChange={(event) => setSource(event.target.value)}
            >
              <option value="">All sources</option>
              {sourceOptions.map((value) => (
                <option key={value} value={value}>
                  {displaySourceKind(value)}
                </option>
              ))}
            </select>
            <select className="form-control px-3 py-2 text-sm" value={topic} onChange={(event) => setTopic(event.target.value)}>
              <option value="">All topics</option>
              {topicOptions.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
            <div>
              <input
                className="form-control w-full px-3 py-2 text-sm"
                value={keyword}
                onChange={(event) => setKeyword(event.target.value)}
                placeholder="Structured keyword"
                list="graph-keywords"
              />
              <datalist id="graph-keywords">
                {facets.keywords.slice(0, 150).map((value) => (
                  <option key={value} value={value} />
                ))}
              </datalist>
            </div>
            <select
              className="form-control px-3 py-2 text-sm"
              value={status}
              onChange={(event) => setStatus(event.target.value)}
            >
              <option value="">All enrichment statuses</option>
              {facets.statuses.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
            <label className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">
              From
              <input
                type="date"
                className="form-control mt-2 w-full px-3 py-2 text-sm"
                value={dateFrom}
                onChange={(event) => setDateFrom(event.target.value)}
              />
            </label>
            <label className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">
              To
              <input
                type="date"
                className="form-control mt-2 w-full px-3 py-2 text-sm"
                value={dateTo}
                onChange={(event) => setDateTo(event.target.value)}
              />
            </label>
            <div className="rounded-2xl border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.84)] px-4 py-3 text-sm text-[color:var(--ink-soft)]">
              Use <strong>Include documents</strong> when you want a mixed document-plus-concept graph instead of a
              cleaner aggregate concept map.
            </div>
          </div>

          <div className="mt-4 grid gap-3 md:grid-cols-3">
            <label className="rounded-2xl border border-[color:var(--line)] bg-[color:rgba(8,18,30,0.84)] px-4 py-3 text-sm text-[color:var(--ink-soft)]">
              <span className="flex items-center gap-3">
                <input
                  type="checkbox"
                  checked={includeDocuments}
                  onChange={(event) => setIncludeDocuments(event.target.checked)}
                />
                Include documents in graph
              </span>
            </label>
            <label className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">
              Max Nodes
              <input
                type="number"
                min={1}
                max={240}
                className="form-control mt-2 w-full px-3 py-2 text-sm"
                value={maxNodesInput}
                onChange={(event) => setMaxNodesInput(event.target.value)}
              />
            </label>
            <label className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">
              Max Edges
              <input
                type="number"
                min={0}
                max={480}
                className="form-control mt-2 w-full px-3 py-2 text-sm"
                value={maxEdgesInput}
                onChange={(event) => setMaxEdgesInput(event.target.value)}
              />
            </label>
          </div>
        </article>

        <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1">
          <article className="panel p-4">
            <p className="text-xs uppercase tracking-[0.1em]">Matching Docs</p>
            <p className="mt-1 text-2xl font-semibold">{loading ? "..." : fmt(data?.summary.matching_documents || 0)}</p>
          </article>
          <article className="panel p-4">
            <p className="text-xs uppercase tracking-[0.1em]">Returned Slice</p>
            <p className="mt-1 text-sm font-semibold">
              {loading
                ? "..."
                : `${fmt(data?.summary.returned_nodes || 0)} nodes / ${fmt(data?.summary.returned_edges || 0)} edges`}
            </p>
          </article>
          <article className="panel p-4">
            <p className="text-xs uppercase tracking-[0.1em]">Underlying Graph</p>
            <p className="mt-1 text-sm font-semibold">
              {loading ? "..." : `${fmt(data?.summary.node_count || 0)} nodes / ${fmt(data?.summary.edge_count || 0)} edges`}
            </p>
          </article>
          <article className="panel p-4">
            <p className="text-xs uppercase tracking-[0.1em]">Date Range</p>
            <p className="mt-1 text-sm font-semibold">
              {loading
                ? "..."
                : data?.summary.start_date
                  ? `${fmtDateOnly(data.summary.start_date)} to ${fmtDateOnly(data.summary.end_date)}`
                  : "No dated documents"}
            </p>
          </article>
        </section>
      </section>

      {error ? <p className="callout callout-error">{error}</p> : null}
      {!loading && (data?.summary.returned_nodes || 0) < (data?.summary.node_count || 0) ? (
        <p className="callout callout-info">
          Showing the strongest {fmt(data?.summary.returned_nodes || 0)} node(s) and {fmt(data?.summary.returned_edges || 0)} edge(s)
          from a larger filtered graph. Increase the caps if you need a denser slice.
        </p>
      ) : null}
      <section className="grid gap-4 xl:grid-cols-[1.2fr_0.8fr]">
        <article className="panel reveal reveal-delay-2 p-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="text-xl font-semibold">Strongest Relationships</h2>
              <p className="mt-1 text-sm text-[color:var(--ink-soft)]">
                Relationship strength is currently document-count based, with aggregate concept edges prioritized in the
                returned slice.
              </p>
            </div>
            <a
              href={buildGraphUrl({
                q,
                org,
                source,
                topic,
                keyword,
                status,
                dateFrom,
                dateTo,
                includeDocuments,
                maxNodes,
                maxEdges
              })}
              className="link-inline text-sm"
            >
              Copyable URL State
            </a>
          </div>

          {loading ? (
            <div className="mt-4 rounded-2xl border border-[color:var(--line)] bg-[color:rgba(8,18,30,0.84)] px-4 py-16 text-sm">
              Building graph slice...
            </div>
          ) : edges.length === 0 ? (
            <div className="mt-4 rounded-2xl border border-[color:var(--line)] bg-[color:rgba(8,18,30,0.84)] px-4 py-16 text-sm">
              No relationships matched the current filters.
            </div>
          ) : (
            <div className="mt-4 space-y-3">
              {edges.slice(0, 18).map((edge) => {
                const sourceNode = nodeById.get(edge.source);
                const targetNode = nodeById.get(edge.target);
                const width = maxEdgeWeight > 0 ? Math.max(8, Math.round((edge.weight / maxEdgeWeight) * 100)) : 8;

                return (
                  <div
                    key={edge.id}
                    className="rounded-2xl border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.84)] px-4 py-4"
                  >
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div>
                        <p className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">
                          {displayEdgeKind(edge.kind)}
                        </p>
                        <p className="mt-1 text-base font-semibold">
                          {sourceNode?.label || edge.source}
                          {" -> "}
                          {targetNode?.label || edge.target}
                        </p>
                      </div>
                      <div className="text-right text-sm text-[color:var(--ink-soft)]">
                        <p>{fmt(edge.document_count)} docs</p>
                        <p>{fmt(edge.weight)} weight</p>
                      </div>
                    </div>
                    <div className="mt-3 h-2 overflow-hidden rounded-full bg-[color:rgba(79,213,255,0.12)]">
                      <div
                        className="h-full rounded-full bg-[linear-gradient(90deg,rgba(79,213,255,0.95),rgba(240,155,61,0.8))]"
                        style={{ width: `${width}%` }}
                      />
                    </div>
                    <div className="mt-3 flex flex-wrap items-center justify-between gap-3 text-xs text-[color:var(--ink-faint)]">
                      <span>{edge.evidence_doc_ids.length} sample doc id(s) attached</span>
                      <a href={buildEdgeCorpusHref(edge, nodeById, baseCorpusFilters)} className="link-inline">
                        Open matching docs
                      </a>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </article>

        <article className="panel p-5">
          <h2 className="text-xl font-semibold">Central Nodes</h2>
          <p className="mt-1 text-sm text-[color:var(--ink-soft)]">
            Degree reflects the returned graph slice, not the hidden overflow beyond the current caps.
          </p>

          {loading ? (
            <p className="mt-4 text-sm">Loading nodes...</p>
          ) : nodes.length === 0 ? (
            <p className="mt-4 text-sm">Apply narrower filters or include documents to surface more graph structure.</p>
          ) : (
            <div className="mt-4 space-y-3">
              {nodes.slice(0, 12).map((node) => {
                const externalUrl = readString(node.metadata.url);
                return (
                  <div
                    key={node.id}
                    className="rounded-2xl border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.84)] px-4 py-3"
                  >
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div>
                        <p className="text-xs font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">
                          {displayNodeKind(node.kind)}
                        </p>
                        <p className="mt-1 text-base font-semibold">{node.label}</p>
                      </div>
                      <div className="text-right text-sm text-[color:var(--ink-soft)]">
                        <p>{fmt(node.document_count)} docs</p>
                        <p>{fmt(node.degree)} links</p>
                      </div>
                    </div>
                    <div className="mt-3 flex flex-wrap items-center gap-3 text-xs">
                      <a href={buildNodeCorpusHref(node, baseCorpusFilters)} className="link-inline">
                        Open matching docs
                      </a>
                      {node.kind === "document" && externalUrl ? (
                        <a href={externalUrl} target="_blank" rel="noreferrer" className="link-inline">
                          Open source
                        </a>
                      ) : null}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </article>
      </section>

      <section className="grid gap-4 lg:grid-cols-3">
        <article className="panel p-5 lg:col-span-2">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="text-xl font-semibold">Node Inventory</h2>
              <p className="mt-1 text-sm text-[color:var(--ink-soft)]">
                Returned nodes are grouped by graph role so you can inspect the shape of the slice before moving to a
                fuller graph store.
              </p>
            </div>
            <a href={buildCorpusHref(baseCorpusFilters)} className="link-inline text-sm">
              Open filtered corpus
            </a>
          </div>

          <div className="mt-4 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {NODE_KIND_ORDER.map((kind) => {
              const group = groupedNodes[kind];
              return (
                <div
                  key={kind}
                  className="rounded-2xl border border-[color:var(--line)] bg-[color:rgba(8,18,30,0.84)] p-4"
                >
                  <div className="flex items-center justify-between gap-3">
                    <p className="text-sm font-semibold">{displayNodeKind(kind)}</p>
                    <span className="text-xs text-[color:var(--ink-faint)]">{fmt(group.length)}</span>
                  </div>
                  {loading ? (
                    <p className="mt-3 text-sm text-[color:var(--ink-soft)]">Loading...</p>
                  ) : group.length === 0 ? (
                    <p className="mt-3 text-sm text-[color:var(--ink-soft)]">No nodes in this slice.</p>
                  ) : (
                    <div className="mt-3 space-y-2">
                      {group.slice(0, 8).map((node) => (
                        <div key={node.id} className="flex items-start justify-between gap-3 text-sm">
                          <a href={buildNodeCorpusHref(node, baseCorpusFilters)} className="link-inline">
                            {node.label}
                          </a>
                          <span className="text-[color:var(--ink-faint)]">{fmt(node.document_count)}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </article>

        <article className="panel p-5">
          <h2 className="text-xl font-semibold">Graph Readout</h2>
          {loading ? (
            <p className="mt-3 text-sm">Loading summary...</p>
          ) : (
            <div className="mt-3 space-y-3 text-sm">
              <div className="rounded-2xl border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.84)] px-4 py-3">
                <p className="text-xs uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Mode</p>
                <p className="mt-1 text-lg font-semibold">
                  {data?.summary.include_documents ? "Document + Concept Graph" : "Aggregate Concept Graph"}
                </p>
              </div>
              <div className="rounded-2xl border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.84)] px-4 py-3">
                <p className="text-xs uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Returned Node Mix</p>
                <div className="mt-2 flex flex-wrap gap-2">
                  {NODE_KIND_ORDER.filter((kind) => groupedNodes[kind].length > 0).map((kind) => (
                    <span key={kind} className="tone-chip">
                      {displayNodeKind(kind)}: {fmt(groupedNodes[kind].length)}
                    </span>
                  ))}
                  {NODE_KIND_ORDER.every((kind) => groupedNodes[kind].length === 0) ? (
                    <span className="text-[color:var(--ink-faint)]">No returned nodes</span>
                  ) : null}
                </div>
              </div>
              <div className="rounded-2xl border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.84)] px-4 py-3">
                <p className="text-xs uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Full Graph Node Mix</p>
                <div className="mt-2 flex flex-wrap gap-2">
                  {Object.entries(data?.summary.nodes_by_kind || {})
                    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
                    .map(([kind, count]) => (
                      <span key={kind} className="tone-chip">
                        {kind}: {fmt(count)}
                      </span>
                    ))}
                </div>
              </div>
              <div className="rounded-2xl border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.84)] px-4 py-3">
                <p className="text-xs uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Filter State</p>
                <div className="mt-2 flex flex-wrap gap-2">
                  {topic ? <span className="tone-chip">Topic: {topic}</span> : null}
                  {keyword ? <span className="tone-chip">Keyword: {keyword}</span> : null}
                  {q ? <span className="tone-chip">Search: {q}</span> : null}
                  {org ? <span className="tone-chip">Org: {org}</span> : null}
                  {source ? <span className="tone-chip">Source: {displaySourceKind(source)}</span> : null}
                  {status ? <span className="tone-chip">Status: {status}</span> : null}
                  {!topic && !keyword && !q && !org && !source && !status ? (
                    <span className="text-[color:var(--ink-faint)]">All corpus documents</span>
                  ) : null}
                </div>
              </div>
            </div>
          )}
        </article>
      </section>
    </div>
  );
}

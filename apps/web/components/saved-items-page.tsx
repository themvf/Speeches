"use client";

import Link from "next/link";
import { Fragment, useCallback, useEffect, useMemo, useState } from "react";
import { DEFAULT_LIST_ID, useSavedItems, type SavedItem, type SavedList } from "@/hooks/use-saved-items";

type SavedFilter = "all" | SavedItem["type"];
type ListFilter = "all" | string;
type JobAnalysisKind = "position" | "stance" | "summary" | "";

interface ApiEnvelope<T> {
  ok: boolean;
  data?: T;
  error?: string;
}

interface DocumentListItem {
  document_id: string;
  title: string;
  organization: string;
  source_kind: string;
  doc_type: string;
  speaker: string;
  url: string;
  date: string;
  published_at: string;
  word_count: number;
  keywords: string[];
  topics: string[];
  sentiment_label: "positive" | "negative" | "neutral" | "";
  sentiment_score: number;
}

interface DocumentsData {
  items: DocumentListItem[];
  page: number;
  page_size: number;
  total: number;
}

interface DocumentDetailData {
  metadata: {
    document_id: string;
    published_at: string;
  };
  content: {
    full_text: string;
    paragraphs: string[];
    sentences: string[];
  };
  enrichment: {
    status: string;
    summary: string;
    tags: string[];
    keywords: string[];
    entities: string[];
    evidence_spans: Array<Record<string, unknown>>;
    stance: Record<string, unknown>;
    comment_position: Record<string, unknown>;
    confidence: number;
  };
  review: {
    decision: string;
    notes: string;
    reviewed_at: string;
  };
  sentiment: {
    score: number;
    label: string;
    rationale: string;
    status: string;
  } | null;
}

const FILTERS: Array<{ id: SavedFilter; label: string }> = [
  { id: "all", label: "All" },
  { id: "article", label: "Articles" },
  { id: "doc", label: "Documents" },
];

const SOURCE_KIND_LABELS: Record<string, string> = {
  sec_speech: "SEC Speeches & Statements",
  sec_tm_faq: "SEC Trading & Markets FAQ",
  sec_enforcement_litigation: "SEC Enforcement Litigation",
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
  uploaded: "Uploaded",
};

const SOURCE_KIND_TYPE_LABELS: Record<string, string> = {
  sec_speech: "Speech",
  sec_tm_faq: "FAQ",
  sec_enforcement_litigation: "Litigation Release",
  finra_regulatory_notice: "Regulatory Notice",
  finra_comment_letter: "Comment Letter",
  finra_key_topic: "Key Topic",
  doj_usao_press_release: "Press Release",
  federal_reserve_speech_testimony: "Testimony",
  cftc_press_release: "Press Release",
  cftc_public_statement_remark: "Statement",
  jdsupra_article: "Article",
  investmentnews_article: "Article",
  citywire_article: "Article",
  congress_crs_product: "CRS Product",
  newsapi_article: "News Article",
  uploaded: "Uploaded Document",
};

function formatSavedAt(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Saved recently";
  }
  return date.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function fmt(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

function fmtDateOnly(value: string | undefined): string {
  if (!value) return "-";
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? value : d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function displayOrganization(value: string | undefined): string {
  const raw = String(value || "").trim();
  if (!raw) return "Unknown";
  const lowered = raw.toLowerCase();
  if (lowered === "financial news" || lowered === "financials news") {
    return "News";
  }
  return raw;
}

function displaySourceKind(value: string | undefined): string {
  const raw = String(value || "").trim();
  if (!raw) return "Unknown";
  return SOURCE_KIND_LABELS[raw] || raw.split("_").filter(Boolean).map((part) => part.charAt(0).toUpperCase() + part.slice(1)).join(" ");
}

function normalizeTypeLabel(value: string | undefined): string {
  const normalized = String(value || "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (!normalized) return "";
  return normalized.split(" ").map((part) => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase()).join(" ");
}

function displayType(docType: string | undefined, sourceKind: string | undefined): string {
  const normalized = normalizeTypeLabel(docType);
  if (normalized) return normalized;
  return SOURCE_KIND_TYPE_LABELS[String(sourceKind || "")] || "Document";
}

function typeClass(typeLabel: string): string {
  const t = typeLabel.toLowerCase();
  if (t.includes("litigation")) return "type-chip type-litigation";
  if (t.includes("regulatory notice")) return "type-chip type-regulatory";
  if (t.includes("speech") || t.includes("statement")) return "type-chip type-speech";
  if (t.includes("testimony")) return "type-chip type-testimony";
  if (t.includes("news")) return "type-chip type-news";
  if (t.includes("press release")) return "type-chip type-press";
  if (t.includes("faq")) return "type-chip type-faq";
  if (t.includes("key topic")) return "type-chip type-key-topic";
  return "type-chip type-default";
}

function statusClass(value: string): string {
  const s = String(value || "").toLowerCase();
  if (["enriched", "reviewed", "success"].includes(s)) return "status-chip status-success";
  if (["fallback_enriched", "queued", "running"].includes(s)) return "status-chip status-warn";
  if (["failed", "rejected"].includes(s)) return "status-chip status-failure";
  return "status-chip status-neutral";
}

function sentimentChipClass(label: string): string {
  if (label === "positive") return "type-chip type-chip--positive";
  if (label === "negative") return "type-chip type-chip--negative";
  return "type-chip type-default";
}

function sentimentLabel(label: string, score: number): string {
  const sign = score > 0 ? "+" : "";
  const pct = `${sign}${Math.round(score * 100)}`;
  if (label === "positive") return `Positive ${pct}`;
  if (label === "negative") return `Negative ${pct}`;
  return "Neutral";
}

function formatAnalysisLabel(value: string): string {
  const normalized = String(value || "").trim();
  if (!normalized) return "";
  return normalized.split("_").filter(Boolean).map((part) => part.charAt(0).toUpperCase() + part.slice(1)).join(" ");
}

function readStringField(value: unknown, key: string): string {
  if (!value || typeof value !== "object") return "";
  const out = (value as Record<string, unknown>)[key];
  return typeof out === "string" ? out.trim() : "";
}

function readNumberField(value: unknown, key: string): number {
  if (!value || typeof value !== "object") return 0;
  const out = Number.parseFloat(String((value as Record<string, unknown>)[key] ?? "0"));
  return Number.isFinite(out) ? out : 0;
}

function pickPrimaryAnalysis(detail: DocumentDetailData | null | undefined): {
  kind: JobAnalysisKind;
  label: string;
  tone: string;
  rationale: string;
  confidence: number;
} {
  if (!detail) {
    return { kind: "", label: "", tone: "", rationale: "", confidence: 0 };
  }

  const positionLabel = readStringField(detail.enrichment.comment_position, "label").toLowerCase();
  const positionRationale = readStringField(detail.enrichment.comment_position, "rationale");
  const positionConfidence = readNumberField(detail.enrichment.comment_position, "confidence");
  if (positionLabel && positionLabel !== "not_applicable" && positionLabel !== "unclear") {
    return {
      kind: "position",
      label: positionLabel,
      tone: positionLabel,
      rationale: positionRationale,
      confidence: Math.max(0, Math.min(1, positionConfidence)),
    };
  }

  const stanceLabel = readStringField(detail.enrichment.stance, "label").toLowerCase();
  const stanceTarget = readStringField(detail.enrichment.stance, "target");
  if (stanceLabel && stanceLabel !== "unclear" && stanceLabel !== "not_applicable") {
    return {
      kind: "stance",
      label: stanceTarget ? `${stanceLabel} (${stanceTarget})` : stanceLabel,
      tone: stanceLabel,
      rationale: "",
      confidence: Math.max(0, Math.min(1, Number(detail.enrichment.confidence || 0))),
    };
  }

  if (detail.enrichment.summary) {
    return { kind: "summary", label: "Summary", tone: "", rationale: "", confidence: 0 };
  }

  return { kind: "", label: "", tone: "", rationale: "", confidence: 0 };
}

function analysisChipClass(value: string): string {
  const label = String(value || "").toLowerCase();
  if (["supportive", "supports", "aligned", "favorable"].includes(label)) return "status-chip status-success";
  if (["opposed", "opposes", "critical", "negative", "adverse"].includes(label)) return "status-chip status-failure";
  if (["mixed", "qualified", "partially_supportive"].includes(label)) return "status-chip status-warn";
  return "status-chip status-neutral";
}

function renderToneChips(items: string[], emptyLabel: string) {
  if (!items.length) {
    return <span className="text-xs text-[color:var(--ink-faint)]">{emptyLabel}</span>;
  }
  return (
    <div className="flex flex-wrap gap-2">
      {items.map((item) => (
        <span key={item} className="tone-chip">
          {item}
        </span>
      ))}
    </div>
  );
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, { ...init, cache: "no-store" });
  const envelope = (await res.json()) as ApiEnvelope<T>;
  if (!res.ok || !envelope.ok || !envelope.data) {
    throw new Error(envelope.error || `Request failed with ${res.status}`);
  }
  return envelope.data;
}

function savedDocumentId(item: SavedItem): string {
  if (item.metadata?.documentId) {
    return item.metadata.documentId;
  }
  return item.id.startsWith("doc:") ? item.id.slice(4) : "";
}

function documentMetadataFromListItem(item: DocumentListItem): NonNullable<SavedItem["metadata"]> {
  return {
    documentId: item.document_id,
    organization: item.organization,
    sourceKind: item.source_kind,
    docType: item.doc_type,
    speaker: item.speaker,
    date: item.date,
    publishedAt: item.published_at,
    wordCount: item.word_count,
    keywords: item.keywords || [],
    topics: item.topics || [],
    sentimentLabel: item.sentiment_label,
    sentimentScore: item.sentiment_score,
  };
}

function matchesSearch(item: SavedItem, query: string, listNames: string[]): boolean {
  if (!query) return true;
  const metadata = item.metadata;
  return [
    item.title,
    item.source,
    item.topic,
    item.url,
    metadata?.organization,
    metadata?.sourceKind,
    metadata?.docType,
    metadata?.speaker,
    ...(metadata?.keywords || []),
    ...(metadata?.topics || []),
    ...listNames,
  ].filter(Boolean).join(" ").toLowerCase().includes(query);
}

function sortBySavedAt(items: SavedItem[]): SavedItem[] {
  return [...items].sort((a, b) => {
    const left = new Date(a.savedAt).getTime() || 0;
    const right = new Date(b.savedAt).getTime() || 0;
    return right - left;
  });
}

function itemListNames(item: SavedItem, listById: Map<string, SavedList>): string[] {
  return item.listIds.map((id) => listById.get(id)?.name).filter((name): name is string => Boolean(name));
}

function ListPicker({
  item,
  lists,
  setItemLists,
}: {
  item: SavedItem;
  lists: SavedList[];
  setItemLists: (id: string, listIds: string[]) => void;
}) {
  const toggleList = (listId: string) => {
    const current = item.listIds.includes(listId);
    const next = current ? item.listIds.filter((id) => id !== listId) : [...item.listIds, listId];
    setItemLists(item.id, next.length ? next : [DEFAULT_LIST_ID]);
  };

  return (
    <details className="relative">
      <summary className="btn-muted inline-flex cursor-pointer list-none px-3 py-1.5 text-xs">
        Lists
      </summary>
      <div className="absolute right-0 z-20 mt-2 w-56 rounded-xl border border-[color:var(--line)] bg-[color:rgba(6,15,24,0.98)] p-3 shadow-2xl">
        <div className="grid gap-2">
          {lists.map((list) => (
            <label key={list.id} className="flex items-center gap-2 text-xs text-[color:var(--ink-soft)]">
              <input
                type="checkbox"
                checked={item.listIds.includes(list.id)}
                onChange={() => toggleList(list.id)}
              />
              <span>{list.name}</span>
            </label>
          ))}
        </div>
      </div>
    </details>
  );
}

function AnalysisPanel({
  detail,
  loading,
  error,
  retry,
}: {
  detail: DocumentDetailData | undefined;
  loading: boolean;
  error: string;
  retry: () => void;
}) {
  const primaryAnalysis = pickPrimaryAnalysis(detail);

  if (loading) {
    return <p className="text-sm text-[color:var(--ink-soft)]">Loading analysis...</p>;
  }

  if (error) {
    return (
      <div className="flex flex-wrap items-center gap-3">
        <p className="text-sm text-[color:var(--danger)]">{error}</p>
        <button type="button" className="link-inline text-xs" onClick={retry}>
          Retry
        </button>
      </div>
    );
  }

  if (!detail) {
    return <p className="text-sm text-[color:var(--ink-faint)]">No analysis is available for this document.</p>;
  }

  return (
    <div className="grid gap-3 lg:grid-cols-[1.45fr_0.55fr]">
      <div>
        <div className="flex flex-wrap gap-2">
          <span className={statusClass(detail.enrichment.status)}>{detail.enrichment.status || "not_enriched"}</span>
          <span className="tone-chip">Review: {detail.review.decision || "pending"}</span>
          {primaryAnalysis.kind === "position" ? (
            <span className={analysisChipClass(primaryAnalysis.tone)}>
              Position: {formatAnalysisLabel(primaryAnalysis.label)}
            </span>
          ) : primaryAnalysis.kind === "stance" ? (
            <span className={analysisChipClass(primaryAnalysis.tone)}>
              Stance: {formatAnalysisLabel(primaryAnalysis.label)}
            </span>
          ) : null}
          {primaryAnalysis.confidence > 0 ? (
            <span className="tone-chip">Confidence: {Math.round(primaryAnalysis.confidence * 100)}%</span>
          ) : null}
        </div>
        <p className="mt-2 text-left text-sm leading-6 text-[color:var(--ink-soft)]">
          {detail.enrichment.summary || "No summary is available for this document yet."}
        </p>
        {primaryAnalysis.rationale ? (
          <p className="mt-2 text-left text-xs leading-5 text-[color:var(--ink-faint)]">{primaryAnalysis.rationale}</p>
        ) : null}
      </div>
      <div className="grid gap-2">
        <div>
          <p className="mb-1.5 text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">Tags</p>
          {renderToneChips(detail.enrichment.tags, "No tags yet")}
        </div>
        <div>
          <p className="mb-1.5 text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">Keywords</p>
          {renderToneChips(detail.enrichment.keywords, "No keywords yet")}
        </div>
      </div>
    </div>
  );
}

export function SavedItemsPage() {
  const { items, lists, listById, loaded, remove, clear, updateItem, setItemLists, createList } = useSavedItems();
  const [filter, setFilter] = useState<SavedFilter>("all");
  const [selectedListId, setSelectedListId] = useState<ListFilter>("all");
  const [query, setQuery] = useState("");
  const [newListName, setNewListName] = useState("");
  const [expandedDocs, setExpandedDocs] = useState<Record<string, boolean>>({});
  const [docDetails, setDocDetails] = useState<Record<string, DocumentDetailData>>({});
  const [docDetailLoading, setDocDetailLoading] = useState<Record<string, boolean>>({});
  const [docDetailError, setDocDetailError] = useState<Record<string, string>>({});

  const normalizedQuery = query.trim().toLowerCase();
  const counts = useMemo(() => ({
    all: items.length,
    article: items.filter((item) => item.type === "article").length,
    doc: items.filter((item) => item.type === "doc").length,
  }), [items]);

  const listCounts = useMemo(() => {
    const out = new Map<string, number>();
    lists.forEach((list) => out.set(list.id, 0));
    items.forEach((item) => {
      item.listIds.forEach((listId) => out.set(listId, (out.get(listId) || 0) + 1));
    });
    return out;
  }, [items, lists]);

  const visibleItems = useMemo(() => {
    const filtered = items.filter((item) => {
      const typeMatches = filter === "all" || item.type === filter;
      const listMatches = selectedListId === "all" || item.listIds.includes(selectedListId);
      return typeMatches && listMatches && matchesSearch(item, normalizedQuery, itemListNames(item, listById));
    });
    return sortBySavedAt(filtered);
  }, [filter, items, listById, normalizedQuery, selectedListId]);

  const visibleDocs = visibleItems.filter((item) => item.type === "doc");
  const visibleArticles = visibleItems.filter((item) => item.type === "article");

  useEffect(() => {
    if (!loaded) return;
    const missing = items.filter((item) => item.type === "doc" && savedDocumentId(item) && (!item.metadata?.sourceKind || !item.metadata?.docType));
    if (!missing.length) return;

    const controller = new AbortController();
    const ids = Array.from(new Set(missing.map(savedDocumentId).filter(Boolean)));
    void fetchJson<DocumentsData>(`/api/documents?doc_ids=${encodeURIComponent(ids.join(","))}&page_size=${ids.length}`, {
      signal: controller.signal,
    }).then((data) => {
      data.items.forEach((doc) => {
        const savedItem = missing.find((item) => savedDocumentId(item) === doc.document_id);
        if (!savedItem) return;
        const primaryTopic = (doc.topics || [])[0] || (doc.keywords || [])[0];
        updateItem(savedItem.id, {
          title: doc.title || savedItem.title,
          url: doc.url || savedItem.url,
          source: displayOrganization(doc.organization),
          topic: primaryTopic,
          metadata: documentMetadataFromListItem(doc),
        });
      });
    }).catch(() => {
      // Keep the older saved record usable even if hydration fails.
    });

    return () => controller.abort();
  }, [items, loaded, updateItem]);

  const loadDocDetail = useCallback(async (item: SavedItem) => {
    const docId = savedDocumentId(item);
    if (!docId || docDetails[docId] || docDetailLoading[docId]) return;

    setDocDetailLoading((prev) => ({ ...prev, [docId]: true }));
    setDocDetailError((prev) => ({ ...prev, [docId]: "" }));
    try {
      const detail = await fetchJson<DocumentDetailData>(`/api/documents/${encodeURIComponent(docId)}`);
      setDocDetails((prev) => ({ ...prev, [docId]: detail }));
    } catch (err) {
      setDocDetailError((prev) => ({
        ...prev,
        [docId]: err instanceof Error ? err.message : "Failed to load analysis.",
      }));
    } finally {
      setDocDetailLoading((prev) => ({ ...prev, [docId]: false }));
    }
  }, [docDetailLoading, docDetails]);

  const toggleDocAnalysis = useCallback((item: SavedItem) => {
    const docId = savedDocumentId(item);
    if (!docId) return;

    setExpandedDocs((prev) => {
      const nextValue = !prev[docId];
      return { ...prev, [docId]: nextValue };
    });

    if (!docDetails[docId] && !docDetailLoading[docId]) {
      void loadDocDetail(item);
    }
  }, [docDetailLoading, docDetails, loadDocDetail]);

  const clearSavedItems = () => {
    if (window.confirm("Remove all saved items?")) {
      clear();
    }
  };

  const addList = () => {
    const list = createList(newListName);
    if (list) {
      setNewListName("");
    }
  };

  return (
    <div className="mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-6 px-4 py-6 md:px-8 md:py-10">
      <header className="panel hero-panel reveal p-6 md:p-8">
        <span className="kicker">Saved</span>
        <h1 className="mt-3 text-3xl font-bold leading-tight md:text-5xl">Saved Research</h1>
        <p className="mt-3 max-w-3xl text-base text-[color:var(--ink-soft)] md:text-lg">
          {loaded ? `${counts.all} saved item${counts.all === 1 ? "" : "s"}` : "Loading saved items"}
        </p>
      </header>

      <section className="grid gap-4 lg:grid-cols-[280px_minmax(0,1fr)]">
        <aside className="panel h-fit p-4">
          <p className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">Library</p>
          <div className="mt-3 grid gap-2">
            {FILTERS.map((item) => {
              const active = filter === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  aria-pressed={active}
                  onClick={() => setFilter(item.id)}
                  className={`flex items-center justify-between rounded-lg border px-3 py-2 text-left text-sm font-semibold transition ${
                    active
                      ? "border-[color:var(--accent)] bg-[color:rgba(79,213,255,0.14)] text-[color:var(--ink)]"
                      : "border-[color:var(--line)] bg-[color:rgba(9,22,36,0.5)] text-[color:var(--ink-soft)] hover:border-[color:var(--line-strong)] hover:text-[color:var(--ink)]"
                  }`}
                >
                  <span>{item.label}</span>
                  <span className="text-xs text-[color:var(--ink-faint)]">{counts[item.id]}</span>
                </button>
              );
            })}
          </div>

          <div className="mt-5">
            <p className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">Lists</p>
            <div className="mt-3 grid gap-2">
              <button
                type="button"
                aria-pressed={selectedListId === "all"}
                onClick={() => setSelectedListId("all")}
                className={`flex items-center justify-between rounded-lg border px-3 py-2 text-left text-sm font-semibold transition ${
                  selectedListId === "all"
                    ? "border-[color:var(--accent)] bg-[color:rgba(79,213,255,0.14)] text-[color:var(--ink)]"
                    : "border-[color:var(--line)] bg-[color:rgba(9,22,36,0.5)] text-[color:var(--ink-soft)] hover:border-[color:var(--line-strong)] hover:text-[color:var(--ink)]"
                }`}
              >
                <span>All Lists</span>
                <span className="text-xs text-[color:var(--ink-faint)]">{items.length}</span>
              </button>
              {lists.map((list) => {
                const active = selectedListId === list.id;
                return (
                  <button
                    key={list.id}
                    type="button"
                    aria-pressed={active}
                    onClick={() => setSelectedListId(list.id)}
                    className={`flex items-center justify-between rounded-lg border px-3 py-2 text-left text-sm font-semibold transition ${
                      active
                        ? "border-[color:var(--accent)] bg-[color:rgba(79,213,255,0.14)] text-[color:var(--ink)]"
                        : "border-[color:var(--line)] bg-[color:rgba(9,22,36,0.5)] text-[color:var(--ink-soft)] hover:border-[color:var(--line-strong)] hover:text-[color:var(--ink)]"
                    }`}
                  >
                    <span>{list.name}</span>
                    <span className="text-xs text-[color:var(--ink-faint)]">{listCounts.get(list.id) || 0}</span>
                  </button>
                );
              })}
            </div>
            <div className="mt-3 flex gap-2">
              <input
                value={newListName}
                onChange={(event) => setNewListName(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    addList();
                  }
                }}
                placeholder="New list"
                className="form-control min-w-0 flex-1 px-3 py-2 text-sm"
              />
              <button type="button" onClick={addList} className="btn-solid px-3 py-2 text-sm">
                Add
              </button>
            </div>
          </div>

          {items.length > 0 ? (
            <button type="button" onClick={clearSavedItems} className="btn-muted mt-4 w-full px-3 py-2 text-sm">
              Clear Saved
            </button>
          ) : null}
        </aside>

        <section className="panel overflow-hidden">
          <div className="flex flex-col gap-3 border-b border-[color:var(--line)] p-4 md:flex-row md:items-center md:justify-between">
            <div>
              <h2 className="text-xl font-semibold">Items</h2>
              <p className="mt-1 text-sm text-[color:var(--ink-faint)]">{visibleItems.length} shown</p>
            </div>
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search saved items"
              className="form-control w-full px-3 py-2 text-sm md:w-72"
            />
          </div>

          {!loaded ? (
            <div className="p-5 text-sm text-[color:var(--ink-faint)]">Loading saved items...</div>
          ) : visibleItems.length === 0 ? (
            <div className="p-6">
              <p className="text-sm text-[color:var(--ink-soft)]">
                {items.length === 0 ? "No saved items yet." : "No saved items match the current filters."}
              </p>
              {items.length === 0 ? (
                <Link href="/" className="btn-solid mt-4 inline-flex px-4 py-2 text-sm">
                  Open Home
                </Link>
              ) : null}
            </div>
          ) : (
            <div className="grid gap-5 p-4">
              {visibleDocs.length > 0 ? (
                <div className="feed-table-wrap">
                  <table className="feed-table">
                    <thead>
                      <tr>
                        <th>Title</th>
                        <th>Source</th>
                        <th>Type</th>
                        <th>Speaker</th>
                        <th>Keywords</th>
                        <th>Date</th>
                        <th>Lists</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {visibleDocs.map((item) => {
                        const docId = savedDocumentId(item);
                        const metadata = item.metadata || {};
                        const typeLabel = displayType(metadata.docType, metadata.sourceKind);
                        const isExpanded = !!expandedDocs[docId];
                        const detail = docDetails[docId];
                        const detailLoading = !!docDetailLoading[docId];
                        const detailError = docDetailError[docId] || "";
                        const analysisActionLabel = detailLoading ? "Loading Analysis..." : isExpanded ? "Hide Analysis" : "Open Analysis";
                        const keywords = (metadata.keywords || []).slice(0, 4).join(", ") || (metadata.topics || []).slice(0, 4).join(", ") || item.topic || "-";
                        const listNames = itemListNames(item, listById);

                        return (
                          <Fragment key={item.id}>
                            <tr>
                              <td>
                                <p className="feed-title">{item.title || "Untitled"}</p>
                                <p className="mt-1 text-xs text-[color:var(--ink-faint)]">Saved {formatSavedAt(item.savedAt)}</p>
                                <div className="mt-2 flex flex-wrap items-center gap-2">
                                  {item.url ? (
                                    <a href={item.url} target="_blank" rel="noreferrer" className="link-inline inline-block text-xs">
                                      Open source
                                    </a>
                                  ) : null}
                                  {docId ? (
                                    <button
                                      type="button"
                                      className={`rounded-full border px-3 py-1 text-xs font-semibold tracking-[0.04em] transition ${
                                        isExpanded
                                          ? "border-[color:var(--accent)] bg-[color:rgba(79,213,255,0.12)] text-[color:var(--ink)]"
                                          : "border-[color:var(--line)] text-[color:var(--ink-soft)] hover:border-[color:var(--accent)] hover:text-[color:var(--ink)]"
                                      }`}
                                      onClick={() => toggleDocAnalysis(item)}
                                    >
                                      {analysisActionLabel}
                                    </button>
                                  ) : null}
                                </div>
                              </td>
                              <td className="text-xs">
                                <span className="tone-chip">{displayOrganization(metadata.organization || item.source)}</span>
                                <p className="feed-subtle mt-2">{metadata.wordCount ? `${fmt(metadata.wordCount)} words` : displaySourceKind(metadata.sourceKind)}</p>
                              </td>
                              <td className="text-xs">
                                <span className={typeClass(typeLabel)}>{typeLabel}</span>
                                {metadata.sentimentLabel ? (
                                  <p className="mt-1.5">
                                    <span
                                      className={sentimentChipClass(metadata.sentimentLabel)}
                                      title={`Tone score: ${(metadata.sentimentScore || 0) > 0 ? "+" : ""}${Number(metadata.sentimentScore || 0).toFixed(2)}`}
                                    >
                                      {sentimentLabel(metadata.sentimentLabel, Number(metadata.sentimentScore || 0))}
                                    </span>
                                  </p>
                                ) : null}
                              </td>
                              <td className="text-xs">{metadata.speaker || "-"}</td>
                              <td className="text-xs">{keywords}</td>
                              <td className="text-xs">{fmtDateOnly(metadata.publishedAt || metadata.date)}</td>
                              <td className="text-xs">
                                <div className="mb-2 flex flex-wrap gap-1">
                                  {listNames.map((name) => <span key={name} className="tone-chip">{name}</span>)}
                                </div>
                                <ListPicker item={item} lists={lists} setItemLists={setItemLists} />
                              </td>
                              <td>
                                <button type="button" onClick={() => remove(item.id)} className="btn-muted px-3 py-1.5 text-xs">
                                  Remove
                                </button>
                              </td>
                            </tr>
                            {isExpanded ? (
                              <tr>
                                <td colSpan={8} className="bg-[color:rgba(8,18,30,0.82)] px-4 py-3">
                                  <AnalysisPanel
                                    detail={detail}
                                    loading={detailLoading}
                                    error={detailError}
                                    retry={() => void loadDocDetail(item)}
                                  />
                                </td>
                              </tr>
                            ) : null}
                          </Fragment>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              ) : null}

              {visibleArticles.length > 0 ? (
                <div className="grid gap-3">
                  {visibleArticles.map((item) => {
                    const listNames = itemListNames(item, listById);
                    return (
                      <article key={item.id} className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(8,18,30,0.74)] p-4">
                        <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                          <div className="min-w-0">
                            <div className="flex flex-wrap items-center gap-2">
                              <span className="type-chip type-news">Article</span>
                              <span className="tone-chip">{item.source || "Unknown source"}</span>
                              {item.topic ? <span className="tone-chip">{item.topic}</span> : null}
                              {listNames.map((name) => <span key={name} className="tone-chip">{name}</span>)}
                            </div>
                            <h3 className="mt-2 text-lg font-semibold leading-snug text-[color:var(--ink)]">{item.title || "Untitled"}</h3>
                            <p className="mt-1 text-xs text-[color:var(--ink-faint)]">Saved {formatSavedAt(item.savedAt)}</p>
                          </div>
                          <div className="flex flex-wrap items-center gap-2 md:justify-end">
                            <ListPicker item={item} lists={lists} setItemLists={setItemLists} />
                            {item.url ? (
                              <a href={item.url} target="_blank" rel="noreferrer" className="btn-solid px-3 py-2 text-sm">
                                Open Source
                              </a>
                            ) : null}
                            <button type="button" onClick={() => remove(item.id)} className="btn-muted px-3 py-2 text-sm">
                              Remove
                            </button>
                          </div>
                        </div>
                      </article>
                    );
                  })}
                </div>
              ) : null}
            </div>
          )}
        </section>
      </section>
    </div>
  );
}

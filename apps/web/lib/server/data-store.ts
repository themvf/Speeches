import fs from "node:fs";
import path from "node:path";

import { downloadGcsJson } from "@/lib/server/gcs-loader";
import { getDataSourceConfig } from "@/lib/server/env";
import {
  type CustomDocumentRecord,
  type CustomDocumentsPayload,
  type DocumentListItem,
  type EnrichmentEntry,
  type EnrichmentStatePayload,
  type NewsConnectorSettingsPayload
} from "@/lib/server/types";

const CUSTOM_DOCS_BLOB = "custom_documents.json";
const ENRICHMENT_BLOB = "document_enrichment_state.json";
const SETTINGS_BLOB = "news_connector_settings.json";

const CACHE_TTL_MS = 15_000;

type CacheEntry<T> = {
  loadedAt: number;
  data: T;
};

const cache = new Map<string, CacheEntry<unknown>>();

function normalizeString(value: unknown): string {
  return String(value ?? "").trim();
}

function normalizeWordCount(value: unknown): number {
  const n = Number.parseInt(String(value ?? ""), 10);
  return Number.isFinite(n) && n >= 0 ? n : 0;
}

function normalizeCustomDocument(record: unknown): CustomDocumentRecord | null {
  if (!record || typeof record !== "object") {
    return null;
  }

  const src = record as Record<string, unknown>;
  const metadataRaw = src.metadata && typeof src.metadata === "object" ? (src.metadata as Record<string, unknown>) : {};
  const contentRaw = src.content && typeof src.content === "object" ? (src.content as Record<string, unknown>) : {};

  const metadata = {
    document_id: normalizeString(metadataRaw.document_id),
    title: normalizeString(metadataRaw.title),
    speaker: normalizeString(metadataRaw.speaker),
    date: normalizeString(metadataRaw.date),
    url: normalizeString(metadataRaw.url),
    word_count: normalizeWordCount(metadataRaw.word_count),
    organization: normalizeString(metadataRaw.organization),
    doc_type: normalizeString(metadataRaw.doc_type),
    source_filename: normalizeString(metadataRaw.source_filename),
    source_format: normalizeString(metadataRaw.source_format),
    source_local_path: normalizeString(metadataRaw.source_local_path),
    source_gcs_path: normalizeString(metadataRaw.source_gcs_path),
    tags: normalizeString(metadataRaw.tags),
    source_kind: normalizeString(metadataRaw.source_kind),
    source_family: normalizeString(metadataRaw.source_family),
    source_index_url: normalizeString(metadataRaw.source_index_url),
    published_date: normalizeString(metadataRaw.published_date),
    updated_date: normalizeString(metadataRaw.updated_date),
    last_reviewed_or_updated: normalizeString(metadataRaw.last_reviewed_or_updated),
    ...metadataRaw
  };

  const paragraphs = Array.isArray(contentRaw.paragraphs)
    ? contentRaw.paragraphs.map((item) => normalizeString(item)).filter(Boolean)
    : [];
  const sentences = Array.isArray(contentRaw.sentences)
    ? contentRaw.sentences.map((item) => normalizeString(item)).filter(Boolean)
    : [];

  return {
    metadata,
    content: {
      full_text: normalizeString(contentRaw.full_text),
      paragraphs,
      sentences,
      ...contentRaw
    },
    validation: src.validation && typeof src.validation === "object" ? (src.validation as Record<string, unknown>) : {}
  } as CustomDocumentRecord;
}

function normalizeCustomDocumentsPayload(payload: unknown): CustomDocumentsPayload {
  if (!payload || typeof payload !== "object") {
    return { updated_at: "", documents: [] };
  }

  const src = payload as Record<string, unknown>;
  const docsRaw = Array.isArray(src.documents) ? src.documents : [];
  const documents = docsRaw.map((item) => normalizeCustomDocument(item)).filter(Boolean) as CustomDocumentRecord[];

  return {
    updated_at: normalizeString(src.updated_at),
    documents
  };
}

function normalizeEnrichmentEntry(docId: string, value: unknown): EnrichmentEntry {
  const src = value && typeof value === "object" ? (value as Record<string, unknown>) : {};
  const enrichmentRaw = src.enrichment && typeof src.enrichment === "object" ? (src.enrichment as Record<string, unknown>) : {};
  const reviewRaw = src.review && typeof src.review === "object" ? (src.review as Record<string, unknown>) : {};

  return {
    doc_id: normalizeString(src.doc_id || docId),
    organization: normalizeString(src.organization),
    org_key: normalizeString(src.org_key),
    title: normalizeString(src.title),
    speaker: normalizeString(src.speaker),
    date: normalizeString(src.date),
    url: normalizeString(src.url),
    doc_type: normalizeString(src.doc_type),
    word_count: normalizeWordCount(src.word_count),
    status: normalizeString(src.status),
    error: normalizeString(src.error),
    model: normalizeString(src.model),
    pipeline_version: normalizeString(src.pipeline_version),
    updated_at: normalizeString(src.updated_at),
    enrichment: {
      summary: normalizeString(enrichmentRaw.summary),
      tags: Array.isArray(enrichmentRaw.tags) ? enrichmentRaw.tags.map((item) => normalizeString(item)).filter(Boolean) : [],
      keywords: Array.isArray(enrichmentRaw.keywords)
        ? enrichmentRaw.keywords.map((item) => normalizeString(item)).filter(Boolean)
        : [],
      entities: Array.isArray(enrichmentRaw.entities)
        ? enrichmentRaw.entities.map((item) => normalizeString(item)).filter(Boolean)
        : [],
      stance: enrichmentRaw.stance && typeof enrichmentRaw.stance === "object" ? enrichmentRaw.stance : {},
      evidence_spans: Array.isArray(enrichmentRaw.evidence_spans)
        ? enrichmentRaw.evidence_spans.filter((item) => item && typeof item === "object")
        : [],
      confidence: Number.parseFloat(String(enrichmentRaw.confidence ?? "0")) || 0,
      ...enrichmentRaw
    },
    review: {
      decision: normalizeString(reviewRaw.decision),
      notes: normalizeString(reviewRaw.notes),
      reviewed_at: normalizeString(reviewRaw.reviewed_at),
      ...reviewRaw
    },
    reward: src.reward && typeof src.reward === "object" ? (src.reward as Record<string, unknown>) : {},
    auto_review: src.auto_review && typeof src.auto_review === "object" ? (src.auto_review as Record<string, unknown>) : {},
    ...src
  } as EnrichmentEntry;
}

function normalizeEnrichmentStatePayload(payload: unknown): EnrichmentStatePayload {
  if (!payload || typeof payload !== "object") {
    return { version: 1, pipeline_version: "v1", updated_at: "", entries: {} };
  }

  const src = payload as Record<string, unknown>;
  const entriesRaw = src.entries && typeof src.entries === "object" ? (src.entries as Record<string, unknown>) : {};
  const entries: Record<string, EnrichmentEntry> = {};

  for (const [docId, value] of Object.entries(entriesRaw)) {
    entries[docId] = normalizeEnrichmentEntry(docId, value);
  }

  return {
    version: Number.parseInt(String(src.version ?? "1"), 10) || 1,
    pipeline_version: normalizeString(src.pipeline_version || "v1"),
    updated_at: normalizeString(src.updated_at),
    entries
  };
}

function normalizeNewsSettingsPayload(payload: unknown): NewsConnectorSettingsPayload {
  if (!payload || typeof payload !== "object") {
    return {
      updated_at: "",
      query: "",
      lookback_days: 7,
      max_pages: 4,
      page_size: 50,
      target_count: 100,
      sort_by: "publishedAt",
      organization_label: "Financial News",
      domains: "",
      exclude_domains: "",
      tags_csv: ""
    };
  }

  const src = payload as Record<string, unknown>;

  return {
    updated_at: normalizeString(src.updated_at),
    query: normalizeString(src.query),
    lookback_days: Number.parseInt(String(src.lookback_days ?? "7"), 10) || 7,
    max_pages: Number.parseInt(String(src.max_pages ?? "4"), 10) || 4,
    page_size: Number.parseInt(String(src.page_size ?? "50"), 10) || 50,
    target_count: Number.parseInt(String(src.target_count ?? "100"), 10) || 100,
    sort_by: normalizeString(src.sort_by || "publishedAt"),
    organization_label: normalizeString(src.organization_label || "Financial News"),
    domains: normalizeString(src.domains),
    exclude_domains: normalizeString(src.exclude_domains),
    tags_csv: normalizeString(src.tags_csv)
  };
}

function findProjectRootWithData(startDir: string): string {
  let current = path.resolve(startDir);
  for (let i = 0; i < 7; i += 1) {
    const candidate = path.join(current, "data", CUSTOM_DOCS_BLOB);
    if (fs.existsSync(candidate)) {
      return current;
    }
    const parent = path.dirname(current);
    if (parent === current) {
      break;
    }
    current = parent;
  }
  return path.resolve(startDir);
}

function resolveDataDirPath(): string {
  const cfg = getDataSourceConfig();
  if (cfg.dataDirPath) {
    return path.isAbsolute(cfg.dataDirPath) ? cfg.dataDirPath : path.resolve(process.cwd(), cfg.dataDirPath);
  }
  const root = findProjectRootWithData(process.cwd());
  return path.join(root, "data");
}

function readLocalJson(fileName: string): unknown | null {
  const dataDir = resolveDataDirPath();
  const filePath = path.join(dataDir, fileName);
  if (!fs.existsSync(filePath)) {
    return null;
  }
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf-8"));
  } catch {
    return null;
  }
}

async function loadFromSource<T>(
  cacheKey: string,
  blobName: string,
  normalize: (payload: unknown) => T,
  emptyFactory: () => T
): Promise<T> {
  const now = Date.now();
  const hit = cache.get(cacheKey);
  if (hit && now - hit.loadedAt < CACHE_TTL_MS) {
    return hit.data as T;
  }

  const cfg = getDataSourceConfig();
  let raw: unknown | null = null;

  if (cfg.mode === "gcs" || cfg.mode === "auto") {
    raw = await downloadGcsJson<unknown>(blobName);
  }
  if (raw === null && (cfg.mode === "local" || cfg.mode === "auto")) {
    raw = readLocalJson(blobName);
  }

  const normalized = raw === null ? emptyFactory() : normalize(raw);
  cache.set(cacheKey, { loadedAt: now, data: normalized });
  return normalized;
}

export async function loadCustomDocuments(): Promise<CustomDocumentsPayload> {
  return loadFromSource(
    "custom_documents",
    CUSTOM_DOCS_BLOB,
    normalizeCustomDocumentsPayload,
    () => ({ updated_at: "", documents: [] })
  );
}

export async function loadEnrichmentState(): Promise<EnrichmentStatePayload> {
  return loadFromSource(
    "enrichment_state",
    ENRICHMENT_BLOB,
    normalizeEnrichmentStatePayload,
    () => ({ version: 1, pipeline_version: "v1", updated_at: "", entries: {} })
  );
}

export async function loadNewsConnectorSettings(): Promise<NewsConnectorSettingsPayload> {
  return loadFromSource(
    "news_connector_settings",
    SETTINGS_BLOB,
    normalizeNewsSettingsPayload,
    () => ({
      updated_at: "",
      query: "",
      lookback_days: 7,
      max_pages: 4,
      page_size: 50,
      target_count: 100,
      sort_by: "publishedAt",
      organization_label: "Financial News",
      domains: "",
      exclude_domains: "",
      tags_csv: ""
    })
  );
}

export function buildDocumentListItems(
  customDocs: CustomDocumentsPayload,
  enrichmentState: EnrichmentStatePayload
): DocumentListItem[] {
  const entries = enrichmentState.entries || {};

  return customDocs.documents.map((doc) => {
    const m = doc.metadata || ({} as CustomDocumentRecord["metadata"]);
    const docId = normalizeString(m.document_id);
    const enrich = entries[docId];
    const reviewDecision = normalizeString(enrich?.review?.decision || "pending") || "pending";

    return {
      document_id: docId,
      title: normalizeString(m.title),
      organization: normalizeString(m.organization),
      source_kind: normalizeString(m.source_kind),
      doc_type: normalizeString(m.doc_type),
      speaker: normalizeString(m.speaker),
      url: normalizeString(m.url),
      date: normalizeString(m.date),
      published_at: normalizeString(m.published_date) || normalizeString(m.date),
      word_count: normalizeWordCount(m.word_count),
      ingest_status: "existing",
      enrichment_status: normalizeString(enrich?.status || "not_enriched") || "not_enriched",
      review_decision: reviewDecision,
      updated_at:
        normalizeString(m.last_reviewed_or_updated) || normalizeString(m.updated_date) || normalizeString(enrich?.updated_at)
    };
  });
}

export function parseComparableDate(value: string): number {
  const parsed = new Date(value);
  const ms = parsed.getTime();
  return Number.isNaN(ms) ? 0 : ms;
}
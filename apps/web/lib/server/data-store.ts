import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";

import { getDataSourceConfig } from "@/lib/server/env";
import { downloadGcsJson, uploadGcsJson } from "@/lib/server/gcs-loader";
import {
  type CustomDocumentRecord,
  type CustomDocumentsPayload,
  type DocumentListItem,
  type DocumentsFacets,
  type EnrichmentEntry,
  type EnrichmentStatePayload,
  type NewsConnectorSettingsPayload
} from "@/lib/server/types";

const SEC_SPEECHES_GCS_BLOB = "all_speeches.json";
const SEC_SPEECHES_LOCAL_FILE = "all_speeches_final.json";
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

function splitCsv(value: string): string[] {
  return String(value || "")
    .split(",")
    .map((item) => normalizeString(item))
    .filter(Boolean);
}

function dedupList(items: string[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const item of items) {
    const key = normalizeString(item).toLowerCase();
    if (!key || seen.has(key)) {
      continue;
    }
    seen.add(key);
    out.push(normalizeString(item));
  }
  return out;
}

const TOPIC_ACRONYMS = new Set(["SEC", "DOJ", "FINRA", "CFTC", "FOMC", "FDIC", "OCC", "CFPB", "AML", "KYC", "ESG"]);

function canonicalFacetToken(value: string): string {
  return normalizeString(value)
    .toLowerCase()
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function formatFacetLabel(value: string): string {
  const normalized = normalizeString(value)
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (!normalized) {
    return "";
  }
  return normalized
    .split(" ")
    .map((word) => {
      const upper = word.toUpperCase();
      if (TOPIC_ACRONYMS.has(upper) || /^[A-Z]{2,}$/.test(word)) {
        return upper;
      }
      if (/^\d+$/.test(word)) {
        return word;
      }
      return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
    })
    .join(" ");
}

function normalizeOrgLabel(value: unknown): string {
  const label = normalizeString(value);
  if (!label) {
    return "SEC";
  }
  const lower = label.toLowerCase();
  if (lower === "financial news" || lower === "financials news") {
    return "News";
  }
  return label;
}

function normalizeNewsOrgLabel(value: unknown): string {
  const label = normalizeString(value);
  if (!label) {
    return "News";
  }
  const lower = label.toLowerCase();
  if (lower === "financial news" || lower === "financials news") {
    return "News";
  }
  return label;
}

function orgKeyFromLabel(label: string): string {
  const cleaned = String(label)
    .split("")
    .map((ch) => (/[a-z0-9]/i.test(ch) ? ch.toLowerCase() : "_"))
    .join("")
    .replace(/^_+|_+$/g, "");
  return cleaned || "sec";
}

function inferSourceKind(metadataRaw: Record<string, unknown>): string {
  const explicit = normalizeString(metadataRaw.source_kind).toLowerCase();
  if (explicit) {
    return explicit;
  }

  const url = normalizeString(metadataRaw.url).toLowerCase();
  const docType = normalizeString(metadataRaw.doc_type).toLowerCase();

  if (url.includes("/newsroom/speeches-statements/")) {
    return "sec_speech";
  }
  if (url.includes("/rules-regulations/public-comments/") || url.includes("/comments/")) {
    return "sec_rule_comment";
  }
  if (url.includes("/rules-regulations/") && docType.includes("release")) {
    return "sec_rule_release";
  }
  if (docType === "regulatory notice") {
    return "finra_regulatory_notice";
  }
  if (docType === "comment letter") {
    return "finra_comment_letter";
  }
  if (docType === "rulemaking document") {
    return "regulations_gov_rule";
  }
  if (docType === "public comment") {
    return "regulations_gov_comment";
  }
  if (docType === "key topic") {
    return "finra_key_topic";
  }
  if (url.includes("/trading-markets-frequently-asked-questions/")) {
    return "sec_tm_faq";
  }
  if (url.includes("/enforcement-litigation/litigation-releases/")) {
    return "sec_enforcement_litigation";
  }
  if ((url.includes("/usao-") || url.includes("/usao/")) && url.includes("/pr/")) {
    return "doj_usao_press_release";
  }
  if (url.includes("/pressroom/pressreleases/")) {
    return "cftc_press_release";
  }
  if (url.includes("/pressroom/speechestestimony/")) {
    return "cftc_public_statement_remark";
  }
  if (url.includes("/crs-product/")) {
    return "congress_crs_product";
  }
  if (["speech", "statement", "remarks"].includes(docType)) {
    return "sec_speech";
  }
  return "document";
}

function corpusDocId(record: Record<string, unknown>, fullText: string): string {
  const existing = normalizeString(record.document_id);
  if (existing) {
    return existing;
  }

  const orgLabel = normalizeOrgLabel(record.organization || record.org || "SEC");
  const stable = [
    orgKeyFromLabel(orgLabel),
    normalizeString(record.url),
    normalizeString(record.title),
    normalizeString(record.speaker),
    normalizeString(record.date)
  ].join("|");

  const key = stable.replace(/\|/g, "").trim() ? stable : fullText.slice(0, 1000);
  return createHash("sha256").update(key).digest("hex").slice(0, 24);
}

function normalizeCustomDocument(record: unknown): CustomDocumentRecord | null {
  if (!record || typeof record !== "object") {
    return null;
  }

  const src = record as Record<string, unknown>;
  const metadataRaw = src.metadata && typeof src.metadata === "object" ? (src.metadata as Record<string, unknown>) : {};
  const contentRaw = src.content && typeof src.content === "object" ? (src.content as Record<string, unknown>) : {};

  const paragraphs = Array.isArray(contentRaw.paragraphs)
    ? contentRaw.paragraphs.map((item) => normalizeString(item)).filter(Boolean)
    : [];
  const sentences = Array.isArray(contentRaw.sentences)
    ? contentRaw.sentences.map((item) => normalizeString(item)).filter(Boolean)
    : [];
  const fullText = normalizeString(contentRaw.full_text);

  const metadata = {
    document_id: normalizeString(metadataRaw.document_id) || corpusDocId(metadataRaw, fullText),
    title: normalizeString(metadataRaw.title),
    speaker: normalizeString(metadataRaw.speaker),
    date: normalizeString(metadataRaw.date),
    url: normalizeString(metadataRaw.url),
    word_count: normalizeWordCount(metadataRaw.word_count),
    organization: normalizeOrgLabel(metadataRaw.organization),
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
    notice_type: normalizeString(metadataRaw.notice_type),
    notice_number: normalizeString(metadataRaw.notice_number),
    notice_title: normalizeString(metadataRaw.notice_title),
    notice_url: normalizeString(metadataRaw.notice_url),
    file_number: normalizeString(metadataRaw.file_number),
    release_numbers: Array.isArray(metadataRaw.release_numbers)
      ? metadataRaw.release_numbers.map((item) => normalizeString(item)).filter(Boolean)
      : [],
    rule_type: normalizeString(metadataRaw.rule_type),
    sec_issue_date: normalizeString(metadataRaw.sec_issue_date),
    federal_register_publish_date: normalizeString(metadataRaw.federal_register_publish_date),
    source_notice_url: normalizeString(metadataRaw.source_notice_url),
    comment_url: normalizeString(metadataRaw.comment_url),
    comments_url: normalizeString(metadataRaw.comments_url),
    commenter_name: normalizeString(metadataRaw.commenter_name),
    commenter_org: normalizeString(metadataRaw.commenter_org),
    letter_type: normalizeString(metadataRaw.letter_type),
    effective_date: normalizeString(metadataRaw.effective_date),
    comment_deadline: normalizeString(metadataRaw.comment_deadline),
    pdf_url: normalizeString(metadataRaw.pdf_url),
    discovery_source: normalizeString(metadataRaw.discovery_source),
    input_url: normalizeString(metadataRaw.input_url),
    docket_id: normalizeString(metadataRaw.docket_id),
    docket_url: normalizeString(metadataRaw.docket_url),
    document_url: normalizeString(metadataRaw.document_url),
    rule_url: normalizeString(metadataRaw.rule_url),
    comment_id: normalizeString(metadataRaw.comment_id),
    comment_page_url: normalizeString(metadataRaw.comment_page_url),
    resolved_content_url: normalizeString(metadataRaw.resolved_content_url),
    attachment_urls: Array.isArray(metadataRaw.attachment_urls)
      ? metadataRaw.attachment_urls.map((item) => normalizeString(item)).filter(Boolean)
      : [],
    extraction_mode: normalizeString(metadataRaw.extraction_mode),
    extraction_warnings: Array.isArray(metadataRaw.extraction_warnings)
      ? metadataRaw.extraction_warnings.map((item) => normalizeString(item)).filter(Boolean)
      : [],
    summary: normalizeString(metadataRaw.summary)
  };

  return {
    metadata,
    content: {
      full_text: fullText,
      paragraphs,
      sentences
    },
    validation: src.validation && typeof src.validation === "object" ? (src.validation as Record<string, unknown>) : {}
  } as CustomDocumentRecord;
}

function normalizeSecSpeechRecord(speech: unknown): CustomDocumentRecord | null {
  if (!speech || typeof speech !== "object") {
    return null;
  }

  const src = speech as Record<string, unknown>;
  const metadataRaw = src.metadata && typeof src.metadata === "object" ? (src.metadata as Record<string, unknown>) : {};
  const contentRaw = src.content && typeof src.content === "object" ? (src.content as Record<string, unknown>) : {};

  const paragraphs = Array.isArray(contentRaw.paragraphs)
    ? contentRaw.paragraphs.map((item) => normalizeString(item)).filter(Boolean)
    : [];
  const sentences = Array.isArray(contentRaw.sentences)
    ? contentRaw.sentences.map((item) => normalizeString(item)).filter(Boolean)
    : [];

  const fullText = normalizeString(contentRaw.full_text);
  const wordCount = normalizeWordCount(metadataRaw.word_count) || (fullText ? fullText.split(/\s+/).filter(Boolean).length : 0);

  const organization = normalizeOrgLabel(metadataRaw.organization || metadataRaw.org || "SEC");
  const sourceKind = inferSourceKind(metadataRaw);
  const docType = normalizeString(metadataRaw.doc_type) || "Speech";
  const publishedDate = normalizeString(metadataRaw.published_date) || normalizeString(metadataRaw.date);
  const updatedDate = normalizeString(metadataRaw.updated_date) || normalizeString(metadataRaw.extraction_date);

  const metadata = {
    document_id: corpusDocId(metadataRaw, fullText),
    title: normalizeString(metadataRaw.title),
    speaker: normalizeString(metadataRaw.speaker),
    date: normalizeString(metadataRaw.date),
    url: normalizeString(metadataRaw.url),
    word_count: wordCount,
    organization,
    doc_type: docType,
    source_filename: normalizeString(metadataRaw.source_filename),
    source_format: normalizeString(metadataRaw.source_format) || "html",
    source_local_path: normalizeString(metadataRaw.source_local_path),
    source_gcs_path: normalizeString(metadataRaw.source_gcs_path),
    tags: normalizeString(metadataRaw.tags),
    source_kind: sourceKind,
    source_family: normalizeString(metadataRaw.source_family) || sourceKind,
    source_index_url: normalizeString(metadataRaw.source_index_url),
    published_date: publishedDate,
    updated_date: updatedDate,
    last_reviewed_or_updated: normalizeString(metadataRaw.last_reviewed_or_updated) || updatedDate || publishedDate
  };

  return {
    metadata,
    content: {
      full_text: fullText,
      paragraphs,
      sentences
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

function normalizeSecSpeechesPayload(payload: unknown): CustomDocumentsPayload {
  if (!payload || typeof payload !== "object") {
    return { updated_at: "", documents: [] };
  }

  const src = payload as Record<string, unknown>;
  const speechesRaw = Array.isArray(src.speeches) ? src.speeches : [];
  const documents = speechesRaw.map((item) => normalizeSecSpeechRecord(item)).filter(Boolean) as CustomDocumentRecord[];

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
      comment_position:
        enrichmentRaw.comment_position && typeof enrichmentRaw.comment_position === "object"
          ? enrichmentRaw.comment_position
          : {},
      evidence_spans: Array.isArray(enrichmentRaw.evidence_spans)
        ? enrichmentRaw.evidence_spans.filter((item) => item && typeof item === "object")
        : [],
      confidence: Number.parseFloat(String(enrichmentRaw.confidence ?? "0")) || 0
    },
    review: {
      decision: normalizeString(reviewRaw.decision),
      notes: normalizeString(reviewRaw.notes),
      reviewed_at: normalizeString(reviewRaw.reviewed_at)
    },
    reward: src.reward && typeof src.reward === "object" ? (src.reward as Record<string, unknown>) : {},
    auto_review: src.auto_review && typeof src.auto_review === "object" ? (src.auto_review as Record<string, unknown>) : {}
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
      organization_label: "News",
      domains: "",
      exclude_domains: "",
      tags_csv: "",
      doj_usao_exclude_terms: ""
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
    organization_label: normalizeNewsOrgLabel(src.organization_label || "News"),
    domains: normalizeString(src.domains),
    exclude_domains: normalizeString(src.exclude_domains),
    tags_csv: normalizeString(src.tags_csv),
    doj_usao_exclude_terms: normalizeString(src.doj_usao_exclude_terms)
  };
}

function findProjectRootWithData(startDir: string): string {
  let current = path.resolve(startDir);
  for (let i = 0; i < 7; i += 1) {
    const candidate = path.join(current, "data", SEC_SPEECHES_LOCAL_FILE);
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

function localDataFilePath(fileName: string): string {
  return path.join(resolveDataDirPath(), fileName);
}

function readLocalJson(fileName: string): unknown | null {
  const filePath = localDataFilePath(fileName);
  if (!fs.existsSync(filePath)) {
    return null;
  }
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf-8"));
  } catch {
    return null;
  }
}

function writeLocalJson(fileName: string, payload: unknown): boolean {
  try {
    const filePath = localDataFilePath(fileName);
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
    fs.writeFileSync(filePath, JSON.stringify(payload, null, 2), "utf-8");
    return true;
  } catch {
    return false;
  }
}

interface SourceLoadConfig<T> {
  cacheKey: string;
  gcsBlobName: string;
  localFileName: string;
  normalize: (payload: unknown) => T;
  emptyFactory: () => T;
}

async function loadFromSource<T>(config: SourceLoadConfig<T>): Promise<T> {
  const now = Date.now();
  const hit = cache.get(config.cacheKey);
  if (hit && now - hit.loadedAt < CACHE_TTL_MS) {
    return hit.data as T;
  }

  const cfg = getDataSourceConfig();
  let raw: unknown | null = null;

  if (cfg.mode === "gcs" || cfg.mode === "auto") {
    raw = await downloadGcsJson<unknown>(config.gcsBlobName);
  }
  if (raw === null && (cfg.mode === "local" || cfg.mode === "auto")) {
    raw = readLocalJson(config.localFileName);
  }

  const normalized = raw === null ? config.emptyFactory() : config.normalize(raw);
  cache.set(config.cacheKey, { loadedAt: now, data: normalized });
  return normalized;
}

function clearCacheKey(cacheKey: string): void {
  cache.delete(cacheKey);
}

export async function loadSecSpeeches(): Promise<CustomDocumentsPayload> {
  return loadFromSource({
    cacheKey: "sec_speeches",
    gcsBlobName: SEC_SPEECHES_GCS_BLOB,
    localFileName: SEC_SPEECHES_LOCAL_FILE,
    normalize: normalizeSecSpeechesPayload,
    emptyFactory: () => ({ updated_at: "", documents: [] })
  });
}

export async function loadCustomDocuments(): Promise<CustomDocumentsPayload> {
  return loadFromSource({
    cacheKey: "custom_documents",
    gcsBlobName: CUSTOM_DOCS_BLOB,
    localFileName: CUSTOM_DOCS_BLOB,
    normalize: normalizeCustomDocumentsPayload,
    emptyFactory: () => ({ updated_at: "", documents: [] })
  });
}

export async function loadEnrichmentState(): Promise<EnrichmentStatePayload> {
  return loadFromSource({
    cacheKey: "enrichment_state",
    gcsBlobName: ENRICHMENT_BLOB,
    localFileName: ENRICHMENT_BLOB,
    normalize: normalizeEnrichmentStatePayload,
    emptyFactory: () => ({ version: 1, pipeline_version: "v1", updated_at: "", entries: {} })
  });
}

export async function loadNewsConnectorSettings(): Promise<NewsConnectorSettingsPayload> {
  return loadFromSource({
    cacheKey: "news_connector_settings",
    gcsBlobName: SETTINGS_BLOB,
    localFileName: SETTINGS_BLOB,
    normalize: normalizeNewsSettingsPayload,
    emptyFactory: () => ({
      updated_at: "",
      query: "",
      lookback_days: 7,
      max_pages: 4,
      page_size: 50,
      target_count: 100,
      sort_by: "publishedAt",
      organization_label: "News",
      domains: "",
      exclude_domains: "",
      tags_csv: "",
      doj_usao_exclude_terms: ""
    })
  });
}

export async function saveNewsConnectorSettings(payload: Partial<NewsConnectorSettingsPayload>): Promise<{
  saved: boolean;
  local_saved: boolean;
  remote_saved: boolean;
  settings: NewsConnectorSettingsPayload;
}> {
  const existing = await loadNewsConnectorSettings();
  const normalized = normalizeNewsSettingsPayload({
    ...existing,
    ...payload,
    updated_at: new Date().toISOString()
  });

  const cfg = getDataSourceConfig();
  let remoteSaved = false;
  let localSaved = false;

  if (cfg.mode === "gcs" || cfg.mode === "auto") {
    remoteSaved = await uploadGcsJson(SETTINGS_BLOB, normalized);
  }
  if (cfg.mode === "local" || cfg.mode === "auto" || !remoteSaved) {
    localSaved = writeLocalJson(SETTINGS_BLOB, normalized);
  }

  clearCacheKey("news_connector_settings");

  return {
    saved: remoteSaved || localSaved,
    local_saved: localSaved,
    remote_saved: remoteSaved,
    settings: normalized
  };
}

export async function loadCorpusDocuments(): Promise<CustomDocumentRecord[]> {
  const [secPayload, customPayload] = await Promise.all([loadSecSpeeches(), loadCustomDocuments()]);

  const dedup = new Map<string, CustomDocumentRecord>();

  for (const doc of secPayload.documents || []) {
    const id = normalizeString(doc.metadata?.document_id);
    if (id) {
      dedup.set(id, doc);
    }
  }
  for (const doc of customPayload.documents || []) {
    const id = normalizeString(doc.metadata?.document_id);
    if (id) {
      dedup.set(id, doc);
    }
  }

  return [...dedup.values()];
}

export function buildDocumentListItems(
  corpusDocs: CustomDocumentRecord[],
  enrichmentState: EnrichmentStatePayload
): DocumentListItem[] {
  const entries = enrichmentState.entries || {};

  return corpusDocs.map((doc) => {
    const m = doc.metadata || ({} as CustomDocumentRecord["metadata"]);
    const docId = normalizeString(m.document_id);
    const enrich = entries[docId];
    const reviewDecision = normalizeString(enrich?.review?.decision || "pending") || "pending";

    const metadataTags = splitCsv(normalizeString(m.tags));
    const enrichTags = Array.isArray(enrich?.enrichment?.tags)
      ? enrich?.enrichment?.tags.map((item) => normalizeString(item)).filter(Boolean)
      : [];
    const keywords = Array.isArray(enrich?.enrichment?.keywords)
      ? enrich?.enrichment?.keywords.map((item) => normalizeString(item)).filter(Boolean)
      : [];

    const topics = dedupList([...enrichTags, ...metadataTags]);
    const tags = dedupList([...metadataTags, ...enrichTags]);

    return {
      document_id: docId,
      title: normalizeString(m.title),
      organization: normalizeOrgLabel(m.organization),
      source_kind: normalizeString(m.source_kind) || inferSourceKind((m as unknown as Record<string, unknown>) || {}),
      doc_type: normalizeString(m.doc_type) || "Document",
      speaker: normalizeString(m.speaker),
      url: normalizeString(m.url),
      date: normalizeString(m.date),
      published_at: normalizeString(m.published_date) || normalizeString(m.date),
      word_count: normalizeWordCount(m.word_count),
      tags,
      keywords: dedupList(keywords),
      topics,
      ingest_status: "existing",
      enrichment_status: normalizeString(enrich?.status || "not_enriched") || "not_enriched",
      review_decision: reviewDecision,
      updated_at:
        normalizeString(m.last_reviewed_or_updated) || normalizeString(m.updated_date) || normalizeString(enrich?.updated_at)
    };
  });
}

export function buildDocumentsFacets(items: DocumentListItem[]): DocumentsFacets {
  const sources = dedupList(items.map((item) => item.source_kind));
  const organizations = dedupList(items.map((item) => item.organization));
  const topicCounts = new Map<string, { label: string; count: number }>();
  for (const item of items) {
    const uniqueTopicKeys = new Set<string>();
    for (const topic of item.topics || []) {
      const key = canonicalFacetToken(topic);
      const label = formatFacetLabel(topic);
      if (!label) {
        continue;
      }
      if (uniqueTopicKeys.has(key)) {
        continue;
      }
      uniqueTopicKeys.add(key);
      const current = topicCounts.get(key);
      if (current) {
        current.count += 1;
      } else {
        topicCounts.set(key, { label, count: 1 });
      }
    }
  }
  const topics = [...topicCounts.values()]
    .map((entry) => entry.label)
    .sort((a, b) => a.localeCompare(b));
  const keyTopics = [...topicCounts.values()]
    .sort((a, b) => (b.count - a.count) || a.label.localeCompare(b.label))
    .slice(0, 10)
    .map((entry) => entry.label);
  const keywords = dedupList(items.flatMap((item) => item.keywords || []));
  const statuses = dedupList(items.map((item) => item.enrichment_status));

  return {
    sources,
    organizations,
    topics,
    key_topics: keyTopics,
    keywords,
    statuses
  };
}

export function parseComparableDate(value: string): number {
  const parsed = new Date(value);
  const ms = parsed.getTime();
  return Number.isNaN(ms) ? 0 : ms;
}

import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";

import { getDataSourceConfig } from "@/lib/server/env";
import { downloadGcsJson, uploadGcsJson } from "@/lib/server/gcs-loader";
import {
  type CustomDocumentMetadata,
  type CustomDocumentRecord,
  type CustomDocumentsPayload,
  type DocumentListItem,
  type DocumentsFacets,
  type EnrichmentEntry,
  type EnrichmentStatePayload,
  type NewsConnectorSettingsPayload,
  type RuleSummariesPayload,
  type TrendItem,
  type TrendsPayload
} from "@/lib/server/types";
import {
  fullTextToDocumentContent,
  loadMetadataOnlyFeed,
  projectionRowsToCorpusAndEnrichment,
  projectionRowsToEnrichmentState,
} from "@/lib/server/document-metadata-feed";
import {
  getAllMirroredDocumentMetadata,
  getMirroredDocumentFacets,
  getMirroredDocumentFeedMetadata,
  getMirroredDocumentListPage,
  getMirroredNoticeDocuments,
  getNewsConnectorSettingsRow,
  isDocumentEnrichmentProjectionAvailable,
  saveNewsConnectorSettingsRow,
  type MirroredDocumentListOptions,
  type NeonDocumentFacetData,
} from "@/lib/server/neon";

const SEC_SPEECHES_GCS_BLOB = "all_speeches.json";
const SEC_SPEECHES_LOCAL_FILE = "all_speeches_final.json";
const CUSTOM_DOCS_BLOB = "custom_documents.json";
const ENRICHMENT_BLOB = "document_enrichment_state.json";
const RULE_SUMMARIES_BLOB = "rule_summaries.json";
const TRENDS_BLOB = "trends_daily.json";

const CACHE_TTL_MS = 120_000;

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

// Enrichment entities are stored as {name, type, mentions} objects (see
// _normalize_enrichment_payload in run_financial_news_pipeline.py), not
// strings - running them through normalizeString() produced the literal
// label "[object Object]" for every entity, which collapsed all entity
// nodes in the knowledge graph into one garbage node. Extract the name;
// plain-string entries (older metadata shapes) pass through unchanged.
function entityLabel(item: unknown): string {
  if (item && typeof item === "object" && !Array.isArray(item)) {
    return normalizeString((item as Record<string, unknown>).name);
  }
  return normalizeString(item);
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

function extractPublishedDateFromText(value: unknown): string {
  const text = normalizeString(value);
  if (!text) {
    return "";
  }
  const match = text.match(/\bPublished Date\s*:\s*([A-Za-z]{3,9}\.?\s+\d{1,2},\s+\d{4})\b/i);
  return match ? normalizeString(match[1]).replace(/\bSept\./i, "Sep").replace(/\./g, "") : "";
}

function normalizeDocumentPublishedDate(metadata: Record<string, unknown>, fullText: string): string {
  const sourceKind = normalizeString(metadata.source_kind).toLowerCase();
  if (sourceKind === "finra_regulatory_notice") {
    return (
      extractPublishedDateFromText(fullText) ||
      normalizeString(metadata.published_date) ||
      normalizeString(metadata.date)
    );
  }
  return normalizeString(metadata.published_date) || normalizeString(metadata.date);
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

  const sourceKind = normalizeString(metadataRaw.source_kind) || inferSourceKind(metadataRaw);
  const publishedDate = normalizeDocumentPublishedDate({ ...metadataRaw, source_kind: sourceKind }, fullText);

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
    source_kind: sourceKind,
    source_family: normalizeString(metadataRaw.source_family),
    source_index_url: normalizeString(metadataRaw.source_index_url),
    published_date: publishedDate,
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
    release_no: normalizeString(metadataRaw.release_no),
    case_id: normalizeString(metadataRaw.case_id),
    subject_text: normalizeString(metadataRaw.subject_text),
    case_summary: normalizeString(metadataRaw.case_summary),
    action_type: normalizeString(metadataRaw.action_type),
    forum: normalizeString(metadataRaw.forum),
    outcome_status: normalizeString(metadataRaw.outcome_status),
    alleged_violations: Array.isArray(metadataRaw.alleged_violations)
      ? metadataRaw.alleged_violations.map((item) => normalizeString(item)).filter(Boolean)
      : splitCsv(normalizeString(metadataRaw.alleged_violations)),
    entities: Array.isArray(metadataRaw.entities)
      ? metadataRaw.entities.map((item) => entityLabel(item)).filter(Boolean)
      : splitCsv(normalizeString(metadataRaw.entities)),
    respondents: Array.isArray(metadataRaw.respondents)
      ? metadataRaw.respondents.map((item) => normalizeString(item)).filter(Boolean)
      : splitCsv(normalizeString(metadataRaw.respondents)),
    sanctions: Array.isArray(metadataRaw.sanctions)
      ? metadataRaw.sanctions.map((item) => normalizeString(item)).filter(Boolean)
      : splitCsv(normalizeString(metadataRaw.sanctions)),
    sanctions_text: normalizeString(metadataRaw.sanctions_text),
    detail_url: normalizeString(metadataRaw.detail_url),
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
    summary: normalizeString(metadataRaw.summary),
    source_name: normalizeString(metadataRaw.source_name),
    authors: Array.isArray(metadataRaw.authors)
      ? metadataRaw.authors.map((item) => normalizeString(item)).filter(Boolean)
      : splitCsv(normalizeString(metadataRaw.authors)),
    keywords: Array.isArray(metadataRaw.keywords)
      ? metadataRaw.keywords.map((item) => normalizeString(item)).filter(Boolean)
      : splitCsv(normalizeString(metadataRaw.keywords)),
    apify_actor_id: normalizeString(metadataRaw.apify_actor_id),
    apify_raw_keys: Array.isArray(metadataRaw.apify_raw_keys)
      ? metadataRaw.apify_raw_keys.map((item) => normalizeString(item)).filter(Boolean)
      : []
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
  const publishedDate = normalizeDocumentPublishedDate({ ...metadataRaw, source_kind: sourceKind }, fullText);
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
        ? enrichmentRaw.entities.map((item) => entityLabel(item)).filter(Boolean)
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
    enforcement_analysis:
      src.enforcement_analysis && typeof src.enforcement_analysis === "object"
        ? src.enforcement_analysis as Record<string, unknown>
        : undefined,
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

function normalizeRuleSummariesPayload(payload: unknown): RuleSummariesPayload {
  if (!payload || typeof payload !== "object") {
    return {
      version: 1,
      updated_at: "",
      generated_at: "",
      custom_documents_updated_at: "",
      enrichment_state_updated_at: "",
      totals: {
        notices: 0,
        comments: 0,
        enriched_comments: 0,
        pending_review_comments: 0
      },
      groups: []
    };
  }

  const src = payload as Record<string, unknown>;
  const totalsRaw = src.totals && typeof src.totals === "object" ? (src.totals as Record<string, unknown>) : {};
  const groupsRaw = Array.isArray(src.groups) ? src.groups : [];

  return {
    version: Number.parseInt(String(src.version ?? "1"), 10) || 1,
    updated_at: normalizeString(src.updated_at),
    generated_at: normalizeString(src.generated_at),
    custom_documents_updated_at: normalizeString(src.custom_documents_updated_at),
    enrichment_state_updated_at: normalizeString(src.enrichment_state_updated_at),
    totals: {
      notices: normalizeWordCount(totalsRaw.notices),
      comments: normalizeWordCount(totalsRaw.comments),
      enriched_comments: normalizeWordCount(totalsRaw.enriched_comments),
      pending_review_comments: normalizeWordCount(totalsRaw.pending_review_comments)
    },
    groups: groupsRaw
      .filter((item) => item && typeof item === "object")
      .map((item) => {
        const group = item as Record<string, unknown>;
        const overviewRaw = group.overview && typeof group.overview === "object" ? (group.overview as Record<string, unknown>) : {};
        const positionCountsRaw =
          overviewRaw.position_counts && typeof overviewRaw.position_counts === "object"
            ? (overviewRaw.position_counts as Record<string, unknown>)
            : {};
        const topTopicsRaw = Array.isArray(overviewRaw.top_topics) ? overviewRaw.top_topics : [];
        return {
          notice_key: normalizeString(group.notice_key),
          source_kind: normalizeString(group.source_kind),
          source_family: normalizeString(group.source_family),
          source_family_label: normalizeString(group.source_family_label),
          group_type_label: normalizeString(group.group_type_label),
          group_identifier_label: normalizeString(group.group_identifier_label),
          group_identifier: normalizeString(group.group_identifier),
          notice_document_id: normalizeString(group.notice_document_id),
          notice_number: normalizeString(group.notice_number),
          docket_id: normalizeString(group.docket_id),
          title: normalizeString(group.title),
          summary: normalizeString(group.summary),
          organization: normalizeString(group.organization),
          url: normalizeString(group.url),
          pdf_url: normalizeString(group.pdf_url),
          published_at: normalizeString(group.published_at),
          effective_date: normalizeString(group.effective_date),
          comment_deadline: normalizeString(group.comment_deadline),
          tags: Array.isArray(group.tags) ? group.tags.map((value) => normalizeString(value)).filter(Boolean) : [],
          keywords: Array.isArray(group.keywords) ? group.keywords.map((value) => normalizeString(value)).filter(Boolean) : [],
          enrichment_status: normalizeString(group.enrichment_status),
          review_decision: normalizeString(group.review_decision),
          comment_count: normalizeWordCount(group.comment_count),
          latest_comment_at: normalizeString(group.latest_comment_at),
          overview: {
            total_comments: normalizeWordCount(overviewRaw.total_comments),
            enriched_comments: normalizeWordCount(overviewRaw.enriched_comments),
            position_counts: Object.fromEntries(
              Object.entries(positionCountsRaw).map(([key, value]) => [normalizeString(key), normalizeWordCount(value)])
            ),
            top_topics: topTopicsRaw
              .filter((topic) => topic && typeof topic === "object")
              .map((topic) => {
                const itemRaw = topic as Record<string, unknown>;
                return {
                  label: normalizeString(itemRaw.label),
                  count: normalizeWordCount(itemRaw.count),
                  share: Number.parseFloat(String(itemRaw.share ?? "0")) || 0
                };
              })
              .filter((topic) => topic.label)
          },
          comment_document_ids: Array.isArray(group.comment_document_ids)
            ? group.comment_document_ids.map((value) => normalizeString(value)).filter(Boolean)
            : [],
          comments: Array.isArray(group.comments)
            ? group.comments
                .filter((c) => c && typeof c === "object")
                .map((c) => {
                  const comment = c as Record<string, unknown>;
                  const posRaw = comment.comment_position && typeof comment.comment_position === "object"
                    ? (comment.comment_position as Record<string, unknown>)
                    : {};
                  return {
                    document_id: normalizeString(comment.document_id),
                    source_kind: normalizeString(comment.source_kind),
                    source_family: normalizeString(comment.source_family),
                    title: normalizeString(comment.title),
                    commenter_name: normalizeString(comment.commenter_name),
                    commenter_org: normalizeString(comment.commenter_org),
                    speaker: normalizeString(comment.speaker),
                    url: normalizeString(comment.url),
                    comment_url: normalizeString(comment.comment_url),
                    pdf_url: normalizeString(comment.pdf_url),
                    resolved_content_url: normalizeString(comment.resolved_content_url),
                    published_at: normalizeString(comment.published_at),
                    summary: normalizeString(comment.summary),
                    tags: Array.isArray(comment.tags) ? comment.tags.map((v) => normalizeString(v)).filter(Boolean) : [],
                    keywords: Array.isArray(comment.keywords) ? comment.keywords.map((v) => normalizeString(v)).filter(Boolean) : [],
                    enrichment_status: normalizeString(comment.enrichment_status),
                    review_decision: normalizeString(comment.review_decision),
                    comment_position: {
                      label: normalizeString(posRaw.label || "unclear"),
                      confidence: Math.max(0, Math.min(1, Number.parseFloat(String(posRaw.confidence ?? "0")) || 0)),
                      rationale: normalizeString(posRaw.rationale)
                    }
                  };
                })
            : undefined
        };
      })
      .filter((group) => group.notice_key)
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
  } catch (err) {
    console.error(`[data-store] readLocalJson failed for ${fileName}:`, err);
    return null;
  }
}

function writeLocalJson(fileName: string, payload: unknown): boolean {
  try {
    const filePath = localDataFilePath(fileName);
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
    fs.writeFileSync(filePath, JSON.stringify(payload, null, 2), "utf-8");
    return true;
  } catch (err) {
    console.error(`[data-store] writeLocalJson failed for ${fileName}:`, err);
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

export function invalidateDocumentCaches(): void {
  clearCacheKey("sec_speeches");
  clearCacheKey("custom_documents");
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

export async function saveEnrichmentState(payload: EnrichmentStatePayload): Promise<{
  saved: boolean;
  local_saved: boolean;
  remote_saved: boolean;
  state: EnrichmentStatePayload;
}> {
  const normalized = normalizeEnrichmentStatePayload({
    ...payload,
    updated_at: new Date().toISOString()
  });

  const cfg = getDataSourceConfig();
  let remoteSaved = false;
  let localSaved = false;

  if (cfg.mode === "gcs" || cfg.mode === "auto") {
    remoteSaved = await uploadGcsJson(ENRICHMENT_BLOB, normalized);
  }
  if (cfg.mode === "local" || cfg.mode === "auto" || !remoteSaved) {
    localSaved = writeLocalJson(ENRICHMENT_BLOB, normalized);
  }

  clearCacheKey("enrichment_state");

  return {
    saved: remoteSaved || localSaved,
    local_saved: localSaved,
    remote_saved: remoteSaved,
    state: normalized
  };
}

export async function loadRuleSummaries(): Promise<RuleSummariesPayload> {
  return loadFromSource({
    cacheKey: "rule_summaries",
    gcsBlobName: RULE_SUMMARIES_BLOB,
    localFileName: RULE_SUMMARIES_BLOB,
    normalize: normalizeRuleSummariesPayload,
    emptyFactory: () => ({
      version: 1,
      updated_at: "",
      generated_at: "",
      custom_documents_updated_at: "",
      enrichment_state_updated_at: "",
      totals: {
        notices: 0,
        comments: 0,
        enriched_comments: 0,
        pending_review_comments: 0
      },
      groups: []
    })
  });
}

// Neon is the sole store for this settings row (see neon.ts's
// getNewsConnectorSettingsRow/saveNewsConnectorSettingsRow, and
// neon_feeds.get_news_connector_settings on the Python read side).
// news_connector_settings.json had exactly one writer (this admin route) and
// one Python reader, so this is a straight cutover rather than a dual-read -
// no GCS fallback, matching the fail-closed posture the other Neon-backed
// readers in this file already use.
function emptyNewsConnectorSettings(): NewsConnectorSettingsPayload {
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

export async function loadNewsConnectorSettings(): Promise<NewsConnectorSettingsPayload> {
  const now = Date.now();
  const cacheKey = "news_connector_settings";
  const hit = cache.get(cacheKey);
  if (hit && now - hit.loadedAt < CACHE_TTL_MS) {
    return hit.data as NewsConnectorSettingsPayload;
  }

  let normalized: NewsConnectorSettingsPayload;
  try {
    const row = await getNewsConnectorSettingsRow();
    normalized = normalizeNewsSettingsPayload(row);
  } catch (error) {
    console.error("[data-store] loadNewsConnectorSettings failed closed:", error);
    normalized = emptyNewsConnectorSettings();
  }

  cache.set(cacheKey, { loadedAt: now, data: normalized });
  return normalized;
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

  let remoteSaved = false;
  try {
    await saveNewsConnectorSettingsRow(normalized as unknown as Record<string, unknown>);
    remoteSaved = true;
  } catch (error) {
    console.error("[data-store] saveNewsConnectorSettings failed:", error);
  }

  clearCacheKey("news_connector_settings");

  return {
    saved: remoteSaved,
    local_saved: false,
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

export type CorpusDocumentsLoadResult = {
  documents: CustomDocumentRecord[];
  source: "neon" | "unavailable";
  warning?: string;
};

export type NewsFeedDocumentsLoadResult = {
  documents: DocumentListItem[];
  source: "neon" | "unavailable";
  metadata_only: true;
  warning?: string;
};

export type NeonDocumentListLoadResult = {
  items: DocumentListItem[];
  total: number;
  facets: DocumentsFacets;
  warnings: string[];
};

// Metadata-only compatibility reader. It intentionally fails closed instead
// of turning a Neon outage into a full GCS download. New interactive readers
// should prefer the bounded page/detail functions below.
export async function loadCorpusDocumentsFromNeon(): Promise<CorpusDocumentsLoadResult> {
  try {
    const neonRows = await getAllMirroredDocumentMetadata();
    const documents: CustomDocumentRecord[] = [];
    for (const row of neonRows) {
      const id = normalizeString(row.document_id);
      if (!id) continue;
      const metadata = (row.metadata && typeof row.metadata === "object" ? row.metadata : {}) as unknown as CustomDocumentMetadata;
      documents.push({
        metadata: { ...metadata, document_id: id },
        content: { full_text: "", paragraphs: [], sentences: [] }
      });
    }

    return { documents, source: "neon" };
  } catch (error) {
    console.error("[data-store] loadCorpusDocumentsFromNeon failed closed:", error);
    return { documents: [], source: "unavailable", warning: "Neon corpus read failed; GCS fallback is disabled" };
  }
}

export type NoticeDocumentsLoadResult = {
  documents: CustomDocumentRecord[];
  enrichment: EnrichmentStatePayload;
  source: "neon" | "unavailable";
  warning?: string;
};

/**
 * Notices/rulemakings and their comments, read from Neon rather than the
 * legacy `rule_summaries.json` -> `custom_documents.json` chain.
 *
 * Fails closed, deliberately: the GCS snapshot for these source kinds stopped
 * moving when scheduled snapshot egress was paused (SEC-20), so falling back
 * to it would serve stale data that looks current. An empty result is treated
 * as a failure for the same reason the caller surfaces a warning - "backend
 * returned nothing" and "your filters matched nothing" rendered identically
 * for weeks, which is why this outage went unnoticed.
 */
export async function loadNoticeDocumentsFromNeon(
  sourceKinds: string[],
  sourceFamilies: string[] = []
): Promise<NoticeDocumentsLoadResult> {
  const emptyEnrichment: EnrichmentStatePayload = {
    version: 1,
    pipeline_version: "",
    updated_at: "",
    entries: {}
  };

  try {
    const rows = await getMirroredNoticeDocuments({ sourceKinds, sourceFamilies });

    if (rows.length === 0) {
      return {
        documents: [],
        enrichment: emptyEnrichment,
        source: "unavailable",
        warning: "Neon returned no notice or comment records; the GCS fallback is intentionally disabled"
      };
    }

    const documents: CustomDocumentRecord[] = [];
    for (const row of rows) {
      const id = normalizeString(row.document_id);
      if (!id) continue;
      const metadata = (row.metadata && typeof row.metadata === "object" ? row.metadata : {}) as unknown as CustomDocumentMetadata;
      documents.push({
        metadata: { ...metadata, document_id: id },
        content: fullTextToDocumentContent(row.full_text)
      });
    }

    return {
      documents,
      enrichment: projectionRowsToEnrichmentState(rows),
      source: "neon"
    };
  } catch (error) {
    console.error("[data-store] loadNoticeDocumentsFromNeon failed closed:", error);
    return {
      documents: [],
      enrichment: emptyEnrichment,
      source: "unavailable",
      warning: "Neon notice read failed; the GCS fallback is intentionally disabled"
    };
  }
}

export const DEFAULT_NEWS_FEED_PINNED_SOURCE_KINDS = [
  "sec_speech",
  "bloomberg_apify_article",
  "bloomberg_public_article",
  "substack_public_article",
  "newsapi_article",
  "sifma_news_item",
  "ici_news_item",
  "isda_news_item",
  "mfa_news_item",
  "fia_news_item",
  "aba_news_item",
  "bpi_news_item",
  "icba_news_item",
  "lsta_news_item",
  "federal_reserve_speech_testimony",
  "treasury_featured_story",
  "treasury_press_release",
  "treasury_statement_remark",
  "cftc_press_release",
  "cftc_public_statement_remark",
  "sec_tm_faq",
  "sec_press_release_rss",
  "sec_federal_register",
  "sec_pcaob_rulemaking",
  "pcaob_update",
  "msrb_press_release",
  "finra_regulatory_notice",
  "finra_awc",
  "jdsupra_article",
  "investmentnews_article",
  "citywire_article",
  "therecord_media_article",
  "wired_article",
  "tripwire_article",
  "akamai_blog_article",
  "ritholtz_article",
  "ft_portfolios_market_commentary",
  "liberty_street_economics_article",
  "wealth_of_common_sense_article",
  "congress_crs_product",
  "wsj_dow_jones",
  "wsj_rss_article",
  "sec_youtube_video",
  "youtube_video",
  "reddit_post",
  "sec_enforcement_litigation",
  "sec_administrative_proceeding",
  "sec_trading_suspension"
];

/**
 * Cost-contained news-feed reader. The successful path reads a bounded Neon
 * metadata projection and intentionally does not touch all_speeches.json,
 * custom_documents.json, or document_enrichment_state.json. Full GCS data is
 * never used as an automatic fallback, and document detail remains unchanged.
 */
export async function loadNewsFeedDocumentsFromNeon(
  options: { limit?: number; pinnedSourceKinds?: string[]; pinnedSourceKindLimit?: number; maxDateMs?: number } = {}
): Promise<NewsFeedDocumentsLoadResult> {
  const pinnedSourceKinds = options.pinnedSourceKinds ?? DEFAULT_NEWS_FEED_PINNED_SOURCE_KINDS;
  const listOptions = {
    ...options,
    pinnedSourceKinds,
  };

  const result = await loadMetadataOnlyFeed(
    () => getMirroredDocumentFeedMetadata({
      limit: options.limit,
      pinnedSourceKinds,
      pinnedSourceKindLimit: options.pinnedSourceKindLimit,
    }),
    (rows) => {
      const { documents: corpusDocs, enrichment } = projectionRowsToCorpusAndEnrichment(rows);
      return selectNewsFeedDocuments(
        buildDocumentListItems(corpusDocs, enrichment),
        listOptions
      );
    }
  );

  if (result.source === "neon" && !(await isDocumentEnrichmentProjectionAvailable())) {
    result.warning = "Neon enrichment projection is not available yet; using metadata-only document cards";
  }

  if (result.warning) {
    console.error(`[data-store] ${result.warning}`);
  }
  return result;
}

export function buildDocumentsFacetsFromNeon(data: NeonDocumentFacetData): DocumentsFacets {
  const topicCounts = new Map<string, { label: string; count: number }>();
  for (const raw of data.topicCounts) {
    const key = canonicalFacetToken(raw.value);
    const label = formatFacetLabel(raw.value);
    if (!key || !label) continue;
    const current = topicCounts.get(key);
    if (current) {
      current.count += Math.max(0, Number(raw.count || 0));
    } else {
      topicCounts.set(key, { label, count: Math.max(0, Number(raw.count || 0)) });
    }
  }

  const topics = [...topicCounts.values()]
    .map((entry) => entry.label)
    .sort((a, b) => a.localeCompare(b));
  const keyTopics = [...topicCounts.values()]
    .sort((a, b) => (b.count - a.count) || a.label.localeCompare(b.label))
    .slice(0, 10)
    .map((entry) => entry.label);

  return {
    sources: dedupList(data.sources),
    organizations: dedupList(data.organizations),
    topics,
    key_topics: keyTopics,
    keywords: dedupList(data.keywords),
    statuses: dedupList(data.statuses),
  };
}

/** Bounded, GCS-free projection used by the interactive document browser. */
export async function loadDocumentListPageFromNeon(
  options: MirroredDocumentListOptions
): Promise<NeonDocumentListLoadResult> {
  const [page, facetData, enrichmentAvailable] = await Promise.all([
    getMirroredDocumentListPage(options),
    getMirroredDocumentFacets(),
    isDocumentEnrichmentProjectionAvailable(),
  ]);
  // A window count is attached to returned rows. If the requested offset is
  // beyond the last page there is no row to carry it, so make one bounded
  // first-page probe to preserve the real total in the API response.
  const total = page.rows.length === 0 && (options.page ?? 1) > 1
    ? (await getMirroredDocumentListPage({ ...options, page: 1, pageSize: 1 })).total
    : page.total;
  const { documents, enrichment } = projectionRowsToCorpusAndEnrichment(page.rows);
  return {
    items: buildDocumentListItems(documents, enrichment),
    total,
    facets: buildDocumentsFacetsFromNeon(facetData),
    warnings: enrichmentAvailable ? [] : ["enrichment_state_unavailable"],
  };
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
    const metadataKeywords = Array.isArray(m.keywords)
      ? m.keywords.map((item) => normalizeString(item)).filter(Boolean)
      : splitCsv(normalizeString(m.keywords));

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
      published_at: normalizeString(m.published_at) || normalizeString(m.published_date) || normalizeString(m.date),
      word_count: normalizeWordCount(m.word_count),
      tags,
      keywords: dedupList([...keywords, ...metadataKeywords]),
      topics,
      ingest_status: "existing",
      enrichment_status: normalizeString(enrich?.status || "not_enriched") || "not_enriched",
      enrichment_summary: normalizeString(enrich?.enrichment?.summary || m.summary),
      enrichment_model: normalizeString(enrich?.model),
      enrichment_confidence:
        typeof enrich?.enrichment?.confidence === "number" ? enrich.enrichment.confidence : 0,
      review_decision: reviewDecision,
      updated_at:
        normalizeString(m.last_reviewed_or_updated) ||
        normalizeString(m.updated_date) ||
        normalizeString(m.extraction_date) ||
        normalizeString(enrich?.updated_at),
      sentiment_label: (enrich?.sentiment?.label as "positive" | "negative" | "neutral") || "",
      sentiment_score: typeof enrich?.sentiment?.score === "number" ? enrich.sentiment.score : 0,
    };
  });
}

export function selectNewsFeedDocuments(
  items: DocumentListItem[],
  options: { limit?: number; pinnedSourceKinds?: string[]; pinnedSourceKindLimit?: number; maxDateMs?: number } = {}
): DocumentListItem[] {
  const feedDocumentLimit = Math.max(0, options.limit ?? 250);
  const pinnedSourceKindLimit =
    options.pinnedSourceKindLimit === undefined
      ? Number.POSITIVE_INFINITY
      : Math.max(0, options.pinnedSourceKindLimit);
  const latestVisibleDateMs = options.maxDateMs ?? endOfTodayMs();
  const pinnedSourceKinds = new Set(
    options.pinnedSourceKinds ?? DEFAULT_NEWS_FEED_PINNED_SOURCE_KINDS
  );
  const dated = items
    .map((item) => ({ item, dateMs: parseComparableDate(item.published_at || item.date) }))
    .filter(({ dateMs }) => dateMs > 0 && dateMs <= latestVisibleDateMs)
    .sort((a, b) => b.dateMs - a.dateMs)
    .map(({ item }) => item);

  const selected = new Map<string, DocumentListItem>();
  const selectedCountsBySourceKind = new Map<string, number>();
  const add = (item: DocumentListItem) => {
    if (!item.document_id || selected.has(item.document_id)) return;
    selected.set(item.document_id, item);
    selectedCountsBySourceKind.set(
      item.source_kind,
      (selectedCountsBySourceKind.get(item.source_kind) ?? 0) + 1
    );
  };

  dated.slice(0, feedDocumentLimit).forEach(add);

  for (const item of dated) {
    if (pinnedSourceKinds.has(item.source_kind)) {
      if ((selectedCountsBySourceKind.get(item.source_kind) ?? 0) >= pinnedSourceKindLimit) {
        continue;
      }
      add(item);
    }
  }

  return [...selected.values()]
    .sort((a, b) => parseComparableDate(b.published_at || b.date) - parseComparableDate(a.published_at || a.date));
}

function endOfTodayMs(): number {
  const today = new Date();
  today.setHours(23, 59, 59, 999);
  return today.getTime();
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

function normalizeTrendsPayload(payload: unknown): TrendsPayload {
  if (!payload || typeof payload !== "object") {
    return { version: 1, generated_at: "", trend_count: 0, trends: [] };
  }
  const src = payload as Record<string, unknown>;
  const trendsRaw = Array.isArray(src.trends) ? src.trends : [];

  const trends: TrendItem[] = trendsRaw
    .filter((item) => item && typeof item === "object")
    .map((item) => {
      const t = item as Record<string, unknown>;
      const sparklineRaw = Array.isArray(t.sparkline) ? t.sparkline : [];
      return {
        id: normalizeString(t.id),
        label: normalizeString(t.label),
        canonical_tag: normalizeString(t.canonical_tag),
        cluster_tags: Array.isArray(t.cluster_tags) ? t.cluster_tags.map((v) => normalizeString(v)).filter(Boolean) : [],
        description: normalizeString(t.description),
        total_mentions: Number.parseInt(String(t.total_mentions ?? "0"), 10) || 0,
        recent_mentions: Number.parseInt(String(t.recent_mentions ?? "0"), 10) || 0,
        growth_pct: Number.parseFloat(String(t.growth_pct ?? "0")) || 0,
        first_seen: normalizeString(t.first_seen),
        last_seen: normalizeString(t.last_seen),
        sparkline: sparklineRaw
          .filter((pt) => pt && typeof pt === "object")
          .map((pt) => {
            const p = pt as Record<string, unknown>;
            return {
              date: normalizeString(p.date),
              count: Number.parseInt(String(p.count ?? "0"), 10) || 0
            };
          }),
        top_doc_ids: Array.isArray(t.top_doc_ids) ? t.top_doc_ids.map((v) => normalizeString(v)).filter(Boolean) : [],
        top_docs: Array.isArray(t.top_docs)
          ? t.top_docs.map((d) => {
              const doc = d as Record<string, unknown>;
              return {
                id: normalizeString(doc.id),
                title: normalizeString(doc.title),
                date: normalizeString(doc.date),
                source_kind: normalizeString(doc.source_kind),
                url: normalizeString(doc.url),
                summary: normalizeString(doc.summary),
              };
            }).filter((d) => d.id)
          : [],
        sources: Array.isArray(t.sources) ? t.sources.map((v) => normalizeString(v)).filter(Boolean) : []
      };
    })
    .filter((t) => t.id);

  return {
    version: Number.parseInt(String(src.version ?? "1"), 10) || 1,
    generated_at: normalizeString(src.generated_at),
    trend_count: trends.length,
    trends
  };
}

export async function loadTrendsData(): Promise<TrendsPayload> {
  return loadFromSource({
    cacheKey: "trends_daily",
    gcsBlobName: TRENDS_BLOB,
    localFileName: TRENDS_BLOB,
    normalize: normalizeTrendsPayload,
    emptyFactory: () => ({ version: 1, generated_at: "", trend_count: 0, trends: [] })
  });
}

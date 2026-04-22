import { createHash } from "node:crypto";

import type { IntelligenceEvidenceArticle } from "../intelligence-types.ts";
import {
  focusAreasForProductCategory,
  type NormalizedTheme,
  type ProductCategory,
  type ProductFocusArea
} from "../theme-intelligence.ts";
import type { CustomDocumentRecord, DocumentListItem, EnrichmentStatePayload } from "./types.ts";

const STORED_EVIDENCE_MAX_RECORDS = 30;
const STORED_EVIDENCE_CACHE_TTL_MS = 60_000;
const TOKEN_ONLY_PATTERNS = new Set(["AML", "BSA", "CIP", "KYC", "OFAC", "SAR"]);

const storedEvidenceCache = new Map<string, { loadedAt: number; articles: IntelligenceEvidenceArticle[] }>();

const PATTERN_ALIASES: Readonly<Record<string, readonly string[]>> = {
  SANCTIONS: ["SANCTIONS", "SANCTION", "SANCTIONED", "TREASURY_SANCTIONS", "OFAC_SANCTIONS"],
  OFAC: ["OFAC", "OFAC_SANCTIONS"],
  AML: ["AML", "AML_CFT", "AML_KYC", "ANTI_MONEY_LAUNDERING"],
  BSA: ["BSA", "BANK_SECRECY_ACT"],
  MONEY_LAUNDERING: ["MONEY_LAUNDERING", "ANTI_MONEY_LAUNDERING"],
  ANTI_MONEY_LAUNDERING: ["ANTI_MONEY_LAUNDERING", "AML"],
  KYC: ["KYC", "KNOW_YOUR_CUSTOMER", "AML_KYC"],
  CIP: ["CIP", "CUSTOMER_IDENTIFICATION_PROGRAM"],
  CUSTOMER_IDENTIFICATION: ["CUSTOMER_IDENTIFICATION", "CUSTOMER_IDENTIFICATION_PROGRAM"],
  BENEFICIAL_OWNERSHIP: ["BENEFICIAL_OWNERSHIP", "BENEFICIAL_OWNER"],
  ILLICIT_FINANCE: ["ILLICIT_FINANCE", "ILLICIT_FINANCING"],
  SUSPICIOUS_ACTIVITY: ["SUSPICIOUS_ACTIVITY", "SUSPICIOUS_ACTIVITY_REPORT"],
  SAR: ["SAR", "SUSPICIOUS_ACTIVITY_REPORT"],
  TERRORIST_FINANCING: ["TERRORIST_FINANCING", "TERRORISM_FINANCING"]
};

function normalizeMatchText(value: string): string {
  return value
    .toUpperCase()
    .replace(/&/g, " AND ")
    .replace(/[^A-Z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function textMatchesPattern(value: string, pattern: string): boolean {
  const haystack = normalizeMatchText(value);
  const needle = normalizeMatchText(pattern);
  if (!needle) return false;

  if (needle.length <= 3 || TOKEN_ONLY_PATTERNS.has(needle)) {
    return haystack.split("_").includes(needle);
  }

  return haystack.includes(needle);
}

function aliasesForPattern(pattern: string): readonly string[] {
  const normalized = normalizeMatchText(pattern);
  return PATTERN_ALIASES[normalized] ?? [pattern];
}

function matchedFocusTerms(haystack: string, focusArea: ProductFocusArea): string[] {
  return focusArea.raw_patterns.filter((pattern) => aliasesForPattern(pattern).some((alias) => textMatchesPattern(haystack, alias)));
}

function sourceName(item: DocumentListItem): string {
  return item.speaker || item.organization || item.source_kind || "News";
}

function excerptForDocument(item: DocumentListItem, summary: string, fullText: string): string {
  if (summary.trim()) {
    return summary.trim();
  }

  const sentence = fullText
    .replace(/\s+/g, " ")
    .split(/(?<=[.!?])\s+/)
    .find((part) => part.length > 60);

  return sentence?.slice(0, 240) ?? `${sourceName(item)} article matched to the selected AML focus.`;
}

function relatedThemesForFocusArea(focusArea: ProductFocusArea): NormalizedTheme[] {
  if (focusArea.id === "aml_sanctions") {
    return ["SANCTIONS"];
  }
  return ["REGULATION"];
}

function storedArticleId(category: ProductCategory, documentId: string): string {
  const hash = createHash("sha1").update(`${category}:${documentId}`).digest("hex").slice(0, 12);
  return `stored-${category.toLowerCase()}-${hash}`;
}

function parseComparableDate(value: string): number {
  const parsed = new Date(value);
  const ms = parsed.getTime();
  return Number.isNaN(ms) ? 0 : ms;
}

function recencyScore(item: DocumentListItem): number {
  const dateMs = parseComparableDate(item.published_at || item.date);
  if (!dateMs) return 0;

  const ageDays = Math.max(0, (Date.now() - dateMs) / 86_400_000);
  return Math.max(0, 20 - Math.min(20, ageDays));
}

function buildMatchHaystack(item: DocumentListItem, fullText: string): string {
  return [
    item.title,
    item.url,
    item.speaker,
    item.organization,
    item.doc_type,
    ...(item.tags || []),
    ...(item.topics || []),
    ...(item.keywords || []),
    fullText
  ].join("\n");
}

function mapStoredArticleToEvidence(
  category: ProductCategory,
  item: DocumentListItem,
  focusArea: ProductFocusArea,
  matchedTerms: readonly string[],
  fullText: string,
  summary: string,
  index: number
): IntelligenceEvidenceArticle {
  return {
    id: storedArticleId(category, item.document_id),
    headline: item.title || `${sourceName(item)} article`,
    url: item.url,
    source: sourceName(item),
    timestamp: item.published_at || item.date || "Stored news",
    excerpt: excerptForDocument(item, summary, fullText),
    explanation: `Matched ${focusArea.label}: ${matchedTerms.join(", ")}`,
    relatedThemes: relatedThemesForFocusArea(focusArea),
    matchedTerms: [...matchedTerms],
    focusAreaId: focusArea.id,
    focusAreaLabel: focusArea.label,
    clusterId: `${category.toLowerCase()}-${focusArea.id}`,
    credibility: Math.max(60, 92 - index),
    impact: Math.max(55, 90 - index)
  };
}

function enrichmentSummaryById(enrichment: EnrichmentStatePayload): Map<string, string> {
  const summaries = new Map<string, string>();
  for (const [docId, entry] of Object.entries(enrichment.entries || {})) {
    const summary = String(entry.enrichment?.summary || "").trim();
    if (summary) {
      summaries.set(docId, summary);
    }
  }
  return summaries;
}

export function buildFullTextByDocumentId(corpusDocs: readonly CustomDocumentRecord[]): Map<string, string> {
  const fullTextById = new Map<string, string>();

  for (const doc of corpusDocs) {
    const docId = String(doc.metadata?.document_id || "").trim();
    if (!docId) continue;
    fullTextById.set(docId, String(doc.content?.full_text || ""));
  }

  return fullTextById;
}

export function mapStoredDocumentsToProductCategoryEvidence(
  category: ProductCategory,
  items: readonly DocumentListItem[],
  fullTextById: ReadonlyMap<string, string>,
  summaryById: ReadonlyMap<string, string> = new Map(),
  focusAreas: readonly ProductFocusArea[] = focusAreasForProductCategory(category)
): IntelligenceEvidenceArticle[] {
  const ranked: {
    item: DocumentListItem;
    focusArea: ProductFocusArea;
    matchedTerms: string[];
    fullText: string;
    summary: string;
    score: number;
  }[] = [];

  for (const item of items) {
    if (item.source_kind !== "newsapi_article") {
      continue;
    }

    const fullText = fullTextById.get(item.document_id) || "";
    const haystack = buildMatchHaystack(item, fullText);
    const matches = focusAreas
      .map((focusArea) => ({ focusArea, matchedTerms: matchedFocusTerms(haystack, focusArea) }))
      .filter((match) => match.matchedTerms.length > 0)
      .sort((a, b) => b.matchedTerms.length - a.matchedTerms.length || a.focusArea.label.localeCompare(b.focusArea.label));

    const bestMatch = matches[0];
    if (!bestMatch) {
      continue;
    }

    ranked.push({
      item,
      focusArea: bestMatch.focusArea,
      matchedTerms: bestMatch.matchedTerms,
      fullText,
      summary: summaryById.get(item.document_id) || "",
      score: bestMatch.matchedTerms.length * 40 + recencyScore(item)
    });
  }

  return ranked
    .sort((a, b) => b.score - a.score || parseComparableDate(b.item.published_at || b.item.date) - parseComparableDate(a.item.published_at || a.item.date))
    .slice(0, STORED_EVIDENCE_MAX_RECORDS)
    .map((match, index) =>
      mapStoredArticleToEvidence(
        category,
        match.item,
        match.focusArea,
        match.matchedTerms,
        match.fullText,
        match.summary,
        index
      )
    );
}

export async function fetchStoredEvidenceForProductCategory(
  category: ProductCategory,
  focusId?: string | null
): Promise<IntelligenceEvidenceArticle[]> {
  const focusAreas = focusAreasForProductCategory(category).filter((focusArea) => !focusId || focusArea.id === focusId);
  if (focusAreas.length === 0) {
    return [];
  }

  const cacheKey = `${category}:${focusId ?? "all"}`;
  const cached = storedEvidenceCache.get(cacheKey);
  if (cached && Date.now() - cached.loadedAt < STORED_EVIDENCE_CACHE_TTL_MS) {
    return cached.articles;
  }

  const { buildDocumentListItems, loadCorpusDocuments, loadEnrichmentState } = await import("./data-store.ts");
  const [corpusDocs, enrichment] = await Promise.all([loadCorpusDocuments(), loadEnrichmentState()]);
  const items = buildDocumentListItems(corpusDocs, enrichment);
  const articles = mapStoredDocumentsToProductCategoryEvidence(
    category,
    items,
    buildFullTextByDocumentId(corpusDocs),
    enrichmentSummaryById(enrichment),
    focusAreas
  );

  storedEvidenceCache.set(cacheKey, { loadedAt: Date.now(), articles });
  return articles;
}

import type { CustomDocumentRecord, DocumentListItem } from "@/lib/server/types";
import { normalizeText } from "@/lib/server/api-utils";
import { parseComparableDate } from "@/lib/server/data-store";

export interface DocumentListFilters {
  q?: string;
  org?: string;
  sourceKind?: string;
  topic?: string;
  keyword?: string;
  tag?: string;
  status?: string;
  fromDate?: Date | null;
  toDate?: Date | null;
}

export function normalizeFacetToken(value: string): string {
  return normalizeText(value)
    .toLowerCase()
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function buildFullTextById(corpusDocs: CustomDocumentRecord[]): Map<string, string> {
  const fullTextById = new Map<string, string>();

  for (const doc of corpusDocs) {
    const docId = String(doc.metadata?.document_id || "").trim();
    if (!docId) {
      continue;
    }
    fullTextById.set(docId, String(doc.content?.full_text || "").toLowerCase());
  }

  return fullTextById;
}

export function filterDocumentListItems(
  items: DocumentListItem[],
  fullTextById: Map<string, string>,
  filters: DocumentListFilters
): DocumentListItem[] {
  const q = normalizeText(filters.q).toLowerCase();
  const org = normalizeText(filters.org);
  const sourceKind = normalizeText(filters.sourceKind);
  const topic = normalizeFacetToken(filters.topic || "");
  const keyword = normalizeFacetToken(filters.keyword || "");
  const tag = normalizeFacetToken(filters.tag || "");
  const status = normalizeText(filters.status);
  const fromDate = filters.fromDate || null;
  const toDate = filters.toDate || null;

  return items.filter((item) => {
    if (org && item.organization !== org) {
      return false;
    }
    if (sourceKind && item.source_kind !== sourceKind) {
      return false;
    }
    if (status && item.enrichment_status !== status) {
      return false;
    }
    if (topic) {
      const hasTopic = (item.topics || []).some((value) => {
        const token = normalizeFacetToken(value);
        return token === topic || token.includes(topic);
      });
      if (!hasTopic) {
        return false;
      }
    }
    if (keyword) {
      const hasKeyword = (item.keywords || []).some((value) => {
        const token = normalizeFacetToken(value);
        return token === keyword || token.includes(keyword);
      });
      if (!hasKeyword) {
        return false;
      }
    }
    if (tag) {
      const hasTag = (item.tags || []).some((value) => {
        const token = normalizeFacetToken(value);
        return token === tag || token.includes(tag);
      });
      if (!hasTag) {
        return false;
      }
    }

    const itemDateMs = parseComparableDate(item.published_at || item.date);
    if (fromDate && itemDateMs && itemDateMs < fromDate.getTime()) {
      return false;
    }
    if (toDate && itemDateMs && itemDateMs > toDate.getTime()) {
      return false;
    }

    if (!q) {
      return true;
    }

    const haystack = [
      item.title,
      item.organization,
      item.source_kind,
      item.doc_type,
      item.speaker,
      item.url,
      ...(item.tags || []),
      ...(item.topics || []),
      ...(item.keywords || []),
      fullTextById.get(item.document_id) || ""
    ]
      .join("\n")
      .toLowerCase();

    return haystack.includes(q);
  });
}

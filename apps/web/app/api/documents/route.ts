import {
  buildDocumentListItems,
  buildDocumentsFacets,
  loadCorpusDocuments,
  loadEnrichmentState,
  parseComparableDate
} from "@/lib/server/data-store";
import { createRequestId, fail, normalizeText, ok, parseDate, toInt } from "@/lib/server/api-utils";

export const runtime = "nodejs";

function normalizeFacetToken(value: string): string {
  return normalizeText(value)
    .toLowerCase()
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export async function GET(request: Request) {
  const requestId = createRequestId();

  try {
    const url = new URL(request.url);
    const q = normalizeText(url.searchParams.get("q")).toLowerCase();
    const org = normalizeText(url.searchParams.get("org"));
    const sourceKind = normalizeText(url.searchParams.get("source_kind") || url.searchParams.get("source"));
    const topic = normalizeFacetToken(url.searchParams.get("topic") || "");
    const keyword = normalizeFacetToken(url.searchParams.get("keyword") || "");
    const status = normalizeText(url.searchParams.get("status"));
    const sort = normalizeText(url.searchParams.get("sort")) || "date_desc";

    const page = toInt(url.searchParams.get("page"), 1, 1, 99999);
    const pageSize = toInt(url.searchParams.get("page_size"), 25, 1, 100);
    const fromDate = parseDate(url.searchParams.get("date_from"));
    const toDate = parseDate(url.searchParams.get("date_to"));

    const [corpusDocs, enrichment] = await Promise.all([loadCorpusDocuments(), loadEnrichmentState()]);
    const items = buildDocumentListItems(corpusDocs, enrichment);
    const facets = buildDocumentsFacets(items);

    const fullTextById = new Map<string, string>();
    for (const doc of corpusDocs) {
      const docId = String(doc.metadata?.document_id || "").trim();
      if (!docId) {
        continue;
      }
      fullTextById.set(docId, String(doc.content?.full_text || "").toLowerCase());
    }

    let filtered = items.filter((item) => {
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

    const sorters: Record<string, (a: (typeof filtered)[number], b: (typeof filtered)[number]) => number> = {
      date_desc: (a, b) => parseComparableDate(b.published_at || b.date) - parseComparableDate(a.published_at || a.date),
      date_asc: (a, b) => parseComparableDate(a.published_at || a.date) - parseComparableDate(b.published_at || b.date),
      updated_desc: (a, b) => parseComparableDate(b.updated_at) - parseComparableDate(a.updated_at)
    };

    filtered = filtered.sort(sorters[sort] || sorters.date_desc);

    const total = filtered.length;
    const start = (page - 1) * pageSize;
    const end = start + pageSize;

    const payload = {
      items: filtered.slice(start, end),
      page,
      page_size: pageSize,
      total,
      facets
    };

    return ok(payload, requestId);
  } catch (error) {
    return fail(
      `Failed to list documents: ${error instanceof Error ? error.message : "Unknown error"}`,
      "DOCUMENT_LIST_FAILED",
      500,
      requestId
    );
  }
}

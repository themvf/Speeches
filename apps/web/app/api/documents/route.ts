import {
  buildDocumentListItems,
  buildDocumentsFacets,
  loadCorpusDocuments,
  loadEnrichmentState,
  parseComparableDate
} from "@/lib/server/data-store";
import { buildFullTextById, filterDocumentListItems, normalizeFacetToken } from "@/lib/server/document-query";
import { createRequestId, fail, normalizeText, ok, parseDate, toInt } from "@/lib/server/api-utils";

export const runtime = "nodejs";

export async function GET(request: Request) {
  const requestId = createRequestId();

  try {
    const url = new URL(request.url);
    const q = normalizeText(url.searchParams.get("q")).toLowerCase();
    const org = normalizeText(url.searchParams.get("org"));
    const sourceKind = normalizeText(url.searchParams.get("source_kind") || url.searchParams.get("source"));
    const topic = normalizeFacetToken(url.searchParams.get("topic") || "");
    const keyword = normalizeFacetToken(url.searchParams.get("keyword") || "");
    const tag = normalizeFacetToken(url.searchParams.get("tag") || "");
    const status = normalizeText(url.searchParams.get("status"));
    const sort = normalizeText(url.searchParams.get("sort")) || "date_desc";

    const page = toInt(url.searchParams.get("page"), 1, 1, 99999);
    const pageSize = toInt(url.searchParams.get("page_size"), 25, 1, 100);
    const fromDate = parseDate(url.searchParams.get("date_from"));
    const toDate = parseDate(url.searchParams.get("date_to"));

    const [corpusDocs, enrichment] = await Promise.all([loadCorpusDocuments(), loadEnrichmentState()]);
    const items = buildDocumentListItems(corpusDocs, enrichment);
    const facets = buildDocumentsFacets(items);
    const fullTextById = buildFullTextById(corpusDocs);

    let filtered = filterDocumentListItems(items, fullTextById, {
      q,
      org,
      sourceKind,
      topic,
      keyword,
      tag,
      status,
      fromDate,
      toDate
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

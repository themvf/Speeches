import {
  loadDocumentListPageFromNeon,
} from "@/lib/server/data-store";
import { normalizeFacetToken } from "@/lib/server/document-query";
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
    const hasDocumentIdsFilter = url.searchParams.has("doc_ids");
    const docIdsParam = normalizeText(url.searchParams.get("doc_ids"));
    const documentIds = docIdsParam
      ? docIdsParam.split(",").slice(0, 100).map((s) => s.trim()).filter(Boolean)
      : [];

    let result;
    try {
      result = await loadDocumentListPageFromNeon({
        q: hasDocumentIdsFilter ? "" : q,
        organization: org,
        sourceKind,
        topic,
        keyword,
        tag,
        status,
        fromDate,
        toDate,
        documentIds,
        hasDocumentIdsFilter,
        sort: ["date_asc", "updated_desc"].includes(sort)
          ? sort as "date_asc" | "updated_desc"
          : "date_desc",
        page,
        pageSize,
      });
    } catch (error) {
      console.error("Failed to load Neon document projection", { requestId, error });
      return fail(
        "Document corpus is temporarily unavailable.",
        "DOCUMENT_CORPUS_UNAVAILABLE",
        503,
        requestId
      );
    }

    const payload = {
      items: result.items,
      page,
      page_size: pageSize,
      total: result.total,
      facets: result.facets,
      warnings: result.warnings
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

import { fullTextToDocumentContent, projectionRowsToEnrichmentState } from "@/lib/server/document-metadata-feed";
import { createRequestId, fail, normalizeText, ok } from "@/lib/server/api-utils";
import { getMirroredDocumentDetail } from "@/lib/server/neon";

export const runtime = "nodejs";

export async function GET(
  _request: Request,
  context: { params: Promise<{ documentId: string }> }
) {
  const requestId = createRequestId();

  try {
    const { documentId } = await context.params;
    const docId = normalizeText(documentId);
    if (!docId) {
      return fail("Document ID is required.", "DOCUMENT_ID_REQUIRED", 400, requestId);
    }

    const row = await getMirroredDocumentDetail(docId);
    if (!row) {
      return fail("Document not found.", "DOCUMENT_NOT_FOUND", 404, requestId);
    }

    const metadata = row.metadata && typeof row.metadata === "object" ? row.metadata : {};
    const enrichmentState = projectionRowsToEnrichmentState([row]);
    const enrichEntry = enrichmentState.entries?.[docId];
    const content = fullTextToDocumentContent(row.full_text);

    const payload = {
      metadata: {
        ...metadata,
        document_id: docId,
        published_at: String(metadata.published_at || metadata.published_date || metadata.date || "")
      },
      content,
      enrichment: {
        status: String(enrichEntry?.status || "not_enriched"),
        model: String(enrichEntry?.model || ""),
        summary: String(enrichEntry?.enrichment?.summary || ""),
        tags: Array.isArray(enrichEntry?.enrichment?.tags) ? enrichEntry?.enrichment?.tags : [],
        keywords: Array.isArray(enrichEntry?.enrichment?.keywords) ? enrichEntry?.enrichment?.keywords : [],
        entities: Array.isArray(enrichEntry?.enrichment?.entities) ? enrichEntry?.enrichment?.entities : [],
        evidence_spans: Array.isArray(enrichEntry?.enrichment?.evidence_spans)
          ? enrichEntry?.enrichment?.evidence_spans
          : [],
        stance:
          enrichEntry?.enrichment?.stance && typeof enrichEntry?.enrichment?.stance === "object"
            ? enrichEntry?.enrichment?.stance
            : {},
        comment_position:
          enrichEntry?.enrichment?.comment_position && typeof enrichEntry?.enrichment?.comment_position === "object"
            ? enrichEntry?.enrichment?.comment_position
            : {},
        confidence: Number.parseFloat(String(enrichEntry?.enrichment?.confidence ?? "0")) || 0
      },
      review: {
        decision: String(enrichEntry?.review?.decision || "pending"),
        notes: String(enrichEntry?.review?.notes || ""),
        reviewed_at: String(enrichEntry?.review?.reviewed_at || "")
      },
      sentiment: enrichEntry?.sentiment
        ? {
            score: Number(enrichEntry.sentiment.score ?? 0),
            label: String(enrichEntry.sentiment.label || "neutral"),
            rationale: String(enrichEntry.sentiment.rationale || ""),
            status: String(enrichEntry.sentiment.status || ""),
          }
        : null,
    };

    return ok(payload, requestId);
  } catch (error) {
    console.error("Failed to load Neon document detail", { requestId, error });
    return fail(
      "Failed to load document detail.",
      "DOCUMENT_DETAIL_FAILED",
      500,
      requestId
    );
  }
}

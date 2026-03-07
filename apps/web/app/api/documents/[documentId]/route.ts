import { loadCorpusDocuments, loadEnrichmentState } from "@/lib/server/data-store";
import { createRequestId, fail, normalizeText, ok } from "@/lib/server/api-utils";

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

    const [corpus, enrichmentState] = await Promise.all([loadCorpusDocuments(), loadEnrichmentState()]);

    const doc = corpus.find((item) => String(item.metadata?.document_id || "").trim() === docId);
    if (!doc) {
      return fail("Document not found.", "DOCUMENT_NOT_FOUND", 404, requestId);
    }

    const enrichEntry = enrichmentState.entries?.[docId];

    const payload = {
      metadata: {
        ...(doc.metadata || {}),
        document_id: docId,
        published_at: String(doc.metadata?.published_date || doc.metadata?.date || "")
      },
      content: {
        full_text: String(doc.content?.full_text || ""),
        paragraphs: Array.isArray(doc.content?.paragraphs) ? doc.content?.paragraphs : [],
        sentences: Array.isArray(doc.content?.sentences) ? doc.content?.sentences : []
      },
      enrichment: {
        status: String(enrichEntry?.status || "not_enriched"),
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
      }
    };

    return ok(payload, requestId);
  } catch (error) {
    return fail(
      `Failed to load document detail: ${error instanceof Error ? error.message : "Unknown error"}`,
      "DOCUMENT_DETAIL_FAILED",
      500,
      requestId
    );
  }
}

import { createRequestId, fail, normalizeText, ok } from "@/lib/server/api-utils";
import { loadCorpusDocuments } from "@/lib/server/data-store";
import { generateEnforcementAnalysis } from "@/lib/server/enforcement-analysis";
import { getClientIp, getGenerateGlobalLimiter, getGenerateIpLimiter, isRateLimited } from "@/lib/server/rate-limit";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  const requestId = createRequestId();
  const ip = getClientIp(request.headers);
  if (await isRateLimited(getGenerateIpLimiter(), ip) || await isRateLimited(getGenerateGlobalLimiter(), "global")) {
    return fail("Rate limit exceeded. Please slow down.", "RATE_LIMITED", 429, requestId);
  }

  try {
    const body = await request.json().catch(() => ({})) as Record<string, unknown>;
    const documentId = normalizeText(body.document_id);
    if (!documentId) {
      return fail("document_id is required.", "DOCUMENT_ID_REQUIRED", 400, requestId);
    }

    const corpus = await loadCorpusDocuments();
    const doc = corpus.find((item) => normalizeText(item.metadata?.document_id) === documentId);
    if (!doc) {
      return fail("Document not found.", "DOCUMENT_NOT_FOUND", 404, requestId);
    }

    const { model, analysis } = await generateEnforcementAnalysis(doc, normalizeText(body.model));
    return ok({
      document_id: documentId,
      generated_at: new Date().toISOString(),
      model,
      analysis,
    }, requestId);
  } catch (error) {
    console.error("[enforcement/analyze]", error);
    return fail(
      `Failed to generate enforcement analysis: ${error instanceof Error ? error.message : "Unknown error"}`,
      "ENFORCEMENT_ANALYSIS_FAILED",
      500,
      requestId
    );
  }
}

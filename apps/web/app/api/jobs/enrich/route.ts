import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import { triggerEnrichJob } from "@/lib/server/github-actions";

export const runtime = "nodejs";

export async function POST(request: Request) {
  const requestId = createRequestId();

  try {
    let body: Record<string, unknown> = {};
    try {
      body = (await request.json()) as Record<string, unknown>;
    } catch {
      body = {};
    }

    const limit = Math.max(1, Number.parseInt(String(body.limit ?? "25"), 10) || 25);
    const modeRaw = String(body.mode ?? "only_missing_or_failed").trim();
    const mode = ["all", "only_missing_or_failed"].includes(modeRaw) ? modeRaw : "only_missing_or_failed";
    const sourceKind = String(body.source_kind ?? "newsapi_article").trim() || "newsapi_article";
    const heuristicOnly = String(body.heuristic_only ?? "false").toLowerCase() === "true";
    const model = String(body.model ?? "").trim();

    const payload = await triggerEnrichJob({
      limit,
      mode,
      sourceKind,
      heuristicOnly,
      model
    });

    return ok(payload, requestId);
  } catch (error) {
    return fail(
      `Failed to trigger enrichment job: ${error instanceof Error ? error.message : "Unknown error"}`,
      "JOB_ENRICH_TRIGGER_FAILED",
      500,
      requestId
    );
  }
}
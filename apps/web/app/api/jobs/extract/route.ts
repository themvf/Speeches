import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import { getJobExecutionMode } from "@/lib/server/env";
import { triggerExtractJob } from "@/lib/server/github-actions";
import { runLocalExtractJob } from "@/lib/server/local-extract";

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

    const connectorRaw = String(body.connector ?? "sec_speech").trim();
    const connector = connectorRaw || "sec_speech";
    const selectionRaw = String(body.selection ?? "new_or_updated").trim();
    const selection = ["new_or_updated", "all"].includes(selectionRaw) ? selectionRaw : "new_or_updated";
    const limit = Math.max(1, Number.parseInt(String(body.limit ?? "25"), 10) || 25);
    const maxPages = Math.max(1, Number.parseInt(String(body.max_pages ?? "5"), 10) || 5);
    const baseUrl = String(body.base_url ?? "").trim();
    const includePdfs = String(body.include_pdfs ?? "true").toLowerCase() === "true";
    const includeRss = String(body.include_rss ?? "true").toLowerCase() === "true";

    const executionMode = getJobExecutionMode();
    if (executionMode === "local") {
      const payload = await runLocalExtractJob({
        connector,
        selection,
        limit,
        maxPages,
        baseUrl,
        includePdfs,
        includeRss
      });
      return ok(payload, requestId);
    }

    const payload = await triggerExtractJob({
      connector,
      selection,
      limit,
      maxPages,
      baseUrl,
      includePdfs,
      includeRss
    });

    return ok(payload, requestId);
  } catch (error) {
    return fail(
      `Failed to trigger extraction job: ${error instanceof Error ? error.message : "Unknown error"}`,
      "JOB_EXTRACT_TRIGGER_FAILED",
      500,
      requestId
    );
  }
}

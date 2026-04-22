import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import { triggerIngestJob } from "@/lib/server/github-actions";

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

    const limit = Math.max(1, Number.parseInt(String(body.limit ?? "10"), 10) || 10);
    const lookbackDays = Math.max(1, Number.parseInt(String(body.lookback_days ?? "7"), 10) || 7);
    const selectionRaw = String(body.selection ?? "new_or_updated").trim();
    const selection = ["new_or_updated", "all"].includes(selectionRaw) ? selectionRaw : "new_or_updated";
    const query = String(body.query ?? "").trim();
    const maxPages = Math.max(0, Number.parseInt(String(body.max_pages ?? "0"), 10) || 0);
    const pageSize = Math.max(0, Number.parseInt(String(body.page_size ?? "0"), 10) || 0);
    const targetCount = Math.max(0, Number.parseInt(String(body.target_count ?? "0"), 10) || 0);
    const domains = String(body.domains ?? "").trim();
    const tagsCsv = String(body.tags_csv ?? "").trim();

    const payload = await triggerIngestJob({
      limit,
      lookbackDays,
      selection,
      query,
      maxPages,
      pageSize,
      targetCount,
      domains,
      tagsCsv
    });

    return ok(payload, requestId);
  } catch (error) {
    return fail(
      `Failed to trigger ingest job: ${error instanceof Error ? error.message : "Unknown error"}`,
      "JOB_INGEST_TRIGGER_FAILED",
      500,
      requestId
    );
  }
}

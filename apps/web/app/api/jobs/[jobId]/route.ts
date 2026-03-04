import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import { getJobSummary } from "@/lib/server/github-actions";

export const runtime = "nodejs";

export async function GET(
  _request: Request,
  context: { params: Promise<{ jobId: string }> }
) {
  const requestId = createRequestId();

  try {
    const { jobId } = await context.params;
    const payload = await getJobSummary(jobId);
    return ok(payload, requestId);
  } catch (error) {
    return fail(
      `Failed to fetch job status: ${error instanceof Error ? error.message : "Unknown error"}`,
      "JOB_STATUS_FAILED",
      500,
      requestId
    );
  }
}
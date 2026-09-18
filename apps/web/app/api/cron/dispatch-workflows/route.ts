import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";
import { checkCronAuth, ok, fail } from "@/lib/server/api-utils";
import { DISPATCH_TARGETS, dispatchIfDue, GITHUB_REPO } from "@/lib/server/github-dispatch";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";
export const maxDuration = 30;

// Presses workflow_dispatch on the workflows whose value depends on cadence. See github-dispatch.ts
// for why GitHub's own schedule trigger is not enough. Vercel cron sends CRON_SECRET as a bearer
// token automatically, so checkCronAuth (which already accepts it) needs no extra wiring.
async function run(req: NextRequest): Promise<NextResponse> {
  const auth = checkCronAuth(req);
  if (!auth.ok) return fail(auth.error, "UNAUTHORIZED", auth.status);

  const token = process.env.GITHUB_DISPATCH_TOKEN ?? "";
  if (!token) {
    // Visible rather than silent: without the token this endpoint does nothing at all, and a quiet
    // no-op would look identical to a healthy run.
    return fail(
      "GITHUB_DISPATCH_TOKEN is not set; no workflow can be dispatched.",
      "DISPATCH_NOT_CONFIGURED",
      503,
    );
  }

  const results = await Promise.all(
    DISPATCH_TARGETS.map((target) => dispatchIfDue(target, { token, repo: GITHUB_REPO })),
  );
  const failures = results.filter((r) => r.status === "failed");
  return ok({
    repo: GITHUB_REPO,
    ranAt: new Date().toISOString(),
    dispatched: results.filter((r) => r.status === "dispatched").length,
    skipped: results.filter((r) => r.status === "skipped").length,
    failed: failures.length,
    results,
    warnings: failures.map((r) => `${r.workflow}: ${r.detail}`),
  });
}

export async function GET(req: NextRequest): Promise<NextResponse> {
  return run(req);
}

export async function POST(req: NextRequest): Promise<NextResponse> {
  return run(req);
}

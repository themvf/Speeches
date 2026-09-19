import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";
import { checkCronAuth, ok, fail } from "@/lib/server/api-utils";
import { getGithubActionsConfig } from "@/lib/server/env";
import { DISPATCH_TARGETS, dispatchIfDue } from "@/lib/server/github-dispatch";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";
export const maxDuration = 30;

// Presses workflow_dispatch on the workflows whose value depends on cadence. See github-dispatch.ts
// for why GitHub's own schedule trigger is not enough. Uses the admin job-runner's existing GitHub
// credentials, so no new secret. Vercel cron sends CRON_SECRET as a bearer token automatically, and
// checkCronAuth already accepts it.
async function run(req: NextRequest): Promise<NextResponse> {
  const auth = checkCronAuth(req);
  if (!auth.ok) return fail(auth.error, "UNAUTHORIZED", auth.status);

  const cfg = getGithubActionsConfig();
  if (!cfg.enabled) {
    // Visible rather than silent: unconfigured, this endpoint does nothing at all, and a quiet no-op
    // would look identical to a healthy run.
    const why = cfg.missingRequiredEnv.length
      ? `missing ${cfg.missingRequiredEnv.join(", ")}`
      : "GITHUB_ACTIONS_ENABLED is false";
    return fail(
      `GitHub Actions dispatch is not configured (${why}); no workflow can be dispatched.`,
      "DISPATCH_NOT_CONFIGURED",
      503,
    );
  }

  const results = await Promise.all(DISPATCH_TARGETS.map((target) => dispatchIfDue(target)));
  const failures = results.filter((r) => r.status === "failed");
  return ok({
    repo: `${cfg.owner}/${cfg.repo}`,
    ref: cfg.ref,
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

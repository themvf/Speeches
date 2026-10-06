import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";
import { checkCronAuth, ok, fail } from "@/lib/server/api-utils";
import { getGithubActionsConfig } from "@/lib/server/env";
import { WINDOW_MINUTES, dispatchTick, inWindow } from "@/lib/server/github-dispatch";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";
export const maxDuration = 30;

// Presses workflow_dispatch on every scheduled database job, but only in the first WINDOW_MINUTES of
// each UTC hour so the RegIntel Neon database can suspend in between. See github-dispatch.ts. Uses the admin job-runner's existing GitHub
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

  const now = new Date();
  if (!inWindow(now)) {
    // Outside the window: no GitHub calls and no dispatches, by design.
    return ok({ ranAt: now.toISOString(), window: `minutes 0-${WINDOW_MINUTES - 1} UTC`, inWindow: false, dispatched: 0, skipped: 0, failed: 0, results: [], warnings: [] });
  }
  const results = await dispatchTick(undefined, { now });
  const failures = results.filter((r) => r.status === "failed");
  return ok({
    repo: `${cfg.owner}/${cfg.repo}`,
    ref: cfg.ref,
    ranAt: now.toISOString(),
    inWindow: true,
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

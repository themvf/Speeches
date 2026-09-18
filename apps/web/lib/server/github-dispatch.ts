// Dispatch GitHub Actions workflows from a Vercel cron.
//
// Why this exists: GitHub treats the `schedule` event as best effort and drops most fires in a repo
// with many scheduled workflows. Measured 2026-09-18, two independent hourly workflows in this repo
// (crypto-social-rolling, bloomberg-public-hourly) each ran 6 or fewer times in 20 hours, with gaps of
// 2.5 to 5.5 hours. `workflow_dispatch` is an explicit API call and is not throttled that way, so a
// reliable scheduler pressing the button gives the cadence the cron string asks for.
//
// The workflows themselves are unchanged: same runner, same Python, same secrets, same pre-flight test
// gate. Their `schedule:` triggers stay in place as a fallback. That is safe because we read the real
// last-run time before dispatching, so a schedule fire and a Vercel tick never stack up.
export type DispatchTarget = {
  workflow: string; // workflow file name, e.g. "crypto-social-rolling.yml"
  everyMinutes: number; // dispatch only if the last run is at least this old
  reason: string; // why this one needs a reliable trigger; surfaced in the response
};

// Only workflows whose value genuinely depends on cadence belong here. Everything else can ride
// GitHub's own scheduler, where a few hours of drift costs nothing.
export const DISPATCH_TARGETS: DispatchTarget[] = [
  {
    workflow: "crypto-social-rolling.yml",
    everyMinutes: 60,
    reason: "ZCAT/ZEC/KNOTS open one search window an hour; missed fires leave windows pending",
  },
];

export type DispatchOutcome = {
  workflow: string;
  status: "dispatched" | "skipped" | "failed";
  detail: string;
  lastRunAt?: string | null;
};

export type FetchLike = (url: string, init?: RequestInit) => Promise<Response>;

export const GITHUB_REPO = process.env.GITHUB_DISPATCH_REPO ?? "themvf/Speeches";
const API_VERSION = "2022-11-28";

function githubHeaders(token: string): Record<string, string> {
  return {
    Authorization: `Bearer ${token}`,
    Accept: "application/vnd.github+json",
    "X-GitHub-Api-Version": API_VERSION,
  };
}

/**
 * When the workflow last started, from GitHub itself rather than from stored state. Using the real
 * run history means a fire from GitHub's own `schedule` trigger counts, so the two schedulers can
 * coexist without doubling up. Returns null when there is no run history to read.
 */
export async function lastRunAt(
  workflow: string,
  options: { token: string; repo?: string; fetchImpl?: FetchLike },
): Promise<Date | null> {
  const repo = options.repo ?? GITHUB_REPO;
  const doFetch = options.fetchImpl ?? (fetch as FetchLike);
  const url = `https://api.github.com/repos/${repo}/actions/workflows/${workflow}/runs?per_page=1`;
  const response = await doFetch(url, { headers: githubHeaders(options.token) });
  if (!response.ok) throw new Error(`run history unavailable (HTTP ${response.status})`);
  const body = (await response.json()) as { workflow_runs?: { created_at?: string }[] };
  const created = body.workflow_runs?.[0]?.created_at;
  if (!created) return null;
  const parsed = new Date(created);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

/** True when the last run is old enough that this target is due. A null `last` always fires. */
export function isDue(target: DispatchTarget, last: Date | null, now: Date): boolean {
  if (!last) return true;
  const elapsedMinutes = (now.getTime() - last.getTime()) / 60_000;
  // A minute of slack: ticks are not perfectly on the second, and one landing at 59m59s should count
  // as due rather than waiting a whole extra period.
  return elapsedMinutes >= target.everyMinutes - 1;
}

/**
 * Run one target if it is due. Returns a described outcome rather than throwing, so one failing
 * workflow never hides the others in the same tick.
 */
export async function dispatchIfDue(
  target: DispatchTarget,
  options: { token: string; ref?: string; repo?: string; now?: Date; fetchImpl?: FetchLike },
): Promise<DispatchOutcome> {
  const repo = options.repo ?? GITHUB_REPO;
  const ref = options.ref ?? "main";
  const now = options.now ?? new Date();
  const doFetch = options.fetchImpl ?? (fetch as FetchLike);
  try {
    const last = await lastRunAt(target.workflow, { token: options.token, repo, fetchImpl: doFetch });
    const lastIso = last ? last.toISOString() : null;
    if (!isDue(target, last, now)) {
      const mins = last ? Math.round((now.getTime() - last.getTime()) / 60_000) : 0;
      return {
        workflow: target.workflow,
        status: "skipped",
        detail: `ran ${mins}m ago, under the ${target.everyMinutes}m period`,
        lastRunAt: lastIso,
      };
    }
    const url = `https://api.github.com/repos/${repo}/actions/workflows/${target.workflow}/dispatches`;
    const response = await doFetch(url, {
      method: "POST",
      headers: { ...githubHeaders(options.token), "Content-Type": "application/json" },
      body: JSON.stringify({ ref }),
    });
    // GitHub answers a successful dispatch with 204 No Content and an empty body.
    if (response.status === 204) {
      return { workflow: target.workflow, status: "dispatched", detail: `ref ${ref}`, lastRunAt: lastIso };
    }
    const body = await response.text().catch(() => "");
    return {
      workflow: target.workflow,
      status: "failed",
      detail: `HTTP ${response.status}${body ? `: ${body.slice(0, 200)}` : ""}`,
      lastRunAt: lastIso,
    };
  } catch (error) {
    return {
      workflow: target.workflow,
      status: "failed",
      detail: error instanceof Error ? error.message : "dispatch failed",
    };
  }
}

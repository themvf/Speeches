import { getGithubActionsConfig } from "./env.ts";

// Dispatch GitHub Actions workflows on a schedule, from a Vercel cron.
//
// Why this exists: GitHub treats the `schedule` event as best effort and drops most fires in a repo
// with many scheduled workflows. Measured 2026-09-18, two independent hourly workflows in this repo
// (crypto-social-rolling, bloomberg-public-hourly) each ran six or fewer times in twenty hours, with
// gaps of 2.5 to 5.5 hours. `workflow_dispatch` is an explicit API call and is not throttled that
// way, so a reliable scheduler pressing the button gives the cadence the cron string asks for.
//
// Credentials are the ones the admin job-runner already uses (`getGithubActionsConfig`:
// GITHUB_ACTIONS_TOKEN / GITHUB_REPO_OWNER / GITHUB_REPO_NAME / GITHUB_DEFAULT_REF), so this needs no
// new secret. The workflows themselves are untouched: same runner, Python, secrets and pre-flight
// test gate, and their `schedule:` triggers stay as a fallback. That is safe because we read the real
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
  {
    workflow: "launchpad-archive.yml",
    everyMinutes: 5,
    reason:
      "GeckoTerminal's new_pools feed reached back only 11 minutes at the launch rate measured 2026-09-19, so a missed sweep loses launches permanently",
  },
];

export type DispatchOutcome = {
  workflow: string;
  status: "dispatched" | "skipped" | "failed";
  detail: string;
  lastRunAt?: string | null;
};

export type FetchLike = (url: string, init?: RequestInit) => Promise<Response>;

type Creds = { token: string; owner: string; repo: string; ref: string };

const API_VERSION = "2022-11-28";

/** Credentials for the dispatch calls, defaulting to the admin job-runner's existing configuration. */
export function dispatchCredentials(overrides?: Partial<Creds>): Creds {
  const cfg = getGithubActionsConfig();
  return {
    token: overrides?.token ?? cfg.token,
    owner: overrides?.owner ?? cfg.owner,
    repo: overrides?.repo ?? cfg.repo,
    ref: overrides?.ref ?? cfg.ref,
  };
}

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
  creds: Creds,
  fetchImpl: FetchLike = fetch as FetchLike,
): Promise<Date | null> {
  const url = `https://api.github.com/repos/${creds.owner}/${creds.repo}/actions/workflows/${workflow}/runs?per_page=1`;
  const response = await fetchImpl(url, { headers: githubHeaders(creds.token) });
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
  options: { now?: Date; fetchImpl?: FetchLike } & Partial<Creds> = {},
): Promise<DispatchOutcome> {
  const creds = dispatchCredentials(options);
  const now = options.now ?? new Date();
  const doFetch = options.fetchImpl ?? (fetch as FetchLike);
  try {
    const last = await lastRunAt(target.workflow, creds, doFetch);
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
    const url = `https://api.github.com/repos/${creds.owner}/${creds.repo}/actions/workflows/${target.workflow}/dispatches`;
    const response = await doFetch(url, {
      method: "POST",
      headers: { ...githubHeaders(creds.token), "Content-Type": "application/json" },
      body: JSON.stringify({ ref: creds.ref }),
    });
    // GitHub answers a successful dispatch with 204 No Content and an empty body.
    if (response.status === 204) {
      return { workflow: target.workflow, status: "dispatched", detail: `ref ${creds.ref}`, lastRunAt: lastIso };
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

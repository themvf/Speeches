import { getGithubActionsConfig } from "./env.ts";

// Dispatch GitHub Actions workflows on a schedule, from a Vercel cron.
//
// Two jobs, in this order of importance:
//
// 1. Let the RegIntel Neon database sleep (2026-10-05). Neon suspends a compute only after about five
//    idle minutes, and jobs writing to it at scattered minutes (:00, :05, :17, :20, :37, ...) kept it
//    awake most of the day. Every database job is therefore started inside one window at the top of
//    each UTC hour (WINDOW_MINUTES), and nothing is dispatched for the rest of the hour. The database
//    wakes once per hour, does everything, and can suspend for roughly forty minutes.
//
// 2. Reliable cadence. GitHub treats the `schedule` event as best effort and drops most fires in this
//    repo (measured 2026-09-18: hourly workflows ran six or fewer times in twenty hours, with gaps of
//    2.5 to 5.5 hours). `workflow_dispatch` is an explicit API call and is not throttled that way.
//
// The workflows listed here have no `schedule:` trigger any more; a GitHub fire at a random minute
// would wake the database outside the window. Exception by design: macro-sync-watchdog.yml keeps its
// own GitHub schedule, because a watchdog started by this dispatcher could not notice this dispatcher
// stopping.
//
// Credentials are the ones the admin job-runner already uses (`getGithubActionsConfig`:
// GITHUB_ACTIONS_TOKEN / GITHUB_REPO_OWNER / GITHUB_REPO_NAME / GITHUB_DEFAULT_REF).

/** Dispatching happens only during minutes 0..WINDOW_MINUTES-1 of each UTC hour. */
export const WINDOW_MINUTES = 16;

/** UTC hours, optionally limited to UTC weekdays (0 = Sunday ... 6 = Saturday). */
export type Slot = { hoursUtc: number[]; daysUtc?: number[] };

export type DispatchTarget = {
  workflow: string; // workflow file name, e.g. "crypto-social-rolling.yml"
  reason: string; // why it runs on this cadence; surfaced in the response
  /**
   * The workflow's GitHub `concurrency` group. At most one run per lane is ever active: GitHub keeps
   * only one pending run per group and cancels the rest, so starting two lane-mates in the same
   * window would silently drop one. The dispatcher starts the next lane-mate on a later tick instead.
   */
  lane: string;
  /** Unpinned cadence in minutes (a multiple of 60). Exactly one of `everyMinutes` or `slots`. */
  everyMinutes?: number;
  /** Pinned UTC hours. A slot missed because the window ran out is caught up in the next window. */
  slots?: Slot[];
  /** workflow_dispatch inputs. Needed where a dispatched run's defaults differ from a scheduled run's. */
  inputs?: Record<string, string>;
};

const range = (start: number, end: number, step = 1) =>
  Array.from({ length: Math.floor((end - start) / step) + 1 }, (_, i) => start + i * step);
const WEEKDAYS = [1, 2, 3, 4, 5];
const TUE_TO_SAT = [2, 3, 4, 5, 6];
// Workflows whose scheduled runs are gated by a repository variable get `scheduled: "true"`. Their
// job-level `if` treats that input like a schedule event, so the variable still switches them off.
const AS_SCHEDULED = { scheduled: "true" };

// Order matters only within a lane: earlier targets start first.
export const DISPATCH_TARGETS: DispatchTarget[] = [
  // --- crypto-social-pilot lane: five workflows share one concurrency group ---
  {
    workflow: "crypto-social-rolling.yml",
    lane: "crypto-social-pilot",
    everyMinutes: 60,
    reason: "ZCAT/ZEC/KNOTS open one search window an hour; missed fires leave windows pending",
  },
  {
    workflow: "crypto-social-watchers.yml",
    lane: "crypto-social-pilot",
    slots: [{ hoursUtc: range(0, 22, 2) }],
    reason: "every two hours, as its former cron",
  },
  {
    workflow: "crypto-market-history.yml",
    lane: "crypto-social-pilot",
    slots: [{ hoursUtc: [2, 8, 14, 20] }],
    reason: "hourly candles for tracked coins, four times a day as its former cron",
  },
  {
    workflow: "crypto-social-history.yml",
    lane: "crypto-social-pilot",
    slots: [{ hoursUtc: [9] }],
    // A dispatched run defaults `execute` to false; a scheduled run executed.
    inputs: { execute: "true" },
    reason: "daily, formerly 08:43 UTC",
  },
  {
    workflow: "crypto-social-pons.yml",
    lane: "crypto-social-pilot",
    slots: [{ hoursUtc: [10] }],
    reason: "daily after the history pull, formerly 09:13 UTC",
  },
  // --- hourly collectors ---
  {
    workflow: "reddit-attention-sweep-hourly.yml",
    lane: "reddit-attention-sweep",
    everyMinutes: 60,
    reason: "hourly Reddit sweep feeding the Attention board",
  },
  // --- sec20-neon-corpus-writers lane ---
  {
    workflow: "bloomberg-public-hourly.yml",
    lane: "sec20-neon-corpus-writers",
    everyMinutes: 60,
    inputs: AS_SCHEDULED,
    reason: "hourly Bloomberg headlines; still switched by ENABLE_NEON_PILOT_SCHEDULES",
  },
  {
    workflow: "financial-news-daily.yml",
    lane: "sec20-neon-corpus-writers",
    slots: [{ hoursUtc: [8, 11, 12] }],
    // A dispatched run defaults ingest_limit to 10; a scheduled run had no limit.
    inputs: { ...AS_SCHEDULED, ingest_limit: "" },
    reason: "three times a day as its former crons; still switched by ENABLE_NEON_PILOT_SCHEDULES",
  },
  {
    workflow: "neon-legacy-compat-sync.yml",
    lane: "sec20-neon-corpus-writers",
    slots: [{ hoursUtc: [10] }],
    inputs: AS_SCHEDULED,
    reason: "daily, formerly 09:43 UTC; still switched by ENABLE_NEON_LEGACY_COMPAT_SYNC",
  },
  // --- single-workflow lanes ---
  {
    workflow: "intelligence-fusion.yml",
    lane: "intelligence-fusion-materializer",
    slots: [{ hoursUtc: range(0, 21, 3) }],
    reason:
      "every three hours: the 15-minute cron actually ran about 2.5 times a day, and each run is ~10 minutes of database work",
  },
  {
    workflow: "capital-formation-connectors.yml",
    lane: "capital-formation-connectors",
    slots: [{ hoursUtc: [0, 6, 12, 18] }],
    inputs: AS_SCHEDULED,
    reason: "every six hours as its former cron; still switched by ENABLE_CAPITAL_FORMATION_CONNECTORS",
  },
  {
    workflow: "polymarket-earnings-sync.yml",
    lane: "polymarket-earnings-sync",
    // 21:00 rather than 20:25 so the after-close run still follows the 4pm ET close.
    slots: [{ hoursUtc: [1, 13, 21] }],
    reason: "9am, after-close and evening ET passes",
  },
  {
    workflow: "polymarket-macro-sync.yml",
    lane: "polymarket-macro-sync",
    slots: [{ hoursUtc: [14, 19], daysUtc: WEEKDAYS }, { hoursUtc: [1] }],
    reason: "weekday 14:00/19:00 UTC passes plus a daily 01:00 pass, as its former crons",
  },
  {
    workflow: "polymarket-invariants.yml",
    lane: "polymarket-invariants",
    slots: [{ hoursUtc: [16] }],
    reason: "daily invariant audit",
  },
  {
    workflow: "rule-comment-ingest.yml",
    lane: "rule-comment-ingest",
    slots: [{ hoursUtc: [14] }],
    inputs: AS_SCHEDULED,
    reason: "daily, formerly 13:35 UTC; still switched by ENABLE_RULE_COMMENT_INGEST",
  },
  {
    workflow: "stock-attention-daily.yml",
    lane: "stock-attention-daily",
    // 01:00, not 00:00: the 00:00 Reddit sweep must finish before the previous UTC day is rolled up.
    slots: [{ hoursUtc: [1] }],
    reason: "aggregates the just-closed UTC day",
  },
  {
    workflow: "attention-outcomes-daily.yml",
    lane: "attention-outcomes-daily",
    slots: [{ hoursUtc: [2] }],
    inputs: AS_SCHEDULED,
    reason: "daily after the attention rollup; still switched by ENABLE_ATTENTION_OUTCOMES",
  },
  {
    workflow: "backpack-monitor.yml",
    lane: "backpack-daily-capture",
    slots: [{ hoursUtc: [1] }],
    reason: "daily Backpack capture, formerly 00:30 UTC",
  },
  {
    workflow: "bp-holder-intel.yml",
    lane: "bp-holder-intel",
    everyMinutes: 24 * 60,
    // A dispatch carries no inputs, so the workflow runs its default daily mode: holdings plus transaction history.
    reason:
      "the top-200 BP holders' holdings and trades have no other refresh; GitHub delays the daily Backpack job by hours and drops fires",
  },
  {
    workflow: "document-ticker-index.yml",
    lane: "document-ticker-index",
    slots: [{ hoursUtc: [6] }],
    // A dispatched run defaults dry_run to true; a scheduled run wrote.
    inputs: { dry_run: "false" },
    reason: "daily incremental ticker index, formerly 05:45 UTC",
  },
  {
    workflow: "rates-credit-daily.yml",
    lane: "rates-credit-daily",
    slots: [{ hoursUtc: [2], daysUtc: TUE_TO_SAT }],
    reason: "Tuesday-Saturday after the US session, formerly 01:30 UTC",
  },
  {
    workflow: "filing-catalyst-sync.yml",
    lane: "filing-catalyst-sync",
    slots: [
      { hoursUtc: range(10, 22, 2), daysUtc: WEEKDAYS },
      { hoursUtc: [0, 2], daysUtc: TUE_TO_SAT },
    ],
    inputs: { mode: "detect" },
    reason: "intraday 8-K/Form 4 detection during EDGAR hours",
  },
  {
    workflow: "filing-catalyst-sync.yml",
    lane: "filing-catalyst-sync",
    slots: [{ hoursUtc: [9], daysUtc: WEEKDAYS }],
    // The workflow chose reconcile mode from the cron string; a dispatch must say so.
    inputs: { mode: "reconcile" },
    reason: "daily reconcile of yesterday's form index",
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

// How many recent runs to search when the newest one was skipped. A daily target behind an hourly
// gated schedule sees at most 24 skipped fires per period, so 30 always reaches past the period.
const SKIPPED_LOOKBACK = 30;

type RunSummary = { created_at?: string; conclusion?: string | null; status?: string };

const ACTIVE_STATUSES = new Set(["queued", "in_progress", "waiting", "requested", "pending"]);

async function recentRuns(workflow: string, creds: Creds, fetchImpl: FetchLike, perPage: number): Promise<RunSummary[]> {
  const url = `https://api.github.com/repos/${creds.owner}/${creds.repo}/actions/workflows/${workflow}/runs?per_page=${perPage}&exclude_pull_requests=true`;
  const response = await fetchImpl(url, { headers: githubHeaders(creds.token) });
  if (!response.ok) throw new Error(`run history unavailable (HTTP ${response.status})`);
  const body = (await response.json()) as { workflow_runs?: RunSummary[] };
  return body.workflow_runs ?? [];
}

function startedAt(run: RunSummary | undefined): Date | null {
  if (!run?.created_at) return null;
  const parsed = new Date(run.created_at);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function isActive(run: RunSummary | undefined): boolean {
  if (!run) return false;
  if (run.status) return ACTIVE_STATUSES.has(run.status);
  return run.conclusion === null;
}

export type RunState = { last: Date | null; active: boolean };

/**
 * When the workflow last started and whether its newest run is still going, from GitHub itself.
 *
 * A run GitHub created only to skip (a gated `schedule` fire such as bp-holder-intel.yml's opt-in
 * hourly trigger) did no work, so it does not count. Counting it made a daily target look fresh
 * forever: from 2026-09-27 to 2026-10-02 the BP holdings refresh never ran. Queued and in-progress
 * runs do count, so a running job is never doubled.
 */
export async function runState(workflow: string, creds: Creds, fetchImpl: FetchLike = fetch as FetchLike): Promise<RunState> {
  const [latest] = await recentRuns(workflow, creds, fetchImpl, 1);
  if (latest?.conclusion !== "skipped") return { last: startedAt(latest), active: isActive(latest) };
  const real = (await recentRuns(workflow, creds, fetchImpl, SKIPPED_LOOKBACK)).find((r) => r.conclusion !== "skipped");
  // Every run in the window was skipped: the last real run is older than all of them, so treat it as due.
  return { last: startedAt(real), active: false };
}

export function inWindow(now: Date): boolean {
  return now.getUTCMinutes() < WINDOW_MINUTES;
}

function hourStart(now: Date): Date {
  const start = new Date(now);
  start.setUTCMinutes(0, 0, 0);
  return start;
}

function slotMatches(slots: Slot[], at: Date): boolean {
  return slots.some(
    (slot) => slot.hoursUtc.includes(at.getUTCHours()) && (!slot.daysUtc || slot.daysUtc.includes(at.getUTCDay())),
  );
}

/**
 * The instant a target's previous run must predate for it to be due now.
 *
 * Pinned targets: the start of the most recent matching slot, so a slot missed because its window
 * ran out (a busy lane, a Vercel outage) is caught up in the next window rather than skipped.
 * Cadence targets: the start of the current hour minus the period less one hour, so a run anywhere in
 * the previous due window counts and a lane delay never pushes the cadence later and later.
 */
export function dueBoundary(target: DispatchTarget, now: Date): Date | null {
  const start = hourStart(now);
  if (target.slots) {
    for (let back = 0; back <= 8 * 24; back += 1) {
      const candidate = new Date(start.getTime() - back * 3_600_000);
      if (slotMatches(target.slots, candidate)) return candidate;
    }
    return null;
  }
  const period = target.everyMinutes ?? 60;
  return new Date(start.getTime() - (period - 60) * 60_000);
}

/** True when the last real run predates this target's due boundary. No history always fires. */
export function isDue(target: DispatchTarget, last: Date | null, now: Date): boolean {
  const boundary = dueBoundary(target, now);
  if (!boundary) return false;
  if (!last) return true;
  return last.getTime() < boundary.getTime();
}

async function dispatch(target: DispatchTarget, creds: Creds, fetchImpl: FetchLike): Promise<{ ok: boolean; detail: string }> {
  const url = `https://api.github.com/repos/${creds.owner}/${creds.repo}/actions/workflows/${target.workflow}/dispatches`;
  const body: { ref: string; inputs?: Record<string, string> } = { ref: creds.ref };
  if (target.inputs && Object.keys(target.inputs).length) body.inputs = target.inputs;
  const response = await fetchImpl(url, {
    method: "POST",
    headers: { ...githubHeaders(creds.token), "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  // GitHub answers a successful dispatch with 204 No Content and an empty body.
  if (response.status === 204) return { ok: true, detail: `ref ${creds.ref}` };
  const text = await response.text().catch(() => "");
  return { ok: false, detail: `HTTP ${response.status}${text ? `: ${text.slice(0, 200)}` : ""}` };
}

/**
 * One cron tick. Outside the window it does nothing and makes no GitHub calls. Inside it, it reads
 * every target's run state, then starts each due target whose lane is idle, at most one per lane.
 * Every target gets a described outcome; one failure never hides the others.
 */
export async function dispatchTick(
  targets: DispatchTarget[] = DISPATCH_TARGETS,
  options: { now?: Date; fetchImpl?: FetchLike } & Partial<Creds> = {},
): Promise<DispatchOutcome[]> {
  const now = options.now ?? new Date();
  if (!inWindow(now)) return [];
  const creds = dispatchCredentials(options);
  const doFetch = options.fetchImpl ?? (fetch as FetchLike);

  const workflows = [...new Set(targets.map((t) => t.workflow))];
  const states = new Map<string, RunState | Error>();
  await Promise.all(
    workflows.map(async (workflow) => {
      try {
        states.set(workflow, await runState(workflow, creds, doFetch));
      } catch (error) {
        states.set(workflow, error instanceof Error ? error : new Error("run history unavailable"));
      }
    }),
  );

  // A lane is busy when any of its workflows has a run going, or when its history is unreadable
  // (we cannot prove it idle, and starting a second run could cancel a queued one).
  const busyLanes = new Set<string>();
  for (const target of targets) {
    const state = states.get(target.workflow);
    if (state instanceof Error || state?.active) busyLanes.add(target.lane);
  }

  const outcomes: DispatchOutcome[] = [];
  for (const target of targets) {
    const state = states.get(target.workflow);
    if (state instanceof Error) {
      outcomes.push({ workflow: target.workflow, status: "failed", detail: state.message });
      continue;
    }
    const lastIso = state?.last ? state.last.toISOString() : null;
    if (!isDue(target, state?.last ?? null, now)) {
      outcomes.push({ workflow: target.workflow, status: "skipped", detail: "not due", lastRunAt: lastIso });
      continue;
    }
    if (busyLanes.has(target.lane)) {
      outcomes.push({ workflow: target.workflow, status: "skipped", detail: `due; waiting for lane ${target.lane}`, lastRunAt: lastIso });
      continue;
    }
    try {
      const result = await dispatch(target, creds, doFetch);
      busyLanes.add(target.lane);
      outcomes.push({ workflow: target.workflow, status: result.ok ? "dispatched" : "failed", detail: result.detail, lastRunAt: lastIso });
    } catch (error) {
      outcomes.push({
        workflow: target.workflow,
        status: "failed",
        detail: error instanceof Error ? error.message : "dispatch failed",
        lastRunAt: lastIso,
      });
    }
  }
  return outcomes;
}

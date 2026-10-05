import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import {
  isDue,
  dueBoundary,
  dispatchTick,
  inWindow,
  runState,
  DISPATCH_TARGETS,
  WINDOW_MINUTES,
  type DispatchTarget,
  type FetchLike,
} from "./github-dispatch.ts";

// Credentials are passed explicitly here so the tests never depend on the ambient environment.
const CREDS = { token: "t", owner: "o", repo: "r", ref: "main" };
const at = (iso: string) => new Date(iso);
// Monday 2026-10-05, 12:04 UTC: inside the window.
const NOW = at("2026-10-05T12:04:00Z");

type Run = { created_at: string; conclusion?: string | null; status?: string };

// Serves per-workflow run histories, newest first, and records every call.
function github(histories: Record<string, Run[]>, dispatchStatus = 204) {
  const calls: string[] = [];
  const bodies: Record<string, unknown> = {};
  const fetchImpl: FetchLike = async (url, init) => {
    const workflow = /workflows\/([^/]+)\//.exec(url)?.[1] ?? "?";
    if (init?.method === "POST") {
      calls.push(`POST ${workflow}`);
      bodies[workflow] = JSON.parse(String(init.body));
      return new Response(dispatchStatus === 204 ? null : "nope", { status: dispatchStatus });
    }
    calls.push(`GET ${workflow}`);
    const perPage = Number(new URL(url).searchParams.get("per_page"));
    return new Response(JSON.stringify({ workflow_runs: (histories[workflow] ?? []).slice(0, perPage) }), { status: 200 });
  };
  return { calls, bodies, fetchImpl, posts: () => calls.filter((c) => c.startsWith("POST")) };
}

const hourly: DispatchTarget = { workflow: "a.yml", lane: "a", everyMinutes: 60, reason: "t" };
const done = (iso: string): Run => ({ created_at: iso, status: "completed", conclusion: "success" });

test("the window is the first WINDOW_MINUTES of each UTC hour", () => {
  assert.equal(WINDOW_MINUTES, 16);
  assert.equal(inWindow(at("2026-10-05T12:00:00Z")), true);
  assert.equal(inWindow(at("2026-10-05T12:15:59Z")), true);
  assert.equal(inWindow(at("2026-10-05T12:16:00Z")), false);
  assert.equal(inWindow(at("2026-10-05T12:45:00Z")), false);
});

test("outside the window a tick makes no GitHub calls at all", async () => {
  const gh = github({});
  const out = await dispatchTick([hourly], { ...CREDS, now: at("2026-10-05T12:30:00Z"), fetchImpl: gh.fetchImpl });
  assert.deepEqual(out, []);
  assert.deepEqual(gh.calls, []);
});

test("hourly cadence: a run in this hour's window counts, last hour's does not", () => {
  assert.equal(isDue(hourly, null, NOW), true, "no history always fires");
  assert.equal(isDue(hourly, at("2026-10-05T11:02:00Z"), NOW), true);
  assert.equal(isDue(hourly, at("2026-10-05T11:59:00Z"), NOW), true, "a stray run late last hour does not suppress this window");
  assert.equal(isDue(hourly, at("2026-10-05T12:01:00Z"), NOW), false);
});

test("a lane delay never drifts a cadence later", () => {
  const daily: DispatchTarget = { ...hourly, everyMinutes: 24 * 60 };
  // Yesterday's run started 12 minutes into its window because the lane was busy.
  assert.equal(isDue(daily, at("2026-10-04T12:12:00Z"), NOW), true);
  assert.equal(isDue(daily, at("2026-10-05T00:10:00Z"), NOW), false);
});

test("pinned slots fire in their hour and catch up a missed slot in the next window", () => {
  const pinned: DispatchTarget = { workflow: "p.yml", lane: "p", slots: [{ hoursUtc: [2, 8, 14, 20] }], reason: "t" };
  assert.equal(dueBoundary(pinned, NOW)?.toISOString(), "2026-10-05T08:00:00.000Z");
  assert.equal(isDue(pinned, at("2026-10-05T08:03:00Z"), NOW), false, "the 08:00 slot already ran");
  assert.equal(isDue(pinned, at("2026-10-05T02:05:00Z"), NOW), true, "the 08:00 slot was missed, so it runs now");
});

test("weekday slots skip weekends", () => {
  const weekdays: DispatchTarget = { workflow: "w.yml", lane: "w", slots: [{ hoursUtc: [9], daysUtc: [1, 2, 3, 4, 5] }], reason: "t" };
  // Monday 09:00 is the latest slot; Saturday and Sunday 09:00 are not slots.
  assert.equal(dueBoundary(weekdays, NOW)?.toISOString(), "2026-10-05T09:00:00.000Z");
  assert.equal(dueBoundary(weekdays, at("2026-10-04T09:05:00Z"))?.toISOString(), "2026-10-02T09:00:00.000Z");
});

test("a due target is dispatched and 204 is the success signal", async () => {
  const gh = github({ "a.yml": [done("2026-10-05T11:01:00Z")] });
  const [out] = await dispatchTick([hourly], { ...CREDS, now: NOW, fetchImpl: gh.fetchImpl });
  assert.equal(out.status, "dispatched");
  assert.equal(out.detail, "ref main");
  assert.deepEqual(gh.posts(), ["POST a.yml"]);
});

test("lane-mates start one at a time, and never while a lane-mate is running", async () => {
  const first: DispatchTarget = { workflow: "x.yml", lane: "shared", everyMinutes: 60, reason: "t" };
  const second: DispatchTarget = { workflow: "y.yml", lane: "shared", everyMinutes: 60, reason: "t" };
  const old = [done("2026-10-05T11:01:00Z")];

  const idle = github({ "x.yml": old, "y.yml": old });
  const out = await dispatchTick([first, second], { ...CREDS, now: NOW, fetchImpl: idle.fetchImpl });
  assert.deepEqual(idle.posts(), ["POST x.yml"], "only the first lane-mate starts this tick");
  assert.match(out[1].detail, /waiting for lane shared/);

  const running = github({ "x.yml": [{ created_at: "2026-10-05T12:02:00Z", status: "in_progress", conclusion: null }], "y.yml": old });
  await dispatchTick([first, second], { ...CREDS, now: NOW, fetchImpl: running.fetchImpl });
  assert.deepEqual(running.posts(), [], "a running lane-mate holds the lane");
});

test("an unreadable history fails that target and holds its lane, without hiding the others", async () => {
  const gh = github({ "b.yml": [done("2026-10-05T11:01:00Z")] });
  const broken: FetchLike = async (url, init) =>
    url.includes("/a.yml/") && init?.method !== "POST" ? new Response("no", { status: 401 }) : gh.fetchImpl(url, init);
  const other: DispatchTarget = { workflow: "b.yml", lane: "b", everyMinutes: 60, reason: "t" };
  const out = await dispatchTick([hourly, other], { ...CREDS, now: NOW, fetchImpl: broken });
  assert.equal(out[0].status, "failed");
  assert.match(out[0].detail, /run history unavailable \(HTTP 401\)/);
  assert.equal(out[1].status, "dispatched");
});

test("a rejected dispatch reports the status rather than throwing", async () => {
  const gh = github({}, 403);
  const [out] = await dispatchTick([hourly], { ...CREDS, now: NOW, fetchImpl: gh.fetchImpl });
  assert.equal(out.status, "failed");
  assert.match(out.detail, /HTTP 403: nope/);
});

test("inputs are sent with the dispatch", async () => {
  const withInputs: DispatchTarget = { ...hourly, inputs: { dry_run: "false" } };
  const gh = github({});
  await dispatchTick([withInputs], { ...CREDS, now: NOW, fetchImpl: gh.fetchImpl });
  assert.deepEqual(gh.bodies["a.yml"], { ref: "main", inputs: { dry_run: "false" } });
});

// bp-holder-intel.yml's hourly schedule is gated off, so GitHub creates a skipped run every few hours.
const skipped = (iso: string): Run => ({ created_at: iso, status: "completed", conclusion: "skipped" });

test("skipped schedule fires do not count as runs", async () => {
  const runs = [skipped("2026-10-05T11:20:00Z"), skipped("2026-10-05T08:20:00Z"), done("2026-10-04T06:00:00Z")];
  const state = await runState("w.yml", CREDS, github({ "w.yml": runs }).fetchImpl);
  assert.equal(state.last?.toISOString(), "2026-10-04T06:00:00.000Z");
  assert.equal(state.active, false);
});

test("a run in progress is active and is never doubled", async () => {
  const state = await runState("w.yml", CREDS, github({ "w.yml": [{ created_at: "2026-10-05T12:01:00Z", status: "queued", conclusion: null }] }).fetchImpl);
  assert.equal(state.active, true);
});

// --- the real target list ---

test("every target has exactly one cadence, a lane, and slot hours in range", () => {
  for (const t of DISPATCH_TARGETS) {
    assert.ok(t.lane, `${t.workflow} needs a lane`);
    assert.ok((t.everyMinutes === undefined) !== (t.slots === undefined), `${t.workflow}: exactly one of everyMinutes/slots`);
    if (t.everyMinutes !== undefined) assert.equal(t.everyMinutes % 60, 0, `${t.workflow}: everyMinutes must be whole hours`);
    for (const slot of t.slots ?? []) {
      for (const h of slot.hoursUtc) assert.ok(Number.isInteger(h) && h >= 0 && h <= 23, `${t.workflow}: hour ${h}`);
      for (const d of slot.daysUtc ?? []) assert.ok(Number.isInteger(d) && d >= 0 && d <= 6, `${t.workflow}: day ${d}`);
    }
  }
});

// Paths are relative to apps/web, where the test script runs.
const workflowsDir = path.join(process.cwd(), "..", "..", ".github", "workflows");
const readWorkflow = (name: string) => fs.readFileSync(path.join(workflowsDir, name), "utf-8");

test("each target's lane is the workflow's own concurrency group", () => {
  for (const t of DISPATCH_TARGETS) {
    const group = /^concurrency:\s*\n\s+group:\s*([^\s#]+)/m.exec(readWorkflow(t.workflow))?.[1];
    assert.equal(group, t.lane, `${t.workflow}: lane must match its concurrency group`);
  }
});

test("dispatched workflows have no GitHub schedule that could wake the database off-window", () => {
  for (const t of DISPATCH_TARGETS) {
    assert.doesNotMatch(readWorkflow(t.workflow), /^\s+schedule:/m, `${t.workflow} still has a schedule trigger`);
  }
});

test("every dispatched input is declared by its workflow", () => {
  for (const t of DISPATCH_TARGETS) {
    const source = readWorkflow(t.workflow);
    const dispatchBlock = /workflow_dispatch:\s*\n((?:\s{4,}.*\n|\s*\n)*)/.exec(source)?.[1] ?? "";
    for (const name of Object.keys(t.inputs ?? {})) {
      assert.match(dispatchBlock, new RegExp(`^\\s+${name}:`, "m"), `${t.workflow} does not declare input ${name}`);
    }
  }
});

test("variable-gated workflows still honour their switch when dispatched", () => {
  for (const t of DISPATCH_TARGETS) {
    const source = readWorkflow(t.workflow);
    if (!/github\.event_name != 'schedule'/.test(source) && !/inputs\.scheduled/.test(source)) continue;
    assert.equal(t.inputs?.scheduled, "true", `${t.workflow} is gated; the dispatcher must send scheduled=true`);
    assert.match(source, /inputs\.scheduled != 'true'/, `${t.workflow}: the gate must treat scheduled=true like a schedule event`);
  }
});

test("the rolling collector is still hourly", () => {
  const rolling = DISPATCH_TARGETS.find((t) => t.workflow === "crypto-social-rolling.yml");
  assert.ok(rolling, "the workflow this was built for must be listed");
  assert.equal(rolling.everyMinutes, 60);
});

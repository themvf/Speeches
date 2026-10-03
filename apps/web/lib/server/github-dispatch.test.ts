import test from "node:test";
import assert from "node:assert/strict";
import { isDue, dispatchIfDue, DISPATCH_TARGETS, type DispatchTarget, type FetchLike } from "./github-dispatch.ts";

// Credentials are passed explicitly here so the tests never depend on the ambient environment.
const CREDS = { token: "t", owner: "o", repo: "r", ref: "main" };

const target: DispatchTarget = { workflow: "w.yml", everyMinutes: 60, reason: "test" };
const NOW = new Date("2026-09-18T12:00:00Z");
const runsBody = (createdAt: string | null) =>
  new Response(JSON.stringify({ workflow_runs: createdAt ? [{ created_at: createdAt }] : [] }), { status: 200 });

type Run = { created_at: string; conclusion: string | null };
// Serves the newest `per_page` runs, as the GitHub API does; a fresh Response per call.
const history = (runs: Run[]) => (url: string) =>
  new Response(JSON.stringify({ workflow_runs: runs.slice(0, Number(new URL(url).searchParams.get("per_page"))) }), { status: 200 });

// Records every call so a test can assert a dispatch did or did not happen.
function stub(handlers: { runs?: Response | ((url: string) => Response); dispatch?: Response }) {
  const calls: string[] = [];
  const fetchImpl: FetchLike = async (url, init) => {
    calls.push(`${init?.method ?? "GET"} ${url.includes("/runs") ? "runs" : "dispatch"}`);
    if (url.includes("/runs")) return typeof handlers.runs === "function" ? handlers.runs(url) : handlers.runs ?? runsBody(null);
    return handlers.dispatch ?? new Response(null, { status: 204 });
  };
  return { calls, fetchImpl };
}

test("isDue respects the period with a minute of slack", () => {
  assert.equal(isDue(target, null, NOW), true, "no history always fires");
  assert.equal(isDue(target, new Date("2026-09-18T11:00:00Z"), NOW), true, "exactly one period");
  assert.equal(isDue(target, new Date("2026-09-18T11:00:01Z"), NOW), true, "59m59s counts as due");
  assert.equal(isDue(target, new Date("2026-09-18T11:30:00Z"), NOW), false, "half a period is not due");
});

test("a recent run is skipped without dispatching", async () => {
  const { calls, fetchImpl } = stub({ runs: runsBody("2026-09-18T11:45:00Z") });
  const out = await dispatchIfDue(target, { ...CREDS, now: NOW, fetchImpl });
  assert.equal(out.status, "skipped");
  assert.match(out.detail, /15m ago/);
  assert.equal(out.lastRunAt, "2026-09-18T11:45:00.000Z");
  assert.deepEqual(calls, ["GET runs"], "never posts a dispatch when not due");
});

test("a stale run is dispatched and 204 is the success signal", async () => {
  const { calls, fetchImpl } = stub({ runs: runsBody("2026-09-18T08:00:00Z") });
  const out = await dispatchIfDue(target, { ...CREDS, now: NOW, fetchImpl });
  assert.equal(out.status, "dispatched");
  assert.equal(out.detail, "ref main");
  assert.deepEqual(calls, ["GET runs", "POST dispatch"]);
});

// bp-holder-intel.yml's hourly schedule is gated off, so GitHub creates a skipped run every few hours.
const skipped = (at: string): Run => ({ created_at: at, conclusion: "skipped" });

test("skipped schedule fires do not count as runs", async () => {
  const daily: DispatchTarget = { workflow: "w.yml", everyMinutes: 24 * 60, reason: "test" };
  const runs = [skipped("2026-09-18T11:20:00Z"), skipped("2026-09-18T08:20:00Z"), { created_at: "2026-09-17T06:00:00Z", conclusion: "success" }];
  const { calls, fetchImpl } = stub({ runs: history(runs) });
  const out = await dispatchIfDue(daily, { ...CREDS, now: NOW, fetchImpl });
  assert.equal(out.status, "dispatched", "the last real run is 30 hours old");
  assert.equal(out.lastRunAt, "2026-09-17T06:00:00.000Z");
  assert.deepEqual(calls, ["GET runs", "GET runs", "POST dispatch"]);
});

test("a recent real run behind skipped fires still holds the dispatch back", async () => {
  const daily: DispatchTarget = { workflow: "w.yml", everyMinutes: 24 * 60, reason: "test" };
  const runs = [skipped("2026-09-18T11:20:00Z"), { created_at: "2026-09-18T06:00:00Z", conclusion: "failure" }];
  const out = await dispatchIfDue(daily, { ...CREDS, now: NOW, fetchImpl: stub({ runs: history(runs) }).fetchImpl });
  assert.equal(out.status, "skipped");
  assert.equal(out.lastRunAt, "2026-09-18T06:00:00.000Z");
});

test("a window of only skipped fires is due, and a run in progress is never doubled", async () => {
  const daily: DispatchTarget = { workflow: "w.yml", everyMinutes: 24 * 60, reason: "test" };
  const allSkipped = Array.from({ length: 40 }, (_, i) => skipped(new Date(NOW.getTime() - (i + 1) * 3_600_000).toISOString()));
  assert.equal((await dispatchIfDue(daily, { ...CREDS, now: NOW, fetchImpl: stub({ runs: history(allSkipped) }).fetchImpl })).status, "dispatched");
  const running = [{ created_at: "2026-09-18T11:59:00Z", conclusion: null }, ...allSkipped];
  const { calls, fetchImpl } = stub({ runs: history(running) });
  assert.equal((await dispatchIfDue(daily, { ...CREDS, now: NOW, fetchImpl })).status, "skipped");
  assert.deepEqual(calls, ["GET runs"], "an unfinished newest run is read once and counts");
});

test("a rejected dispatch reports the status rather than throwing", async () => {
  const { fetchImpl } = stub({ runs: runsBody(null), dispatch: new Response("Resource not accessible", { status: 403 }) });
  const out = await dispatchIfDue(target, { ...CREDS, now: NOW, fetchImpl });
  assert.equal(out.status, "failed");
  assert.match(out.detail, /HTTP 403: Resource not accessible/);
});

test("unreadable run history fails the target, not the tick", async () => {
  const fetchImpl: FetchLike = async () => new Response("nope", { status: 401 });
  const out = await dispatchIfDue(target, { ...CREDS, now: NOW, fetchImpl });
  assert.equal(out.status, "failed");
  assert.match(out.detail, /run history unavailable \(HTTP 401\)/);
});

test("the rolling collector is a declared target", () => {
  const rolling = DISPATCH_TARGETS.find((t) => t.workflow === "crypto-social-rolling.yml");
  assert.ok(rolling, "the workflow this was built for must be listed");
  assert.equal(rolling.everyMinutes, 60);
});

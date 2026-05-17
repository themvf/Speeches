/**
 * Smoke tests for core app API endpoints.
 * Run against production or a custom BASE_URL env var.
 *
 * Exit codes:
 *   0 = all critical tests passed (warnings allowed)
 *   1 = one or more critical tests failed
 */

import assert from "node:assert/strict";

const BASE_URL = (process.env.APP_SMOKE_BASE_URL ?? "https://speeches-zeta.vercel.app").replace(/\/$/, "");
const TIMEOUT_MS = 15_000;

type Severity = "critical" | "warn";

type TestResult = {
  name: string;
  passed: boolean;
  severity: Severity;
  durationMs: number;
  error?: string;
};

async function fetchWithTimeout(url: string): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    return await fetch(url, {
      signal: controller.signal,
      headers: { "user-agent": "PolicyResearchHub/1.0 smoke-test" },
      cache: "no-store",
    });
  } finally {
    clearTimeout(timer);
  }
}

async function runTest(
  name: string,
  severity: Severity,
  fn: () => Promise<void>
): Promise<TestResult> {
  const start = Date.now();
  try {
    await fn();
    return { name, passed: true, severity, durationMs: Date.now() - start };
  } catch (err) {
    return {
      name,
      passed: false,
      severity,
      durationMs: Date.now() - start,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function yesterday(): string {
  const d = new Date();
  d.setUTCDate(d.getUTCDate() - 1);
  return d.toISOString().slice(0, 10);
}

// ── Tests ────────────────────────────────────────────────────────────────────

const results: TestResult[] = [];

// Feed
results.push(await runTest("GET /api/intel/feed", "critical", async () => {
  const res = await fetchWithTimeout(`${BASE_URL}/api/intel/feed?limit=5`);
  assert.ok(res.ok, `HTTP ${res.status}`);
  const json = await res.json() as { ok: boolean; data?: { articles?: unknown[] } };
  assert.equal(json.ok, true, "ok must be true");
  assert.ok(Array.isArray(json.data?.articles), "data.articles must be an array");
}));

// Trends
results.push(await runTest("GET /api/trends", "critical", async () => {
  const res = await fetchWithTimeout(`${BASE_URL}/api/trends`);
  assert.ok(res.ok, `HTTP ${res.status}`);
  const json = await res.json() as { ok: boolean; data?: { trends?: unknown[] } };
  assert.equal(json.ok, true, "ok must be true");
  assert.ok(Array.isArray(json.data?.trends), "data.trends must be an array");
}));

// Metrics
results.push(await runTest("GET /api/metrics", "critical", async () => {
  const res = await fetchWithTimeout(`${BASE_URL}/api/metrics`);
  assert.ok(res.ok, `HTTP ${res.status}`);
  const json = await res.json() as { ok: boolean; data?: Record<string, unknown> };
  assert.equal(json.ok, true, "ok must be true");
  assert.ok(json.data && typeof json.data === "object", "data must be an object");
}));

// Market crypto
results.push(await runTest("GET /api/market/crypto", "critical", async () => {
  const res = await fetchWithTimeout(`${BASE_URL}/api/market/crypto?limit=5`);
  assert.ok(res.ok, `HTTP ${res.status}`);
  const json = await res.json() as { ok: boolean; data?: { coins?: unknown[] } };
  assert.equal(json.ok, true, "ok must be true");
  assert.ok(Array.isArray(json.data?.coins), "data.coins must be an array");
}));

// Market bonds
results.push(await runTest("GET /api/market/bonds", "critical", async () => {
  const res = await fetchWithTimeout(`${BASE_URL}/api/market/bonds`);
  assert.ok(res.ok, `HTTP ${res.status}`);
  const json = await res.json() as { ok: boolean; data?: Record<string, unknown> };
  assert.equal(json.ok, true, "ok must be true");
  assert.ok(json.data && typeof json.data === "object", "data must be an object");
}));

// Recap (yesterday — may legitimately be empty, just must not crash)
results.push(await runTest("GET /api/intel/recap", "critical", async () => {
  const res = await fetchWithTimeout(`${BASE_URL}/api/intel/recap?date=${yesterday()}`);
  assert.ok(res.status === 200 || res.status === 404, `unexpected HTTP ${res.status}`);
  const json = await res.json() as { ok: boolean };
  // 404 with ok:false is acceptable — means no recap generated yet for that date
  assert.ok(typeof json.ok === "boolean", "response must have ok field");
}));

// Search (vector-store-dependent — warn only)
results.push(await runTest("GET /api/search", "warn", async () => {
  const res = await fetchWithTimeout(`${BASE_URL}/api/search?q=enforcement&topK=3`);
  assert.ok(res.ok, `HTTP ${res.status}`);
  const json = await res.json() as { ok: boolean; data?: unknown };
  assert.equal(json.ok, true, "ok must be true");
  assert.ok(json.data !== undefined, "data must be present");
}));

// ── Report ───────────────────────────────────────────────────────────────────

const passed = results.filter((r) => r.passed);
const failed = results.filter((r) => !r.passed);
const criticalFailures = failed.filter((r) => r.severity === "critical");

console.log(`\nSmoke test results — ${BASE_URL}\n${"─".repeat(60)}`);
for (const r of results) {
  const status = r.passed ? "PASS" : r.severity === "warn" ? "WARN" : "FAIL";
  const ms = `${r.durationMs}ms`;
  console.log(`  ${status.padEnd(4)}  ${r.name.padEnd(35)} ${ms.padStart(7)}`);
  if (!r.passed && r.error) {
    console.log(`        ${r.error}`);
  }
}
console.log(`${"─".repeat(60)}`);
console.log(`  ${passed.length}/${results.length} passed  |  ${criticalFailures.length} critical failure(s)\n`);

if (criticalFailures.length > 0) {
  process.exit(1);
}

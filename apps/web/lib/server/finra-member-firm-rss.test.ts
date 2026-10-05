import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";

/**
 * The firm rotation offset is floor(now / BATCH_SLOT_MS) * batchSize. If the
 * slot is shorter than the cron interval, the offset advances by more than one
 * batch per invocation and the firms in between are never fetched - a silent
 * coverage hole rather than a slower rotation. These two numbers live in
 * different files, so pin them together.
 */

// Read from source rather than importing: the module resolves "@/..." path
// aliases that bare `node --test` cannot, and this assertion is about two
// declared configuration values, not runtime behaviour.
function batchSlotMs(): number {
  const source = fs.readFileSync(
    path.join(process.cwd(), "lib/server/finra-member-firm-rss.ts"),
    "utf-8"
  );
  const match = /BATCH_SLOT_MS\s*=\s*(\d+)\s*\*\s*60_000/.exec(source);
  assert.ok(match, "BATCH_SLOT_MS must be declared as <minutes> * 60_000");
  return Number(match[1]) * 60_000;
}

function cronIntervalMs(): number {
  const configPath = path.join(process.cwd(), "vercel.json");
  const config = JSON.parse(fs.readFileSync(configPath, "utf-8")) as {
    crons?: Array<{ path: string; schedule: string }>;
  };
  const cron = (config.crons || []).find((entry) => entry.path === "/api/intel/rss-refresh");
  assert.ok(cron, "vercel.json must schedule /api/intel/rss-refresh");

  const [minuteField, hourField] = cron.schedule.trim().split(/\s+/);
  // Hourly at a fixed minute ("0 * * * *") is a 60-minute interval.
  if (/^\d+$/.test(minuteField) && hourField === "*") return 60 * 60_000;
  const match = /^\*\/(\d+)$/.exec(minuteField);
  assert.ok(match, `expected "*/N" or a fixed minute with an hourly hour field, got "${cron.schedule}"`);
  return Number(match[1]) * 60_000;
}

test("firm rotation slot matches the rss-refresh cron interval", () => {
  assert.equal(
    batchSlotMs(),
    cronIntervalMs(),
    "BATCH_SLOT_MS and the vercel.json rss-refresh cron must stay in lockstep, " +
      "or each run skips the firms between consecutive offsets"
  );
});

test("a full rotation still fits inside the 7-day Google News window", () => {
  const registry = JSON.parse(
    fs.readFileSync(path.join(process.cwd(), "lib/generated/finra-member-firms.json"), "utf-8")
  ) as { firms: Array<{ name?: string; rssUrl?: string }> };

  const firms = registry.firms.filter((firm) => firm.name && firm.rssUrl).length;
  const source = fs.readFileSync(path.join(process.cwd(), "lib/server/finra-member-firm-rss.ts"), "utf-8");
  const batchSize = Number(/DEFAULT_BATCH_SIZE\s*=\s*(\d+)/.exec(source)?.[1]);
  assert.ok(batchSize > 0, "DEFAULT_BATCH_SIZE must be declared as a number");
  const cycleMs = Math.ceil(firms / batchSize) * batchSlotMs();
  const sevenDaysMs = 7 * 24 * 60 * 60_000;

  assert.ok(
    cycleMs < sevenDaysMs,
    `full rotation takes ${(cycleMs / 3_600_000).toFixed(1)}h, which must stay under the ` +
      `168h "when:7d" query window or firm news is missed outright`
  );
});

test("the rss-refresh cron stays on minute 0, where the 03:00 UTC maintenance tick can fire", () => {
  const config = JSON.parse(fs.readFileSync(path.join(process.cwd(), "vercel.json"), "utf-8")) as {
    crons?: Array<{ path: string; schedule: string }>;
  };
  const cron = (config.crons || []).find((entry) => entry.path === "/api/intel/rss-refresh");
  assert.ok(cron);
  assert.equal(cron.schedule.trim().split(/\s+/)[0], "0", "route.ts checks getUTCMinutes() === 0 at 03:00 UTC");
});

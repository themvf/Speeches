import assert from "node:assert/strict";
import test from "node:test";

import { FRED_MACRO_DEFINITIONS } from "./fred-macro.ts";
import {
  addDays,
  buildCalendarEntries,
  calendarReleaseDefinitions,
  DAILY_REFRESH_RELEASE_IDS,
  fetchFredReleaseCalendar,
  parseReleaseDates,
  releaseName,
  toIsoDate,
} from "./fred-calendar.ts";

/** Swaps in a fetch stub for one call and always restores the original. */
async function withFetch<T>(
  handler: (url: URL) => { status?: number; body?: unknown },
  run: () => Promise<T>,
): Promise<{ result: T; urls: URL[] }> {
  const original = globalThis.fetch;
  const urls: URL[] = [];
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const url = new URL(String(input));
    urls.push(url);
    const { status = 200, body = { release_dates: [] } } = handler(url);
    return {
      ok: status >= 200 && status < 300,
      status,
      json: async () => body,
      text: async () => JSON.stringify(body),
    } as Response;
  }) as typeof globalThis.fetch;
  try {
    return { result: await run(), urls };
  } finally {
    globalThis.fetch = original;
  }
}

test("every tracked series carries a FRED release id", () => {
  for (const definition of FRED_MACRO_DEFINITIONS) {
    assert.equal(
      Number.isInteger(definition.releaseId) && definition.releaseId > 0,
      true,
      `${definition.seriesId} is missing a releaseId`,
    );
  }
});

test("every calendar release id resolves to a pinned name", () => {
  for (const releaseId of calendarReleaseDefinitions().keys()) {
    assert.equal(
      releaseName(releaseId).startsWith("FRED release "),
      false,
      `release ${releaseId} has no pinned name`,
    );
  }
});

test("groups indicators that share a release and drops daily refreshes", () => {
  const byRelease = calendarReleaseDefinitions();

  // Employment Situation carries four of the tracked series.
  const employmentSituation = byRelease.get(50)?.map((definition) => definition.id) ?? [];
  assert.deepEqual(new Set(employmentSituation), new Set([
    "nonfarm_payrolls",
    "unemployment_rate",
    "average_hourly_earnings_growth",
    "labor_force_participation",
  ]));

  // New Residential Construction carries two.
  assert.equal(byRelease.get(27)?.length, 2);

  for (const releaseId of DAILY_REFRESH_RELEASE_IDS) {
    assert.equal(byRelease.has(releaseId), false, `daily release ${releaseId} should be excluded`);
  }
});

test("parses release dates and ignores malformed or foreign rows", () => {
  const dates = parseReleaseDates({
    release_dates: [
      { release_id: 10, date: "2026-09-10" },
      { release_id: 10, date: "not-a-date" },
      { release_id: 10 },
      { release_id: 46, date: "2026-09-11" },
      { date: "2026-09-12" },
    ],
  }, 10);
  assert.deepEqual(dates, ["2026-09-10", "2026-09-12"]);
});

test("builds sorted, deduped entries inside the horizon", () => {
  const byRelease = calendarReleaseDefinitions();
  const entries = buildCalendarEntries(
    [
      { releaseId: 46, dates: ["2026-09-11"] },
      { releaseId: 10, dates: ["2026-09-10", "2026-09-10", "2026-08-01", "2026-12-01"] },
      { releaseId: 50, dates: ["2026-09-04"] },
    ],
    byRelease,
    "2026-08-20",
    "2026-11-18",
  );

  assert.deepEqual(entries.map((entry) => `${entry.date} ${entry.releaseName}`), [
    "2026-09-04 Employment Situation",
    "2026-09-10 Consumer Price Index",
    "2026-09-11 Producer Price Index",
  ]);
  assert.equal(entries[0].indicators.length, 4);
  assert.equal(entries[1].releaseUrl, "https://fred.stlouisfed.org/release?rid=10");
});

test("skips releases that map to no tracked indicator", () => {
  const entries = buildCalendarEntries(
    [{ releaseId: 999, dates: ["2026-09-10"] }],
    calendarReleaseDefinitions(),
    "2026-08-20",
    "2026-11-18",
  );
  assert.deepEqual(entries, []);
});

test("horizon dates advance in whole UTC days", () => {
  const start = new Date("2026-08-20T23:30:00Z");
  assert.equal(toIsoDate(start), "2026-08-20");
  assert.equal(toIsoDate(addDays(start, 90)), "2026-11-18");
  // Crossing a DST boundary must not lose or gain a day.
  assert.equal(toIsoDate(addDays(new Date("2026-10-30T12:00:00Z"), 7)), "2026-11-06");
});

test("requests future dates over the horizon window for each tracked release", async () => {
  const { result, urls } = await withFetch(
    (url) => ({
      body: {
        release_dates: [{ release_id: Number(url.searchParams.get("release_id")), date: "2026-09-15" }],
      },
    }),
    () => fetchFredReleaseCalendar(90, "test-key", new Date("2026-08-20T12:00:00Z")),
  );

  assert.equal(urls.length, calendarReleaseDefinitions().size);
  for (const url of urls) {
    // Without this flag FRED returns only past dates, which would make the
    // whole calendar empty.
    assert.equal(url.searchParams.get("include_release_dates_with_no_data"), "true");
    assert.equal(url.searchParams.get("realtime_start"), "2026-08-20");
    assert.equal(url.searchParams.get("realtime_end"), "2026-11-18");
    assert.equal(url.searchParams.get("file_type"), "json");
    assert.equal(url.pathname.endsWith("/fred/release/dates"), true);
  }

  assert.equal(result.horizonDays, 90);
  assert.equal(result.source, "FRED");
  assert.equal(result.warnings, undefined);
  assert.equal(result.entries.length, calendarReleaseDefinitions().size);
  assert.equal(result.entries.every((entry) => entry.date === "2026-09-15"), true);
});

test("a failing release degrades to a warning instead of an empty calendar", async () => {
  const { result } = await withFetch(
    (url) => (url.searchParams.get("release_id") === "10"
      ? { status: 500, body: { error_message: "boom" } }
      : { body: { release_dates: [{ release_id: Number(url.searchParams.get("release_id")), date: "2026-09-15" }] } }),
    () => fetchFredReleaseCalendar(90, "test-key", new Date("2026-08-20T12:00:00Z")),
  );

  assert.deepEqual(result.warnings, ["Consumer Price Index schedule unavailable."]);
  assert.equal(result.entries.length, calendarReleaseDefinitions().size - 1);
  assert.equal(result.entries.some((entry) => entry.releaseId === 10), false);
});

test("throws when every release schedule fails", async () => {
  await assert.rejects(
    () => withFetch(
      () => ({ status: 500, body: {} }),
      () => fetchFredReleaseCalendar(90, "test-key", new Date("2026-08-20T12:00:00Z")),
    ),
    /schedule failed \(500\)/,
  );
});

test("refuses to call FRED without an API key", async () => {
  await assert.rejects(
    () => fetchFredReleaseCalendar(90, "", new Date("2026-08-20T12:00:00Z")),
    /FRED_API_KEY is not configured/,
  );
});

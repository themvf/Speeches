import assert from "node:assert/strict";
import test from "node:test";

import {
  daysUntil,
  localIsoDate,
  matchContract,
  nextReleaseByIndicator,
  relativeDayLabel,
} from "./macro-calendar-display.ts";
import type { MacroCalendarEntry, MacroPredictionEvent } from "./server/types.ts";

function entry(date: string, releaseId: number, indicatorIds: string[]): MacroCalendarEntry {
  return {
    date,
    releaseId,
    releaseName: `Release ${releaseId}`,
    releaseUrl: `https://fred.stlouisfed.org/release?rid=${releaseId}`,
    indicators: indicatorIds.map((id) => ({
      id: id as MacroCalendarEntry["indicators"][number]["id"],
      label: id,
      seriesId: id.toUpperCase(),
      group: "headline",
    })),
  };
}

function contract(
  overrides: Partial<MacroPredictionEvent> & Pick<MacroPredictionEvent, "mappingKey" | "indicatorIds">,
): MacroPredictionEvent {
  return {
    eventId: overrides.mappingKey,
    slug: overrides.mappingKey,
    title: overrides.mappingKey,
    url: `https://polymarket.com/event/${overrides.mappingKey}`,
    theme: "inflation",
    matchKind: "related_signal",
    matchNote: "note",
    endDate: null,
    volume: 0,
    liquidity: 0,
    leadingOutcome: {
      marketId: "m1",
      conditionId: "c1",
      label: "Yes",
      probability: 0.62,
      oneDayChange: null,
      volume: 0,
      liquidity: 0,
      closed: false,
    },
    outcomes: [],
    ...overrides,
  };
}

test("derives the local calendar day, not the UTC one", () => {
  // 2026-08-21T02:00Z is still 2026-08-20 for anyone at UTC-5.
  const lateUtc = new Date("2026-08-21T02:00:00Z");
  const expected = [
    lateUtc.getFullYear(),
    String(lateUtc.getMonth() + 1).padStart(2, "0"),
    String(lateUtc.getDate()).padStart(2, "0"),
  ].join("-");
  assert.equal(localIsoDate(lateUtc), expected);
});

test("counts whole days between plain dates", () => {
  assert.equal(daysUntil("2026-08-20", "2026-08-20"), 0);
  assert.equal(daysUntil("2026-08-21", "2026-08-20"), 1);
  assert.equal(daysUntil("2026-11-18", "2026-08-20"), 90);
  // A date already past reads as negative rather than throwing.
  assert.equal(daysUntil("2026-08-19", "2026-08-20"), -1);
  assert.equal(daysUntil("nonsense", "2026-08-20"), 0);
});

test("labels the countdown in human terms", () => {
  assert.equal(relativeDayLabel(-3), "Today");
  assert.equal(relativeDayLabel(0), "Today");
  assert.equal(relativeDayLabel(1), "Tomorrow");
  assert.equal(relativeDayLabel(6), "In 6 days");
  assert.equal(relativeDayLabel(9), "Next week");
  assert.equal(relativeDayLabel(21), "In 3 weeks");
});

test("prefers an exact-series contract over a related signal", () => {
  const cpiRelease = entry("2026-09-10", 10, ["cpi_inflation"]);
  const related = contract({ mappingKey: "related", indicatorIds: ["cpi_inflation"] });
  const exact = contract({ mappingKey: "exact", indicatorIds: ["cpi_inflation"], matchKind: "exact_series" });

  assert.equal(matchContract(cpiRelease, [related, exact])?.mappingKey, "exact");
  assert.equal(matchContract(cpiRelease, [related])?.mappingKey, "related");
  assert.equal(matchContract(cpiRelease, []), null);
});

test("ignores contracts for other indicators or with no leading outcome", () => {
  const cpiRelease = entry("2026-09-10", 10, ["cpi_inflation"]);
  const otherIndicator = contract({ mappingKey: "jobs", indicatorIds: ["nonfarm_payrolls"] });
  const noLeader = contract({ mappingKey: "quiet", indicatorIds: ["cpi_inflation"], leadingOutcome: null });

  assert.equal(matchContract(cpiRelease, [otherIndicator, noLeader]), null);
});

test("maps the earliest upcoming release onto each indicator it updates", () => {
  const next = nextReleaseByIndicator(
    [
      entry("2026-08-10", 50, ["nonfarm_payrolls", "unemployment_rate"]), // already past
      entry("2026-09-04", 50, ["nonfarm_payrolls", "unemployment_rate"]),
      entry("2026-10-02", 50, ["nonfarm_payrolls", "unemployment_rate"]),
      entry("2026-09-10", 10, ["cpi_inflation"]),
    ],
    "2026-08-20",
  );

  assert.equal(next.get("nonfarm_payrolls"), "2026-09-04");
  assert.equal(next.get("unemployment_rate"), "2026-09-04");
  assert.equal(next.get("cpi_inflation"), "2026-09-10");
  assert.equal(next.get("housing_starts"), undefined);
});

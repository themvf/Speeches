import type {
  MacroCalendarEntry,
  MacroPredictionEvent,
  MarketMacroIndicatorId,
} from "@/lib/server/types";

/**
 * Local calendar day as YYYY-MM-DD. Calendar entries are plain days with no
 * timezone, so comparing them against a UTC "today" would mislabel a release
 * by one day for anyone whose local date has not caught up to UTC.
 */
export function localIsoDate(now: Date): string {
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

export function daysUntil(date: string, today: string): number {
  const target = Date.parse(`${date}T00:00:00Z`);
  const start = Date.parse(`${today}T00:00:00Z`);
  if (Number.isNaN(target) || Number.isNaN(start)) return 0;
  return Math.round((target - start) / 86_400_000);
}

export function relativeDayLabel(days: number): string {
  if (days <= 0) return "Today";
  if (days === 1) return "Tomorrow";
  if (days < 7) return `In ${days} days`;
  if (days < 14) return "Next week";
  return `In ${Math.round(days / 7)} weeks`;
}

export function formatCalendarDate(date: string, options: Intl.DateTimeFormatOptions): string {
  return new Date(`${date}T00:00:00Z`).toLocaleDateString("en-US", { ...options, timeZone: "UTC" });
}

/**
 * The one contract whose indicators overlap this release, preferring an exact
 * series match over a related signal so a CPI row shows the CPI bracket market
 * rather than an adjacent inflation contract.
 */
export function matchContract(
  entry: MacroCalendarEntry,
  contracts: MacroPredictionEvent[],
): MacroPredictionEvent | null {
  const indicatorIds = new Set<MarketMacroIndicatorId>(entry.indicators.map((indicator) => indicator.id));
  const matches = contracts.filter(
    (contract) =>
      contract.leadingOutcome && contract.indicatorIds.some((indicatorId) => indicatorIds.has(indicatorId)),
  );
  if (!matches.length) return null;
  return matches.find((contract) => contract.matchKind === "exact_series") ?? matches[0];
}

/**
 * Earliest upcoming release date per indicator. Several indicators share one
 * release (all four Employment Situation series, for instance), so this maps
 * the calendar's release-shaped rows back onto individual indicator cards.
 */
export function nextReleaseByIndicator(
  entries: MacroCalendarEntry[],
  today: string,
): Map<MarketMacroIndicatorId, string> {
  const next = new Map<MarketMacroIndicatorId, string>();
  for (const entry of entries) {
    if (entry.date < today) continue;
    for (const indicator of entry.indicators) {
      const existing = next.get(indicator.id);
      if (!existing || entry.date < existing) next.set(indicator.id, entry.date);
    }
  }
  return next;
}

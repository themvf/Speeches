import { FRED_MACRO_DEFINITIONS, type FredMacroDefinition } from "./fred-macro.ts";
import type {
  MacroCalendarEntry,
  MacroCalendarIndicatorRef,
  MarketMacroCalendarData,
} from "./types.ts";

const FRED_API_BASE = "https://api.stlouisfed.org/fred";

/**
 * Release schedules move rarely (agencies publish them months ahead), so this
 * caches far longer than the 15-minute observation cache in fred-macro.ts.
 */
export const FRED_CALENDAR_CACHE_SECONDS = 6 * 60 * 60;

export const DEFAULT_CALENDAR_HORIZON_DAYS = 90;
export const MIN_CALENDAR_HORIZON_DAYS = 7;
export const MAX_CALENDAR_HORIZON_DAYS = 365;

/**
 * `fred/release/dates` returns only a release_id, never the release name, so
 * the names are pinned here. Verified against each series' FRED page (the
 * "Series from <name>" release link) on 2026-08-20.
 */
const RELEASE_NAMES: Readonly<Record<number, string>> = {
  9: "Advance Monthly Sales for Retail and Food Services",
  10: "Consumer Price Index",
  13: "G.17 Industrial Production and Capacity Utilization",
  17: "H.10 Foreign Exchange Rates",
  18: "H.15 Selected Interest Rates",
  20: "H.4.1 Factors Affecting Reserve Balances",
  21: "H.6 Money Stock Measures",
  27: "New Residential Construction",
  46: "Producer Price Index",
  50: "Employment Situation",
  53: "Gross Domestic Product",
  54: "Personal Income and Outlays",
  180: "Unemployment Insurance Weekly Claims Report",
  187: "St. Louis Fed Financial Stress Index",
  190: "Primary Mortgage Market Survey",
  192: "Job Openings and Labor Turnover Survey",
  221: "Chicago Fed National Financial Conditions Index",
  304: "Interest Rate Spreads",
  445: "Secured Overnight Financing Rate Data",
  456: "Sahm Rule Recession Indicator",
};

/**
 * Scheduled publication time per release, as Eastern wall-clock "HH:MM".
 *
 * FRED serves dates with no time of day, and the eight publishers behind these
 * releases have no common schedule feed -- so these are pinned constants. They
 * are safe as constants because the values move on the order of decades (CPI
 * has been 8:30 ET for as long as anyone has cared), and each was read off the
 * publisher's own live schedule on 2026-08-21 rather than recalled:
 *
 *   BLS     bls.gov/schedule/2026/MM_sched.htm              (10, 46, 50, 192)
 *   Census  census.gov/economic-indicators/calendar-listview.html  (9, 27)
 *   BEA     bea.gov/news/schedule                           (53, 54)
 *   Fed     federalreserve.gov/releases/{g17,h6}/current/, feeds/h41.xml
 *
 * Deliberately Eastern wall clock, NOT a UTC offset: the same 8:30 ET release
 * is 12:30Z in summer and 13:30Z in winter. Times are rendered labelled "ET",
 * so no timezone conversion is needed anywhere and DST cannot skew them.
 *
 * Releases absent from this map render date-only, which is the correct
 * behaviour rather than a guess -- see CALENDAR_TIME_UNKNOWN_RELEASE_IDS.
 */
const RELEASE_TIMES_ET: Readonly<Record<number, string>> = {
  9: "08:30", // Advance Monthly Retail Sales -- Census
  10: "08:30", // Consumer Price Index -- BLS
  13: "09:15", // G.17 Industrial Production -- Federal Reserve
  20: "16:30", // H.4.1 Reserve Balances -- Federal Reserve
  21: "13:00", // H.6 Money Stock -- Federal Reserve
  27: "08:30", // New Residential Construction -- Census
  46: "08:30", // Producer Price Index -- BLS
  50: "08:30", // Employment Situation -- BLS
  53: "08:30", // Gross Domestic Product -- BEA
  54: "08:30", // Personal Income and Outlays -- BEA
  // DOL publishes the weekly claims schedule only as a PDF and blocks the ETA
  // press index, so this one is corroborated rather than read off a schedule:
  // FRED ingests ICSA at ~08:34 ET, consistent with the long-standing 8:30.
  180: "08:30", // Unemployment Insurance Weekly Claims -- DOL/ETA
  190: "12:00", // Primary Mortgage Market Survey -- Freddie Mac ("Thursdays at 12 p.m. ET")
  192: "10:00", // Job Openings and Labor Turnover Survey -- BLS
  221: "08:30", // Chicago Fed NFCI ("8:30 a.m. ET on Wednesday")
};

/**
 * Releases with no published time. STLFSI's page states no release time, and
 * the Sahm Rule is a FRED-computed series with no press release of its own
 * (it lands with the Employment Situation). Asserting a time for either would
 * be a guess dressed as data.
 */
export const CALENDAR_TIME_UNKNOWN_RELEASE_IDS: ReadonlySet<number> = new Set([187, 456]);

export function releaseTimeEt(releaseId: number): string | null {
  return RELEASE_TIMES_ET[releaseId] ?? null;
}

/**
 * Releases that are a daily market-rate refresh rather than a scheduled
 * economic event. Each publishes every business day, so including them would
 * add ~65 rows per quarter apiece and bury CPI/payrolls in the noise. Their
 * indicator cards still render on the Macro tab; only the calendar skips them.
 */
export const DAILY_REFRESH_RELEASE_IDS: ReadonlySet<number> = new Set([
  17, // H.10 Foreign Exchange Rates -> trade_weighted_dollar
  18, // H.15 Selected Interest Rates -> effective_fed_funds
  304, // Interest Rate Spreads -> yield_curve_10y2y, breakeven_inflation_10y
  445, // Secured Overnight Financing Rate Data -> sofr
]);

interface FredReleaseDatesPayload {
  release_dates?: Array<{ release_id?: number; date?: string }>;
}

export function calendarReleaseDefinitions(): Map<number, FredMacroDefinition[]> {
  const byRelease = new Map<number, FredMacroDefinition[]>();
  for (const definition of FRED_MACRO_DEFINITIONS) {
    if (DAILY_REFRESH_RELEASE_IDS.has(definition.releaseId)) continue;
    const existing = byRelease.get(definition.releaseId);
    if (existing) existing.push(definition);
    else byRelease.set(definition.releaseId, [definition]);
  }
  return byRelease;
}

export function toIsoDate(date: Date): string {
  return date.toISOString().slice(0, 10);
}

export function addDays(date: Date, days: number): Date {
  const next = new Date(date.getTime());
  next.setUTCDate(next.getUTCDate() + days);
  return next;
}

export function releaseUrl(releaseId: number): string {
  return `https://fred.stlouisfed.org/release?rid=${releaseId}`;
}

export function releaseName(releaseId: number): string {
  return RELEASE_NAMES[releaseId] ?? `FRED release ${releaseId}`;
}

function indicatorRef(definition: FredMacroDefinition): MacroCalendarIndicatorRef {
  return {
    id: definition.id,
    label: definition.label,
    seriesId: definition.seriesId,
    group: definition.group,
  };
}

/**
 * Turns per-release date lists into one chronologically sorted list of dated
 * release rows. Dates outside [startDate, endDate] are dropped: FRED honours
 * the realtime window, but the horizon is a promise this payload makes to the
 * UI, so it is enforced here rather than trusted upstream.
 */
export function buildCalendarEntries(
  releaseDates: Array<{ releaseId: number; dates: string[] }>,
  definitionsByRelease: Map<number, FredMacroDefinition[]>,
  startDate: string,
  endDate: string,
): MacroCalendarEntry[] {
  const entries: MacroCalendarEntry[] = [];
  for (const { releaseId, dates } of releaseDates) {
    const definitions = definitionsByRelease.get(releaseId);
    if (!definitions?.length) continue;
    // A release can repeat a date (revisions carry their own row), and the
    // calendar wants one row per (release, date).
    for (const date of [...new Set(dates)].sort()) {
      if (date < startDate || date > endDate) continue;
      entries.push({
        date,
        timeEt: releaseTimeEt(releaseId),
        releaseId,
        releaseName: releaseName(releaseId),
        releaseUrl: releaseUrl(releaseId),
        indicators: definitions.map(indicatorRef),
      });
    }
  }
  // Within a day, order by publication time so the morning prints lead;
  // releases with no known time sort last rather than pretending to be at
  // midnight.
  return entries.sort((left, right) => {
    if (left.date !== right.date) return left.date.localeCompare(right.date);
    if (left.timeEt !== right.timeEt) {
      if (!left.timeEt) return 1;
      if (!right.timeEt) return -1;
      return left.timeEt.localeCompare(right.timeEt);
    }
    return left.releaseName.localeCompare(right.releaseName);
  });
}

export function parseReleaseDates(payload: FredReleaseDatesPayload, releaseId: number): string[] {
  return (payload.release_dates ?? []).flatMap((entry) => {
    const date = String(entry.date ?? "");
    const matchesRelease = entry.release_id === undefined || entry.release_id === releaseId;
    return matchesRelease && /^\d{4}-\d{2}-\d{2}$/.test(date) ? [date] : [];
  });
}

async function fetchReleaseDates(
  releaseId: number,
  apiKey: string,
  startDate: string,
  endDate: string,
): Promise<string[]> {
  const url = new URL(`${FRED_API_BASE}/release/dates`);
  url.searchParams.set("api_key", apiKey);
  url.searchParams.set("file_type", "json");
  url.searchParams.set("release_id", String(releaseId));
  url.searchParams.set("realtime_start", startDate);
  url.searchParams.set("realtime_end", endDate);
  // Without this, FRED omits exactly the rows this calendar exists to show:
  // the default "excludes future release dates which may be available in the
  // FRED release calendar" (fred/release/dates docs).
  url.searchParams.set("include_release_dates_with_no_data", "true");
  url.searchParams.set("sort_order", "asc");

  const response = await fetch(url.toString(), {
    headers: { Accept: "application/json" },
    next: { revalidate: FRED_CALENDAR_CACHE_SECONDS },
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`FRED release ${releaseId} schedule failed (${response.status}): ${body.slice(0, 180)}`);
  }
  return parseReleaseDates((await response.json()) as FredReleaseDatesPayload, releaseId);
}

export async function fetchFredReleaseCalendar(
  horizonDays = DEFAULT_CALENDAR_HORIZON_DAYS,
  apiKey = String(process.env.FRED_API_KEY ?? "").trim(),
  now = new Date(),
): Promise<MarketMacroCalendarData> {
  if (!apiKey) throw new Error("FRED_API_KEY is not configured.");

  const startDate = toIsoDate(now);
  const endDate = toIsoDate(addDays(now, horizonDays));
  const definitionsByRelease = calendarReleaseDefinitions();
  const releaseIds = [...definitionsByRelease.keys()];

  const results = await Promise.allSettled(
    releaseIds.map((releaseId) => fetchReleaseDates(releaseId, apiKey, startDate, endDate)),
  );

  const releaseDates: Array<{ releaseId: number; dates: string[] }> = [];
  const warnings: string[] = [];
  results.forEach((result, index) => {
    const releaseId = releaseIds[index];
    if (result.status === "fulfilled") releaseDates.push({ releaseId, dates: result.value });
    else warnings.push(`${releaseName(releaseId)} schedule unavailable.`);
  });

  if (!releaseDates.length) {
    const firstFailure = results.find((result) => result.status === "rejected");
    throw firstFailure && firstFailure.status === "rejected"
      ? firstFailure.reason
      : new Error("FRED returned no release schedules.");
  }

  return {
    entries: buildCalendarEntries(releaseDates, definitionsByRelease, startDate, endDate),
    horizonDays,
    generatedAt: new Date().toISOString(),
    cacheSeconds: FRED_CALENDAR_CACHE_SECONDS,
    source: "FRED",
    ...(warnings.length ? { warnings } : {}),
  };
}

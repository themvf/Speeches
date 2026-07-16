import type {
  MacroPredictionEvent,
  MacroPredictionMatchKind,
  MacroPredictionOutcome,
  MacroPredictionTheme,
  MarketMacroIndicatorId,
  MarketMacroPredictionsData,
} from "./types.ts";

const GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events";
export const POLYMARKET_MACRO_CACHE_SECONDS = 5 * 60;

const DISCOVERY_TAGS = [
  "fed",
  "cpi-release",
  "inflation",
  "gdp",
  "nfp",
  "freddie-mac",
  "economic-policy",
] as const;

interface GammaMarket {
  id?: string;
  conditionId?: string;
  question?: string;
  slug?: string;
  outcomes?: string;
  outcomePrices?: string;
  groupItemTitle?: string;
  groupItemThreshold?: string;
  volume?: string | number;
  volumeNum?: number;
  liquidity?: string | number;
  liquidityNum?: number;
  oneDayPriceChange?: number;
  closed?: boolean;
}

export interface GammaEvent {
  id?: string;
  slug?: string;
  title?: string;
  endDate?: string;
  volume?: string | number;
  liquidity?: string | number;
  active?: boolean;
  closed?: boolean;
  markets?: GammaMarket[];
}

interface MacroMappingDefinition {
  key: string;
  titlePattern: RegExp;
  theme: MacroPredictionTheme;
  matchKind: MacroPredictionMatchKind;
  matchNote: string;
  indicatorIds: MarketMacroIndicatorId[];
  selection: "nearest" | "highest_volume";
}

export const MACRO_EVENT_MAPPINGS: readonly MacroMappingDefinition[] = [
  {
    key: "fed_next_decision",
    titlePattern: /^Fed Decision in .+\?$/i,
    theme: "fed_policy",
    matchKind: "related_signal",
    matchNote: "Contract resolves on the target range's upper bound; the FRED card shows the effective traded rate.",
    indicatorIds: ["effective_fed_funds"],
    selection: "nearest",
  },
  {
    key: "nonfarm_payrolls_next",
    titlePattern: /^How many jobs added in .+\?$/i,
    theme: "labor",
    matchKind: "exact_series",
    matchNote: "Monthly bracket distribution for the same BLS nonfarm-payroll release shown on the FRED card.",
    indicatorIds: ["nonfarm_payrolls"],
    selection: "nearest",
  },
  {
    key: "unemployment_next",
    titlePattern: /^(January|February|March|April|May|June|July|August|September|October|November|December) Unemployment Rate$/i,
    theme: "labor",
    matchKind: "exact_series",
    matchNote: "Monthly bracket distribution for the same BLS U-3 unemployment series.",
    indicatorIds: ["unemployment_rate"],
    selection: "nearest",
  },
  {
    key: "headline_cpi_monthly_next",
    titlePattern: /^(January|February|March|April|May|June|July|August|September|October|November|December) Inflation(?: US)? - Monthly$/i,
    theme: "inflation",
    matchKind: "related_signal",
    matchNote: "Forecasts headline CPI month-over-month; the FRED card displays the year-over-year transformation.",
    indicatorIds: ["cpi_inflation"],
    selection: "nearest",
  },
  {
    key: "headline_cpi_annual_next",
    titlePattern: /^(January|February|March|April|May|June|July|August|September|October|November|December) Inflation(?: US)? - Annual$/i,
    theme: "inflation",
    matchKind: "exact_series",
    matchNote: "Monthly release brackets for headline CPI year-over-year, matching the FRED card's transformation.",
    indicatorIds: ["cpi_inflation"],
    selection: "nearest",
  },
  {
    key: "core_cpi_next",
    titlePattern: /^Core CPI MoM - /i,
    theme: "inflation",
    matchKind: "related_signal",
    matchNote: "Contract forecasts Core CPI month-over-month; the mapped cards show headline CPI and Core PCE year-over-year.",
    indicatorIds: ["cpi_inflation", "core_pce_inflation"],
    selection: "nearest",
  },
  {
    key: "us_gdp_next",
    titlePattern: /^US GDP growth in Q[1-4] \d{4}\?$/i,
    theme: "growth",
    matchKind: "exact_series",
    matchNote: "Quarterly real GDP growth bracket distribution matching the corresponding BEA/FRED release.",
    indicatorIds: ["real_gdp_growth"],
    selection: "nearest",
  },
] as const;

function finiteNumber(value: unknown): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function parseJsonArray(value: unknown): unknown[] {
  if (Array.isArray(value)) return value;
  if (typeof value !== "string") return [];
  try {
    const parsed = JSON.parse(value) as unknown;
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function outcomeLabel(market: GammaMarket, eventTitle: string, marketCount: number): string {
  const grouped = String(market.groupItemTitle || market.groupItemThreshold || "").trim();
  if (grouped) return grouped;
  const question = String(market.question || "").trim();
  if (marketCount === 1 || question === eventTitle) return "Yes";
  return question || "Outcome";
}

function normalizeOutcomes(event: GammaEvent): MacroPredictionOutcome[] {
  const markets = event.markets ?? [];
  return markets.flatMap((market) => {
    const outcomes = parseJsonArray(market.outcomes).map(String);
    const prices = parseJsonArray(market.outcomePrices).map(Number);
    const yesIndex = outcomes.findIndex((outcome) => outcome.toLowerCase() === "yes");
    const probability = yesIndex >= 0 ? prices[yesIndex] : Number.NaN;
    if (!Number.isFinite(probability)) return [];
    return [{
      marketId: String(market.id || market.conditionId || market.slug || ""),
      conditionId: String(market.conditionId || ""),
      label: outcomeLabel(market, String(event.title || ""), markets.length),
      probability: Math.max(0, Math.min(1, probability)),
      oneDayChange: Number.isFinite(Number(market.oneDayPriceChange))
        ? Number(market.oneDayPriceChange)
        : null,
      volume: finiteNumber(market.volumeNum ?? market.volume),
      liquidity: finiteNumber(market.liquidityNum ?? market.liquidity),
      closed: Boolean(market.closed),
    }];
  });
}

function eventTimestamp(event: GammaEvent): number {
  const parsed = Date.parse(String(event.endDate || ""));
  return Number.isFinite(parsed) ? parsed : Number.POSITIVE_INFINITY;
}

function selectEvent(
  events: GammaEvent[],
  mapping: MacroMappingDefinition,
  now: Date,
): GammaEvent | null {
  const candidates = events.filter((event) => mapping.titlePattern.test(String(event.title || "")));
  if (!candidates.length) return null;
  if (mapping.selection === "nearest") {
    const nowMs = now.getTime();
    const future = candidates.filter((event) => eventTimestamp(event) >= nowMs);
    return [...(future.length ? future : candidates)]
      .sort((left, right) => eventTimestamp(left) - eventTimestamp(right))[0] ?? null;
  }
  return [...candidates]
    .sort((left, right) => finiteNumber(right.volume) - finiteNumber(left.volume))[0] ?? null;
}

export function buildMacroPredictionEvents(
  rawEvents: GammaEvent[],
  now = new Date(),
): MacroPredictionEvent[] {
  const uniqueEvents = [...new Map(
    rawEvents
      .filter((event) => event.id || event.slug)
      .map((event) => [String(event.id || event.slug), event]),
  ).values()];

  return MACRO_EVENT_MAPPINGS.flatMap((mapping) => {
    const event = selectEvent(uniqueEvents, mapping, now);
    if (!event) return [];
    const outcomes = normalizeOutcomes(event);
    if (!outcomes.length) return [];
    const openOutcomes = outcomes.filter((outcome) => !outcome.closed);
    const leadingPool = openOutcomes.length ? openOutcomes : outcomes;
    const leadingOutcome = [...leadingPool]
      .sort((left, right) => right.probability - left.probability)[0] ?? null;
    const slug = String(event.slug || "");
    return [{
      mappingKey: mapping.key,
      eventId: String(event.id || slug),
      slug,
      title: String(event.title || "Polymarket macro event"),
      url: "https://polymarket.com/event/" + encodeURIComponent(slug),
      theme: mapping.theme,
      matchKind: mapping.matchKind,
      matchNote: mapping.matchNote,
      indicatorIds: [...mapping.indicatorIds],
      endDate: event.endDate || null,
      volume: finiteNumber(event.volume),
      liquidity: finiteNumber(event.liquidity),
      leadingOutcome,
      outcomes,
    }];
  });
}

async function fetchEventsByTag(tagSlug: string): Promise<GammaEvent[]> {
  const url = new URL(GAMMA_EVENTS_URL);
  url.searchParams.set("tag_slug", tagSlug);
  url.searchParams.set("active", "true");
  url.searchParams.set("closed", "false");
  url.searchParams.set("order", "volume");
  url.searchParams.set("ascending", "false");
  url.searchParams.set("limit", "100");
  const response = await fetch(url, {
    headers: { Accept: "application/json", "User-Agent": "PolicyResearchHub/1.0" },
    next: { revalidate: POLYMARKET_MACRO_CACHE_SECONDS },
  });
  if (!response.ok) {
    throw new Error("Polymarket " + tagSlug + " discovery failed (" + response.status + ").");
  }
  return response.json() as Promise<GammaEvent[]>;
}

export async function fetchPolymarketMacroPredictions(): Promise<MarketMacroPredictionsData> {
  const results = await Promise.allSettled(DISCOVERY_TAGS.map(fetchEventsByTag));
  const successful = results.flatMap((result) => result.status === "fulfilled" ? result.value : []);
  if (!successful.length) {
    const firstFailure = results.find((result) => result.status === "rejected");
    throw firstFailure && firstFailure.status === "rejected"
      ? firstFailure.reason
      : new Error("Polymarket macro discovery returned no events.");
  }
  const events = buildMacroPredictionEvents(successful);
  const failedTags = results.filter((result) => result.status === "rejected").length;
  const warning = failedTags > 0
    ? failedTags + " of " + DISCOVERY_TAGS.length + " Polymarket discovery categories were unavailable."
    : events.length === 0
      ? "No supported US macro contracts are currently active."
      : undefined;
  return {
    events,
    generatedAt: new Date().toISOString(),
    cacheSeconds: POLYMARKET_MACRO_CACHE_SECONDS,
    source: "Polymarket Gamma API",
    ...(warning ? { warning } : {}),
  };
}

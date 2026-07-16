import type {
  MarketMacroIndicator,
  MarketMacroIndicatorId,
  MarketMacroPoint,
  MarketMacroUnit,
} from "./types.ts";

const FRED_API_BASE = "https://api.stlouisfed.org/fred";
export const FRED_MACRO_CACHE_SECONDS = 15 * 60;

type FredUnits = "lin" | "chg" | "pc1";

export interface FredMacroDefinition {
  id: MarketMacroIndicatorId;
  seriesId: string;
  label: string;
  description: string;
  unit: MarketMacroUnit;
  units: FredUnits;
  limit: number;
}

interface FredObservationPayload {
  observations?: Array<{ date?: string; value?: string }>;
}

interface FredSeriesPayload {
  seriess?: Array<{
    frequency?: string;
    last_updated?: string;
  }>;
}

export const FRED_MACRO_DEFINITIONS: readonly FredMacroDefinition[] = [
  {
    id: "real_gdp_growth",
    seriesId: "A191RL1Q225SBEA",
    label: "Real GDP Growth",
    description: "Inflation-adjusted economic growth at a seasonally adjusted annual rate.",
    unit: "percent",
    units: "lin",
    limit: 20,
  },
  {
    id: "cpi_inflation",
    seriesId: "CPIAUCSL",
    label: "CPI Inflation",
    description: "Headline consumer inflation measured as the percent change from one year ago.",
    unit: "percent",
    units: "pc1",
    limit: 60,
  },
  {
    id: "nonfarm_payrolls",
    seriesId: "PAYEMS",
    label: "Nonfarm Payrolls",
    description: "Monthly change in US payroll employment, excluding farm workers.",
    unit: "thousands",
    units: "chg",
    limit: 60,
  },
  {
    id: "unemployment_rate",
    seriesId: "UNRATE",
    label: "Unemployment Rate",
    description: "Share of the US labor force that is unemployed and actively seeking work.",
    unit: "percent",
    units: "lin",
    limit: 60,
  },
  {
    id: "effective_fed_funds",
    seriesId: "DFF",
    label: "Effective Fed Funds Rate",
    description: "Volume-weighted overnight rate for federal funds transactions.",
    unit: "percent",
    units: "lin",
    limit: 365,
  },
  {
    id: "yield_curve_10y2y",
    seriesId: "T10Y2Y",
    label: "10Y-2Y Treasury Spread",
    description: "Difference between 10-year and 2-year Treasury yields; negative values indicate inversion.",
    unit: "percentage_points",
    units: "lin",
    limit: 365,
  },
] as const;

function fredUrl(path: string, apiKey: string, params: Record<string, string | number>): string {
  const url = new URL(`${FRED_API_BASE}/${path}`);
  url.searchParams.set("api_key", apiKey);
  url.searchParams.set("file_type", "json");
  for (const [key, value] of Object.entries(params)) url.searchParams.set(key, String(value));
  return url.toString();
}

async function fetchFredJson<T>(url: string): Promise<T> {
  const response = await fetch(url, {
    headers: { Accept: "application/json" },
    next: { revalidate: FRED_MACRO_CACHE_SECONDS },
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`FRED request failed (${response.status}): ${body.slice(0, 180)}`);
  }
  return response.json() as Promise<T>;
}

export function parseFredObservations(payload: FredObservationPayload): MarketMacroPoint[] {
  return (payload.observations ?? [])
    .flatMap((observation) => {
      const value = Number(observation.value);
      return observation.date && Number.isFinite(value)
        ? [{ date: observation.date, value }]
        : [];
    })
    .sort((left, right) => left.date.localeCompare(right.date));
}

export function buildMacroIndicator(
  definition: FredMacroDefinition,
  points: MarketMacroPoint[],
  metadata: { frequency?: string; lastUpdated?: string },
): MarketMacroIndicator {
  const current = points.at(-1);
  if (!current) throw new Error(`FRED returned no observations for ${definition.seriesId}`);
  const previous = points.at(-2) ?? null;
  return {
    id: definition.id,
    fredSeriesId: definition.seriesId,
    label: definition.label,
    description: definition.description,
    frequency: metadata.frequency || "Unknown",
    unit: definition.unit,
    value: current.value,
    previousValue: previous?.value ?? null,
    change: previous ? current.value - previous.value : null,
    observationDate: current.date,
    lastUpdated: metadata.lastUpdated || "",
    points,
    sourceUrl: `https://fred.stlouisfed.org/series/${definition.seriesId}`,
  };
}

async function fetchMacroIndicator(definition: FredMacroDefinition, apiKey: string): Promise<MarketMacroIndicator> {
  const [observations, series] = await Promise.all([
    fetchFredJson<FredObservationPayload>(fredUrl("series/observations", apiKey, {
      series_id: definition.seriesId,
      units: definition.units,
      sort_order: "desc",
      limit: definition.limit,
    })),
    fetchFredJson<FredSeriesPayload>(fredUrl("series", apiKey, { series_id: definition.seriesId })),
  ]);
  const metadata = series.seriess?.[0];
  return buildMacroIndicator(definition, parseFredObservations(observations), {
    frequency: metadata?.frequency,
    lastUpdated: metadata?.last_updated,
  });
}

export async function fetchFredMacroIndicators(apiKey = String(process.env.FRED_API_KEY ?? "").trim()): Promise<MarketMacroIndicator[]> {
  if (!apiKey) throw new Error("FRED_API_KEY is not configured.");
  return Promise.all(FRED_MACRO_DEFINITIONS.map((definition) => fetchMacroIndicator(definition, apiKey)));
}

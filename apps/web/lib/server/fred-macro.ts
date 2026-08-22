import type {
  MarketMacroGroup,
  MarketMacroIndicator,
  MarketMacroIndicatorId,
  MarketMacroPoint,
  MarketMacroUnit,
} from "./types.ts";

const FRED_API_BASE = "https://api.stlouisfed.org/fred";
export const FRED_MACRO_CACHE_SECONDS = 15 * 60;

type FredUnits = "lin" | "chg" | "pch" | "pc1";

export interface FredMacroDefinition {
  id: MarketMacroIndicatorId;
  seriesId: string;
  /**
   * FRED release this series belongs to (`fred/series/release`). Several
   * indicators share one release -- the four Employment Situation series,
   * for instance -- which is what lets the release calendar show a single
   * dated row per release rather than one per series.
   */
  releaseId: number;
  label: string;
  description: string;
  unit: MarketMacroUnit;
  group: MarketMacroGroup;
  priority: number;
  units: FredUnits;
  limit: number;
  scale?: number;
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
    id: "real_gdp_growth", seriesId: "A191RL1Q225SBEA", releaseId: 53, label: "Real GDP Growth",
    description: "Inflation-adjusted economic growth at a seasonally adjusted annual rate.",
    unit: "percent", group: "headline", priority: 1, units: "lin", limit: 20,
  },
  {
    id: "cpi_inflation", seriesId: "CPIAUCSL", releaseId: 10, label: "CPI Inflation",
    description: "Headline consumer inflation measured as the percent change from one year ago.",
    unit: "percent", group: "headline", priority: 2, units: "pc1", limit: 60,
  },
  {
    id: "nonfarm_payrolls", seriesId: "PAYEMS", releaseId: 50, label: "Nonfarm Payrolls",
    description: "Monthly change in US payroll employment, excluding farm workers.",
    unit: "thousands", group: "headline", priority: 3, units: "chg", limit: 60,
  },
  {
    id: "unemployment_rate", seriesId: "UNRATE", releaseId: 50, label: "Unemployment Rate",
    description: "Share of the US labor force that is unemployed and actively seeking work.",
    unit: "percent", group: "headline", priority: 4, units: "lin", limit: 60,
  },
  {
    id: "effective_fed_funds", seriesId: "DFF", releaseId: 18, label: "Effective Fed Funds Rate",
    description: "Volume-weighted overnight rate for federal funds transactions.",
    unit: "percent", group: "headline", priority: 5, units: "lin", limit: 365,
  },
  {
    id: "yield_curve_10y2y", seriesId: "T10Y2Y", releaseId: 304, label: "10Y-2Y Treasury Spread",
    description: "Difference between 10-year and 2-year Treasury yields; negative values indicate inversion.",
    unit: "percentage_points", group: "headline", priority: 6, units: "lin", limit: 365,
  },
  {
    id: "retail_sales_growth", seriesId: "RSAFS", releaseId: 9, label: "Retail Sales",
    description: "Monthly change in advance estimates of US retail and food-services sales.",
    unit: "percent", group: "activity", priority: 1, units: "pch", limit: 60,
  },
  {
    id: "industrial_production_growth", seriesId: "INDPRO", releaseId: 13, label: "Industrial Production",
    description: "Monthly change in real output from manufacturing, mining, and utilities.",
    unit: "percent", group: "activity", priority: 2, units: "pch", limit: 60,
  },
  {
    id: "core_pce_inflation", seriesId: "PCEPILFE", releaseId: 54, label: "Core PCE Inflation",
    description: "The Federal Reserve's preferred underlying inflation measure, excluding food and energy.",
    unit: "percent", group: "inflation", priority: 1, units: "pc1", limit: 60,
  },
  {
    id: "breakeven_inflation_10y", seriesId: "T10YIE", releaseId: 304, label: "10Y Breakeven Inflation",
    description: "Market-implied average inflation over the next decade from nominal and inflation-protected Treasuries.",
    unit: "percent", group: "inflation", priority: 2, units: "lin", limit: 365,
  },
  {
    id: "producer_price_inflation", seriesId: "PPIFIS", releaseId: 46, label: "Producer Price Inflation",
    description: "Year-over-year change in the Producer Price Index for final demand.",
    unit: "percent", group: "inflation", priority: 3, units: "pc1", limit: 60,
  },
  {
    id: "initial_claims", seriesId: "ICSA", releaseId: 180, label: "Initial Jobless Claims",
    description: "New weekly claims for unemployment insurance, shown in thousands of people.",
    unit: "thousands_level", group: "labor", priority: 1, units: "lin", limit: 156, scale: 0.001,
  },
  {
    id: "average_hourly_earnings_growth", seriesId: "CES0500000003", releaseId: 50, label: "Average Hourly Earnings",
    description: "Year-over-year wage growth for all private nonfarm employees.",
    unit: "percent", group: "labor", priority: 2, units: "pc1", limit: 60,
  },
  {
    id: "labor_force_participation", seriesId: "CIVPART", releaseId: 50, label: "Labor Force Participation",
    description: "Share of the civilian working-age population employed or actively looking for work.",
    unit: "percent", group: "labor", priority: 3, units: "lin", limit: 60,
  },
  {
    id: "job_openings", seriesId: "JTSJOL", releaseId: 192, label: "Job Openings",
    description: "Total nonfarm job openings reported by employers, in thousands.",
    unit: "thousands_level", group: "labor", priority: 4, units: "lin", limit: 60,
  },
  {
    id: "sahm_rule", seriesId: "SAHMREALTIME", releaseId: 456, label: "Sahm Rule Indicator",
    description: "Real-time recession signal; a reading of 0.50 percentage points or more triggers the rule.",
    unit: "percentage_points", group: "labor", priority: 5, units: "lin", limit: 60,
  },
  {
    id: "national_financial_conditions", seriesId: "NFCI", releaseId: 221, label: "National Financial Conditions",
    description: "Chicago Fed index of financial conditions; positive values indicate tighter-than-average conditions.",
    unit: "index", group: "financial", priority: 1, units: "lin", limit: 156,
  },
  {
    id: "financial_stress", seriesId: "STLFSI4", releaseId: 187, label: "Financial Stress Index",
    description: "St. Louis Fed measure of market stress; values above zero indicate above-average stress.",
    unit: "index", group: "financial", priority: 2, units: "lin", limit: 156,
  },
  {
    id: "fed_balance_sheet", seriesId: "WALCL", releaseId: 20, label: "Federal Reserve Assets",
    description: "Total assets held by the Federal Reserve, shown in trillions of dollars.",
    unit: "trillions", group: "financial", priority: 3, units: "lin", limit: 156, scale: 0.000001,
  },
  {
    id: "m2_money_stock", seriesId: "M2SL", releaseId: 21, label: "M2 Money Stock",
    description: "Broad money supply including cash, checking deposits, and readily convertible near money.",
    unit: "trillions", group: "financial", priority: 4, units: "lin", limit: 60, scale: 0.001,
  },
  {
    id: "sofr", seriesId: "SOFR", releaseId: 445, label: "Secured Overnight Financing Rate",
    description: "Broad measure of the cost of borrowing cash overnight collateralized by Treasury securities.",
    unit: "percent", group: "financial", priority: 5, units: "lin", limit: 365,
  },
  {
    id: "credit_spread_baa", seriesId: "BAA10Y", releaseId: 304, label: "Baa Credit Spread",
    description: "Moody's Baa corporate bond yield over the 10-year Treasury; widens as credit risk is repriced.",
    unit: "percentage_points", group: "financial", priority: 7, units: "lin", limit: 365,
  },
  {
    id: "credit_conditions", seriesId: "NFCICREDIT", releaseId: 221, label: "Credit Conditions",
    description: "Chicago Fed credit subindex; positive values indicate tighter-than-average credit conditions.",
    unit: "index", group: "financial", priority: 8, units: "lin", limit: 156,
  },
  {
    id: "real_yield_10y", seriesId: "DFII10", releaseId: 18, label: "10Y Real Yield",
    description: "Inflation-protected 10-year Treasury yield - the rate through which policy actually transmits.",
    unit: "percent", group: "financial", priority: 9, units: "lin", limit: 365,
  },
  {
    id: "trade_weighted_dollar", seriesId: "DTWEXBGS", releaseId: 17, label: "Trade-Weighted US Dollar",
    description: "Broad index of the US dollar against currencies of major trading partners.",
    unit: "index", group: "financial", priority: 6, units: "lin", limit: 365,
  },
  {
    id: "housing_starts", seriesId: "HOUST", releaseId: 27, label: "Housing Starts",
    description: "Annualized pace of newly started privately owned homes, in thousands.",
    unit: "thousands_level", group: "housing", priority: 1, units: "lin", limit: 60,
  },
  {
    id: "building_permits", seriesId: "PERMIT", releaseId: 27, label: "Building Permits",
    description: "Annualized pace of permits for new privately owned housing units, in thousands.",
    unit: "thousands_level", group: "housing", priority: 2, units: "lin", limit: 60,
  },
  {
    id: "mortgage_rate_30y", seriesId: "MORTGAGE30US", releaseId: 190, label: "30Y Mortgage Rate",
    description: "Average US interest rate on a 30-year fixed-rate mortgage.",
    unit: "percent", group: "housing", priority: 3, units: "lin", limit: 156,
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
      return observation.date && Number.isFinite(value) ? [{ date: observation.date, value }] : [];
    })
    .sort((left, right) => left.date.localeCompare(right.date));
}

export function buildMacroIndicator(
  definition: FredMacroDefinition,
  rawPoints: MarketMacroPoint[],
  metadata: { frequency?: string; lastUpdated?: string },
): MarketMacroIndicator {
  const scale = definition.scale ?? 1;
  const points = rawPoints.map((point) => ({ ...point, value: point.value * scale }));
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
    group: definition.group,
    priority: definition.priority,
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

export async function fetchFredMacroIndicators(
  apiKey = String(process.env.FRED_API_KEY ?? "").trim(),
): Promise<MarketMacroIndicator[]> {
  if (!apiKey) throw new Error("FRED_API_KEY is not configured.");

  const results = await Promise.allSettled(
    FRED_MACRO_DEFINITIONS.map((definition) => fetchMacroIndicator(definition, apiKey)),
  );
  const indicators = results.flatMap((result) => result.status === "fulfilled" ? [result.value] : []);
  if (!indicators.length) {
    const firstFailure = results.find((result) => result.status === "rejected");
    throw firstFailure && firstFailure.status === "rejected"
      ? firstFailure.reason
      : new Error("FRED returned no macro indicators.");
  }
  return indicators;
}

import type { MarketMacroIndicator, MarketMacroIndicatorId } from "@/lib/server/types";

export interface Signal {
  text: string;
  alert?: boolean;
}

const direction = (indicator: MarketMacroIndicator, up: string, down: string): Signal => ({
  text: (indicator.change ?? 0) >= 0 ? up : down,
});

export const SIGNALS: Record<MarketMacroIndicatorId, (indicator: MarketMacroIndicator) => Signal> = {
  real_gdp_growth: (indicator) => direction(indicator, "Growth accelerating", "Growth slowing"),
  cpi_inflation: (indicator) => direction(indicator, "Inflation heating", "Inflation cooling"),
  nonfarm_payrolls: (indicator) => ({ text: indicator.value >= 0 ? "Jobs added" : "Jobs lost", alert: indicator.value < 0 }),
  unemployment_rate: (indicator) => direction(indicator, "Labor market softening", "Labor market firming"),
  effective_fed_funds: (indicator) => ({ text: (indicator.change ?? 0) === 0 ? "Policy rate steady" : (indicator.change ?? 0) > 0 ? "Policy tightening" : "Policy easing" }),
  yield_curve_10y2y: (indicator) => ({ text: indicator.value < 0 ? "Yield curve inverted" : "Yield curve positive", alert: indicator.value < 0 }),
  retail_sales_growth: (indicator) => ({ text: indicator.value >= 0 ? "Consumer spending rising" : "Consumer spending falling", alert: indicator.value < 0 }),
  industrial_production_growth: (indicator) => ({ text: indicator.value >= 0 ? "Industrial output rising" : "Industrial output falling", alert: indicator.value < 0 }),
  core_pce_inflation: (indicator) => direction(indicator, "Core inflation heating", "Core inflation cooling"),
  breakeven_inflation_10y: (indicator) => ({ text: indicator.value > 2.5 ? "Expectations elevated" : indicator.value < 1.5 ? "Expectations subdued" : "Expectations near target", alert: indicator.value > 2.5 }),
  producer_price_inflation: (indicator) => direction(indicator, "Pipeline inflation heating", "Pipeline inflation cooling"),
  initial_claims: (indicator) => direction(indicator, "Layoffs rising", "Layoffs easing"),
  average_hourly_earnings_growth: (indicator) => direction(indicator, "Wage growth strengthening", "Wage growth cooling"),
  labor_force_participation: (indicator) => direction(indicator, "Participation rising", "Participation falling"),
  job_openings: (indicator) => direction(indicator, "Labor demand rising", "Labor demand cooling"),
  sahm_rule: (indicator) => ({ text: indicator.value >= 0.5 ? "Recession signal triggered" : "No recession signal", alert: indicator.value >= 0.5 }),
  national_financial_conditions: (indicator) => ({ text: indicator.value > 0 ? "Conditions tighter than average" : "Conditions looser than average", alert: indicator.value > 0 }),
  financial_stress: (indicator) => ({ text: indicator.value > 0 ? "Stress above average" : "Stress below average", alert: indicator.value > 0 }),
  fed_balance_sheet: (indicator) => direction(indicator, "Fed assets expanding", "Fed assets contracting"),
  m2_money_stock: (indicator) => direction(indicator, "Money supply expanding", "Money supply contracting"),
  sofr: (indicator) => direction(indicator, "Overnight funding cost rising", "Overnight funding cost falling"),
  credit_spread_baa: (indicator) => ({ text: (indicator.change ?? 0) >= 0 ? "Credit spreads widening" : "Credit spreads tightening", alert: (indicator.change ?? 0) > 0 }),
  credit_conditions: (indicator) => ({ text: indicator.value > 0 ? "Credit tighter than average" : "Credit looser than average", alert: indicator.value > 0 }),
  real_yield_10y: (indicator) => ({ text: indicator.value < 0 ? "Real yield negative" : (indicator.change ?? 0) >= 0 ? "Real yield rising" : "Real yield falling", alert: indicator.value < 0 }),
  trade_weighted_dollar: (indicator) => direction(indicator, "Dollar strengthening", "Dollar weakening"),
  housing_starts: (indicator) => direction(indicator, "Homebuilding rising", "Homebuilding falling"),
  building_permits: (indicator) => direction(indicator, "Building pipeline rising", "Building pipeline falling"),
  mortgage_rate_30y: (indicator) => direction(indicator, "Housing finance tightening", "Housing finance easing"),
};

/**
 * Resolve an indicator's signal, tolerating an id this bundle has never heard
 * of.
 *
 * SIGNALS is an exhaustive Record over MarketMacroIndicatorId, so adding a
 * union member without a signal still fails to compile - that guarantee is
 * worth keeping. But it is a COMPILE-time guarantee about one build, and the
 * client and the API are not always the same build: a browser holding an older
 * page bundle across a deploy keeps fetching fresh data into old code. When
 * that deploy added indicators, the old bundle looked them up, got undefined,
 * and called it - "ta[n.id] is not a function", the whole Macro tab replaced
 * by an error boundary.
 *
 * Same family as the Python-column/TS-reader deploy-order trap in CLAUDE.md,
 * inverted: here the API gained data ahead of the code that renders it. One
 * unknown indicator now costs its own badge, not the page.
 */
export function signalFor(indicator: MarketMacroIndicator): Signal {
  const resolve = (SIGNALS as Partial<Record<MarketMacroIndicatorId, (i: MarketMacroIndicator) => Signal>>)[
    indicator.id
  ];
  if (resolve) return resolve(indicator);
  if (indicator.change === null || indicator.change === 0) return { text: "Updated" };
  return { text: indicator.change > 0 ? "Rising" : "Falling" };
}


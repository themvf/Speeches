import type { Portfolio, PortfolioHolding, PortfolioQuote, PortfoliosPayload } from "./portfolio-types.ts";

export type HoldingMetrics = {
  cost: number;
  marketValue: number | null;
  gain: number | null;
  gainPct: number | null;
  todayGain: number | null;
};

export type PortfolioMetrics = {
  cost: number;
  marketValue: number | null;
  gain: number | null;
  gainPct: number | null;
  todayGain: number | null;
  pricedHoldings: number;
};

const MAX_PORTFOLIOS = 50;
const MAX_HOLDINGS = 200;

function finitePositive(value: unknown): number | null {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) && number > 0 ? number : null;
}

export function normalizeSymbol(value: unknown): string {
  return String(value ?? "").trim().toUpperCase().replace(/[^A-Z0-9.^=-]/g, "").slice(0, 16);
}

export function normalizeHolding(value: unknown): PortfolioHolding | null {
  if (!value || typeof value !== "object") return null;
  const raw = value as Partial<PortfolioHolding>;
  const symbol = normalizeSymbol(raw.symbol);
  const shares = finitePositive(raw.shares);
  const tradePrice = finitePositive(raw.tradePrice);
  if (!symbol || !shares || !tradePrice) return null;
  const createdAt = String(raw.createdAt || new Date().toISOString());
  return {
    id: String(raw.id || `holding:${symbol}:${createdAt}`),
    symbol,
    shares,
    tradePrice,
    tradeDate: raw.tradeDate ? String(raw.tradeDate).slice(0, 10) : undefined,
    createdAt,
  };
}
export function normalizePortfolio(value: unknown): Portfolio | null {
  if (!value || typeof value !== "object") return null;
  const raw = value as Partial<Portfolio>;
  const name = String(raw.name ?? "").replace(/\s+/g, " ").trim().slice(0, 80);
  const id = String(raw.id ?? "").trim();
  if (!name || !id) return null;
  const createdAt = String(raw.createdAt || new Date().toISOString());
  return {
    id,
    name,
    holdings: (Array.isArray(raw.holdings) ? raw.holdings : [])
      .map(normalizeHolding)
      .filter((holding): holding is PortfolioHolding => Boolean(holding))
      .slice(0, MAX_HOLDINGS),
    createdAt,
    updatedAt: String(raw.updatedAt || createdAt),
  };
}

export function normalizePortfoliosPayload(value: unknown): PortfoliosPayload {
  const raw = value && typeof value === "object" ? value as Partial<PortfoliosPayload> : {};
  return {
    portfolios: (Array.isArray(raw.portfolios) ? raw.portfolios : [])
      .map(normalizePortfolio)
      .filter((portfolio): portfolio is Portfolio => Boolean(portfolio))
      .slice(0, MAX_PORTFOLIOS),
    updatedAt: String(raw.updatedAt || new Date(0).toISOString()),
  };
}

export function holdingMetrics(holding: PortfolioHolding, quote?: PortfolioQuote): HoldingMetrics {
  const cost = holding.shares * holding.tradePrice;
  if (!quote) return { cost, marketValue: null, gain: null, gainPct: null, todayGain: null };
  const marketValue = holding.shares * quote.price;
  const gain = marketValue - cost;
  return {
    cost,
    marketValue,
    gain,
    gainPct: cost ? (gain / cost) * 100 : null,
    todayGain: holding.shares * quote.change,
  };
}

export function portfolioMetrics(portfolio: Portfolio, quotes: Map<string, PortfolioQuote>): PortfolioMetrics {
  let cost = 0;
  let marketValue = 0;
  let todayGain = 0;
  let pricedHoldings = 0;
  for (const holding of portfolio.holdings) {
    const metrics = holdingMetrics(holding, quotes.get(holding.symbol));
    cost += metrics.cost;
    if (metrics.marketValue != null) {
      marketValue += metrics.marketValue;
      todayGain += metrics.todayGain ?? 0;
      pricedHoldings += 1;
    }
  }
  const complete = pricedHoldings === portfolio.holdings.length && portfolio.holdings.length > 0;
  const gain = complete ? marketValue - cost : null;
  return {
    cost,
    marketValue: complete ? marketValue : null,
    gain,
    gainPct: gain == null || !cost ? null : (gain / cost) * 100,
    todayGain: complete ? todayGain : null,
    pricedHoldings,
  };
}
// End portfolio calculations.

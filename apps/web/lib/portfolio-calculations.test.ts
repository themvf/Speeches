import assert from "node:assert/strict";
import test from "node:test";
import { holdingMetrics, normalizePortfoliosPayload, portfolioMetrics } from "./portfolio-calculations.ts";
import type { Portfolio, PortfolioQuote } from "./portfolio-types.ts";

const portfolio: Portfolio = {
  id: "p1", name: "Core", createdAt: "2026-01-01", updatedAt: "2026-01-01",
  holdings: [
    { id: "h1", symbol: "AAPL", shares: 10, tradePrice: 100, createdAt: "2026-01-01" },
    { id: "h2", symbol: "MSFT", shares: 5, tradePrice: 200, createdAt: "2026-01-01" },
  ],
};
const quote: PortfolioQuote = {
  symbol: "AAPL", name: "Apple", price: 120, change: 2, pct: 1.69, status: "OPEN",
  asOf: "2026-01-02T21:00:00.000Z", currency: "USD", provider: "Yahoo Finance",
};

test("calculates cost basis, unrealized return, and today's change", () => {
  assert.deepEqual(holdingMetrics(portfolio.holdings[0], quote), {
    cost: 1000, marketValue: 1200, gain: 200, gainPct: 20, todayGain: 20,
  });
});

test("does not present a partial quote total as the whole portfolio", () => {
  const metrics = portfolioMetrics(portfolio, new Map([["AAPL", quote]]));
  assert.equal(metrics.cost, 2000);
  assert.equal(metrics.marketValue, null);
  assert.equal(metrics.gain, null);
  assert.equal(metrics.pricedHoldings, 1);
});

test("normalizes symbols and rejects invalid position values", () => {
  const payload = normalizePortfoliosPayload({
    portfolios: [{ ...portfolio, holdings: [
      { ...portfolio.holdings[0], symbol: " aapl$ " },
      { ...portfolio.holdings[1], shares: 0 },
    ] }],
    updatedAt: "2026-01-02",
  });
  assert.equal(payload.portfolios[0].holdings.length, 1);
  assert.equal(payload.portfolios[0].holdings[0].symbol, "AAPL");
});

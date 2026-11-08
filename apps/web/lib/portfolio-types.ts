export interface PortfolioHolding {
  id: string;
  symbol: string;
  shares: number;
  tradePrice: number;
  tradeDate?: string;
  createdAt: string;
}

export interface Portfolio {
  id: string;
  name: string;
  holdings: PortfolioHolding[];
  createdAt: string;
  updatedAt: string;
}

export interface PortfoliosPayload {
  portfolios: Portfolio[];
  updatedAt: string;
}

export interface PortfolioQuote {
  symbol: string;
  name: string;
  price: number;
  change: number;
  pct: number;
  status: "OPEN" | "PRE" | "AFTER" | "CLOSED";
  asOf: string;
  currency: string;
  provider: "Yahoo Finance";
}

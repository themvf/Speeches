export type MarketSectorCompany = {
  symbol: string;
  name: string;
  aliases?: string[];
};

export const MARKET_SECTOR_COMPANIES: Record<string, MarketSectorCompany[]> = {
  "Technology": [
    { symbol: "NVDA", name: "Nvidia" }, { symbol: "AAPL", name: "Apple" },
    { symbol: "MSFT", name: "Microsoft" }, { symbol: "AVGO", name: "Broadcom" },
    { symbol: "AMD", name: "AMD", aliases: ["Advanced Micro Devices"] },
    { symbol: "MU", name: "Micron", aliases: ["Micron Technology"] },
    { symbol: "INTC", name: "Intel" }, { symbol: "CSCO", name: "Cisco", aliases: ["Cisco Systems"] },
    { symbol: "AMAT", name: "Applied Materials" }, { symbol: "LRCX", name: "Lam Research" },
  ],
  "Communication Services": [
    { symbol: "GOOGL", name: "Alphabet", aliases: ["Google"] },
    { symbol: "META", name: "Meta", aliases: ["Meta Platforms", "Facebook"] },
    { symbol: "NFLX", name: "Netflix" }, { symbol: "TMUS", name: "T-Mobile", aliases: ["T-Mobile US"] },
    { symbol: "VZ", name: "Verizon", aliases: ["Verizon Communications"] }, { symbol: "T", name: "AT&T" },
    { symbol: "DIS", name: "Disney", aliases: ["Walt Disney"] }, { symbol: "CMCSA", name: "Comcast" },
    { symbol: "CHTR", name: "Charter", aliases: ["Charter Communications"] }, { symbol: "EA", name: "Electronic Arts" },
  ],
  "Consumer Cyclical": [
    { symbol: "AMZN", name: "Amazon" }, { symbol: "TSLA", name: "Tesla" },
    { symbol: "HD", name: "Home Depot" }, { symbol: "MCD", name: "McDonald's" },
    { symbol: "TJX", name: "TJX", aliases: ["TJX Companies"] }, { symbol: "BKNG", name: "Booking Holdings" },
    { symbol: "SBUX", name: "Starbucks" }, { symbol: "LOW", name: "Lowe's" },
    { symbol: "MAR", name: "Marriott", aliases: ["Marriott International"] }, { symbol: "HLT", name: "Hilton", aliases: ["Hilton Worldwide"] },
  ],
  "Consumer Defensive": [
    { symbol: "WMT", name: "Walmart" }, { symbol: "COST", name: "Costco" },
    { symbol: "PG", name: "P&G", aliases: ["Procter & Gamble"] }, { symbol: "KO", name: "Coca-Cola" },
    { symbol: "PM", name: "Philip Morris" }, { symbol: "PEP", name: "PepsiCo" },
    { symbol: "MDLZ", name: "Mondelez" }, { symbol: "MO", name: "Altria" },
    { symbol: "CL", name: "Colgate-Palmolive" }, { symbol: "TGT", name: "Target" },
  ],
  "Energy": [
    { symbol: "XOM", name: "Exxon Mobil" }, { symbol: "CVX", name: "Chevron" },
    { symbol: "COP", name: "ConocoPhillips" }, { symbol: "EOG", name: "EOG Resources" },
    { symbol: "SLB", name: "SLB", aliases: ["Schlumberger"] }, { symbol: "WMB", name: "Williams", aliases: ["Williams Companies"] },
    { symbol: "MPC", name: "Marathon Petroleum" }, { symbol: "OKE", name: "ONEOK" },
    { symbol: "PSX", name: "Phillips 66" }, { symbol: "VLO", name: "Valero" },
  ],
  "Financial Services": [
    { symbol: "BRK-B", name: "Berkshire Hathaway" }, { symbol: "JPM", name: "JPMorgan", aliases: ["JPMorgan Chase"] },
    { symbol: "V", name: "Visa" }, { symbol: "MA", name: "Mastercard" },
    { symbol: "BAC", name: "Bank of America" }, { symbol: "WFC", name: "Wells Fargo" },
    { symbol: "GS", name: "Goldman Sachs" }, { symbol: "MS", name: "Morgan Stanley" },
    { symbol: "AXP", name: "American Express" }, { symbol: "SPGI", name: "S&P Global" },
  ],
  "Healthcare": [
    { symbol: "LLY", name: "Eli Lilly" }, { symbol: "JNJ", name: "Johnson & Johnson", aliases: ["J&J"] },
    { symbol: "ABBV", name: "AbbVie" }, { symbol: "UNH", name: "UnitedHealth", aliases: ["UnitedHealth Group"] },
    { symbol: "MRK", name: "Merck" }, { symbol: "ABT", name: "Abbott", aliases: ["Abbott Laboratories"] },
    { symbol: "TMO", name: "Thermo Fisher", aliases: ["Thermo Fisher Scientific"] }, { symbol: "ISRG", name: "Intuitive Surgical" },
    { symbol: "AMGN", name: "Amgen" }, { symbol: "DHR", name: "Danaher" },
  ],
  "Industrials": [
    { symbol: "GE", name: "GE Aerospace" }, { symbol: "CAT", name: "Caterpillar" },
    { symbol: "RTX", name: "RTX", aliases: ["RTX Corporation", "Raytheon Technologies"] }, { symbol: "UBER", name: "Uber" },
    { symbol: "UNP", name: "Union Pacific" }, { symbol: "HON", name: "Honeywell" },
    { symbol: "BA", name: "Boeing" }, { symbol: "ETN", name: "Eaton" },
    { symbol: "DE", name: "Deere", aliases: ["John Deere"] }, { symbol: "ADP", name: "ADP", aliases: ["Automatic Data Processing"] },
  ],
  "Basic Materials": [
    { symbol: "LIN", name: "Linde" }, { symbol: "SHW", name: "Sherwin-Williams" },
    { symbol: "FCX", name: "Freeport-McMoRan" }, { symbol: "NEM", name: "Newmont" },
    { symbol: "ECL", name: "Ecolab" }, { symbol: "CTVA", name: "Corteva" },
    { symbol: "APD", name: "Air Products" }, { symbol: "VMC", name: "Vulcan Materials" },
    { symbol: "MLM", name: "Martin Marietta" }, { symbol: "NUE", name: "Nucor" },
  ],
  "Real Estate": [
    { symbol: "PLD", name: "Prologis" }, { symbol: "AMT", name: "American Tower" },
    { symbol: "EQIX", name: "Equinix" }, { symbol: "WELL", name: "Welltower" },
    { symbol: "SPG", name: "Simon Property", aliases: ["Simon Property Group"] }, { symbol: "DLR", name: "Digital Realty" },
    { symbol: "O", name: "Realty Income" }, { symbol: "PSA", name: "Public Storage" },
    { symbol: "CBRE", name: "CBRE" }, { symbol: "CCI", name: "Crown Castle" },
  ],
  "Utilities": [
    { symbol: "NEE", name: "NextEra Energy" }, { symbol: "SO", name: "Southern Co.", aliases: ["Southern Company"] },
    { symbol: "DUK", name: "Duke Energy" }, { symbol: "CEG", name: "Constellation Energy" },
    { symbol: "AEP", name: "American Electric Power" }, { symbol: "SRE", name: "Sempra" },
    { symbol: "VST", name: "Vistra" }, { symbol: "D", name: "Dominion Energy" },
    { symbol: "EXC", name: "Exelon" }, { symbol: "PEG", name: "PSEG", aliases: ["Public Service Enterprise Group"] },
  ],
};

const COMPANY_BY_SYMBOL = new Map(
  Object.values(MARKET_SECTOR_COMPANIES)
    .flat()
    .map((company) => [company.symbol.toUpperCase(), company])
);

export function findMarketSectorCompany(symbol: string): MarketSectorCompany | null {
  return COMPANY_BY_SYMBOL.get(String(symbol || "").trim().toUpperCase()) ?? null;
}

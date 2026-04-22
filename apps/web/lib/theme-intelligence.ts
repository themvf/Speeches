export type ThemeCategory = "MACRO" | "FINANCIAL_SYSTEM" | "GEOPOLITICS" | "REAL_ECONOMY" | "MODERN_THEMES";

export type ProductCategory =
  | "SECURITIES_REGULATION"
  | "CAPITAL_FORMATION"
  | "AML"
  | "ENFORCEMENT"
  | "AI_TECH"
  | "CRYPTO"
  | "CREDIT_MARKETS"
  | "FINANCIAL_MARKETS"
  | "ECONOMIC_GROWTH";

export type NormalizedTheme =
  | "INFLATION"
  | "INTEREST_RATES"
  | "CENTRAL_BANK"
  | "ECONOMIC_GROWTH"
  | "LABOR_MARKET"
  | "BANKING"
  | "CREDIT_MARKETS"
  | "LIQUIDITY"
  | "FINANCIAL_MARKETS"
  | "REGULATION"
  | "GEOPOLITICS"
  | "CONFLICT"
  | "TRADE"
  | "SANCTIONS"
  | "ENERGY"
  | "SUPPLY_CHAIN"
  | "COMMODITIES"
  | "TECHNOLOGY"
  | "CRYPTO"
  | "CORPORATE_ACTIVITY"
  | "AI";

export type ThemeSeverity = "CRITICAL" | "HIGH" | "NORMAL";
export type SignalTrendDirection = "ACCELERATING" | "RISING" | "STABLE" | "FALLING";
export type MarketImpactDirection = "UP" | "DOWN" | "MIXED";

export type RawThemeInput = string | readonly string[];

export interface ThemeDefinition {
  normalized_theme: NormalizedTheme;
  category: ThemeCategory;
  weight: number;
  raw_patterns: readonly string[];
}

export interface ThemeMatch {
  normalized_theme: NormalizedTheme;
  category: ThemeCategory;
  weight: number;
  matched_raw_themes: string[];
  matched_patterns: string[];
}

export interface ThemeCombinationSignal {
  id: string;
  label: string;
  themes: NormalizedTheme[];
}

export interface ThemeArticleInput {
  id?: string;
  title?: string;
  raw_themes: RawThemeInput;
  context?: ThemeContextInput;
}

export interface ThemeFrequencyContext {
  current_mentions?: number;
  baseline_mentions?: number;
  previous_mentions?: number;
}

export interface ThemeContextInput {
  window_label?: string;
  previous_score?: number;
  baseline_score?: number;
  confidence?: number;
  theme_mentions?: Partial<Record<NormalizedTheme, ThemeFrequencyContext>>;
}

export interface ThemeTrendSignal {
  window_label: string;
  current_score: number;
  previous_score: number;
  baseline_score: number;
  delta: number;
  delta_pct: number;
  direction: SignalTrendDirection;
}

export interface ThemeFrequencySignal {
  normalized_theme: NormalizedTheme;
  category: ThemeCategory;
  weight: number;
  current_mentions: number;
  baseline_mentions: number;
  previous_mentions: number;
  delta_mentions: number;
  spike_pct: number;
  anomaly_score: number;
  weighted_intensity: number;
  theme_score: number;
  contribution_pct: number;
  rank: number;
  classification: "PRIMARY" | "SECONDARY" | "BACKGROUND";
  matched_raw_themes: string[];
  matched_patterns: string[];
}

export interface ProductSubcategorySignal {
  label: string;
  score: number;
  contribution_pct: number;
  rank: number;
  themes: NormalizedTheme[];
  matched_raw_themes: string[];
  matched_patterns: string[];
}

export interface ProductCategorySignal {
  category: ProductCategory;
  label: string;
  score: number;
  contribution_pct: number;
  rank: number;
  subcategories: ProductSubcategorySignal[];
  themes: NormalizedTheme[];
  matched_raw_themes: string[];
  matched_patterns: string[];
}

export interface ProductFocusArea {
  id: string;
  category: ProductCategory;
  label: string;
  raw_patterns: readonly string[];
  normalized_themes: readonly NormalizedTheme[];
}

export interface SignalModel {
  severity_score: number;
  trend_score: number;
  breadth_score: number;
  confidence_score: number;
  composite_score: number;
}

export interface SignalInterpretation {
  headline: string;
  drivers: string[];
  implications: string[];
}

export interface MarketImpactSignal {
  asset: string;
  direction: MarketImpactDirection;
  themes: NormalizedTheme[];
  rationale: string;
  confidence: number;
}

export interface ThemeSignal {
  id?: string;
  title?: string;
  raw_themes: string[];
  full_theme_list: string[];
  normalized_themes: NormalizedTheme[];
  normalized_theme_list: NormalizedTheme[];
  theme_weights: Partial<Record<NormalizedTheme, number>>;
  total_score: number;
  total_signal_score: number;
  severity: ThemeSeverity;
  matches: ThemeMatch[];
  category_scores: Partial<Record<ThemeCategory, number>>;
  product_category_scores: Partial<Record<ProductCategory, number>>;
  product_categories: ProductCategorySignal[];
  primary_product_category: ProductCategorySignal | null;
  secondary_product_categories: ProductCategorySignal[];
  market_mappings: Partial<Record<NormalizedTheme, string[]>>;
  combination_signals: ThemeCombinationSignal[];
  trend: ThemeTrendSignal;
  frequency_signals: ThemeFrequencySignal[];
  primary_driver: ThemeFrequencySignal | null;
  primary_drivers: ThemeFrequencySignal[];
  secondary_drivers: ThemeFrequencySignal[];
  background_context: ThemeFrequencySignal[];
  signal_model: SignalModel;
  interpretation: SignalInterpretation;
  market_impacts: MarketImpactSignal[];
}

export const THEME_MAPPING: readonly ThemeDefinition[] = [
  { normalized_theme: "INFLATION", category: "MACRO", weight: 10, raw_patterns: ["ECON_INFLATION", "INFLATION", "CPI", "PRICES"] },
  { normalized_theme: "INTEREST_RATES", category: "MACRO", weight: 10, raw_patterns: ["ECON_INTEREST_RATES", "FED", "RATE_HIKE", "RATE_CUT"] },
  { normalized_theme: "CENTRAL_BANK", category: "MACRO", weight: 10, raw_patterns: ["CENTRAL_BANK", "FEDERAL_RESERVE", "ECB", "MONETARY_POLICY"] },
  { normalized_theme: "ECONOMIC_GROWTH", category: "MACRO", weight: 9, raw_patterns: ["ECON_GROWTH", "GDP", "RECESSION", "SLOWDOWN"] },
  { normalized_theme: "LABOR_MARKET", category: "MACRO", weight: 8, raw_patterns: ["UNEMPLOYMENT", "JOBS", "WAGES", "LABOR"] },
  { normalized_theme: "BANKING", category: "FINANCIAL_SYSTEM", weight: 10, raw_patterns: ["ECON_BANKING", "BANK_FAILURE", "BANK_RUN"] },
  { normalized_theme: "CREDIT_MARKETS", category: "FINANCIAL_SYSTEM", weight: 9, raw_patterns: ["ECON_CREDIT", "DEBT", "DEFAULT", "BOND_MARKET"] },
  { normalized_theme: "LIQUIDITY", category: "FINANCIAL_SYSTEM", weight: 9, raw_patterns: ["LIQUIDITY", "FUNDING_STRESS", "CASH_FLOW"] },
  { normalized_theme: "FINANCIAL_MARKETS", category: "FINANCIAL_SYSTEM", weight: 8, raw_patterns: ["FINANCIAL_MARKET", "STOCK_MARKET", "EQUITIES"] },
  { normalized_theme: "REGULATION", category: "FINANCIAL_SYSTEM", weight: 7, raw_patterns: ["REGULATION", "SEC", "POLICY", "COMPLIANCE"] },
  { normalized_theme: "GEOPOLITICS", category: "GEOPOLITICS", weight: 9, raw_patterns: ["GEOPOLITICAL", "DIPLOMACY", "FOREIGN_POLICY"] },
  { normalized_theme: "CONFLICT", category: "GEOPOLITICS", weight: 10, raw_patterns: ["WAR", "MILITARY", "ATTACK", "DEFENSE"] },
  { normalized_theme: "TRADE", category: "GEOPOLITICS", weight: 8, raw_patterns: ["TRADE", "EXPORTS", "IMPORTS", "TARIFFS"] },
  { normalized_theme: "SANCTIONS", category: "GEOPOLITICS", weight: 9, raw_patterns: ["SANCTIONS", "EMBARGO", "RESTRICTIONS"] },
  { normalized_theme: "ENERGY", category: "REAL_ECONOMY", weight: 10, raw_patterns: ["OIL", "NATURAL_GAS", "ENERGY_SUPPLY", "OPEC"] },
  { normalized_theme: "SUPPLY_CHAIN", category: "REAL_ECONOMY", weight: 9, raw_patterns: ["SUPPLY_CHAIN", "LOGISTICS", "SHIPPING"] },
  { normalized_theme: "COMMODITIES", category: "REAL_ECONOMY", weight: 8, raw_patterns: ["GOLD", "METALS", "AGRICULTURE", "RAW_MATERIALS"] },
  { normalized_theme: "TECHNOLOGY", category: "MODERN_THEMES", weight: 7, raw_patterns: ["TECHNOLOGY", "SEMICONDUCTOR", "SOFTWARE"] },
  { normalized_theme: "CRYPTO", category: "MODERN_THEMES", weight: 7, raw_patterns: ["CRYPTOCURRENCY", "BITCOIN", "BLOCKCHAIN"] },
  { normalized_theme: "CORPORATE_ACTIVITY", category: "MODERN_THEMES", weight: 6, raw_patterns: ["EARNINGS", "MERGERS", "ACQUISITIONS", "LAYOFFS"] },
  { normalized_theme: "AI", category: "MODERN_THEMES", weight: 8, raw_patterns: ["AI", "ARTIFICIAL_INTELLIGENCE", "MACHINE_LEARNING", "DEEP_LEARNING", "LLM", "GENERATIVE_AI"] }
];

export const PRODUCT_CATEGORY_ORDER: readonly ProductCategory[] = [
  "SECURITIES_REGULATION",
  "CAPITAL_FORMATION",
  "AML",
  "ENFORCEMENT",
  "AI_TECH",
  "CRYPTO",
  "CREDIT_MARKETS",
  "FINANCIAL_MARKETS",
  "ECONOMIC_GROWTH"
];

export const PRODUCT_CATEGORY_LABELS: Readonly<Record<ProductCategory, string>> = {
  SECURITIES_REGULATION: "Securities Regulation",
  CAPITAL_FORMATION: "Capital Formation",
  AML: "AML",
  ENFORCEMENT: "Enforcement",
  AI_TECH: "AI & Tech",
  CRYPTO: "Crypto",
  CREDIT_MARKETS: "Credit Markets",
  FINANCIAL_MARKETS: "Financial Markets",
  ECONOMIC_GROWTH: "Economic Growth"
};

const PRODUCT_CATEGORY_DISPLAY_THEMES: Readonly<Record<ProductCategory, readonly NormalizedTheme[]>> = {
  SECURITIES_REGULATION: ["REGULATION", "FINANCIAL_MARKETS", "CORPORATE_ACTIVITY"],
  CAPITAL_FORMATION: ["CORPORATE_ACTIVITY", "FINANCIAL_MARKETS", "CREDIT_MARKETS"],
  AML: ["SANCTIONS"],
  ENFORCEMENT: ["REGULATION", "FINANCIAL_MARKETS", "CRYPTO"],
  AI_TECH: ["AI", "TECHNOLOGY"],
  CRYPTO: ["CRYPTO", "REGULATION"],
  CREDIT_MARKETS: ["BANKING", "INTEREST_RATES", "CREDIT_MARKETS", "LIQUIDITY"],
  FINANCIAL_MARKETS: ["FINANCIAL_MARKETS", "COMMODITIES", "GEOPOLITICS", "CONFLICT"],
  ECONOMIC_GROWTH: ["INFLATION", "CENTRAL_BANK", "ECONOMIC_GROWTH", "LABOR_MARKET", "TRADE", "SUPPLY_CHAIN", "ENERGY"]
};

export const PRODUCT_FOCUS_AREAS: readonly ProductFocusArea[] = [
  {
    id: "aml_sanctions",
    category: "AML",
    label: "Sanctions",
    raw_patterns: ["SANCTIONS", "OFAC"],
    normalized_themes: []
  },
  {
    id: "aml_bsa",
    category: "AML",
    label: "AML / BSA",
    raw_patterns: ["AML", "BSA", "FINCEN", "MONEY_LAUNDERING", "ANTI_MONEY_LAUNDERING"],
    normalized_themes: []
  },
  {
    id: "aml_kyc_ownership",
    category: "AML",
    label: "KYC / Ownership",
    raw_patterns: ["KYC", "CIP", "CUSTOMER_IDENTIFICATION", "BENEFICIAL_OWNERSHIP"],
    normalized_themes: []
  },
  {
    id: "aml_illicit_finance",
    category: "AML",
    label: "Illicit Finance",
    raw_patterns: ["ILLICIT_FINANCE", "SUSPICIOUS_ACTIVITY", "SAR", "TERRORIST_FINANCING"],
    normalized_themes: []
  }
];

type ProductSubcategoryDefinition = {
  label: string;
  normalized_themes: readonly NormalizedTheme[];
  raw_patterns: readonly string[];
  weight: number;
};

type ProductCategoryDefinition = {
  category: ProductCategory;
  label: string;
  subcategories: readonly ProductSubcategoryDefinition[];
};

export const PRODUCT_TAXONOMY: readonly ProductCategoryDefinition[] = [
  {
    category: "SECURITIES_REGULATION",
    label: PRODUCT_CATEGORY_LABELS.SECURITIES_REGULATION,
    subcategories: [
      { label: "Rulemaking / Compliance", normalized_themes: ["REGULATION"], raw_patterns: ["SEC", "REGULATION", "COMPLIANCE", "POLICY", "RULEMAKING"], weight: 7 },
      { label: "Disclosure", normalized_themes: [], raw_patterns: ["DISCLOSURE", "REPORTING", "10_K", "10_Q", "8_K", "PUBLIC_COMPANY"], weight: 7 },
      { label: "Market Structure", normalized_themes: [], raw_patterns: ["MARKET_STRUCTURE", "EXCHANGE", "TRADING_RULES", "CLEARING", "SETTLEMENT"], weight: 8 },
      { label: "Broker-Dealer / Adviser Regulation", normalized_themes: [], raw_patterns: ["BROKER_DEALER", "INVESTMENT_ADVISER", "ADVISER", "FIDUCIARY"], weight: 7 },
      { label: "Corporate Governance", normalized_themes: [], raw_patterns: ["CORPORATE_GOVERNANCE", "PROXY", "BOARD", "SHAREHOLDER"], weight: 6 }
    ]
  },
  {
    category: "CAPITAL_FORMATION",
    label: PRODUCT_CATEGORY_LABELS.CAPITAL_FORMATION,
    subcategories: [
      { label: "IPOs / Offerings", normalized_themes: [], raw_patterns: ["IPO", "OFFERING", "EQUITY_OFFERING", "DEBT_OFFERING", "SECURITIES_OFFERING"], weight: 8 },
      { label: "Private Markets", normalized_themes: [], raw_patterns: ["PRIVATE_MARKETS", "PRIVATE_CREDIT", "PRIVATE_EQUITY", "VENTURE_CAPITAL"], weight: 8 },
      { label: "Fundraising", normalized_themes: [], raw_patterns: ["CAPITAL_RAISE", "FUNDRAISING", "FINANCING", "CAPITAL_FORMATION"], weight: 8 },
      { label: "M&A / SPACs", normalized_themes: [], raw_patterns: ["MERGERS", "ACQUISITIONS", "SPAC", "DEALMAKING"], weight: 7 }
    ]
  },
  {
    category: "AML",
    label: PRODUCT_CATEGORY_LABELS.AML,
    subcategories: [
      { label: "AML / BSA", normalized_themes: [], raw_patterns: ["AML", "BSA", "FINCEN", "MONEY_LAUNDERING", "ANTI_MONEY_LAUNDERING"], weight: 9 },
      { label: "KYC / Beneficial Ownership", normalized_themes: [], raw_patterns: ["KYC", "CIP", "CUSTOMER_IDENTIFICATION", "BENEFICIAL_OWNERSHIP"], weight: 8 },
      { label: "Sanctions Compliance", normalized_themes: [], raw_patterns: ["SANCTIONS", "OFAC"], weight: 9 },
      { label: "Illicit Finance", normalized_themes: [], raw_patterns: ["ILLICIT_FINANCE", "SUSPICIOUS_ACTIVITY", "SAR", "TERRORIST_FINANCING"], weight: 9 }
    ]
  },
  {
    category: "ENFORCEMENT",
    label: PRODUCT_CATEGORY_LABELS.ENFORCEMENT,
    subcategories: [
      { label: "SEC / FINRA Enforcement", normalized_themes: [], raw_patterns: ["ENFORCEMENT", "SEC_ACTION", "FINRA", "DISCIPLINARY", "VIOLATION"], weight: 9 },
      { label: "Fraud / Manipulation", normalized_themes: [], raw_patterns: ["FRAUD", "MARKET_MANIPULATION", "INSIDER_TRADING", "MISREPRESENTATION"], weight: 10 },
      { label: "Penalties / Settlements", normalized_themes: [], raw_patterns: ["PENALTY", "SETTLEMENT", "FINE", "SANCTIONED"], weight: 8 },
      { label: "Litigation / Investigations", normalized_themes: [], raw_patterns: ["LITIGATION", "INVESTIGATION", "DOJ", "LAWSUIT", "CHARGES"], weight: 8 }
    ]
  },
  {
    category: "AI_TECH",
    label: PRODUCT_CATEGORY_LABELS.AI_TECH,
    subcategories: [
      { label: "Artificial Intelligence", normalized_themes: ["AI"], raw_patterns: ["AI", "ARTIFICIAL_INTELLIGENCE", "GENERATIVE_AI", "MACHINE_LEARNING", "DEEP_LEARNING", "LLM"], weight: 8 },
      { label: "Semiconductors", normalized_themes: ["TECHNOLOGY", "AI"], raw_patterns: ["SEMICONDUCTOR", "CHIP", "GPU", "ACCELERATOR"], weight: 8 },
      { label: "Software / Cloud", normalized_themes: ["TECHNOLOGY"], raw_patterns: ["SOFTWARE", "CLOUD", "DATA_CENTER", "INFRASTRUCTURE"], weight: 7 },
      { label: "Cybersecurity", normalized_themes: ["TECHNOLOGY"], raw_patterns: ["CYBERSECURITY", "DATA_SECURITY", "HACK", "BREACH"], weight: 7 }
    ]
  },
  {
    category: "CRYPTO",
    label: PRODUCT_CATEGORY_LABELS.CRYPTO,
    subcategories: [
      { label: "Digital Assets", normalized_themes: ["CRYPTO"], raw_patterns: ["CRYPTOCURRENCY", "DIGITAL_ASSET", "TOKEN", "TOKENIZATION"], weight: 7 },
      { label: "Bitcoin / Ethereum", normalized_themes: ["CRYPTO"], raw_patterns: ["BITCOIN", "BTC", "ETHEREUM", "ETH"], weight: 7 },
      { label: "Stablecoins / DeFi", normalized_themes: ["CRYPTO"], raw_patterns: ["STABLECOIN", "DEFI", "DECENTRALIZED_FINANCE"], weight: 7 },
      { label: "Custody / Exchanges", normalized_themes: ["CRYPTO"], raw_patterns: ["CRYPTO_EXCHANGE", "CUSTODY", "WALLET", "BLOCKCHAIN"], weight: 7 }
    ]
  },
  {
    category: "CREDIT_MARKETS",
    label: PRODUCT_CATEGORY_LABELS.CREDIT_MARKETS,
    subcategories: [
      { label: "Interest Rates", normalized_themes: ["INTEREST_RATES"], raw_patterns: ["ECON_INTEREST_RATES", "RATE_HIKE", "RATE_CUT", "YIELDS"], weight: 10 },
      { label: "Credit Spreads / Debt", normalized_themes: ["CREDIT_MARKETS"], raw_patterns: ["ECON_CREDIT", "CREDIT_SPREADS", "DEBT", "BOND_MARKET"], weight: 9 },
      { label: "Defaults", normalized_themes: ["CREDIT_MARKETS"], raw_patterns: ["DEFAULT", "BANKRUPTCY", "DISTRESSED_DEBT"], weight: 9 },
      { label: "Banking Stress", normalized_themes: ["BANKING"], raw_patterns: ["ECON_BANKING", "BANK_FAILURE", "BANK_RUN", "REGIONAL_BANK"], weight: 10 },
      { label: "Liquidity / Funding", normalized_themes: ["LIQUIDITY"], raw_patterns: ["LIQUIDITY", "FUNDING_STRESS", "CASH_FLOW"], weight: 9 }
    ]
  },
  {
    category: "FINANCIAL_MARKETS",
    label: PRODUCT_CATEGORY_LABELS.FINANCIAL_MARKETS,
    subcategories: [
      { label: "Equities / Market Structure", normalized_themes: ["FINANCIAL_MARKETS"], raw_patterns: ["FINANCIAL_MARKET", "STOCK_MARKET", "EQUITIES", "ETF", "INDEX"], weight: 8 },
      { label: "Volatility / Risk Sentiment", normalized_themes: ["FINANCIAL_MARKETS", "GEOPOLITICS", "CONFLICT"], raw_patterns: ["VOLATILITY", "VIX", "RISK_SENTIMENT", "GEOPOLITICAL", "WAR", "ATTACK"], weight: 8 },
      { label: "Commodities", normalized_themes: ["COMMODITIES"], raw_patterns: ["GOLD", "METALS", "AGRICULTURE", "RAW_MATERIALS", "COMMODITIES"], weight: 8 },
      { label: "Derivatives / Cross-Asset", normalized_themes: ["FINANCIAL_MARKETS"], raw_patterns: ["DERIVATIVES", "OPTIONS", "FUTURES", "CROSS_ASSET"], weight: 8 }
    ]
  },
  {
    category: "ECONOMIC_GROWTH",
    label: PRODUCT_CATEGORY_LABELS.ECONOMIC_GROWTH,
    subcategories: [
      { label: "GDP / Recession", normalized_themes: ["ECONOMIC_GROWTH"], raw_patterns: ["ECON_GROWTH", "GDP", "RECESSION", "SLOWDOWN"], weight: 9 },
      { label: "Inflation", normalized_themes: ["INFLATION"], raw_patterns: ["ECON_INFLATION", "INFLATION", "CPI", "PRICES"], weight: 10 },
      { label: "Labor Market", normalized_themes: ["LABOR_MARKET"], raw_patterns: ["UNEMPLOYMENT", "JOBS", "WAGES", "LABOR"], weight: 8 },
      { label: "Monetary Policy", normalized_themes: ["CENTRAL_BANK", "INTEREST_RATES"], raw_patterns: ["CENTRAL_BANK", "FEDERAL_RESERVE", "ECB", "MONETARY_POLICY"], weight: 9 },
      { label: "Trade / Supply Chain", normalized_themes: ["TRADE", "SUPPLY_CHAIN"], raw_patterns: ["TRADE", "EXPORTS", "IMPORTS", "TARIFFS", "SUPPLY_CHAIN", "SHIPPING", "LOGISTICS"], weight: 8 },
      { label: "Energy Prices", normalized_themes: ["ENERGY"], raw_patterns: ["OIL", "NATURAL_GAS", "ENERGY_SUPPLY", "OPEC"], weight: 9 }
    ]
  }
];

export const THEME_WEIGHTS: Readonly<Record<NormalizedTheme, number>> = THEME_MAPPING.reduce(
  (acc, theme) => ({ ...acc, [theme.normalized_theme]: theme.weight }),
  {} as Record<NormalizedTheme, number>
);

export const MARKET_THEME_MAP: Readonly<Partial<Record<NormalizedTheme, readonly string[]>>> = {
  INFLATION: ["TIP", "GLD", "TLT"],
  INTEREST_RATES: ["TLT", "IEF", "XLF"],
  CENTRAL_BANK: ["SPY", "QQQ", "TLT", "DXY"],
  ECONOMIC_GROWTH: ["SPY", "IWM", "HYG"],
  LABOR_MARKET: ["SPY", "XLY", "TLT"],
  BANKING: ["XLF", "KRE", "KBE"],
  CREDIT_MARKETS: ["HYG", "LQD", "JNK"],
  LIQUIDITY: ["SPY", "TLT", "DXY"],
  FINANCIAL_MARKETS: ["SPY", "QQQ", "VIX"],
  REGULATION: ["XLF", "KRE", "QQQ"],
  GEOPOLITICS: ["DXY", "GLD", "VIX"],
  CONFLICT: ["GLD", "USO", "ITA"],
  TRADE: ["FXI", "EEM", "IYT"],
  SANCTIONS: ["USO", "DXY", "EEM"],
  ENERGY: ["XLE", "USO", "UNG"],
  SUPPLY_CHAIN: ["IYT", "XLI", "XLY"],
  COMMODITIES: ["GLD", "DBA", "DBC"],
  TECHNOLOGY: ["QQQ", "SMH", "XLK"],
  CRYPTO: ["BTC", "ETH", "COIN"],
  CORPORATE_ACTIVITY: ["SPY", "HYG", "IWM"],
  AI: ["QQQ", "SMH", "NVDA", "XLK"]
};

export const THEME_COMBINATIONS: readonly { id: string; label: string; themes: readonly NormalizedTheme[] }[] = [
  { id: "inflation_energy_shock", label: "Inflation + Energy", themes: ["INFLATION", "ENERGY"] },
  { id: "bank_credit_stress", label: "Banking + Credit Stress", themes: ["BANKING", "CREDIT_MARKETS"] },
  { id: "geopolitical_supply_shock", label: "Conflict + Supply Chain", themes: ["CONFLICT", "SUPPLY_CHAIN"] },
  { id: "policy_rate_shock", label: "Central Bank + Rates", themes: ["CENTRAL_BANK", "INTEREST_RATES"] }
];

const MARKET_IMPACT_RULES: Readonly<Partial<Record<NormalizedTheme, readonly Omit<MarketImpactSignal, "themes">[]>>> = {
  INFLATION: [
    { asset: "TLT", direction: "DOWN", rationale: "Higher inflation pressure usually weighs on long-duration bonds.", confidence: 74 },
    { asset: "GLD", direction: "UP", rationale: "Inflation shocks can increase demand for hard-asset hedges.", confidence: 68 },
    { asset: "QQQ", direction: "DOWN", rationale: "Higher discount-rate expectations pressure long-duration growth equities.", confidence: 70 }
  ],
  INTEREST_RATES: [
    { asset: "TLT", direction: "DOWN", rationale: "Rate pressure directly affects long-duration Treasuries.", confidence: 78 },
    { asset: "QQQ", direction: "DOWN", rationale: "Higher rates reduce the present value of growth cash flows.", confidence: 72 },
    { asset: "DXY", direction: "UP", rationale: "Rate repricing can support the dollar.", confidence: 66 }
  ],
  CENTRAL_BANK: [
    { asset: "TLT", direction: "MIXED", rationale: "Policy communication can move both rate path and growth expectations.", confidence: 62 },
    { asset: "SPY", direction: "MIXED", rationale: "Policy sensitivity can shift broad equity risk appetite.", confidence: 60 }
  ],
  ENERGY: [
    { asset: "XLE", direction: "UP", rationale: "Energy supply and price shocks usually support energy equities.", confidence: 76 },
    { asset: "USO", direction: "UP", rationale: "Oil-linked themes map directly to crude exposure.", confidence: 78 },
    { asset: "QQQ", direction: "DOWN", rationale: "Energy-driven inflation can pressure rate-sensitive equities.", confidence: 61 }
  ],
  BANKING: [
    { asset: "KRE", direction: "DOWN", rationale: "Banking stress is most direct for regional bank risk.", confidence: 80 },
    { asset: "XLF", direction: "DOWN", rationale: "Financial-sector risk premium rises when banking themes spike.", confidence: 72 }
  ],
  CREDIT_MARKETS: [
    { asset: "HYG", direction: "DOWN", rationale: "Credit stress tends to widen spreads and pressure high-yield credit.", confidence: 78 },
    { asset: "LQD", direction: "DOWN", rationale: "Investment-grade credit can weaken when default risk rises.", confidence: 66 }
  ],
  LIQUIDITY: [
    { asset: "SPY", direction: "DOWN", rationale: "Funding stress can reduce risk appetite.", confidence: 66 },
    { asset: "DXY", direction: "UP", rationale: "Liquidity stress can increase demand for dollars.", confidence: 64 }
  ],
  CONFLICT: [
    { asset: "GLD", direction: "UP", rationale: "Conflict risk can increase safe-haven demand.", confidence: 70 },
    { asset: "VIX", direction: "UP", rationale: "Conflict escalation usually increases volatility demand.", confidence: 72 }
  ],
  SUPPLY_CHAIN: [
    { asset: "IYT", direction: "DOWN", rationale: "Shipping and logistics disruption can pressure transport exposure.", confidence: 65 },
    { asset: "XLI", direction: "DOWN", rationale: "Industrial margins can weaken when supply-chain costs rise.", confidence: 61 }
  ],
  TECHNOLOGY: [
    { asset: "QQQ", direction: "UP", rationale: "Technology theme acceleration can support growth and mega-cap tech exposure.", confidence: 64 },
    { asset: "SMH", direction: "UP", rationale: "Semiconductor-linked themes map directly to chip exposure.", confidence: 72 }
  ],
  AI: [
    { asset: "SMH", direction: "UP", rationale: "AI infrastructure demand maps directly to semiconductor exposure.", confidence: 76 },
    { asset: "QQQ", direction: "UP", rationale: "AI adoption can support mega-cap technology risk appetite.", confidence: 70 },
    { asset: "NVDA", direction: "UP", rationale: "AI compute and model deployment coverage are direct inputs to accelerator demand.", confidence: 72 }
  ],
  CRYPTO: [
    { asset: "BTC", direction: "UP", rationale: "Crypto theme acceleration maps directly to bitcoin sentiment.", confidence: 65 },
    { asset: "COIN", direction: "UP", rationale: "Crypto activity can support exchange and infrastructure exposure.", confidence: 58 }
  ],
  SANCTIONS: [
    { asset: "USO", direction: "UP", rationale: "Sanctions can tighten commodity supply expectations.", confidence: 64 },
    { asset: "DXY", direction: "UP", rationale: "Sanctions stress can support dollar demand.", confidence: 59 }
  ],
  TRADE: [
    { asset: "EEM", direction: "DOWN", rationale: "Trade friction can pressure export-sensitive emerging markets.", confidence: 60 },
    { asset: "IYT", direction: "DOWN", rationale: "Tariff and trade risk can weaken transport-linked demand.", confidence: 57 }
  ]
};

export function parseRawThemes(input: RawThemeInput | null | undefined): string[] {
  const parts = Array.isArray(input) ? input : String(input ?? "").split(/[,;\n|]+/);
  const seen = new Set<string>();
  const rawThemes: string[] = [];

  for (const part of parts) {
    const theme = String(part ?? "").trim();
    const key = theme.toUpperCase();
    if (!theme || seen.has(key)) {
      continue;
    }
    seen.add(key);
    rawThemes.push(theme);
  }

  return rawThemes;
}

export function severityForScore(totalScore: number): ThemeSeverity {
  if (totalScore >= 25) {
    return "CRITICAL";
  }
  if (totalScore >= 15) {
    return "HIGH";
  }
  return "NORMAL";
}

function normalizeForMatch(value: string): string {
  return value
    .toUpperCase()
    .replace(/[^A-Z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function matchesPattern(rawTheme: string, pattern: string): boolean {
  const rawUpper = rawTheme.toUpperCase();
  const patternUpper = pattern.toUpperCase();
  const rawNormalized = normalizeForMatch(rawTheme);
  const patternNormalized = normalizeForMatch(pattern);

  if (patternNormalized.length <= 3) {
    return rawNormalized.split("_").includes(patternNormalized);
  }

  return rawUpper.includes(patternUpper) || rawNormalized.includes(patternNormalized);
}

export function themesForProductCategory(category: ProductCategory): NormalizedTheme[] {
  const definition = PRODUCT_TAXONOMY.find((item) => item.category === category);
  const taxonomyThemes = definition?.subcategories.flatMap((subcategory) => subcategory.normalized_themes) ?? [];

  return [...new Set([...taxonomyThemes, ...PRODUCT_CATEGORY_DISPLAY_THEMES[category]])].sort(sortThemes);
}

export function focusAreasForProductCategory(category: ProductCategory): ProductFocusArea[] {
  return PRODUCT_FOCUS_AREAS.filter((focusArea) => focusArea.category === category).map((focusArea) => ({ ...focusArea }));
}

function sortThemes(a: NormalizedTheme, b: NormalizedTheme): number {
  return THEME_WEIGHTS[b] - THEME_WEIGHTS[a] || a.localeCompare(b);
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function roundTo(value: number, decimals = 2): number {
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}

function pctChange(current: number, baseline: number): number {
  if (baseline <= 0) {
    return current > 0 ? 100 : 0;
  }
  return Math.round(((current - baseline) / baseline) * 100);
}

function trendDirection(delta: number, deltaPct: number): SignalTrendDirection {
  if (delta >= 10 || deltaPct >= 50) {
    return "ACCELERATING";
  }
  if (delta > 2) {
    return "RISING";
  }
  if (delta < -2) {
    return "FALLING";
  }
  return "STABLE";
}

export function detectThemeCombinations(normalizedThemes: readonly NormalizedTheme[]): ThemeCombinationSignal[] {
  const themeSet = new Set(normalizedThemes);
  return THEME_COMBINATIONS.filter((combo) => combo.themes.every((theme) => themeSet.has(theme))).map((combo) => ({
    id: combo.id,
    label: combo.label,
    themes: [...combo.themes]
  }));
}

function buildFrequencySignals(
  normalizedThemes: readonly NormalizedTheme[],
  matchesByTheme: ReadonlyMap<NormalizedTheme, ThemeMatch>,
  context: ThemeContextInput | undefined
): ThemeFrequencySignal[] {
  const baseSignals = normalizedThemes
    .map((theme) => {
      const match = matchesByTheme.get(theme);
      const contextMentions = context?.theme_mentions?.[theme];
      const fallbackMentions = Math.max(match?.matched_raw_themes.length ?? 0, 1);
      const currentMentions = Math.max(0, Math.round(contextMentions?.current_mentions ?? fallbackMentions));
      const baselineMentions = Math.max(0, Math.round(contextMentions?.baseline_mentions ?? currentMentions));
      const previousMentions = Math.max(0, Math.round(contextMentions?.previous_mentions ?? baselineMentions));
      const deltaMentions = currentMentions - baselineMentions;
      const spikePct = pctChange(currentMentions, baselineMentions);
      const ratio = baselineMentions > 0 ? currentMentions / baselineMentions : currentMentions > 0 ? 2 : 0;
      const anomalyScore = Math.round(clamp(ratio * 25, 0, 100));
      const weight = THEME_WEIGHTS[theme];
      const themeScore = roundTo(currentMentions * 0.4 + Math.max(0, spikePct) * 0.4 + weight * 0.2);

      return {
        normalized_theme: theme,
        category: match?.category ?? THEME_MAPPING.find((item) => item.normalized_theme === theme)!.category,
        weight,
        current_mentions: currentMentions,
        baseline_mentions: baselineMentions,
        previous_mentions: previousMentions,
        delta_mentions: deltaMentions,
        spike_pct: spikePct,
        anomaly_score: anomalyScore,
        weighted_intensity: Math.round(weight * ratio),
        theme_score: themeScore,
        contribution_pct: 0,
        rank: 0,
        classification: "BACKGROUND" as const,
        matched_raw_themes: match?.matched_raw_themes ?? [],
        matched_patterns: match?.matched_patterns ?? []
      };
    })
    .sort((a, b) => b.theme_score - a.theme_score || b.current_mentions - a.current_mentions || b.weight - a.weight);
  const totalSignalScore = baseSignals.reduce((sum, signal) => sum + signal.theme_score, 0);

  return baseSignals.map((signal, index) => ({
    ...signal,
    contribution_pct: totalSignalScore > 0 ? roundTo((signal.theme_score / totalSignalScore) * 100, 1) : 0,
    rank: index + 1,
    classification: index < 3 ? "PRIMARY" : index < 8 ? "SECONDARY" : "BACKGROUND"
  }));
}

function buildProductCategorySignals(
  rawThemes: readonly string[],
  frequencySignals: readonly ThemeFrequencySignal[]
): ProductCategorySignal[] {
  const frequencyByTheme = new Map(frequencySignals.map((signal) => [signal.normalized_theme, signal]));
  const categorySignals = PRODUCT_TAXONOMY.map((definition) => {
    const subcategories = definition.subcategories
      .map((subcategory) => {
        const themeSignals = subcategory.normalized_themes
          .map((theme) => frequencyByTheme.get(theme))
          .filter((signal): signal is ThemeFrequencySignal => Boolean(signal));
        const themeScore = themeSignals.reduce((sum, signal) => sum + signal.theme_score, 0);
        const matchedRawThemes = rawThemes.filter((rawTheme) =>
          subcategory.raw_patterns.some((pattern) => matchesPattern(rawTheme, pattern))
        );
        const matchedPatterns = subcategory.raw_patterns.filter((pattern) =>
          rawThemes.some((rawTheme) => matchesPattern(rawTheme, pattern))
        );
        const rawScore = matchedRawThemes.length * subcategory.weight * 0.6;
        const score = roundTo(themeScore + rawScore);

        return {
          label: subcategory.label,
          score,
          contribution_pct: 0,
          rank: 0,
          themes: [...new Set(themeSignals.map((signal) => signal.normalized_theme))].sort(sortThemes),
          matched_raw_themes: [...new Set(matchedRawThemes)],
          matched_patterns: [...new Set(matchedPatterns)]
        };
      })
      .filter((subcategory) => subcategory.score > 0)
      .sort((a, b) => b.score - a.score || a.label.localeCompare(b.label));

    const score = roundTo(subcategories.reduce((sum, subcategory) => sum + subcategory.score, 0));
    const themes = [...new Set(subcategories.flatMap((subcategory) => subcategory.themes))].sort(sortThemes);
    const matchedRawThemes = [...new Set(subcategories.flatMap((subcategory) => subcategory.matched_raw_themes))];
    const matchedPatterns = [...new Set(subcategories.flatMap((subcategory) => subcategory.matched_patterns))];

    return {
      category: definition.category,
      label: definition.label,
      score,
      contribution_pct: 0,
      rank: 0,
      subcategories,
      themes,
      matched_raw_themes: matchedRawThemes,
      matched_patterns: matchedPatterns
    };
  })
    .filter((category) => category.score > 0)
    .sort((a, b) => b.score - a.score || PRODUCT_CATEGORY_ORDER.indexOf(a.category) - PRODUCT_CATEGORY_ORDER.indexOf(b.category));

  const totalProductScore = categorySignals.reduce((sum, category) => sum + category.score, 0);

  return categorySignals.map((category, categoryIndex) => {
    const totalSubcategoryScore = category.subcategories.reduce((sum, subcategory) => sum + subcategory.score, 0);
    return {
      ...category,
      contribution_pct: totalProductScore > 0 ? roundTo((category.score / totalProductScore) * 100, 1) : 0,
      rank: categoryIndex + 1,
      subcategories: category.subcategories.map((subcategory, subcategoryIndex) => ({
        ...subcategory,
        contribution_pct: totalSubcategoryScore > 0 ? roundTo((subcategory.score / totalSubcategoryScore) * 100, 1) : 0,
        rank: subcategoryIndex + 1
      }))
    };
  });
}

function buildTrendSignal(totalScore: number, context: ThemeContextInput | undefined): ThemeTrendSignal {
  const previousScore = Math.max(0, Math.round(context?.previous_score ?? totalScore));
  const baselineScore = Math.max(0, Math.round(context?.baseline_score ?? previousScore));
  const delta = totalScore - previousScore;
  const deltaPct = pctChange(totalScore, previousScore);

  return {
    window_label: context?.window_label ?? "current window",
    current_score: totalScore,
    previous_score: previousScore,
    baseline_score: baselineScore,
    delta,
    delta_pct: deltaPct,
    direction: trendDirection(delta, deltaPct)
  };
}

function buildSignalModel(
  totalScore: number,
  normalizedThemes: readonly NormalizedTheme[],
  categoryScores: Partial<Record<ThemeCategory, number>>,
  trend: ThemeTrendSignal,
  context: ThemeContextInput | undefined
): SignalModel {
  const severityScore = Math.round(clamp(totalScore * 3, 0, 100));
  const trendScore = Math.round(clamp(50 + trend.delta * 4 + trend.delta_pct * 0.18, 0, 100));
  const breadthScore = Math.round(clamp(normalizedThemes.length * 8 + Object.keys(categoryScores).length * 10, 0, 100));
  const confidenceScore = Math.round(clamp(context?.confidence ?? 55 + normalizedThemes.length * 5, 0, 100));
  const compositeScore = Math.round(severityScore * 0.35 + trendScore * 0.3 + breadthScore * 0.2 + confidenceScore * 0.15);

  return {
    severity_score: severityScore,
    trend_score: trendScore,
    breadth_score: breadthScore,
    confidence_score: confidenceScore,
    composite_score: compositeScore
  };
}

function buildInterpretation(
  normalizedThemes: readonly NormalizedTheme[],
  combinations: readonly ThemeCombinationSignal[],
  primaryDriver: ThemeFrequencySignal | null
): SignalInterpretation {
  const comboIds = new Set(combinations.map((combo) => combo.id));
  const themeSet = new Set(normalizedThemes);

  if (comboIds.has("inflation_energy_shock")) {
    return {
      headline: "Inflation Shock",
      drivers: ["Oil-linked price pressure", "Policy sensitivity from inflation themes"],
      implications: ["Rate expectations up", "Long-duration equities under pressure", "Energy exposure supported"]
    };
  }
  if (comboIds.has("bank_credit_stress")) {
    return {
      headline: "Financial Stress",
      drivers: ["Banking stress", "Credit-market deterioration"],
      implications: ["Regional bank risk up", "High-yield spreads vulnerable", "Liquidity preference rising"]
    };
  }
  if (comboIds.has("geopolitical_supply_shock")) {
    return {
      headline: "Geopolitical Supply Shock",
      drivers: ["Conflict escalation", "Logistics and shipping disruption"],
      implications: ["Volatility risk up", "Transport and industrial margins vulnerable", "Safe-haven demand supported"]
    };
  }
  if (comboIds.has("policy_rate_shock")) {
    return {
      headline: "Policy Rate Shock",
      drivers: ["Central bank sensitivity", "Rate repricing"],
      implications: ["Duration risk up", "USD support possible", "Growth equity multiple pressure"]
    };
  }
  if (themeSet.has("AI")) {
    return {
      headline: "AI Infrastructure Signal",
      drivers: [primaryDriver ? `${primaryDriver.normalized_theme.replace(/_/g, " ")} acceleration` : "AI and compute-demand themes"],
      implications: ["Semiconductor exposure in focus", "Mega-cap technology sensitivity up", "Regulatory and liquidity overlap can change signal quality"]
    };
  }
  if (themeSet.has("TECHNOLOGY") || themeSet.has("CRYPTO")) {
    return {
      headline: "Modern Risk Appetite Signal",
      drivers: [primaryDriver ? `${primaryDriver.normalized_theme.replace(/_/g, " ")} acceleration` : "Technology and digital-asset themes"],
      implications: ["QQQ and SMH sensitivity up", "Crypto beta in focus", "Narrative risk depends on regulation overlap"]
    };
  }
  if (primaryDriver) {
    return {
      headline: `${primaryDriver.normalized_theme.replace(/_/g, " ")} Signal`,
      drivers: [`${primaryDriver.current_mentions} mentions vs ${primaryDriver.baseline_mentions} baseline`],
      implications: ["Monitor velocity and breadth before treating the signal as persistent"]
    };
  }
  return {
    headline: "No Material Theme Signal",
    drivers: ["No normalized finance themes detected"],
    implications: ["No market mapping generated"]
  };
}

function buildMarketImpacts(
  normalizedThemes: readonly NormalizedTheme[],
  frequencySignals: readonly ThemeFrequencySignal[]
): MarketImpactSignal[] {
  const frequencyByTheme = new Map(frequencySignals.map((item) => [item.normalized_theme, item]));
  const impacts = new Map<string, MarketImpactSignal>();

  for (const theme of normalizedThemes) {
    const rules = MARKET_IMPACT_RULES[theme] ?? [];
    const frequency = frequencyByTheme.get(theme);
    const intensityBoost = frequency ? clamp(Math.round(frequency.anomaly_score / 10), 0, 10) : 0;

    for (const rule of rules) {
      const key = `${rule.asset}:${rule.direction}`;
      const existing = impacts.get(key);
      if (existing) {
        existing.themes = [...new Set([...existing.themes, theme])];
        existing.confidence = Math.min(95, Math.max(existing.confidence, rule.confidence + intensityBoost));
      } else {
        impacts.set(key, {
          ...rule,
          confidence: Math.min(95, rule.confidence + intensityBoost),
          themes: [theme]
        });
      }
    }
  }

  return [...impacts.values()].sort((a, b) => b.confidence - a.confidence || a.asset.localeCompare(b.asset)).slice(0, 8);
}

export function scoreThemeArticle(article: ThemeArticleInput): ThemeSignal {
  const rawThemes = parseRawThemes(article.raw_themes);
  const matchesByTheme = new Map<NormalizedTheme, ThemeMatch>();

  for (const rawTheme of rawThemes) {
    for (const definition of THEME_MAPPING) {
      const matchedPatterns = definition.raw_patterns.filter((pattern) => matchesPattern(rawTheme, pattern));
      if (matchedPatterns.length === 0) {
        continue;
      }

      const existing = matchesByTheme.get(definition.normalized_theme);
      if (existing) {
        existing.matched_raw_themes = [...new Set([...existing.matched_raw_themes, rawTheme])];
        existing.matched_patterns = [...new Set([...existing.matched_patterns, ...matchedPatterns])];
      } else {
        matchesByTheme.set(definition.normalized_theme, {
          normalized_theme: definition.normalized_theme,
          category: definition.category,
          weight: definition.weight,
          matched_raw_themes: [rawTheme],
          matched_patterns: matchedPatterns
        });
      }
    }
  }

  const normalizedThemes = [...matchesByTheme.keys()].sort(sortThemes);
  const matches = normalizedThemes.map((theme) => matchesByTheme.get(theme)!);
  const themeWeights = normalizedThemes.reduce<Partial<Record<NormalizedTheme, number>>>((acc, theme) => {
    acc[theme] = THEME_WEIGHTS[theme];
    return acc;
  }, {});
  const categoryScores = matches.reduce<Partial<Record<ThemeCategory, number>>>((acc, match) => {
    acc[match.category] = (acc[match.category] ?? 0) + match.weight;
    return acc;
  }, {});
  const marketMappings = normalizedThemes.reduce<Partial<Record<NormalizedTheme, string[]>>>((acc, theme) => {
    const assets = MARKET_THEME_MAP[theme];
    if (assets) {
      acc[theme] = [...assets];
    }
    return acc;
  }, {});
  const combinations = detectThemeCombinations(normalizedThemes);
  const frequencySignals = buildFrequencySignals(normalizedThemes, matchesByTheme, article.context);
  const productCategories = buildProductCategorySignals(rawThemes, frequencySignals);
  const primaryProductCategory = productCategories[0] ?? null;
  const secondaryProductCategories = productCategories.slice(1, 4);
  const productCategoryScores = productCategories.reduce<Partial<Record<ProductCategory, number>>>((acc, category) => {
    acc[category.category] = category.score;
    return acc;
  }, {});
  const totalScore = roundTo(frequencySignals.reduce((sum, signal) => sum + signal.theme_score, 0));
  const trend = buildTrendSignal(totalScore, article.context);
  const primaryDriver = frequencySignals[0] ?? null;
  const primaryDrivers = frequencySignals.filter((signal) => signal.classification === "PRIMARY");
  const secondaryDrivers = frequencySignals.filter((signal) => signal.classification === "SECONDARY");
  const backgroundContext = frequencySignals.filter((signal) => signal.classification === "BACKGROUND");
  const signalModel = buildSignalModel(totalScore, normalizedThemes, categoryScores, trend, article.context);
  const interpretation = buildInterpretation(normalizedThemes, combinations, primaryDriver);
  const marketImpacts = buildMarketImpacts(normalizedThemes, frequencySignals);

  return {
    id: article.id,
    title: article.title,
    raw_themes: rawThemes,
    full_theme_list: rawThemes,
    normalized_themes: normalizedThemes,
    normalized_theme_list: normalizedThemes,
    theme_weights: themeWeights,
    total_score: totalScore,
    total_signal_score: totalScore,
    severity: severityForScore(totalScore),
    matches,
    category_scores: categoryScores,
    product_category_scores: productCategoryScores,
    product_categories: productCategories,
    primary_product_category: primaryProductCategory,
    secondary_product_categories: secondaryProductCategories,
    market_mappings: marketMappings,
    combination_signals: combinations,
    trend,
    frequency_signals: frequencySignals,
    primary_driver: primaryDriver,
    primary_drivers: primaryDrivers,
    secondary_drivers: secondaryDrivers,
    background_context: backgroundContext,
    signal_model: signalModel,
    interpretation,
    market_impacts: marketImpacts
  };
}

export function scoreThemeArticles(articles: readonly ThemeArticleInput[]): ThemeSignal[] {
  return articles.map(scoreThemeArticle);
}

export type TopicRuleRecommendation = {
  topicKey: string;
  label: string;
  sortOrder: number;
  focus: string;
  suggestedKeywords: string[];
  broadTerms: string[];
  notes: string[];
};

export const TOPIC_RULE_RECOMMENDATIONS: readonly TopicRuleRecommendation[] = [
  {
    topicKey: "SECURITIES_REGULATION",
    label: "Securities Regulation",
    sortOrder: 10,
    focus: "SEC and SRO rulemaking, disclosure, registration, market structure, and regulated intermediaries.",
    suggestedKeywords: [
      "securities regulation",
      "SEC rulemaking",
      "rulemaking",
      "disclosure",
      "registration statement",
      "Exchange Act",
      "Securities Act",
      "Regulation NMS",
      "market structure",
      "broker-dealer",
      "investment adviser",
      "fiduciary",
      "proxy",
      "shareholder",
      "corporate governance",
      "Form PF",
      "EDGAR",
      "investment company"
    ],
    broadTerms: ["sec", "securities", "investor", "exchange", "registration"],
    notes: [
      "Use explicit rule or filing terms instead of agency names alone.",
      "Keep investor-only stories out unless they mention disclosure, advice, or registration."
    ]
  },
  {
    topicKey: "CAPITAL_FORMATION",
    label: "Capital Formation",
    sortOrder: 20,
    focus: "Public offerings, exempt offerings, private capital, crowdfunding, and capital access policy.",
    suggestedKeywords: [
      "capital formation",
      "IPO",
      "initial public offering",
      "public offering",
      "secondary offering",
      "private offering",
      "private placement",
      "Reg D",
      "Regulation D",
      "Reg A",
      "Regulation A",
      "Regulation Crowdfunding",
      "exempt offering",
      "crowdfunding",
      "SPAC",
      "de-SPAC",
      "venture capital",
      "startup funding",
      "private markets",
      "emerging growth company",
      "small business capital"
    ],
    broadTerms: ["capital", "offering", "funding", "venture", "startup"],
    notes: [
      "Generic capital and funding terms over-match credit, bank capital, and macro stories.",
      "Phrase-level offering terms are safer than a standalone offering match."
    ]
  },
  {
    topicKey: "AML",
    label: "AML",
    sortOrder: 30,
    focus: "Anti-money-laundering controls, BSA duties, sanctions compliance, KYC, and illicit finance.",
    suggestedKeywords: [
      "AML",
      "anti-money laundering",
      "anti money laundering",
      "Bank Secrecy Act",
      "BSA",
      "FinCEN",
      "OFAC",
      "KYC",
      "know your customer",
      "customer identification program",
      "CIP",
      "beneficial ownership",
      "suspicious activity report",
      "SAR",
      "suspicious activity",
      "sanctions compliance",
      "illicit finance",
      "terrorist financing",
      "money laundering"
    ],
    broadTerms: ["finra", "sanctions"],
    notes: [
      "FINRA alone is an organization, not an AML signal.",
      "Sanctions should stay only when paired with compliance, OFAC, or AML language."
    ]
  },
  {
    topicKey: "ENFORCEMENT",
    label: "Enforcement",
    sortOrder: 40,
    focus: "Agency enforcement actions, litigation, penalties, settlements, fraud, and misconduct.",
    suggestedKeywords: [
      "enforcement",
      "litigation release",
      "administrative proceeding",
      "cease-and-desist",
      "injunction",
      "disgorgement",
      "civil penalty",
      "penalty",
      "fine",
      "fraud",
      "charges",
      "complaint",
      "settlement",
      "indictment",
      "prosecution",
      "lawsuit",
      "investigation",
      "violation",
      "misconduct",
      "disciplinary action",
      "AWC",
      "insider trading",
      "market manipulation",
      "misrepresentation"
    ],
    broadTerms: ["fine"],
    notes: [
      "This category is generally strong; add enforcement-specific legal outcomes for better recall.",
      "Fine is useful but weak on its own in consumer and market articles."
    ]
  },
  {
    topicKey: "AI_TECH",
    label: "AI & Tech",
    sortOrder: 50,
    focus: "AI, predictive analytics, financial technology, automation, cyber risk, and regulated technology infrastructure.",
    suggestedKeywords: [
      "AI",
      "artificial intelligence",
      "generative AI",
      "machine learning",
      "large language model",
      "LLM",
      "predictive data analytics",
      "algorithmic trading",
      "algorithm",
      "automation",
      "robo-adviser",
      "fintech",
      "cybersecurity",
      "cyber risk",
      "data breach",
      "Reg SCI",
      "technology infrastructure",
      "cloud",
      "data center"
    ],
    broadTerms: ["technology"],
    notes: [
      "Keep AI as an exact acronym, but do not rely on generic technology alone.",
      "Cyber and infrastructure terms catch regulated tech stories that are not strictly AI."
    ]
  },
  {
    topicKey: "CRYPTO",
    label: "Crypto",
    sortOrder: 60,
    focus: "Crypto assets, tokenization, stablecoins, custody, staking, DeFi, and digital-asset market infrastructure.",
    suggestedKeywords: [
      "crypto",
      "cryptocurrency",
      "crypto asset",
      "crypto assets",
      "digital asset",
      "digital assets",
      "digital asset securities",
      "bitcoin",
      "BTC",
      "ethereum",
      "ETH",
      "blockchain",
      "distributed ledger",
      "tokenization",
      "tokenized securities",
      "stablecoin",
      "staking",
      "DeFi",
      "decentralized finance",
      "NFT",
      "wallet",
      "custody",
      "crypto exchange"
    ],
    broadTerms: ["token"],
    notes: [
      "Token is useful but noisy without crypto, digital asset, or securities context.",
      "Custody overlaps with adviser rules; pairing with crypto terms keeps this category cleaner."
    ]
  },
  {
    topicKey: "CREDIT_MARKETS",
    label: "Credit Markets",
    sortOrder: 70,
    focus: "Rates, bonds, private credit, spreads, loans, securitization, defaults, and credit stress.",
    suggestedKeywords: [
      "credit markets",
      "credit spread",
      "credit spreads",
      "corporate bond",
      "bond market",
      "fixed income",
      "high yield",
      "investment grade",
      "private credit",
      "leveraged loan",
      "loan",
      "mortgage",
      "MBS",
      "ABS",
      "CLO",
      "securitization",
      "debt financing",
      "distressed debt",
      "default",
      "bankruptcy",
      "Treasury yield",
      "yield curve",
      "funding stress",
      "liquidity"
    ],
    broadTerms: ["credit", "bond", "debt", "yield"],
    notes: [
      "Credit, debt, bond, and yield are broad; phrase-level market terms classify more cleanly.",
      "Liquidity is relevant here when it is attached to funding or market stress."
    ]
  },
  {
    topicKey: "FINANCIAL_MARKETS",
    label: "Financial Markets",
    sortOrder: 80,
    focus: "Trading, equities, derivatives, ETFs, volatility, exchanges, clearing, settlement, and market plumbing.",
    suggestedKeywords: [
      "financial markets",
      "market structure",
      "stock market",
      "equity market",
      "equities",
      "trading",
      "exchange trading",
      "volatility",
      "VIX",
      "S&P 500",
      "Nasdaq",
      "Dow Jones",
      "options",
      "futures",
      "derivatives",
      "swaps",
      "ETF",
      "exchange-traded fund",
      "clearing",
      "settlement",
      "execution quality",
      "ATS",
      "alternative trading system",
      "market maker",
      "order routing"
    ],
    broadTerms: ["market", "stock", "equity", "dow"],
    notes: [
      "Market alone catches almost every finance article.",
      "Use market-structure, instrument, or index terms for cleaner topic routing."
    ]
  },
  {
    topicKey: "ECONOMIC_GROWTH",
    label: "Economic Growth",
    sortOrder: 90,
    focus: "Macro growth, inflation, labor, monetary policy, fiscal policy, trade, tariffs, and supply-chain pressure.",
    suggestedKeywords: [
      "economic growth",
      "GDP",
      "recession",
      "slowdown",
      "inflation",
      "CPI",
      "PCE",
      "prices",
      "Federal Reserve",
      "FOMC",
      "monetary policy",
      "interest rate",
      "rate cut",
      "rate hike",
      "unemployment",
      "labor market",
      "wages",
      "jobs report",
      "fiscal policy",
      "tariff",
      "tariffs",
      "trade policy",
      "supply chain",
      "energy prices"
    ],
    broadTerms: ["economy", "growth", "fed", "jobs"],
    notes: [
      "Fed alone can mean many things; FOMC, rate path, or monetary policy is cleaner.",
      "Jobs is safer as jobs report or labor-market language."
    ]
  },
  {
    topicKey: "PREDICTION_MARKETS",
    label: "Prediction Markets",
    sortOrder: 100,
    focus: "Event contracts, prediction exchanges, binary-event products, and CFTC market-structure disputes.",
    suggestedKeywords: [
      "prediction market",
      "prediction markets",
      "event contract",
      "event contracts",
      "binary options",
      "binary event",
      "prediction exchange",
      "betting market",
      "forecast market",
      "Kalshi",
      "Polymarket",
      "CFTC event contracts",
      "CFTC prediction markets",
      "political betting",
      "sports event contract",
      "odds contract"
    ],
    broadTerms: ["forecast", "odds", "contract", "prediction"],
    notes: [
      "Contract and forecast create false positives in ordinary business coverage.",
      "Event-contract phrases are the best precision terms."
    ]
  },
  {
    topicKey: "CYBER_OPERATIONAL_RESILIENCE",
    label: "Cybersecurity & Operational Resilience",
    sortOrder: 110,
    focus: "Cybersecurity, cyber incidents, operational resilience, outages, vendor risk, and technology controls.",
    suggestedKeywords: [
      "cybersecurity",
      "cyber risk",
      "cyber incident",
      "data breach",
      "ransomware",
      "operational resilience",
      "business continuity",
      "incident response",
      "Reg SCI",
      "systems compliance",
      "third-party risk",
      "vendor risk",
      "data security",
      "information security",
      "access controls",
      "technology outage"
    ],
    broadTerms: ["security", "technology", "systems"],
    notes: [
      "Use cyber, resilience, or outage context rather than security alone.",
      "Vendor and third-party terms are strongest when paired with operational or data-risk language."
    ]
  },
  {
    topicKey: "ADVISERS_PRIVATE_FUNDS",
    label: "Investment Advisers & Private Funds",
    sortOrder: 120,
    focus: "Investment adviser regulation, private funds, fiduciary duties, custody, conflicts, valuation, and fee disclosure.",
    suggestedKeywords: [
      "investment adviser",
      "investment advisers",
      "private fund",
      "private funds",
      "hedge fund",
      "private equity fund",
      "Form ADV",
      "Form PF",
      "fiduciary duty",
      "custody rule",
      "safeguarding rule",
      "adviser compliance",
      "fund fees",
      "fee disclosure",
      "conflicts of interest",
      "valuation",
      "side letter",
      "preferential treatment"
    ],
    broadTerms: ["fund", "adviser", "fees", "valuation"],
    notes: [
      "Private fund and adviser phrases avoid pulling in generic mutual-fund or market stories.",
      "Fee and valuation terms should carry adviser, fund, custody, or conflict context."
    ]
  },
  {
    topicKey: "RETAIL_SALES_PRACTICES",
    label: "Retail Investor Protection & Sales Practices",
    sortOrder: 130,
    focus: "Retail investor protection, broker recommendations, suitability, Reg BI, complex products, and vulnerable investors.",
    suggestedKeywords: [
      "retail investor",
      "investor protection",
      "sales practice",
      "suitability",
      "Regulation Best Interest",
      "Reg BI",
      "best interest",
      "broker recommendation",
      "complex products",
      "senior investor",
      "financial exploitation",
      "account recommendation",
      "rollover recommendation",
      "investor education",
      "misleading communication",
      "fair dealing"
    ],
    broadTerms: ["investor", "recommendation", "education"],
    notes: [
      "Investor alone is too broad; retain retail, sales-practice, Reg BI, or suitability context.",
      "This topic complements enforcement by focusing on conduct risk before or outside a charged action."
    ]
  },
  {
    topicKey: "MARKET_STRUCTURE_EXECUTION",
    label: "Market Structure & Execution Quality",
    sortOrder: 140,
    focus: "Execution quality, order routing, market data, trading venues, market makers, and NMS structure.",
    suggestedKeywords: [
      "market structure",
      "best execution",
      "order routing",
      "payment for order flow",
      "PFOF",
      "execution quality",
      "national market system",
      "Regulation NMS",
      "tick size",
      "odd lots",
      "market data",
      "consolidated tape",
      "ATS",
      "alternative trading system",
      "dark pool",
      "market maker",
      "exchange fees"
    ],
    broadTerms: ["market", "trading", "exchange"],
    notes: [
      "This is a more precise child topic of financial markets.",
      "Prioritize execution, routing, venue, and market-data terms over generic trading language."
    ]
  },
  {
    topicKey: "CORPORATE_DISCLOSURE_GOVERNANCE",
    label: "Corporate Disclosure & Governance",
    sortOrder: 150,
    focus: "Public company disclosure, periodic reports, proxy matters, governance, ownership reporting, and board oversight.",
    suggestedKeywords: [
      "corporate disclosure",
      "public company disclosure",
      "materiality",
      "risk factors",
      "MD&A",
      "10-K",
      "10-Q",
      "8-K",
      "proxy statement",
      "shareholder proposal",
      "corporate governance",
      "board oversight",
      "executive compensation",
      "insider reporting",
      "beneficial ownership",
      "Schedule 13D",
      "Schedule 13G"
    ],
    broadTerms: ["disclosure", "governance", "shareholder"],
    notes: [
      "This separates issuer disclosure and governance from general securities regulation.",
      "Form names and proxy terms provide the cleanest classification signals."
    ]
  }
];

export const TOPIC_RULE_RECOMMENDATION_BY_KEY: Readonly<Record<string, TopicRuleRecommendation>> =
  Object.fromEntries(TOPIC_RULE_RECOMMENDATIONS.map((item) => [item.topicKey, item])) as Record<string, TopicRuleRecommendation>;

export function formatTopicRuleKeywords(keywords: readonly string[]): string {
  return keywords.join(", ");
}

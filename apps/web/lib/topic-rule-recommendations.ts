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
    topicKey: "PRE_IPO",
    label: "Pre-IPO & Private Secondary Markets",
    sortOrder: 25,
    focus: "Pre-IPO share transactions, private-company secondary markets, tender offers, employee liquidity, and restricted stock.",
    suggestedKeywords: [
      "pre-IPO shares",
      "pre IPO shares",
      "pre-IPO marketplace",
      "private shares",
      "private company shares",
      "private secondary",
      "secondary market",
      "private secondary market",
      "private company tender offer",
      "employee liquidity",
      "late-stage private company",
      "unicorn secondary",
      "restricted stock",
      "tender offer",
      "secondary sale"
    ],
    broadTerms: ["restricted stock", "tender offer"],
    notes: [
      "This narrows the old Pre-IPO topic so ordinary IPO and private-placement coverage stays in Capital Formation.",
      "The best signals are private-secondary and employee-liquidity phrases rather than IPO by itself."
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
    label: "Artificial Intelligence",
    sortOrder: 50,
    focus: "Artificial intelligence, generative AI, machine learning, model governance, predictive analytics, and AI-related market or compliance risk.",
    suggestedKeywords: [
      "AI",
      "artificial intelligence",
      "generative AI",
      "GenAI",
      "machine learning",
      "ML",
      "deep learning",
      "neural network",
      "large language model",
      "LLM",
      "AI model",
      "foundation model",
      "transformer",
      "natural language processing",
      "NLP",
      "generative model",
      "synthetic data",
      "hallucination",
      "prompt engineering",
      "fine-tuning",
      "retrieval augmented generation",
      "RAG",
      "alignment",
      "responsible AI",
      "explainability",
      "bias",
      "fairness",
      "governance",
      "autonomous",
      "agentic",
      "reasoning",
      "model governance",
      "model risk",
      "algorithmic accountability",
      "predictive data analytics",
      "algorithmic trading",
      "algorithm",
      "automation",
      "robo-adviser"
    ],
    broadTerms: ["algorithm", "automation"],
    notes: [
      "This is now intentionally AI-only; fintech, regtech, cyber, and infrastructure stories should route to their own topics.",
      "AI is retained as an exact acronym, but generic technology terms are excluded to reduce false positives."
    ]
  },
  {
    topicKey: "TECH",
    label: "Fintech & RegTech Infrastructure",
    sortOrder: 55,
    focus: "Financial technology, regulatory technology, compliance automation, broker platforms, and regulated infrastructure.",
    suggestedKeywords: [
      "fintech",
      "regtech",
      "compliance technology",
      "surveillance technology",
      "trade surveillance",
      "regulatory reporting technology",
      "cloud migration",
      "API integration",
      "digital onboarding",
      "robo-adviser",
      "broker platform",
      "wealthtech",
      "compliance automation",
      "supervisory technology",
      "technology infrastructure",
      "financial technology",
      "regulatory technology"
    ],
    broadTerms: ["fintech", "regtech"],
    notes: [
      "This repurposes the old generic Tech bucket into a financial and regulatory infrastructure topic.",
      "Avoid standalone tech, software, hardware, platform, digital, cloud, and API unless paired with financial or regulatory context."
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
      "PredictIt",
      "CFTC event contracts",
      "CFTC prediction markets",
      "political betting",
      "sports event contract",
      "odds contract",
      "Commodity Futures Trading Commission",
      "CME Group",
      "Robinhood",
      "prediction-markets"
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
  },
  {
    topicKey: "COMMODITIES_ENERGY_MARKETS",
    label: "Commodities & Energy Markets",
    sortOrder: 160,
    focus: "Oil, gas, power, commodities futures, supply disruption, and energy-market risk.",
    suggestedKeywords: [
      "commodities",
      "commodities futures",
      "commodity futures",
      "oil prices",
      "crude oil",
      "brent crude",
      "WTI",
      "natural gas",
      "energy markets",
      "energy prices",
      "power markets",
      "electricity markets",
      "CFTC",
      "Strait of Hormuz",
      "shipping chokepoint",
      "supply disruption",
      "geopolitical risk premium"
    ],
    broadTerms: ["commodities", "oil", "gas", "energy"],
    notes: [
      "This prevents oil, gas, and commodities-futures coverage from being forced into generic Financial Markets.",
      "Prefer market, futures, or supply-risk phrases over standalone energy terms when precision matters."
    ]
  },
  {
    topicKey: "GEOPOLITICAL_TRADE_RISK",
    label: "Geopolitical & Trade Risk",
    sortOrder: 170,
    focus: "Geopolitical events, trade policy, tariffs, export controls, sanctions risk, and cross-border supply-chain disruption.",
    suggestedKeywords: [
      "geopolitical risk",
      "trade policy",
      "tariff",
      "tariffs",
      "export controls",
      "import restrictions",
      "supply chain",
      "supply chains",
      "sanctions risk",
      "foreign policy",
      "shipping lanes",
      "Strait of Hormuz",
      "Iran",
      "China",
      "national security",
      "cross-border restrictions",
      "trade war",
      "maritime security"
    ],
    broadTerms: ["Iran", "China", "national security"],
    notes: [
      "Country names are intentionally broad and should be monitored for false positives.",
      "The highest precision terms are trade-policy, export-control, sanctions-risk, and shipping-lane phrases."
    ]
  },
  {
    topicKey: "SRO_RULEMAKING_ARBITRATION",
    label: "SRO Rulemaking & Arbitration",
    sortOrder: 180,
    focus: "FINRA and exchange rule filings, SRO notices, arbitration procedures, and dispute-resolution infrastructure.",
    suggestedKeywords: [
      "proposed rule change",
      "rule filing",
      "SR-FINRA",
      "SR-CBOE",
      "SR-NYSE",
      "SR-NASDAQ",
      "FINRA arbitration",
      "dispute resolution",
      "arbitration procedure",
      "self-regulatory organization",
      "SRO rulemaking",
      "FINRA rules",
      "arbitration forum",
      "customer arbitration",
      "industry arbitration"
    ],
    broadTerms: ["rule filing", "dispute resolution"],
    notes: [
      "This separates procedural SRO rulemaking from general Securities Regulation.",
      "SR-form prefixes are clean signals for exchange and FINRA rule filings."
    ]
  },
  {
    topicKey: "BANKING_PAYMENTS",
    label: "Banking & Payments",
    sortOrder: 190,
    focus: "Bank supervision, deposits, bank capital, payment rails, card networks, and money movement infrastructure.",
    suggestedKeywords: [
      "bank",
      "banking",
      "deposits",
      "deposit insurance",
      "payments",
      "payment rails",
      "card network",
      "ACH",
      "wire transfer",
      "cross-border payments",
      "real-time payments",
      "instant payments",
      "bank merger",
      "bank capital",
      "liquidity coverage",
      "net interest income",
      "Basel",
      "Federal Reserve supervision"
    ],
    broadTerms: ["bank", "banking", "payments"],
    notes: [
      "Bank alone can be broad, but recent corpus coverage supports separating banking from macro and credit.",
      "Payments terms are useful for fintech, stablecoin, and banking-infrastructure crossovers."
    ]
  },
  {
    topicKey: "CONSUMER_PROTECTION_DECEPTIVE_PRACTICES",
    label: "Consumer Protection & Deceptive Practices",
    sortOrder: 200,
    focus: "FTC actions, deceptive advertising, subscription traps, unfair practices, junk fees, and consumer redress.",
    suggestedKeywords: [
      "FTC",
      "consumer protection",
      "deceptive advertising",
      "deceptive claims",
      "subscription scheme",
      "unfair practices",
      "unfair or deceptive",
      "junk fees",
      "refunds",
      "consumer fraud",
      "misleading claims",
      "negative option",
      "dark patterns",
      "telemarketing",
      "consumer redress"
    ],
    broadTerms: ["FTC", "refunds"],
    notes: [
      "This prevents FTC and consumer-fraud items from being misrouted into generic Enforcement only.",
      "Best used with unfair, deceptive, subscription, fee, or redress language."
    ]
  },
  {
    topicKey: "DATA_PRIVACY_DIGITAL_IDENTITY",
    label: "Data Privacy & Digital Identity",
    sortOrder: 210,
    focus: "Privacy, identity verification, personal data, biometrics, data brokers, and digital identity policy.",
    suggestedKeywords: [
      "data privacy",
      "personal data",
      "consumer data",
      "digital identity",
      "identity verification",
      "eID",
      "biometrics",
      "privacy rule",
      "data broker",
      "data security",
      "identity fraud",
      "AI identity",
      "personal identification code",
      "authentication",
      "credential theft"
    ],
    broadTerms: ["authentication", "data security"],
    notes: [
      "This separates privacy and identity issues from cyber incident response.",
      "Digital identity terms are useful for AI-agent, KYC, and e-government coverage."
    ]
  },
  {
    topicKey: "INVESTMENT_PRODUCTS_DERIVATIVES",
    label: "Investment Products & Derivatives",
    sortOrder: 220,
    focus: "ETFs, derivatives, structured products, funds, annuities, and retail investment product design.",
    suggestedKeywords: [
      "ETF",
      "exchange-traded fund",
      "options",
      "futures",
      "swaps",
      "derivatives",
      "structured product",
      "annuity",
      "mutual fund",
      "closed-end fund",
      "interval fund",
      "leveraged ETF",
      "inverse ETF",
      "covered call ETF",
      "yield product",
      "retail structured notes"
    ],
    broadTerms: ["options", "futures", "derivatives"],
    notes: [
      "This separates product wrappers and derivatives from generic Financial Markets.",
      "It will overlap with Crypto and Market Structure, but the overlap is analytically useful when products are the story."
    ]
  }
];

export const TOPIC_RULE_RECOMMENDATION_BY_KEY: Readonly<Record<string, TopicRuleRecommendation>> =
  Object.fromEntries(TOPIC_RULE_RECOMMENDATIONS.map((item) => [item.topicKey, item])) as Record<string, TopicRuleRecommendation>;

export function formatTopicRuleKeywords(keywords: readonly string[]): string {
  return keywords.join(", ");
}

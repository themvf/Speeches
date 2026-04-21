import type {
  IntelligenceSeedProfile,
  NarrativeRank,
  SystemTrend
} from "@/lib/intelligence-types";
export const SYSTEM_TRENDS: readonly SystemTrend[] = [
  { label: "Inflation Pressure", changePct: 67, direction: "up" },
  { label: "Trade Friction", changePct: 52, direction: "up" },
  { label: "Banking Stress", changePct: 41, direction: "up" },
  { label: "AI Risk", changePct: -18, direction: "down" }
];

export const NARRATIVE_LEADERBOARD: readonly NarrativeRank[] = [
  { label: "Geopolitical Supply Shock", severity: "CRITICAL", summary: "Conflict, sanctions, and logistics headlines are converging." },
  { label: "Inflation Reacceleration", severity: "CRITICAL", summary: "Energy costs are feeding back into rates coverage." },
  { label: "Banking Liquidity Pressure", severity: "HIGH", summary: "Funding stress is spreading into credit-market coverage." },
  { label: "AI Risk Rotation", severity: "HIGH", summary: "AI infrastructure remains firm while corporate quality is mixed." }
];

export const WHAT_CHANGED: readonly string[] = [
  "Trade conflict coverage doubled across policy and shipping sources.",
  "Energy-linked inflation stories are accelerating into rates commentary.",
  "Banking stress broadened from regional lenders into credit spreads.",
  "AI optimism cooled slightly as cost-control headlines increased."
];

export const INTELLIGENCE_PROFILES: readonly IntelligenceSeedProfile[] = [
  {
    id: "macro",
    label: "Inflation Shock",
    rawThemes: "ECON_INFLATION; CPI; CENTRAL_BANK; FEDERAL_RESERVE; OIL; ENERGY",
    oneLineSummary: "Energy-driven inflation pressure is rising quickly across global macro coverage.",
    narrative:
      "An inflation narrative is re-emerging, driven primarily by oil supply concerns and reinforced by central bank sensitivity to renewed price pressure.",
    whatChanged: [
      "Energy stories moved from commodity coverage into inflation and rates commentary.",
      "Central bank coverage is now framing oil risk as a policy constraint.",
      "Transport-cost headlines broadened the signal beyond crude prices."
    ],
    context: {
      window_label: "last 2 hours",
      previous_score: 18,
      baseline_score: 16,
      confidence: 86,
      theme_mentions: {
        INFLATION: { current_mentions: 120, baseline_mentions: 40, previous_mentions: 52 },
        CENTRAL_BANK: { current_mentions: 76, baseline_mentions: 31, previous_mentions: 42 },
        ENERGY: { current_mentions: 88, baseline_mentions: 29, previous_mentions: 34 }
      }
    },
    clusters: [
      {
        id: "energy-supply",
        title: "Energy Supply Shock Narrative",
        volume: 88,
        changePct: 203,
        importance: "Primary Driver",
        firstSeen: "2h ago",
        peakAcceleration: "last 30 min",
        summary: "Oil supply concerns are dominating global coverage, with multiple regions reporting disruption risk and tighter physical-market conditions.",
        takeaways: ["Oil supply concerns are now the lead narrative.", "Coverage has expanded beyond crude into refined products."]
      },
      {
        id: "policy-sensitivity",
        title: "Central Bank Inflation Sensitivity",
        volume: 76,
        changePct: 145,
        importance: "Secondary",
        firstSeen: "90 min ago",
        peakAcceleration: "last 45 min",
        summary: "Coverage is connecting renewed price pressure to central bank reaction functions, especially around rate-cut timing and inflation credibility.",
        takeaways: ["Rate-cut timing is being repriced.", "Policy credibility is becoming a repeated frame."]
      },
      {
        id: "consumer-prices",
        title: "Consumer Price Pass-Through",
        volume: 42,
        changePct: 91,
        importance: "Emerging",
        firstSeen: "54 min ago",
        peakAcceleration: "last 20 min",
        summary: "Articles increasingly frame energy costs as a near-term input to transport, food, and consumer-price expectations.",
        takeaways: ["Transport and food channels are being cited more often.", "Pass-through narrative is still early but rising."]
      }
    ],
    coverage: {
      totalArticles: 132,
      sourceCount: 17,
      regionCount: 5,
      changePct: 180,
      diversity: "High",
      dimensions: ["policy", "commodities", "logistics"]
    },
    sourceDistribution: [
      { source: "Reuters", count: 22 },
      { source: "Bloomberg", count: 18 },
      { source: "Financial Times", count: 15 },
      { source: "CNBC", count: 10 }
    ],
    keyTakeaways: [
      "Energy supply risk is now the strongest inflation driver.",
      "Rates coverage is reacting to renewed price-pressure headlines.",
      "Transport and consumer-price pass-through are broadening the narrative."
    ],
    evidence: [
      {
        id: "macro-1",
        headline: "Oil prices surge as supply concerns intensify",
        source: "Reuters",
        timestamp: "12 min ago",
        excerpt: "Energy traders pointed to tighter supply expectations and renewed geopolitical risk premiums.",
        explanation: "Driving energy-linked inflation narrative",
        relatedThemes: ["ENERGY", "INFLATION"],
        clusterId: "energy-supply",
        credibility: 94,
        impact: 96
      },
      {
        id: "macro-2",
        headline: "Central bank officials flag caution on inflation outlook",
        source: "Bloomberg",
        timestamp: "19 min ago",
        excerpt: "Policy-sensitive coverage emphasized that price shocks could complicate the rate path.",
        explanation: "Reinforcing policy sensitivity to price pressure",
        relatedThemes: ["CENTRAL_BANK", "INFLATION", "INTEREST_RATES"],
        clusterId: "policy-sensitivity",
        credibility: 92,
        impact: 90
      },
      {
        id: "macro-3",
        headline: "Fuel and freight costs rise across regional markets",
        source: "Financial Times",
        timestamp: "31 min ago",
        excerpt: "Transport costs are being cited as a channel for broader goods inflation.",
        explanation: "Connecting oil shock to consumer-price pass-through",
        relatedThemes: ["ENERGY", "SUPPLY_CHAIN", "INFLATION"],
        clusterId: "consumer-prices",
        credibility: 91,
        impact: 82
      },
      {
        id: "macro-4",
        headline: "Bond traders pare back expectations for near-term cuts",
        source: "MarketWatch",
        timestamp: "44 min ago",
        excerpt: "Rates commentary shifted toward inflation risk rather than growth weakness.",
        explanation: "Linking inflation narrative to rates pressure",
        relatedThemes: ["INFLATION", "INTEREST_RATES", "CENTRAL_BANK"],
        clusterId: "policy-sensitivity",
        credibility: 83,
        impact: 79
      },
      {
        id: "macro-5",
        headline: "Commodity desks cite broader repricing in energy-linked contracts",
        source: "Dow Jones",
        timestamp: "58 min ago",
        excerpt: "Coverage broadened from crude into refined products and transport-sensitive inputs.",
        explanation: "Showing breadth beyond a single oil-price move",
        relatedThemes: ["ENERGY", "COMMODITIES"],
        clusterId: "energy-supply",
        credibility: 87,
        impact: 74
      }
    ]
  },
  {
    id: "bank",
    label: "Bank Stress",
    rawThemes: "ECON_BANKING; BANK_FAILURE; BANK_RUN; ECON_CREDIT; LIQUIDITY; FUNDING_STRESS",
    oneLineSummary: "Banking and credit-stress coverage is accelerating, with funding pressure becoming the core narrative.",
    narrative:
      "Financial-stress coverage is clustering around bank balance-sheet risk, liquidity preference, and spillover into credit markets.",
    whatChanged: [
      "Funding-pressure headlines moved from bank equities into credit-market coverage.",
      "Money-market flow stories started confirming liquidity preference.",
      "Regional bank stress is now showing up in broader risk-appetite framing."
    ],
    context: {
      window_label: "last 2 hours",
      previous_score: 20,
      baseline_score: 14,
      confidence: 88,
      theme_mentions: {
        BANKING: { current_mentions: 92, baseline_mentions: 18, previous_mentions: 29 },
        CREDIT_MARKETS: { current_mentions: 68, baseline_mentions: 22, previous_mentions: 33 },
        LIQUIDITY: { current_mentions: 54, baseline_mentions: 19, previous_mentions: 25 }
      }
    },
    clusters: [
      {
        id: "regional-banks",
        title: "Regional Bank Stress",
        volume: 92,
        changePct: 411,
        importance: "Primary Driver",
        firstSeen: "2h ago",
        peakAcceleration: "last 25 min",
        summary: "Coverage is concentrated around deposit sensitivity, unrealized losses, and renewed concern over smaller bank funding channels.",
        takeaways: ["Funding sensitivity is the repeated frame.", "Regional lenders are the main equity transmission channel."]
      },
      {
        id: "credit-spreads",
        title: "Credit Spread Contagion",
        volume: 68,
        changePct: 209,
        importance: "Secondary",
        firstSeen: "75 min ago",
        peakAcceleration: "last 35 min",
        summary: "Credit-market stories are increasingly connecting bank stress to widening spreads and weaker risk appetite.",
        takeaways: ["High-yield spreads are the clearest market channel.", "Credit coverage confirms spillover beyond bank equities."]
      }
    ],
    coverage: {
      totalArticles: 118,
      sourceCount: 14,
      regionCount: 3,
      changePct: 241,
      diversity: "High",
      dimensions: ["banks", "credit", "liquidity"]
    },
    sourceDistribution: [
      { source: "Bloomberg", count: 24 },
      { source: "Reuters", count: 20 },
      { source: "Wall Street Journal", count: 13 },
      { source: "Dow Jones", count: 8 }
    ],
    keyTakeaways: [
      "Funding concerns are leading the financial-stress narrative.",
      "Credit markets are beginning to validate the banking signal.",
      "Liquidity preference is visible in money-market coverage."
    ],
    evidence: [
      {
        id: "bank-1",
        headline: "Regional lenders fall as funding concerns resurface",
        source: "Bloomberg",
        timestamp: "8 min ago",
        excerpt: "Shares weakened as investors revisited deposit stability and securities-book exposure.",
        explanation: "Driving banking-stress narrative",
        relatedThemes: ["BANKING", "LIQUIDITY"],
        clusterId: "regional-banks",
        credibility: 92,
        impact: 94
      },
      {
        id: "bank-2",
        headline: "Credit desks report renewed pressure in high-yield spreads",
        source: "Reuters",
        timestamp: "17 min ago",
        excerpt: "Traders cited bank-risk headlines as a catalyst for reduced appetite in lower-quality credit.",
        explanation: "Linking banking stress to credit markets",
        relatedThemes: ["CREDIT_MARKETS", "BANKING"],
        clusterId: "credit-spreads",
        credibility: 94,
        impact: 88
      },
      {
        id: "bank-3",
        headline: "Money-market flows rise as investors seek liquidity",
        source: "Wall Street Journal",
        timestamp: "36 min ago",
        excerpt: "Cash-like products saw renewed inflows amid questions about financial-sector resilience.",
        explanation: "Confirming liquidity preference",
        relatedThemes: ["LIQUIDITY", "BANKING"],
        clusterId: "regional-banks",
        credibility: 90,
        impact: 76
      }
    ]
  },
  {
    id: "geopolitical",
    label: "Trade Conflict",
    rawThemes: "TRADE_SANCTIONS; TARIFFS; GEOPOLITICAL; MILITARY_ATTACK; SUPPLY_CHAIN",
    oneLineSummary: "Trade and conflict coverage is converging around sanctions, logistics disruption, and supply-chain risk.",
    narrative:
      "A geopolitical supply-shock narrative is building as sanctions headlines, conflict coverage, and logistics stories reinforce each other.",
    whatChanged: [
      "Sanctions headlines expanded into shipping and export-channel evidence.",
      "Freight-rate coverage is converting conflict risk into supply-chain pressure.",
      "Trade policy stories are now reinforcing the geopolitical risk signal."
    ],
    context: {
      window_label: "last 2 hours",
      previous_score: 24,
      baseline_score: 19,
      confidence: 82,
      theme_mentions: {
        CONFLICT: { current_mentions: 71, baseline_mentions: 15, previous_mentions: 22 },
        SANCTIONS: { current_mentions: 63, baseline_mentions: 20, previous_mentions: 28 },
        SUPPLY_CHAIN: { current_mentions: 49, baseline_mentions: 16, previous_mentions: 19 },
        TRADE: { current_mentions: 57, baseline_mentions: 24, previous_mentions: 35 },
        GEOPOLITICS: { current_mentions: 45, baseline_mentions: 18, previous_mentions: 25 }
      }
    },
    clusters: [
      {
        id: "sanctions",
        title: "Sanctions Escalation",
        volume: 63,
        changePct: 215,
        importance: "Primary Driver",
        firstSeen: "2h ago",
        peakAcceleration: "last 40 min",
        summary: "Coverage is clustering around new restrictions, export controls, and retaliatory trade measures.",
        takeaways: ["Sanctions are targeting transport and export channels.", "Trade-policy escalation risk remains high."]
      },
      {
        id: "shipping",
        title: "Logistics Disruption",
        volume: 49,
        changePct: 206,
        importance: "Secondary",
        firstSeen: "84 min ago",
        peakAcceleration: "last 30 min",
        summary: "Shipping and logistics stories are increasingly tied to conflict risk and rerouting costs.",
        takeaways: ["Freight-rate evidence is already reacting.", "Rerouting costs connect geopolitical risk to inflation risk."]
      }
    ],
    coverage: {
      totalArticles: 126,
      sourceCount: 16,
      regionCount: 6,
      changePct: 196,
      diversity: "High",
      dimensions: ["policy", "shipping", "energy"]
    },
    sourceDistribution: [
      { source: "Reuters", count: 21 },
      { source: "Financial Times", count: 16 },
      { source: "Lloyd's List", count: 12 },
      { source: "Bloomberg", count: 11 }
    ],
    keyTakeaways: [
      "Sanctions are moving from policy headlines into transport evidence.",
      "Shipping costs are already reacting to conflict risk.",
      "Energy and inflation signals are likely downstream beneficiaries."
    ],
    evidence: [
      {
        id: "geo-1",
        headline: "New sanctions package targets energy and shipping channels",
        source: "Reuters",
        timestamp: "15 min ago",
        excerpt: "Restrictions focused on export flows and transport intermediaries.",
        explanation: "Driving sanctions and trade-friction narrative",
        relatedThemes: ["SANCTIONS", "TRADE", "ENERGY"],
        clusterId: "sanctions",
        credibility: 94,
        impact: 91
      },
      {
        id: "geo-2",
        headline: "Freight rates jump as carriers reroute vessels",
        source: "Lloyd's List",
        timestamp: "27 min ago",
        excerpt: "Shipping costs moved higher as operators avoided higher-risk corridors.",
        explanation: "Turning conflict risk into supply-chain pressure",
        relatedThemes: ["SUPPLY_CHAIN", "CONFLICT"],
        clusterId: "shipping",
        credibility: 86,
        impact: 84
      },
      {
        id: "geo-3",
        headline: "Officials warn trade restrictions could broaden",
        source: "Financial Times",
        timestamp: "41 min ago",
        excerpt: "Policy officials framed export controls as likely to remain in focus.",
        explanation: "Extending the narrative into trade policy",
        relatedThemes: ["TRADE", "SANCTIONS", "GEOPOLITICS"],
        clusterId: "sanctions",
        credibility: 91,
        impact: 78
      }
    ]
  },
  {
    id: "modern",
    label: "AI Risk Rotation",
    rawThemes: "GENERATIVE_AI; AI; SEMICONDUCTOR; SOFTWARE; CRYPTOCURRENCY; BITCOIN; EARNINGS; LAYOFFS; SEC; FUNDING_STRESS",
    oneLineSummary: "AI infrastructure is the lead driver, with crypto participation and corporate-cost narratives adding breadth.",
    narrative:
      "Modern-market coverage is led by AI model deployment, semiconductor demand, and data-center investment, while crypto flows, regulation, and funding conditions shape the quality of the signal.",
    whatChanged: [
      "AI infrastructure coverage remains the lead thread while generic technology is secondary.",
      "Crypto participation is rising but contributes less than AI and semiconductors.",
      "Regulation and funding stories are acting as quality filters for the signal."
    ],
    context: {
      window_label: "last 2 hours",
      previous_score: 12,
      baseline_score: 10,
      confidence: 74,
      theme_mentions: {
        AI: { current_mentions: 74, baseline_mentions: 26, previous_mentions: 39 },
        TECHNOLOGY: { current_mentions: 58, baseline_mentions: 43, previous_mentions: 47 },
        CRYPTO: { current_mentions: 37, baseline_mentions: 21, previous_mentions: 24 },
        CORPORATE_ACTIVITY: { current_mentions: 29, baseline_mentions: 24, previous_mentions: 22 },
        REGULATION: { current_mentions: 18, baseline_mentions: 15, previous_mentions: 14 },
        LIQUIDITY: { current_mentions: 14, baseline_mentions: 10, previous_mentions: 11 }
      }
    },
    clusters: [
      {
        id: "ai-infra",
        title: "AI Infrastructure Demand",
        volume: 74,
        changePct: 185,
        importance: "Primary Driver",
        firstSeen: "2h ago",
        peakAcceleration: "last 55 min",
        summary: "Coverage is concentrated around model deployment, chip demand, data-center spend, and AI infrastructure capacity.",
        takeaways: ["AI model deployment is the strongest thread.", "Semiconductor demand remains the clearest market channel."]
      },
      {
        id: "crypto-flow",
        title: "Digital-Asset Participation",
        volume: 37,
        changePct: 76,
        importance: "Secondary",
        firstSeen: "68 min ago",
        peakAcceleration: "last 35 min",
        summary: "Crypto coverage is rising, with flow stories and exchange-linked headlines supporting participation.",
        takeaways: ["Bitcoin participation is improving.", "Crypto beta is supportive but less broad than AI infrastructure."]
      },
      {
        id: "policy-funding",
        title: "Regulation and Funding Conditions",
        volume: 24,
        changePct: 36,
        importance: "Emerging",
        firstSeen: "51 min ago",
        peakAcceleration: "last 20 min",
        summary: "Coverage is linking AI and crypto participation to compliance scrutiny, cash-flow discipline, and financing conditions.",
        takeaways: ["Regulation is a quality filter.", "Funding conditions are a secondary constraint."]
      }
    ],
    coverage: {
      totalArticles: 83,
      sourceCount: 11,
      regionCount: 3,
      changePct: 42,
      diversity: "Medium",
      dimensions: ["AI", "semiconductors", "crypto", "regulation"]
    },
    sourceDistribution: [
      { source: "Bloomberg", count: 14 },
      { source: "CNBC", count: 11 },
      { source: "CoinDesk", count: 9 },
      { source: "Reuters", count: 7 }
    ],
    keyTakeaways: [
      "AI is now separate from general technology in the signal.",
      "Crypto participation is rising but still narrower than AI coverage.",
      "Regulation and funding conditions reduce the quality of the signal."
    ],
    evidence: [
      {
        id: "modern-1",
        headline: "Chipmakers rise as AI infrastructure demand broadens",
        source: "Bloomberg",
        timestamp: "11 min ago",
        excerpt: "Semiconductor coverage pointed to expanding data-center investment and strong forward demand.",
        explanation: "Driving AI infrastructure and chip-demand coverage",
        relatedThemes: ["AI", "TECHNOLOGY"],
        clusterId: "ai-infra",
        credibility: 92,
        impact: 84
      },
      {
        id: "modern-2",
        headline: "Bitcoin-linked shares gain with renewed crypto flows",
        source: "CoinDesk",
        timestamp: "24 min ago",
        excerpt: "Digital-asset coverage focused on participation and exchange-traded flow momentum.",
        explanation: "Driving crypto beta",
        relatedThemes: ["CRYPTO"],
        clusterId: "crypto-flow",
        credibility: 80,
        impact: 74
      },
      {
        id: "modern-3",
        headline: "Tech earnings coverage focuses on AI capex and layoffs",
        source: "CNBC",
        timestamp: "52 min ago",
        excerpt: "Corporate stories paired AI investment with cost-cutting and margin discipline.",
        explanation: "Adding corporate-activity context",
        relatedThemes: ["AI", "TECHNOLOGY", "CORPORATE_ACTIVITY"],
        clusterId: "ai-infra",
        credibility: 78,
        impact: 66
      },
      {
        id: "modern-4",
        headline: "AI platforms face renewed compliance and funding scrutiny",
        source: "Reuters",
        timestamp: "59 min ago",
        excerpt: "Coverage tied model deployment to compliance requirements, cash-flow discipline, and financing conditions.",
        explanation: "Adding regulation and liquidity context",
        relatedThemes: ["AI", "REGULATION", "LIQUIDITY"],
        clusterId: "policy-funding",
        credibility: 88,
        impact: 62
      }
    ]
  }
];

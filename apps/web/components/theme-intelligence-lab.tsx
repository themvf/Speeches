"use client";

import { useMemo, useState } from "react";
import {
  scoreThemeArticle,
  THEME_MAPPING,
  type MarketImpactDirection,
  type NormalizedTheme,
  type SignalTrendDirection,
  type ThemeCategory,
  type ThemeContextInput,
  type ThemeFrequencySignal,
  type ThemeSeverity
} from "@/lib/theme-intelligence";

type ArticleSort = "recency" | "credibility" | "impact";
type EvidenceTab = "clusters" | "articles" | "impact";

type EvidenceArticle = {
  id: string;
  headline: string;
  source: string;
  timestamp: string;
  excerpt: string;
  explanation: string;
  relatedThemes: NormalizedTheme[];
  clusterId: string;
  credibility: number;
  impact: number;
};

type EvidenceCluster = {
  id: string;
  title: string;
  volume: number;
  changePct: number;
  importance: "Primary Driver" | "Secondary" | "Emerging";
  firstSeen: string;
  peakAcceleration: string;
  summary: string;
  takeaways: string[];
};

type IntelligenceProfile = {
  id: string;
  label: string;
  rawThemes: string;
  context: ThemeContextInput;
  oneLineSummary: string;
  narrative: string;
  coverage: {
    totalArticles: number;
    sourceCount: number;
    regionCount: number;
    changePct: number;
    diversity: "High" | "Medium" | "Low";
    dimensions: string[];
  };
  sourceDistribution: { source: string; count: number }[];
  keyTakeaways: string[];
  evidence: EvidenceArticle[];
  clusters: EvidenceCluster[];
};

type SystemTrend = {
  label: string;
  changePct: number;
  direction: "up" | "down";
};

type NarrativeRank = {
  label: string;
  severity: ThemeSeverity;
  summary: string;
};

const CATEGORY_LABELS: Record<ThemeCategory, string> = {
  MACRO: "Macro",
  FINANCIAL_SYSTEM: "Financial System",
  GEOPOLITICS: "Geopolitics",
  REAL_ECONOMY: "Real Economy",
  MODERN_THEMES: "Modern Themes"
};

const SYSTEM_TRENDS: readonly SystemTrend[] = [
  { label: "Inflation Pressure", changePct: 67, direction: "up" },
  { label: "Trade Friction", changePct: 52, direction: "up" },
  { label: "Banking Stress", changePct: 41, direction: "up" },
  { label: "AI Risk", changePct: -18, direction: "down" }
];

const NARRATIVE_LEADERBOARD: readonly NarrativeRank[] = [
  { label: "Geopolitical Supply Shock", severity: "CRITICAL", summary: "Conflict, sanctions, and logistics headlines are converging." },
  { label: "Inflation Reacceleration", severity: "CRITICAL", summary: "Energy costs are feeding back into rates coverage." },
  { label: "Banking Liquidity Pressure", severity: "HIGH", summary: "Funding stress is spreading into credit-market coverage." },
  { label: "AI Risk Rotation", severity: "HIGH", summary: "AI infrastructure remains firm while corporate quality is mixed." }
];

const WHAT_CHANGED: readonly string[] = [
  "Trade conflict coverage doubled across policy and shipping sources.",
  "Energy-linked inflation stories are accelerating into rates commentary.",
  "Banking stress broadened from regional lenders into credit spreads.",
  "AI optimism cooled slightly as cost-control headlines increased."
];

const PROFILES: readonly IntelligenceProfile[] = [
  {
    id: "macro",
    label: "Inflation Shock",
    rawThemes: "ECON_INFLATION; CPI; CENTRAL_BANK; FEDERAL_RESERVE; OIL; ENERGY",
    oneLineSummary: "Energy-driven inflation pressure is rising quickly across global macro coverage.",
    narrative:
      "An inflation narrative is re-emerging, driven primarily by oil supply concerns and reinforced by central bank sensitivity to renewed price pressure.",
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

const SEVERITY_STYLE: Record<ThemeSeverity, { color: string; background: string; border: string }> = {
  CRITICAL: { color: "#ff6b7f", background: "rgba(255,107,127,0.12)", border: "rgba(255,107,127,0.34)" },
  HIGH: { color: "var(--warn)", background: "rgba(242,171,67,0.12)", border: "rgba(242,171,67,0.32)" },
  NORMAL: { color: "var(--ok)", background: "rgba(65,211,157,0.1)", border: "rgba(65,211,157,0.28)" }
};

const TREND_STYLE: Record<SignalTrendDirection, { color: string; label: string }> = {
  ACCELERATING: { color: "var(--danger)", label: "Accelerating" },
  RISING: { color: "var(--warn)", label: "Rising" },
  STABLE: { color: "var(--accent)", label: "Stable" },
  FALLING: { color: "var(--ok)", label: "Cooling" }
};

const DIRECTION_STYLE: Record<MarketImpactDirection, { color: string; background: string; label: string }> = {
  UP: { color: "var(--ok)", background: "rgba(65,211,157,0.1)", label: "Up" },
  DOWN: { color: "var(--danger)", background: "rgba(255,107,127,0.1)", label: "Down" },
  MIXED: { color: "var(--warn)", background: "rgba(242,171,67,0.1)", label: "Mixed" }
};

function formatTheme(theme: string): string {
  return theme.replace(/_/g, " ");
}

function formatSigned(value: number): string {
  return value > 0 ? `+${value}` : String(value);
}

function formatPct(value: number): string {
  return `${formatSigned(value)}%`;
}

function formatContribution(value: number): string {
  return `${Number.isInteger(value) ? value : value.toFixed(1)}%`;
}

function sortArticles(articles: readonly EvidenceArticle[], sort: ArticleSort): EvidenceArticle[] {
  const minutes = (value: string) => Number.parseInt(value, 10) || 999;
  return [...articles].sort((a, b) => {
    if (sort === "credibility") return b.credibility - a.credibility || b.impact - a.impact;
    if (sort === "impact") return b.impact - a.impact || b.credibility - a.credibility;
    return minutes(a.timestamp) - minutes(b.timestamp);
  });
}

function expandEvidenceArticles(profile: IntelligenceProfile, count = 30): EvidenceArticle[] {
  if (profile.evidence.length === 0) {
    return [];
  }

  return Array.from({ length: count }, (_, index) => {
    const base = profile.evidence[index % profile.evidence.length];
    const cycle = Math.floor(index / profile.evidence.length);
    const cluster = profile.clusters.find((item) => item.id === base.clusterId);

    if (cycle === 0) {
      return base;
    }

    return {
      ...base,
      id: `${base.id}-expanded-${cycle}`,
      headline: `${base.headline} (${cluster?.title ?? "follow-up"} ${cycle + 1})`,
      timestamp: `${12 + index * 4} min ago`,
      excerpt: `${base.excerpt} Related coverage continues to reinforce the same theme cluster.`,
      impact: Math.max(1, base.impact - cycle * 2),
      credibility: Math.max(1, base.credibility - cycle)
    };
  });
}

function getThemeEvidence(theme: NormalizedTheme, profile: IntelligenceProfile) {
  const articles = profile.evidence.filter((article) => article.relatedThemes.includes(theme));
  const clusterIds = new Set(articles.map((article) => article.clusterId));
  const clusters = profile.clusters.filter((cluster) => clusterIds.has(cluster.id));

  return {
    articles,
    clusters
  };
}

function getProfileIdForTheme(theme: NormalizedTheme): string {
  const directProfile = PROFILES.find((profile) => scoreThemeArticle({ raw_themes: profile.rawThemes, context: profile.context }).normalized_theme_list.includes(theme));
  if (directProfile) {
    return directProfile.id;
  }

  const definition = THEME_MAPPING.find((item) => item.normalized_theme === theme);
  if (!definition) {
    return PROFILES[0].id;
  }

  if (definition.category === "FINANCIAL_SYSTEM") {
    return "bank";
  }
  if (definition.category === "GEOPOLITICS") {
    return "geopolitical";
  }
  if (definition.category === "REAL_ECONOMY") {
    return theme === "SUPPLY_CHAIN" ? "geopolitical" : "macro";
  }
  if (definition.category === "MODERN_THEMES") {
    return "modern";
  }
  return "macro";
}

function StatusBadge({ severity }: { severity: ThemeSeverity }) {
  const style = SEVERITY_STYLE[severity];
  return (
    <span className="inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold uppercase" style={{ color: style.color, background: style.background, border: `1px solid ${style.border}` }}>
      {severity}
    </span>
  );
}

const EVIDENCE_TABS: readonly { id: EvidenceTab; label: string }[] = [
  { id: "clusters", label: "Clusters" },
  { id: "articles", label: "Articles" },
  { id: "impact", label: "Impact" }
];

function SystemOverview() {
  return (
    <section className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.62)] p-4">
      <div className="grid gap-4 xl:grid-cols-3">
        <div>
          <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Top Trends</p>
          <div className="mt-2 flex flex-wrap gap-2">
            {SYSTEM_TRENDS.map((trend) => (
              <span
                key={trend.label}
                className="rounded-full border border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.45)] px-3 py-1 text-xs font-semibold text-[color:var(--ink)]"
              >
                {trend.label}{" "}
                <span className="tabular-nums" style={{ color: trend.direction === "up" ? "var(--warn)" : "var(--ok)" }}>
                  {formatPct(trend.changePct)}
                </span>
              </span>
            ))}
          </div>
        </div>

        <div>
          <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">What Changed</p>
          <div className="mt-2 space-y-1">
            {WHAT_CHANGED.slice(0, 3).map((item) => (
              <p key={item} className="line-clamp-1 text-xs text-[color:var(--ink)]">
                - {item}
              </p>
            ))}
          </div>
        </div>

        <div>
          <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Dominant Narratives</p>
          <div className="mt-2 space-y-1.5">
            {NARRATIVE_LEADERBOARD.slice(0, 3).map((item, index) => (
              <div key={item.label} className="flex items-center justify-between gap-3 text-xs">
                <span className="min-w-0 truncate font-semibold text-[color:var(--ink)]">
                  {index + 1}. {item.label}
                </span>
                <span className="shrink-0 rounded-full border border-[color:var(--line-soft)] px-2 py-0.5 text-[10px] font-semibold text-[color:var(--ink-faint)]">
                  {item.severity}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

function AllThemesStrip({
  selectedTheme,
  activeThemes,
  onThemeSelect
}: {
  selectedTheme: NormalizedTheme | null;
  activeThemes: readonly NormalizedTheme[];
  onThemeSelect: (theme: NormalizedTheme) => void;
}) {
  const groupedThemes = THEME_MAPPING.reduce<Record<ThemeCategory, NormalizedTheme[]>>(
    (acc, item) => {
      acc[item.category].push(item.normalized_theme);
      return acc;
    },
    {
      MACRO: [],
      FINANCIAL_SYSTEM: [],
      GEOPOLITICS: [],
      REAL_ECONOMY: [],
      MODERN_THEMES: []
    }
  );
  const activeThemeSet = new Set(activeThemes);

  return (
    <section className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.52)] p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">All Themes</p>
          <h2 className="mt-1 text-base font-semibold text-[color:var(--ink)]" style={{ letterSpacing: 0 }}>
            Full normalized taxonomy
          </h2>
        </div>
        <span className="rounded-full border border-[color:var(--line-soft)] px-3 py-1 text-xs text-[color:var(--ink-faint)]">
          {THEME_MAPPING.length} themes visible
        </span>
      </div>

      <div className="mt-4 grid gap-3 xl:grid-cols-5">
        {(Object.keys(groupedThemes) as ThemeCategory[]).map((category) => (
          <div key={category}>
            <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">{CATEGORY_LABELS[category]}</p>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {groupedThemes[category].map((theme) => (
                <button
                  key={theme}
                  type="button"
                  onClick={() => onThemeSelect(theme)}
                  aria-pressed={selectedTheme === theme}
                  className={`min-h-7 rounded-full border px-2.5 py-0.5 text-[11px] font-semibold transition-colors ${
                    selectedTheme === theme
                      ? "border-[color:var(--line-strong)] bg-[color:rgba(79,213,255,0.18)] text-[color:var(--ink)]"
                      : activeThemeSet.has(theme)
                        ? "border-[color:rgba(79,213,255,0.3)] bg-[color:rgba(79,213,255,0.08)] text-[color:var(--accent)] hover:border-[color:var(--line-strong)]"
                        : "border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.38)] text-[color:var(--ink)] hover:border-[color:var(--line-strong)]"
                  }`}
                >
                  {formatTheme(theme)}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function SignalCard({
  profile,
  active,
  onClick
}: {
  profile: IntelligenceProfile;
  active: boolean;
  onClick: () => void;
}) {
  const signal = scoreThemeArticle({ raw_themes: profile.rawThemes, context: profile.context });
  const trend = TREND_STYLE[signal.trend.direction];

  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`min-h-[112px] rounded-xl border p-4 text-left transition-colors ${
        active
          ? "border-[color:var(--line-strong)] bg-[color:rgba(79,213,255,0.13)]"
          : "border-[color:var(--line)] bg-[color:rgba(9,21,34,0.58)] hover:border-[color:var(--line-strong)]"
      }`}
    >
      <div className="flex items-center justify-between gap-3">
        <span className="text-sm font-semibold text-[color:var(--ink)]">{profile.label}</span>
        <StatusBadge severity={signal.severity} />
      </div>
      <p className="mt-2 line-clamp-2 text-xs text-[color:var(--ink-faint)]">{profile.oneLineSummary}</p>
      <div className="mt-2 flex items-center justify-between gap-3 text-xs">
        <span className="font-semibold" style={{ color: trend.color }}>
          {trend.label}
        </span>
        <span className="text-[color:var(--ink-faint)]">{formatSigned(signal.trend.delta)} vs prior window</span>
      </div>
    </button>
  );
}

function EvidenceItem({ article }: { article: EvidenceArticle }) {
  return (
    <article className="rounded-lg border border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.48)] p-3">
      <h3 className="text-sm font-semibold text-[color:var(--ink)]" style={{ letterSpacing: 0 }}>
        {article.headline}
      </h3>
      <p className="mt-1 text-[11px] font-semibold uppercase text-[color:var(--ink-faint)]">
        {article.source} - {article.timestamp}
      </p>
      <p className="mt-2 text-xs text-[color:var(--ink-faint)]">{article.excerpt}</p>
      <p className="mt-2 text-xs font-medium text-[color:var(--accent)]">-&gt; {article.explanation}</p>
    </article>
  );
}

function ThemeDetailPanel({
  theme,
  profile,
  driver
}: {
  theme: NormalizedTheme;
  profile: IntelligenceProfile;
  driver: ThemeFrequencySignal | undefined;
}) {
  const definition = THEME_MAPPING.find((item) => item.normalized_theme === theme);
  const evidence = getThemeEvidence(theme, profile);
  const active = Boolean(driver);

  return (
    <section className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.58)] p-4">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Selected Theme</p>
          <h2 className="mt-1 text-xl font-semibold text-[color:var(--ink)]" style={{ letterSpacing: 0 }}>
            {formatTheme(theme)}
          </h2>
          <p className="mt-1 text-xs text-[color:var(--ink-faint)]">
            {definition ? CATEGORY_LABELS[definition.category] : "Theme"} - Priority {definition?.weight ?? 0}/10
          </p>
        </div>
        <span
          className={`rounded-full border px-3 py-1 text-xs font-semibold ${
            active
              ? "border-[color:rgba(79,213,255,0.32)] bg-[color:rgba(79,213,255,0.1)] text-[color:var(--accent)]"
              : "border-[color:var(--line-soft)] text-[color:var(--ink-faint)]"
          }`}
        >
          {active ? `Active in ${profile.label}` : `Not active in ${profile.label}`}
        </span>
      </div>

      {driver ? (
        <div className="mt-4 grid gap-3 md:grid-cols-3">
          <div className="rounded-lg border border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.38)] p-3">
            <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Mentions</p>
            <p className="mt-1 text-2xl font-semibold tabular-nums text-[color:var(--ink)]">{driver.current_mentions}</p>
          </div>
          <div className="rounded-lg border border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.38)] p-3">
            <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Spike</p>
            <p className="mt-1 text-2xl font-semibold tabular-nums text-[color:var(--accent)]">{formatPct(driver.spike_pct)}</p>
          </div>
          <div className="rounded-lg border border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.38)] p-3">
            <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Contribution</p>
            <p className="mt-1 text-2xl font-semibold tabular-nums text-[color:var(--ink)]">{formatContribution(driver.contribution_pct)}</p>
          </div>
        </div>
      ) : (
        <p className="mt-4 rounded-lg border border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.35)] p-3 text-sm text-[color:var(--ink-faint)]">
          This theme is part of the full taxonomy, but it is not contributing to the selected signal. Choose a signal where it appears to inspect contribution and evidence.
        </p>
      )}

      {evidence.clusters.length > 0 || evidence.articles.length > 0 ? (
        <div className="mt-4 grid gap-4 lg:grid-cols-[0.85fr_1.15fr]">
          <div>
            <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Linked Clusters</p>
            <div className="mt-2 flex flex-wrap gap-2">
              {evidence.clusters.map((cluster) => (
                <span key={cluster.id} className="rounded-full border border-[color:var(--line-soft)] px-3 py-1 text-xs text-[color:var(--ink)]">
                  {cluster.title}
                </span>
              ))}
            </div>
          </div>
          <div>
            <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Supporting Evidence</p>
            <div className="mt-2 space-y-2">
              {evidence.articles.slice(0, 3).map((article) => (
                <div key={article.id} className="rounded-lg border border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.35)] p-3">
                  <p className="text-sm font-semibold text-[color:var(--ink)]">{article.headline}</p>
                  <p className="mt-1 text-[11px] uppercase text-[color:var(--ink-faint)]">
                    {article.source} - {article.timestamp}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}

function DriverMetricRow({ driver, profile }: { driver: ThemeFrequencySignal; profile: IntelligenceProfile }) {
  const evidence = getThemeEvidence(driver.normalized_theme, profile);
  const firstArticle = evidence.articles[0];
  const firstCluster = evidence.clusters[0];

  return (
    <div className="rounded-lg border border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.38)] p-3">
      <div className="grid gap-2 sm:grid-cols-[minmax(0,1fr)_82px_78px_94px] sm:items-center">
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-[color:var(--ink)]">{formatTheme(driver.normalized_theme)}</p>
          {firstArticle ? (
            <p className="mt-1 line-clamp-1 text-[11px] text-[color:var(--ink-faint)]">
              Evidence: {firstArticle.explanation}
            </p>
          ) : (
            <p className="mt-1 text-[11px] text-[color:var(--ink-faint)]">Evidence link pending</p>
          )}
        </div>
        <p className="text-xs tabular-nums text-[color:var(--ink-faint)]">{driver.current_mentions} mentions</p>
        <p className="text-xs font-semibold tabular-nums text-[color:var(--accent)]">{formatPct(driver.spike_pct)}</p>
        <p className="text-xs font-semibold tabular-nums text-[color:var(--ink)]">{formatContribution(driver.contribution_pct)}</p>
      </div>
      {firstCluster ? (
        <p className="mt-2 text-[11px] text-[color:var(--ink-faint)]">
          Cluster: {firstCluster.title} - {evidence.articles.length} supporting {evidence.articles.length === 1 ? "article" : "articles"}
        </p>
      ) : null}
    </div>
  );
}

function CompactSignalPanel({
  profile,
  signal,
  trend,
  primaryDrivers,
  secondaryDrivers,
  additionalDriversOpen,
  onToggleAdditionalDrivers
}: {
  profile: IntelligenceProfile;
  signal: ReturnType<typeof scoreThemeArticle>;
  trend: { color: string; label: string };
  primaryDrivers: readonly ThemeFrequencySignal[];
  secondaryDrivers: readonly ThemeFrequencySignal[];
  additionalDriversOpen: boolean;
  onToggleAdditionalDrivers: () => void;
}) {
  const visibleDrivers = additionalDriversOpen ? [...primaryDrivers, ...secondaryDrivers] : primaryDrivers;

  return (
    <section className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.62)] p-4">
      <div className="grid gap-4 xl:grid-cols-[minmax(0,0.72fr)_minmax(360px,0.28fr)]">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-full border border-[color:var(--line-soft)] px-3 py-1 text-xs font-semibold text-[color:var(--ink-faint)]">
              {profile.context.window_label}
            </span>
            <StatusBadge severity={signal.severity} />
            <span className="rounded-full border border-[color:var(--line-soft)] px-3 py-1 text-xs font-semibold" style={{ color: trend.color }}>
              {trend.label}
            </span>
          </div>
          <h1 className="mt-3 text-2xl font-semibold text-[color:var(--ink)]" style={{ letterSpacing: 0 }}>
            {profile.label}
          </h1>
          <p className="mt-2 max-w-4xl text-sm text-[color:var(--ink-soft)]">{profile.oneLineSummary}</p>
          <p className="mt-2 max-w-4xl text-xs leading-5 text-[color:var(--ink-faint)]">{profile.narrative}</p>
        </div>

        <div className="rounded-lg border border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.42)] p-3">
          <div className="grid grid-cols-[minmax(0,1fr)_78px_70px_88px] gap-2 text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">
            <span>Drivers</span>
            <span>Mentions</span>
            <span>Spike</span>
            <span>Contribution</span>
          </div>
          <div className="mt-2 divide-y divide-[color:var(--line-soft)]">
            {visibleDrivers.map((driver) => (
              <div key={driver.normalized_theme} className="grid grid-cols-[minmax(0,1fr)_78px_70px_88px] gap-2 py-2 text-xs">
                <span className="truncate font-semibold text-[color:var(--ink)]">{formatTheme(driver.normalized_theme)}</span>
                <span className="tabular-nums text-[color:var(--ink-faint)]">{driver.current_mentions}</span>
                <span className="font-semibold tabular-nums text-[color:var(--accent)]">{formatPct(driver.spike_pct)}</span>
                <span className="font-semibold tabular-nums text-[color:var(--ink)]">{formatContribution(driver.contribution_pct)}</span>
              </div>
            ))}
          </div>
          {secondaryDrivers.length > 0 ? (
            <button
              type="button"
              onClick={onToggleAdditionalDrivers}
              aria-expanded={additionalDriversOpen}
              className="mt-2 min-h-8 rounded-lg border border-[color:var(--line-soft)] px-3 text-xs font-semibold text-[color:var(--accent)] hover:border-[color:var(--line-strong)]"
            >
              {additionalDriversOpen ? "Hide Additional Drivers" : `Show Additional Drivers (${secondaryDrivers.length})`}
            </button>
          ) : null}
        </div>
      </div>
    </section>
  );
}

function EvidenceListSection({ articles }: { articles: readonly EvidenceArticle[] }) {
  return (
    <section className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.58)] p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Evidence</p>
          <h2 className="mt-1 text-xl font-semibold text-[color:var(--ink)]" style={{ letterSpacing: 0 }}>
            Supporting articles
          </h2>
        </div>
        <span className="rounded-full border border-[color:var(--line-soft)] px-3 py-1 text-xs text-[color:var(--ink-faint)]">{articles.length} articles shown</span>
      </div>
      <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {articles.map((article) => (
          <EvidenceItem key={article.id} article={article} />
        ))}
      </div>
    </section>
  );
}

function SignalCompositionPanel({
  drivers,
  profile
}: {
  drivers: readonly ThemeFrequencySignal[];
  profile: IntelligenceProfile;
}) {
  return (
    <section className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.58)] p-4">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Signal Composition</p>
          <h2 className="mt-1 text-xl font-semibold text-[color:var(--ink)]" style={{ letterSpacing: 0 }}>
            Ranked contribution by theme
          </h2>
        </div>
        <span className="rounded-full border border-[color:var(--line-soft)] px-3 py-1 text-xs text-[color:var(--ink-faint)]">
          All {drivers.length} contributing themes shown
        </span>
      </div>

      <div className="mt-4 space-y-3">
        {drivers.map((driver) => {
          const evidence = getThemeEvidence(driver.normalized_theme, profile);
          return (
            <div key={driver.normalized_theme}>
              <div className="flex items-center justify-between gap-3 text-xs">
                <span className="font-semibold text-[color:var(--ink)]">{formatTheme(driver.normalized_theme)}</span>
                <span className="font-semibold tabular-nums text-[color:var(--ink)]">{formatContribution(driver.contribution_pct)}</span>
              </div>
              <div className="mt-1 h-2 overflow-hidden rounded-full bg-[color:rgba(148,163,184,0.14)]">
                <div className="h-full rounded-full bg-[color:var(--accent)]" style={{ width: `${Math.max(4, Math.min(100, driver.contribution_pct))}%` }} />
              </div>
              {evidence.articles[0] ? (
                <p className="mt-1 text-[11px] text-[color:var(--ink-faint)]">
                  Driven by {evidence.articles[0].headline}
                </p>
              ) : null}
            </div>
          );
        })}
      </div>

      <div className="mt-4 rounded-lg border border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.32)] p-3">
        <p className="text-xs font-semibold text-[color:var(--ink)]">Showing top drivers based on:</p>
        <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-[color:var(--ink-faint)]">
          <span className="rounded-full border border-[color:var(--line-soft)] px-2 py-0.5">mention volume</span>
          <span className="rounded-full border border-[color:var(--line-soft)] px-2 py-0.5">spike vs baseline</span>
          <span className="rounded-full border border-[color:var(--line-soft)] px-2 py-0.5">contribution to signal</span>
        </div>
      </div>

      <div className="mt-4">
        <h3 className="text-sm font-semibold text-[color:var(--ink)]">Full Signal Composition</h3>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full min-w-[560px] border-separate border-spacing-y-2 text-left text-xs">
            <thead className="text-[10px] uppercase text-[color:var(--ink-faint)]">
              <tr>
                <th className="px-3 font-semibold">Theme</th>
                <th className="px-3 font-semibold">Mentions</th>
                <th className="px-3 font-semibold">Spike</th>
                <th className="px-3 font-semibold">Contribution</th>
                <th className="px-3 font-semibold">Evidence</th>
              </tr>
            </thead>
            <tbody>
              {drivers.map((driver) => {
                const evidence = getThemeEvidence(driver.normalized_theme, profile);
                return (
                  <tr key={driver.normalized_theme} className="bg-[color:rgba(6,15,24,0.38)] text-[color:var(--ink)]">
                    <td className="rounded-l-lg border-y border-l border-[color:var(--line-soft)] px-3 py-2 font-semibold">{formatTheme(driver.normalized_theme)}</td>
                    <td className="border-y border-[color:var(--line-soft)] px-3 py-2 tabular-nums">{driver.current_mentions}</td>
                    <td className="border-y border-[color:var(--line-soft)] px-3 py-2 tabular-nums text-[color:var(--accent)]">{formatPct(driver.spike_pct)}</td>
                    <td className="border-y border-[color:var(--line-soft)] px-3 py-2 tabular-nums">{formatContribution(driver.contribution_pct)}</td>
                    <td className="rounded-r-lg border-y border-r border-[color:var(--line-soft)] px-3 py-2 text-[color:var(--ink-faint)]">
                      {evidence.clusters[0]?.title ?? evidence.articles[0]?.source ?? "Pending"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

export function ThemeIntelligenceLab() {
  const [profileId, setProfileId] = useState<string>(PROFILES[0].id);
  const [activeTab, setActiveTab] = useState<EvidenceTab>("clusters");
  const [selectedClusterId, setSelectedClusterId] = useState<string>("all");
  const [articleSort, setArticleSort] = useState<ArticleSort>("impact");
  const [additionalDriversOpen, setAdditionalDriversOpen] = useState(true);
  const [selectedTheme, setSelectedTheme] = useState<NormalizedTheme | null>(null);
  const profile = PROFILES.find((item) => item.id === profileId) ?? PROFILES[0];
  const signal = useMemo(() => scoreThemeArticle({ raw_themes: profile.rawThemes, context: profile.context }), [profile]);
  const trend = TREND_STYLE[signal.trend.direction];
  const evidenceArticles = useMemo(() => expandEvidenceArticles(profile, 30), [profile]);
  const selectedArticles = selectedClusterId === "all" ? evidenceArticles : evidenceArticles.filter((article) => article.clusterId === selectedClusterId);
  const sortedArticles = sortArticles(selectedArticles, articleSort);
  const selectedCluster = profile.clusters.find((cluster) => cluster.id === selectedClusterId);
  const selectedVolume = selectedCluster?.volume ?? profile.coverage.totalArticles;
  const additionalSelectedArticles = Math.max(0, selectedVolume - selectedArticles.length);
  const primaryDrivers = signal.primary_drivers;
  const secondaryDrivers = signal.secondary_drivers;
  const backgroundCount = signal.background_context.length;
  const selectedThemeForDisplay = selectedTheme ?? signal.primary_driver?.normalized_theme ?? null;
  const selectedThemeDriver = selectedThemeForDisplay ? signal.frequency_signals.find((driver) => driver.normalized_theme === selectedThemeForDisplay) : undefined;
  const handleThemeSelect = (theme: NormalizedTheme) => {
    setSelectedTheme(theme);
    setProfileId(getProfileIdForTheme(theme));
    setSelectedClusterId("all");
    setActiveTab("clusters");
    setAdditionalDriversOpen(true);
  };

  return (
    <div className="space-y-6">
      <SystemOverview />

      <section>
        <div className="mb-3 flex items-center justify-between gap-3">
          <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Highest Alerts</p>
          <span className="text-xs text-[color:var(--ink-faint)]">Click an alert or choose a theme below</span>
        </div>
        <div className="grid gap-3 lg:grid-cols-4">
          {PROFILES.map((item) => (
            <SignalCard
              key={item.id}
              profile={item}
              active={item.id === profileId}
              onClick={() => {
                setProfileId(item.id);
                setSelectedTheme(null);
                setSelectedClusterId("all");
                setActiveTab("clusters");
                setAdditionalDriversOpen(true);
              }}
            />
          ))}
        </div>
      </section>

      <AllThemesStrip
        selectedTheme={selectedThemeForDisplay}
        activeThemes={signal.normalized_theme_list}
        onThemeSelect={handleThemeSelect}
      />

      {selectedThemeForDisplay ? <ThemeDetailPanel theme={selectedThemeForDisplay} profile={profile} driver={selectedThemeDriver} /> : null}

      <CompactSignalPanel
        profile={profile}
        signal={signal}
        trend={trend}
        primaryDrivers={primaryDrivers}
        secondaryDrivers={secondaryDrivers}
        additionalDriversOpen={additionalDriversOpen}
        onToggleAdditionalDrivers={() => setAdditionalDriversOpen((open) => !open)}
      />

      <EvidenceListSection articles={evidenceArticles} />

      <SignalCompositionPanel
        drivers={signal.frequency_signals}
        profile={profile}
      />

      <section className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.58)] p-4">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">Signal Detail</p>
            <h2 className="mt-1 text-xl font-semibold text-[color:var(--ink)]" style={{ letterSpacing: 0 }}>
              Clusters, articles, and impact
            </h2>
          </div>
          <div className="flex flex-wrap gap-2 text-xs text-[color:var(--ink-faint)]">
            <span
              title={`Coverage diversity: ${profile.coverage.diversity}. Dimensions: ${profile.coverage.dimensions.join(", ")}.`}
              className="rounded-full border border-[color:var(--line-soft)] px-3 py-1"
            >
              {profile.coverage.totalArticles} articles
            </span>
            <span className="rounded-full border border-[color:var(--line-soft)] px-3 py-1">{profile.coverage.sourceCount} sources</span>
            <span className="rounded-full border border-[color:var(--line-soft)] px-3 py-1">{profile.coverage.regionCount} regions</span>
            <span className="rounded-full border border-[color:var(--line-soft)] px-3 py-1 text-[color:var(--accent)]">{formatPct(profile.coverage.changePct)}</span>
          </div>
        </div>

        <div className="mt-4 flex flex-wrap gap-2 border-b border-[color:var(--line-soft)] pb-3" role="tablist" aria-label="Evidence detail views">
          {EVIDENCE_TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={activeTab === tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`min-h-9 rounded-lg border px-3 text-sm font-semibold transition-colors ${
                activeTab === tab.id
                  ? "border-[color:var(--line-strong)] bg-[color:rgba(79,213,255,0.13)] text-[color:var(--ink)]"
                  : "border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.35)] text-[color:var(--ink-faint)] hover:border-[color:var(--line-strong)]"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {activeTab === "clusters" ? (
          <div className="mt-4 grid gap-3 lg:grid-cols-2">
            {profile.clusters.map((cluster, index) => (
              <article key={cluster.id} className="rounded-lg border border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.45)] p-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <p className="text-[10px] font-semibold uppercase text-[color:var(--ink-faint)]">{cluster.importance}</p>
                    <h3 className="mt-1 text-sm font-semibold text-[color:var(--ink)]">
                      {index + 1}. {cluster.title}
                    </h3>
                  </div>
                  <span className="rounded-full border border-[color:rgba(242,171,67,0.24)] bg-[color:rgba(242,171,67,0.09)] px-2 py-0.5 text-xs font-semibold text-[color:var(--warn)]">
                    {formatPct(cluster.changePct)}
                  </span>
                </div>
                <p className="mt-2 text-xs leading-5 text-[color:var(--ink-soft)]">{cluster.summary}</p>
                <div className="mt-3 flex flex-wrap gap-2 text-[11px] text-[color:var(--ink-faint)]">
                  <span>{cluster.volume} articles</span>
                  <span>First seen {cluster.firstSeen}</span>
                  <span>Peak {cluster.peakAcceleration}</span>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    setSelectedClusterId(cluster.id);
                    setActiveTab("articles");
                  }}
                  className="mt-3 min-h-9 rounded-lg border border-[color:var(--line-soft)] px-3 text-xs font-semibold text-[color:var(--accent)] hover:border-[color:var(--line-strong)]"
                >
                  View Articles
                </button>
              </article>
            ))}
          </div>
        ) : null}

        {activeTab === "articles" ? (
          <div className="mt-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => setSelectedClusterId("all")}
                  aria-pressed={selectedClusterId === "all"}
                  className={`min-h-9 rounded-lg border px-3 text-xs font-semibold ${
                    selectedClusterId === "all"
                      ? "border-[color:var(--line-strong)] bg-[color:rgba(79,213,255,0.13)] text-[color:var(--ink)]"
                      : "border-[color:var(--line-soft)] text-[color:var(--ink-faint)]"
                  }`}
                >
                  All Coverage
                </button>
                {profile.clusters.map((cluster) => (
                  <button
                    key={cluster.id}
                    type="button"
                    onClick={() => setSelectedClusterId(cluster.id)}
                    aria-pressed={selectedClusterId === cluster.id}
                    className={`min-h-9 rounded-lg border px-3 text-xs font-semibold ${
                      selectedClusterId === cluster.id
                        ? "border-[color:var(--line-strong)] bg-[color:rgba(79,213,255,0.13)] text-[color:var(--ink)]"
                        : "border-[color:var(--line-soft)] text-[color:var(--ink-faint)]"
                    }`}
                  >
                    {cluster.title}
                  </button>
                ))}
              </div>
              <select aria-label="Sort articles" className="form-control min-h-10 px-3 text-sm" value={articleSort} onChange={(event) => setArticleSort(event.target.value as ArticleSort)}>
                <option value="impact">Sort by impact</option>
                <option value="credibility">Sort by source credibility</option>
                <option value="recency">Sort by recency</option>
              </select>
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              {profile.sourceDistribution.map((item) => (
                <span key={item.source} className="rounded-full border border-[color:var(--line-soft)] px-3 py-1 text-xs text-[color:var(--ink-faint)]">
                  {item.source} ({item.count})
                </span>
              ))}
            </div>
            <p className="mt-3 text-xs text-[color:var(--ink-faint)]">
              Representative sample shown. {additionalSelectedArticles} more articles reinforce this {selectedCluster ? "cluster" : "signal"}.
            </p>
            <div className="mt-3 max-h-[520px] space-y-3 overflow-y-auto pr-1">
              {sortedArticles.map((article) => (
                <EvidenceItem key={article.id} article={article} />
              ))}
            </div>
          </div>
        ) : null}

        {activeTab === "impact" ? (
          <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {signal.market_impacts.slice(0, 6).map((impact) => {
              const style = DIRECTION_STYLE[impact.direction];
              const linkedEvidence = profile.evidence.find((article) => article.relatedThemes.some((theme) => impact.themes.includes(theme)));
              return (
                <div key={`${impact.asset}-${impact.direction}`} className="rounded-lg border border-[color:var(--line-soft)] bg-[color:rgba(6,15,24,0.45)] p-3">
                  <div className="flex items-center justify-between gap-3">
                    <span className="font-mono text-base font-semibold text-[color:var(--ink)]">{impact.asset}</span>
                    <span className="rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase" style={{ color: style.color, background: style.background }}>
                      {style.label}
                    </span>
                  </div>
                  <p className="mt-2 text-xs text-[color:var(--ink-faint)]">{impact.rationale}</p>
                  {linkedEvidence ? (
                    <p className="mt-3 border-t border-[color:var(--line-soft)] pt-2 text-xs text-[color:var(--accent)]">
                      Evidence: {linkedEvidence.source} - {linkedEvidence.explanation.toLowerCase()}
                    </p>
                  ) : null}
                </div>
              );
            })}
          </div>
        ) : null}
      </section>
    </div>
  );
}

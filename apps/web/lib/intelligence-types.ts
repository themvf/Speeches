import type {
  NormalizedTheme,
  ThemeContextInput,
  ThemeSeverity,
  ThemeSignal
} from "@/lib/theme-intelligence";

export type IntelligenceEvidenceArticle = {
  id: string;
  headline: string;
  url?: string;
  source: string;
  timestamp: string;
  excerpt: string;
  explanation: string;
  relatedThemes: NormalizedTheme[];
  clusterId: string;
  credibility: number;
  impact: number;
};

export type IntelligenceEvidenceCluster = {
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

export type IntelligenceSeedProfile = {
  id: string;
  label: string;
  rawThemes: string;
  context: ThemeContextInput;
  oneLineSummary: string;
  narrative: string;
  whatChanged: string[];
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
  evidence: IntelligenceEvidenceArticle[];
  clusters: IntelligenceEvidenceCluster[];
};

export type IntelligenceProfile = IntelligenceSeedProfile & {
  signal: ThemeSignal;
};

export type SystemTrend = {
  label: string;
  changePct: number;
  direction: "up" | "down";
};

export type NarrativeRank = {
  label: string;
  severity: ThemeSeverity;
  summary: string;
};

export type IntelligenceSignalsData = {
  generatedAt: string;
  source: "seed" | "gdelt-doc";
  systemTrends: SystemTrend[];
  whatChanged: string[];
  narrativeLeaderboard: NarrativeRank[];
  profiles: IntelligenceProfile[];
};

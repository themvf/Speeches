import {
  INTELLIGENCE_PROFILES,
  NARRATIVE_LEADERBOARD,
  SYSTEM_TRENDS,
  WHAT_CHANGED
} from "@/lib/intelligence-seed";
import { sourceSearchUrl } from "@/lib/intelligence-links";
import type {
  IntelligenceEvidenceArticle,
  IntelligenceSeedProfile,
  IntelligenceSignalsData
} from "@/lib/intelligence-types";
import { scoreThemeArticle } from "@/lib/theme-intelligence";

function withArticleUrls(profile: IntelligenceSeedProfile): IntelligenceSeedProfile {
  return {
    ...profile,
    evidence: profile.evidence.map((article) => ({
      ...article,
      url: article.url ?? sourceSearchUrl(article)
    }))
  };
}

export function buildIntelligenceSignalsData(): IntelligenceSignalsData {
  return {
    generatedAt: new Date().toISOString(),
    source: "seed",
    systemTrends: [...SYSTEM_TRENDS],
    whatChanged: [...WHAT_CHANGED],
    narrativeLeaderboard: [...NARRATIVE_LEADERBOARD],
    profiles: INTELLIGENCE_PROFILES.map(withArticleUrls).map((profile) => ({
      ...profile,
      signal: scoreThemeArticle({
        id: profile.id,
        title: profile.label,
        raw_themes: profile.rawThemes,
        context: profile.context
      })
    }))
  };
}

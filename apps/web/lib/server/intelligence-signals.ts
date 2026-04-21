import {
  INTELLIGENCE_PROFILES,
  NARRATIVE_LEADERBOARD,
  SYSTEM_TRENDS,
  WHAT_CHANGED
} from "@/lib/intelligence-seed";
import type { IntelligenceSignalsData } from "@/lib/intelligence-types";
import { scoreThemeArticle } from "@/lib/theme-intelligence";

export function buildIntelligenceSignalsData(): IntelligenceSignalsData {
  return {
    generatedAt: new Date().toISOString(),
    source: "seed",
    systemTrends: [...SYSTEM_TRENDS],
    whatChanged: [...WHAT_CHANGED],
    narrativeLeaderboard: [...NARRATIVE_LEADERBOARD],
    profiles: INTELLIGENCE_PROFILES.map((profile) => ({
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

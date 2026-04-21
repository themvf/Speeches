import { createHash } from "node:crypto";

import type { IntelligenceEvidenceArticle, IntelligenceProfile } from "../intelligence-types";
import type { NormalizedTheme } from "../theme-intelligence";

const GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc";
const GDELT_DOC_TIMEOUT_MS = 20_000;
const GDELT_DOC_CACHE_TTL_MS = 5 * 60 * 1_000;
const GDELT_DOC_MAX_RECORDS = 30;
const GDELT_DOC_TIMESPAN = "24h";

type GdeltDocArticle = {
  url?: unknown;
  url_mobile?: unknown;
  title?: unknown;
  seendate?: unknown;
  socialimage?: unknown;
  domain?: unknown;
  language?: unknown;
  sourcecountry?: unknown;
};

type CacheEntry = {
  loadedAt: number;
  articles: IntelligenceEvidenceArticle[];
};

const gdeltEvidenceCache = new Map<string, CacheEntry>();

const THEME_QUERY_TERMS: Readonly<Record<NormalizedTheme, readonly string[]>> = {
  INFLATION: ["inflation", "consumer prices", "CPI"],
  INTEREST_RATES: ["interest rates", "rate hike", "rate cut", "bond yields"],
  CENTRAL_BANK: ["central bank", "Federal Reserve", "ECB", "monetary policy"],
  ECONOMIC_GROWTH: ["GDP", "economic growth", "recession", "slowdown"],
  LABOR_MARKET: ["jobs", "wages", "unemployment", "labor market"],
  BANKING: ["banking stress", "regional banks", "bank failure", "bank run"],
  CREDIT_MARKETS: ["credit markets", "debt", "default", "bond market", "credit spreads"],
  LIQUIDITY: ["liquidity", "funding stress", "cash flow"],
  FINANCIAL_MARKETS: ["stock market", "equities", "financial markets"],
  REGULATION: ["regulation", "SEC", "compliance", "policy"],
  GEOPOLITICS: ["geopolitical risk", "diplomacy", "foreign policy"],
  CONFLICT: ["war", "military", "attack", "defense"],
  TRADE: ["trade", "exports", "imports", "tariffs"],
  SANCTIONS: ["sanctions", "embargo", "restrictions"],
  ENERGY: ["oil", "natural gas", "energy supply", "OPEC"],
  SUPPLY_CHAIN: ["supply chain", "shipping", "logistics", "freight"],
  COMMODITIES: ["gold", "metals", "agriculture", "raw materials", "commodities"],
  TECHNOLOGY: ["technology", "semiconductor", "software"],
  CRYPTO: ["cryptocurrency", "bitcoin", "blockchain"],
  CORPORATE_ACTIVITY: ["earnings", "mergers", "acquisitions", "layoffs"],
  AI: ["artificial intelligence", "generative AI", "machine learning", "AI infrastructure", "LLM"]
};

const PROFILE_QUERY_PLANS: Readonly<Record<string, readonly string[]>> = {
  macro: ["oil inflation", "energy inflation", "central bank inflation"],
  bank: ["bank credit", "banking liquidity", "credit spreads"],
  geopolitical: ["trade sanctions", "shipping sanctions", "supply chain disruption"],
  modern: ["artificial intelligence semiconductor", "AI chip demand", "generative AI investment"]
};

function normalizeString(value: unknown): string {
  return String(value ?? "").trim();
}

function unique<T>(items: readonly T[]): T[] {
  return Array.from(new Set(items));
}

function quoteTerm(term: string): string {
  const normalized = term.trim();
  if (!normalized) return "";
  return /\s/.test(normalized) ? `"${normalized.replace(/"/g, "")}"` : normalized;
}

function isHttpUrl(value: string): boolean {
  try {
    const parsed = new URL(value);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

function articleId(profileId: string, url: string): string {
  const hash = createHash("sha1").update(url).digest("hex").slice(0, 12);
  return `gdelt-${profileId}-${hash}`;
}

function parseGdeltSeenDate(value: string): Date | null {
  const match = value.match(/^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z$/);
  if (!match) return null;
  const [, year, month, day, hour, minute, second] = match;
  const date = new Date(Date.UTC(Number(year), Number(month) - 1, Number(day), Number(hour), Number(minute), Number(second)));
  return Number.isNaN(date.getTime()) ? null : date;
}

function formatGdeltSeenDate(value: string): string {
  const date = parseGdeltSeenDate(value);
  if (!date) return value || "GDELT";
  return date.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "UTC",
    timeZoneName: "short"
  });
}

function sourceFromArticle(article: GdeltDocArticle): string {
  const domain = normalizeString(article.domain);
  if (domain) return domain;

  const url = normalizeString(article.url);
  if (!url) return "GDELT";

  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return "GDELT";
  }
}

function themesForArticle(article: GdeltDocArticle, profile: IntelligenceProfile): NormalizedTheme[] {
  const title = normalizeString(article.title).toLowerCase();
  const matched = profile.signal.normalized_theme_list.filter((theme) => {
    const terms = THEME_QUERY_TERMS[theme] ?? [];
    return terms.some((term) => title.includes(term.toLowerCase()));
  });

  if (matched.length > 0) {
    return matched;
  }

  return profile.signal.primary_drivers.map((driver) => driver.normalized_theme).slice(0, 3);
}

export function buildGdeltDocQuery(profile: IntelligenceProfile): string {
  const profileQuery = PROFILE_QUERY_PLANS[profile.id]?.[0];
  if (profileQuery) {
    return profileQuery;
  }

  const rankedThemes = unique([
    ...profile.signal.primary_drivers.map((driver) => driver.normalized_theme),
    ...profile.signal.secondary_drivers.map((driver) => driver.normalized_theme),
    ...profile.signal.normalized_theme_list
  ]).slice(0, 5);

  const terms = unique(rankedThemes.flatMap((theme) => THEME_QUERY_TERMS[theme] ?? [theme])).slice(0, 16);
  const queryTerms = terms.map(quoteTerm).filter(Boolean);
  return queryTerms.slice(0, 6).join(" ");
}

export function buildGdeltDocQueries(profile: IntelligenceProfile): string[] {
  const plannedQueries = PROFILE_QUERY_PLANS[profile.id];
  if (plannedQueries?.length) {
    return [...plannedQueries];
  }

  return [buildGdeltDocQuery(profile)];
}

export function buildGdeltDocUrl(query: string): string {
  const url = new URL(GDELT_DOC_ENDPOINT);
  url.searchParams.set("query", query);
  url.searchParams.set("mode", "artlist");
  url.searchParams.set("format", "json");
  url.searchParams.set("maxrecords", String(GDELT_DOC_MAX_RECORDS));
  url.searchParams.set("timespan", GDELT_DOC_TIMESPAN);
  url.searchParams.set("sort", "datedesc");
  return url.toString();
}

export function mapGdeltDocArticlesToEvidence(
  profile: IntelligenceProfile,
  articles: readonly GdeltDocArticle[]
): IntelligenceEvidenceArticle[] {
  const seenUrls = new Set<string>();
  const evidence: IntelligenceEvidenceArticle[] = [];

  for (const [index, article] of articles.entries()) {
    const url = normalizeString(article.url);
    const title = normalizeString(article.title);
    if (!url || !title || !isHttpUrl(url) || seenUrls.has(url)) {
      continue;
    }

    seenUrls.add(url);

    const source = sourceFromArticle(article);
    const language = normalizeString(article.language);
    const country = normalizeString(article.sourcecountry);
    const relatedThemes = themesForArticle(article, profile);
    const primaryTheme = relatedThemes[0] ?? profile.signal.primary_driver?.normalized_theme;

    evidence.push({
      id: articleId(profile.id, url),
      headline: title,
      url,
      source,
      timestamp: formatGdeltSeenDate(normalizeString(article.seendate)),
      excerpt: [
        "Live GDELT DOC 2.0 article match.",
        language ? `Language: ${language}.` : "",
        country ? `Source country: ${country}.` : ""
      ].filter(Boolean).join(" "),
      explanation: primaryTheme
        ? `Matched live ${formatThemeForSentence(primaryTheme)} coverage for this signal`
        : "Matched live GDELT coverage for this signal",
      relatedThemes,
      clusterId: profile.clusters[0]?.id ?? "gdelt-doc",
      credibility: Math.max(60, 92 - index),
      impact: Math.max(55, 90 - index)
    });

    if (evidence.length >= GDELT_DOC_MAX_RECORDS) {
      break;
    }
  }

  return evidence;
}

function formatThemeForSentence(theme: NormalizedTheme): string {
  return theme.toLowerCase().replace(/_/g, " ");
}

export async function fetchGdeltEvidenceForProfile(profile: IntelligenceProfile): Promise<IntelligenceEvidenceArticle[]> {
  for (const query of buildGdeltDocQueries(profile)) {
    const cached = gdeltEvidenceCache.get(query);
    if (cached && Date.now() - cached.loadedAt < GDELT_DOC_CACHE_TTL_MS) {
      if (cached.articles.length > 0) {
        return cached.articles;
      }
      continue;
    }

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), GDELT_DOC_TIMEOUT_MS);

    try {
      const response = await fetch(buildGdeltDocUrl(query), {
        signal: controller.signal,
        headers: {
          "accept": "application/json",
          "user-agent": "PolicyResearchHub/1.0 IntelBeta GDELT evidence retrieval"
        },
        next: { revalidate: 300 }
      });

      if (!response.ok) {
        gdeltEvidenceCache.set(query, { loadedAt: Date.now(), articles: [] });
        continue;
      }

      const payload = (await response.json()) as { articles?: GdeltDocArticle[] };
      const evidence = mapGdeltDocArticlesToEvidence(profile, Array.isArray(payload.articles) ? payload.articles : []);
      gdeltEvidenceCache.set(query, { loadedAt: Date.now(), articles: evidence });

      if (evidence.length > 0) {
        return evidence;
      }
    } catch {
      gdeltEvidenceCache.set(query, { loadedAt: Date.now(), articles: [] });
    } finally {
      clearTimeout(timeout);
    }
  }

  return [];
}

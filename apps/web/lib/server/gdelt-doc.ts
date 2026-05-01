import { createHash } from "node:crypto";

import type { IntelligenceEvidenceArticle, IntelligenceProfile } from "../intelligence-types";
import {
  focusAreasForProductCategory,
  type NormalizedTheme,
  type ProductCategory,
  type ProductFocusArea
} from "../theme-intelligence.ts";

const GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc";
const GDELT_DOC_TIMEOUT_MS = 20_000;
const GDELT_DOC_CACHE_TTL_MS = 5 * 60 * 1_000;
const GDELT_DOC_MAX_RECORDS = 30;
const GDELT_DOC_MAX_RECORDS_PER_FOCUS = 6;
const GDELT_DOC_CATEGORY_MAX_RECORDS_PER_QUERY = 8;
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

type CategoryDocQueryPlan = {
  focusArea: ProductFocusArea;
  query: string;
  terms: readonly string[];
};

const gdeltEvidenceCache = new Map<string, CacheEntry>();
const gdeltCategoryEvidenceCache = new Map<string, CacheEntry>();

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

export const CATEGORY_DOC_QUERY_TERMS: Readonly<Partial<Record<ProductCategory, Readonly<Record<string, readonly string[]>>>>> = {
  AML: {
    aml_sanctions: ["OFAC", "sanctions"],
    aml_bsa: ["money laundering", "anti-money laundering", "FinCEN"],
    aml_kyc_ownership: ["beneficial ownership", "KYC", "customer identification"],
    aml_illicit_finance: ["suspicious activity", "terrorist financing", "illicit finance"]
  },
  CAPITAL_FORMATION: {
    capital_public_offerings: ["IPO", "initial public offering", "public offering"],
    capital_private_capital: ["private credit", "private equity", "venture capital"],
    capital_debt_financing: ["bond issuance", "debt offering", "credit facility"],
    capital_strategic_transactions: ["SPAC", "merger", "acquisition"],
    capital_access_policy: ["crowdfunding", "Reg A", "exempt offering"]
  }
};

export const CATEGORY_PATTERN_ALIASES: Readonly<Record<string, readonly string[]>> = {
  SANCTIONS: ["SANCTIONS", "SANCTION", "SANCTIONED"],
  OFAC: ["OFAC"],
  AML: ["AML", "ANTI_MONEY_LAUNDERING"],
  BSA: ["BSA", "BANK_SECRECY_ACT"],
  FINCEN: ["FINCEN", "FINANCIAL_CRIMES_ENFORCEMENT_NETWORK"],
  MONEY_LAUNDERING: ["MONEY_LAUNDERING", "ANTI_MONEY_LAUNDERING"],
  ANTI_MONEY_LAUNDERING: ["ANTI_MONEY_LAUNDERING", "AML"],
  KYC: ["KYC", "KNOW_YOUR_CUSTOMER"],
  CIP: ["CIP", "CUSTOMER_IDENTIFICATION_PROGRAM"],
  CUSTOMER_IDENTIFICATION: ["CUSTOMER_IDENTIFICATION", "CUSTOMER_IDENTIFICATION_PROGRAM"],
  BENEFICIAL_OWNERSHIP: ["BENEFICIAL_OWNERSHIP", "BENEFICIAL_OWNER"],
  ILLICIT_FINANCE: ["ILLICIT_FINANCE", "ILLICIT_FINANCING"],
  SUSPICIOUS_ACTIVITY: ["SUSPICIOUS_ACTIVITY", "SUSPICIOUS_ACTIVITY_REPORT"],
  SAR: ["SAR", "SUSPICIOUS_ACTIVITY_REPORT"],
  TERRORIST_FINANCING: ["TERRORIST_FINANCING", "TERRORISM_FINANCING"],
  IPO: ["IPO", "INITIAL_PUBLIC_OFFERING", "GO_PUBLIC", "PUBLIC_DEBUT"],
  INITIAL_PUBLIC_OFFERING: ["INITIAL_PUBLIC_OFFERING", "IPO", "GO_PUBLIC", "PUBLIC_DEBUT"],
  SECURITIES_OFFERING: ["SECURITIES_OFFERING", "SECURITIES_ISSUANCE"],
  EQUITY_OFFERING: ["EQUITY_OFFERING", "STOCK_OFFERING", "SHARE_OFFERING"],
  PUBLIC_OFFERING: ["PUBLIC_OFFERING", "GO_PUBLIC"],
  SECONDARY_OFFERING: ["SECONDARY_OFFERING", "FOLLOW_ON_OFFERING"],
  SHARE_SALE: ["SHARE_SALE"],
  STOCK_LISTING: ["STOCK_LISTING", "GO_PUBLIC"],
  PRIVATE_MARKETS: ["PRIVATE_MARKETS", "PRIVATE_MARKET"],
  PRIVATE_EQUITY: ["PRIVATE_EQUITY"],
  PRIVATE_CREDIT: ["PRIVATE_CREDIT", "DIRECT_LENDING"],
  VENTURE_CAPITAL: ["VENTURE_CAPITAL", "VC"],
  FUNDRAISING: ["FUNDRAISING", "FUNDING_ROUND", "RAISES_FUNDS"],
  STARTUP_FUNDING: ["STARTUP_FUNDING", "SEED_FUNDING", "FUNDING_ROUND"],
  PRIVATE_PLACEMENT: ["PRIVATE_PLACEMENT"],
  REG_D: ["REG_D", "REGULATION_D"],
  ACCREDITED_INVESTOR: ["ACCREDITED_INVESTOR", "ACCREDITED_INVESTORS"],
  CAPITAL_RAISE: ["CAPITAL_RAISE", "CAPITAL_RAISING", "RAISES_CAPITAL"],
  DEBT_OFFERING: ["DEBT_OFFERING", "DEBT_ISSUANCE"],
  BOND_ISSUANCE: ["BOND_ISSUANCE", "BOND_OFFERING"],
  CORPORATE_BONDS: ["CORPORATE_BONDS", "CORPORATE_BOND"],
  DEBT_FINANCING: ["DEBT_FINANCING"],
  REFINANCING: ["REFINANCING", "REFINANCE"],
  LEVERAGED_LOAN: ["LEVERAGED_LOAN", "LEVERAGED_LOANS"],
  CREDIT_FACILITY: ["CREDIT_FACILITY", "CREDIT_LINE"],
  HIGH_YIELD: ["HIGH_YIELD", "JUNK_BOND"],
  INVESTMENT_GRADE: ["INVESTMENT_GRADE"],
  "M&A": ["M_AND_A", "MERGER", "ACQUISITION", "ACQUIRE"],
  DEALMAKING: ["DEALMAKING"],
  TAKEOVER_BID: ["TAKEOVER_BID"],
  BUYOUT: ["BUYOUT"],
  SPAC: ["SPAC", "SPECIAL_PURPOSE_ACQUISITION_COMPANY"],
  DE_SPAC: ["DE_SPAC", "DESPAC"],
  MERGER_AGREEMENT: ["MERGER_AGREEMENT"],
  STRATEGIC_TRANSACTION: ["STRATEGIC_TRANSACTION"],
  CAPITAL_FORMATION: ["CAPITAL_FORMATION", "CAPITAL_ACCESS"],
  SMALL_BUSINESS_CAPITAL: ["SMALL_BUSINESS_CAPITAL", "SMALL_BUSINESS_FINANCING"],
  EMERGING_GROWTH_COMPANY: ["EMERGING_GROWTH_COMPANY"],
  CROWDFUNDING: ["CROWDFUNDING"],
  REG_CF: ["REG_CF", "REGULATION_CROWDFUNDING"],
  REG_A: ["REG_A", "REGULATION_A"],
  REGULATION_A: ["REGULATION_A", "REG_A"],
  EXEMPT_OFFERING: ["EXEMPT_OFFERING", "EXEMPT_OFFERINGS"]
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

function buildOrQuery(terms: readonly string[]): string {
  const queryTerms = unique(terms.map(quoteTerm).filter(Boolean));
  if (queryTerms.length === 0) {
    return "";
  }
  return queryTerms.length === 1 ? queryTerms[0] : `(${queryTerms.join(" OR ")})`;
}

function normalizeMatchText(value: string): string {
  return value
    .toUpperCase()
    .replace(/&/g, " AND ")
    .replace(/[^A-Z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function matchTokens(value: string): string[] {
  return normalizeMatchText(value).split("_").filter(Boolean);
}

function containsTokenSequence(haystack: readonly string[], needle: readonly string[]): boolean {
  if (needle.length === 0 || needle.length > haystack.length) {
    return false;
  }

  for (let index = 0; index <= haystack.length - needle.length; index += 1) {
    if (needle.every((token, offset) => haystack[index + offset] === token)) {
      return true;
    }
  }

  return false;
}

function textMatchesCategoryPattern(value: string, pattern: string): boolean {
  const haystack = matchTokens(value);
  const needle = matchTokens(pattern);
  if (needle.length === 0) return false;
  if (needle.length === 1) {
    return haystack.includes(needle[0]);
  }
  return containsTokenSequence(haystack, needle);
}

function aliasesForCategoryPattern(pattern: string): readonly string[] {
  const normalized = normalizeMatchText(pattern);
  return CATEGORY_PATTERN_ALIASES[normalized] ?? [pattern];
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

export function buildGdeltDocCategoryQueries(category: ProductCategory, focusId?: string | null): string[] {
  return categoryDocQueryPlans(category, focusId).map((plan) => plan.query);
}

function categoryDocQueryPlans(category: ProductCategory, focusId?: string | null): CategoryDocQueryPlan[] {
  const focusAreas = focusAreasForProductCategory(category).filter((focusArea) => !focusId || focusArea.id === focusId);
  const queryPlans = CATEGORY_DOC_QUERY_TERMS[category];

  return focusAreas
    .flatMap((focusArea) => {
      const terms = queryPlans?.[focusArea.id] ?? focusArea.raw_patterns;
      return terms.map((term) => ({
        focusArea,
        query: `${quoteTerm(term)} sourcelang:english`,
        terms: [term]
      }));
    });
}

export function buildGdeltDocUrl(query: string, options?: { maxRecords?: number; timespan?: string }): string {
  const url = new URL(GDELT_DOC_ENDPOINT);
  url.searchParams.set("query", query);
  url.searchParams.set("mode", "artlist");
  url.searchParams.set("format", "json");
  url.searchParams.set("maxrecords", String(options?.maxRecords ?? GDELT_DOC_MAX_RECORDS));
  url.searchParams.set("timespan", options?.timespan ?? GDELT_DOC_TIMESPAN);
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

function categoryArticleId(category: ProductCategory, url: string): string {
  const hash = createHash("sha1").update(`${category}:${url}`).digest("hex").slice(0, 12);
  return `gdelt-doc-${category.toLowerCase()}-${hash}`;
}

function matchedFocusTermsForArticle(article: GdeltDocArticle, focusArea: ProductFocusArea): string[] {
  const haystack = [
    normalizeString(article.title),
    normalizeString(article.url),
    normalizeString(article.domain)
  ].join(" ");

  return focusArea.raw_patterns.filter((pattern) => {
    return aliasesForCategoryPattern(pattern).some((alias) => textMatchesCategoryPattern(haystack, alias));
  });
}

export function mapGdeltDocArticlesToProductCategoryEvidence(
  category: ProductCategory,
  articles: readonly GdeltDocArticle[],
  focusAreas: readonly ProductFocusArea[] = focusAreasForProductCategory(category)
): IntelligenceEvidenceArticle[] {
  const evidence: IntelligenceEvidenceArticle[] = [];
  const seenUrls = new Set<string>();

  for (const article of articles) {
    const url = normalizeString(article.url);
    const title = normalizeString(article.title);
    if (!url || !title || !isHttpUrl(url) || seenUrls.has(url)) {
      continue;
    }

    const matchedFocusAreas = focusAreas
      .map((focusArea) => ({ focusArea, matchedTerms: matchedFocusTermsForArticle(article, focusArea) }))
      .filter((item) => item.matchedTerms.length > 0)
      .sort((a, b) => b.matchedTerms.length - a.matchedTerms.length || a.focusArea.label.localeCompare(b.focusArea.label));
    const bestMatch = matchedFocusAreas[0];
    if (!bestMatch) {
      continue;
    }

    seenUrls.add(url);
    const source = sourceFromArticle(article);
    evidence.push({
      id: categoryArticleId(category, url),
      headline: title,
      url,
      source,
      timestamp: formatGdeltSeenDate(normalizeString(article.seendate)),
      excerpt: `Live GDELT DOC 2.0 match on category terms: ${bestMatch.matchedTerms.join(", ")}.`,
      explanation: `Matched ${bestMatch.focusArea.label}: ${bestMatch.matchedTerms.join(", ")}`,
      relatedThemes: [...bestMatch.focusArea.normalized_themes],
      matchedTerms: bestMatch.matchedTerms,
      focusAreaId: bestMatch.focusArea.id,
      focusAreaLabel: bestMatch.focusArea.label,
      clusterId: `${category.toLowerCase()}-${bestMatch.focusArea.id}`,
      credibility: Math.max(60, 92 - evidence.length),
      impact: Math.max(55, 90 - evidence.length)
    });

    if (evidence.length >= GDELT_DOC_MAX_RECORDS) {
      break;
    }
  }

  return evidence;
}

function mapGdeltDocArticlesToKnownFocusAreaEvidence(
  category: ProductCategory,
  articles: readonly GdeltDocArticle[],
  focusArea: ProductFocusArea,
  matchedTerms: readonly string[],
  seenUrls: Set<string>,
  existingCount: number,
  limit: number
): IntelligenceEvidenceArticle[] {
  const evidence: IntelligenceEvidenceArticle[] = [];
  if (limit <= 0) {
    return evidence;
  }

  for (const article of articles) {
    const url = normalizeString(article.url);
    const title = normalizeString(article.title);
    if (!url || !title || !isHttpUrl(url) || seenUrls.has(url)) {
      continue;
    }

    seenUrls.add(url);
    const source = sourceFromArticle(article);
    const index = existingCount + evidence.length;
    evidence.push({
      id: categoryArticleId(category, url),
      headline: title,
      url,
      source,
      timestamp: formatGdeltSeenDate(normalizeString(article.seendate)),
      excerpt: `Live GDELT DOC 2.0 article returned by ${focusArea.label} query.`,
      explanation: `Matched ${focusArea.label}: ${matchedTerms.join(", ")}`,
      relatedThemes: [...focusArea.normalized_themes],
      matchedTerms: [...matchedTerms],
      focusAreaId: focusArea.id,
      focusAreaLabel: focusArea.label,
      clusterId: `${category.toLowerCase()}-${focusArea.id}`,
      credibility: Math.max(60, 92 - index),
      impact: Math.max(55, 90 - index)
    });

    if (existingCount + evidence.length >= GDELT_DOC_MAX_RECORDS || evidence.length >= limit) {
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

async function fetchGdeltDocArticlesForQuery(
  query: string,
  options?: { maxRecords?: number; timespan?: string }
): Promise<GdeltDocArticle[]> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), GDELT_DOC_TIMEOUT_MS);

  try {
    const response = await fetch(buildGdeltDocUrl(query, options), {
      signal: controller.signal,
      headers: {
        "accept": "application/json",
        "user-agent": "PolicyResearchHub/1.0 IntelBeta GDELT category retrieval"
      },
      next: { revalidate: 300 }
    });

    if (!response.ok) {
      return [];
    }

    const payload = (await response.json()) as { articles?: GdeltDocArticle[] };
    return Array.isArray(payload.articles) ? payload.articles : [];
  } catch {
    return [];
  } finally {
    clearTimeout(timeout);
  }
}

export async function fetchGdeltDocEvidenceForProductCategory(
  category: ProductCategory,
  focusId?: string | null
): Promise<IntelligenceEvidenceArticle[]> {
  const cacheKey = `category:${category}:${focusId ?? "all"}`;
  const cached = gdeltCategoryEvidenceCache.get(cacheKey);
  if (cached && Date.now() - cached.loadedAt < GDELT_DOC_CACHE_TTL_MS) {
    return cached.articles;
  }

  const focusAreas = focusAreasForProductCategory(category).filter((focusArea) => !focusId || focusArea.id === focusId);
  if (focusAreas.length === 0) {
    return [];
  }

  const plans = categoryDocQueryPlans(category, focusId);
  const results = await Promise.allSettled(plans.map(async (plan) => ({
    plan,
    articles: await fetchGdeltDocArticlesForQuery(plan.query, { maxRecords: GDELT_DOC_CATEGORY_MAX_RECORDS_PER_QUERY })
  })));
  const seenUrls = new Set<string>();
  const evidence: IntelligenceEvidenceArticle[] = [];
  const focusCounts = new Map<string, number>();

  for (const result of results) {
    if (result.status !== "fulfilled") {
      continue;
    }

    const strictEvidence = mapGdeltDocArticlesToProductCategoryEvidence(category, result.value.articles, [result.value.plan.focusArea])
      .filter((article) => article.url && !seenUrls.has(article.url));
    let focusCount = focusCounts.get(result.value.plan.focusArea.id) ?? 0;
    for (const article of strictEvidence) {
      seenUrls.add(article.url!);
      evidence.push(article);
      focusCounts.set(result.value.plan.focusArea.id, ++focusCount);
      if (evidence.length >= GDELT_DOC_MAX_RECORDS) {
        break;
      }
      if (focusCount >= GDELT_DOC_MAX_RECORDS_PER_FOCUS) {
        break;
      }
    }
    if (evidence.length >= GDELT_DOC_MAX_RECORDS) {
      break;
    }
  }

  gdeltCategoryEvidenceCache.set(cacheKey, { loadedAt: Date.now(), articles: evidence });
  return evidence;
}

import { createHash } from "node:crypto";
import { inflateRawSync } from "node:zlib";

import type { IntelligenceEvidenceArticle, IntelligenceProfile } from "../intelligence-types";
import {
  focusAreasForProductCategory,
  scoreThemeArticle,
  type NormalizedTheme,
  type ProductCategory,
  type ProductFocusArea
} from "../theme-intelligence.ts";

const GDELT_GKG_UPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt";
const GDELT_GKG_TIMEOUT_MS = 8_000;
const GDELT_GKG_CACHE_TTL_MS = 5 * 60 * 1_000;
const GDELT_GKG_MAX_RECORDS = 30;
const GDELT_GKG_ARCHIVE_COUNT = 6;
const PROFILE_EVIDENCE_TERMS: Readonly<Record<string, readonly string[]>> = {
  macro: ["inflation", "oil", "energy", "central-bank", "central bank", "rates", "price", "prices", "cpi"],
  bank: ["bank", "banking", "credit", "debt", "liquidity", "funding", "default", "bonds"],
  geopolitical: ["trade", "sanction", "shipping", "supply", "tariff", "war", "attack", "military", "conflict"],
  modern: ["ai", "artificial-intelligence", "artificial intelligence", "semiconductor", "chip", "crypto", "bitcoin", "blockchain", "software", "earnings", "layoff"]
};

const PROFILE_TERM_THEMES: Readonly<Record<string, readonly { terms: readonly string[]; theme: NormalizedTheme }[]>> = {
  macro: [
    { terms: ["inflation", "cpi", "price", "prices"], theme: "INFLATION" },
    { terms: ["oil", "energy"], theme: "ENERGY" },
    { terms: ["central-bank", "central bank"], theme: "CENTRAL_BANK" },
    { terms: ["rates", "rate"], theme: "INTEREST_RATES" }
  ],
  bank: [
    { terms: ["bank", "banking"], theme: "BANKING" },
    { terms: ["credit", "debt", "default", "bonds"], theme: "CREDIT_MARKETS" },
    { terms: ["liquidity", "funding"], theme: "LIQUIDITY" }
  ],
  geopolitical: [
    { terms: ["war", "attack", "military", "conflict"], theme: "CONFLICT" },
    { terms: ["trade", "tariff"], theme: "TRADE" },
    { terms: ["sanction"], theme: "SANCTIONS" },
    { terms: ["shipping", "supply"], theme: "SUPPLY_CHAIN" }
  ],
  modern: [
    { terms: ["ai", "artificial-intelligence", "artificial intelligence"], theme: "AI" },
    { terms: ["semiconductor", "chip", "software"], theme: "TECHNOLOGY" },
    { terms: ["crypto", "bitcoin", "blockchain"], theme: "CRYPTO" },
    { terms: ["earnings", "layoff"], theme: "CORPORATE_ACTIVITY" }
  ]
};

export type GdeltGkgRecord = {
  recordId: string;
  date: string;
  source: string;
  url: string;
  rawThemes: string[];
  normalizedThemes: NormalizedTheme[];
};

type GkgArchiveCacheEntry = {
  loadedAt: number;
  records: GdeltGkgRecord[];
};

const gkgArchiveCache = new Map<string, GkgArchiveCacheEntry>();
const gkgEvidenceCache = new Map<string, { loadedAt: number; articles: IntelligenceEvidenceArticle[] }>();
const gkgCategoryEvidenceCache = new Map<string, { loadedAt: number; articles: IntelligenceEvidenceArticle[] }>();

function isHttpUrl(value: string): boolean {
  try {
    const parsed = new URL(value);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

async function fetchText(url: string, timeoutMs: number): Promise<string> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      signal: controller.signal,
      headers: { "user-agent": "PolicyResearchHub/1.0 IntelBeta GDELT GKG retrieval" },
      cache: "no-store"
    });
    if (!response.ok) {
      throw new Error(`GDELT request failed with ${response.status}`);
    }
    return await response.text();
  } finally {
    clearTimeout(timeout);
  }
}

async function fetchBuffer(url: string, timeoutMs: number): Promise<Buffer> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      signal: controller.signal,
      headers: { "user-agent": "PolicyResearchHub/1.0 IntelBeta GDELT GKG retrieval" },
      cache: "no-store"
    });
    if (!response.ok) {
      throw new Error(`GDELT archive request failed with ${response.status}`);
    }
    return Buffer.from(await response.arrayBuffer());
  } finally {
    clearTimeout(timeout);
  }
}

function findEndOfCentralDirectory(buffer: Buffer): number {
  for (let offset = buffer.length - 22; offset >= 0; offset -= 1) {
    if (buffer.readUInt32LE(offset) === 0x06054b50) {
      return offset;
    }
  }
  return -1;
}

export function extractFirstZipEntryText(buffer: Buffer): string {
  let localHeaderOffset = 0;
  let compressionMethod = buffer.readUInt16LE(8);
  let compressedSize = buffer.readUInt32LE(18);

  if (buffer.readUInt32LE(0) !== 0x04034b50 || compressedSize === 0) {
    const endOffset = findEndOfCentralDirectory(buffer);
    if (endOffset < 0) {
      throw new Error("Invalid GDELT ZIP archive.");
    }

    const centralDirectoryOffset = buffer.readUInt32LE(endOffset + 16);
    if (buffer.readUInt32LE(centralDirectoryOffset) !== 0x02014b50) {
      throw new Error("Invalid GDELT ZIP central directory.");
    }

    compressionMethod = buffer.readUInt16LE(centralDirectoryOffset + 10);
    compressedSize = buffer.readUInt32LE(centralDirectoryOffset + 20);
    localHeaderOffset = buffer.readUInt32LE(centralDirectoryOffset + 42);
  }

  if (buffer.readUInt32LE(localHeaderOffset) !== 0x04034b50) {
    throw new Error("Invalid GDELT ZIP local header.");
  }

  const fileNameLength = buffer.readUInt16LE(localHeaderOffset + 26);
  const extraLength = buffer.readUInt16LE(localHeaderOffset + 28);
  const dataStart = localHeaderOffset + 30 + fileNameLength + extraLength;
  const compressed = buffer.subarray(dataStart, dataStart + compressedSize);

  if (compressionMethod === 0) {
    return compressed.toString("utf8");
  }
  if (compressionMethod === 8) {
    return inflateRawSync(compressed).toString("utf8");
  }
  throw new Error(`Unsupported GDELT ZIP compression method ${compressionMethod}.`);
}

export function parseGdeltGkgManifest(manifest: string): string | null {
  const line = manifest
    .split(/\r?\n/)
    .map((item) => item.trim())
    .find((item) => item.includes(".gkg.csv.zip"));

  return line?.split(/\s+/).at(-1) ?? null;
}

function parseGdeltTimestamp(value: string): Date | null {
  const match = value.match(/^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})$/);
  if (!match) return null;
  const [, year, month, day, hour, minute, second] = match;
  const date = new Date(Date.UTC(Number(year), Number(month) - 1, Number(day), Number(hour), Number(minute), Number(second)));
  return Number.isNaN(date.getTime()) ? null : date;
}

function formatGdeltTimestamp(value: string): string {
  const date = parseGdeltTimestamp(value);
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

function timestampFromArchiveUrl(url: string): string | null {
  return url.match(/\/(\d{14})\.gkg\.csv\.zip$/)?.[1] ?? null;
}

function formatTimestampForArchive(date: Date): string {
  const parts = [
    date.getUTCFullYear(),
    date.getUTCMonth() + 1,
    date.getUTCDate(),
    date.getUTCHours(),
    date.getUTCMinutes(),
    date.getUTCSeconds()
  ];

  return `${parts[0]}${String(parts[1]).padStart(2, "0")}${String(parts[2]).padStart(2, "0")}${String(parts[3]).padStart(2, "0")}${String(parts[4]).padStart(2, "0")}${String(parts[5]).padStart(2, "0")}`;
}

export function buildGdeltGkgArchiveUrls(latestArchiveUrl: string, archiveCount = GDELT_GKG_ARCHIVE_COUNT): string[] {
  const latestTimestamp = timestampFromArchiveUrl(latestArchiveUrl);
  const latestDate = latestTimestamp ? parseGdeltTimestamp(latestTimestamp) : null;
  if (!latestDate) {
    return [latestArchiveUrl];
  }

  return Array.from({ length: archiveCount }, (_, index) => {
    const date = new Date(latestDate.getTime() - index * 15 * 60 * 1_000);
    return latestArchiveUrl.replace(`${latestTimestamp}.gkg.csv.zip`, `${formatTimestampForArchive(date)}.gkg.csv.zip`);
  });
}

function parseRawThemeTokens(themes: string, v2Themes: string): string[] {
  const tokens = new Set<string>();

  for (const field of [themes, v2Themes]) {
    for (const item of field.split(";")) {
      const token = item.split(",")[0]?.trim();
      if (token) {
        tokens.add(token);
      }
    }
  }

  return [...tokens];
}

function normalizeSource(source: string, url: string): string {
  if (source.trim()) {
    return source.trim().replace(/^www\./, "");
  }
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return "GDELT";
  }
}

export function parseGdeltGkgCsv(text: string): GdeltGkgRecord[] {
  const records: GdeltGkgRecord[] = [];

  for (const line of text.split(/\r?\n/)) {
    if (!line) continue;
    const columns = line.split("\t");
    const recordId = columns[0] ?? "";
    const date = columns[1] ?? "";
    const source = normalizeSource(columns[3] ?? "", columns[4] ?? "");
    const url = (columns[4] ?? "").trim();
    const rawThemes = parseRawThemeTokens(columns[7] ?? "", columns[8] ?? "");

    if (!recordId || !date || !isHttpUrl(url) || rawThemes.length === 0) {
      continue;
    }

    const signal = scoreThemeArticle({ id: recordId, raw_themes: rawThemes });

    records.push({
      recordId,
      date,
      source,
      url,
      rawThemes,
      normalizedThemes: signal.normalized_theme_list
    });
  }

  return records;
}

async function fetchGkgArchiveRecords(url: string): Promise<GdeltGkgRecord[]> {
  const cached = gkgArchiveCache.get(url);
  if (cached && Date.now() - cached.loadedAt < GDELT_GKG_CACHE_TTL_MS) {
    return cached.records;
  }

  const archive = await fetchBuffer(url, GDELT_GKG_TIMEOUT_MS);
  const csv = extractFirstZipEntryText(archive);
  const records = parseGdeltGkgCsv(csv);
  gkgArchiveCache.set(url, { loadedAt: Date.now(), records });
  return records;
}

async function fetchRecentGkgRecords(archiveCount = GDELT_GKG_ARCHIVE_COUNT): Promise<GdeltGkgRecord[]> {
  const manifest = await fetchText(GDELT_GKG_UPDATE_URL, GDELT_GKG_TIMEOUT_MS);
  const latestArchiveUrl = parseGdeltGkgManifest(manifest);
  if (!latestArchiveUrl) {
    return [];
  }

  const archiveUrls = buildGdeltGkgArchiveUrls(latestArchiveUrl, archiveCount);
  const results = await Promise.allSettled(archiveUrls.map((archiveUrl) => fetchGkgArchiveRecords(archiveUrl)));

  return results.flatMap((result) => result.status === "fulfilled" ? result.value : []);
}

function articleId(profileId: string, url: string): string {
  const hash = createHash("sha1").update(url).digest("hex").slice(0, 12);
  return `gkg-${profileId}-${hash}`;
}

function categoryArticleId(category: ProductCategory, url: string): string {
  const hash = createHash("sha1").update(`${category}:${url}`).digest("hex").slice(0, 12);
  return `gkg-${category.toLowerCase()}-${hash}`;
}

function titleCase(value: string): string {
  return value
    .split(/\s+/)
    .filter(Boolean)
    .map((word) => {
      const lower = word.toLowerCase();
      if (["ai", "api", "cpi", "ecb", "fed", "gdp", "ipo", "llm", "opec", "sec", "uk", "us"].includes(lower)) {
        return lower.toUpperCase();
      }
      return `${lower.charAt(0).toUpperCase()}${lower.slice(1)}`;
    })
    .join(" ");
}

function cleanUrlSegment(segment: string): string {
  const cleaned = decodeURIComponent(segment)
    .replace(/\.[a-z0-9]+$/i, "")
    .replace(/[_-]+/g, " ")
    .replace(/\b\d{4}\b|\b\d{2,}\b/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  return cleaned
    .split(/\s+/)
    .filter((word) => !/^(?=.*[a-z])(?=.*\d)[a-z0-9]{6,}$/i.test(word))
    .filter((word) => !/^article$/i.test(word))
    .join(" ")
    .trim();
}

function headlineFromUrl(url: string, source: string): string {
  try {
    const parsed = new URL(url);
    const candidates = parsed.pathname
      .split("/")
      .filter(Boolean)
      .map(cleanUrlSegment)
      .filter((part) => /[a-zA-Z]/.test(part) && part.length > 5)
      .sort((a, b) => b.split(/\s+/).length - a.split(/\s+/).length || b.length - a.length);

    const headline = candidates[0];
    if (!headline) {
      return `${source} article`;
    }

    return headline ? titleCase(headline).slice(0, 140) : `${source} article`;
  } catch {
    return `${source} article`;
  }
}

function formatThemesForSentence(themes: readonly NormalizedTheme[]): string {
  return themes
    .slice(0, 3)
    .map((theme) => theme.toLowerCase().replace(/_/g, " "))
    .join(" + ");
}

function urlTextForRecord(record: GdeltGkgRecord): string {
  return decodeURIComponent(record.url).toLowerCase().replace(/[_/-]+/g, " ");
}

function urlTextIncludesTerm(urlText: string, term: string): boolean {
  const normalized = term.toLowerCase();
  if (normalized.length <= 3 && /^[a-z0-9]+$/.test(normalized)) {
    return new RegExp(`(^|[^a-z0-9])${normalized}([^a-z0-9]|$)`).test(urlText);
  }
  return urlText.includes(normalized);
}

function normalizeMatchText(value: string): string {
  return value
    .toUpperCase()
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

function textMatchesSourcePattern(value: string, pattern: string): boolean {
  const haystack = matchTokens(value);
  const needle = matchTokens(pattern);
  if (needle.length === 0) return false;
  if (needle.length === 1) {
    return haystack.includes(needle[0]);
  }
  return containsTokenSequence(haystack, needle);
}

function matchedFocusTerms(record: GdeltGkgRecord, focusArea: ProductFocusArea): string[] {
  const urlSourceText = [
    record.url,
    record.source
  ].join(" ");
  const sourceText = [
    record.url,
    record.source,
    record.rawThemes.join(" ")
  ].join(" ");

  return focusArea.raw_patterns.filter((pattern) => {
    const normalizedPattern = normalizeMatchText(pattern);

    if (focusArea.id === "aml_sanctions" && normalizedPattern === "SANCTIONS") {
      return (
        textMatchesSourcePattern(urlSourceText, "SANCTION") ||
        textMatchesSourcePattern(urlSourceText, "SANCTIONS") ||
        textMatchesSourcePattern(urlSourceText, "SANCTIONED")
      );
    }

    return textMatchesSourcePattern(sourceText, pattern);
  });
}

function inferredThemesFromUrl(record: GdeltGkgRecord, profile: IntelligenceProfile): NormalizedTheme[] {
  const urlText = urlTextForRecord(record);
  const inferred = new Set<NormalizedTheme>();

  for (const item of PROFILE_TERM_THEMES[profile.id] ?? []) {
    if (item.terms.some((term) => urlTextIncludesTerm(urlText, term))) {
      inferred.add(item.theme);
    }
  }

  return [...inferred];
}

function relevanceScore(record: GdeltGkgRecord, profile: IntelligenceProfile): number {
  const primary = new Set(profile.signal.primary_drivers.map((driver) => driver.normalized_theme));
  const secondary = new Set(profile.signal.secondary_drivers.map((driver) => driver.normalized_theme));
  const profileThemes = new Set(profile.signal.normalized_theme_list);
  const urlText = urlTextForRecord(record);
  const urlTermScore = (PROFILE_EVIDENCE_TERMS[profile.id] ?? []).reduce((score, term) => {
    return urlTextIncludesTerm(urlText, term) ? score + 12 : score;
  }, 0);

  const themeScore = record.normalizedThemes.reduce((score, theme) => {
    if (primary.has(theme)) return score + 10;
    if (secondary.has(theme)) return score + 5;
    if (profileThemes.has(theme)) return score + 2;
    return score;
  }, 0);

  if (themeScore <= 0) {
    return 0;
  }

  return themeScore + urlTermScore + inferredThemesFromUrl(record, profile).length * 8;
}

function mapGkgRecordToEvidence(profile: IntelligenceProfile, record: GdeltGkgRecord, index: number): IntelligenceEvidenceArticle {
  const inferredThemes = inferredThemesFromUrl(record, profile);
  const relevantThemes = [...new Set([
    ...record.normalizedThemes.filter((theme) => profile.signal.normalized_theme_list.includes(theme)),
    ...inferredThemes.filter((theme) => profile.signal.normalized_theme_list.includes(theme))
  ])];
  const relatedThemes = relevantThemes.length > 0 ? relevantThemes : record.normalizedThemes;
  const cluster = profile.clusters[index % Math.max(profile.clusters.length, 1)];

  return {
    id: articleId(profile.id, record.url),
    headline: headlineFromUrl(record.url, record.source),
    url: record.url,
    source: record.source,
    timestamp: formatGdeltTimestamp(record.date),
    excerpt: `Live GDELT article matched to ${profile.label}.`,
    explanation: `Matched ${formatThemesForSentence(relatedThemes)} coverage`,
    relatedThemes,
    clusterId: cluster?.id ?? "gdelt-gkg",
    credibility: Math.max(60, 92 - index),
    impact: Math.max(55, 90 - index)
  };
}

export async function fetchGdeltGkgEvidenceForProfile(profile: IntelligenceProfile): Promise<IntelligenceEvidenceArticle[]> {
  const cached = gkgEvidenceCache.get(profile.id);
  if (cached && Date.now() - cached.loadedAt < GDELT_GKG_CACHE_TTL_MS) {
    return cached.articles;
  }

  const seenUrls = new Set<string>();
  const rankedRecords: { record: GdeltGkgRecord; relevance: number }[] = [];
  const records = await fetchRecentGkgRecords();

  for (const record of records) {
    if (seenUrls.has(record.url)) {
      continue;
    }

    const relevance = relevanceScore(record, profile);
    if (relevance <= 0) {
      continue;
    }

    seenUrls.add(record.url);
    rankedRecords.push({ record, relevance });

    if (rankedRecords.length >= GDELT_GKG_MAX_RECORDS * 2) {
      break;
    }
  }

  const articles: IntelligenceEvidenceArticle[] = [];
  const seenHeadlines = new Set<string>();

  for (const { record } of rankedRecords.sort((a, b) => b.relevance - a.relevance || b.record.date.localeCompare(a.record.date))) {
    const article = mapGkgRecordToEvidence(profile, record, articles.length);
    const headlineKey = `${article.source}:${article.headline}`.toLowerCase();
    if (seenHeadlines.has(headlineKey)) {
      continue;
    }
    seenHeadlines.add(headlineKey);
    articles.push(article);
    if (articles.length >= GDELT_GKG_MAX_RECORDS) {
      break;
    }
  }

  gkgEvidenceCache.set(profile.id, { loadedAt: Date.now(), articles });
  return articles;
}

function mapGkgRecordToCategoryEvidence(
  category: ProductCategory,
  record: GdeltGkgRecord,
  focusArea: ProductFocusArea,
  matchedTerms: readonly string[],
  index: number
): IntelligenceEvidenceArticle {
  const headline = headlineFromUrl(record.url, record.source);
  return {
    id: categoryArticleId(category, record.url),
    headline,
    url: record.url,
    source: record.source,
    timestamp: formatGdeltTimestamp(record.date),
    excerpt: `Live GDELT GKG match on source-side terms: ${matchedTerms.join(", ")}.`,
    explanation: `Matched ${focusArea.label}: ${matchedTerms.join(", ")}`,
    relatedThemes: record.normalizedThemes,
    matchedTerms: [...matchedTerms],
    focusAreaId: focusArea.id,
    focusAreaLabel: focusArea.label,
    clusterId: `${category.toLowerCase()}-${focusArea.id}`,
    credibility: Math.max(60, 92 - index),
    impact: Math.max(55, 90 - index)
  };
}

export async function fetchGdeltGkgEvidenceForProductCategory(
  category: ProductCategory,
  focusId?: string | null,
  options?: { archiveCount?: number }
): Promise<IntelligenceEvidenceArticle[]> {
  const focusAreas = focusAreasForProductCategory(category).filter((focusArea) => !focusId || focusArea.id === focusId);
  if (focusAreas.length === 0) {
    return [];
  }

  const archiveCount = options?.archiveCount ?? GDELT_GKG_ARCHIVE_COUNT;
  const cacheKey = `${category}:${focusId ?? "all"}:${archiveCount}`;
  const cached = gkgCategoryEvidenceCache.get(cacheKey);
  if (cached && Date.now() - cached.loadedAt < GDELT_GKG_CACHE_TTL_MS) {
    return cached.articles;
  }

  const records = await fetchRecentGkgRecords(archiveCount);
  const articles = mapGdeltGkgRecordsToProductCategoryEvidence(category, records, focusAreas);
  gkgCategoryEvidenceCache.set(cacheKey, { loadedAt: Date.now(), articles });
  return articles;
}

export function mapGdeltGkgRecordsToProductCategoryEvidence(
  category: ProductCategory,
  records: readonly GdeltGkgRecord[],
  focusAreas: readonly ProductFocusArea[] = focusAreasForProductCategory(category)
): IntelligenceEvidenceArticle[] {
  const rankedRecords: { record: GdeltGkgRecord; focusArea: ProductFocusArea; matchedTerms: string[]; score: number }[] = [];
  const seenUrls = new Set<string>();

  for (const record of records) {
    if (seenUrls.has(record.url)) {
      continue;
    }

    const matchedFocusAreas = focusAreas
      .map((focusArea) => ({ focusArea, matchedTerms: matchedFocusTerms(record, focusArea) }))
      .filter((item) => item.matchedTerms.length > 0)
      .sort((a, b) => b.matchedTerms.length - a.matchedTerms.length);

    const bestMatch = matchedFocusAreas[0];
    if (!bestMatch) {
      continue;
    }

    seenUrls.add(record.url);
    const urlText = urlTextForRecord(record);
    const score = bestMatch.matchedTerms.length * 20 + bestMatch.matchedTerms.reduce((sum, term) => {
      return urlTextIncludesTerm(urlText, term) ? sum + 10 : sum;
    }, 0);
    rankedRecords.push({
      record,
      focusArea: bestMatch.focusArea,
      matchedTerms: bestMatch.matchedTerms,
      score
    });

    if (rankedRecords.length >= GDELT_GKG_MAX_RECORDS * 3) {
      break;
    }
  }

  return rankedRecords
    .sort((a, b) => b.score - a.score || b.record.date.localeCompare(a.record.date))
    .slice(0, GDELT_GKG_MAX_RECORDS)
    .map((item, index) => mapGkgRecordToCategoryEvidence(category, item.record, item.focusArea, item.matchedTerms, index));
}

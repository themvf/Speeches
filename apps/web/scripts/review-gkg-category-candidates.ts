import {
  buildGdeltGkgArchiveUrls,
  extractFirstZipEntryText,
  parseGdeltGkgCsv,
  parseGdeltGkgManifest,
  type GdeltGkgRecord
} from "../lib/server/gdelt-gkg.ts";
import { CATEGORY_DOC_QUERY_TERMS, CATEGORY_PATTERN_ALIASES } from "../lib/server/gdelt-doc.ts";
import {
  focusAreasForProductCategory,
  PRODUCT_CATEGORY_LABELS,
  type ProductCategory,
  type ProductFocusArea
} from "../lib/theme-intelligence.ts";

const GDELT_GKG_UPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt";
const FETCH_TIMEOUT_MS = 12_000;
const DEFAULT_ARCHIVE_COUNT = 48;
const DEFAULT_MAX_ROWS = 50;
const FETCH_CONCURRENCY = 6;
const DEFAULT_FORMAT = "markdown";

type CandidateRow = {
  rawDate: string;
  timestamp: string;
  source: string;
  headline: string;
  focus: string;
  visibleTerms: string[];
  rawThemeTerms: string[];
  rawThemes: string[];
  normalizedThemes: string[];
  url: string;
};

function getArg(flag: string): string | null {
  const index = process.argv.indexOf(flag);
  if (index < 0) return null;
  return process.argv[index + 1] ?? null;
}

function parseCategoryArg(value: string | null): ProductCategory {
  const normalized = String(value ?? "CAPITAL_FORMATION").trim().toUpperCase();
  const categories: ProductCategory[] = [
    "SECURITIES_REGULATION",
    "CAPITAL_FORMATION",
    "AML",
    "ENFORCEMENT",
    "AI_TECH",
    "CRYPTO",
    "CREDIT_MARKETS",
    "FINANCIAL_MARKETS",
    "ECONOMIC_GROWTH"
  ];

  if (categories.includes(normalized as ProductCategory)) {
    return normalized as ProductCategory;
  }

  throw new Error(`Unsupported category '${value}'.`);
}

function parseNumberArg(value: string | null, fallback: number): number {
  const parsed = Number.parseInt(String(value ?? ""), 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function parseFormatArg(value: string | null): "markdown" | "csv" {
  const normalized = String(value ?? DEFAULT_FORMAT).trim().toLowerCase();
  return normalized === "csv" ? "csv" : "markdown";
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

function textMatchesPattern(value: string, pattern: string): boolean {
  const haystack = matchTokens(value);
  const needle = matchTokens(pattern);
  if (needle.length === 0) return false;
  if (needle.length === 1) {
    return haystack.includes(needle[0]);
  }
  return containsTokenSequence(haystack, needle);
}

function aliasesForPattern(pattern: string): readonly string[] {
  const normalized = normalizeMatchText(pattern);
  return CATEGORY_PATTERN_ALIASES[normalized] ?? [pattern];
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

    return titleCase(headline).slice(0, 160);
  } catch {
    return `${source} article`;
  }
}

function parseGdeltTimestamp(value: string): Date | null {
  const match = value.match(/^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})$/);
  if (!match) return null;
  const [, year, month, day, hour, minute, second] = match;
  const date = new Date(Date.UTC(Number(year), Number(month) - 1, Number(day), Number(hour), Number(minute), Number(second)));
  return Number.isNaN(date.getTime()) ? null : date;
}

function formatTimestamp(value: string): string {
  const date = parseGdeltTimestamp(value);
  if (!date) return value || "GDELT";
  return date.toLocaleString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "UTC",
    timeZoneName: "short"
  });
}

async function fetchText(url: string): Promise<string> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

  try {
    const response = await fetch(url, {
      signal: controller.signal,
      headers: { "user-agent": "PolicyResearchHub/1.0 CapitalFormation GKG review" },
      cache: "no-store"
    });
    if (!response.ok) {
      throw new Error(`Request failed with ${response.status}`);
    }
    return await response.text();
  } finally {
    clearTimeout(timeout);
  }
}

async function fetchBuffer(url: string): Promise<Buffer> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

  try {
    const response = await fetch(url, {
      signal: controller.signal,
      headers: { "user-agent": "PolicyResearchHub/1.0 CapitalFormation GKG review" },
      cache: "no-store"
    });
    if (!response.ok) {
      throw new Error(`Archive request failed with ${response.status}`);
    }
    return Buffer.from(await response.arrayBuffer());
  } finally {
    clearTimeout(timeout);
  }
}

async function fetchArchiveRecords(url: string): Promise<GdeltGkgRecord[]> {
  const archive = await fetchBuffer(url);
  const csv = extractFirstZipEntryText(archive);
  return parseGdeltGkgCsv(csv);
}

async function fetchRecentRecords(archiveCount: number): Promise<GdeltGkgRecord[]> {
  const manifest = await fetchText(GDELT_GKG_UPDATE_URL);
  const latestArchiveUrl = parseGdeltGkgManifest(manifest);
  if (!latestArchiveUrl) {
    throw new Error("Could not parse latest GKG archive URL.");
  }

  const archiveUrls = buildGdeltGkgArchiveUrls(latestArchiveUrl, archiveCount);
  const records: GdeltGkgRecord[] = [];

  for (let index = 0; index < archiveUrls.length; index += FETCH_CONCURRENCY) {
    const batch = archiveUrls.slice(index, index + FETCH_CONCURRENCY);
    const results = await Promise.allSettled(batch.map((url) => fetchArchiveRecords(url)));
    for (const result of results) {
      if (result.status === "fulfilled") {
        records.push(...result.value);
      }
    }
  }

  return records;
}

function matchedVisibleTerms(record: GdeltGkgRecord, focusArea: ProductFocusArea): string[] {
  const visibleText = [record.url, record.source, headlineFromUrl(record.url, record.source)].join(" ");
  const terms = CATEGORY_DOC_QUERY_TERMS[focusArea.category]?.[focusArea.id] ?? focusArea.raw_patterns;
  return terms.filter((term) => textMatchesPattern(visibleText, term));
}

function matchedRawThemeTerms(record: GdeltGkgRecord, focusArea: ProductFocusArea): string[] {
  const normalizedThemes = new Set(record.rawThemes.map((theme) => normalizeMatchText(theme)));

  return focusArea.raw_patterns.filter((pattern) => {
    return aliasesForPattern(pattern).some((alias) => normalizedThemes.has(normalizeMatchText(alias)));
  });
}

function candidateRows(records: readonly GdeltGkgRecord[], focusAreas: readonly ProductFocusArea[]): CandidateRow[] {
  const seenUrls = new Set<string>();
  const seenHeadlineKeys = new Set<string>();
  const rows: CandidateRow[] = [];

  for (const record of records) {
    if (seenUrls.has(record.url)) {
      continue;
    }

    const matches = focusAreas
      .map((focusArea) => {
        const visibleTerms = matchedVisibleTerms(record, focusArea);
        const rawThemeTerms = matchedRawThemeTerms(record, focusArea);
        return { focusArea, visibleTerms, rawThemeTerms };
      })
      .filter((item) => item.visibleTerms.length > 0 || item.rawThemeTerms.length > 0)
      .sort((a, b) => {
        const scoreA = a.visibleTerms.length * 3 + a.rawThemeTerms.length;
        const scoreB = b.visibleTerms.length * 3 + b.rawThemeTerms.length;
        return scoreB - scoreA || a.focusArea.label.localeCompare(b.focusArea.label);
      });

    const bestMatch = matches[0];
    if (!bestMatch) {
      continue;
    }

    const headline = headlineFromUrl(record.url, record.source);
    const headlineKey = normalizeMatchText(headline);
    if (seenHeadlineKeys.has(headlineKey)) {
      continue;
    }

    seenUrls.add(record.url);
    seenHeadlineKeys.add(headlineKey);
    rows.push({
      rawDate: record.date,
      timestamp: formatTimestamp(record.date),
      source: record.source,
      headline,
      focus: bestMatch.focusArea.label,
      visibleTerms: bestMatch.visibleTerms,
      rawThemeTerms: bestMatch.rawThemeTerms,
      rawThemes: record.rawThemes.slice(0, 8),
      normalizedThemes: record.normalizedThemes,
      url: record.url
    });
  }

  return rows.sort((a, b) => b.rawDate.localeCompare(a.rawDate));
}

function escapePipe(value: string): string {
  return value.replace(/\|/g, "\\|").replace(/\s+/g, " ").trim();
}

function escapeCsv(value: string): string {
  const normalized = value.replace(/\r?\n/g, " ").trim();
  return `"${normalized.replace(/"/g, "\"\"")}"`;
}

async function main() {
  const category = parseCategoryArg(getArg("--category"));
  const archiveCount = parseNumberArg(getArg("--archives"), DEFAULT_ARCHIVE_COUNT);
  const maxRows = parseNumberArg(getArg("--max"), DEFAULT_MAX_ROWS);
  const format = parseFormatArg(getArg("--format"));
  const focusAreas = focusAreasForProductCategory(category);
  const records = await fetchRecentRecords(archiveCount);
  const rows = candidateRows(records, focusAreas).slice(0, maxRows);

  if (format === "csv") {
    console.log("timestamp,source,focus,visible_hits,raw_theme_hits,raw_gkg_themes,normalized_themes,headline,url");
    for (const row of rows) {
      console.log([
        escapeCsv(row.timestamp),
        escapeCsv(row.source),
        escapeCsv(row.focus),
        escapeCsv(row.visibleTerms.join(", ")),
        escapeCsv(row.rawThemeTerms.join(", ")),
        escapeCsv(row.rawThemes.join(", ")),
        escapeCsv(row.normalizedThemes.join(", ")),
        escapeCsv(row.headline),
        escapeCsv(row.url)
      ].join(","));
    }
    return;
  }

  console.log(`# ${PRODUCT_CATEGORY_LABELS[category]} GKG Review`);
  console.log(``);
  console.log(`- category: ${category}`);
  console.log(`- archives scanned: ${archiveCount}`);
  console.log(`- raw GKG records scanned: ${records.length}`);
  console.log(`- candidate rows: ${rows.length}`);
  console.log(``);
  console.log(`| timestamp | source | focus | visible hits | raw theme hits | raw GKG themes | normalized themes | headline | url |`);
  console.log(`|---|---|---|---|---|---|---|---|---|`);

  if (rows.length === 0) {
    console.log(`| - | - | - | No candidates found | - | - | - | - | - |`);
    return;
  }

  for (const row of rows) {
    console.log(
      `| ${escapePipe(row.timestamp)} | ${escapePipe(row.source)} | ${escapePipe(row.focus)} | ${escapePipe(row.visibleTerms.join(", "))} | ${escapePipe(row.rawThemeTerms.join(", "))} | ${escapePipe(row.rawThemes.join(", "))} | ${escapePipe(row.normalizedThemes.join(", "))} | ${escapePipe(row.headline)} | ${escapePipe(row.url)} |`
    );
  }
}

await main();

import {
  buildGdeltGkgArchiveUrls,
  extractFirstZipEntryText,
  parseGdeltGkgCsv,
  parseGdeltGkgManifest,
  type GdeltGkgRecord
} from "../lib/server/gdelt-gkg.ts";

const GDELT_GKG_UPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt";
const FETCH_TIMEOUT_MS = 12_000;
const FETCH_CONCURRENCY = 6;
const DEFAULT_ARCHIVE_COUNT = 96;
const DEFAULT_MAX_PER_THEME = 10;

type ThemeSampleRow = {
  requestedTheme: string;
  timestamp: string;
  source: string;
  headline: string;
  url: string;
  rawThemes: string[];
  normalizedThemes: string[];
};

function getArg(flag: string): string | null {
  const index = process.argv.indexOf(flag);
  if (index < 0) return null;
  return process.argv[index + 1] ?? null;
}

function parseNumberArg(value: string | null, fallback: number): number {
  const parsed = Number.parseInt(String(value ?? ""), 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function parseThemesArg(value: string | null): string[] {
  return String(value ?? "")
    .split(/[,\n]/)
    .map((item) => item.trim().toUpperCase())
    .filter(Boolean);
}

function normalizeTheme(theme: string): string {
  return theme.trim().toUpperCase();
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
      headers: { "user-agent": "PolicyResearchHub/1.0 GKG theme sample review" },
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
      headers: { "user-agent": "PolicyResearchHub/1.0 GKG theme sample review" },
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

function escapeCsv(value: string): string {
  return `"${String(value ?? "").replace(/"/g, "\"\"")}"`;
}

function buildThemeSampleRows(records: readonly GdeltGkgRecord[], requestedThemes: readonly string[], maxPerTheme: number): ThemeSampleRow[] {
  const requested = new Set(requestedThemes.map(normalizeTheme));
  const themeCounts = new Map<string, number>(requestedThemes.map((theme) => [normalizeTheme(theme), 0]));
  const seenPerTheme = new Map<string, Set<string>>(requestedThemes.map((theme) => [normalizeTheme(theme), new Set<string>()]));
  const rows: ThemeSampleRow[] = [];

  for (const record of [...records].sort((a, b) => b.date.localeCompare(a.date))) {
    for (const rawTheme of record.rawThemes) {
      const normalizedRawTheme = normalizeTheme(rawTheme);
      if (!requested.has(normalizedRawTheme)) {
        continue;
      }

      const currentCount = themeCounts.get(normalizedRawTheme) ?? 0;
      if (currentCount >= maxPerTheme) {
        continue;
      }

      const headline = headlineFromUrl(record.url, record.source);
      const dedupeKey = `${headline.toLowerCase()}|${record.source.toLowerCase()}`;
      const seenKeys = seenPerTheme.get(normalizedRawTheme);
      if (seenKeys?.has(dedupeKey)) {
        continue;
      }

      seenKeys?.add(dedupeKey);
      themeCounts.set(normalizedRawTheme, currentCount + 1);
      rows.push({
        requestedTheme: normalizedRawTheme,
        timestamp: formatTimestamp(record.date),
        source: record.source,
        headline,
        url: record.url,
        rawThemes: record.rawThemes,
        normalizedThemes: record.normalizedThemes
      });
    }

    if ([...themeCounts.values()].every((count) => count >= maxPerTheme)) {
      break;
    }
  }

  return rows;
}

async function main() {
  const themes = parseThemesArg(getArg("--themes"));
  if (themes.length === 0) {
    throw new Error("Pass --themes with a comma-separated list of raw GKG themes.");
  }

  const archives = parseNumberArg(getArg("--archives"), DEFAULT_ARCHIVE_COUNT);
  const maxPerTheme = parseNumberArg(getArg("--max-per-theme"), DEFAULT_MAX_PER_THEME);
  const records = await fetchRecentRecords(archives);
  const rows = buildThemeSampleRows(records, themes, maxPerTheme);

  console.log("requested_theme,timestamp,source,headline,url,normalized_themes,raw_themes");
  for (const row of rows) {
    console.log([
      escapeCsv(row.requestedTheme),
      escapeCsv(row.timestamp),
      escapeCsv(row.source),
      escapeCsv(row.headline),
      escapeCsv(row.url),
      escapeCsv(row.normalizedThemes.join(", ")),
      escapeCsv(row.rawThemes.join(", "))
    ].join(","));
  }
}

await main();

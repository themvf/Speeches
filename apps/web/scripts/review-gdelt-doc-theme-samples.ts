export {};

const GDELT_DOC_ENDPOINT = "http://api.gdeltproject.org/api/v2/doc/doc";
const FETCH_TIMEOUT_MS = 60_000;

type GdeltDocArticle = {
  url?: string;
  title?: string;
  seendate?: string;
  domain?: string;
  language?: string;
  sourcecountry?: string;
};

type GdeltDocResponse = {
  articles?: GdeltDocArticle[];
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

function buildUrl(theme: string, maxRecords: number, timespan: string): string {
  const url = new URL(GDELT_DOC_ENDPOINT);
  url.searchParams.set("query", `theme:${theme}`);
  url.searchParams.set("mode", "artlist");
  url.searchParams.set("format", "json");
  url.searchParams.set("sort", "datedesc");
  url.searchParams.set("maxrecords", String(maxRecords));
  url.searchParams.set("timespan", timespan);
  return url.toString();
}

function escapeCsv(value: string): string {
  return `"${String(value ?? "").replace(/"/g, "\"\"")}"`;
}

function parseSeenDate(value: string): Date | null {
  const match = value.match(/^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z$/);
  if (!match) return null;
  const [, year, month, day, hour, minute, second] = match;
  const date = new Date(Date.UTC(Number(year), Number(month) - 1, Number(day), Number(hour), Number(minute), Number(second)));
  return Number.isNaN(date.getTime()) ? null : date;
}

function formatSeenDate(value: string): string {
  const date = parseSeenDate(value);
  if (!date) return value || "";
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

async function fetchJson(url: string): Promise<GdeltDocResponse> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

  try {
    const response = await fetch(url, {
      signal: controller.signal,
      headers: { "user-agent": "PolicyResearchHub/1.0 DOC theme sample review" },
      cache: "no-store"
    });
    if (!response.ok) {
      throw new Error(`Request failed with ${response.status}`);
    }
    return await response.json() as GdeltDocResponse;
  } finally {
    clearTimeout(timeout);
  }
}

async function main() {
  const themes = parseThemesArg(getArg("--themes"));
  if (themes.length === 0) {
    throw new Error("Pass --themes with a comma-separated list of raw GDELT themes.");
  }

  const maxPerTheme = parseNumberArg(getArg("--max-per-theme"), 10);
  const timespan = getArg("--timespan") ?? "7days";

  console.log("requested_theme,timestamp,domain,title,url,language,source_country");

  for (const theme of themes) {
    const url = buildUrl(theme, maxPerTheme, timespan);

    try {
      const result = await fetchJson(url);
      const articles = result.articles ?? [];

      for (const article of articles) {
        console.log([
          escapeCsv(theme),
          escapeCsv(formatSeenDate(article.seendate ?? "")),
          escapeCsv(article.domain ?? ""),
          escapeCsv(article.title ?? ""),
          escapeCsv(article.url ?? ""),
          escapeCsv(article.language ?? ""),
          escapeCsv(article.sourcecountry ?? "")
        ].join(","));
      }
    } catch (error) {
      console.log([
        escapeCsv(theme),
        escapeCsv(""),
        escapeCsv(""),
        escapeCsv(`ERROR: ${String(error)}`),
        escapeCsv(url),
        escapeCsv(""),
        escapeCsv("")
      ].join(","));
    }
  }
}

await main();

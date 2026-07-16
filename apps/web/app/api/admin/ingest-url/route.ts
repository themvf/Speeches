import { type NextRequest } from "next/server";
import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import { extractArticle } from "@/lib/server/url-article-extractor";
import { upsertRssArticles } from "@/lib/server/neon";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Manual admin URL ingests land under this feed key so they're
// distinguishable from scheduled RSS sources in the feed.
const FEED_KEY = "manual_url";
const FETCH_TIMEOUT_MS = 20_000;
const MAX_BYTES = 5_000_000; // don't slurp giant pages into memory

const BROWSER_HEADERS: Record<string, string> = {
  "User-Agent":
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
  Accept: "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
  "Accept-Language": "en-US,en;q=0.5",
};

// Admin-only route (middleware gates /api/admin/*), but the user supplies the
// URL, so refuse anything but public http(s) to avoid SSRF into localhost /
// link-local / private ranges.
function validateUrl(raw: string): { url: URL } | { error: string } {
  let candidate = raw.trim();
  if (!candidate) return { error: "URL is required." };
  if (!/^https?:\/\//i.test(candidate)) candidate = "https://" + candidate;
  let url: URL;
  try {
    url = new URL(candidate);
  } catch {
    return { error: "Not a valid URL." };
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    return { error: "Only http(s) URLs are supported." };
  }
  const host = url.hostname.toLowerCase();
  const blocked =
    host === "localhost" ||
    host.endsWith(".localhost") ||
    host === "0.0.0.0" ||
    host === "::1" ||
    /^127\./.test(host) ||
    /^10\./.test(host) ||
    /^192\.168\./.test(host) ||
    /^169\.254\./.test(host) ||
    /^172\.(1[6-9]|2\d|3[01])\./.test(host);
  if (blocked) return { error: "Refusing to fetch a private/loopback address." };
  return { url };
}

async function fetchHtml(url: string): Promise<{ html: string } | { error: string }> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  try {
    const resp = await fetch(url, { headers: BROWSER_HEADERS, signal: controller.signal, redirect: "follow" });
    if (!resp.ok) return { error: `Fetch failed: HTTP ${resp.status}` };
    const type = resp.headers.get("content-type") ?? "";
    if (type && !/html|xml|text/i.test(type)) return { error: `Unsupported content-type: ${type}` };
    const buf = await resp.arrayBuffer();
    if (buf.byteLength > MAX_BYTES) return { error: "Page too large to process." };
    return { html: new TextDecoder("utf-8").decode(buf) };
  } catch (err) {
    const msg = err instanceof Error ? err.message : "unknown error";
    return { error: controller.signal.aborted ? "Fetch timed out." : `Fetch error: ${msg}` };
  } finally {
    clearTimeout(timer);
  }
}

export async function POST(req: NextRequest) {
  const requestId = createRequestId();
  let body: { url?: string; dryRun?: boolean };
  try {
    body = await req.json();
  } catch {
    return fail("Invalid JSON body.", "BAD_BODY", 400, requestId);
  }

  const validated = validateUrl(body.url ?? "");
  if ("error" in validated) return fail(validated.error, "BAD_URL", 400, requestId);
  const canonicalUrl = validated.url.toString();

  const fetched = await fetchHtml(canonicalUrl);
  if ("error" in fetched) return fail(fetched.error, "FETCH_FAILED", 502, requestId);

  const article = extractArticle(fetched.html, canonicalUrl);
  if (article.wordCount === 0 && !article.description) {
    return fail("No readable article text could be extracted from that page.", "NO_CONTENT", 422, requestId);
  }

  // dryRun previews the extraction without touching the feed.
  if (body.dryRun) {
    return ok({ added: false, feedKey: FEED_KEY, article }, requestId);
  }

  const stats = await upsertRssArticles(
    [{
      guid: canonicalUrl,
      title: article.title,
      url: canonicalUrl,
      description: article.description,
      author: article.author,
      publishedAt: article.publishedAt ? new Date(article.publishedAt) : new Date(),
    }],
    FEED_KEY,
  );
  const outcome = stats.inserted ? "added" : stats.updated ? "updated" : "already present (unchanged)";

  return ok({ added: true, outcome, feedKey: FEED_KEY, article }, requestId);
}

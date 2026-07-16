// Server-side web-page structure analyzer + readable-text extractor (SEC-35).
// Port of the user's BeautifulSoup script to cheerio: hidden-node detection,
// JSON-LD extraction, text-visibility stats, and main-article extraction.
// Pure (takes HTML, returns a report) so it's unit-testable without a fetch.

import * as cheerio from "cheerio";
import type { CheerioAPI } from "cheerio";

export interface HiddenNode {
  tag: string;
  hiddenBy: string;
  textPreview: string;
}

export interface TextVisibility {
  totalWords: number;
  hiddenWords: number;
  visibleWords: number;
  hiddenRatio: number;
}

export interface ExtractedArticle {
  title: string;
  author: string;
  publishedAt: string | null; // ISO, from JSON-LD/meta when available
  description: string;        // meta/OG/JSON-LD summary, else lead of the body
  readableText: string;
  wordCount: number;
  jsonLdCount: number;
  hiddenNodes: HiddenNode[];
  textVisibility: TextVisibility;
}

const HIDDEN_STYLE_RE = /(display\s*:\s*none|visibility\s*:\s*hidden)/i;
const HIDING_CLASSES = ["hidden", "collapsed", "invisible", "d-none", "visually-hidden", "sr-only"];
const JUNK_TAGS = "script,style,noscript,iframe,nav,footer,header,aside,button,form,svg,canvas";
const BLOCK_TAGS = "p,h1,h2,h3,h4,li,blockquote";
const MIN_PARAGRAPH_LENGTH = 20;

function wordCount(text: string): number {
  const t = text.trim();
  return t ? t.split(/\s+/).length : 0;
}

function analyzeHiddenNodes($: CheerioAPI): HiddenNode[] {
  const results: HiddenNode[] = [];
  const seen = new Set<unknown>();

  $("[style]").each((_, el) => {
    const style = $(el).attr("style") ?? "";
    if (HIDDEN_STYLE_RE.test(style)) {
      seen.add(el);
      results.push({
        tag: (el as { tagName?: string }).tagName ?? "?",
        hiddenBy: `inline style (${style.slice(0, 60)})`,
        textPreview: $(el).text().trim().slice(0, 100),
      });
    }
  });

  for (const cls of HIDING_CLASSES) {
    $(`.${cls}`).each((_, el) => {
      if (seen.has(el)) return;
      // class_ match in the Python used exact token membership; cheerio's
      // .class selector already does token matching.
      seen.add(el);
      results.push({
        tag: (el as { tagName?: string }).tagName ?? "?",
        hiddenBy: `class='${cls}'`,
        textPreview: $(el).text().trim().slice(0, 100),
      });
    });
  }
  return results;
}

function extractJsonLd($: CheerioAPI): unknown[] {
  const out: unknown[] = [];
  $('script[type="application/ld+json"]').each((_, el) => {
    let raw = $(el).text().trim();
    if (!raw) return;
    if (raw.startsWith("<!--")) raw = raw.slice(4);
    if (raw.endsWith("-->")) raw = raw.slice(0, -3);
    try {
      out.push(JSON.parse(raw));
    } catch {
      // ignore malformed blocks, matching the script
    }
  });
  return out;
}

function calculateTextVisibility(html: string): TextVisibility {
  const $ = cheerio.load(html);
  $("script,style,noscript,iframe,nav,footer").remove();
  const totalWords = wordCount($.root().text());
  let hiddenText = "";
  $("[style]").each((_, el) => {
    if (HIDDEN_STYLE_RE.test($(el).attr("style") ?? "")) hiddenText += " " + $(el).text().trim();
  });
  const hiddenWords = wordCount(hiddenText);
  const visibleWords = Math.max(totalWords - hiddenWords, 0);
  return {
    totalWords,
    hiddenWords,
    visibleWords,
    hiddenRatio: totalWords ? hiddenWords / totalWords : 0,
  };
}

function extractReadableText(html: string): string {
  const $ = cheerio.load(html);
  $("[style]").each((_, el) => {
    if (HIDDEN_STYLE_RE.test($(el).attr("style") ?? "")) $(el).remove();
  });
  $(JUNK_TAGS).remove();

  // Element-based selections throughout (never $.root(), which types as a
  // Document and can't be .find()'d) - cheerio.load normalizes to a full
  // document so <body> always exists as the last real fallback.
  let container = $("body").first();
  if ($("article").first().length) container = $("article").first();
  else if ($("main").first().length) container = $("main").first();
  else if ($('[role="main"]').first().length) container = $('[role="main"]').first();

  const blocks: string[] = [];
  container.find(BLOCK_TAGS).each((_, el) => {
    const text = $(el).text().trim().replace(/\s+/g, " ");
    if (text.length >= MIN_PARAGRAPH_LENGTH) blocks.push(text);
  });
  return blocks.join("\n\n");
}

// JSON-LD is the most reliable byline/date source; fall back to meta tags.
function firstJsonLdArticle(blocks: unknown[]): Record<string, unknown> | null {
  const flat: Record<string, unknown>[] = [];
  const visit = (node: unknown) => {
    if (Array.isArray(node)) node.forEach(visit);
    else if (node && typeof node === "object") {
      const obj = node as Record<string, unknown>;
      flat.push(obj);
      if (Array.isArray(obj["@graph"])) (obj["@graph"] as unknown[]).forEach(visit);
    }
  };
  blocks.forEach(visit);
  const isArticle = (t: unknown) =>
    typeof t === "string" && /article|newsarticle|reportagenewsarticle|blogposting/i.test(t);
  return flat.find((o) => isArticle(o["@type"]) || (Array.isArray(o["@type"]) && o["@type"].some(isArticle))) ?? flat[0] ?? null;
}

function jsonLdAuthor(article: Record<string, unknown> | null): string {
  if (!article) return "";
  const a = article["author"];
  if (typeof a === "string") return a;
  if (Array.isArray(a)) {
    return a.map((x) => (x && typeof x === "object" ? String((x as Record<string, unknown>).name ?? "") : String(x))).filter(Boolean).join(", ");
  }
  if (a && typeof a === "object") return String((a as Record<string, unknown>).name ?? "");
  return "";
}

export function extractArticle(html: string, url: string): ExtractedArticle {
  const $ = cheerio.load(html);
  const jsonLd = extractJsonLd($);
  const ld = firstJsonLdArticle(jsonLd);

  const metaContent = (selector: string) => $(selector).attr("content")?.trim() || "";
  const title =
    (ld && typeof ld["headline"] === "string" ? (ld["headline"] as string) : "") ||
    metaContent('meta[property="og:title"]') ||
    $("title").first().text().trim() ||
    url;

  const author =
    jsonLdAuthor(ld) ||
    metaContent('meta[name="author"]') ||
    metaContent('meta[property="article:author"]');

  const publishedRaw =
    (ld && typeof ld["datePublished"] === "string" ? (ld["datePublished"] as string) : "") ||
    metaContent('meta[property="article:published_time"]') ||
    metaContent('meta[name="date"]');
  let publishedAt: string | null = null;
  if (publishedRaw) {
    const d = new Date(publishedRaw);
    if (!Number.isNaN(d.getTime())) publishedAt = d.toISOString();
  }

  const readableText = extractReadableText(html);
  const metaDescription =
    metaContent('meta[property="og:description"]') ||
    metaContent('meta[name="description"]') ||
    (ld && typeof ld["description"] === "string" ? (ld["description"] as string) : "");
  // Feed snippet: the page's own summary if present, else the lead of the body.
  const description = (metaDescription || readableText.slice(0, 600)).trim();

  return {
    title: title.slice(0, 500),
    author: author.slice(0, 300),
    publishedAt,
    description,
    readableText,
    wordCount: wordCount(readableText),
    jsonLdCount: jsonLd.length,
    hiddenNodes: analyzeHiddenNodes($),
    textVisibility: calculateTextVisibility(html),
  };
}

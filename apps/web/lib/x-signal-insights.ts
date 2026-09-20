import { decodeEntities } from "./intel-topic-matching.ts";

/** Client-safe view of an ingested post; the original objects are retained. */
export type XSignalPost = {
  id: number;
  guid?: string;
  feed_key: string;
  feed_label?: string | null;
  title: string;
  description: string;
  author: string;
  url: string;
  published_at: string | null;
  fetched_at?: string;
  analysis?: unknown;
};

export type XSignalWindow = "24h" | "7d" | "all";

export type XSignalTakeaway = {
  post: XSignalPost;
  text: string;
  kind: "analysis" | "excerpt";
  whyItMatters: string | null;
  watchNext: string | null;
};

export type XSignalInsights = {
  posts: XSignalPost[];
  takeaways: XSignalTakeaway[];
  accountCount: number;
  latestPublishedAt: string | null;
  undatedCount: number;
};

const X_FEED_PREFIX = "x_public_timeline_";
const X_HOSTS = new Set(["x.com", "www.x.com", "mobile.x.com", "twitter.com", "www.twitter.com", "mobile.twitter.com"]);
const USERNAME = /^[a-z0-9_]{1,15}$/i;
const DAY_MS = 24 * 60 * 60 * 1000;
const MAX_EXCERPT = 260;

function cleanText(value: unknown): string {
  if (typeof value !== "string") return "";
  // Decode numeric entities first: the shared decoder uses UTF-16 code units,
  // which cannot represent emoji and other code points above U+FFFF correctly.
  const numericDecoded = value.replace(/<[^>]+>/g, " ").replace(/&#(x[\da-f]+|\d+);/gi, (_, entity: string) => {
    const codePoint = parseInt(entity.replace(/^x/i, ""), /^x/i.test(entity) ? 16 : 10);
    return codePoint > 0 && codePoint <= 0x10ffff && !(codePoint >= 0xd800 && codePoint <= 0xdfff)
      ? String.fromCodePoint(codePoint) : "\ufffd";
  });
  return decodeEntities(numericDecoded).replace(/\s+/g, " ").trim();
}

function normalized(value: string): string {
  return cleanText(value).normalize("NFKC").toLowerCase();
}

function xUrl(value: string | undefined): URL | null {
  if (!value) return null;
  try {
    const url = new URL(value);
    return (url.protocol === "https:" || url.protocol === "http:") && X_HOSTS.has(url.hostname.toLowerCase())
      ? url : null;
  } catch {
    return null;
  }
}

export function isXSignalPost(post: XSignalPost): boolean {
  return post.feed_key.toLowerCase().startsWith(X_FEED_PREFIX)
    || /^\/(?:[a-z0-9_]{1,15}|i\/web)\/(?:status|statuses)\/\d+(?:\/|$)/i.test(xUrl(post.url)?.pathname ?? "");
}

function canonicalHandle(post: XSignalPost): string | null {
  const url = xUrl(post.url);
  const username = url?.pathname.match(/^\/([a-z0-9_]{1,15})(?:\/(?:status|statuses)\/\d+(?:\/.*)?)?\/?$/i)?.[1];
  if (username && !["i", "home", "search", "explore", "intent"].includes(username.toLowerCase())) {
    return username.toLowerCase();
  }
  const feedHandle = post.feed_key.toLowerCase().startsWith(X_FEED_PREFIX)
    ? post.feed_key.slice(X_FEED_PREFIX.length) : "";
  if (USERNAME.test(feedHandle)) return feedHandle.toLowerCase();
  return post.author.match(/(?:^|[\s(])@([a-z0-9_]{1,15})(?![a-z0-9_])/i)?.[1].toLowerCase() ?? null;
}

function publicationTime(post: XSignalPost): number | null {
  const value = post.published_at ? Date.parse(post.published_at) : NaN;
  return Number.isFinite(value) ? value : null;
}

function validAnalysis(post: XSignalPost): Record<string, unknown> | null {
  const value = post.analysis;
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const analysis = value as Record<string, unknown>;
  return analysis.status === "enriched" && analysis.fallback === false ? analysis : null;
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.map(cleanText).filter(Boolean) : [];
}

function postIdentity(post: XSignalPost): string {
  // Tweet IDs are global; account casing, old hosts and tracking parameters are irrelevant.
  for (const value of [post.url, post.guid]) {
    const id = xUrl(value)?.pathname.match(/\/(?:status|statuses)\/(\d+)(?:\/|$)/i)?.[1];
    if (id) return `tweet:${id}`;
  }
  // Our own ingestion GUID can also identify a tweet whose canonical URL is absent.
  const guidId = post.guid?.match(/^x:[a-z0-9_]+:(\d{10,})$/i)?.[1];
  if (guidId) return `tweet:${guidId}`;
  return post.guid ? `guid:${normalized(post.feed_key)}:${normalized(post.guid)}` : `row:${post.id}`;
}

function matchesQuery(post: XSignalPost, query: string): boolean {
  if (!query) return true;
  const pattern = queryPattern(query);
  const analysis = validAnalysis(post);
  const fields = [post.title, post.description, post.author, canonicalHandle(post) ?? "", ...stringList(analysis?.entities)];
  return fields.some((field) => pattern.test(normalized(field)));
}

function queryPattern(query: string): RegExp {
  const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  // A Unicode word boundary keeps Backpack distinct from backpacking and BackpackPro.
  return new RegExp(`(?<![\\p{L}\\p{N}_])${escaped}(?![\\p{L}\\p{N}_])`, "iu");
}

function excerpt(value: string): string {
  const text = cleanText(value);
  if (text.length <= MAX_EXCERPT) return text;
  const available = text.slice(0, MAX_EXCERPT - 1);
  const sentenceEnds = [...available.matchAll(/[.!?](?=\s|$)/g)];
  const sentenceEnd = sentenceEnds.at(-1)?.index;
  if (sentenceEnd !== undefined && sentenceEnd >= 60) return available.slice(0, sentenceEnd + 1);
  const wordEnd = available.lastIndexOf(" ");
  return `${available.slice(0, wordEnd >= 60 ? wordEnd : available.length).trimEnd()}…`;
}

function contentIdentity(value: string): string {
  return normalized(value).replace(/https?:\/\/\S+/g, "").replace(/[^\p{L}\p{N}]+/gu, " ").trim();
}

function relevantExcerpt(value: string, query: string): string {
  const source = cleanText(value).normalize("NFKC");
  const match = query ? queryPattern(query).exec(source) : null;
  if (!match || match.index + match[0].length <= MAX_EXCERPT) return excerpt(source);
  const preceding = source.slice(0, match.index);
  const sentenceBoundary = [...preceding.matchAll(/[.!?]\s+/g)].at(-1);
  const sentenceStart = sentenceBoundary ? sentenceBoundary.index + sentenceBoundary[0].length : 0;
  // Prefer the whole matching sentence; if it is long, start at a nearby word.
  const start = match.index - sentenceStart < 100 ? sentenceStart : source.lastIndexOf(" ", match.index - 80) + 1;
  return excerpt(`${start > 0 ? "… " : ""}${source.slice(start)}`);
}

export function buildXSignalInsights(
  posts: readonly XSignalPost[],
  { query, window, now }: { query: string; window: XSignalWindow; now: number },
): XSignalInsights {
  const search = normalized(query).replace(/^[@#]/, "");
  const cutoff = window === "all" ? -Infinity : now - (window === "24h" ? DAY_MS : 7 * DAY_MS);
  const candidates = posts.filter((post) => {
    if (!isXSignalPost(post)) return false;
    const published = publicationTime(post);
    // fetched_at is ingestion time and must never make an old or undated post look new.
    if (published === null ? window !== "all" : published > now || published < cutoff) return false;
    return matchesQuery(post, search);
  }).sort((left, right) => {
    const leftTime = publicationTime(left) ?? -Infinity;
    const rightTime = publicationTime(right) ?? -Infinity;
    if (leftTime !== rightTime) return rightTime - leftTime;
    const enrichment = Number(validAnalysis(right) !== null) - Number(validAnalysis(left) !== null);
    return enrichment || right.description.length - left.description.length || left.id - right.id;
  });

  const seenPosts = new Set<string>();
  const matched = candidates.filter((post) => {
    const identity = postIdentity(post);
    if (seenPosts.has(identity)) return false;
    seenPosts.add(identity);
    return true;
  });
  const seenContent = new Set<string>();
  const seenTakeaways = new Set<string>();
  const takeaways: XSignalTakeaway[] = [];
  for (const post of matched) {
    const analysis = validAnalysis(post);
    const thesis = cleanText(analysis?.thesis);
    const source = cleanText(post.description) || cleanText(post.title);
    const text = thesis ? excerpt(thesis) : relevantExcerpt(source, search);
    const sourceIdentity = contentIdentity(source);
    const takeawayIdentity = contentIdentity(text);
    if (!text || !takeawayIdentity || (sourceIdentity && seenContent.has(sourceIdentity)) || seenTakeaways.has(takeawayIdentity)) continue;
    if (sourceIdentity) seenContent.add(sourceIdentity);
    seenTakeaways.add(takeawayIdentity);
    takeaways.push({
      post,
      text,
      kind: thesis ? "analysis" : "excerpt",
      whyItMatters: thesis ? excerpt(stringList(analysis?.why_it_matters)[0] ?? "") || null : null,
      watchNext: thesis ? excerpt(stringList(analysis?.follow_up_questions)[0] ?? "") || null : null,
    });
    if (takeaways.length === 3) break;
  }

  const accounts = new Set(matched.map((post) => canonicalHandle(post) || normalized(post.author) || normalized(post.feed_key)).filter(Boolean));
  return {
    posts: matched,
    takeaways,
    accountCount: accounts.size,
    latestPublishedAt: matched.find((post) => publicationTime(post) !== null)?.published_at ?? null,
    undatedCount: matched.filter((post) => publicationTime(post) === null).length,
  };
}

export function suggestedXSignalSubjects(posts: readonly XSignalPost[]): string[] {
  // Prefer an enriched copy if the same tweet arrived from multiple feeds.
  const xPosts = posts.filter(isXSignalPost).sort((left, right) => Number(validAnalysis(right) !== null) - Number(validAnalysis(left) !== null));
  const subjects = new Map<string, { label: string; count: number }>();
  const seenPosts = new Set<string>();
  for (const post of xPosts) {
    const identity = postIdentity(post);
    if (seenPosts.has(identity)) continue;
    seenPosts.add(identity);
    const entities = new Map(stringList(validAnalysis(post)?.entities).map((label) => [normalized(label), label]));
    for (const [key, label] of entities) {
      const existing = subjects.get(key);
      subjects.set(key, { label: existing?.label ?? label, count: (existing?.count ?? 0) + 1 });
    }
  }
  if (subjects.size) return [...subjects.values()].sort((left, right) => right.count - left.count || left.label.localeCompare(right.label)).slice(0, 5).map(({ label }) => label);
  return [...new Set(xPosts.map(canonicalHandle).filter((handle): handle is string => handle !== null))].slice(0, 5).map((handle) => `@${handle}`);
}

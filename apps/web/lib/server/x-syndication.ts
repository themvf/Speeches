import type { RssArticle } from "@/lib/server/rss-fetcher";

const SYNDICATION_TIMELINE_URL = "https://cdn.syndication.twimg.com/timeline/timeline";
const X_API_BASE_URL = "https://api.x.com/2";
export const X_TIMELINE_FEED_KEY_PREFIX = "x_public_timeline_";
const X_USERNAME_RE = /^[A-Za-z0-9_]{1,15}$/;
const MAX_TIMELINE_ACCOUNTS_PER_RUN = 12;
const DEFAULT_TIMELINE_LIMIT = 20;

type UnknownRecord = Record<string, unknown>;

export type XTimelineFetchResult = {
  username: string;
  articles: RssArticle[];
  fetched: number;
  provider?: "x-api" | "syndication";
  error?: string;
};

type XApiUser = {
  id?: string;
  name?: string;
  username?: string;
};

type XApiTweet = {
  id?: string;
  text?: string;
  created_at?: string;
};

type XApiPayload = {
  data?: unknown;
  errors?: Array<{ title?: string; detail?: string; status?: number }>;
  meta?: { result_count?: number };
};

function text(value: unknown): string {
  return String(value ?? "").replace(/\s+/g, " ").trim();
}

function record(value: unknown): UnknownRecord | null {
  return value && typeof value === "object" && !Array.isArray(value) ? value as UnknownRecord : null;
}

function records(value: unknown): UnknownRecord[] {
  return Array.isArray(value) ? value.map(record).filter(Boolean) as UnknownRecord[] : [];
}

function stripJsonCallback(value: string): string {
  const raw = value.trim();
  const match = raw.match(/^[\w$.]+\(([\s\S]*)\);?$/);
  return match ? match[1].trim() : raw;
}

function hashText(value: string): string {
  let h = 0;
  for (let i = 0; i < value.length; i++) {
    h = (Math.imul(31, h) + value.charCodeAt(i)) | 0;
  }
  return (h >>> 0).toString(16);
}

function cleanTweetText(value: string): string {
  return value
    .replace(/\s*https:\/\/t\.co\/[A-Za-z0-9]+$/g, "")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, "\"")
    .replace(/&#39;/g, "'")
    .replace(/\s+/g, " ")
    .trim();
}

function parseXDate(value: string): Date | null {
  if (!value) return null;
  const parsed = new Date(value);
  return Number.isFinite(parsed.getTime()) ? parsed : null;
}

function xBearerToken(): string {
  return String(process.env.X_BEARER_TOKEN || process.env.TWITTER_BEARER_TOKEN || "").trim();
}

function xApiUrl(path: string, params: Record<string, string>): string {
  const url = new URL(`${X_API_BASE_URL}${path}`);
  for (const [key, value] of Object.entries(params)) {
    if (value) url.searchParams.set(key, value);
  }
  return url.toString();
}

function xApiError(payload: XApiPayload | null, status: number, body: string): string {
  const errors = Array.isArray(payload?.errors) ? payload.errors : [];
  const detail = errors
    .map((error) => text(error.detail) || text(error.title))
    .filter(Boolean)
    .join("; ");
  return detail || body.slice(0, 240) || `X API request failed: ${status}`;
}

async function fetchXApiJson(url: string, token: string, timeoutMs: number): Promise<XApiPayload> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      headers: {
        "Authorization": `Bearer ${token}`,
        "Accept": "application/json",
      },
      cache: "no-store",
      signal: controller.signal,
    });
    const body = await response.text();
    let payload: XApiPayload | null = null;
    try {
      payload = body ? JSON.parse(body) as XApiPayload : {};
    } catch {
      payload = null;
    }
    if (!response.ok || !payload) {
      throw new Error(xApiError(payload, response.status, body));
    }
    return payload;
  } finally {
    clearTimeout(timer);
  }
}

export function normalizeXUsername(input: string): string {
  const raw = String(input || "").trim();
  let cleaned = raw.replace(/^@+/, "").trim();
  if (/^https?:\/\//i.test(cleaned)) {
    try {
      const url = new URL(cleaned);
      if (!/(^|\.)x\.com$/i.test(url.hostname) && !/(^|\.)twitter\.com$/i.test(url.hostname)) {
        return "";
      }
      cleaned = url.pathname.split("/").filter(Boolean)[0] || "";
    } catch {
      return "";
    }
  } else if (/[/?#]/.test(cleaned)) {
    return "";
  }
  return X_USERNAME_RE.test(cleaned) ? cleaned : "";
}

export function parseXTimelineAccounts(raw: string): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const part of String(raw || "").split(/[\s,;]+/g)) {
    const username = normalizeXUsername(part);
    const key = username.toLowerCase();
    if (!username || seen.has(key)) continue;
    seen.add(key);
    out.push(username);
    if (out.length >= MAX_TIMELINE_ACCOUNTS_PER_RUN) break;
  }
  return out;
}

export function xTimelineFeedKey(username: string): string {
  return `${X_TIMELINE_FEED_KEY_PREFIX}${normalizeXUsername(username).toLowerCase()}`;
}

export function xTimelineFeedLabel(username: string): string {
  return `X: @${normalizeXUsername(username)}`;
}

export function xTimelineFeedUrl(username: string): string {
  return `https://x.com/${normalizeXUsername(username)}`;
}

export function isXTimelineFeedKey(feedKey: string): boolean {
  return String(feedKey || "").trim().toLowerCase().startsWith(X_TIMELINE_FEED_KEY_PREFIX);
}

export function xTimelineUsernameFromFeed(feed: { feed_key?: string | null; feed_url?: string | null; label?: string | null }): string {
  const fromUrl = normalizeXUsername(String(feed.feed_url || ""));
  if (fromUrl) return fromUrl;

  const fromLabel = normalizeXUsername(String(feed.label || "").replace(/^X:\s*/i, ""));
  if (fromLabel) return fromLabel;

  const key = String(feed.feed_key || "").trim().toLowerCase();
  if (key.startsWith(X_TIMELINE_FEED_KEY_PREFIX)) {
    return normalizeXUsername(key.slice(X_TIMELINE_FEED_KEY_PREFIX.length));
  }
  return "";
}

function authorFromTweet(tweet: UnknownRecord): UnknownRecord {
  const direct = record(tweet.user);
  if (direct) return direct;

  const core = record(tweet.core);
  const userResults = record(record(core?.user_results)?.result);
  const legacy = record(userResults?.legacy);
  return legacy || {};
}

function legacyTweet(tweet: UnknownRecord): UnknownRecord {
  return record(tweet.legacy) || tweet;
}

function tweetId(tweet: UnknownRecord, legacy: UnknownRecord): string {
  return text(tweet.id_str) || text(legacy.id_str) || text(tweet.rest_id) || text(tweet.id);
}

function tweetText(tweet: UnknownRecord, legacy: UnknownRecord): string {
  return cleanTweetText(text(legacy.full_text) || text(legacy.text) || text(tweet.full_text) || text(tweet.text));
}

function coerceTweetArticle(tweet: UnknownRecord, fallbackUsername: string): RssArticle | null {
  const legacy = legacyTweet(tweet);
  const body = tweetText(tweet, legacy);
  if (!body) return null;

  const author = authorFromTweet(tweet);
  const username = normalizeXUsername(text(author.screen_name) || text(author.userName) || fallbackUsername);
  if (!username) return null;

  const id = tweetId(tweet, legacy);
  const createdAt = text(legacy.created_at) || text(tweet.created_at);
  const publishedAt = parseXDate(createdAt);
  const url = id ? `https://x.com/${username}/status/${id}` : xTimelineFeedUrl(username);
  const authorName = text(author.name);
  const guid = id ? `x:${username.toLowerCase()}:${id}` : `x:${username.toLowerCase()}:${hashText(`${body}:${createdAt}`)}`;

  return {
    guid,
    title: body.slice(0, 180),
    url,
    description: body,
    author: authorName ? `${authorName} (@${username})` : `@${username}`,
    publishedAt,
  };
}

function looksLikeTweet(value: UnknownRecord): boolean {
  const legacy = legacyTweet(value);
  return Boolean(tweetText(value, legacy) && (tweetId(value, legacy) || text(legacy.created_at) || text(value.created_at)));
}

function collectTweetNodes(value: unknown, out: UnknownRecord[], seenObjects: WeakSet<object>): void {
  if (!value || typeof value !== "object") return;
  if (seenObjects.has(value)) return;
  seenObjects.add(value);

  if (Array.isArray(value)) {
    for (const item of value) collectTweetNodes(item, out, seenObjects);
    return;
  }

  const obj = value as UnknownRecord;
  if (looksLikeTweet(obj)) {
    out.push(obj);
  }

  for (const child of Object.values(obj)) {
    collectTweetNodes(child, out, seenObjects);
  }
}

function parseSyndicationPayload(payload: unknown, username: string, limit: number): RssArticle[] {
  const root = record(payload);
  if (!root) return [];

  const instructions = records(record(record(root.data)?.timeline)?.instructions);
  const explicitTweets: UnknownRecord[] = [];
  for (const instruction of instructions) {
    for (const entry of records(instruction.entries)) {
      if (!text(entry.entryId).startsWith("tweet-")) continue;
      const content = record(record(record(entry.content)?.item)?.content);
      const tweet = record(content?.tweet);
      if (tweet) explicitTweets.push(tweet);
    }
  }

  const candidates = explicitTweets.length ? explicitTweets : (() => {
    const found: UnknownRecord[] = [];
    collectTweetNodes(root, found, new WeakSet<object>());
    return found;
  })();

  const seen = new Set<string>();
  const articles: RssArticle[] = [];
  for (const candidate of candidates) {
    const article = coerceTweetArticle(candidate, username);
    if (!article || seen.has(article.guid)) continue;
    seen.add(article.guid);
    articles.push(article);
    if (articles.length >= limit) break;
  }
  return articles;
}

function xApiUserFromPayload(payload: XApiPayload): XApiUser | null {
  const data = record(payload.data);
  if (!data) return null;
  const id = text(data.id);
  const username = normalizeXUsername(text(data.username));
  if (!id || !username) return null;
  return {
    id,
    username,
    name: text(data.name),
  };
}

function xApiTweetsFromPayload(payload: XApiPayload): XApiTweet[] {
  const tweets = Array.isArray(payload.data)
    ? payload.data.map(record).filter((tweet): tweet is UnknownRecord => Boolean(tweet))
    : [];
  return tweets.map((tweet) => ({
        id: text(tweet.id),
        text: cleanTweetText(text(tweet.text)),
        created_at: text(tweet.created_at),
      }));
}

function xApiTweetToArticle(tweet: XApiTweet, user: XApiUser): RssArticle | null {
  const username = normalizeXUsername(user.username || "");
  const id = text(tweet.id);
  const body = cleanTweetText(text(tweet.text));
  if (!username || !id || !body) return null;
  return {
    guid: `x:${username.toLowerCase()}:${id}`,
    title: body.slice(0, 180),
    url: `https://x.com/${username}/status/${id}`,
    description: body,
    author: user.name ? `${user.name} (@${username})` : `@${username}`,
    publishedAt: parseXDate(text(tweet.created_at)),
  };
}

async function fetchXTimelineViaApi(usernameInput: string, opts: { limit?: number; timeoutMs?: number } = {}): Promise<XTimelineFetchResult> {
  const username = normalizeXUsername(usernameInput);
  const token = xBearerToken();
  if (!username) {
    return { username: usernameInput, articles: [], fetched: 0, provider: "x-api", error: "Invalid X username." };
  }
  if (!token) {
    return { username, articles: [], fetched: 0, provider: "x-api", error: "X_BEARER_TOKEN is not configured." };
  }

  const limit = Math.max(1, Math.min(50, Math.round(Number(opts.limit || DEFAULT_TIMELINE_LIMIT))));
  const timeoutMs = opts.timeoutMs || 15_000;

  try {
    const userPayload = await fetchXApiJson(
      xApiUrl(`/users/by/username/${encodeURIComponent(username)}`, {
        "user.fields": "id,name,username",
      }),
      token,
      timeoutMs
    );
    const user = xApiUserFromPayload(userPayload);
    if (!user?.id) {
      return { username, articles: [], fetched: 0, provider: "x-api", error: `X user @${username} was not found.` };
    }

    const maxResults = Math.max(5, Math.min(100, limit));
    const tweetPayload = await fetchXApiJson(
      xApiUrl(`/users/${encodeURIComponent(user.id)}/tweets`, {
        max_results: String(maxResults),
        exclude: "retweets,replies",
        "tweet.fields": "created_at,lang,public_metrics",
      }),
      token,
      timeoutMs
    );
    const seen = new Set<string>();
    const articles: RssArticle[] = [];
    for (const tweet of xApiTweetsFromPayload(tweetPayload)) {
      const article = xApiTweetToArticle(tweet, user);
      if (!article || seen.has(article.guid)) continue;
      seen.add(article.guid);
      articles.push(article);
      if (articles.length >= limit) break;
    }
    return { username: user.username || username, articles, fetched: articles.length, provider: "x-api" };
  } catch (error) {
    return {
      username,
      articles: [],
      fetched: 0,
      provider: "x-api",
      error: error instanceof Error ? error.message : "Unknown X API fetch error.",
    };
  }
}

async function fetchXTimelineViaSyndication(usernameInput: string, opts: { limit?: number; timeoutMs?: number } = {}): Promise<XTimelineFetchResult> {
  const username = normalizeXUsername(usernameInput);
  if (!username) {
    return { username: usernameInput, articles: [], fetched: 0, provider: "syndication", error: "Invalid X username." };
  }

  const limit = Math.max(1, Math.min(50, Math.round(Number(opts.limit || DEFAULT_TIMELINE_LIMIT))));
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), opts.timeoutMs || 15_000);
  const url = new URL(SYNDICATION_TIMELINE_URL);
  url.searchParams.set("screen_name", username);
  url.searchParams.set("count", String(limit));
  url.searchParams.set("lang", "en");
  url.searchParams.set("withReplies", "false");
  url.searchParams.set("withVoice", "false");

  try {
    const response = await fetch(url, {
      headers: {
        "User-Agent": "Mozilla/5.0 (compatible; PolicyHubBot/1.0)",
        "Referer": "https://platform.twitter.com/",
        "Accept": "application/json",
      },
      cache: "no-store",
      signal: controller.signal,
    });
    const body = await response.text();
    if (!response.ok) {
      return { username, articles: [], fetched: 0, provider: "syndication", error: `X syndication fetch failed: ${response.status}` };
    }
    if (!body.trim()) {
      return { username, articles: [], fetched: 0, provider: "syndication", error: "X syndication returned an empty response." };
    }

    const payload = JSON.parse(stripJsonCallback(body)) as unknown;
    const articles = parseSyndicationPayload(payload, username, limit);
    return { username, articles, fetched: articles.length, provider: "syndication" };
  } catch (error) {
    return {
      username,
      articles: [],
      fetched: 0,
      provider: "syndication",
      error: error instanceof Error ? error.message : "Unknown X syndication fetch error.",
    };
  } finally {
    clearTimeout(timer);
  }
}

export async function fetchXTimeline(usernameInput: string, opts: { limit?: number; timeoutMs?: number } = {}): Promise<XTimelineFetchResult> {
  if (!xBearerToken()) {
    return fetchXTimelineViaSyndication(usernameInput, opts);
  }

  const apiResult = await fetchXTimelineViaApi(usernameInput, opts);
  if (apiResult.articles.length > 0 || process.env.X_SYNDICATION_FALLBACK === "0") {
    return apiResult;
  }

  const fallback = await fetchXTimelineViaSyndication(usernameInput, opts);
  if (fallback.articles.length > 0) {
    return fallback;
  }

  return {
    ...apiResult,
    error: [apiResult.error, fallback.error ? `Syndication fallback: ${fallback.error}` : ""].filter(Boolean).join(" "),
  };
}

export async function fetchXTimelineBatch(accounts: string[], opts: { limit?: number; timeoutMs?: number } = {}): Promise<XTimelineFetchResult[]> {
  const normalized = parseXTimelineAccounts(accounts.join(","));
  const results: XTimelineFetchResult[] = [];
  for (const username of normalized) {
    results.push(await fetchXTimeline(username, opts));
  }
  return results;
}

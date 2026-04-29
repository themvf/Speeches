import { neon } from "@neondatabase/serverless";
import type { RssArticle } from "@/lib/server/rss-fetcher";
import { WSJ_FEEDS } from "@/lib/server/rss-fetcher";

export type StoredRssArticle = {
  id: number;
  guid: string;
  feed_key: string;
  title: string;
  url: string;
  description: string;
  author: string;
  published_at: string | null;
  tone_label: "positive" | "neutral" | "negative" | null;
  fetched_at: string;
};

export type RssFeed = {
  id: number;
  label: string;
  feed_url: string;
  feed_key: string;
  active: boolean;
  added_at: string;
};

let _sql: ReturnType<typeof neon> | null = null;

function getSql() {
  if (!_sql) {
    const url = process.env.DATABASE_URL;
    if (!url) throw new Error("DATABASE_URL env var is not set");
    _sql = neon(url);
  }
  return _sql;
}

function deriveFeedKey(feedUrl: string): string {
  try {
    const u = new URL(feedUrl);
    return (u.hostname + u.pathname)
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_|_$/g, "")
      .slice(0, 60);
  } catch {
    return feedUrl.replace(/[^a-z0-9]+/gi, "_").slice(0, 60);
  }
}

export async function ensureSchema(): Promise<void> {
  const sql = getSql();
  await sql`
    CREATE TABLE IF NOT EXISTS rss_articles (
      id           SERIAL PRIMARY KEY,
      guid         TEXT UNIQUE NOT NULL,
      feed_key     TEXT NOT NULL,
      title        TEXT NOT NULL,
      url          TEXT NOT NULL,
      description  TEXT,
      author       TEXT,
      published_at TIMESTAMPTZ,
      tone_label   TEXT CHECK (tone_label IN ('positive','neutral','negative')),
      fetched_at   TIMESTAMPTZ NOT NULL DEFAULT now()
    )
  `;
  await sql`CREATE INDEX IF NOT EXISTS rss_articles_fetched_at ON rss_articles (fetched_at DESC)`;
  await sql`CREATE INDEX IF NOT EXISTS rss_articles_feed_key ON rss_articles (feed_key)`;
  await sql`
    CREATE TABLE IF NOT EXISTS rss_feeds (
      id       SERIAL PRIMARY KEY,
      label    TEXT NOT NULL,
      feed_url TEXT UNIQUE NOT NULL,
      feed_key TEXT UNIQUE NOT NULL,
      active   BOOLEAN NOT NULL DEFAULT true,
      added_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
  `;
  await seedDefaultFeeds(sql);
}

async function seedDefaultFeeds(sql: ReturnType<typeof neon>): Promise<void> {
  const existing = (await sql`SELECT COUNT(*) AS n FROM rss_feeds`) as unknown as { n: string }[];
  if (parseInt(existing[0]?.n ?? "0", 10) > 0) return;
  for (const [key, { label, feedUrl }] of Object.entries(WSJ_FEEDS)) {
    await sql`
      INSERT INTO rss_feeds (label, feed_url, feed_key)
      VALUES (${label}, ${feedUrl}, ${key})
      ON CONFLICT (feed_url) DO NOTHING
    `;
  }
}

export async function getFeeds(onlyActive = false): Promise<RssFeed[]> {
  const sql = getSql();
  const rows = onlyActive
    ? await sql`SELECT * FROM rss_feeds WHERE active = true ORDER BY added_at ASC`
    : await sql`SELECT * FROM rss_feeds ORDER BY added_at ASC`;
  return rows as unknown as RssFeed[];
}

export async function addFeed(label: string, feedUrl: string): Promise<RssFeed> {
  const sql = getSql();
  const feedKey = deriveFeedKey(feedUrl);
  const rows = (await sql`
    INSERT INTO rss_feeds (label, feed_url, feed_key)
    VALUES (${label.trim()}, ${feedUrl.trim()}, ${feedKey})
    ON CONFLICT (feed_url) DO UPDATE SET label = EXCLUDED.label, active = true
    RETURNING *
  `) as unknown as RssFeed[];
  return rows[0];
}

export async function toggleFeed(id: number, active: boolean): Promise<void> {
  const sql = getSql();
  await sql`UPDATE rss_feeds SET active = ${active} WHERE id = ${id}`;
}

export async function deleteFeed(id: number): Promise<void> {
  const sql = getSql();
  await sql`DELETE FROM rss_feeds WHERE id = ${id}`;
}

export async function upsertRssArticles(articles: RssArticle[], feedKey: string): Promise<number> {
  if (articles.length === 0) return 0;
  const sql = getSql();
  let inserted = 0;
  for (const a of articles) {
    const result = (await sql`
      INSERT INTO rss_articles (guid, feed_key, title, url, description, author, published_at)
      VALUES (
        ${a.guid},
        ${feedKey},
        ${a.title},
        ${a.url},
        ${a.description ?? ""},
        ${a.author ?? ""},
        ${a.publishedAt ? a.publishedAt.toISOString() : null}
      )
      ON CONFLICT (guid) DO NOTHING
      RETURNING id
    `) as unknown as { id: number }[];
    if (result.length > 0) inserted++;
  }
  return inserted;
}

export async function getRecentArticles(opts: {
  limit?: number;
  feedKey?: string;
  since?: Date;
} = {}): Promise<StoredRssArticle[]> {
  const sql = getSql();
  const limit = opts.limit ?? 50;
  const feedKey = opts.feedKey ?? null;
  const since = opts.since ? opts.since.toISOString() : null;

  let query;
  if (feedKey && since) {
    query = sql`
      SELECT * FROM rss_articles
      WHERE feed_key = ${feedKey} AND fetched_at > ${since}
      ORDER BY fetched_at DESC
      LIMIT ${limit}
    `;
  } else if (feedKey) {
    query = sql`
      SELECT * FROM rss_articles
      WHERE feed_key = ${feedKey}
      ORDER BY fetched_at DESC
      LIMIT ${limit}
    `;
  } else if (since) {
    query = sql`
      SELECT * FROM rss_articles
      WHERE fetched_at > ${since}
      ORDER BY fetched_at DESC
      LIMIT ${limit}
    `;
  } else {
    query = sql`
      SELECT * FROM rss_articles
      ORDER BY fetched_at DESC
      LIMIT ${limit}
    `;
  }
  return (await query) as unknown as StoredRssArticle[];
}

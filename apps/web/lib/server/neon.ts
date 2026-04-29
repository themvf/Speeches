import { neon } from "@neondatabase/serverless";
import type { RssArticle } from "@/lib/server/rss-fetcher";

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

let _sql: ReturnType<typeof neon> | null = null;

function getSql() {
  if (!_sql) {
    const url = process.env.DATABASE_URL;
    if (!url) throw new Error("DATABASE_URL env var is not set");
    _sql = neon(url);
  }
  return _sql;
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

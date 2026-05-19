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

export type StoredRssTopicRule = {
  id: number;
  topic_key: string;
  label: string;
  keywords: string;
  active: boolean;
  sort_order: number;
  updated_at: string;
};

export type RecapSource = { title: string; url: string; source_type: "article" | "document"; source_kind?: string; speaker?: string };

export type DailyRecapRow = {
  id: number;
  recap_date: string;
  topic_key: string;
  topic_label: string;
  summary: string;
  article_count: number;
  positive_count: number;
  negative_count: number;
  neutral_count: number;
  sources: RecapSource[];
  generated_at: string;
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

const DEFAULT_TOPIC_RULES: Array<{
  topicKey: string;
  label: string;
  keywords: string;
  sortOrder: number;
}> = [
  {
    topicKey: "SECURITIES_REGULATION",
    label: "Securities Regulation",
    keywords: "sec, securities, disclosure, investor, exchange, registration",
    sortOrder: 10,
  },
  {
    topicKey: "CAPITAL_FORMATION",
    label: "Capital Formation",
    keywords: "ipo, spac, capital, offering, funding, venture, startup",
    sortOrder: 20,
  },
  {
    topicKey: "AML",
    label: "AML",
    keywords: "aml, money laundering, sanctions, bsa, finra, anti-money",
    sortOrder: 30,
  },
  {
    topicKey: "ENFORCEMENT",
    label: "Enforcement",
    keywords: "enforcement, fine, penalty, fraud, charges, lawsuit, settlement, indictment",
    sortOrder: 40,
  },
  {
    topicKey: "AI_TECH",
    label: "AI & Tech",
    keywords: "ai, artificial intelligence, machine learning, technology, fintech, automation",
    sortOrder: 50,
  },
  {
    topicKey: "CRYPTO",
    label: "Crypto",
    keywords: "crypto, bitcoin, blockchain, digital asset, stablecoin, ethereum, defi, nft",
    sortOrder: 60,
  },
  {
    topicKey: "CREDIT_MARKETS",
    label: "Credit Markets",
    keywords: "credit, bond, debt, yield, loan, lending, mortgage, default",
    sortOrder: 70,
  },
  {
    topicKey: "FINANCIAL_MARKETS",
    label: "Financial Markets",
    keywords: "market, stock, equity, trading, volatility, s&p, nasdaq, dow",
    sortOrder: 80,
  },
  {
    topicKey: "ECONOMIC_GROWTH",
    label: "Economic Growth",
    keywords: "economy, gdp, growth, inflation, fed, federal reserve, recession, jobs",
    sortOrder: 90,
  },
  {
    topicKey: "PREDICTION_MARKETS",
    label: "Prediction Markets",
    keywords: "prediction market, polymarket, kalshi, betting market, forecast, odds, contract",
    sortOrder: 100,
  },
];

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

function inferToneLabel(
  title: string,
  description: string,
  feedKey: string
): "positive" | "neutral" | "negative" {
  const titleLower = title.toLowerCase();
  const descLower = description.toLowerCase();
  const fullLower = `${titleLower} ${descLower}`;

  const weightedSignals: Array<{ label: "positive" | "negative"; weight: number; patterns: string[] }> = [
    {
      label: "positive",
      weight: 3,
      patterns: [
        "landmark victory",
        "major victory",
        "strikes a blow against",
        "hailed",
        "praised",
        "applauded",
        "breakthrough",
        "surges",
        "boosts",
        "eases fears",
        "beats expectations",
        "on track",
        "to reach ballot",
      ],
    },
    {
      label: "negative",
      weight: 3,
      patterns: [
        "beware",
        "murderous",
        "evil",
        "rackets",
        "frustrated",
        "legal threats",
        "less leverage than expected",
        "reckless",
        "dangerously",
        "alarming",
        "overreaching",
        "disastrous",
        "slammed",
        "blasted",
        "crisis",
        "collapse",
      ],
    },
  ];

  let score = 0;
  for (const signal of weightedSignals) {
    for (const pattern of signal.patterns) {
      const inTitle = titleLower.includes(pattern);
      const inDesc = descLower.includes(pattern);
      if (!inTitle && !inDesc) continue;
      const delta = signal.weight * (inTitle ? 2 : 1);
      score += signal.label === "positive" ? delta : -delta;
    }
  }

  const opinionBoostNegative = [
    "the left",
    "the right",
    "shouldn't",
    "no evil",
    "casualty",
    "threat",
    "war",
    "fight",
  ];
  const opinionBoostPositive = ["win", "victory", "success", "benefit", "improves"];

  if (feedKey === "wsj_opinion") {
    score += opinionBoostPositive.filter((pattern) => fullLower.includes(pattern)).length;
    score -= opinionBoostNegative.filter((pattern) => fullLower.includes(pattern)).length;
    if (score === 0 && (titleLower.includes("?") || descLower.includes("?"))) {
      score -= 1;
    }
  }

  if (score >= 2) return "positive";
  if (score <= -2) return "negative";
  return "neutral";
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
  await sql`
    CREATE TABLE IF NOT EXISTS rss_topic_rules (
      id         SERIAL PRIMARY KEY,
      topic_key  TEXT UNIQUE NOT NULL,
      label      TEXT NOT NULL,
      keywords   TEXT NOT NULL DEFAULT '',
      active     BOOLEAN NOT NULL DEFAULT true,
      sort_order INTEGER NOT NULL DEFAULT 100,
      updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
  `;
  await sql`
    CREATE TABLE IF NOT EXISTS recap_settings (
      id         SERIAL PRIMARY KEY,
      topic_keys TEXT NOT NULL DEFAULT '',
      updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
  `;
  await sql`
    CREATE TABLE IF NOT EXISTS daily_recaps (
      id             SERIAL PRIMARY KEY,
      recap_date     DATE NOT NULL,
      topic_key      TEXT NOT NULL,
      topic_label    TEXT NOT NULL,
      summary        TEXT NOT NULL,
      article_count  INTEGER NOT NULL DEFAULT 0,
      positive_count INTEGER NOT NULL DEFAULT 0,
      negative_count INTEGER NOT NULL DEFAULT 0,
      neutral_count  INTEGER NOT NULL DEFAULT 0,
      sources        TEXT NOT NULL DEFAULT '[]',
      generated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
      UNIQUE (recap_date, topic_key)
    )
  `;
  await sql`ALTER TABLE daily_recaps ADD COLUMN IF NOT EXISTS sources TEXT NOT NULL DEFAULT '[]'`;
  await seedDefaultFeeds(sql);
  await seedDefaultTopicRules(sql);
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

async function seedDefaultTopicRules(sql: ReturnType<typeof neon>): Promise<void> {
  const existing = (await sql`SELECT COUNT(*) AS n FROM rss_topic_rules`) as unknown as { n: string }[];
  if (parseInt(existing[0]?.n ?? "0", 10) > 0) return;
  for (const rule of DEFAULT_TOPIC_RULES) {
    await sql`
      INSERT INTO rss_topic_rules (topic_key, label, keywords, active, sort_order)
      VALUES (${rule.topicKey}, ${rule.label}, ${rule.keywords}, true, ${rule.sortOrder})
      ON CONFLICT (topic_key) DO NOTHING
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

export async function getTopicRules(onlyActive = true): Promise<StoredRssTopicRule[]> {
  await ensureSchema();
  const sql = getSql();
  const rows = onlyActive
    ? await sql`
        SELECT * FROM rss_topic_rules
        WHERE active = true
        ORDER BY sort_order ASC, label ASC
      `
    : await sql`
        SELECT * FROM rss_topic_rules
        ORDER BY sort_order ASC, label ASC
      `;
  return rows as unknown as StoredRssTopicRule[];
}

export async function addTopicRule(data: {
  topicKey: string;
  label: string;
  keywords: string;
  active: boolean;
  sortOrder: number;
}): Promise<StoredRssTopicRule> {
  const sql = getSql();
  const rows = (await sql`
    INSERT INTO rss_topic_rules (topic_key, label, keywords, active, sort_order)
    VALUES (${data.topicKey}, ${data.label}, ${data.keywords}, ${data.active}, ${data.sortOrder})
    RETURNING *
  `) as unknown as StoredRssTopicRule[];
  return rows[0];
}

export async function updateTopicRule(
  id: number,
  data: { label?: string; keywords?: string; active?: boolean; sortOrder?: number }
): Promise<void> {
  const sql = getSql();
  await sql`
    UPDATE rss_topic_rules SET
      label      = COALESCE(${data.label ?? null}, label),
      keywords   = COALESCE(${data.keywords ?? null}, keywords),
      active     = COALESCE(${data.active ?? null}, active),
      sort_order = COALESCE(${data.sortOrder ?? null}, sort_order),
      updated_at = NOW()
    WHERE id = ${id}
  `;
}

export async function deleteTopicRule(id: number): Promise<void> {
  const sql = getSql();
  await sql`DELETE FROM rss_topic_rules WHERE id = ${id}`;
}

export async function upsertRssArticles(articles: RssArticle[], feedKey: string): Promise<number> {
  if (articles.length === 0) return 0;
  const sql = getSql();
  let inserted = 0;
  for (const a of articles) {
    const toneLabel = inferToneLabel(a.title, a.description ?? "", feedKey);
    const result = (await sql`
      INSERT INTO rss_articles (guid, feed_key, title, url, description, author, published_at, tone_label)
      VALUES (
        ${a.guid},
        ${feedKey},
        ${a.title},
        ${a.url},
        ${a.description ?? ""},
        ${a.author ?? ""},
        ${a.publishedAt ? a.publishedAt.toISOString() : null},
        ${toneLabel}
      )
      ON CONFLICT (guid) DO UPDATE
      SET title = EXCLUDED.title,
          url = EXCLUDED.url,
          description = EXCLUDED.description,
          author = EXCLUDED.author,
          published_at = EXCLUDED.published_at,
          feed_key = EXCLUDED.feed_key,
          tone_label = CASE
            WHEN rss_articles.tone_label IS NULL THEN EXCLUDED.tone_label
            WHEN rss_articles.tone_label = 'neutral' AND EXCLUDED.tone_label <> 'neutral' THEN EXCLUDED.tone_label
            ELSE rss_articles.tone_label
          END
      RETURNING id, (xmax = 0) AS inserted
    `) as unknown as { id: number; inserted: boolean }[];
    if (result[0]?.inserted) inserted++;
  }
  return inserted;
}

export async function getRecentArticles(opts: {
  limit?: number;
  feedKey?: string;
  since?: Date;
  until?: Date;
} = {}): Promise<StoredRssArticle[]> {
  const sql = getSql();
  const limit = opts.limit ?? 50;
  const feedKey = opts.feedKey ?? null;
  const since = opts.since ? opts.since.toISOString() : null;
  const until = opts.until ? opts.until.toISOString() : null;

  let query;
  if (feedKey && since && until) {
    query = sql`SELECT * FROM rss_articles WHERE feed_key = ${feedKey} AND COALESCE(published_at, fetched_at) > ${since} AND COALESCE(published_at, fetched_at) <= ${until} ORDER BY COALESCE(published_at, fetched_at) DESC LIMIT ${limit}`;
  } else if (feedKey && since) {
    query = sql`SELECT * FROM rss_articles WHERE feed_key = ${feedKey} AND COALESCE(published_at, fetched_at) > ${since} ORDER BY COALESCE(published_at, fetched_at) DESC LIMIT ${limit}`;
  } else if (feedKey) {
    query = sql`SELECT * FROM rss_articles WHERE feed_key = ${feedKey} ORDER BY fetched_at DESC LIMIT ${limit}`;
  } else if (since && until) {
    query = sql`SELECT * FROM rss_articles WHERE COALESCE(published_at, fetched_at) > ${since} AND COALESCE(published_at, fetched_at) <= ${until} ORDER BY COALESCE(published_at, fetched_at) DESC LIMIT ${limit}`;
  } else if (since) {
    query = sql`SELECT * FROM rss_articles WHERE COALESCE(published_at, fetched_at) > ${since} ORDER BY COALESCE(published_at, fetched_at) DESC LIMIT ${limit}`;
  } else {
    query = sql`SELECT * FROM rss_articles ORDER BY fetched_at DESC LIMIT ${limit}`;
  }
  return (await query) as unknown as StoredRssArticle[];
}

export async function getRecapSettings(): Promise<string[]> {
  await ensureSchema();
  const sql = getSql();
  const rows = (await sql`SELECT topic_keys FROM recap_settings LIMIT 1`) as unknown as { topic_keys: string }[];
  const raw = rows[0]?.topic_keys ?? "";
  return raw ? raw.split(",").map((k) => k.trim()).filter(Boolean) : [];
}

export async function saveRecapSettings(topicKeys: string[]): Promise<void> {
  await ensureSchema();
  const sql = getSql();
  const value = topicKeys.join(",");
  const existing = (await sql`SELECT id FROM recap_settings LIMIT 1`) as unknown as { id: number }[];
  if (existing.length > 0) {
    await sql`UPDATE recap_settings SET topic_keys = ${value}, updated_at = now() WHERE id = ${existing[0].id}`;
  } else {
    await sql`INSERT INTO recap_settings (topic_keys) VALUES (${value})`;
  }
}

export async function getTodaysRecap(date?: string): Promise<DailyRecapRow[]> {
  await ensureSchema();
  const sql = getSql();
  const rows = date
    ? (await sql`
        SELECT * FROM daily_recaps
        WHERE recap_date = ${date}::date
        ORDER BY topic_label ASC
      `) as unknown as (Omit<DailyRecapRow, "sources"> & { sources: string })[]
    : (await sql`
        SELECT * FROM daily_recaps
        WHERE recap_date = CURRENT_DATE
        ORDER BY topic_label ASC
      `) as unknown as (Omit<DailyRecapRow, "sources"> & { sources: string })[];
  return rows.map((r) => {
    let sources: RecapSource[] = [];
    try {
      sources = JSON.parse(r.sources || "[]") as RecapSource[];
    } catch (err) {
      console.error(`[neon] getTodaysRecap: failed to parse sources for topic ${r.topic_key}:`, err);
    }
    return { ...r, sources };
  });
}

export async function saveRecapRows(rows: Omit<DailyRecapRow, "id" | "generated_at">[]): Promise<void> {
  await ensureSchema();
  const sql = getSql();
  for (const row of rows) {
    const sourcesJson = JSON.stringify(row.sources ?? []);
    await sql`
      INSERT INTO daily_recaps (recap_date, topic_key, topic_label, summary, article_count, positive_count, negative_count, neutral_count, sources)
      VALUES (${row.recap_date}, ${row.topic_key}, ${row.topic_label}, ${row.summary}, ${row.article_count}, ${row.positive_count}, ${row.negative_count}, ${row.neutral_count}, ${sourcesJson})
      ON CONFLICT (recap_date, topic_key) DO UPDATE
      SET summary        = EXCLUDED.summary,
          topic_label    = EXCLUDED.topic_label,
          article_count  = EXCLUDED.article_count,
          positive_count = EXCLUDED.positive_count,
          negative_count = EXCLUDED.negative_count,
          neutral_count  = EXCLUDED.neutral_count,
          sources        = EXCLUDED.sources,
          generated_at   = now()
    `;
  }
}

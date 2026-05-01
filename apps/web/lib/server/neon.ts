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

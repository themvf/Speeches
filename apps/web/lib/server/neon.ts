import { neon } from "@neondatabase/serverless";
import { createHash } from "crypto";
import type { FeedAnalysis } from "@/lib/server/feed-analysis";
import type { RssArticle } from "@/lib/server/rss-fetcher";
import { DEFAULT_RSS_FEEDS } from "@/lib/server/rss-fetcher";
import { TOPIC_RULE_RECOMMENDATIONS, formatTopicRuleKeywords } from "@/lib/topic-rule-recommendations";

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
  analysis?: StoredRssArticleAnalysis | null;
};

export type MentionType = "keyword" | "individual" | "entity" | "topic";

export type StoredRssArticleAnalysis = {
  article_id: number;
  guid: string;
  source_hash: string;
  status: "pending" | "enriched" | "failed" | "stale";
  model: string;
  generated_at: string;
  thesis: string;
  why_it_matters: string[];
  risk_signals: string[];
  follow_up_questions: string[];
  keywords: string[];
  individuals: string[];
  entities: string[];
  topics: string[];
  analysis_text: string;
  fallback: boolean;
  error: string;
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
  refresh_interval_minutes: number;
  last_refresh_at: string | null;
  added_at: string;
};

let _sql: ReturnType<typeof neon> | null = null;

const DEFAULT_TOPIC_RULES = TOPIC_RULE_RECOMMENDATIONS.map((rule) => ({
  topicKey: rule.topicKey,
  label: rule.label,
  keywords: formatTopicRuleKeywords(rule.suggestedKeywords),
  sortOrder: rule.sortOrder,
}));

const TOPIC_TAXONOMY_UPSERT_KEYS = new Set([
  "AI_TECH",
  "PRE_IPO",
  "PREDICTION_MARKETS",
  "TECH",
  "COMMODITIES_ENERGY_MARKETS",
  "GEOPOLITICAL_TRADE_RISK",
  "SRO_RULEMAKING_ARBITRATION",
  "BANKING_PAYMENTS",
  "CONSUMER_PROTECTION_DECEPTIVE_PRACTICES",
  "DATA_PRIVACY_DIGITAL_IDENTITY",
  "INVESTMENT_PRODUCTS_DERIVATIVES",
]);

const DEPRECATED_TOPIC_RULE_KEYS = ["PREMARKETS", "AI"] as const;

const DEPRECATED_RSS_FEED_KEYS = [
  "bleepingcomputer",
  "the_hacker_news",
  "dark_reading",
  "securityweek",
  "microsoft_security_blog",
] as const;

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

function textArray(value: unknown, maxItems = 30): string[] {
  if (!Array.isArray(value)) return [];
  const seen = new Set<string>();
  const out: string[] = [];
  for (const item of value) {
    const text = String(item ?? "").replace(/\s+/g, " ").trim().slice(0, 160);
    const key = text.toLowerCase();
    if (!text || seen.has(key)) continue;
    seen.add(key);
    out.push(text);
    if (out.length >= maxItems) break;
  }
  return out;
}

function normalizeMention(value: string): string {
  return String(value || "")
    .toLowerCase()
    .replace(/['"“”‘’]/g, "")
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function rssArticleSourceHash(article: Pick<StoredRssArticle, "title" | "description" | "url" | "published_at" | "feed_key">): string {
  return createHash("sha256")
    .update([
      article.feed_key || "",
      article.title || "",
      article.description || "",
      article.url || "",
      article.published_at || "",
    ].join("\n"))
    .digest("hex");
}

function buildAnalysisText(article: StoredRssArticle, analysis: FeedAnalysis, topics: string[]): string {
  return [
    `Title: ${article.title}`,
    `Source: ${article.feed_key}`,
    article.author ? `Author: ${article.author}` : "",
    article.published_at ? `Published: ${article.published_at}` : "",
    article.url ? `URL: ${article.url}` : "",
    topics.length ? `Topics: ${topics.join(", ")}` : "",
    "",
    `Thesis: ${analysis.thesis}`,
    "",
    "Why it matters:",
    ...analysis.why_it_matters.map((item) => `- ${item}`),
    "",
    "Risk signals:",
    ...analysis.risk_signals.map((item) => `- ${item}`),
    "",
    "Follow-up:",
    ...analysis.follow_up_questions.map((item) => `- ${item}`),
    "",
    analysis.keywords.length ? `Keywords: ${analysis.keywords.join(", ")}` : "",
    analysis.individuals.length ? `Individuals: ${analysis.individuals.join(", ")}` : "",
    analysis.entities.length ? `Entities: ${analysis.entities.join(", ")}` : "",
    "",
    article.description || "",
  ].filter(Boolean).join("\n");
}

function normalizeAnalysisRow(row: Record<string, unknown> | null | undefined): StoredRssArticleAnalysis | null {
  if (!row) return null;
  return {
    article_id: Number(row.article_id || 0),
    guid: String(row.guid || ""),
    source_hash: String(row.source_hash || ""),
    status: String(row.status || "pending") as StoredRssArticleAnalysis["status"],
    model: String(row.model || ""),
    generated_at: String(row.generated_at || ""),
    thesis: String(row.thesis || ""),
    why_it_matters: textArray(row.why_it_matters, 8),
    risk_signals: textArray(row.risk_signals, 8),
    follow_up_questions: textArray(row.follow_up_questions, 8),
    keywords: textArray(row.keywords, 20),
    individuals: textArray(row.individuals, 20),
    entities: textArray(row.entities, 30),
    topics: textArray(row.topics, 20),
    analysis_text: String(row.analysis_text || ""),
    fallback: Boolean(row.fallback),
    error: String(row.error || ""),
  };
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
  await sql`CREATE INDEX IF NOT EXISTS rss_articles_published_at ON rss_articles (published_at DESC)`;
  await sql`
    CREATE TABLE IF NOT EXISTS rss_article_analysis (
      article_id          INTEGER PRIMARY KEY REFERENCES rss_articles(id) ON DELETE CASCADE,
      guid                TEXT NOT NULL,
      source_hash         TEXT NOT NULL,
      status              TEXT NOT NULL DEFAULT 'pending',
      model               TEXT NOT NULL DEFAULT '',
      generated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
      thesis              TEXT NOT NULL DEFAULT '',
      why_it_matters      JSONB NOT NULL DEFAULT '[]'::jsonb,
      risk_signals        JSONB NOT NULL DEFAULT '[]'::jsonb,
      follow_up_questions JSONB NOT NULL DEFAULT '[]'::jsonb,
      keywords            TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
      individuals         TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
      entities            TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
      topics              TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
      analysis_text       TEXT NOT NULL DEFAULT '',
      fallback            BOOLEAN NOT NULL DEFAULT false,
      error               TEXT NOT NULL DEFAULT ''
    )
  `;
  await sql`CREATE INDEX IF NOT EXISTS rss_article_analysis_guid ON rss_article_analysis (guid)`;
  await sql`CREATE INDEX IF NOT EXISTS rss_article_analysis_status ON rss_article_analysis (status)`;
  await sql`CREATE INDEX IF NOT EXISTS rss_article_analysis_keywords ON rss_article_analysis USING GIN (keywords)`;
  await sql`CREATE INDEX IF NOT EXISTS rss_article_analysis_individuals ON rss_article_analysis USING GIN (individuals)`;
  await sql`CREATE INDEX IF NOT EXISTS rss_article_analysis_entities ON rss_article_analysis USING GIN (entities)`;
  await sql`
    CREATE TABLE IF NOT EXISTS intelligence_mentions (
      id               BIGSERIAL PRIMARY KEY,
      source_type      TEXT NOT NULL,
      source_id        TEXT NOT NULL,
      mention_type     TEXT NOT NULL,
      value            TEXT NOT NULL,
      normalized_value TEXT NOT NULL,
      confidence       DOUBLE PRECISION NOT NULL DEFAULT 1,
      generated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
      UNIQUE (source_type, source_id, mention_type, normalized_value)
    )
  `;
  await sql`CREATE INDEX IF NOT EXISTS intelligence_mentions_lookup ON intelligence_mentions (mention_type, normalized_value)`;
  await sql`CREATE INDEX IF NOT EXISTS intelligence_mentions_source ON intelligence_mentions (source_type, source_id)`;
  await sql`
    CREATE TABLE IF NOT EXISTS rss_feeds (
      id                       SERIAL PRIMARY KEY,
      label                    TEXT NOT NULL,
      feed_url                 TEXT UNIQUE NOT NULL,
      feed_key                 TEXT UNIQUE NOT NULL,
      active                   BOOLEAN NOT NULL DEFAULT true,
      refresh_interval_minutes INTEGER NOT NULL DEFAULT 10,
      last_refresh_at          TIMESTAMPTZ,
      added_at                 TIMESTAMPTZ NOT NULL DEFAULT now()
    )
  `;
  await sql`ALTER TABLE rss_feeds ADD COLUMN IF NOT EXISTS refresh_interval_minutes INTEGER NOT NULL DEFAULT 10`;
  await sql`ALTER TABLE rss_feeds ADD COLUMN IF NOT EXISTS last_refresh_at TIMESTAMPTZ`;
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
  await applyFeedSourceMigrations(sql);
  await seedDefaultTopicRules(sql);
  await applyTopicTaxonomyMigrations(sql);
}

async function seedDefaultFeeds(sql: ReturnType<typeof neon>): Promise<void> {
  for (const [key, { label, feedUrl, refreshIntervalMinutes }] of Object.entries(DEFAULT_RSS_FEEDS)) {
    const intervalMinutes = Math.max(1, Math.round(Number(refreshIntervalMinutes || 10)));
    await sql`
      INSERT INTO rss_feeds (label, feed_url, feed_key, refresh_interval_minutes)
      VALUES (${label}, ${feedUrl}, ${key}, ${intervalMinutes})
      ON CONFLICT (feed_url) DO UPDATE SET
        label = EXCLUDED.label,
        feed_key = EXCLUDED.feed_key,
        refresh_interval_minutes = EXCLUDED.refresh_interval_minutes
    `;
  }
}

async function applyFeedSourceMigrations(sql: ReturnType<typeof neon>): Promise<void> {
  await sql`
    UPDATE rss_feeds
    SET active = false
    WHERE feed_key = ANY(${DEPRECATED_RSS_FEED_KEYS})
       OR feed_url = ANY(${[
         "https://www.bleepingcomputer.com/feed/",
         "https://feeds.feedburner.com/TheHackersNews",
         "https://www.darkreading.com/rss.xml",
         "https://www.securityweek.com/feed/",
         "https://www.microsoft.com/en-us/security/blog/feed/",
       ]})
  `;
}

async function seedDefaultTopicRules(sql: ReturnType<typeof neon>): Promise<void> {
  for (const rule of DEFAULT_TOPIC_RULES) {
    await sql`
      INSERT INTO rss_topic_rules (topic_key, label, keywords, active, sort_order)
      VALUES (${rule.topicKey}, ${rule.label}, ${rule.keywords}, true, ${rule.sortOrder})
      ON CONFLICT (topic_key) DO NOTHING
    `;
  }
}

async function applyTopicTaxonomyMigrations(sql: ReturnType<typeof neon>): Promise<void> {
  for (const rule of DEFAULT_TOPIC_RULES.filter((item) => TOPIC_TAXONOMY_UPSERT_KEYS.has(item.topicKey))) {
    await sql`
      INSERT INTO rss_topic_rules (topic_key, label, keywords, active, sort_order, updated_at)
      VALUES (${rule.topicKey}, ${rule.label}, ${rule.keywords}, true, ${rule.sortOrder}, NOW())
      ON CONFLICT (topic_key) DO UPDATE
      SET
        label = EXCLUDED.label,
        keywords = EXCLUDED.keywords,
        active = true,
        sort_order = EXCLUDED.sort_order,
        updated_at = NOW()
    `;
  }

  await sql`
    UPDATE rss_topic_rules
    SET active = false, updated_at = NOW()
    WHERE topic_key = ANY(${DEPRECATED_TOPIC_RULE_KEYS})
      AND active = true
  `;
}

export async function getFeeds(onlyActive = false, opts: { dueOnly?: boolean } = {}): Promise<RssFeed[]> {
  await ensureSchema();
  const sql = getSql();
  let rows;
  if (onlyActive && opts.dueOnly) {
    rows = await sql`
      SELECT *
      FROM rss_feeds
      WHERE active = true
        AND (
          last_refresh_at IS NULL
          OR last_refresh_at <= now() - (GREATEST(refresh_interval_minutes, 1) * INTERVAL '1 minute')
        )
      ORDER BY last_refresh_at ASC NULLS FIRST, added_at ASC
    `;
  } else {
    rows = onlyActive
      ? await sql`SELECT * FROM rss_feeds WHERE active = true ORDER BY added_at ASC`
      : await sql`SELECT * FROM rss_feeds ORDER BY added_at ASC`;
  }
  return rows as unknown as RssFeed[];
}

export async function addFeed(label: string, feedUrl: string, refreshIntervalMinutes = 10): Promise<RssFeed> {
  const sql = getSql();
  const feedKey = deriveFeedKey(feedUrl);
  const intervalMinutes = Math.max(1, Math.round(Number(refreshIntervalMinutes || 10)));
  const rows = (await sql`
    INSERT INTO rss_feeds (label, feed_url, feed_key, refresh_interval_minutes)
    VALUES (${label.trim()}, ${feedUrl.trim()}, ${feedKey}, ${intervalMinutes})
    ON CONFLICT (feed_url) DO UPDATE SET
      label = EXCLUDED.label,
      refresh_interval_minutes = EXCLUDED.refresh_interval_minutes,
      active = true
    RETURNING *
  `) as unknown as RssFeed[];
  return rows[0];
}

export async function markFeedRefreshed(feedKey: string): Promise<void> {
  const sql = getSql();
  await sql`UPDATE rss_feeds SET last_refresh_at = now() WHERE feed_key = ${feedKey}`;
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
          published_at = COALESCE(EXCLUDED.published_at, rss_articles.published_at),
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

export async function getRssArticleById(articleId: number): Promise<StoredRssArticle | null> {
  await ensureSchema();
  const sql = getSql();
  const rows = (await sql`
    SELECT a.*, to_jsonb(ra.*) AS analysis
    FROM rss_articles a
    LEFT JOIN rss_article_analysis ra ON ra.article_id = a.id
    WHERE a.id = ${articleId}
    LIMIT 1
  `) as unknown as Array<StoredRssArticle & { analysis?: Record<string, unknown> | null }>;
  const row = rows[0];
  return row ? { ...row, analysis: normalizeAnalysisRow(row.analysis) } : null;
}

export async function getRssArticlesNeedingAnalysis(limit = 10): Promise<StoredRssArticle[]> {
  await ensureSchema();
  const sql = getSql();
  const cappedLimit = Math.max(1, Math.min(100, limit));
  const refreshForDeepSeek =
    String(process.env.FEED_ANALYSIS_PROVIDER || "").trim().toLowerCase() !== "openai";
  const rows = (await sql`
    SELECT a.*, to_jsonb(ra.*) AS analysis
    FROM rss_articles a
    LEFT JOIN rss_article_analysis ra ON ra.article_id = a.id
    WHERE ra.article_id IS NULL
       OR ra.status IN ('pending', 'failed', 'stale')
       OR (
         ${refreshForDeepSeek}
         AND ra.status = 'enriched'
         AND (
           ra.fallback = true
           OR ra.model NOT ILIKE 'deepseek%'
           OR length(COALESCE(ra.thesis, '')) < 40
           OR jsonb_array_length(COALESCE(ra.why_it_matters, '[]'::jsonb)) < 2
           OR jsonb_array_length(COALESCE(ra.risk_signals, '[]'::jsonb)) < 2
           OR jsonb_array_length(COALESCE(ra.follow_up_questions, '[]'::jsonb)) < 2
         )
       )
    ORDER BY COALESCE(a.published_at, a.fetched_at) DESC
    LIMIT ${cappedLimit}
  `) as unknown as Array<StoredRssArticle & { analysis?: Record<string, unknown> | null }>;
  const direct = rows.map((row) => ({ ...row, analysis: normalizeAnalysisRow(row.analysis) }));
  if (direct.length >= cappedLimit) return direct;

  const recentRows = (await sql`
    SELECT a.*, to_jsonb(ra.*) AS analysis
    FROM rss_articles a
    LEFT JOIN rss_article_analysis ra ON ra.article_id = a.id
    WHERE ra.article_id IS NOT NULL
    ORDER BY COALESCE(a.published_at, a.fetched_at) DESC
    LIMIT ${Math.max(cappedLimit * 4, 25)}
  `) as unknown as Array<StoredRssArticle & { analysis?: Record<string, unknown> | null }>;
  const stale = recentRows
    .map((row) => ({ ...row, analysis: normalizeAnalysisRow(row.analysis) }))
    .filter((row) => row.analysis?.source_hash && row.analysis.source_hash !== rssArticleSourceHash(row));
  return [...direct, ...stale].slice(0, cappedLimit);
}

export async function saveRssArticleAnalysis(article: StoredRssArticle, analysis: FeedAnalysis, topics: string[]): Promise<StoredRssArticleAnalysis> {
  await ensureSchema();
  const sql = getSql();
  const sourceHash = rssArticleSourceHash(article);
  const cleanedTopics = textArray(topics, 20);
  const analysisText = buildAnalysisText(article, analysis, cleanedTopics);
  const whyJson = JSON.stringify(textArray(analysis.why_it_matters, 8));
  const riskJson = JSON.stringify(textArray(analysis.risk_signals, 8));
  const followJson = JSON.stringify(textArray(analysis.follow_up_questions, 8));
  const keywords = textArray(analysis.keywords, 20);
  const individuals = textArray(analysis.individuals, 20);
  const entities = textArray(analysis.entities, 30);

  const rows = (await sql`
    INSERT INTO rss_article_analysis (
      article_id, guid, source_hash, status, model, generated_at, thesis,
      why_it_matters, risk_signals, follow_up_questions,
      keywords, individuals, entities, topics, analysis_text, fallback, error
    )
    VALUES (
      ${article.id}, ${article.guid}, ${sourceHash}, 'enriched', ${analysis.model}, ${analysis.generated_at},
      ${analysis.thesis}, ${whyJson}::jsonb, ${riskJson}::jsonb, ${followJson}::jsonb,
      ${keywords}, ${individuals}, ${entities}, ${cleanedTopics}, ${analysisText}, ${analysis.fallback}, ''
    )
    ON CONFLICT (article_id) DO UPDATE SET
      guid = EXCLUDED.guid,
      source_hash = EXCLUDED.source_hash,
      status = EXCLUDED.status,
      model = EXCLUDED.model,
      generated_at = EXCLUDED.generated_at,
      thesis = EXCLUDED.thesis,
      why_it_matters = EXCLUDED.why_it_matters,
      risk_signals = EXCLUDED.risk_signals,
      follow_up_questions = EXCLUDED.follow_up_questions,
      keywords = EXCLUDED.keywords,
      individuals = EXCLUDED.individuals,
      entities = EXCLUDED.entities,
      topics = EXCLUDED.topics,
      analysis_text = EXCLUDED.analysis_text,
      fallback = EXCLUDED.fallback,
      error = ''
    RETURNING *
  `) as unknown as Record<string, unknown>[];

  await saveIntelligenceMentions("rss_article", String(article.id), [
    ...keywords.map((value) => ({ type: "keyword" as const, value })),
    ...individuals.map((value) => ({ type: "individual" as const, value })),
    ...entities.map((value) => ({ type: "entity" as const, value })),
    ...cleanedTopics.map((value) => ({ type: "topic" as const, value })),
  ]);

  return normalizeAnalysisRow(rows[0]) as StoredRssArticleAnalysis;
}

export async function saveRssArticleAnalysisFailure(article: StoredRssArticle, error: string): Promise<void> {
  await ensureSchema();
  const sql = getSql();
  await sql`
    INSERT INTO rss_article_analysis (article_id, guid, source_hash, status, error)
    VALUES (${article.id}, ${article.guid}, ${rssArticleSourceHash(article)}, 'failed', ${String(error || "").slice(0, 800)})
    ON CONFLICT (article_id) DO UPDATE SET
      source_hash = EXCLUDED.source_hash,
      status = 'failed',
      error = EXCLUDED.error,
      generated_at = now()
  `;
}

export async function deleteInvalidCouponArticles(): Promise<number> {
  await ensureSchema();
  const sql = getSql();
  const rows = (await sql`
    DELETE FROM rss_articles
    WHERE title ILIKE '%coupon%'
       OR title ILIKE '%promo code%'
       OR title ILIKE '%promo-code%'
       OR title ILIKE '%discount code%'
       OR title ILIKE '%discount coupon%'
       OR url ILIKE '%coupon%'
       OR url ILIKE '%promo-code%'
       OR url ILIKE '%promo-codes%'
       OR url ILIKE '%discount-code%'
       OR url ILIKE '%discount-coupon%'
       OR description ILIKE '%coupon%'
       OR description ILIKE '%promo code%'
       OR description ILIKE '%promo-code%'
       OR description ILIKE '%discount code%'
       OR description ILIKE '%discount coupon%'
    RETURNING id
  `) as unknown as Array<{ id: number }>;
  return rows.length;
}

export async function saveIntelligenceMentions(
  sourceType: string,
  sourceId: string,
  mentions: Array<{ type: MentionType; value: string; confidence?: number }>
): Promise<void> {
  const sql = getSql();
  await sql`DELETE FROM intelligence_mentions WHERE source_type = ${sourceType} AND source_id = ${sourceId}`;
  for (const mention of mentions) {
    const value = String(mention.value || "").trim();
    const normalized = normalizeMention(value);
    if (!value || !normalized) continue;
    await sql`
      INSERT INTO intelligence_mentions (source_type, source_id, mention_type, value, normalized_value, confidence)
      VALUES (${sourceType}, ${sourceId}, ${mention.type}, ${value}, ${normalized}, ${mention.confidence ?? 1})
      ON CONFLICT (source_type, source_id, mention_type, normalized_value) DO UPDATE SET
        value = EXCLUDED.value,
        confidence = EXCLUDED.confidence,
        generated_at = now()
    `;
  }
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
    query = sql`SELECT a.*, to_jsonb(ra.*) AS analysis FROM rss_articles a LEFT JOIN rss_article_analysis ra ON ra.article_id = a.id WHERE a.feed_key = ${feedKey} AND COALESCE(a.published_at, a.fetched_at) > ${since} AND COALESCE(a.published_at, a.fetched_at) <= ${until} ORDER BY COALESCE(a.published_at, a.fetched_at) DESC LIMIT ${limit}`;
  } else if (feedKey && since) {
    query = sql`SELECT a.*, to_jsonb(ra.*) AS analysis FROM rss_articles a LEFT JOIN rss_article_analysis ra ON ra.article_id = a.id WHERE a.feed_key = ${feedKey} AND COALESCE(a.published_at, a.fetched_at) > ${since} ORDER BY COALESCE(a.published_at, a.fetched_at) DESC LIMIT ${limit}`;
  } else if (feedKey) {
    query = sql`SELECT a.*, to_jsonb(ra.*) AS analysis FROM rss_articles a LEFT JOIN rss_article_analysis ra ON ra.article_id = a.id WHERE a.feed_key = ${feedKey} ORDER BY COALESCE(a.published_at, a.fetched_at) DESC LIMIT ${limit}`;
  } else if (since && until) {
    query = sql`SELECT a.*, to_jsonb(ra.*) AS analysis FROM rss_articles a LEFT JOIN rss_article_analysis ra ON ra.article_id = a.id WHERE COALESCE(a.published_at, a.fetched_at) > ${since} AND COALESCE(a.published_at, a.fetched_at) <= ${until} ORDER BY COALESCE(a.published_at, a.fetched_at) DESC LIMIT ${limit}`;
  } else if (since) {
    query = sql`SELECT a.*, to_jsonb(ra.*) AS analysis FROM rss_articles a LEFT JOIN rss_article_analysis ra ON ra.article_id = a.id WHERE COALESCE(a.published_at, a.fetched_at) > ${since} ORDER BY COALESCE(a.published_at, a.fetched_at) DESC LIMIT ${limit}`;
  } else {
    query = sql`SELECT a.*, to_jsonb(ra.*) AS analysis FROM rss_articles a LEFT JOIN rss_article_analysis ra ON ra.article_id = a.id ORDER BY COALESCE(a.published_at, a.fetched_at) DESC LIMIT ${limit}`;
  }
  const rows = (await query) as unknown as Array<StoredRssArticle & { analysis?: Record<string, unknown> | null }>;
  return rows.map((row) => ({ ...row, analysis: normalizeAnalysisRow(row.analysis) }));
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

import { neon } from "@neondatabase/serverless";
import { createHash } from "crypto";
import type { FeedAnalysis } from "@/lib/server/feed-analysis";
import { isAllowedRssArticleForIngestion } from "@/lib/server/rss-ingestion-filter";
import { isEnglishRssArticle, shouldEnglishOnlyFilterFeed } from "@/lib/server/rss-language-filter";
import type { RssArticle } from "@/lib/server/rss-fetcher";
import { DEFAULT_RSS_FEEDS } from "@/lib/server/rss-fetcher";
import { TOPIC_RULE_RECOMMENDATIONS, formatTopicRuleKeywords } from "@/lib/topic-rule-recommendations";
import { canonicalEntityLabel, normalizeMentionValue } from "@/lib/server/entity-aliases";

// Cap on re-queuing a "strengthened" (model under-delivered, padded with
// boilerplate) analysis for re-analysis, so a chronically-weak article isn't
// re-billed on every refresh with no forward progress. Exported so
// feed-analysis.ts's pure-JS mirror of this gate can't drift from the SQL.
export const MAX_STRENGTHEN_ATTEMPTS = 3;

export type StoredRssArticle = {
  id: number;
  guid: string;
  feed_key: string;
  feed_label?: string | null;
  title: string;
  url: string;
  description: string;
  author: string;
  published_at: string | null;
  tone_label: "positive" | "neutral" | "negative" | null;
  fetched_at: string;
  analysis?: StoredRssArticleAnalysis | null;
  matched_finra_firms?: string[];
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
  strengthened: boolean;
  strengthen_attempts: number;
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
  // True when the model call failed/timed out/returned an invalid response
  // and the row holds the templated buildSourceFallbackSummary() text
  // instead of a real LLM summary. Surfaced in the UI so a degraded recap
  // never looks indistinguishable from a real one.
  fallback: boolean;
  generated_at: string;
};

export type RssFeed = {
  id: number;
  label: string;
  feed_url: string;
  feed_key: string;
  active: boolean;
  active_manually_set?: boolean;
  refresh_interval_minutes: number;
  last_refresh_at: string | null;
  last_error: string | null;
  consecutive_failures: number;
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
  "PONZI_INVESTOR_FRAUD",
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
  "dark_reading",
  "securityweek",
  "microsoft_security_blog",
] as const;

const ACTIVE_RSS_FEED_KEYS = [
  "the_hacker_news",
  "welivesecurity",
  "sophos_security_operations",
] as const;

const FEED_LABEL_CORRECTIONS: Record<string, string> = {
  cls_blue_sky_blog: "CLS Blue Sky Blog",
  harvard_corp_gov_forum: "Harvard Corporate Governance Forum",
  rss_nytimes_com_services_xml_rss_nyt_dealbook_xml: "NYT DealBook",
  rss_nytimes_com_services_xml_rss_nyt_economy_xml: "NYT Economy",
  search_cnbc_com_rs_search_combinedcms_view_xml: "CNBC",
  the_corporate_counsel_net: "The Corporate Counsel",
  www_centralbanking_com_feeds_rss_category_central_banks_fina: "Central Banking",
};

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

function keepAllowedLanguageArticle<T extends { feed_key: string; title?: string | null; description?: string | null; author?: string | null }>(article: T): boolean {
  return !shouldEnglishOnlyFilterFeed(article.feed_key) || isEnglishRssArticle(article);
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

// The normalized_value normalization moved to entity-aliases.ts (as
// normalizeMentionValue) so the alias map and the Python port share one
// definition; this wrapper keeps existing call sites readable.
function normalizeMention(value: string): string {
  return normalizeMentionValue(value);
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
    strengthened: Boolean(row.strengthened),
    strengthen_attempts: Number(row.strengthen_attempts || 0),
    error: String(row.error || ""),
  };
}

let _schemaEnsured: Promise<void> | null = null;

// ensureSchema() runs ~24 DDL/migration statements and used to be called,
// unmemoized, from nearly every exported function below - meaning a single
// request (e.g. the 10-minute rss-refresh cron) could re-run the full check
// 6-10+ times. Cache the in-flight/completed promise per process lifetime so
// it only actually executes once; a failure clears the cache so the next
// call retries instead of permanently wedging.
export async function ensureSchema(): Promise<void> {
  if (!_schemaEnsured) {
    _schemaEnsured = ensureSchemaUncached().catch((err) => {
      _schemaEnsured = null;
      throw err;
    });
  }
  return _schemaEnsured;
}

async function ensureSchemaUncached(): Promise<void> {
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
      strengthened        BOOLEAN NOT NULL DEFAULT false,
      strengthen_attempts INTEGER NOT NULL DEFAULT 0,
      error               TEXT NOT NULL DEFAULT ''
    )
  `;
  await sql`ALTER TABLE rss_article_analysis ADD COLUMN IF NOT EXISTS strengthened BOOLEAN NOT NULL DEFAULT false`;
  await sql`ALTER TABLE rss_article_analysis ADD COLUMN IF NOT EXISTS strengthen_attempts INTEGER NOT NULL DEFAULT 0`;
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
  await sql`ALTER TABLE rss_feeds ADD COLUMN IF NOT EXISTS active_manually_set BOOLEAN NOT NULL DEFAULT false`;
  await sql`ALTER TABLE rss_feeds ADD COLUMN IF NOT EXISTS last_error TEXT`;
  await sql`ALTER TABLE rss_feeds ADD COLUMN IF NOT EXISTS consecutive_failures INTEGER NOT NULL DEFAULT 0`;
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
      fallback       BOOLEAN NOT NULL DEFAULT false,
      generated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
      UNIQUE (recap_date, topic_key)
    )
  `;
  await sql`ALTER TABLE daily_recaps ADD COLUMN IF NOT EXISTS sources TEXT NOT NULL DEFAULT '[]'`;
  await sql`ALTER TABLE daily_recaps ADD COLUMN IF NOT EXISTS fallback BOOLEAN NOT NULL DEFAULT false`;
  await seedDefaultFeeds(sql);
  await applyFeedSourceMigrations(sql);
  await seedDefaultTopicRules(sql);
  await applyTopicTaxonomyMigrations(sql);
}

async function seedDefaultFeeds(sql: ReturnType<typeof neon>): Promise<void> {
  for (const [key, { label, feedUrl, refreshIntervalMinutes }] of Object.entries(DEFAULT_RSS_FEEDS)) {
    const intervalMinutes = Math.max(1, Math.round(Number(refreshIntervalMinutes || 10)));
    const keyRows = (await sql`
      SELECT id
      FROM rss_feeds
      WHERE feed_key = ${key}
      LIMIT 1
    `) as unknown as Array<{ id: number }>;
    const conflictingUrlRows = (await sql`
      SELECT id
      FROM rss_feeds
      WHERE feed_url = ${feedUrl}
        AND feed_key <> ${key}
      LIMIT 1
    `) as unknown as Array<{ id: number }>;

    if (keyRows.length > 0) {
      if (conflictingUrlRows.length > 0) {
        await sql`
          UPDATE rss_feeds
          SET
            label = ${label},
            refresh_interval_minutes = ${intervalMinutes}
          WHERE feed_key = ${key}
        `;
      } else {
        await sql`
          UPDATE rss_feeds
          SET
            label = ${label},
            feed_url = ${feedUrl},
            refresh_interval_minutes = ${intervalMinutes}
          WHERE feed_key = ${key}
        `;
      }
      continue;
    }

    await sql`
      INSERT INTO rss_feeds (label, feed_url, feed_key, refresh_interval_minutes)
      VALUES (${label}, ${feedUrl}, ${key}, ${intervalMinutes})
      ON CONFLICT (feed_url) DO UPDATE SET
        label = EXCLUDED.label,
        refresh_interval_minutes = EXCLUDED.refresh_interval_minutes
    `;
  }
}

async function applyFeedSourceMigrations(sql: ReturnType<typeof neon>): Promise<void> {
  for (const [feedKey, label] of Object.entries(FEED_LABEL_CORRECTIONS)) {
    await sql`
      UPDATE rss_feeds
      SET label = ${label}
      WHERE feed_key = ${feedKey}
    `;
  }

  await sql`
    UPDATE rss_feeds
    SET active = false
    WHERE active_manually_set = false
      AND (
        feed_key = ANY(${DEPRECATED_RSS_FEED_KEYS})
        OR feed_url = ANY(${[
          "https://www.bleepingcomputer.com/feed/",
          "https://www.darkreading.com/rss.xml",
          "https://www.securityweek.com/feed/",
          "https://www.microsoft.com/en-us/security/blog/feed/",
        ]})
      )
  `;
  await sql`
    UPDATE rss_feeds
    SET active = true
    WHERE active_manually_set = false
      AND feed_key = ANY(${ACTIVE_RSS_FEED_KEYS})
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

export async function upsertFeedSource(
  feedKey: string,
  label: string,
  feedUrl: string,
  refreshIntervalMinutes = 180
): Promise<RssFeed> {
  await ensureSchema();
  const sql = getSql();
  const intervalMinutes = Math.max(1, Math.round(Number(refreshIntervalMinutes || 180)));
  const rows = (await sql`
    INSERT INTO rss_feeds (label, feed_url, feed_key, refresh_interval_minutes, active)
    VALUES (${label.trim()}, ${feedUrl.trim()}, ${feedKey.trim()}, ${intervalMinutes}, true)
    ON CONFLICT (feed_key) DO UPDATE SET
      label = EXCLUDED.label,
      feed_url = EXCLUDED.feed_url,
      refresh_interval_minutes = EXCLUDED.refresh_interval_minutes,
      active = true
    RETURNING *
  `) as unknown as RssFeed[];
  return rows[0];
}

export async function markFeedRefreshed(feedKey: string, error?: string | null): Promise<void> {
  const sql = getSql();
  if (error) {
    await sql`
      UPDATE rss_feeds
      SET last_refresh_at = now(),
          last_error = ${error},
          consecutive_failures = consecutive_failures + 1
      WHERE feed_key = ${feedKey}
    `;
    return;
  }
  await sql`
    UPDATE rss_feeds
    SET last_refresh_at = now(),
        last_error = NULL,
        consecutive_failures = 0
    WHERE feed_key = ${feedKey}
  `;
}

export async function toggleFeed(id: number, active: boolean): Promise<void> {
  await ensureSchema();
  const sql = getSql();
  await sql`UPDATE rss_feeds SET active = ${active}, active_manually_set = true WHERE id = ${id}`;
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
    SELECT a.*, f.label AS feed_label, to_jsonb(ra.*) AS analysis
    FROM rss_articles a
    LEFT JOIN rss_feeds f ON f.feed_key = a.feed_key
    LEFT JOIN rss_article_analysis ra ON ra.article_id = a.id
    WHERE a.id = ${articleId}
    LIMIT 1
  `) as unknown as Array<StoredRssArticle & { analysis?: Record<string, unknown> | null }>;
  const row = rows[0];
  const article = row ? { ...row, analysis: normalizeAnalysisRow(row.analysis) } : null;
  return article && keepAllowedLanguageArticle(article) ? article : null;
}

export async function getRssArticlesNeedingAnalysis(limit = 10, opts: { feedKeys?: string[] } = {}): Promise<StoredRssArticle[]> {
  await ensureSchema();
  const sql = getSql();
  const cappedLimit = Math.max(1, Math.min(100, limit));
  const feedKeys = Array.from(new Set((opts.feedKeys || []).map((item) => String(item || "").trim()).filter(Boolean)));
  const includeAnyFeed = feedKeys.length === 0;
  const refreshForDeepSeek =
    String(process.env.FEED_ANALYSIS_PROVIDER || "").trim().toLowerCase() !== "openai";
  const rows = (await sql`
    SELECT a.*, f.label AS feed_label, to_jsonb(ra.*) AS analysis
    FROM rss_articles a
    LEFT JOIN rss_feeds f ON f.feed_key = a.feed_key
    LEFT JOIN rss_article_analysis ra ON ra.article_id = a.id
    WHERE (${includeAnyFeed} OR a.feed_key = ANY(${feedKeys}))
      AND (
        ra.article_id IS NULL
        OR ra.status IN ('pending', 'failed', 'stale')
        OR (
          ${refreshForDeepSeek}
          AND ra.status = 'enriched'
          AND (
            ra.fallback = true
            -- Padded (boilerplate-filled) analyses look complete by item count
            -- but the model underdelivered; re-queue them, capped so a
            -- chronically-underperforming article isn't re-billed forever.
            OR (ra.strengthened = true AND ra.strengthen_attempts < ${MAX_STRENGTHEN_ATTEMPTS})
            OR ra.model NOT ILIKE 'deepseek%'
            OR length(COALESCE(ra.thesis, '')) < 40
            OR jsonb_array_length(COALESCE(ra.why_it_matters, '[]'::jsonb)) < 2
            OR jsonb_array_length(COALESCE(ra.risk_signals, '[]'::jsonb)) < 2
            OR jsonb_array_length(COALESCE(ra.follow_up_questions, '[]'::jsonb)) < 2
          )
        )
      )
    ORDER BY COALESCE(a.published_at, a.fetched_at) DESC
    LIMIT ${cappedLimit}
  `) as unknown as Array<StoredRssArticle & { analysis?: Record<string, unknown> | null }>;
  const direct = rows
    .map((row) => ({ ...row, analysis: normalizeAnalysisRow(row.analysis) }))
    .filter(keepAllowedLanguageArticle);
  if (direct.length >= cappedLimit) return direct;

  const recentRows = (await sql`
    SELECT a.*, f.label AS feed_label, to_jsonb(ra.*) AS analysis
    FROM rss_articles a
    LEFT JOIN rss_feeds f ON f.feed_key = a.feed_key
    LEFT JOIN rss_article_analysis ra ON ra.article_id = a.id
    WHERE (${includeAnyFeed} OR a.feed_key = ANY(${feedKeys}))
      AND ra.article_id IS NOT NULL
    ORDER BY COALESCE(a.published_at, a.fetched_at) DESC
    LIMIT ${Math.max(cappedLimit * 4, 25)}
  `) as unknown as Array<StoredRssArticle & { analysis?: Record<string, unknown> | null }>;
  const stale = recentRows
    .map((row) => ({ ...row, analysis: normalizeAnalysisRow(row.analysis) }))
    .filter(keepAllowedLanguageArticle)
    .filter((row) => row.analysis?.source_hash && row.analysis.source_hash !== rssArticleSourceHash(row));
  return [...direct, ...stale].slice(0, cappedLimit);
}

export type BacktestArticle = {
  id: number;
  title: string;
  description: string;
  url: string;
  feed_key: string;
  published_at: string | null;
};

/**
 * Recent articles for the admin topic-rule backtest tool - a live preview of
 * how a candidate keyword set would have matched real ingested articles,
 * distinct from the static suggested-keyword content in
 * topic-rule-recommendations.ts.
 */
export async function getRecentRssArticlesForBacktest(days = 30, limit = 1500): Promise<BacktestArticle[]> {
  await ensureSchema();
  const sql = getSql();
  const cappedDays = Math.max(1, Math.min(180, days));
  const cappedLimit = Math.max(1, Math.min(3000, limit));
  const rows = (await sql`
    SELECT id, title, description, url, feed_key, published_at
    FROM rss_articles
    WHERE COALESCE(published_at, fetched_at) >= now() - (${cappedDays} * INTERVAL '1 day')
    ORDER BY COALESCE(published_at, fetched_at) DESC
    LIMIT ${cappedLimit}
  `) as unknown as BacktestArticle[];
  return rows.map((row) => ({
    id: Number(row.id),
    title: String(row.title || ""),
    description: String(row.description || ""),
    url: String(row.url || ""),
    feed_key: String(row.feed_key || ""),
    published_at: row.published_at ? String(row.published_at) : null,
  }));
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
      keywords, individuals, entities, topics, analysis_text, fallback, strengthened, strengthen_attempts, error
    )
    VALUES (
      ${article.id}, ${article.guid}, ${sourceHash}, 'enriched', ${analysis.model}, ${analysis.generated_at},
      ${analysis.thesis}, ${whyJson}::jsonb, ${riskJson}::jsonb, ${followJson}::jsonb,
      ${keywords}, ${individuals}, ${entities}, ${cleanedTopics}, ${analysisText}, ${analysis.fallback}, ${analysis.strengthened},
      CASE WHEN ${analysis.strengthened} THEN 1 ELSE 0 END, ''
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
      strengthened = EXCLUDED.strengthened,
      -- Increments only while the model keeps under-delivering on the same
      -- article; a clean (non-strengthened) save resets the counter.
      strengthen_attempts = CASE WHEN EXCLUDED.strengthened THEN rss_article_analysis.strengthen_attempts + 1 ELSE 0 END,
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

// Cleanup queries below only need to re-examine articles inserted since the
// last pass - once a row has survived a check it won't newly become a
// coupon/gambling/non-English match on its own. Bounding to a recent window
// (using the existing fetched_at index) turns what used to be a full-table
// sequential scan on every call into a cheap index range scan, since
// rss_articles has no retention cap and only grows.
const RSS_CLEANUP_WINDOW_HOURS = 24 * 14;

export async function deleteInvalidCouponArticles(windowHours = RSS_CLEANUP_WINDOW_HOURS): Promise<number> {
  await ensureSchema();
  const sql = getSql();
  const cappedHours = Math.max(1, Math.min(24 * 365, windowHours));
  const rows = (await sql`
    DELETE FROM rss_articles
    WHERE fetched_at >= now() - (${cappedHours} * INTERVAL '1 hour')
      AND (
        title ILIKE '%coupon%'
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
      )
    RETURNING id
  `) as unknown as Array<{ id: number }>;
  return rows.length;
}

export async function deleteNonEnglishPrNewswireArticles(windowHours = RSS_CLEANUP_WINDOW_HOURS): Promise<number> {
  await ensureSchema();
  const sql = getSql();
  const cappedHours = Math.max(1, Math.min(24 * 365, windowHours));
  const candidates = (await sql`
    SELECT id, title, description, author, feed_key
    FROM rss_articles
    WHERE fetched_at >= now() - (${cappedHours} * INTERVAL '1 hour')
      AND feed_key LIKE 'prnewswire_%'
  `) as unknown as Array<{ id: number; title: string; description: string; author: string; feed_key: string }>;
  const ids = candidates
    .filter((article) => shouldEnglishOnlyFilterFeed(article.feed_key) && !isEnglishRssArticle(article))
    .map((article) => article.id);
  if (ids.length === 0) return 0;

  const rows = (await sql`
    DELETE FROM rss_articles
    WHERE id = ANY(${ids})
    RETURNING id
  `) as unknown as Array<{ id: number }>;
  return rows.length;
}

export async function deleteBlockedRssArticles(
  topicRules: StoredRssTopicRule[],
  windowHours = RSS_CLEANUP_WINDOW_HOURS
): Promise<number> {
  const hasActiveTopicRules = topicRules.some((rule) => rule.active && String(rule.keywords || "").trim());
  if (!hasActiveTopicRules) return 0;

  await ensureSchema();
  const sql = getSql();
  const cappedHours = Math.max(1, Math.min(24 * 365, windowHours));
  const candidates = (await sql`
    SELECT id, guid, title, url, description, author, published_at, feed_key
    FROM rss_articles
    WHERE fetched_at >= now() - (${cappedHours} * INTERVAL '1 hour')
      AND (
        feed_key LIKE 'prnewswire_%'
        OR feed_key LIKE 'google_news_%'
        OR title ~* '(gambling|casino|slots?|sportsbook|wagering|betting|lottery|poker|blackjack|roulette|sweepstakes)'
        OR description ~* '(gambling|casino|slots?|sportsbook|wagering|betting|lottery|poker|blackjack|roulette|sweepstakes)'
      )
  `) as unknown as Array<Pick<StoredRssArticle, "id" | "guid" | "feed_key" | "title" | "url" | "description" | "author" | "published_at">>;

  const ids = candidates
    .filter((article) => !isAllowedRssArticleForIngestion(article.feed_key, {
      guid: article.guid,
      title: article.title,
      url: article.url,
      description: article.description,
      author: article.author,
      publishedAt: article.published_at ? new Date(article.published_at) : null,
    }, topicRules))
    .map((article) => article.id);

  if (ids.length === 0) return 0;

  const rows = (await sql`
    DELETE FROM rss_articles
    WHERE id = ANY(${ids})
    RETURNING id
  `) as unknown as Array<{ id: number }>;
  return rows.length;
}

const DEFAULT_RSS_RETENTION_DAYS = 180;

export type PruneResult = { deletedArticles: number; deletedMentions: number };

// rss_articles has no cap and grows forever (144+ inserts/day from the
// refresh cron alone). rss_article_analysis cascades on article delete via
// its FK, but intelligence_mentions is keyed generically by
// (source_type, source_id) with no FK, so it needs an explicit sweep.
export async function pruneOldRssData(retentionDays = DEFAULT_RSS_RETENTION_DAYS): Promise<PruneResult> {
  await ensureSchema();
  const sql = getSql();
  const cappedDays = Math.max(30, Math.min(1825, retentionDays));

  const deletedArticleRows = (await sql`
    DELETE FROM rss_articles
    WHERE COALESCE(published_at, fetched_at) < now() - (${cappedDays} * INTERVAL '1 day')
    RETURNING id
  `) as unknown as Array<{ id: number }>;

  let deletedMentions = 0;
  if (deletedArticleRows.length > 0) {
    const ids = deletedArticleRows.map((row) => String(row.id));
    const mentionRows = (await sql`
      DELETE FROM intelligence_mentions
      WHERE source_type = 'rss_article'
        AND source_id = ANY(${ids})
      RETURNING id
    `) as unknown as Array<{ id: number }>;
    deletedMentions = mentionRows.length;
  }

  return { deletedArticles: deletedArticleRows.length, deletedMentions };
}

export type PreparedMentionBatch = {
  types: MentionType[];
  values: string[];
  normalizedValues: string[];
  confidences: number[];
};

// The unique constraint is (source_type, source_id, mention_type,
// normalized_value), so within one save call the only possible collisions
// are same-type mentions that normalize to the same value (e.g. differing
// punctuation/casing). A single multi-row INSERT can't hit the same
// ON CONFLICT target twice, so dedupe here - keeping the last occurrence,
// matching what the old sequential per-row upsert loop would have left
// behind (each iteration overwrote the previous one on a collision).
export function prepareMentionBatch(
  mentions: Array<{ type: MentionType; value: string; confidence?: number }>
): PreparedMentionBatch {
  const byKey = new Map<string, { type: MentionType; value: string; normalized: string; confidence: number }>();
  for (const mention of mentions) {
    let value = String(mention.value || "").trim();
    // Entity mentions resolve through the shared alias map (see CLAUDE.md
    // "Entity normalization / alias map") so "SEC" / "Securities and
    // Exchange Commission" / "the Commission" collapse to one
    // normalized_value instead of fragmenting watchlist matches and trend
    // counts. Canonicalizing the label first (then normalizing the result)
    // is the one path that guarantees value and normalized_value agree.
    // Other mention types (keyword/topic/individual) are left untouched:
    // topics come from fixed rule sets and keywords are free-form phrases,
    // so aliasing them would rewrite meaning, not merge duplicates.
    if (mention.type === "entity") {
      value = canonicalEntityLabel(value);
    }
    const normalized = normalizeMention(value);
    if (!value || !normalized) continue;
    const key = `${mention.type} ${normalized}`;
    byKey.set(key, { type: mention.type, value, normalized, confidence: mention.confidence ?? 1 });
  }
  const types: MentionType[] = [];
  const values: string[] = [];
  const normalizedValues: string[] = [];
  const confidences: number[] = [];
  for (const entry of byKey.values()) {
    types.push(entry.type);
    values.push(entry.value);
    normalizedValues.push(entry.normalized);
    confidences.push(entry.confidence);
  }
  return { types, values, normalizedValues, confidences };
}

export async function saveIntelligenceMentions(
  sourceType: string,
  sourceId: string,
  mentions: Array<{ type: MentionType; value: string; confidence?: number }>
): Promise<void> {
  const sql = getSql();
  const batch = prepareMentionBatch(mentions);

  await sql`DELETE FROM intelligence_mentions WHERE source_type = ${sourceType} AND source_id = ${sourceId}`;

  if (batch.types.length === 0) return;

  await sql`
    INSERT INTO intelligence_mentions (source_type, source_id, mention_type, value, normalized_value, confidence)
    SELECT ${sourceType}, ${sourceId}, t, v, n, c
    FROM unnest(
      ${batch.types}::text[],
      ${batch.values}::text[],
      ${batch.normalizedValues}::text[],
      ${batch.confidences}::double precision[]
    ) AS mention(t, v, n, c)
    ON CONFLICT (source_type, source_id, mention_type, normalized_value) DO UPDATE SET
      value = EXCLUDED.value,
      confidence = EXCLUDED.confidence,
      generated_at = now()
  `;
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
    query = sql`SELECT a.*, f.label AS feed_label, to_jsonb(ra.*) AS analysis FROM rss_articles a LEFT JOIN rss_feeds f ON f.feed_key = a.feed_key LEFT JOIN rss_article_analysis ra ON ra.article_id = a.id WHERE a.feed_key = ${feedKey} AND COALESCE(a.published_at, a.fetched_at) > ${since} AND COALESCE(a.published_at, a.fetched_at) <= ${until} ORDER BY COALESCE(a.published_at, a.fetched_at) DESC LIMIT ${limit}`;
  } else if (feedKey && since) {
    query = sql`SELECT a.*, f.label AS feed_label, to_jsonb(ra.*) AS analysis FROM rss_articles a LEFT JOIN rss_feeds f ON f.feed_key = a.feed_key LEFT JOIN rss_article_analysis ra ON ra.article_id = a.id WHERE a.feed_key = ${feedKey} AND COALESCE(a.published_at, a.fetched_at) > ${since} ORDER BY COALESCE(a.published_at, a.fetched_at) DESC LIMIT ${limit}`;
  } else if (feedKey) {
    query = sql`SELECT a.*, f.label AS feed_label, to_jsonb(ra.*) AS analysis FROM rss_articles a LEFT JOIN rss_feeds f ON f.feed_key = a.feed_key LEFT JOIN rss_article_analysis ra ON ra.article_id = a.id WHERE a.feed_key = ${feedKey} ORDER BY COALESCE(a.published_at, a.fetched_at) DESC LIMIT ${limit}`;
  } else if (since && until) {
    query = sql`SELECT a.*, f.label AS feed_label, to_jsonb(ra.*) AS analysis FROM rss_articles a LEFT JOIN rss_feeds f ON f.feed_key = a.feed_key LEFT JOIN rss_article_analysis ra ON ra.article_id = a.id WHERE COALESCE(a.published_at, a.fetched_at) > ${since} AND COALESCE(a.published_at, a.fetched_at) <= ${until} ORDER BY COALESCE(a.published_at, a.fetched_at) DESC LIMIT ${limit}`;
  } else if (since) {
    query = sql`SELECT a.*, f.label AS feed_label, to_jsonb(ra.*) AS analysis FROM rss_articles a LEFT JOIN rss_feeds f ON f.feed_key = a.feed_key LEFT JOIN rss_article_analysis ra ON ra.article_id = a.id WHERE COALESCE(a.published_at, a.fetched_at) > ${since} ORDER BY COALESCE(a.published_at, a.fetched_at) DESC LIMIT ${limit}`;
  } else {
    query = sql`SELECT a.*, f.label AS feed_label, to_jsonb(ra.*) AS analysis FROM rss_articles a LEFT JOIN rss_feeds f ON f.feed_key = a.feed_key LEFT JOIN rss_article_analysis ra ON ra.article_id = a.id ORDER BY COALESCE(a.published_at, a.fetched_at) DESC LIMIT ${limit}`;
  }
  const rows = (await query) as unknown as Array<StoredRssArticle & { analysis?: Record<string, unknown> | null }>;
  return rows
    .map((row) => ({ ...row, analysis: normalizeAnalysisRow(row.analysis) }))
    .filter(keepAllowedLanguageArticle);
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
      INSERT INTO daily_recaps (recap_date, topic_key, topic_label, summary, article_count, positive_count, negative_count, neutral_count, sources, fallback)
      VALUES (${row.recap_date}, ${row.topic_key}, ${row.topic_label}, ${row.summary}, ${row.article_count}, ${row.positive_count}, ${row.negative_count}, ${row.neutral_count}, ${sourcesJson}, ${row.fallback})
      ON CONFLICT (recap_date, topic_key) DO UPDATE
      SET summary        = EXCLUDED.summary,
          topic_label    = EXCLUDED.topic_label,
          article_count  = EXCLUDED.article_count,
          positive_count = EXCLUDED.positive_count,
          negative_count = EXCLUDED.negative_count,
          neutral_count  = EXCLUDED.neutral_count,
          sources        = EXCLUDED.sources,
          fallback       = EXCLUDED.fallback,
          generated_at   = now()
    `;
  }
}

export type NeonMirroredDocumentRow = {
  document_id: string;
  metadata: Record<string, unknown>;
};

// Phase 3 of migrating off custom_documents.json (see CLAUDE.md): a
// read-only query against the `documents` mirror table (Phase 1/2), for
// readers that only need metadata, not full_text - deliberately omits
// full_text from the SELECT since it's the bulk of each row's size and
// unused by e.g. /api/metrics, unlike downloading the full GCS blob which
// has no way to omit it. No ensureSchema() call here: the `documents` table
// is created lazily by the Python-side mirror (neon_feeds.py), not by this
// file's ensureSchema(); if it doesn't exist yet, the query below fails
// naturally and callers are expected to fall back to the GCS blob.
export async function getAllMirroredDocumentMetadata(): Promise<NeonMirroredDocumentRow[]> {
  const sql = getSql();
  const rows = (await sql`SELECT document_id, metadata FROM documents`) as unknown as NeonMirroredDocumentRow[];
  return rows;
}

// ─── Stock attention (docs/stock-attention-spec.md) ─────────────────────────
// Both tables below are Python-owned (created lazily by neon_feeds.py, like
// `documents` above) - deliberately no ensureSchema() here. If they don't
// exist yet (sweep never ran), queries fail naturally and the API route
// returns an empty payload with a warning.

export type DailyStockAttentionRow = {
  attention_date: string;
  ticker: string;
  company: string;
  mention_count: number;
  reddit_count: number;
  news_count: number;
  total_mention_count: number;
  source_count: number;
  subreddit_count: number;
  weighted_score: number;
  mood: string;
  top_source_ids: string; // JSON array of reddit_attention_items.source_id
  price_close: number | null;
  price_pct: number | null;
  volume: number | null;
  volume_vs_20d: number | null;
  divergence: string;
  weighted_mention_count: number;
  quality_flags: string; // JSON array, same convention as top_source_ids
  top_news_ids: string;  // JSON array of rss_articles ids (SEC-4)
  generated_at: string;
};

export type RedditAttentionItemRow = {
  source_id: string;
  kind: string;
  subreddit: string;
  author: string;
  title: string;
  permalink: string;
  created_utc: string;
  score: number;
  mood: string;
};

export async function getLatestStockAttentionDate(): Promise<string | null> {
  const sql = getSql();
  const rows = (await sql`SELECT MAX(attention_date)::text AS latest FROM daily_stock_attention`) as unknown as { latest: string | null }[];
  return rows[0]?.latest ?? null;
}

export async function getDailyStockAttention(date: string, limit = 50): Promise<DailyStockAttentionRow[]> {
  const sql = getSql();
  try {
    return (await sql`
      SELECT attention_date::text AS attention_date, ticker, company, mention_count, reddit_count, news_count,
             total_mention_count, source_count, subreddit_count, weighted_score::float AS weighted_score, mood,
             top_source_ids, price_close::float AS price_close, price_pct::float AS price_pct,
             volume, volume_vs_20d::float AS volume_vs_20d, divergence,
             weighted_mention_count::float AS weighted_mention_count, quality_flags, top_news_ids,
             generated_at::text AS generated_at
      FROM daily_stock_attention
      WHERE attention_date = ${date}::date
      ORDER BY weighted_score DESC, total_mention_count DESC, ticker ASC
      LIMIT ${limit}
    `) as unknown as DailyStockAttentionRow[];
  } catch (err) {
    // Old-schema tolerance (deploy-order rule in CLAUDE.md): top_news_ids
    // is added by the Python rollup's ALTER; until that runs post-deploy,
    // retry without the column so the board renders instead of erroring.
    if (!String(err).includes("top_news_ids")) throw err;
    const rows = (await sql`
      SELECT attention_date::text AS attention_date, ticker, company, mention_count, reddit_count, news_count,
             total_mention_count, source_count, subreddit_count, weighted_score::float AS weighted_score, mood,
             top_source_ids, price_close::float AS price_close, price_pct::float AS price_pct,
             volume, volume_vs_20d::float AS volume_vs_20d, divergence,
             weighted_mention_count::float AS weighted_mention_count, quality_flags,
             generated_at::text AS generated_at
      FROM daily_stock_attention
      WHERE attention_date = ${date}::date
      ORDER BY weighted_score DESC, total_mention_count DESC, ticker ASC
      LIMIT ${limit}
    `) as unknown as Omit<DailyStockAttentionRow, "top_news_ids">[];
    return rows.map((row) => ({ ...row, top_news_ids: "[]" }));
  }
}

export type RssArticleRef = { id: number; title: string; url: string };

// SEC-4: resolve the drawer's top_news_ids to linkable articles. Articles
// older than the RSS retention window may be pruned - callers must treat
// missing ids as normal.
export async function getRssArticlesByIds(ids: number[]): Promise<RssArticleRef[]> {
  if (ids.length === 0) return [];
  const sql = getSql();
  const rows = (await sql`
    SELECT id, title, url FROM rss_articles WHERE id = ANY(${ids})
  `) as unknown as RssArticleRef[];
  return rows;
}

// Item 3: per-ticker history for the detail drawer/chart. Reads
// daily_stock_attention directly (persists indefinitely, per spec §6.3) -
// no separate history table needed.
export async function getStockAttentionHistory(ticker: string, days = 30): Promise<DailyStockAttentionRow[]> {
  const sql = getSql();
  const rows = (await sql`
    SELECT attention_date::text AS attention_date, ticker, company, mention_count, reddit_count, news_count,
           total_mention_count, source_count, subreddit_count, weighted_score::float AS weighted_score, mood,
           top_source_ids, price_close::float AS price_close, price_pct::float AS price_pct,
           volume, volume_vs_20d::float AS volume_vs_20d, divergence,
           weighted_mention_count::float AS weighted_mention_count, quality_flags,
           generated_at::text AS generated_at
    FROM daily_stock_attention
    WHERE ticker = ${ticker}
      AND attention_date >= CURRENT_DATE - (${days} * INTERVAL '1 day')
    ORDER BY attention_date ASC
  `) as unknown as DailyStockAttentionRow[];
  return rows;
}

// ─── Attention sweep config + review queue (enhancement items 4/6) ─────────
// The config is WRITTEN by the admin UI (this side) and read fail-soft by
// the Python sweep/rollup - the reverse ownership of the other attention
// tables, so the CREATE lives here too (kept in lockstep with
// neon_feeds.py's _ensure_attention_config_schema; if you change one,
// change the other). The review queue is written by the Python rollup and
// worked through the admin UI here.

export type AttentionSweepSubreddit = { name: string; tier: number; weight: number; active: boolean };
export type AttentionSweepConfig = {
  subreddits: AttentionSweepSubreddit[];
  bot_blocklist: string[];
  symbol_overrides: { force_ambiguous: string[]; force_unambiguous: string[] };
  author_weighting: { low_diversity_share: number; low_diversity_max_tickers: number; discount: number; min_items?: number };
};

async function ensureAttentionConfigSchema(sql: ReturnType<typeof neon>): Promise<void> {
  await sql`
    CREATE TABLE IF NOT EXISTS attention_sweep_config (
      id         SERIAL PRIMARY KEY,
      config     JSONB NOT NULL,
      updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
  `;
}

export async function getAttentionSweepConfig(): Promise<AttentionSweepConfig | null> {
  const sql = getSql();
  await ensureAttentionConfigSchema(sql);
  const rows = (await sql`SELECT config FROM attention_sweep_config ORDER BY id DESC LIMIT 1`) as unknown as { config: AttentionSweepConfig }[];
  return rows[0]?.config ?? null;
}

export async function saveAttentionSweepConfig(config: AttentionSweepConfig): Promise<void> {
  const sql = getSql();
  await ensureAttentionConfigSchema(sql);
  const json = JSON.stringify(config);
  const existing = (await sql`SELECT id FROM attention_sweep_config ORDER BY id DESC LIMIT 1`) as unknown as { id: number }[];
  if (existing.length > 0) {
    await sql`UPDATE attention_sweep_config SET config = ${json}::jsonb, updated_at = now() WHERE id = ${existing[0].id}`;
  } else {
    await sql`INSERT INTO attention_sweep_config (config) VALUES (${json}::jsonb)`;
  }
}

export type AttentionReviewRow = {
  id: number;
  review_date: string;
  ticker: string;
  status: string;
  sample_source_ids: string;
  created_at: string;
  reviewed_at: string | null;
};

export async function getAttentionReviewQueue(status = "pending", limit = 100): Promise<AttentionReviewRow[]> {
  const sql = getSql();
  const rows = (await sql`
    SELECT id, review_date::text AS review_date, ticker, status, sample_source_ids,
           created_at::text AS created_at, reviewed_at::text AS reviewed_at
    FROM attention_review_queue
    WHERE status = ${status}
    ORDER BY review_date DESC, ticker ASC
    LIMIT ${limit}
  `) as unknown as AttentionReviewRow[];
  return rows;
}

export async function resolveAttentionReviewItem(id: number, status: "legit" | "false_positive"): Promise<AttentionReviewRow | null> {
  const sql = getSql();
  const rows = (await sql`
    UPDATE attention_review_queue
    SET status = ${status}, reviewed_at = now()
    WHERE id = ${id}
    RETURNING id, review_date::text AS review_date, ticker, status, sample_source_ids,
              created_at::text AS created_at, reviewed_at::text AS reviewed_at
  `) as unknown as AttentionReviewRow[];
  return rows[0] ?? null;
}

// ─── Attention Activity + Authors views (see CLAUDE.md plan, 2026-07-12) ───

export type AttentionActivityRow = {
  source_id: string;
  kind: string;
  subreddit: string;
  author: string;
  title: string;
  permalink: string;
  created_utc: string;
  score: number;
  mood: string;
  tickers: string[];
};

export async function getRecentAttentionActivity(hoursBack = 24, limit = 150): Promise<AttentionActivityRow[]> {
  const sql = getSql();
  const cappedHours = Math.max(1, Math.min(72, hoursBack));
  const cappedLimit = Math.max(1, Math.min(300, limit));
  const rows = (await sql`
    SELECT i.source_id, i.kind, i.subreddit, i.author, i.title, i.permalink,
           i.created_utc::text AS created_utc, i.score, i.mood,
           ARRAY(
             SELECT m.value FROM intelligence_mentions m
             WHERE m.source_type IN ('reddit_post', 'reddit_comment')
               AND m.source_id = i.source_id
               AND m.mention_type = 'ticker'
             ORDER BY m.value
           ) AS tickers
    FROM reddit_attention_items i
    WHERE i.created_utc >= now() - (${cappedHours} * INTERVAL '1 hour')
    ORDER BY i.created_utc DESC
    LIMIT ${cappedLimit}
  `) as unknown as AttentionActivityRow[];
  return rows;
}

export type RedditAuthorStatsRow = {
  author: string;
  first_seen: string | null;
  last_seen: string | null;
  items_total: number;
  tickers_distinct: number;
  subreddits_distinct: number;
  top_ticker_share: number;
  top_ticker: string;
  account_created: string | null;
  link_karma: number | null;
};

export async function getRedditAuthorStats(limit = 50): Promise<RedditAuthorStatsRow[]> {
  const sql = getSql();
  const cappedLimit = Math.max(1, Math.min(200, limit));
  try {
    return (await sql`
      SELECT author, first_seen::text AS first_seen, last_seen::text AS last_seen,
             items_total, tickers_distinct, subreddits_distinct,
             top_ticker_share::float AS top_ticker_share, top_ticker,
             account_created::text AS account_created, link_karma
      FROM reddit_author_stats
      WHERE items_total > 0
      ORDER BY items_total DESC, author ASC
      LIMIT ${cappedLimit}
    `) as unknown as RedditAuthorStatsRow[];
  } catch (err) {
    // Old-schema tolerance (deploy-order rule in CLAUDE.md): top_ticker is
    // added by the Python rollup's ALTER, which may not have run yet when
    // this reader deploys. Retry without the column for that one cycle.
    if (!String(err).includes("top_ticker")) throw err;
    const rows = (await sql`
      SELECT author, first_seen::text AS first_seen, last_seen::text AS last_seen,
             items_total, tickers_distinct, subreddits_distinct,
             top_ticker_share::float AS top_ticker_share,
             account_created::text AS account_created, link_karma
      FROM reddit_author_stats
      WHERE items_total > 0
      ORDER BY items_total DESC, author ASC
      LIMIT ${cappedLimit}
    `) as unknown as Omit<RedditAuthorStatsRow, "top_ticker">[];
    return rows.map((row) => ({ ...row, top_ticker: "" }));
  }
}

export type StockAttentionSparklinePoint = { attention_date: string; total_mention_count: number };

// Item 3: batched sparkline history for every ticker on the current
// leaderboard in one query, instead of one round trip per row.
export async function getStockAttentionSparklines(
  tickers: string[],
  days = 14
): Promise<Map<string, StockAttentionSparklinePoint[]>> {
  const map = new Map<string, StockAttentionSparklinePoint[]>();
  if (tickers.length === 0) return map;
  const sql = getSql();
  const rows = (await sql`
    SELECT ticker, attention_date::text AS attention_date, total_mention_count
    FROM daily_stock_attention
    WHERE ticker = ANY(${tickers})
      AND attention_date >= CURRENT_DATE - (${days} * INTERVAL '1 day')
    ORDER BY ticker ASC, attention_date ASC
  `) as unknown as { ticker: string; attention_date: string; total_mention_count: number }[];
  for (const row of rows) {
    const list = map.get(row.ticker) ?? [];
    list.push({ attention_date: row.attention_date, total_mention_count: row.total_mention_count });
    map.set(row.ticker, list);
  }
  return map;
}

export type IntradayMentionRow = { ticker: string; author: string; created_utc: string };

// Item 3c: raw rows for the "hot right now" intraday view. Deliberately
// unaggregated - the API route computes per-author dedup + freshness
// decay at request time (not stored; this is the freshness-decay math
// the daily rollup's §6.2 explicitly keeps out of the persisted rollup).
export async function getIntradayTickerMentions(hoursBack = 24): Promise<IntradayMentionRow[]> {
  const sql = getSql();
  const cappedHours = Math.max(1, Math.min(72, hoursBack));
  const rows = (await sql`
    SELECT m.value AS ticker, i.author, i.created_utc::text AS created_utc
    FROM intelligence_mentions m
    JOIN reddit_attention_items i ON i.source_id = m.source_id
    WHERE m.mention_type = 'ticker'
      AND m.source_type IN ('reddit_post', 'reddit_comment')
      AND i.created_utc >= now() - (${cappedHours} * INTERVAL '1 hour')
  `) as unknown as IntradayMentionRow[];
  return rows;
}

export async function getRedditAttentionItems(sourceIds: string[]): Promise<RedditAttentionItemRow[]> {
  if (sourceIds.length === 0) return [];
  const sql = getSql();
  const rows = (await sql`
    SELECT source_id, kind, subreddit, author, title, permalink,
           created_utc::text AS created_utc, score, mood
    FROM reddit_attention_items
    WHERE source_id = ANY(${sourceIds})
  `) as unknown as RedditAttentionItemRow[];
  return rows;
}

export interface TopicRuleInput {
  topic_key: string;
  label: string;
  keywords: string;
  active: boolean;
  sort_order: number;
}

export interface TopicRuleView {
  topic_key: string;
  label: string;
  keywords: string[];
  keywordMatchers: TopicKeywordMatcher[];
  sort_order: number;
}

export interface TopicArticleInput {
  title: string;
  description?: string | null;
}

export interface TopicMatch {
  rule: TopicRuleView;
  score: number;
}

export interface TopicKeywordMatcher {
  keyword: string;
  pattern: RegExp | null;
  specificity: number;
}

export function decodeEntities(text: string): string {
  return text
    .replace(/&#x([0-9a-fA-F]+);/gi, (_, hex) => String.fromCharCode(parseInt(hex, 16)))
    .replace(/&#(\d+);/g, (_, dec) => String.fromCharCode(parseInt(dec, 10)))
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&nbsp;/g, " ");
}

export function parseKeywords(value: string): string[] {
  return value
    .split(/[\n,]+/)
    .map((item) => item.trim().toLowerCase())
    .filter(Boolean);
}

export function normalizeTopicRules(rules: TopicRuleInput[]): TopicRuleView[] {
  return rules
    .filter((rule) => rule && rule.active)
    .map((rule) => {
      const keywords = parseKeywords(String(rule.keywords || ""));
      return {
        topic_key: String(rule.topic_key || "").trim(),
        label: String(rule.label || "").trim() || String(rule.topic_key || "").trim(),
        keywords,
        keywordMatchers: keywords.map(compileKeywordMatcher),
        sort_order: Number(rule.sort_order || 100),
      };
    })
    .filter((rule) => rule.topic_key && rule.label);
}

export function normalizeMatchText(text: string): string {
  return decodeEntities(text || "")
    .toLowerCase()
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[\u201c\u201d]/g, '"');
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function keywordPattern(keyword: string): RegExp | null {
  const normalized = normalizeMatchText(keyword).replace(/\s+/g, " ").trim();
  if (!normalized) return null;

  const parts = normalized.split(/\s+/).map(escapeRegExp);
  const source = parts.join("[\\s\\-\\u2013\\u2014_/]+");
  return new RegExp(`(^|[^a-z0-9])${source}(?=$|[^a-z0-9])`, "i");
}

function compileKeywordMatcher(keyword: string): TopicKeywordMatcher {
  return {
    keyword,
    pattern: keywordPattern(keyword),
    specificity: keywordSpecificity(keyword),
  };
}

function keywordMatcherScore(matcher: TopicKeywordMatcher, title: string, description: string): number {
  if (!matcher.pattern) {
    return 0;
  }
  if (matcher.pattern.test(title)) {
    return 100 + matcher.specificity;
  }
  if (matcher.pattern.test(description)) {
    return 50 + matcher.specificity;
  }
  return 0;
}

function keywordSpecificity(keyword: string): number {
  const normalized = normalizeMatchText(keyword).replace(/\s+/g, " ").trim();
  const compact = normalized.replace(/[^a-z0-9]/g, "");
  const wordCount = normalized ? normalized.split(/\s+/).length : 0;
  const acronymBoost = compact.length > 0 && compact.length <= 3 ? 8 : 0;
  return Math.min(28, compact.length + Math.max(0, wordCount - 1) * 6 + acronymBoost);
}

export function getTopicMatches(article: TopicArticleInput, rules: TopicRuleView[]): TopicMatch[] {
  const title = normalizeMatchText(article.title);
  const description = normalizeMatchText(article.description ?? "");
  return rules
    .map((rule) => {
      const matchers = rule.keywordMatchers || rule.keywords.map(compileKeywordMatcher);
      const score = matchers.reduce((best, matcher) => Math.max(best, keywordMatcherScore(matcher, title, description)), 0);
      return { rule, score };
    })
    .filter((match) => match.score > 0)
    .sort((a, b) => b.score - a.score || a.rule.sort_order - b.rule.sort_order || a.rule.label.localeCompare(b.rule.label));
}

export function getMatchingTopics(article: TopicArticleInput, rules: TopicRuleView[]): TopicRuleView[] {
  return getTopicMatches(article, rules).map((match) => match.rule);
}

/**
 * Compile a raw keyword list into matchers once, so callers scanning many
 * articles (e.g. the topic-rule backtest tool) don't recompile the same
 * regex per article. Reuses the exact same word-boundary matching as
 * getTopicMatches/getMatchingTopics, so a backtest preview reflects real
 * ingestion-time behavior.
 */
export function compileKeywords(keywords: string[]): TopicKeywordMatcher[] {
  return keywords.map(compileKeywordMatcher);
}

/** Keywords (from precompiled matchers) that match a given article. */
export function matchingKeywordsForArticle(matchers: TopicKeywordMatcher[], article: TopicArticleInput): string[] {
  const title = normalizeMatchText(article.title);
  const description = normalizeMatchText(article.description ?? "");
  return matchers.filter((matcher) => keywordMatcherScore(matcher, title, description) > 0).map((matcher) => matcher.keyword);
}

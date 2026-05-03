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
    .map((rule) => ({
      topic_key: String(rule.topic_key || "").trim(),
      label: String(rule.label || "").trim() || String(rule.topic_key || "").trim(),
      keywords: parseKeywords(String(rule.keywords || "")),
      sort_order: Number(rule.sort_order || 100),
    }))
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

function keywordMatches(text: string, keyword: string): boolean {
  const pattern = keywordPattern(keyword);
  return pattern ? pattern.test(text) : false;
}

function keywordSpecificity(keyword: string): number {
  const normalized = normalizeMatchText(keyword).replace(/\s+/g, " ").trim();
  const compact = normalized.replace(/[^a-z0-9]/g, "");
  const wordCount = normalized ? normalized.split(/\s+/).length : 0;
  const acronymBoost = compact.length > 0 && compact.length <= 3 ? 8 : 0;
  return Math.min(28, compact.length + Math.max(0, wordCount - 1) * 6 + acronymBoost);
}

function keywordScore(keyword: string, title: string, description: string): number {
  const specificity = keywordSpecificity(keyword);
  if (keywordMatches(title, keyword)) {
    return 100 + specificity;
  }
  if (keywordMatches(description, keyword)) {
    return 50 + specificity;
  }
  return 0;
}

export function getTopicMatches(article: TopicArticleInput, rules: TopicRuleView[]): TopicMatch[] {
  const title = normalizeMatchText(article.title);
  const description = normalizeMatchText(article.description ?? "");
  return rules
    .map((rule) => {
      const score = rule.keywords.reduce((best, keyword) => Math.max(best, keywordScore(keyword, title, description)), 0);
      return { rule, score };
    })
    .filter((match) => match.score > 0)
    .sort((a, b) => b.score - a.score || a.rule.sort_order - b.rule.sort_order || a.rule.label.localeCompare(b.rule.label));
}

export function getMatchingTopics(article: TopicArticleInput, rules: TopicRuleView[]): TopicRuleView[] {
  return getTopicMatches(article, rules).map((match) => match.rule);
}

import { decodeEntities } from "@/lib/intel-topic-matching";

export type NewsFeedQualityInput = {
  feed_key?: string | null;
  source_kind?: string | null;
  title?: string | null;
  description?: string | null;
  url?: string | null;
  author?: string | null;
  tags?: string[] | null;
  keywords?: string[] | null;
};

const WIRED_FEED_KEYS = new Set(["wired_security"]);
const WIRED_SOURCE_KINDS = new Set(["wired_article"]);

const COUPON_SPAM_RE = /\b(?:promo[\s-]*codes?|coupon(?:s|[\s-]*codes?)|discount[\s-]*(?:codes?|coupons?))\b/i;
const CJK_RE = /[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]/gu;
const LOW_VALUE_WIRED_RE = /\b(?:best\s+(?:tv\s+shows?|movies?|streaming|shows?|games?|deals?|sales?|gifts?|gadgets?|phones?|laptops?|headphones?)|tv\s+shows?|movie\s+reviews?|movies?\s+to\s+watch|stream(?:ing)?\s+(?:this\s+month|guide|service|services|platforms?)|netflix|hulu|disney\+|hbo|max|peacock|paramount\+|prime\s+video|apple\s+tv|the\s+onion|infowars\s+parody|parody|satire|celebrity|trailer|comic-con|coupon(?:s|[\s-]*codes?)|promo[\s-]*codes?|discount[\s-]*(?:codes?|coupons?)|prime\s+day|black\s+friday|digital\s+(?:content|media|library|libraries)|what\s+you\s+(?:buy|own))\b/i;
const WIRED_RELEVANT_CONTEXT_RE = /\b(?:cybersecurity|cybersecurity\s+and\s+infrastructure\s+security\s+agency|cisa|ransomware|malware|spyware|botnet|phishing|data\s+breach|breached|breach|vulnerabilit(?:y|ies)|zero[\s-]*day|cve-\d{4}|hack(?:ed|er|ers|ing)?|password|passkey|vpn|encryption|surveillance|privacy|data\s+broker|identity\s+theft|content\s+moderation|platform\s+governance|ftc|federal\s+trade\s+commission|doj|department\s+of\s+justice|sec|securities\s+and\s+exchange\s+commission|regulator|regulatory|regulation|lawmakers?|congress|senate|lawsuit|settlement|court\s+order|copyright|antitrust|consumer\s+protection)\b/i;
const GENERAL_LITIGATION_RE = /\b(?:lawsuit|litigation|legal\s+battle|court\s+battle|sues|sued|suing|class\s+action|complaint\s+filed|files?\s+(?:a\s+)?suit|filed\s+(?:a\s+)?lawsuit|settlement\s+agreement|settles?\s+(?:lawsuit|claims?|case)|trial\s+opens?|judge\s+(?:rules|rejects|dismisses)|appeals?\s+court)\b/i;
const LITIGATION_ALLOWED_RE = /\b(?:securities|security-based|securities\s+act|exchange\s+act|sec|securities\s+and\s+exchange\s+commission|investor|shareholder|broker-dealer|investment\s+adviser|investment\s+advisor|crypto|cryptocurrency|digital\s+asset|tokenized|token|stablecoin|bitcoin|ethereum|blockchain|defi|fintech|technology|tech|software|platform|artificial\s+intelligence|ai|cybersecurity|data\s+breach|privacy|ransomware|malware|hack(?:ed|er|ers|ing)?|antitrust)\b/i;

function articleText(input: NewsFeedQualityInput): string {
  return decodeEntities([
    input.title,
    input.description,
    input.author,
    input.url,
  ].filter(Boolean).join(" "));
}

function cjkLanguageMatch(value: string): boolean {
  const matches = value.match(CJK_RE) || [];
  if (matches.length === 0) return false;
  const asciiLetters = (value.match(/[a-z]/gi) || []).length;
  const visibleChars = Array.from(value).filter((char) => !/\s/.test(char)).length;
  const ratio = visibleChars > 0 ? matches.length / visibleChars : 0;
  return matches.length >= 4 && (asciiLetters < 6 || matches.length >= asciiLetters || ratio >= 0.18);
}

function metadataText(input: NewsFeedQualityInput): string {
  return decodeEntities([
    input.title,
    input.url,
    ...(Array.isArray(input.tags) ? input.tags : []),
    ...(Array.isArray(input.keywords) ? input.keywords : []),
  ].filter(Boolean).join(" "));
}

export function isWiredSource(input: NewsFeedQualityInput): boolean {
  const feedKey = String(input.feed_key || "").trim().toLowerCase();
  const sourceKind = String(input.source_kind || "").trim().toLowerCase();
  return WIRED_FEED_KEYS.has(feedKey) || WIRED_SOURCE_KINDS.has(sourceKind);
}

export function isLowValueWiredArticle(input: NewsFeedQualityInput): boolean {
  if (!isWiredSource(input)) return false;
  const text = articleText(input);
  return LOW_VALUE_WIRED_RE.test(text) && !WIRED_RELEVANT_CONTEXT_RE.test(text);
}

export function isCjkLanguageArticle(input: NewsFeedQualityInput): boolean {
  return cjkLanguageMatch(articleText(input));
}

export function isDisallowedGeneralLitigationArticle(input: NewsFeedQualityInput): boolean {
  const text = articleText(input);
  return GENERAL_LITIGATION_RE.test(text) && !LITIGATION_ALLOWED_RE.test(text);
}

export function isInvalidWiredCouponArticle(input: NewsFeedQualityInput): boolean {
  if (!isWiredSource(input)) return false;
  return COUPON_SPAM_RE.test(metadataText(input));
}

export function isBlockedNewsFeedDocument(input: NewsFeedQualityInput): boolean {
  return (
    isCjkLanguageArticle(input) ||
    isInvalidWiredCouponArticle(input) ||
    isLowValueWiredArticle(input) ||
    isDisallowedGeneralLitigationArticle(input)
  );
}

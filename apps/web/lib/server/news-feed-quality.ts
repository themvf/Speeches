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
const LOW_VALUE_WIRED_RE = /\b(?:best\s+(?:tv\s+shows?|movies?|streaming|shows?|games?|deals?|sales?|gifts?|gadgets?|phones?|laptops?|headphones?)|tv\s+shows?|movie\s+reviews?|movies?\s+to\s+watch|stream(?:ing)?\s+(?:this\s+month|guide|service|services|platforms?)|netflix|hulu|disney\+|hbo|max|peacock|paramount\+|prime\s+video|apple\s+tv|the\s+onion|infowars\s+parody|parody|satire|celebrity|trailer|comic-con|coupon(?:s|[\s-]*codes?)|promo[\s-]*codes?|discount[\s-]*(?:codes?|coupons?)|prime\s+day|black\s+friday|digital\s+(?:content|media|library|libraries)|what\s+you\s+(?:buy|own))\b/i;
const WIRED_RELEVANT_CONTEXT_RE = /\b(?:cybersecurity|cybersecurity\s+and\s+infrastructure\s+security\s+agency|cisa|ransomware|malware|spyware|botnet|phishing|data\s+breach|breached|breach|vulnerabilit(?:y|ies)|zero[\s-]*day|cve-\d{4}|hack(?:ed|er|ers|ing)?|password|passkey|vpn|encryption|surveillance|privacy|data\s+broker|identity\s+theft|content\s+moderation|platform\s+governance|ftc|federal\s+trade\s+commission|doj|department\s+of\s+justice|sec|securities\s+and\s+exchange\s+commission|regulator|regulatory|regulation|lawmakers?|congress|senate|lawsuit|settlement|court\s+order|copyright|antitrust|consumer\s+protection)\b/i;

function articleText(input: NewsFeedQualityInput): string {
  return decodeEntities([
    input.title,
    input.description,
    input.author,
    input.url,
  ].filter(Boolean).join(" "));
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

export function isInvalidWiredCouponArticle(input: NewsFeedQualityInput): boolean {
  if (!isWiredSource(input)) return false;
  return COUPON_SPAM_RE.test(metadataText(input));
}

export function isBlockedNewsFeedDocument(input: NewsFeedQualityInput): boolean {
  return isInvalidWiredCouponArticle(input) || isLowValueWiredArticle(input);
}

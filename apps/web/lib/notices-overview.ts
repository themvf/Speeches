export type NoticePositionBucket = "supportive" | "neutral" | "opposed" | "mixed" | "unclear";

export interface NoticeOverviewTopic {
  label: string;
  count: number;
  share: number;
}

export interface NoticeOverview {
  total_comments: number;
  enriched_comments: number;
  position_counts: Record<NoticePositionBucket, number>;
  top_topics: NoticeOverviewTopic[];
}

export interface OverviewCommentInput {
  tags?: string[];
  keywords?: string[];
  enrichment_status?: string;
  comment_position?: {
    label?: string;
    confidence?: number;
  };
}

interface CanonicalTopicRule {
  label: string;
  exact?: string[];
  includes?: string[];
}

interface BuildNoticeOverviewOptions {
  minConfidence?: number;
}

const TOP_TOPIC_LIMIT = 5;
const POSITION_BUCKETS: NoticePositionBucket[] = ["supportive", "neutral", "opposed", "mixed", "unclear"];
const ENRICHED_STATUSES = new Set(["enriched", "fallback_enriched", "reviewed"]);
const NOISE_TOPIC_KEYS = new Set([
  "",
  "sec",
  "finra",
  "regulations gov",
  "rule",
  "rules",
  "rule release",
  "rule releases",
  "rulemaking",
  "rulemakings",
  "public comment",
  "public comments",
  "comment",
  "comments",
  "comment letter",
  "comment letters",
  "notice",
  "notices"
]);
const CANONICAL_TOPIC_RULES: CanonicalTopicRule[] = [
  {
    label: "Climate Disclosure",
    exact: ["climate disclosure", "climate disclosures", "climate risk disclosure", "climate related disclosure"],
    includes: ["climate disclosure", "climate-related", "climate risk"]
  },
  {
    label: "Artificial Intelligence",
    exact: ["ai", "artificial intelligence", "machine learning", "generative ai"],
    includes: ["artificial intelligence", "machine learning", "generative ai"]
  },
  {
    label: "Anti-Money Laundering",
    exact: ["aml", "anti money laundering", "anti-money laundering", "bank secrecy act"],
    includes: ["anti money laundering", "bank secrecy act"]
  },
  {
    label: "Digital Assets",
    exact: [
      "crypto",
      "cryptocurrency",
      "cryptocurrencies",
      "digital asset",
      "digital assets",
      "crypto asset",
      "crypto assets",
      "cryptoasset",
      "cryptoassets",
      "stablecoin",
      "stablecoins",
      "tokenization",
      "tokenized securities"
    ],
    includes: ["digital asset", "crypto asset", "cryptocurrency", "stablecoin", "tokenization"]
  },
  {
    label: "Cybersecurity",
    exact: ["cybersecurity", "cyber security", "cyber risk", "cyber risks"],
    includes: ["cybersecurity", "cyber security", "cyber risk"]
  },
  {
    label: "Best Execution",
    exact: ["best execution"],
    includes: ["best execution"]
  },
  {
    label: "Safeguarding Rule",
    exact: ["safeguarding", "safeguarding rule", "custody rule"],
    includes: ["safeguarding rule", "custody rule"]
  },
  {
    label: "Custody",
    exact: ["custody", "qualified custodian", "qualified custodians"],
    includes: ["qualified custodian", "custody"]
  },
  {
    label: "Conflicts of Interest",
    exact: ["conflict of interest", "conflicts of interest"],
    includes: ["conflict of interest"]
  },
  {
    label: "Recordkeeping",
    exact: ["recordkeeping", "record keeping", "books and records"],
    includes: ["recordkeeping", "record keeping", "books and records"]
  },
  {
    label: "Reporting",
    exact: ["reporting", "report", "reporting requirements", "periodic reporting"],
    includes: ["reporting"]
  },
  {
    label: "Disclosure",
    exact: ["disclosure", "disclosures", "disclosure requirements"],
    includes: ["disclosure"]
  },
  {
    label: "Data Privacy",
    exact: ["privacy", "data privacy", "consumer data", "data portability", "data security"],
    includes: ["data privacy", "consumer data", "data portability"]
  }
];

function normalizeText(value: unknown): string {
  return String(value ?? "").replace(/\s+/g, " ").trim();
}

function titleCase(value: string): string {
  return value
    .split(" ")
    .filter(Boolean)
    .map((part) => {
      if (/^[A-Z0-9]{2,}$/.test(part)) {
        return part;
      }
      return part.charAt(0).toUpperCase() + part.slice(1).toLowerCase();
    })
    .join(" ");
}

function normalizedTopicKey(value: unknown): string {
  return normalizeText(value)
    .toLowerCase()
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function shouldIgnoreTopic(key: string): boolean {
  if (!key || NOISE_TOPIC_KEYS.has(key)) {
    return true;
  }
  if (/^(?:file|notice)\s+/.test(key)) {
    return true;
  }
  if (/^s\d+\s+\d{4}\s+\d{2}$/.test(key)) {
    return true;
  }
  return false;
}

function canonicalTopicLabel(value: unknown): string {
  const key = normalizedTopicKey(value);
  if (shouldIgnoreTopic(key)) {
    return "";
  }

  for (const rule of CANONICAL_TOPIC_RULES) {
    if (rule.exact?.includes(key)) {
      return rule.label;
    }
  }

  for (const rule of CANONICAL_TOPIC_RULES) {
    if (rule.includes?.some((fragment) => key.includes(fragment))) {
      return rule.label;
    }
  }

  return titleCase(key);
}

function positionBucket(value: unknown): NoticePositionBucket {
  const normalized = normalizeText(value).toLowerCase();
  if (
    normalized === "supportive" ||
    normalized === "neutral" ||
    normalized === "opposed" ||
    normalized === "mixed"
  ) {
    return normalized;
  }
  return "unclear";
}

function commentTopics(comment: OverviewCommentInput): string[] {
  const preferred =
    Array.isArray(comment.keywords) && comment.keywords.length > 0
      ? comment.keywords
      : Array.isArray(comment.tags)
        ? comment.tags
        : [];
  const seen = new Set<string>();
  const out: string[] = [];

  for (const raw of preferred) {
    const label = canonicalTopicLabel(raw);
    const key = normalizedTopicKey(label);
    if (!label || !key || seen.has(key)) {
      continue;
    }
    seen.add(key);
    out.push(label);
  }

  return out;
}

export function normalizeOverviewConfidence(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(1, value));
}

export function isEnrichedCommentStatus(value: unknown): boolean {
  return ENRICHED_STATUSES.has(normalizeText(value).toLowerCase());
}

export function filterCommentsByConfidence<T extends OverviewCommentInput>(comments: T[], minConfidence = 0): T[] {
  const threshold = normalizeOverviewConfidence(minConfidence);
  if (threshold <= 0) {
    return comments;
  }
  return comments.filter((comment) => {
    const confidence = normalizeOverviewConfidence(Number(comment.comment_position?.confidence ?? 0));
    return confidence >= threshold;
  });
}

export function emptyNoticeOverview(): NoticeOverview {
  return {
    total_comments: 0,
    enriched_comments: 0,
    position_counts: {
      supportive: 0,
      neutral: 0,
      opposed: 0,
      mixed: 0,
      unclear: 0
    },
    top_topics: []
  };
}

export function buildNoticeOverview<T extends OverviewCommentInput>(
  comments: T[],
  options: BuildNoticeOverviewOptions = {}
): NoticeOverview {
  const overview = emptyNoticeOverview();
  const topicCounts = new Map<string, { label: string; count: number }>();
  const filtered = filterCommentsByConfidence(comments, options.minConfidence ?? 0);

  overview.total_comments = filtered.length;

  for (const comment of filtered) {
    if (isEnrichedCommentStatus(comment.enrichment_status)) {
      overview.enriched_comments += 1;
    }

    overview.position_counts[positionBucket(comment.comment_position?.label)] += 1;

    for (const label of commentTopics(comment)) {
      const key = normalizedTopicKey(label);
      const current = topicCounts.get(key);
      if (current) {
        current.count += 1;
      } else {
        topicCounts.set(key, { label, count: 1 });
      }
    }
  }

  overview.top_topics = [...topicCounts.values()]
    .sort((a, b) => b.count - a.count || a.label.localeCompare(b.label))
    .slice(0, TOP_TOPIC_LIMIT)
    .map((item) => ({
      label: item.label,
      count: item.count,
      share: overview.total_comments > 0 ? item.count / overview.total_comments : 0
    }));

  for (const bucket of POSITION_BUCKETS) {
    overview.position_counts[bucket] ||= 0;
  }

  return overview;
}

export function confidenceFilterLabel(minConfidence: number): string {
  const threshold = normalizeOverviewConfidence(minConfidence);
  if (threshold <= 0) {
    return "All comments";
  }
  return `${Math.round(threshold * 100)}%+ confidence`;
}

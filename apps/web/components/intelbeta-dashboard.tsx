"use client";

import { Fragment, useCallback, useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import type { StoredRssArticle, StoredRssTopicRule } from "@/lib/server/neon";
import type { DocumentListItem } from "@/lib/server/types";
import { BookmarkButton } from "@/components/bookmark-button";
import { useSavedItems } from "@/hooks/use-saved-items";
import {
  decodeEntities,
  getMatchingTopics,
  normalizeTopicRules,
  type TopicRuleView,
} from "@/lib/intel-topic-matching";

type TopicFilter = string | "ALL";
type SidebarTab = "topics" | "sources";
type SourceFilter = "ALL" | "SEC" | "FINRA" | "DOJ" | "WSJ";

type FeedMeta = {
  label: string;
  code: string;
  color: string;
};

type FeedItem = StoredRssArticle & {
  item_type?: "article" | "document";
  document_id?: string;
  organization?: string;
  source_kind?: string;
  doc_type?: string;
  topics?: string[];
  keywords?: string[];
  analysis?: unknown;
};

interface ApiEnvelope<T> {
  ok: boolean;
  data?: T;
  error?: string;
}

interface DocumentDetailData {
  metadata: {
    document_id: string;
    published_at: string;
  };
  content: {
    full_text: string;
    paragraphs: string[];
    sentences: string[];
  };
  enrichment: {
    status: string;
    summary: string;
    tags: string[];
    keywords: string[];
    entities: string[];
    evidence_spans: Array<Record<string, unknown>>;
    stance: Record<string, unknown>;
    comment_position: Record<string, unknown>;
    confidence: number;
  };
  review: {
    decision: string;
    notes: string;
    reviewed_at: string;
  };
  sentiment: {
    score: number;
    label: string;
    rationale: string;
    status: string;
  } | null;
}

interface FeedItemAnalysis {
  thesis: string;
  why_it_matters: string[];
  risk_signals: string[];
  follow_up_questions: string[];
  keywords: string[];
  individuals: string[];
  entities: string[];
  model: string;
  generated_at: string;
  fallback: boolean;
}

function toFeedItemAnalysis(value: unknown): FeedItemAnalysis | undefined {
  if (!value || typeof value !== "object") return undefined;
  const src = value as Partial<FeedItemAnalysis>;
  const thesis = String(src.thesis || "").trim();
  if (!thesis) return undefined;
  return {
    thesis,
    why_it_matters: Array.isArray(src.why_it_matters) ? src.why_it_matters.map(String).filter(Boolean) : [],
    risk_signals: Array.isArray(src.risk_signals) ? src.risk_signals.map(String).filter(Boolean) : [],
    follow_up_questions: Array.isArray(src.follow_up_questions) ? src.follow_up_questions.map(String).filter(Boolean) : [],
    keywords: Array.isArray(src.keywords) ? src.keywords.map(String).filter(Boolean) : [],
    individuals: Array.isArray(src.individuals) ? src.individuals.map(String).filter(Boolean) : [],
    entities: Array.isArray(src.entities) ? src.entities.map(String).filter(Boolean) : [],
    model: String(src.model || ""),
    generated_at: String(src.generated_at || ""),
    fallback: Boolean(src.fallback),
  };
}

const FEED_META: Record<string, FeedMeta> = {
  wsj_us_business: { label: "WSJ Business", code: "WSJB", color: "#63a8ff" },
  wsj_markets: { label: "WSJ Markets", code: "WSJM", color: "#ffc857" },
  wsj_opinion: { label: "WSJ Opinion", code: "WSJO", color: "#b88fff" },
  mw_top_stories: { label: "MarketWatch", code: "MW", color: "#4dd39f" },
  rss_nytimes_com_services_xml_rss_nyt_business_xml: { label: "NYT Business", code: "NYTB", color: "#ffe066" },
  rss_nytimes_com_services_xml_rss_nyt_technology_xml: { label: "NYT Tech", code: "NYTT", color: "#74c0fc" },
  rss_nytimes_com_services_xml_rss_nyt_politics_xml: { label: "NYT Politics", code: "NYTP", color: "#ff8787" },
  coindesk: { label: "CoinDesk", code: "CDSK", color: "#f0b90b" },
  cointelegraph: { label: "Cointelegraph", code: "CTEL", color: "#f3c969" },
  decrypt: { label: "Decrypt", code: "DECR", color: "#8ce99a" },
  the_block: { label: "The Block", code: "BLCK", color: "#63e6be" },
  cisa_cybersecurity_advisories: { label: "CISA", code: "CISA", color: "#4dabf7" },
  bleepingcomputer: { label: "BleepingComputer", code: "BLC", color: "#74c0fc" },
  krebs_on_security: { label: "Krebs on Security", code: "KRBS", color: "#ffa8a8" },
  the_hacker_news: { label: "The Hacker News", code: "THN", color: "#ff8787" },
  dark_reading: { label: "Dark Reading", code: "DARK", color: "#b197fc" },
  securityweek: { label: "SecurityWeek", code: "SECW", color: "#91a7ff" },
  microsoft_security_blog: { label: "Microsoft Security Blog", code: "MSFT", color: "#69db7c" },
};

const SOURCE_FILTERS: Array<{ key: Exclude<SourceFilter, "ALL">; label: string }> = [
  { key: "SEC", label: "SEC" },
  { key: "FINRA", label: "FINRA" },
  { key: "DOJ", label: "DOJ" },
  { key: "WSJ", label: "WSJ" },
];

function getFeedMeta(feedKey: string): FeedMeta {
  if (feedKey.startsWith("document_")) {
    const sourceKind = feedKey.replace(/^document_/, "");
    const normalized = sourceKind.replace(/[_-]+/g, " ").trim();
    const code = normalized
      .split(" ")
      .map((word) => word.charAt(0))
      .join("")
      .toUpperCase()
      .slice(0, 4) || "DOC";
    return {
      label: normalized.replace(/\b\w/g, (ch) => ch.toUpperCase()) || "Document",
      code,
      color: "#4fd5ff",
    };
  }
  return FEED_META[feedKey] ?? {
    label: feedKey,
    code: feedKey.slice(0, 4).toUpperCase(),
    color: "#8fa7c8",
  };
}

function feedSourceLabel(article: FeedItem, source: FeedMeta): string {
  return decodeEntities(article.organization || source.label || article.feed_key || "Unknown");
}

function articleSourceText(article: FeedItem): string {
  const source = getFeedMeta(article.feed_key);
  return [
    article.feed_key,
    source.label,
    article.organization ?? "",
    article.source_kind ?? "",
    article.doc_type ?? "",
    article.url ?? "",
  ].join(" ").toLowerCase();
}

function matchesSourceFilter(article: FeedItem, sourceFilter: SourceFilter): boolean {
  if (sourceFilter === "ALL") return true;
  const text = articleSourceText(article);
  if (sourceFilter === "SEC") {
    return (
      text.includes("sec_") ||
      text.includes("document_sec") ||
      text.includes("securities and exchange commission") ||
      /\bsec\b/.test(text)
    );
  }
  if (sourceFilter === "FINRA") {
    return text.includes("finra") || text.includes("financial industry regulatory authority");
  }
  if (sourceFilter === "DOJ") {
    return (
      text.includes("doj") ||
      text.includes("justice.gov") ||
      text.includes("department of justice") ||
      text.includes("justice department")
    );
  }
  return text.includes("wsj_") || text.includes("wall street journal") || text.includes("wsj.com") || text.includes("dowjones");
}

function savedArticleId(article: FeedItem): string {
  if (article.item_type === "document" && article.document_id) {
    return `document:${article.document_id}`;
  }
  return `article:${article.id}`;
}

const TONE_STYLE: Record<string, { color: string; bg: string; label: string; short: string; glyph: string }> = {
  positive: {
    color: "#41d39d",
    bg: "rgba(65, 211, 157, 0.12)",
    label: "Bullish",
    short: "POS",
    glyph: "▲",
  },
  negative: {
    color: "#ff6b7f",
    bg: "rgba(255, 107, 127, 0.12)",
    label: "Bearish",
    short: "NEG",
    glyph: "▼",
  },
  neutral: {
    color: "#8b95a1",
    bg: "rgba(139, 149, 161, 0.10)",
    label: "Neutral",
    short: "NEU",
    glyph: "◆",
  },
};

function matchesTopic(article: FeedItem, rule: TopicRuleView | null, topicMatchesByArticleId: Map<number, TopicRuleView[]>): boolean {
  if (!rule) return true;
  return (topicMatchesByArticleId.get(article.id) ?? []).some((item) => item.topic_key === rule.topic_key);
}

function matchesSearch(article: FeedItem, searchTerm: string): boolean {
  if (!searchTerm) return true;
  const haystack = [
    article.title,
    article.description ?? "",
    article.author ?? "",
    article.organization ?? "",
    article.source_kind ?? "",
    article.doc_type ?? "",
    ...(article.topics ?? []),
    ...(article.keywords ?? []),
  ].join(" ").toLowerCase();
  return haystack.includes(searchTerm);
}

function formatRelativeTime(dateStr: string | null): string {
  if (!dateStr) return "";
  const ms = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(ms / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return new Date(dateStr).toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

function feedItemDate(article: Pick<FeedItem, "published_at" | "fetched_at">): string {
  return article.published_at || article.fetched_at || "";
}

function feedItemDateMs(article: Pick<FeedItem, "published_at" | "fetched_at">): number {
  const ms = new Date(feedItemDate(article)).getTime();
  return Number.isFinite(ms) ? ms : 0;
}

function formatClock(date: Date): string {
  return date.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
}

function formatUpdated(dateStr: string | null): string {
  if (!dateStr) return "";
  return new Date(dateStr).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
}

function ellipsize(text: string, max = 120): string {
  const value = decodeEntities(text || "");
  return value.length > max ? `${value.slice(0, max - 1).trimEnd()}…` : value;
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });
  const payload = (await res.json()) as ApiEnvelope<T>;
  if (!res.ok || !payload.ok || payload.data === undefined) {
    throw new Error(payload.error || `Request failed with ${res.status}`);
  }
  return payload.data;
}

function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const media = window.matchMedia(query);
    const update = () => setMatches(media.matches);
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, [query]);

  return matches;
}

function statusClass(value: string): string {
  const s = String(value || "").toLowerCase();
  if (["enriched", "reviewed", "success"].includes(s)) return "status-chip status-success";
  if (["fallback_enriched", "queued", "running"].includes(s)) return "status-chip status-warn";
  if (["failed", "rejected"].includes(s)) return "status-chip status-failure";
  return "status-chip status-neutral";
}

function analysisChipClass(value: string): string {
  const label = String(value || "").toLowerCase();
  if (["supportive", "supports", "aligned", "favorable", "positive"].includes(label)) return "status-chip status-success";
  if (["opposed", "opposes", "critical", "negative", "adverse"].includes(label)) return "status-chip status-failure";
  if (["mixed", "qualified", "partially_supportive"].includes(label)) return "status-chip status-warn";
  return "status-chip status-neutral";
}

function formatAnalysisLabel(value: string): string {
  const normalized = String(value || "").trim();
  if (!normalized) return "";
  return normalized
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function readStringField(value: unknown, key: string): string {
  if (!value || typeof value !== "object") return "";
  const out = (value as Record<string, unknown>)[key];
  return typeof out === "string" ? out.trim() : "";
}

function readNumberField(value: unknown, key: string): number {
  if (!value || typeof value !== "object") return 0;
  const out = Number.parseFloat(String((value as Record<string, unknown>)[key] ?? "0"));
  return Number.isFinite(out) ? out : 0;
}

function pickPrimaryAnalysis(detail: DocumentDetailData | null | undefined): {
  kind: "position" | "stance" | "summary" | "";
  label: string;
  tone: string;
  rationale: string;
  confidence: number;
} {
  if (!detail) return { kind: "", label: "", tone: "", rationale: "", confidence: 0 };

  const positionLabel = readStringField(detail.enrichment.comment_position, "label").toLowerCase();
  const positionRationale = readStringField(detail.enrichment.comment_position, "rationale");
  const positionConfidence = readNumberField(detail.enrichment.comment_position, "confidence");
  if (positionLabel && positionLabel !== "not_applicable" && positionLabel !== "unclear") {
    return {
      kind: "position",
      label: positionLabel,
      tone: positionLabel,
      rationale: positionRationale,
      confidence: Math.max(0, Math.min(1, positionConfidence)),
    };
  }

  const stanceLabel = readStringField(detail.enrichment.stance, "label").toLowerCase();
  const stanceTarget = readStringField(detail.enrichment.stance, "target");
  if (stanceLabel && stanceLabel !== "unclear" && stanceLabel !== "not_applicable") {
    return {
      kind: "stance",
      label: stanceTarget ? `${stanceLabel} (${stanceTarget})` : stanceLabel,
      tone: stanceLabel,
      rationale: "",
      confidence: Math.max(0, Math.min(1, Number(detail.enrichment.confidence || 0))),
    };
  }

  if (detail.enrichment.summary) {
    return { kind: "summary", label: "summary_ready", tone: "summary_ready", rationale: "", confidence: 0 };
  }

  return { kind: "", label: "", tone: "", rationale: "", confidence: 0 };
}

function renderAnalysisChips(items: string[], emptyLabel: string) {
  if (!items.length) {
    return <span style={{ color: "#6f819d", fontSize: 12 }}>{emptyLabel}</span>;
  }
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
      {items.slice(0, 8).map((item) => (
        <span key={item} className="tone-chip">
          {item}
        </span>
      ))}
    </div>
  );
}

function articleListSignature(articles: FeedItem[]): string {
  const first = articles[0];
  const last = articles[articles.length - 1];
  const firstDate = first ? feedItemDate(first) : "";
  const lastDate = last ? feedItemDate(last) : "";
  return `${articles.length}:${first?.id ?? ""}:${firstDate}:${last?.id ?? ""}:${lastDate}`;
}

function analysisMapFromFeedItems(items: FeedItem[]): Record<string, FeedItemAnalysis> {
  const out: Record<string, FeedItemAnalysis> = {};
  for (const item of items) {
    const analysis = toFeedItemAnalysis(item.analysis);
    if (analysis) {
      out[savedArticleId(item)] = analysis;
    }
  }
  return out;
}

function stableNegativeId(value: string): number {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) {
    hash = ((hash << 5) - hash + value.charCodeAt(i)) | 0;
  }
  return -Math.max(1, Math.abs(hash));
}

function documentDescription(document: DocumentListItem): string {
  return [
    document.doc_type,
    document.speaker ? `By ${document.speaker}` : "",
    (document.topics || []).slice(0, 5).join(", "),
    (document.keywords || []).slice(0, 5).join(", "),
  ].filter(Boolean).join(" | ");
}

function documentToFeedItem(document: DocumentListItem): FeedItem {
  const publishedAt = document.published_at || document.date || new Date(0).toISOString();
  return {
    id: stableNegativeId(document.document_id),
    guid: document.document_id,
    feed_key: `document_${document.source_kind || "document"}`,
    title: document.title || "Untitled document",
    url: document.url,
    description: documentDescription(document),
    author: document.speaker || document.organization || "Document",
    published_at: publishedAt,
    tone_label: document.sentiment_label || null,
    fetched_at: publishedAt,
    item_type: "document",
    document_id: document.document_id,
    organization: document.organization,
    source_kind: document.source_kind,
    doc_type: document.doc_type,
    topics: document.topics || [],
    keywords: document.keywords || [],
  };
}

function topicRulesSignature(rules: StoredRssTopicRule[]): string {
  return rules
    .map((rule) => `${rule.id}:${rule.topic_key}:${rule.active}:${rule.sort_order}:${rule.updated_at}:${rule.keywords}`)
    .join("|");
}

function TopicPill({ label }: { label: string }) {
  return (
    <span
      style={{
        border: "1px solid rgba(93, 123, 171, 0.32)",
        borderRadius: 4,
        padding: "2px 6px",
        fontSize: 10,
        lineHeight: 1.2,
        letterSpacing: "0.08em",
        color: "#8fa7c8",
        textTransform: "uppercase",
        whiteSpace: "nowrap",
      }}
    >
      {label}
    </span>
  );
}

function ToneChip({ label }: { label: string | null }) {
  const tone = label && TONE_STYLE[label] ? label : "neutral";
  const style = TONE_STYLE[tone];
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 5,
        color: style.color,
        fontSize: 11,
        letterSpacing: "0.10em",
        textTransform: "uppercase",
        whiteSpace: "nowrap",
      }}
    >
      <span>{style.glyph}</span>
      <span>{style.short}</span>
    </span>
  );
}

function TopicButton({
  label,
  active,
  onClick,
  count,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
  count: number;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        width: "100%",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 8,
        textAlign: "left",
        border: "none",
        borderLeft: active ? "2px solid #63a8ff" : "2px solid transparent",
        background: active ? "rgba(67, 112, 186, 0.18)" : "transparent",
        color: active ? "#e6eef9" : "#9ba9bc",
        borderRadius: 4,
        padding: "7px 10px 7px 8px",
        cursor: "pointer",
        fontSize: 13,
        transition: "background 120ms ease, color 120ms ease, border-color 120ms ease",
      }}
      onMouseEnter={(e) => {
        if (!active) {
          (e.currentTarget as HTMLElement).style.background = "rgba(67, 112, 186, 0.08)";
          (e.currentTarget as HTMLElement).style.color = "#dbe6f4";
        }
      }}
      onMouseLeave={(e) => {
        if (!active) {
          (e.currentTarget as HTMLElement).style.background = "transparent";
          (e.currentTarget as HTMLElement).style.color = "#9ba9bc";
        }
      }}
    >
      <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{label}</span>
      <span style={{ color: active ? "#8fb2ea" : "#64728a", fontSize: 12 }}>{count}</span>
    </button>
  );
}

function FeedRow({
  article,
  matchedTopics,
  active,
  onSelect,
  saved,
  onToggleSave,
  analysisOpen,
  analysisLabel,
  onToggleAnalysis,
  compact = false,
}: {
  article: FeedItem;
  matchedTopics: TopicRuleView[];
  active: boolean;
  onSelect: () => void;
  saved: boolean;
  onToggleSave: () => void;
  analysisOpen: boolean;
  analysisLabel: string;
  onToggleAnalysis: () => void;
  compact?: boolean;
}) {
  const source = getFeedMeta(article.feed_key);
  const sourceLabel = feedSourceLabel(article, source);
  const visibleTopics = matchedTopics.slice(0, 3);
  const description = ellipsize(article.description ?? "", article.item_type === "document" ? 120 : 82);
  const analysisButtonStyle = {
    border: analysisOpen ? "1px solid rgba(79,213,255,0.55)" : "1px solid rgba(90,118,162,0.28)",
    background: analysisOpen ? "rgba(79,213,255,0.12)" : "rgba(14,24,39,0.58)",
    color: analysisOpen ? "#e8f7ff" : "#9fb0c7",
    borderRadius: 999,
    padding: compact ? "8px 12px" : "4px 9px",
    fontSize: compact ? 11 : 10,
    fontWeight: 700,
    letterSpacing: "0.06em",
    textTransform: "uppercase" as const,
    cursor: "pointer",
  };

  if (compact) {
    return (
      <div
        style={{
          display: "grid",
          gap: 10,
          padding: "12px 0",
          borderTop: "1px solid rgba(112, 142, 187, 0.12)",
          background: active ? "rgba(67, 112, 186, 0.08)" : "transparent",
          cursor: "pointer",
          minWidth: 0,
        }}
        onClick={onSelect}
      >
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 10 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", minWidth: 0 }}>
            <span style={{ color: "#7f8faa", fontSize: 12, whiteSpace: "nowrap" }}>{formatRelativeTime(feedItemDate(article))}</span>
            <span style={{ color: source.color, fontSize: 12, fontWeight: 700 }}>{sourceLabel}</span>
            <ToneChip label={article.tone_label} />
          </div>
          <div onClick={(e) => e.stopPropagation()}>
            <BookmarkButton saved={saved} onToggle={onToggleSave} />
          </div>
        </div>

        <div style={{ minWidth: 0 }}>
          <a
            href={article.url}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              color: "#edf3fb",
              fontSize: 15,
              fontWeight: 650,
              textDecoration: "none",
              lineHeight: 1.35,
            }}
          >
            {decodeEntities(article.title)}
          </a>
          {description ? (
            <div style={{ color: "#7f8faa", fontSize: 12, marginTop: 5, lineHeight: 1.45 }}>{description}</div>
          ) : null}
          {article.item_type === "document" ? (
            <div style={{ color: "#4fd5ff", fontSize: 10, marginTop: 6, letterSpacing: "0.12em", textTransform: "uppercase" }}>
              Primary document
            </div>
          ) : null}
        </div>

        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {visibleTopics.map((topic) => (
            <TopicPill key={`${article.id}_${topic.topic_key}`} label={topic.label} />
          ))}
        </div>

        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onToggleAnalysis();
          }}
          style={{ ...analysisButtonStyle, justifySelf: "start" }}
        >
          {analysisLabel}
        </button>
      </div>
    );
  }

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "80px 150px 66px minmax(0, 1fr) 220px 24px",
        gap: 14,
        alignItems: "start",
        padding: "10px 0",
        borderTop: "1px solid rgba(112, 142, 187, 0.12)",
        background: active ? "rgba(67, 112, 186, 0.08)" : "transparent",
        cursor: "pointer",
      }}
      onClick={onSelect}
    >
      <div style={{ color: "#7f8faa", fontSize: 12, whiteSpace: "nowrap" }}>{formatRelativeTime(feedItemDate(article))}</div>
      <div style={{ color: source.color, fontSize: 12, fontWeight: 700, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={sourceLabel}>
        {sourceLabel}
      </div>
      <ToneChip label={article.tone_label} />
      <div style={{ minWidth: 0 }}>
        <a
          href={article.url}
          target="_blank"
          rel="noopener noreferrer"
          style={{
            color: "#edf3fb",
            fontSize: 15,
            fontWeight: 600,
            textDecoration: "none",
            lineHeight: 1.35,
          }}
        >
          {decodeEntities(article.title)}
        </a>
        {description ? (
          <div style={{ color: "#7f8faa", fontSize: 12, marginTop: 3, lineHeight: 1.45 }}>{description}</div>
        ) : null}
        {article.item_type === "document" ? (
          <div style={{ color: "#4fd5ff", fontSize: 10, marginTop: 5, letterSpacing: "0.12em", textTransform: "uppercase" }}>
            Primary document
          </div>
        ) : null}
        <div style={{ marginTop: 8 }}>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onToggleAnalysis();
            }}
            style={analysisButtonStyle}
          >
            {analysisLabel}
          </button>
        </div>
      </div>
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", justifyContent: "flex-end" }}>
        {visibleTopics.map((topic) => (
          <TopicPill key={`${article.id}_${topic.topic_key}`} label={topic.label} />
        ))}
      </div>
      <BookmarkButton saved={saved} onToggle={onToggleSave} />
    </div>
  );
}

function FeaturedCard({
  article,
  matchedTopics,
  saved,
  onToggleSave,
  analysisOpen,
  analysisLabel,
  onToggleAnalysis,
  compact = false,
}: {
  article: FeedItem;
  matchedTopics: TopicRuleView[];
  saved: boolean;
  onToggleSave: () => void;
  analysisOpen: boolean;
  analysisLabel: string;
  onToggleAnalysis: () => void;
  compact?: boolean;
}) {
  const source = getFeedMeta(article.feed_key);
  const sourceLabel = feedSourceLabel(article, source);
  const tone = article.tone_label && TONE_STYLE[article.tone_label] ? article.tone_label : "neutral";
  const analysisButtonStyle = {
    border: analysisOpen ? "1px solid rgba(79,213,255,0.55)" : "1px solid rgba(90,118,162,0.28)",
    background: analysisOpen ? "rgba(79,213,255,0.12)" : "rgba(14,24,39,0.58)",
    color: analysisOpen ? "#e8f7ff" : "#9fb0c7",
    borderRadius: 999,
    padding: compact ? "8px 12px" : "5px 10px",
    fontSize: compact ? 11 : 11,
    fontWeight: 700,
    letterSpacing: "0.06em",
    textTransform: "uppercase" as const,
    cursor: "pointer",
  };

  if (compact) {
    return (
      <div
        style={{
          borderTop: "1px solid rgba(112, 142, 187, 0.16)",
          borderBottom: "1px solid rgba(112, 142, 187, 0.16)",
          padding: "14px 0 16px",
          marginBottom: 4,
          display: "grid",
          gap: 11,
          minWidth: 0,
        }}
      >
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 10 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", minWidth: 0 }}>
            <span style={{ color: "#8fa7c8", fontSize: 12, whiteSpace: "nowrap" }}>{formatRelativeTime(feedItemDate(article))}</span>
            <span style={{ color: source.color, fontSize: 12, fontWeight: 700 }}>{sourceLabel}</span>
            <ToneChip label={tone} />
          </div>
          <BookmarkButton saved={saved} onToggle={onToggleSave} size={16} />
        </div>

        <div style={{ minWidth: 0 }}>
          <a
            href={article.url}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              color: "#f4f7fc",
              fontWeight: 700,
              fontSize: 16,
              lineHeight: 1.38,
              textDecoration: "none",
            }}
          >
            {decodeEntities(article.title)}
          </a>
          {article.description ? (
            <div style={{ marginTop: 9, color: "#b8c6d8", fontSize: 13, lineHeight: 1.55 }}>
              {ellipsize(article.description, 180)}
            </div>
          ) : null}
        </div>

        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {matchedTopics.slice(0, 4).map((topic) => (
            <TopicPill key={`${article.id}_${topic.topic_key}`} label={topic.label} />
          ))}
        </div>

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", color: "#8da0bc", fontSize: 11 }}>
          <span>{decodeEntities(article.author || (article.item_type === "document" ? "Document" : "News Desk"))}</span>
          <span aria-hidden="true">/</span>
          <span>{sourceLabel}</span>
          <span aria-hidden="true">/</span>
          <span style={{ color: TONE_STYLE[tone].color, fontWeight: 700 }}>{TONE_STYLE[tone].label}</span>
        </div>

        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onToggleAnalysis();
          }}
          style={{ ...analysisButtonStyle, justifySelf: "start" }}
        >
          {analysisLabel}
        </button>
      </div>
    );
  }

  return (
    <div
      style={{
        borderTop: "1px solid rgba(112, 142, 187, 0.16)",
        borderBottom: "1px solid rgba(112, 142, 187, 0.16)",
        padding: "14px 0 18px",
        marginBottom: 4,
      }}
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "80px 150px 66px minmax(0, 1fr) 240px 24px",
          gap: 14,
          alignItems: "start",
        }}
      >
        <div style={{ color: "#8fa7c8", fontSize: 12 }}>{formatRelativeTime(feedItemDate(article))}</div>
        <div style={{ color: source.color, fontSize: 12, fontWeight: 700, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={sourceLabel}>
          {sourceLabel}
        </div>
        <ToneChip label={tone} />
        <div style={{ minWidth: 0 }}>
          <a
            href={article.url}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              color: "#f4f7fc",
              fontWeight: 700,
              fontSize: 16,
              lineHeight: 1.4,
              textDecoration: "none",
            }}
          >
            {decodeEntities(article.title)}
          </a>
          {article.description ? (
            <div style={{ marginTop: 14 }}>
              <div
                style={{
                  color: "#6e7e98",
                  fontSize: 10,
                  letterSpacing: "0.18em",
                  textTransform: "uppercase",
                  marginBottom: 8,
                }}
              >
                Why It Matters
              </div>
              <div
                style={{
                  color: "#dce7f7",
                  fontSize: 16,
                  lineHeight: 1.65,
                  fontStyle: "italic",
                  fontFamily: '"Iowan Old Style", "Palatino Linotype", serif',
                }}
              >
                {decodeEntities(article.description)}
              </div>
            </div>
          ) : null}
        </div>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "74px minmax(0, 1fr)",
            gap: "6px 10px",
            fontSize: 11,
            color: "#8da0bc",
            alignSelf: "stretch",
          }}
        >
          <div style={{ letterSpacing: "0.12em", textTransform: "uppercase", color: "#5e708a" }}>Author</div>
          <div style={{ color: "#d7e1ef" }}>{decodeEntities(article.author || (article.item_type === "document" ? "Document" : "News Desk"))}</div>
          <div style={{ letterSpacing: "0.12em", textTransform: "uppercase", color: "#5e708a" }}>Source</div>
          <div style={{ color: "#d7e1ef" }}>{sourceLabel}</div>
          <div style={{ letterSpacing: "0.12em", textTransform: "uppercase", color: "#5e708a" }}>Impact</div>
          <div style={{ color: TONE_STYLE[tone].color, fontWeight: 700 }}>{TONE_STYLE[tone].label.toUpperCase()}</div>
          <div style={{ letterSpacing: "0.12em", textTransform: "uppercase", color: "#5e708a" }}>Topics</div>
          <div style={{ color: "#d7e1ef" }}>
            {matchedTopics.length > 0 ? matchedTopics.map((topic) => topic.label).join(", ") : "Unmapped"}
          </div>
        </div>
        <BookmarkButton saved={saved} onToggle={onToggleSave} size={16} />
      </div>
      <div style={{ display: "flex", gap: 8, marginTop: 12, paddingLeft: 214 }}>
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onToggleAnalysis();
          }}
          style={analysisButtonStyle}
        >
          {analysisLabel}
        </button>
      </div>
    </div>
  );
}

function FeedAnalysisPanel({
  article,
  matchedTopics,
  analysis,
  analysisLoading,
  analysisError,
  retryAnalysis,
  detail,
  loading,
  error,
  retry,
  compact = false,
}: {
  article: FeedItem;
  matchedTopics: TopicRuleView[];
  analysis: FeedItemAnalysis | undefined;
  analysisLoading: boolean;
  analysisError: string;
  retryAnalysis: () => void;
  detail: DocumentDetailData | undefined;
  loading: boolean;
  error: string;
  retry: () => void;
  compact?: boolean;
}) {
  const source = getFeedMeta(article.feed_key);
  const sourceLabel = feedSourceLabel(article, source);
  const primaryAnalysis = pickPrimaryAnalysis(detail);
  const tone = article.tone_label && TONE_STYLE[article.tone_label] ? article.tone_label : "neutral";
  const decodedDescription = decodeEntities(article.description || "");
  const topicLabels = matchedTopics.map((topic) => topic.label);
  const articleSummary = decodedDescription || "No feed summary is available for this article yet.";
  const analysisModel = analysis?.fallback ? `${analysis.model} fallback` : analysis?.model;
  const mainGridColumns = compact ? "1fr" : "minmax(0,1.45fr) minmax(220px,0.55fr)";
  const twoColumnGrid = compact ? "1fr" : "repeat(2, minmax(0, 1fr))";
  const threeColumnGrid = compact ? "1fr" : "repeat(3, minmax(0, 1fr))";
  const analysisBlock = analysisLoading ? (
    <p style={{ color: "#9fb0c7", fontSize: 13 }}>Generating analysis...</p>
  ) : analysisError ? (
    <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
      <p style={{ color: "#ff8aa0", fontSize: 13 }}>{analysisError}</p>
      <button type="button" className="link-inline text-xs" onClick={retryAnalysis}>
        Retry
      </button>
    </div>
  ) : analysis ? (
    <div style={{ display: "grid", gap: 14 }}>
      <div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
          <span className={analysisChipClass(tone)}>Tone: {TONE_STYLE[tone].label}</span>
          <span className="tone-chip">Source: {sourceLabel}</span>
          {analysisModel ? <span className="tone-chip">Model: {analysisModel}</span> : null}
        </div>
        <p style={{ marginTop: 10, color: "#dbe7f5", fontSize: 14, fontWeight: 700, lineHeight: 1.55 }}>
          {analysis.thesis}
        </p>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: twoColumnGrid, gap: 14 }}>
        <div>
          <p style={{ marginBottom: 6, color: "#60738f", fontSize: 10, fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase" }}>
            Why It Matters
          </p>
          <ul style={{ margin: 0, paddingLeft: 16, color: "#b7c7dc", fontSize: 12, lineHeight: 1.6 }}>
            {analysis.why_it_matters.map((item) => <li key={item}>{item}</li>)}
          </ul>
        </div>
        <div>
          <p style={{ marginBottom: 6, color: "#60738f", fontSize: 10, fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase" }}>
            Risk Signals
          </p>
          <ul style={{ margin: 0, paddingLeft: 16, color: "#b7c7dc", fontSize: 12, lineHeight: 1.6 }}>
            {analysis.risk_signals.map((item) => <li key={item}>{item}</li>)}
          </ul>
        </div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: threeColumnGrid, gap: 14 }}>
        <div>
          <p style={{ marginBottom: 6, color: "#60738f", fontSize: 10, fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase" }}>
            Keywords
          </p>
          {renderAnalysisChips(analysis.keywords, "No keywords extracted")}
        </div>
        <div>
          <p style={{ marginBottom: 6, color: "#60738f", fontSize: 10, fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase" }}>
            Individuals
          </p>
          {renderAnalysisChips(analysis.individuals, "No individuals identified")}
        </div>
        <div>
          <p style={{ marginBottom: 6, color: "#60738f", fontSize: 10, fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase" }}>
            Entities
          </p>
          {renderAnalysisChips(analysis.entities, "No entities identified")}
        </div>
      </div>
      <div>
        <p style={{ marginBottom: 6, color: "#60738f", fontSize: 10, fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase" }}>
          Follow-Up
        </p>
        <ul style={{ margin: 0, paddingLeft: 16, color: "#8ea0ba", fontSize: 12, lineHeight: 1.6 }}>
          {analysis.follow_up_questions.map((item) => <li key={item}>{item}</li>)}
        </ul>
      </div>
    </div>
  ) : (
    <p style={{ color: "#7f8faa", fontSize: 13 }}>Open analysis is preparing this item.</p>
  );

  if (article.item_type === "document") {
    if (loading) {
      return <p style={{ color: "#9fb0c7", fontSize: 13 }}>Loading analysis...</p>;
    }

    if (error) {
      return (
        <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          <p style={{ color: "#ff8aa0", fontSize: 13 }}>{error}</p>
          <button type="button" className="link-inline text-xs" onClick={retry}>
            Retry
          </button>
        </div>
      );
    }

    if (!detail) {
      return <p style={{ color: "#7f8faa", fontSize: 13 }}>No analysis is available for this document.</p>;
    }

    return (
      <div style={{ display: "grid", gap: 18 }}>
        {analysisBlock}
        <div style={{ height: 1, background: "rgba(112, 142, 187, 0.12)" }} />
        <div style={{ display: "grid", gridTemplateColumns: mainGridColumns, gap: compact ? 14 : 18 }}>
        <div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            <span className={statusClass(detail.enrichment.status)}>{detail.enrichment.status || "not_enriched"}</span>
            <span className="tone-chip">Review: {detail.review.decision || "pending"}</span>
            {primaryAnalysis.kind === "position" ? (
              <span className={analysisChipClass(primaryAnalysis.tone)}>
                Position: {formatAnalysisLabel(primaryAnalysis.label)}
              </span>
            ) : primaryAnalysis.kind === "stance" ? (
              <span className={analysisChipClass(primaryAnalysis.tone)}>
                Stance: {formatAnalysisLabel(primaryAnalysis.label)}
              </span>
            ) : null}
            {primaryAnalysis.confidence > 0 ? (
              <span className="tone-chip">Confidence: {Math.round(primaryAnalysis.confidence * 100)}%</span>
            ) : null}
          </div>
          <p style={{ marginTop: 10, color: "#c6d4e6", fontSize: 13, lineHeight: 1.65 }}>
            {detail.enrichment.summary || "No summary is available for this document yet."}
          </p>
          {primaryAnalysis.rationale ? (
            <p style={{ marginTop: 8, color: "#8899b1", fontSize: 12, lineHeight: 1.55 }}>{primaryAnalysis.rationale}</p>
          ) : null}
        </div>
        <div style={{ display: "grid", gap: 12 }}>
          <div>
            <p style={{ marginBottom: 6, color: "#60738f", fontSize: 10, fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase" }}>
              Tags
            </p>
            {renderAnalysisChips(detail.enrichment.tags, "No tags yet")}
          </div>
          <div>
            <p style={{ marginBottom: 6, color: "#60738f", fontSize: 10, fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase" }}>
              Keywords
            </p>
            {renderAnalysisChips(detail.enrichment.keywords, "No keywords yet")}
          </div>
        </div>
      </div>
      </div>
    );
  }

  return (
    <div style={{ display: "grid", gridTemplateColumns: mainGridColumns, gap: compact ? 14 : 18 }}>
      <div>
        {analysisBlock}
        <p style={{ marginTop: 10, color: "#c6d4e6", fontSize: 13, lineHeight: 1.65 }}>{articleSummary}</p>
      </div>
      <div style={{ display: "grid", gap: 12 }}>
        <div>
          <p style={{ marginBottom: 6, color: "#60738f", fontSize: 10, fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase" }}>
            Matched Topics
          </p>
          {renderAnalysisChips(topicLabels, "No mapped topics")}
        </div>
        <div>
          <p style={{ marginBottom: 6, color: "#60738f", fontSize: 10, fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase" }}>
            Follow-Up
          </p>
          <p style={{ color: "#8ea0ba", fontSize: 12, lineHeight: 1.55 }}>
            Review the source article, then save it if it should be promoted into research follow-up or briefing coverage.
          </p>
        </div>
      </div>
    </div>
  );
}

export function IntelBetaDashboard({
  initialArticles,
  initialTopicRules,
  initialDocuments = [],
}: {
  initialArticles: StoredRssArticle[];
  initialTopicRules: StoredRssTopicRule[];
  initialDocuments?: DocumentListItem[];
}) {
  const [articles, setArticles] = useState<StoredRssArticle[]>(initialArticles);
  const documentFeedItems = useMemo(() => initialDocuments.map(documentToFeedItem), [initialDocuments]);
  const feedItems = useMemo<FeedItem[]>(
    () => [...articles, ...documentFeedItems]
      .sort((a, b) => feedItemDateMs(b) - feedItemDateMs(a)),
    [articles, documentFeedItems]
  );
  const [topicRules, setTopicRules] = useState<StoredRssTopicRule[]>(initialTopicRules);
  const [sidebarTab, setSidebarTab] = useState<SidebarTab>("topics");
  const [selectedTopic, setSelectedTopic] = useState<TopicFilter>("ALL");
  const [selectedSource, setSelectedSource] = useState<SourceFilter>("ALL");
  const [search, setSearch] = useState("");
  const [selectedArticleId, setSelectedArticleId] = useState<number | null>(feedItems[0]?.id ?? null);
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());
  const [newCount, setNewCount] = useState(0);
  const newestFetchedAtRef = useRef<string>(initialArticles[0] ? feedItemDate(initialArticles[0]) : "");
  const articleSignatureRef = useRef(articleListSignature(initialArticles));
  const topicRulesSignatureRef = useRef(topicRulesSignature(initialTopicRules));
  const savedItems = useSavedItems();
  const [expandedAnalysis, setExpandedAnalysis] = useState<Record<string, boolean>>({});
  const [docDetails, setDocDetails] = useState<Record<string, DocumentDetailData>>({});
  const [docDetailLoading, setDocDetailLoading] = useState<Record<string, boolean>>({});
  const [docDetailError, setDocDetailError] = useState<Record<string, string>>({});
  const [feedAnalyses, setFeedAnalyses] = useState<Record<string, FeedItemAnalysis>>(() => analysisMapFromFeedItems(feedItems));
  const [feedAnalysisLoading, setFeedAnalysisLoading] = useState<Record<string, boolean>>({});
  const [feedAnalysisError, setFeedAnalysisError] = useState<Record<string, string>>({});

  const visibleTopicRules = useMemo(() => normalizeTopicRules(topicRules), [topicRules]);
  const topicIndex = useMemo(() => {
    const topicMatchesByArticleId = new Map<number, TopicRuleView[]>();
    const topicCounts = new Map<string, number>();
    const matchedArticles: FeedItem[] = [];

    for (const article of feedItems) {
      const matches = getMatchingTopics(article, visibleTopicRules);
      topicMatchesByArticleId.set(article.id, matches);
      if (matches.length === 0 && article.item_type !== "document") {
        continue;
      }
      matchedArticles.push(article);
      for (const topic of matches) {
        topicCounts.set(topic.topic_key, (topicCounts.get(topic.topic_key) ?? 0) + 1);
      }
    }

    return { topicMatchesByArticleId, topicCounts, matchedArticles };
  }, [feedItems, visibleTopicRules]);
  const matchedArticles = topicIndex.matchedArticles;
  const sourceCounts = useMemo(() => {
    const counts = new Map<SourceFilter, number>();
    for (const article of matchedArticles) {
      for (const source of SOURCE_FILTERS) {
        if (matchesSourceFilter(article, source.key)) {
          counts.set(source.key, (counts.get(source.key) ?? 0) + 1);
        }
      }
    }
    return counts;
  }, [matchedArticles]);
  const selectedRule = selectedTopic === "ALL"
    ? null
    : visibleTopicRules.find((rule) => rule.topic_key === selectedTopic) ?? null;
  const selectedSourceLabel = selectedSource === "ALL"
    ? "All Sources"
    : SOURCE_FILTERS.find((source) => source.key === selectedSource)?.label ?? selectedSource;
  const searchTerm = search.trim().toLowerCase();
  const deferredSearchTerm = useDeferredValue(searchTerm);
  const isMobile = useMediaQuery("(max-width: 760px)");

  useEffect(() => {
    if (selectedTopic !== "ALL" && !visibleTopicRules.some((rule) => rule.topic_key === selectedTopic)) {
      setSelectedTopic("ALL");
    }
  }, [selectedTopic, visibleTopicRules]);

  useEffect(() => {
    const savedAnalyses = analysisMapFromFeedItems(feedItems);
    if (Object.keys(savedAnalyses).length === 0) return;
    setFeedAnalyses((prev) => ({ ...savedAnalyses, ...prev }));
  }, [feedItems]);

  useEffect(() => {
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    let errStreak = 0;
    let mounted = true;

    const poll = async () => {
      try {
        const res = await fetch("/api/intel/feed?limit=400", { cache: "no-store" });
        if (!res.ok) { errStreak++; }
        else {
          const json = (await res.json()) as {
            ok: boolean;
            data: { articles: StoredRssArticle[]; topicRules: StoredRssTopicRule[]; generatedAt: string };
          };
          if (!json.ok) { errStreak++; }
          else {
            errStreak = 0;
            const fresh = json.data.articles;
            const freshRules = json.data.topicRules;
            const newest = fresh[0] ? feedItemDate(fresh[0]) : "";
            let changed = false;
            if (newest && newest > newestFetchedAtRef.current) {
              const added = fresh.filter((article) => feedItemDate(article) > newestFetchedAtRef.current).length;
              newestFetchedAtRef.current = newest;
              setNewCount((count) => count + added);
              if (!selectedArticleId && fresh[0]?.id) {
                setSelectedArticleId(fresh[0].id);
              }
            }
            const nextArticleSignature = articleListSignature(fresh);
            if (nextArticleSignature !== articleSignatureRef.current) {
              articleSignatureRef.current = nextArticleSignature;
              changed = true;
              setArticles(fresh);
            }
            const nextTopicRulesSignature = topicRulesSignature(freshRules);
            if (nextTopicRulesSignature !== topicRulesSignatureRef.current) {
              topicRulesSignatureRef.current = nextTopicRulesSignature;
              changed = true;
              setTopicRules(freshRules);
            }
            if (changed) setLastUpdated(new Date());
          }
        }
      } catch {
        errStreak++;
      }
      if (mounted) {
        // Exponential backoff on errors: 15s → 30s → 60s → 120s (cap)
        const delay = errStreak > 0 ? Math.min(15_000 * (2 ** (errStreak - 1)), 120_000) : 15_000;
        timeoutId = setTimeout(() => { void poll(); }, delay);
      }
    };

    void poll();
    return () => { mounted = false; if (timeoutId) clearTimeout(timeoutId); };
  }, [selectedArticleId]);

  const filtered = useMemo(
    () =>
      matchedArticles.filter(
        (article) =>
          matchesTopic(article, selectedRule, topicIndex.topicMatchesByArticleId) &&
          matchesSourceFilter(article, selectedSource) &&
          matchesSearch(article, deferredSearchTerm)
      ),
    [deferredSearchTerm, matchedArticles, selectedRule, selectedSource, topicIndex.topicMatchesByArticleId]
  );
  useEffect(() => {
    if (filtered.length === 0) {
      setSelectedArticleId(null);
      return;
    }
    if (!filtered.some((article) => article.id === selectedArticleId)) {
      setSelectedArticleId(filtered[0].id);
    }
  }, [filtered, selectedArticleId]);

  const featured = filtered.find((article) => article.id === selectedArticleId) ?? filtered[0] ?? null;

  const loadDocDetail = useCallback(async (documentId: string) => {
    const docId = String(documentId || "").trim();
    if (!docId) return;
    if (docDetails[docId] || docDetailLoading[docId]) return;

    setDocDetailLoading((prev) => ({ ...prev, [docId]: true }));
    setDocDetailError((prev) => ({ ...prev, [docId]: "" }));
    try {
      const detail = await fetchJson<DocumentDetailData>(`/api/documents/${encodeURIComponent(docId)}`);
      setDocDetails((prev) => ({ ...prev, [docId]: detail }));
    } catch (err) {
      setDocDetailError((prev) => ({
        ...prev,
        [docId]: err instanceof Error ? err.message : "Failed to load analysis.",
      }));
    } finally {
      setDocDetailLoading((prev) => ({ ...prev, [docId]: false }));
    }
  }, [docDetailLoading, docDetails]);

  const loadFeedAnalysis = useCallback(async (article: FeedItem) => {
    const itemKey = savedArticleId(article);
    if (feedAnalyses[itemKey] || feedAnalysisLoading[itemKey]) return;
    const source = getFeedMeta(article.feed_key);
    const topics = topicIndex.topicMatchesByArticleId.get(article.id)?.map((topic) => topic.label) ?? [];

    setFeedAnalysisLoading((prev) => ({ ...prev, [itemKey]: true }));
    setFeedAnalysisError((prev) => ({ ...prev, [itemKey]: "" }));
    try {
      const payload = await fetchJson<{ analysis: FeedItemAnalysis }>("/api/intel/analyze", {
        method: "POST",
        body: JSON.stringify({
          article_id: article.item_type === "document" ? "" : article.id,
          guid: article.guid || "",
          title: decodeEntities(article.title || ""),
          description: decodeEntities(article.description || ""),
          url: article.url || "",
          source: article.organization || source.label,
          author: article.author || "",
          published_at: feedItemDate(article),
          tone_label: article.tone_label || "",
          topics,
          item_type: article.item_type || "article",
        }),
      });
      setFeedAnalyses((prev) => ({ ...prev, [itemKey]: payload.analysis }));
    } catch (err) {
      setFeedAnalysisError((prev) => ({
        ...prev,
        [itemKey]: err instanceof Error ? err.message : "Failed to generate analysis.",
      }));
    } finally {
      setFeedAnalysisLoading((prev) => ({ ...prev, [itemKey]: false }));
    }
  }, [feedAnalyses, feedAnalysisLoading, topicIndex.topicMatchesByArticleId]);

  const toggleFeedAnalysis = useCallback((article: FeedItem) => {
    const key = savedArticleId(article);
    setExpandedAnalysis((prev) => {
      const shouldOpen = !prev[key];
      if (isMobile) return shouldOpen ? { [key]: true } : {};
      return { ...prev, [key]: shouldOpen };
    });

    if (!feedAnalyses[key] && !feedAnalysisLoading[key]) {
      void loadFeedAnalysis(article);
    }
    if (article.item_type === "document" && article.document_id && !docDetails[article.document_id] && !docDetailLoading[article.document_id]) {
      void loadDocDetail(article.document_id);
    }
  }, [docDetailLoading, docDetails, feedAnalyses, feedAnalysisLoading, isMobile, loadDocDetail, loadFeedAnalysis]);

  const toggleArticleSave = (article: FeedItem) => {
    const source = getFeedMeta(article.feed_key);
    const primaryTopic = topicIndex.topicMatchesByArticleId.get(article.id)?.[0]?.label;
    savedItems.toggle({
      id: savedArticleId(article),
      type: article.item_type === "document" ? "doc" : "article",
      title: decodeEntities(article.title || "Untitled article"),
      url: article.url,
      source: article.organization || source.label,
      topic: primaryTopic,
      metadata: {
        feedKey: article.feed_key,
        author: article.author,
        publishedAt: article.published_at || "",
        toneLabel: article.tone_label,
        documentId: article.document_id || "",
        sourceKind: article.source_kind || "",
      },
    });
  };

  return (
    <div
      style={{
        minHeight: "82vh",
        color: "#dbe7f5",
        fontFamily: 'var(--font-body), "Segoe UI", sans-serif',
      }}
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: isMobile ? "1fr" : "168px minmax(0, 1fr)",
          gap: 0,
          border: "1px solid rgba(99, 127, 170, 0.18)",
          background: "linear-gradient(180deg, rgba(8,16,28,0.96), rgba(9,20,31,0.96))",
          boxShadow: "0 24px 64px rgba(0,0,0,0.28)",
          overflow: "hidden",
        }}
      >
        <aside
          style={{
            borderRight: isMobile ? "none" : "1px solid rgba(99, 127, 170, 0.16)",
            borderBottom: isMobile ? "1px solid rgba(99, 127, 170, 0.16)" : "none",
            padding: isMobile ? "12px" : "14px 10px 18px",
            background: "linear-gradient(180deg, rgba(8,17,29,0.92), rgba(10,21,34,0.98))",
          }}
        >
          {isMobile ? (
            <div style={{ display: "grid", gap: 10 }}>
              <a
                href="/research"
                style={{
                  display: "block",
                  padding: "9px 10px",
                  borderRadius: 8,
                  border: "1px solid rgba(79,213,255,0.15)",
                  background: "rgba(79,213,255,0.05)",
                  color: "#4fd5ff",
                  fontSize: 12,
                  fontWeight: 600,
                  textDecoration: "none",
                  lineHeight: 1.35,
                }}
              >
                Search all regulatory documents
              </a>
              <label style={{ display: "grid", gap: 6 }}>
                <span style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
                  <button
                    type="button"
                    onClick={() => setSidebarTab("topics")}
                    style={{
                      minHeight: 34,
                      borderRadius: 7,
                      border: sidebarTab === "topics" ? "1px solid rgba(79,213,255,0.52)" : "1px solid rgba(90, 118, 162, 0.18)",
                      background: sidebarTab === "topics" ? "rgba(79,213,255,0.14)" : "rgba(14, 24, 39, 0.72)",
                      color: sidebarTab === "topics" ? "#dbe7f5" : "#8ea0ba",
                      fontSize: 12,
                      fontWeight: 700,
                      cursor: "pointer",
                    }}
                  >
                    Topics
                  </button>
                  <button
                    type="button"
                    onClick={() => setSidebarTab("sources")}
                    style={{
                      minHeight: 34,
                      borderRadius: 7,
                      border: sidebarTab === "sources" ? "1px solid rgba(79,213,255,0.52)" : "1px solid rgba(90, 118, 162, 0.18)",
                      background: sidebarTab === "sources" ? "rgba(79,213,255,0.14)" : "rgba(14, 24, 39, 0.72)",
                      color: sidebarTab === "sources" ? "#dbe7f5" : "#8ea0ba",
                      fontSize: 12,
                      fontWeight: 700,
                      cursor: "pointer",
                    }}
                  >
                    Sources
                  </button>
                </span>
                <span style={{ color: "#5f7390", fontSize: 10, letterSpacing: "0.18em", textTransform: "uppercase" }}>
                  {sidebarTab === "topics" ? "Topics" : "Sources"}
                </span>
                {sidebarTab === "topics" ? (
                  <select
                    value={selectedTopic}
                    onChange={(e) => setSelectedTopic(e.target.value)}
                    style={{
                      width: "100%",
                      minHeight: 40,
                      background: "rgba(14, 24, 39, 0.9)",
                      border: "1px solid rgba(90, 118, 162, 0.28)",
                      color: "#d9e7f7",
                      borderRadius: 6,
                      padding: "8px 10px",
                      fontSize: 13,
                    }}
                  >
                    <option value="ALL">All Topics ({matchedArticles.length})</option>
                    {visibleTopicRules.map((rule) => (
                      <option key={rule.topic_key} value={rule.topic_key}>
                        {rule.label} ({topicIndex.topicCounts.get(rule.topic_key) ?? 0})
                      </option>
                    ))}
                  </select>
                ) : (
                  <select
                    value={selectedSource}
                    onChange={(e) => setSelectedSource(e.target.value as SourceFilter)}
                    style={{
                      width: "100%",
                      minHeight: 40,
                      background: "rgba(14, 24, 39, 0.9)",
                      border: "1px solid rgba(90, 118, 162, 0.28)",
                      color: "#d9e7f7",
                      borderRadius: 6,
                      padding: "8px 10px",
                      fontSize: 13,
                    }}
                  >
                    <option value="ALL">All Sources ({matchedArticles.length})</option>
                    {SOURCE_FILTERS.map((source) => (
                      <option key={source.key} value={source.key}>
                        {source.label} ({sourceCounts.get(source.key) ?? 0})
                      </option>
                    ))}
                  </select>
                )}
              </label>
            </div>
          ) : (
            <>
              <a
                href="/research"
                style={{
                  display: "block",
                  marginBottom: 14,
                  padding: "7px 10px",
                  borderRadius: 8,
                  border: "1px solid rgba(79,213,255,0.15)",
                  background: "rgba(79,213,255,0.05)",
                  color: "#4fd5ff",
                  fontSize: 11,
                  fontWeight: 500,
                  textDecoration: "none",
                  lineHeight: 1.35,
                }}
              >
                Search all regulatory documents
              </a>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, marginBottom: 12 }}>
                <button
                  type="button"
                  onClick={() => setSidebarTab("topics")}
                  style={{
                    minHeight: 32,
                    borderRadius: 7,
                    border: sidebarTab === "topics" ? "1px solid rgba(79,213,255,0.52)" : "1px solid rgba(90, 118, 162, 0.16)",
                    background: sidebarTab === "topics" ? "rgba(79,213,255,0.14)" : "rgba(14, 24, 39, 0.54)",
                    color: sidebarTab === "topics" ? "#dbe7f5" : "#8ea0ba",
                    fontSize: 11,
                    fontWeight: 700,
                    cursor: "pointer",
                  }}
                >
                  Topics
                </button>
                <button
                  type="button"
                  onClick={() => setSidebarTab("sources")}
                  style={{
                    minHeight: 32,
                    borderRadius: 7,
                    border: sidebarTab === "sources" ? "1px solid rgba(79,213,255,0.52)" : "1px solid rgba(90, 118, 162, 0.16)",
                    background: sidebarTab === "sources" ? "rgba(79,213,255,0.14)" : "rgba(14, 24, 39, 0.54)",
                    color: sidebarTab === "sources" ? "#dbe7f5" : "#8ea0ba",
                    fontSize: 11,
                    fontWeight: 700,
                    cursor: "pointer",
                  }}
                >
                  Sources
                </button>
              </div>
              <div style={{ color: "#5f7390", fontSize: 10, letterSpacing: "0.18em", textTransform: "uppercase", marginBottom: 10 }}>
                {sidebarTab === "topics" ? "Topics" : "Sources"}
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                {sidebarTab === "topics" ? (
                  <>
                    <TopicButton
                      label="All Topics"
                      active={selectedTopic === "ALL"}
                      onClick={() => setSelectedTopic("ALL")}
                      count={matchedArticles.length}
                    />
                    {visibleTopicRules.map((rule) => (
                      <TopicButton
                        key={rule.topic_key}
                        label={rule.label}
                        active={selectedTopic === rule.topic_key}
                        onClick={() => setSelectedTopic(rule.topic_key)}
                        count={topicIndex.topicCounts.get(rule.topic_key) ?? 0}
                      />
                    ))}
                  </>
                ) : (
                  <>
                    <TopicButton
                      label="All Sources"
                      active={selectedSource === "ALL"}
                      onClick={() => setSelectedSource("ALL")}
                      count={matchedArticles.length}
                    />
                    {SOURCE_FILTERS.map((source) => (
                      <TopicButton
                        key={source.key}
                        label={source.label}
                        active={selectedSource === source.key}
                        onClick={() => setSelectedSource(source.key)}
                        count={sourceCounts.get(source.key) ?? 0}
                      />
                    ))}
                  </>
                )}
              </div>
            </>
          )}

          {!isMobile ? <div style={{ marginTop: 22 }}>
            <div style={{ color: "#5f7390", fontSize: 10, letterSpacing: "0.18em", textTransform: "uppercase", marginBottom: 10 }}>
              Legend
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 8, fontSize: 12, color: "#8d9fb7" }}>
              {Object.entries(TONE_STYLE).map(([key, value]) => (
                <div key={key} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ width: 6, height: 6, borderRadius: 1, background: value.color }} />
                  <span>{value.label}</span>
                </div>
              ))}
            </div>
          </div> : null}
        </aside>

        <main style={{ minWidth: 0 }}>
          <div
            style={{
              display: "flex",
              alignItems: isMobile ? "stretch" : "center",
              justifyContent: "space-between",
              gap: isMobile ? 10 : 16,
              padding: isMobile ? "12px" : "14px 16px 10px",
              borderBottom: "1px solid rgba(99, 127, 170, 0.16)",
              flexWrap: "wrap",
              flexDirection: isMobile ? "column" : "row",
            }}
          >
            <div
              style={{
                color: "#91a8c7",
                fontSize: 11,
                letterSpacing: "0.16em",
                textTransform: "uppercase",
                fontFamily: '"IBM Plex Mono", "SFMono-Regular", Consolas, monospace',
              }}
            >
              News Feed / {selectedRule ? selectedRule.label : "All"} / {selectedSourceLabel} / {filtered.length} matched ({feedItems.length} total)
            </div>

            <div style={{ display: "flex", alignItems: isMobile ? "stretch" : "center", gap: isMobile ? 10 : 14, flexWrap: "wrap" }}>
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="search..."
                style={{
                  width: isMobile ? "100%" : 220,
                  minHeight: isMobile ? 40 : undefined,
                  background: "rgba(14, 24, 39, 0.9)",
                  border: "1px solid rgba(90, 118, 162, 0.18)",
                  color: "#d9e7f7",
                  borderRadius: 5,
                  padding: "8px 12px",
                  fontSize: 12,
                }}
              />
              <div
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 8,
                  color: "#4dd39f",
                  fontSize: 12,
                  letterSpacing: "0.10em",
                  textTransform: "uppercase",
                  fontFamily: '"IBM Plex Mono", "SFMono-Regular", Consolas, monospace',
                }}
              >
                <span style={{ width: 6, height: 6, borderRadius: 999, background: "#4dd39f", boxShadow: "0 0 10px rgba(77,211,159,0.7)" }} />
                <span>Live {formatClock(lastUpdated)}</span>
              </div>
            </div>
          </div>

          {newCount > 0 ? (
            <button
              onClick={() => setNewCount(0)}
              style={{
                margin: "10px 16px 0",
                border: "1px solid rgba(77, 211, 159, 0.25)",
                background: "rgba(77, 211, 159, 0.08)",
                color: "#4dd39f",
                padding: "7px 12px",
                borderRadius: 4,
                fontSize: 12,
                cursor: "pointer",
              }}
            >
              {newCount} new item{newCount === 1 ? "" : "s"} available
            </button>
          ) : null}

          <div style={{ padding: isMobile ? "0 12px 14px" : "12px 16px 18px" }}>
            {!isMobile ? <div
              style={{
                display: "grid",
                gridTemplateColumns: "80px 150px 66px minmax(0, 1fr) 220px 24px",
                gap: 14,
                paddingBottom: 8,
                color: "#5f7390",
                fontSize: 10,
                letterSpacing: "0.18em",
                textTransform: "uppercase",
                fontFamily: '"IBM Plex Mono", "SFMono-Regular", Consolas, monospace',
              }}
            >
              <div>Time</div>
              <div>Source</div>
              <div>Snt</div>
              <div>Headline</div>
              <div style={{ textAlign: "right" }}>Tags</div>
              <div aria-hidden="true" />
            </div> : null}

            {filtered.length === 0 ? (
              <div style={{ color: "#72839d", fontSize: 13, padding: "28px 0" }}>
                {feedItems.length === 0 ? "No feed items yet." : "No feed items match the current filters."}
              </div>
            ) : (
              filtered.map((article) => {
                const itemKey = savedArticleId(article);
                const matchedTopicsForArticle = topicIndex.topicMatchesByArticleId.get(article.id) ?? [];
                const analysisOpen = !!expandedAnalysis[itemKey];
                const docId = article.document_id || "";
                const detailLoading = docId ? !!docDetailLoading[docId] : false;
                const detailError = docId ? docDetailError[docId] || "" : "";
                const itemAnalysisLoading = !!feedAnalysisLoading[itemKey];
                const itemAnalysisError = feedAnalysisError[itemKey] || "";
                const analysisLabel = itemAnalysisLoading || detailLoading ? "Analyzing..." : analysisOpen ? "Hide Analysis" : "Open Analysis";

                return (
                  <Fragment key={itemKey}>
                    {article.id === featured?.id ? (
                      <FeaturedCard
                        article={article}
                        matchedTopics={matchedTopicsForArticle}
                        saved={savedItems.isSaved(itemKey)}
                        onToggleSave={() => toggleArticleSave(article)}
                        analysisOpen={analysisOpen}
                        analysisLabel={analysisLabel}
                        onToggleAnalysis={() => toggleFeedAnalysis(article)}
                        compact={isMobile}
                      />
                    ) : (
                      <FeedRow
                        article={article}
                        matchedTopics={matchedTopicsForArticle}
                        active={article.id === selectedArticleId}
                        onSelect={() => setSelectedArticleId(article.id)}
                        saved={savedItems.isSaved(itemKey)}
                        onToggleSave={() => toggleArticleSave(article)}
                        analysisOpen={analysisOpen}
                        analysisLabel={analysisLabel}
                        onToggleAnalysis={() => toggleFeedAnalysis(article)}
                        compact={isMobile}
                      />
                    )}
                    {analysisOpen ? (
                      <div
                        style={{
                          borderTop: "1px solid rgba(112, 142, 187, 0.12)",
                          background: "rgba(5, 13, 23, 0.62)",
                          padding: isMobile ? "12px 0 16px" : "14px 16px 16px",
                        }}
                      >
                        <FeedAnalysisPanel
                          article={article}
                          matchedTopics={matchedTopicsForArticle}
                          analysis={feedAnalyses[itemKey]}
                          analysisLoading={itemAnalysisLoading}
                          analysisError={itemAnalysisError}
                          retryAnalysis={() => void loadFeedAnalysis(article)}
                          detail={docId ? docDetails[docId] : undefined}
                          loading={detailLoading}
                          error={detailError}
                          retry={() => {
                            if (docId) void loadDocDetail(docId);
                          }}
                          compact={isMobile}
                        />
                      </div>
                    ) : null}
                  </Fragment>
                );
              })
            )}
          </div>
        </main>
      </div>

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          gap: 8,
          flexDirection: isMobile ? "column" : "row",
          padding: "10px 4px 0",
          color: "#5d708a",
          fontSize: 11,
        }}
      >
        <div>Updated {formatUpdated(lastUpdated.toISOString())}</div>
        <div>{articles.length} articles + {documentFeedItems.length} research documents</div>
      </div>
    </div>
  );
}

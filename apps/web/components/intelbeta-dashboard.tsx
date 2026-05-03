"use client";

import { useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import type { StoredRssArticle, StoredRssTopicRule } from "@/lib/server/neon";
import { BookmarkButton } from "@/components/bookmark-button";
import { useSavedItems } from "@/hooks/use-saved-items";
import {
  decodeEntities,
  getMatchingTopics,
  normalizeTopicRules,
  type TopicRuleView,
} from "@/lib/intel-topic-matching";

type TopicFilter = string | "ALL";

type FeedMeta = {
  label: string;
  code: string;
  color: string;
};

const FEED_META: Record<string, FeedMeta> = {
  wsj_us_business: { label: "WSJ Business", code: "WSJB", color: "#63a8ff" },
  wsj_markets: { label: "WSJ Markets", code: "WSJM", color: "#ffc857" },
  wsj_opinion: { label: "WSJ Opinion", code: "WSJO", color: "#b88fff" },
  mw_top_stories: { label: "MarketWatch", code: "MW", color: "#4dd39f" },
  rss_nytimes_com_services_xml_rss_nyt_business_xml: { label: "NYT Business", code: "NYTB", color: "#ffe066" },
  rss_nytimes_com_services_xml_rss_nyt_technology_xml: { label: "NYT Tech", code: "NYTT", color: "#74c0fc" },
  rss_nytimes_com_services_xml_rss_nyt_politics_xml: { label: "NYT Politics", code: "NYTP", color: "#ff8787" },
};

function getFeedMeta(feedKey: string): FeedMeta {
  return FEED_META[feedKey] ?? {
    label: feedKey,
    code: feedKey.slice(0, 4).toUpperCase(),
    color: "#8fa7c8",
  };
}

function savedArticleId(article: StoredRssArticle): string {
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

function matchesTopic(article: StoredRssArticle, rule: TopicRuleView | null, topicMatchesByArticleId: Map<number, TopicRuleView[]>): boolean {
  if (!rule) return true;
  return (topicMatchesByArticleId.get(article.id) ?? []).some((item) => item.topic_key === rule.topic_key);
}

function matchesSearch(article: StoredRssArticle, searchTerm: string): boolean {
  if (!searchTerm) return true;
  const haystack = `${article.title} ${article.description ?? ""} ${article.author ?? ""}`.toLowerCase();
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

function articleListSignature(articles: StoredRssArticle[]): string {
  const first = articles[0];
  const last = articles[articles.length - 1];
  return `${articles.length}:${first?.id ?? ""}:${first?.fetched_at ?? ""}:${last?.id ?? ""}:${last?.fetched_at ?? ""}`;
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
}: {
  article: StoredRssArticle;
  matchedTopics: TopicRuleView[];
  active: boolean;
  onSelect: () => void;
  saved: boolean;
  onToggleSave: () => void;
}) {
  const source = getFeedMeta(article.feed_key);
  const visibleTopics = matchedTopics.slice(0, 3);
  const description = ellipsize(article.description ?? "", 82);

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "80px 54px 66px minmax(0, 1fr) 220px 24px",
        gap: 14,
        alignItems: "start",
        padding: "10px 0",
        borderTop: "1px solid rgba(112, 142, 187, 0.12)",
        background: active ? "rgba(67, 112, 186, 0.08)" : "transparent",
        cursor: "pointer",
      }}
      onClick={onSelect}
    >
      <div style={{ color: "#7f8faa", fontSize: 12, whiteSpace: "nowrap" }}>{formatRelativeTime(article.fetched_at)}</div>
      <div style={{ color: source.color, fontSize: 12, fontWeight: 700, letterSpacing: "0.08em" }}>{source.code}</div>
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
}: {
  article: StoredRssArticle;
  matchedTopics: TopicRuleView[];
  saved: boolean;
  onToggleSave: () => void;
}) {
  const source = getFeedMeta(article.feed_key);
  const tone = article.tone_label && TONE_STYLE[article.tone_label] ? article.tone_label : "neutral";

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
          gridTemplateColumns: "80px 54px 66px minmax(0, 1fr) 240px 24px",
          gap: 14,
          alignItems: "start",
        }}
      >
        <div style={{ color: "#8fa7c8", fontSize: 12 }}>{formatRelativeTime(article.fetched_at)}</div>
        <div style={{ color: source.color, fontSize: 12, fontWeight: 700, letterSpacing: "0.08em" }}>{source.code}</div>
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
          <div style={{ color: "#d7e1ef" }}>{decodeEntities(article.author || "News Desk")}</div>
          <div style={{ letterSpacing: "0.12em", textTransform: "uppercase", color: "#5e708a" }}>Source</div>
          <div style={{ color: "#d7e1ef" }}>{source.label}</div>
          <div style={{ letterSpacing: "0.12em", textTransform: "uppercase", color: "#5e708a" }}>Impact</div>
          <div style={{ color: TONE_STYLE[tone].color, fontWeight: 700 }}>{TONE_STYLE[tone].label.toUpperCase()}</div>
          <div style={{ letterSpacing: "0.12em", textTransform: "uppercase", color: "#5e708a" }}>Topics</div>
          <div style={{ color: "#d7e1ef" }}>
            {matchedTopics.length > 0 ? matchedTopics.map((topic) => topic.label).join(", ") : "Unmapped"}
          </div>
        </div>
        <BookmarkButton saved={saved} onToggle={onToggleSave} size={16} />
      </div>
    </div>
  );
}

export function IntelBetaDashboard({
  initialArticles,
  initialTopicRules,
}: {
  initialArticles: StoredRssArticle[];
  initialTopicRules: StoredRssTopicRule[];
}) {
  const [articles, setArticles] = useState<StoredRssArticle[]>(initialArticles);
  const [topicRules, setTopicRules] = useState<StoredRssTopicRule[]>(initialTopicRules);
  const [selectedTopic, setSelectedTopic] = useState<TopicFilter>("ALL");
  const [search, setSearch] = useState("");
  const [selectedArticleId, setSelectedArticleId] = useState<number | null>(initialArticles[0]?.id ?? null);
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());
  const [newCount, setNewCount] = useState(0);
  const newestFetchedAtRef = useRef<string>(initialArticles[0]?.fetched_at ?? "");
  const articleSignatureRef = useRef(articleListSignature(initialArticles));
  const topicRulesSignatureRef = useRef(topicRulesSignature(initialTopicRules));
  const savedItems = useSavedItems();

  const visibleTopicRules = useMemo(() => normalizeTopicRules(topicRules), [topicRules]);
  const topicIndex = useMemo(() => {
    const topicMatchesByArticleId = new Map<number, TopicRuleView[]>();
    const topicCounts = new Map<string, number>();
    const matchedArticles: StoredRssArticle[] = [];

    for (const article of articles) {
      const matches = getMatchingTopics(article, visibleTopicRules);
      topicMatchesByArticleId.set(article.id, matches);
      if (matches.length === 0) {
        continue;
      }
      matchedArticles.push(article);
      for (const topic of matches) {
        topicCounts.set(topic.topic_key, (topicCounts.get(topic.topic_key) ?? 0) + 1);
      }
    }

    return { topicMatchesByArticleId, topicCounts, matchedArticles };
  }, [articles, visibleTopicRules]);
  const matchedArticles = topicIndex.matchedArticles;
  const selectedRule = selectedTopic === "ALL"
    ? null
    : visibleTopicRules.find((rule) => rule.topic_key === selectedTopic) ?? null;
  const searchTerm = search.trim().toLowerCase();
  const deferredSearchTerm = useDeferredValue(searchTerm);

  useEffect(() => {
    if (selectedTopic !== "ALL" && !visibleTopicRules.some((rule) => rule.topic_key === selectedTopic)) {
      setSelectedTopic("ALL");
    }
  }, [selectedTopic, visibleTopicRules]);

  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch("/api/intel/feed?limit=400", { cache: "no-store" });
        if (!res.ok) return;
        const json = (await res.json()) as {
          ok: boolean;
          data: { articles: StoredRssArticle[]; topicRules: StoredRssTopicRule[]; generatedAt: string };
        };
        if (!json.ok) return;
        const fresh = json.data.articles;
        const freshRules = json.data.topicRules;
        const newest = fresh[0]?.fetched_at ?? "";
        let changed = false;
        if (newest && newest > newestFetchedAtRef.current) {
          const added = fresh.filter((article) => article.fetched_at > newestFetchedAtRef.current).length;
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
        if (changed) {
          setLastUpdated(new Date());
        }
      } catch {
        // silent; will retry next interval
      }
    };

    void poll();
    const id = setInterval(poll, 15_000);
    return () => clearInterval(id);
  }, []);

  const filtered = useMemo(
    () =>
      matchedArticles.filter(
        (article) => matchesTopic(article, selectedRule, topicIndex.topicMatchesByArticleId) && matchesSearch(article, deferredSearchTerm)
      ),
    [deferredSearchTerm, matchedArticles, selectedRule, topicIndex.topicMatchesByArticleId]
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

  const toggleArticleSave = (article: StoredRssArticle) => {
    const source = getFeedMeta(article.feed_key);
    const primaryTopic = topicIndex.topicMatchesByArticleId.get(article.id)?.[0]?.label;
    savedItems.toggle({
      id: savedArticleId(article),
      type: "article",
      title: decodeEntities(article.title || "Untitled article"),
      url: article.url,
      source: source.label,
      topic: primaryTopic,
      metadata: {
        feedKey: article.feed_key,
        author: article.author,
        publishedAt: article.published_at || "",
        toneLabel: article.tone_label,
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
          gridTemplateColumns: "168px minmax(0, 1fr)",
          gap: 0,
          border: "1px solid rgba(99, 127, 170, 0.18)",
          background: "linear-gradient(180deg, rgba(8,16,28,0.96), rgba(9,20,31,0.96))",
          boxShadow: "0 24px 64px rgba(0,0,0,0.28)",
          overflow: "hidden",
        }}
      >
        <aside
          style={{
            borderRight: "1px solid rgba(99, 127, 170, 0.16)",
            padding: "14px 10px 18px",
            background: "linear-gradient(180deg, rgba(8,17,29,0.92), rgba(10,21,34,0.98))",
          }}
        >
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
            Search all regulatory documents →
          </a>
          <div style={{ color: "#5f7390", fontSize: 10, letterSpacing: "0.18em", textTransform: "uppercase", marginBottom: 10 }}>
            Topics
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
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
          </div>

          <div style={{ marginTop: 22 }}>
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
          </div>
        </aside>

        <main style={{ minWidth: 0 }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: 16,
              padding: "14px 16px 10px",
              borderBottom: "1px solid rgba(99, 127, 170, 0.16)",
              flexWrap: "wrap",
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
              News Feed / {selectedRule ? selectedRule.label : "All"} / {filtered.length} matched ({articles.length} total)
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="search..."
                style={{
                  width: 220,
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

          <div style={{ padding: "12px 16px 18px" }}>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "80px 54px 66px minmax(0, 1fr) 220px 24px",
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
              <div>Src</div>
              <div>Snt</div>
              <div>Headline</div>
              <div style={{ textAlign: "right" }}>Tags</div>
              <div aria-hidden="true" />
            </div>

            {filtered.length === 0 ? (
              <div style={{ color: "#72839d", fontSize: 13, padding: "28px 0" }}>
                {articles.length === 0 ? "No articles yet." : "No articles match the current filters."}
              </div>
            ) : (
              filtered.map((article) =>
                article.id === featured?.id ? (
                  <FeaturedCard
                    key={article.id}
                    article={article}
                    matchedTopics={topicIndex.topicMatchesByArticleId.get(article.id) ?? []}
                    saved={savedItems.isSaved(savedArticleId(article))}
                    onToggleSave={() => toggleArticleSave(article)}
                  />
                ) : (
                  <FeedRow
                    key={article.id}
                    article={article}
                    matchedTopics={topicIndex.topicMatchesByArticleId.get(article.id) ?? []}
                    active={article.id === selectedArticleId}
                    onSelect={() => setSelectedArticleId(article.id)}
                    saved={savedItems.isSaved(savedArticleId(article))}
                    onToggleSave={() => toggleArticleSave(article)}
                  />
                )
              )
            )}
          </div>
        </main>
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", padding: "10px 4px 0", color: "#5d708a", fontSize: 11 }}>
        <div>Updated {formatUpdated(lastUpdated.toISOString())}</div>
        <div>{articles.length} tracked articles</div>
      </div>
    </div>
  );
}

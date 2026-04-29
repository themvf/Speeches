"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { StoredRssArticle, StoredRssTopicRule } from "@/lib/server/neon";

type TopicFilter = string | "ALL";

type TopicRuleView = {
  topic_key: string;
  label: string;
  keywords: string[];
};

function decodeEntities(text: string): string {
  return text
    .replace(/&#x([0-9a-fA-F]+);/gi, (_, hex) => String.fromCharCode(parseInt(hex, 16)))
    .replace(/&#(\d+);/g, (_, dec) => String.fromCharCode(parseInt(dec, 10)))
    .replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"').replace(/&apos;/g, "'").replace(/&nbsp;/g, " ");
}

function parseKeywords(value: string): string[] {
  return value
    .split(/[\n,]+/)
    .map((item) => item.trim().toLowerCase())
    .filter(Boolean);
}

function normalizeTopicRules(rules: StoredRssTopicRule[]): TopicRuleView[] {
  return rules
    .filter((rule) => rule && rule.active)
    .map((rule) => ({
      topic_key: String(rule.topic_key || "").trim(),
      label: String(rule.label || "").trim() || String(rule.topic_key || "").trim(),
      keywords: parseKeywords(String(rule.keywords || "")),
    }))
    .filter((rule) => rule.topic_key && rule.label);
}

const FEED_LABELS: Record<string, string> = {
  wsj_us_business: "WSJ Business",
  wsj_markets: "WSJ Markets",
  wsj_opinion: "WSJ Opinion",
  mw_top_stories: "MarketWatch",
};

const TONE_STYLE: Record<string, { color: string; bg: string; label: string }> = {
  positive: { color: "#41d39d", bg: "rgba(65,211,157,0.12)", label: "Positive" },
  negative: { color: "#ff595e", bg: "rgba(255,89,94,0.12)", label: "Negative" },
  neutral: { color: "#8b95a1", bg: "rgba(139,149,161,0.1)", label: "Neutral" },
};

function matchesTopic(article: StoredRssArticle, rule: TopicRuleView | null): boolean {
  if (!rule) return true;
  if (rule.keywords.length === 0) return false;
  const haystack = `${article.title} ${article.description ?? ""}`.toLowerCase();
  return rule.keywords.some((kw) => haystack.includes(kw));
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

function formatTime(dateStr: string | null): string {
  if (!dateStr) return "";
  return new Date(dateStr).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
}

function ToneBadge({ label }: { label: string | null }) {
  const tone = label && TONE_STYLE[label] ? label : "neutral";
  const s = TONE_STYLE[tone];
  return (
    <span
      style={{
        color: s.color,
        background: s.bg,
        border: `1px solid ${s.color}33`,
        borderRadius: 4,
        fontSize: 11,
        fontWeight: 600,
        padding: "1px 7px",
        letterSpacing: "0.03em",
        whiteSpace: "nowrap",
      }}
    >
      {s.label}
    </span>
  );
}

function ArticleCard({ article }: { article: StoredRssArticle }) {
  return (
    <article
      style={{
        borderBottom: "1px solid rgba(255,255,255,0.07)",
        padding: "14px 0",
        display: "flex",
        flexDirection: "column",
        gap: 5,
      }}
    >
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 12 }}>
        <a
          href={article.url}
          target="_blank"
          rel="noopener noreferrer"
          style={{
            color: "#e8eaed",
            fontWeight: 600,
            fontSize: 14,
            lineHeight: 1.4,
            textDecoration: "none",
            flex: 1,
          }}
          onMouseEnter={(e) => ((e.target as HTMLElement).style.color = "#ffffff")}
          onMouseLeave={(e) => ((e.target as HTMLElement).style.color = "#e8eaed")}
        >
          {decodeEntities(article.title)}
        </a>
        <ToneBadge label={article.tone_label} />
      </div>

      {article.description && (
        <p
          style={{
            color: "#8b95a1",
            fontSize: 13,
            lineHeight: 1.5,
            margin: 0,
            display: "-webkit-box",
            WebkitLineClamp: 2,
            WebkitBoxOrient: "vertical",
            overflow: "hidden",
          }}
        >
          {decodeEntities(article.description ?? "")}
        </p>
      )}

      <div style={{ display: "flex", gap: 10, alignItems: "center", fontSize: 11, color: "#5f6978" }}>
        <span style={{ color: "#6b7a8d", fontWeight: 500 }}>
          {FEED_LABELS[article.feed_key] ?? article.feed_key}
        </span>
        {article.author && <span>· {decodeEntities(article.author)}</span>}
        <span>· {formatRelativeTime(article.fetched_at)}</span>
      </div>
    </article>
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
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());
  const [newCount, setNewCount] = useState(0);
  const newestFetchedAtRef = useRef<string>(initialArticles[0]?.fetched_at ?? "");

  const visibleTopicRules = useMemo(() => normalizeTopicRules(topicRules), [topicRules]);
  const selectedRule = selectedTopic === "ALL"
    ? null
    : visibleTopicRules.find((rule) => rule.topic_key === selectedTopic) ?? null;

  useEffect(() => {
    if (selectedTopic !== "ALL" && !visibleTopicRules.some((rule) => rule.topic_key === selectedTopic)) {
      setSelectedTopic("ALL");
    }
  }, [selectedTopic, visibleTopicRules]);

  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch("/api/intel/feed?limit=100", { cache: "no-store" });
        if (!res.ok) return;
        const json = (await res.json()) as {
          ok: boolean;
          data: { articles: StoredRssArticle[]; topicRules: StoredRssTopicRule[]; generatedAt: string };
        };
        if (!json.ok) return;
        const fresh = json.data.articles;
        const freshRules = json.data.topicRules;
        const newest = fresh[0]?.fetched_at ?? "";
        if (newest && newest > newestFetchedAtRef.current) {
          const added = fresh.filter((a) => a.fetched_at > newestFetchedAtRef.current).length;
          newestFetchedAtRef.current = newest;
          setArticles(fresh);
          setNewCount((c) => c + added);
        } else {
          setArticles(fresh);
        }
        setTopicRules(freshRules);
        setLastUpdated(new Date());
      } catch {
        // silent; will retry next interval
      }
    };

    void poll();
    const id = setInterval(poll, 15_000);
    return () => clearInterval(id);
  }, []);

  const filtered = articles.filter((article) => matchesTopic(article, selectedRule));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 0, minHeight: "80vh" }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          paddingBottom: 16,
          borderBottom: "1px solid rgba(255,255,255,0.1)",
          marginBottom: 0,
        }}
      >
        <div>
          <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: "#e8eaed", letterSpacing: "-0.01em" }}>
            Intel Feed
          </h1>
          <p style={{ margin: 0, fontSize: 12, color: "#5f6978", marginTop: 2 }}>
            WSJ &amp; MarketWatch · live stream
          </p>
        </div>
        <div style={{ fontSize: 12, color: "#5f6978", textAlign: "right" }}>
          <div>Updated {formatTime(lastUpdated.toISOString())}</div>
          <div style={{ color: "#3d4a59" }}>{articles.length} articles</div>
        </div>
      </div>

      {newCount > 0 && (
        <button
          onClick={() => setNewCount(0)}
          style={{
            background: "rgba(65,211,157,0.12)",
            border: "1px solid rgba(65,211,157,0.3)",
            borderRadius: 6,
            color: "#41d39d",
            fontSize: 13,
            fontWeight: 600,
            padding: "8px 16px",
            cursor: "pointer",
            marginTop: 12,
            textAlign: "center",
            width: "100%",
          }}
        >
          {newCount} new article{newCount !== 1 ? "s" : ""} · click to dismiss
        </button>
      )}

      <div style={{ display: "flex", gap: 0, marginTop: 16, alignItems: "flex-start" }}>
        <aside
          style={{
            width: 180,
            flexShrink: 0,
            paddingRight: 20,
            borderRight: "1px solid rgba(255,255,255,0.07)",
            marginRight: 24,
            position: "sticky",
            top: 16,
          }}
        >
          <div style={{ fontSize: 11, fontWeight: 600, color: "#5f6978", letterSpacing: "0.06em", marginBottom: 10 }}>
            TOPICS
          </div>
          <TopicButton
            label="All Topics"
            active={selectedTopic === "ALL"}
            onClick={() => setSelectedTopic("ALL")}
            count={articles.length}
          />
          {visibleTopicRules.map((rule) => {
            const count = articles.filter((article) => matchesTopic(article, rule)).length;
            return (
              <TopicButton
                key={rule.topic_key}
                label={rule.label}
                active={selectedTopic === rule.topic_key}
                onClick={() => setSelectedTopic(rule.topic_key)}
                count={count}
              />
            );
          })}
        </aside>

        <main style={{ flex: 1, minWidth: 0 }}>
          {filtered.length === 0 ? (
            <div style={{ color: "#5f6978", fontSize: 14, padding: "40px 0", textAlign: "center" }}>
              {articles.length === 0
                ? "No articles yet - the feed refreshes every 10 minutes."
                : "No articles match this topic filter."}
            </div>
          ) : (
            filtered.map((article) => <ArticleCard key={article.id} article={article} />)
          )}
        </main>
      </div>
    </div>
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
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        width: "100%",
        textAlign: "left",
        background: active ? "rgba(255,255,255,0.07)" : "transparent",
        border: "none",
        borderRadius: 5,
        color: active ? "#e8eaed" : "#8b95a1",
        fontSize: 13,
        fontWeight: active ? 600 : 400,
        padding: "6px 8px",
        cursor: "pointer",
        marginBottom: 1,
        transition: "background 0.1s, color 0.1s",
      }}
      onMouseEnter={(e) => {
        if (!active) {
          (e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.04)";
          (e.currentTarget as HTMLElement).style.color = "#c8d0da";
        }
      }}
      onMouseLeave={(e) => {
        if (!active) {
          (e.currentTarget as HTMLElement).style.background = "transparent";
          (e.currentTarget as HTMLElement).style.color = "#8b95a1";
        }
      }}
    >
      <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>{label}</span>
      {count > 0 && (
        <span
          style={{
            fontSize: 11,
            color: active ? "#6b7a8d" : "#3d4a59",
            marginLeft: 6,
            flexShrink: 0,
          }}
        >
          {count}
        </span>
      )}
    </button>
  );
}

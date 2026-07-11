"use client";

import { useState } from "react";
import type { DailyRecapRow, RecapSource, StoredRssTopicRule } from "@/lib/server/neon";

const MAX_RETRIES_PER_TOPIC = 3;
const RETRY_BASE_DELAY_MS = 1500;
const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

function renderInline(text: string, keyPrefix: string): React.ReactNode[] {
  return text.split(/(\*\*[^*\n]+\*\*|\*[^*\n]+\*|`[^`\n]+`)/).map((part, j) => {
    if (part.startsWith("**") && part.endsWith("**"))
      return <strong key={`${keyPrefix}-${j}`} className="font-semibold text-[color:var(--ink)]">{part.slice(2, -2)}</strong>;
    if (part.startsWith("*") && part.endsWith("*"))
      return <em key={`${keyPrefix}-${j}`} className="italic">{part.slice(1, -1)}</em>;
    if (part.startsWith("`") && part.endsWith("`"))
      return <code key={`${keyPrefix}-${j}`} className="rounded bg-[rgba(79,213,255,0.1)] px-1 font-mono text-xs text-[color:rgba(79,213,255,0.85)]">{part.slice(1, -1)}</code>;
    return part;
  });
}

function ChatMarkdownSimple({ content }: { content: string }) {
  const lines = content.replace(/\r\n?/g, "\n").trim().split("\n");

  type Block = { type: "p"; text: string } | { type: "ul"; items: string[] };
  const blocks: Block[] = [];

  for (const line of lines) {
    const bulletMatch = line.match(/^[-*]\s+(.+)/);
    if (bulletMatch) {
      const last = blocks[blocks.length - 1];
      if (last?.type === "ul") last.items.push(bulletMatch[1]);
      else blocks.push({ type: "ul", items: [bulletMatch[1]] });
    } else if (line.trim() === "") {
      // skip blank lines
    } else {
      blocks.push({ type: "p", text: line });
    }
  }

  return (
    <div className="space-y-2 text-sm text-[color:var(--ink-soft)]">
      {blocks.map((block, i) => {
        if (block.type === "ul") {
          return (
            <ul key={i} className="space-y-1 pl-1">
              {block.items.map((item, j) => (
                <li key={j} className="flex gap-2 leading-6">
                  <span className="mt-[3px] shrink-0 text-[color:var(--accent)] opacity-60">•</span>
                  <span>{renderInline(item, `${i}-${j}`)}</span>
                </li>
              ))}
            </ul>
          );
        }
        return <p key={i} className="leading-6">{renderInline(block.text, String(i))}</p>;
      })}
    </div>
  );
}

function ToneBar({ positive, negative, neutral }: { positive: number; negative: number; neutral: number }) {
  const total = positive + negative + neutral;
  if (total === 0) return null;
  return (
    <div className="flex items-center gap-3 text-xs text-[color:var(--ink-faint)]">
      {positive > 0 && <span className="text-[#41d39d]">▲ {positive} bullish</span>}
      {negative > 0 && <span className="text-[#ff6b7f]">▼ {negative} bearish</span>}
      {neutral > 0 && <span>◆ {neutral} neutral</span>}
    </div>
  );
}

function SourceLink({ s }: { s: RecapSource }) {
  return (
    <a
      href={s.url}
      target="_blank"
      rel="noopener noreferrer"
      className="text-xs text-[color:var(--ink-faint)] hover:text-[color:var(--accent)] hover:underline"
    >
      {s.title}{s.speaker ? ` — ${s.speaker}` : ""}
    </a>
  );
}

function SourcesList({ sources }: { sources: RecapSource[] }) {
  if (!sources.length) return null;

  const secSpeeches = sources.filter((s) => s.source_type === "document" && s.source_kind === "sec_speech");
  const isNewsLikeDocument = (s: RecapSource) => {
    const sourceKind = String(s.source_kind || "").toLowerCase();
    return s.source_type === "document" && (
      sourceKind.includes("article") ||
      sourceKind.startsWith("google_news_") ||
      sourceKind === "newsapi_article" ||
      sourceKind.includes("public_article")
    );
  };
  const otherDocs = sources.filter((s) => s.source_type === "document" && s.source_kind !== "sec_speech" && !isNewsLikeDocument(s));
  const news = sources.filter((s) => s.source_type !== "document" || isNewsLikeDocument(s));

  return (
    <div className="mt-4 space-y-3 border-t border-[color:var(--line)] pt-3">
      {secSpeeches.length > 0 && (
        <div>
          <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-[color:var(--accent)]">SEC Speeches</p>
          <ul className="space-y-1">
            {secSpeeches.map((s, i) => <li key={i}><SourceLink s={s} /></li>)}
          </ul>
        </div>
      )}
      {otherDocs.length > 0 && (
        <div>
          <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-[color:rgba(79,213,255,0.6)]">Documents</p>
          <ul className="space-y-1">
            {otherDocs.map((s, i) => <li key={i}><SourceLink s={s} /></li>)}
          </ul>
        </div>
      )}
      {news.length > 0 && (
        <div>
          <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-[color:var(--ink-faint)]">News</p>
          <ul className="space-y-1">
            {news.map((s, i) => <li key={i}><SourceLink s={s} /></li>)}
          </ul>
        </div>
      )}
    </div>
  );
}

function RecapCard({ row }: { row: DailyRecapRow }) {
  const generatedAt = new Date(row.generated_at).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
  return (
    <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(8,18,30,0.9)] p-4">
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold uppercase tracking-[0.14em] text-[color:var(--ink)]">
            {row.topic_label}
          </span>
          <span className="text-xs text-[color:var(--ink-faint)]">{row.article_count} articles</span>
          {row.fallback && (
            <span
              title="The model call failed or timed out; this is a templated source list, not an AI-generated summary."
              className="rounded-full border border-[#ffb84d]/40 bg-[#ffb84d]/10 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.08em] text-[#ffb84d]"
            >
              Fallback
            </span>
          )}
        </div>
        <span className="shrink-0 text-[10px] text-[color:var(--ink-faint)]">generated {generatedAt}</span>
      </div>
      <ToneBar positive={row.positive_count} negative={row.negative_count} neutral={row.neutral_count} />
      <div className="mt-3">
        <ChatMarkdownSimple content={row.summary} />
      </div>
      <SourcesList sources={row.sources ?? []} />
    </div>
  );
}

export function RecapDashboard({
  initialTopicRules,
  initialSelectedKeys,
  initialRecap,
}: {
  initialTopicRules: StoredRssTopicRule[];
  initialSelectedKeys: string[];
  initialRecap: DailyRecapRow[];
}) {
  const [selectedKeys, setSelectedKeys] = useState<Set<string>>(new Set(initialSelectedKeys));
  const [settingsSaving, setSettingsSaving] = useState(false);
  const [settingsSaved, setSettingsSaved] = useState(false);

  const [recap, setRecap] = useState<DailyRecapRow[]>(initialRecap);
  const [generating, setGenerating] = useState(false);
  const [generateProgress, setGenerateProgress] = useState<string | null>(null);
  const [generateError, setGenerateError] = useState<string | null>(null);
  const [skippedTopics, setSkippedTopics] = useState<{ topic_key: string; topic_label: string }[]>([]);
  const [loadingDate, setLoadingDate] = useState(false);

  const todayIso = new Date().toISOString().split("T")[0] as string;
  const [viewDate, setViewDate] = useState<string>(todayIso);

  const isToday = viewDate === todayIso;

  const loadDate = async (date: string) => {
    if (date === viewDate) return;
    setViewDate(date);
    setLoadingDate(true);
    setGenerateError(null);
    setSkippedTopics([]);
    try {
      const res = await fetch(`/api/intel/recap?date=${date}`);
      if (!res.ok) { setGenerateError(`Failed to load recap for ${date} (${res.status})`); return; }
      const json = (await res.json()) as { ok: boolean; data?: { recap: DailyRecapRow[] } };
      if (json.ok && json.data) setRecap(json.data.recap);
      else if (!json.ok) setGenerateError((json as { error?: string }).error ?? `No recap found for ${date}`);
    } catch (err) {
      setGenerateError(err instanceof Error ? err.message : "Network error");
    } finally {
      setLoadingDate(false);
    }
  };

  const toggleTopic = (key: string) => {
    setSelectedKeys((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
    setSettingsSaved(false);
  };

  const saveSettings = async () => {
    setSettingsSaving(true);
    try {
      const res = await fetch("/api/intel/recap/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topicKeys: [...selectedKeys] }),
      });
      if (!res.ok) {
        const d = await res.json().catch(() => ({})) as { error?: string };
        setGenerateError(d.error ?? `Save failed (HTTP ${res.status})`);
        return;
      }
      setSettingsSaved(true);
    } catch (err) {
      setGenerateError(err instanceof Error ? err.message : "Network error");
    } finally {
      setSettingsSaving(false);
    }
  };

  const generate = async () => {
    const topicKeys = [...selectedKeys];
    if (topicKeys.length === 0) {
      setGenerateError("No topics selected. Save topic settings first.");
      return;
    }

    setGenerating(true);
    setGenerateProgress("Starting recap generation...");
    setGenerateError(null);
    setSkippedTopics([]);
    try {
      const allSkipped: { topic_key: string; topic_label: string }[] = [];
      const allFailed: { topic_key: string; topic_label: string; error: string }[] = [];
      const allWarnings = new Set<string>();
      let generatedCount = 0;

      // One request per topic, keyed by topic_key (not a positional cursor
      // recomputed server-side each call — see generate/route.ts). A 429
      // from the shared rate limiter retries the same topic with backoff
      // instead of aborting the whole run and losing every topic after it.
      for (let i = 0; i < topicKeys.length; i += 1) {
        const topicKey = topicKeys[i];
        const fallbackLabel = initialTopicRules.find((r) => r.topic_key === topicKey)?.label ?? topicKey;
        setGenerateProgress(`Generating recap topics... ${i + 1}/${topicKeys.length} (${fallbackLabel})`);

        let attempt = 0;
        let settled = false;
        while (!settled) {
          attempt += 1;
          let res: Response;
          try {
            res = await fetch("/api/intel/recap/generate", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ date: viewDate, topicKey }),
            });
          } catch (err) {
            allFailed.push({ topic_key: topicKey, topic_label: fallbackLabel, error: err instanceof Error ? err.message : "Network error" });
            break;
          }

          if (res.status === 429 && attempt < MAX_RETRIES_PER_TOPIC) {
            await sleep(RETRY_BASE_DELAY_MS * attempt);
            continue;
          }

          let json: {
            ok: boolean;
            error?: string;
            data?: {
              topicKey?: string;
              topicLabel?: string;
              status?: "generated" | "skipped" | "failed";
              error?: string;
              warnings?: string[];
            };
          };
          try {
            json = (await res.json()) as typeof json;
          } catch {
            allFailed.push({ topic_key: topicKey, topic_label: fallbackLabel, error: `Server error ${res.status}: ${res.statusText || "non-JSON response"}` });
            break;
          }

          if (!json.ok) {
            allFailed.push({ topic_key: topicKey, topic_label: fallbackLabel, error: json.error ?? `Generation failed (HTTP ${res.status}).` });
            break;
          }

          const label = json.data?.topicLabel ?? fallbackLabel;
          for (const warning of json.data?.warnings ?? []) allWarnings.add(warning);
          if (json.data?.status === "generated") generatedCount += 1;
          else if (json.data?.status === "skipped") allSkipped.push({ topic_key: topicKey, topic_label: label });
          else if (json.data?.status === "failed") allFailed.push({ topic_key: topicKey, topic_label: label, error: json.data.error ?? "Unknown error" });
          settled = true;
        }
      }

      setSkippedTopics(Array.from(new Map(allSkipped.map((topic) => [topic.topic_key, topic])).values()));
      const messages: string[] = [];
      if (allFailed.length > 0) {
        messages.push(`Some topics failed: ${allFailed.map((item) => `${item.topic_label}: ${item.error}`).join("; ")}`);
      }
      messages.push(...allWarnings);
      if (generatedCount === 0 && allSkipped.length === 0 && allFailed.length === 0) {
        messages.push("The recap request completed, but no topic rows were generated. Save recap settings again, then retry.");
      }
      if (messages.length > 0) setGenerateError(messages.join(" "));

      const recapRes = await fetch(`/api/intel/recap?date=${viewDate}`);
      if (!recapRes.ok) {
        setGenerateError(`Failed to reload recap (HTTP ${recapRes.status})`);
        return;
      }
      const recapJson = (await recapRes.json()) as { ok: boolean; data?: { recap: DailyRecapRow[] } };
      if (recapJson.ok && recapJson.data) setRecap(recapJson.data.recap);
    } catch (err) {
      setGenerateError(err instanceof Error ? err.message : String(err));
    } finally {
      setGenerating(false);
      setGenerateProgress(null);
    }
  };

  const topicCols = [
    initialTopicRules.slice(0, Math.ceil(initialTopicRules.length / 2)),
    initialTopicRules.slice(Math.ceil(initialTopicRules.length / 2)),
  ];

  return (
    <div className="space-y-6">
      {/* Settings */}
      <section className="panel p-5">
        <h2 className="text-xs font-semibold uppercase tracking-[0.14em] text-[color:var(--ink-faint)]">Recap Settings</h2>
        <p className="mt-1 text-xs text-[color:var(--ink-faint)]">Choose which topics to include in your daily recap.</p>
        <div className="mt-4 grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-3">
          {topicCols.flat().map((rule) => (
            <label key={rule.topic_key} className="flex cursor-pointer items-center gap-2 text-sm text-[color:var(--ink-soft)]">
              <input
                type="checkbox"
                checked={selectedKeys.has(rule.topic_key)}
                onChange={() => toggleTopic(rule.topic_key)}
                className="h-3.5 w-3.5 rounded accent-[color:var(--accent)]"
              />
              {rule.label}
            </label>
          ))}
        </div>
        <div className="mt-4 flex items-center gap-3">
          <button
            onClick={() => void saveSettings()}
            disabled={settingsSaving}
            className="btn-solid px-4 py-1.5 text-sm disabled:opacity-50"
          >
            {settingsSaving ? "Saving…" : "Save Settings"}
          </button>
          {settingsSaved && <span className="text-xs text-[#41d39d]">Saved</span>}
        </div>
      </section>

      {/* Recap */}
      <section className="panel p-5">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h2 className="text-xs font-semibold uppercase tracking-[0.14em] text-[color:var(--ink-faint)]">
              {isToday ? "Today's Recap" : "Recap"}
            </h2>
            <input
              type="date"
              value={viewDate}
              max={todayIso}
              onChange={(e) => { if (e.target.value) void loadDate(e.target.value); }}
              disabled={loadingDate || generating}
              className="mt-1 rounded-lg border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.95)] px-2 py-0.5 text-xs text-[color:var(--ink-faint)] [color-scheme:dark] disabled:opacity-50"
            />
          </div>
          <button
            onClick={() => void generate()}
            disabled={generating || loadingDate}
            className="btn-solid shrink-0 px-4 py-1.5 text-sm disabled:opacity-50"
          >
            {generating ? "Generating…" : recap.length > 0 ? "Regenerate" : "Generate Recap"}
          </button>
        </div>

        {generateProgress && (
          <p className="mt-3 text-xs text-[color:var(--ink-faint)]">{generateProgress}</p>
        )}

        {generateError && (
          <p className="callout callout-error mt-3">{generateError}</p>
        )}

        {generating && (
          <div className="mt-4 space-y-3">
            {[...selectedKeys].map((key) => {
              const rule = initialTopicRules.find((r) => r.topic_key === key);
              return (
                <div key={key} className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(8,18,30,0.9)] p-4">
                  <div className="text-xs font-semibold uppercase tracking-[0.14em] text-[color:var(--ink-faint)]">
                    {rule?.label ?? key}
                  </div>
                  <div className="mt-2 h-2 w-2/3 animate-pulse rounded bg-[color:rgba(79,213,255,0.15)]" />
                  <div className="mt-1.5 h-2 w-full animate-pulse rounded bg-[color:rgba(79,213,255,0.08)]" />
                  <div className="mt-1.5 h-2 w-4/5 animate-pulse rounded bg-[color:rgba(79,213,255,0.08)]" />
                </div>
              );
            })}
          </div>
        )}

        {loadingDate && (
          <p className="mt-4 text-sm text-[color:var(--ink-faint)]">Loading…</p>
        )}

        {!generating && !loadingDate && recap.length > 0 && (
          <div className="mt-4 space-y-3">
            {recap.map((row) => <RecapCard key={row.topic_key} row={row} />)}
          </div>
        )}

        {!generating && !loadingDate && skippedTopics.length > 0 && (
          <p className="mt-3 text-xs text-[color:var(--ink-faint)]">
            No matching articles for: {skippedTopics.map((t) => t.topic_label).join(", ")}
          </p>
        )}

        {!generating && !loadingDate && recap.length === 0 && skippedTopics.length === 0 && (
          <p className="mt-4 text-sm text-[color:var(--ink-faint)]">
            {isToday ? "No recap for today yet. Select your topics above and click Generate." : "No recap stored for this date."}
          </p>
        )}
      </section>
    </div>
  );
}

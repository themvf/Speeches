"use client";

import { useState } from "react";
import type { DailyRecapRow, StoredRssTopicRule } from "@/lib/server/neon";

function ChatMarkdownSimple({ content }: { content: string }) {
  const blocks = content.replace(/\r\n?/g, "\n").trim().split(/\n{2,}/).filter(Boolean);
  return (
    <div className="space-y-2 text-sm text-[color:var(--ink-soft)]">
      {blocks.map((block, i) => {
        const parts = block.split(/(\*\*[^*\n]+\*\*|\*[^*\n]+\*|`[^`\n]+`)/);
        const rendered = parts.map((part, j) => {
          if (part.startsWith("**") && part.endsWith("**"))
            return <strong key={j} className="font-semibold text-[color:var(--ink)]">{part.slice(2, -2)}</strong>;
          if (part.startsWith("*") && part.endsWith("*"))
            return <em key={j} className="italic">{part.slice(1, -1)}</em>;
          if (part.startsWith("`") && part.endsWith("`"))
            return <code key={j} className="rounded bg-[rgba(79,213,255,0.1)] px-1 font-mono text-xs text-[color:rgba(79,213,255,0.85)]">{part.slice(1, -1)}</code>;
          return part;
        });
        return <p key={i} className="leading-6">{rendered}</p>;
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
        </div>
        <span className="shrink-0 text-[10px] text-[color:var(--ink-faint)]">generated {generatedAt}</span>
      </div>
      <ToneBar positive={row.positive_count} negative={row.negative_count} neutral={row.neutral_count} />
      <div className="mt-3">
        <ChatMarkdownSimple content={row.summary} />
      </div>
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
  const [generateError, setGenerateError] = useState<string | null>(null);

  const today = new Date().toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" });

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
      await fetch("/api/intel/recap/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topicKeys: [...selectedKeys] }),
      });
      setSettingsSaved(true);
    } finally {
      setSettingsSaving(false);
    }
  };

  const generate = async () => {
    setGenerating(true);
    setGenerateError(null);
    try {
      const res = await fetch("/api/intel/recap/generate", { method: "POST" });
      let json: { ok: boolean; error?: string };
      try {
        json = (await res.json()) as { ok: boolean; error?: string };
      } catch {
        setGenerateError(`Server error ${res.status}: ${res.statusText || "non-JSON response"}`);
        return;
      }
      if (!json.ok) {
        setGenerateError(json.error ?? `Generation failed (HTTP ${res.status}).`);
        return;
      }
      const recapRes = await fetch("/api/intel/recap");
      const recapJson = (await recapRes.json()) as { ok: boolean; data?: { recap: DailyRecapRow[] } };
      if (recapJson.ok && recapJson.data) setRecap(recapJson.data.recap);
    } catch (err) {
      setGenerateError(err instanceof Error ? err.message : String(err));
    } finally {
      setGenerating(false);
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
            <h2 className="text-xs font-semibold uppercase tracking-[0.14em] text-[color:var(--ink-faint)]">Today&apos;s Recap</h2>
            <p className="mt-0.5 text-xs text-[color:var(--ink-faint)]">{today}</p>
          </div>
          <button
            onClick={() => void generate()}
            disabled={generating}
            className="btn-solid shrink-0 px-4 py-1.5 text-sm disabled:opacity-50"
          >
            {generating ? "Generating…" : recap.length > 0 ? "Regenerate" : "Generate Recap"}
          </button>
        </div>

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

        {!generating && recap.length > 0 && (
          <div className="mt-4 space-y-3">
            {recap.map((row) => <RecapCard key={row.topic_key} row={row} />)}
          </div>
        )}

        {!generating && recap.length === 0 && (
          <p className="mt-4 text-sm text-[color:var(--ink-faint)]">
            No recap for today yet. Select your topics above and click Generate.
          </p>
        )}
      </section>
    </div>
  );
}

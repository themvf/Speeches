"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

type Settings = {
  enabled: boolean;
  timezone: string;
  hour: number;
  minute: number;
  lookback_hours: number;
  openai_enabled: boolean;
  openai_model: string;
  deepseek_enabled: boolean;
  deepseek_model: string;
  blind_comparison: boolean;
  rough_draft: boolean;
};

type Source = {
  source_id: string;
  title: string;
  description: string;
  url: string;
  publisher: string;
  published_at: string | null;
};

type Output = {
  id: number;
  provider: "openai" | "deepseek";
  model: string;
  status: "completed" | "failed";
  latency_ms: number;
  usage: Record<string, unknown>;
  package: Record<string, unknown> | null;
  error: string;
};

type Run = {
  id: number;
  run_date: string;
  trigger: "manual" | "scheduled";
  status: "running" | "completed" | "partial" | "failed";
  snapshot_hash: string;
  source_count: number;
  source_snapshot: Source[];
  settings_snapshot: Settings;
  error: string;
  started_at: string;
  finished_at: string | null;
  outputs: Output[];
};

type Runtime = { openai_configured: boolean; deepseek_configured: boolean };

const DEFAULT_SETTINGS: Settings = {
  enabled: false,
  timezone: "America/New_York",
  hour: 21,
  minute: 0,
  lookback_hours: 24,
  openai_enabled: true,
  openai_model: "gpt-5.6-luna",
  deepseek_enabled: true,
  deepseek_model: "deepseek-v4-pro",
  blind_comparison: true,
  rough_draft: true,
};

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function records(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.map(record) : [];
}

function strings(value: unknown): string[] {
  return Array.isArray(value) ? value.map((item) => String(item)).filter(Boolean) : [];
}

async function apiBody(response: Response): Promise<Record<string, unknown>> {
  const text = await response.text();
  if (!text) return { ok: response.ok };
  try {
    return JSON.parse(text) as Record<string, unknown>;
  } catch {
    return { ok: false, error: text };
  }
}

function Toggle({ checked, onChange, label, detail, disabled = false }: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  detail: string;
  disabled?: boolean;
}) {
  return (
    <label className={`flex gap-3 rounded-xl border border-[color:var(--line-soft)] bg-[color:rgba(8,18,30,0.52)] p-3 ${disabled ? "opacity-55" : "cursor-pointer"}`}>
      <input type="checkbox" checked={checked} disabled={disabled} onChange={(event) => onChange(event.target.checked)} className="mt-1 h-4 w-4" />
      <span>
        <span className="block text-sm font-semibold text-[color:var(--ink)]">{label}</span>
        <span className="mt-0.5 block text-xs leading-5 text-[color:var(--ink-faint)]">{detail}</span>
      </span>
    </label>
  );
}

function StatusChip({ status }: { status: Run["status"] | Output["status"] }) {
  const tone = status === "completed" ? "status-success" : status === "partial" || status === "running" ? "status-warn" : "status-failure";
  return <span className={`status-chip ${tone}`}>{status}</span>;
}

function OutputView({ output, label, revealProvider }: { output: Output; label: string; revealProvider: boolean }) {
  if (output.status === "failed" || !output.package) {
    return (
      <section className="panel p-5">
        <div className="flex items-center justify-between gap-3"><h3 className="text-lg font-semibold">{label}</h3><StatusChip status="failed" /></div>
        <p className="mt-3 text-sm text-[color:var(--danger)]">{output.error || "Generation failed."}</p>
      </section>
    );
  }
  const pkg = record(output.package);
  const recommendation = record(pkg.editorial_recommendation);
  const candidates = records(pkg.candidates);
  const selected = record(pkg.selected_package);
  const selectedId = String(recommendation.selected_candidate_id || "");
  const draft = typeof pkg.draft === "string" ? pkg.draft : "";
  return (
    <div className="space-y-4">
      <section className="panel p-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="kicker">{label}</p>
            <h2 className="mt-4 text-xl font-semibold text-[color:var(--ink)]">Editorial recommendation</h2>
          </div>
          <div className="text-right text-xs text-[color:var(--ink-faint)]">
            {revealProvider ? <p>{output.provider === "openai" ? "OpenAI" : "DeepSeek"} · {output.model}</p> : <p>Provider hidden</p>}
            <p>{(output.latency_ms / 1000).toFixed(1)} seconds</p>
          </div>
        </div>
        <div className="mt-4 flex items-center gap-3">
          <span className={String(recommendation.decision) === "publish" ? "status-chip status-success" : "status-chip status-warn"}>{String(recommendation.decision || "review")}</span>
          <p className="text-sm leading-6 text-[color:var(--ink-soft)]">{String(recommendation.rationale || "")}</p>
        </div>
      </section>

      <section className="panel p-5">
        <h2 className="text-lg font-semibold text-[color:var(--ink)]">Candidate angles</h2>
        <div className="mt-4 grid gap-3 lg:grid-cols-3">
          {candidates.map((candidate, index) => (
            <article key={String(candidate.candidate_id || index)} className={`rounded-xl border p-4 ${String(candidate.candidate_id) === selectedId ? "border-[color:rgba(79,213,255,0.6)] bg-[color:rgba(79,213,255,0.08)]" : "border-[color:var(--line-soft)]"}`}>
              <div className="flex items-start justify-between gap-2">
                <h3 className="font-semibold text-[color:var(--ink)]">{String(candidate.working_title || "Untitled angle")}</h3>
                {String(candidate.candidate_id) === selectedId ? <span className="status-chip status-success">selected</span> : null}
              </div>
              <p className="mt-2 text-xs italic leading-5 text-[color:var(--ink-faint)]">{String(candidate.subtitle || "")}</p>
              <p className="mt-3 text-sm leading-6 text-[color:var(--ink-soft)]">{String(candidate.thesis || "")}</p>
              <p className="mt-3 text-xs text-[color:var(--ink-faint)]">Support {String(candidate.support_score || "–")}/5 · Originality {String(candidate.originality_score || "–")}/5 · Recap risk {String(candidate.recap_risk || "–")}</p>
            </article>
          ))}
        </div>
      </section>

      {draft ? (
        <section className="panel p-5 md:p-7">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div><p className="kicker">Editable rough draft</p><h2 className="mt-4 text-xl font-semibold">Medium-style article</h2></div>
            <button type="button" className="btn-muted px-3 py-2 text-sm" onClick={() => navigator.clipboard.writeText(draft)}>Copy draft</button>
          </div>
          <div className="mt-6 whitespace-pre-wrap text-[15px] leading-7 text-[color:var(--ink-soft)]">{draft}</div>
        </section>
      ) : null}

      <div className="grid gap-4 lg:grid-cols-2">
        <section className="panel p-5">
          <h2 className="text-lg font-semibold">Outline</h2>
          <div className="mt-4 space-y-3">
            {records(selected.outline).map((section, index) => (
              <div key={`${String(section.heading)}-${index}`} className="rounded-xl border border-[color:var(--line-soft)] p-3">
                <h3 className="text-sm font-semibold">{String(section.heading || "Section")}</h3>
                <p className="mt-1 text-sm leading-6 text-[color:var(--ink-soft)]">{String(section.purpose || "")}</p>
                <p className="mt-2 text-xs text-[color:var(--ink-faint)]">{strings(section.source_ids).join(" · ")}</p>
              </div>
            ))}
          </div>
        </section>
        <section className="panel p-5">
          <h2 className="text-lg font-semibold">Questions for your edit</h2>
          <ul className="mt-4 space-y-3">
            {strings(selected.author_questions).map((question) => <li key={question} className="text-sm leading-6 text-[color:var(--ink-soft)]">• {question}</li>)}
          </ul>
          {strings(pkg.quality_warnings).length ? <>
            <h3 className="mt-6 text-sm font-semibold uppercase tracking-[0.08em] text-[color:var(--warn)]">Quality warnings</h3>
            <ul className="mt-3 space-y-2">{strings(pkg.quality_warnings).map((warning) => <li key={warning} className="text-xs leading-5 text-[color:var(--ink-faint)]">• {warning}</li>)}</ul>
          </> : null}
        </section>
      </div>
    </div>
  );
}

export function ScheduledEditorialDashboard() {
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [runtime, setRuntime] = useState<Runtime>({ openai_configured: false, deepseek_configured: false });
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [selectedOutput, setSelectedOutput] = useState(0);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch("/api/admin/briefings/scheduled", { cache: "no-store" });
      const body = await apiBody(response);
      if (!response.ok || !body.ok) throw new Error(String(body.error || "Unable to load scheduled briefings."));
      const data = record(body.data);
      setSettings(data.settings as Settings);
      setRuntime(data.runtime as Runtime);
      setRuns(data.runs as Run[]);
      setSelectedRunId((current) => current ?? (data.runs as Run[])[0]?.id ?? null);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : String(loadError));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  const selectedRun = useMemo(() => runs.find((run) => run.id === selectedRunId) || runs[0] || null, [runs, selectedRunId]);
  const outputLabels = selectedRun?.outputs.map((output, index) => selectedRun.settings_snapshot.blind_comparison ? `Draft ${String.fromCharCode(65 + index)}` : output.provider === "openai" ? "OpenAI · Luna" : "DeepSeek") || [];

  const save = async (): Promise<boolean> => {
    setSaving(true); setError(null); setNotice(null);
    try {
      const response = await fetch("/api/admin/briefings/scheduled", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ settings }) });
      const body = await apiBody(response);
      if (!response.ok || !body.ok) throw new Error(String(body.error || "Unable to save settings."));
      setSettings(record(body.data).settings as Settings);
      setNotice("Schedule settings saved.");
      return true;
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : String(saveError));
      return false;
    } finally { setSaving(false); }
  };

  const runNow = async () => {
    if (!await save()) return;
    setRunning(true); setNotice("Generating from a frozen source snapshot…"); setError(null);
    try {
      const response = await fetch("/api/admin/briefings/scheduled", { method: "POST" });
      const body = await apiBody(response);
      if (!response.ok || !body.ok) throw new Error(String(body.error || "Generation failed."));
      setNotice("Editorial briefing generated and stored.");
      await load();
      const run = record(record(body.data).run);
      if (run.id) { setSelectedRunId(Number(run.id)); setSelectedOutput(0); }
    } catch (runError) {
      setError(runError instanceof Error ? runError.message : String(runError));
    } finally { setRunning(false); }
  };

  if (loading) return <section className="panel p-6 text-sm text-[color:var(--ink-faint)]">Loading scheduled editorial workspace…</section>;

  return (
    <div className="space-y-5">
      <section className="panel hero-panel overflow-hidden p-5 md:p-7">
        <div className="relative z-10 flex flex-col justify-between gap-5 lg:flex-row lg:items-end">
          <div className="max-w-3xl">
            <p className="kicker">Nightly editorial desk</p>
            <h1 className="mt-4 text-3xl font-semibold text-[color:var(--ink)]">Daily AI Editorial</h1>
            <p className="mt-2 text-sm leading-6 text-[color:var(--ink-soft)]">Turn the last 24 hours of captured AI news into a source-bounded editorial package and editable Medium-style draft. Runs are stored here for comparison and review.</p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button type="button" onClick={() => void save()} disabled={saving || running} className="btn-muted px-4 py-2 text-sm disabled:opacity-50">{saving ? "Saving…" : "Save settings"}</button>
            <button type="button" onClick={() => void runNow()} disabled={saving || running} className="btn-accent px-4 py-2 text-sm disabled:opacity-50">{running ? "Generating…" : "Run now"}</button>
          </div>
        </div>
      </section>

      {error ? <div className="callout callout-error">{error}{error === "Unauthorized" || error.includes("Admin access") ? <> <Link href="/admin/login?next=/briefings/scheduled" className="underline">Sign in as admin</Link>.</> : null}</div> : null}
      {notice ? <div className="callout callout-info">{notice}</div> : null}

      <div className="grid gap-5 xl:grid-cols-[0.9fr_1.3fr]">
        <section className="panel p-5">
          <div className="flex items-start justify-between gap-3"><div><h2 className="text-lg font-semibold">Schedule and output</h2><p className="mt-1 text-xs text-[color:var(--ink-faint)]">9:00 PM Eastern, with daylight-saving adjustment.</p></div><span className={settings.enabled ? "status-chip status-success" : "status-chip status-neutral"}>{settings.enabled ? "active" : "paused"}</span></div>
          <div className="mt-4 space-y-3">
            <Toggle checked={settings.enabled} onChange={(enabled) => setSettings({ ...settings, enabled })} label="Nightly briefing" detail="Allow the Vercel cron route to generate one stored run each evening." />
            <Toggle checked={settings.rough_draft} onChange={(rough_draft) => setSettings({ ...settings, rough_draft })} label="Editable rough draft" detail="Include a 900–1,400 word Medium-style draft alongside the editorial package." />
            <Toggle checked={settings.blind_comparison} onChange={(blind_comparison) => setSettings({ ...settings, blind_comparison })} label="Blind comparison" detail="Show provider outputs as Draft A and Draft B during review." />
          </div>
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            <label className="text-xs font-semibold text-[color:var(--ink-faint)]">Run hour<input className="form-control mt-1 w-full px-3 py-2 text-sm" type="number" min={0} max={23} value={settings.hour} onChange={(event) => setSettings({ ...settings, hour: Number(event.target.value) })} /></label>
            <label className="text-xs font-semibold text-[color:var(--ink-faint)]">Lookback hours<input className="form-control mt-1 w-full px-3 py-2 text-sm" type="number" min={6} max={72} value={settings.lookback_hours} onChange={(event) => setSettings({ ...settings, lookback_hours: Number(event.target.value) })} /></label>
          </div>
        </section>

        <section className="panel p-5">
          <h2 className="text-lg font-semibold">Providers</h2>
          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <div className="rounded-xl border border-[color:var(--line-soft)] p-4">
              <Toggle checked={settings.openai_enabled} onChange={(openai_enabled) => setSettings({ ...settings, openai_enabled })} label="OpenAI" detail={runtime.openai_configured ? "Configured and ready." : "OPENAI_API_KEY is missing; disable this provider until configured."} />
              <label className="mt-3 block text-xs font-semibold text-[color:var(--ink-faint)]">Model<input className="form-control mt-1 w-full px-3 py-2 text-sm" value={settings.openai_model} onChange={(event) => setSettings({ ...settings, openai_model: event.target.value })} /></label>
              <p className="mt-2 text-xs text-[color:var(--ok)]">Default: GPT-5.6 Luna</p>
            </div>
            <div className="rounded-xl border border-[color:var(--line-soft)] p-4">
              <Toggle checked={settings.deepseek_enabled} onChange={(deepseek_enabled) => setSettings({ ...settings, deepseek_enabled })} label="DeepSeek" detail={runtime.deepseek_configured ? "Configured and ready." : "DEEPSEEK_API is missing; disable this provider until configured."} />
              <label className="mt-3 block text-xs font-semibold text-[color:var(--ink-faint)]">Model<input className="form-control mt-1 w-full px-3 py-2 text-sm" value={settings.deepseek_model} onChange={(event) => setSettings({ ...settings, deepseek_model: event.target.value })} /></label>
            </div>
          </div>
        </section>
      </div>

      <section className="panel p-5">
        <div className="flex flex-wrap items-center justify-between gap-3"><div><h2 className="text-lg font-semibold">Saved outputs</h2><p className="mt-1 text-xs text-[color:var(--ink-faint)]">Every run retains its exact source snapshot and provider results.</p></div><button type="button" onClick={() => void load()} className="btn-muted px-3 py-2 text-sm">Refresh</button></div>
        {runs.length ? <div className="mt-4 flex gap-2 overflow-x-auto pb-2">{runs.map((run) => <button key={run.id} type="button" onClick={() => { setSelectedRunId(run.id); setSelectedOutput(0); }} className={`min-w-44 rounded-xl border p-3 text-left ${selectedRun?.id === run.id ? "border-[color:rgba(79,213,255,0.6)] bg-[color:rgba(79,213,255,0.1)]" : "border-[color:var(--line-soft)]"}`}><div className="flex items-center justify-between gap-2"><span className="text-sm font-semibold">{run.run_date}</span><StatusChip status={run.status} /></div><p className="mt-2 text-xs text-[color:var(--ink-faint)]">{run.trigger} · {run.source_count} sources</p></button>)}</div> : <p className="mt-4 text-sm text-[color:var(--ink-faint)]">No stored runs yet. Use Run now to create the first one.</p>}
      </section>

      {selectedRun ? <>
        <section className="panel p-4">
          <div className="flex flex-wrap items-center justify-between gap-3"><div className="flex flex-wrap gap-2">{selectedRun.outputs.map((output, index) => <button key={output.id} type="button" onClick={() => setSelectedOutput(index)} className={selectedOutput === index ? "btn-solid px-4 py-2 text-sm" : "btn-muted px-4 py-2 text-sm"}>{outputLabels[index]}</button>)}</div><p className="text-xs text-[color:var(--ink-faint)]">Snapshot {selectedRun.snapshot_hash.slice(0, 12)} · {selectedRun.source_count} sources</p></div>
        </section>
        {selectedRun.outputs[selectedOutput] ? <OutputView output={selectedRun.outputs[selectedOutput]} label={outputLabels[selectedOutput] || "Draft"} revealProvider={!selectedRun.settings_snapshot.blind_comparison} /> : null}
        <section className="panel p-5"><details><summary className="cursor-pointer text-sm font-semibold">Frozen source snapshot ({selectedRun.source_count})</summary><div className="mt-4 grid gap-3 md:grid-cols-2">{selectedRun.source_snapshot.map((source) => <a key={source.source_id} href={source.url} target="_blank" rel="noopener noreferrer" className="rounded-xl border border-[color:var(--line-soft)] p-3 hover:border-[color:var(--line-strong)]"><p className="text-sm font-semibold">{source.title}</p><p className="mt-1 text-xs text-[color:var(--ink-faint)]">{source.source_id} · {source.publisher}</p><p className="mt-2 line-clamp-3 text-xs leading-5 text-[color:var(--ink-soft)]">{source.description}</p></a>)}</div></details></section>
      </> : null}
    </div>
  );
}

"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type RunStatus = {
  id: number;
  status: string;       // queued | in_progress | completed
  conclusion: string | null; // success | failure | cancelled | null
  created_at: string;
  html_url: string;
};

const POLL_INTERVAL_ACTIVE = 12_000;  // 12s while queued/in_progress
const POLL_INTERVAL_IDLE   = 60_000;  // 60s when completed

function statusLabel(run: RunStatus): string {
  if (run.status === "queued") return "Queued";
  if (run.status === "in_progress") return "Running…";
  if (run.conclusion === "success") return "Succeeded";
  if (run.conclusion === "failure") return "Failed";
  if (run.conclusion === "cancelled") return "Cancelled";
  return run.conclusion ?? run.status;
}

function statusColor(run: RunStatus): string {
  if (run.status === "queued") return "text-[color:var(--ink-faint)]";
  if (run.status === "in_progress") return "text-[color:var(--accent)]";
  if (run.conclusion === "success") return "text-[#41d39d]";
  if (run.conclusion === "failure") return "text-[color:var(--danger)]";
  if (run.conclusion === "cancelled") return "text-[color:var(--warn)]";
  return "text-[color:var(--ink-faint)]";
}

function statusDot(run: RunStatus): string {
  if (run.status === "in_progress") return "animate-pulse bg-[color:var(--accent)]";
  if (run.conclusion === "success") return "bg-[#41d39d]";
  if (run.conclusion === "failure") return "bg-[color:var(--danger)]";
  if (run.conclusion === "cancelled") return "bg-[color:var(--warn)]";
  return "bg-[color:var(--ink-faint)]";
}

function isActive(run: RunStatus): boolean {
  return run.status === "queued" || run.status === "in_progress";
}

export function JobStatusBadge({ workflowFile }: { workflowFile: string }) {
  const [run, setRun] = useState<RunStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);

  const poll = useCallback(async () => {
    try {
      const res = await fetch(`/api/admin/workflow/runs?workflow=${encodeURIComponent(workflowFile)}`);
      const data = await res.json() as { ok: boolean; run?: RunStatus; error?: string };
      if (!mountedRef.current) return;
      if (data.ok) {
        setRun(data.run ?? null);
        setError(null);
        const interval = data.run && isActive(data.run) ? POLL_INTERVAL_ACTIVE : POLL_INTERVAL_IDLE;
        timerRef.current = setTimeout(() => { void poll(); }, interval);
      } else {
        setError(data.error ?? "Unknown error");
      }
    } catch {
      if (mountedRef.current) setError("Network error");
    }
  }, [workflowFile]);

  useEffect(() => {
    mountedRef.current = true;
    void poll();
    return () => {
      mountedRef.current = false;
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [poll]);

  if (error) return null;
  if (!run) return null;

  const relTime = (() => {
    const diffMs = Date.now() - new Date(run.created_at).getTime();
    const mins = Math.floor(diffMs / 60_000);
    if (mins < 1) return "just now";
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    return `${hrs}h ago`;
  })();

  return (
    <a
      href={run.html_url}
      target="_blank"
      rel="noopener noreferrer"
      className={`flex items-center gap-1.5 text-xs hover:underline ${statusColor(run)}`}
      title={`Last run started ${new Date(run.created_at).toLocaleString()}`}
    >
      <span className={`inline-block h-1.5 w-1.5 rounded-full ${statusDot(run)}`} />
      {statusLabel(run)}
      <span className="text-[color:var(--ink-faint)]">· {relTime}</span>
    </a>
  );
}

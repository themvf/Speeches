import { getGithubActionsConfig } from "@/lib/server/env";

interface GithubWorkflowRun {
  id: number;
  status: string;
  conclusion: string | null;
  html_url: string;
  created_at: string;
  updated_at: string;
  run_started_at: string | null;
  event: string;
  name: string;
}

interface GithubArtifact {
  id: number;
  name: string;
  archive_download_url: string;
  expired: boolean;
}

interface GithubRunSummary {
  job_id: string;
  provider: "github_actions";
  workflow: string;
  status: "queued" | "running" | "success" | "failed" | "unknown";
  github_run_id: number;
  html_url: string;
  created_at: string;
  started_at: string;
  updated_at: string;
  finished_at: string;
  conclusion: string;
  artifacts: string[];
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function extractErrorMessage(payload: unknown): string {
  if (!payload || typeof payload !== "object") {
    return "GitHub API error";
  }
  const src = payload as Record<string, unknown>;
  const msg = String(src.message ?? "GitHub API error").trim();
  if (!Array.isArray(src.errors) || src.errors.length === 0) {
    return msg;
  }
  return `${msg}: ${src.errors.map((item) => JSON.stringify(item)).join("; ")}`;
}

async function githubRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const cfg = getGithubActionsConfig();
  if (!cfg.enabled) {
    throw new Error("GitHub Actions integration is not configured.");
  }

  const response = await fetch(`https://api.github.com${path}`, {
    ...init,
    headers: {
      Accept: "application/vnd.github+json",
      Authorization: `Bearer ${cfg.token}`,
      "X-GitHub-Api-Version": "2022-11-28",
      ...(init?.headers || {})
    },
    cache: "no-store"
  });

  if (!response.ok) {
    let body: unknown = null;
    try {
      body = await response.json();
    } catch {
      // Ignore parse failure.
    }
    throw new Error(`GitHub API request failed (${response.status}): ${extractErrorMessage(body)}`);
  }

  if (response.status === 204) {
    return {} as T;
  }

  return (await response.json()) as T;
}

function mapRunStatus(status: string, conclusion: string | null): GithubRunSummary["status"] {
  const normalizedStatus = String(status || "").toLowerCase();
  const normalizedConclusion = String(conclusion || "").toLowerCase();

  if (normalizedStatus === "queued") {
    return "queued";
  }
  if (normalizedStatus === "in_progress" || normalizedStatus === "requested" || normalizedStatus === "waiting") {
    return "running";
  }
  if (normalizedStatus === "completed") {
    if (normalizedConclusion === "success") {
      return "success";
    }
    if (normalizedConclusion) {
      return "failed";
    }
  }
  return "unknown";
}

function parseJobId(jobId: string): number | null {
  const text = String(jobId || "").trim();
  const match = /^gha_(\d+)$/.exec(text);
  if (!match) {
    return null;
  }
  return Number.parseInt(match[1], 10);
}

async function findRunIdAfterDispatch(workflow: string, startedAtMs: number): Promise<number | null> {
  const cfg = getGithubActionsConfig();

  for (let attempt = 0; attempt < 5; attempt += 1) {
    if (attempt > 0) {
      await sleep(1200);
    }

    const payload = await githubRequest<{ workflow_runs?: GithubWorkflowRun[] }>(
      `/repos/${cfg.owner}/${cfg.repo}/actions/workflows/${encodeURIComponent(workflow)}/runs?event=workflow_dispatch&branch=${encodeURIComponent(cfg.ref)}&per_page=10`
    );

    const runs = Array.isArray(payload.workflow_runs) ? payload.workflow_runs : [];
    const recent = runs.find((run) => {
      const createdAtMs = Date.parse(String(run.created_at || ""));
      return Number.isFinite(createdAtMs) && createdAtMs >= startedAtMs - 90_000;
    });

    if (recent && Number.isFinite(recent.id)) {
      return recent.id;
    }
  }

  return null;
}

async function dispatchWorkflow(workflow: string, inputs: Record<string, string>): Promise<{ job_id: string; github_run_id: number }> {
  const cfg = getGithubActionsConfig();
  if (!cfg.enabled) {
    throw new Error("GitHub Actions integration is disabled or missing required environment variables.");
  }

  const startedAtMs = Date.now();

  await githubRequest(`/repos/${cfg.owner}/${cfg.repo}/actions/workflows/${encodeURIComponent(workflow)}/dispatches`, {
    method: "POST",
    body: JSON.stringify({
      ref: cfg.ref,
      inputs
    })
  });

  const runId = await findRunIdAfterDispatch(workflow, startedAtMs);
  if (!runId) {
    throw new Error(
      `Workflow dispatched but run ID was not observed for '${workflow}'. Verify branch/ref configuration and workflow permissions.`
    );
  }
  return {
    job_id: `gha_${runId}`,
    github_run_id: runId
  };
}

export async function triggerIngestJob(payload: {
  limit: number;
  lookbackDays: number;
  selection: string;
}): Promise<{ job_id: string; provider: "github_actions"; status: "queued"; status_url: string; github_run_id: number }> {
  const cfg = getGithubActionsConfig();
  const dispatch = await dispatchWorkflow(cfg.ingestWorkflow, {
    ingest_limit: String(payload.limit),
    lookback_days: String(payload.lookbackDays),
    selection: payload.selection
  });

  return {
    job_id: dispatch.job_id,
    provider: "github_actions",
    status: "queued",
    status_url: `/api/jobs/${dispatch.job_id}`,
    github_run_id: dispatch.github_run_id
  };
}

export async function triggerEnrichJob(payload: {
  limit: number;
  mode: string;
  sourceKind: string;
  heuristicOnly: boolean;
  model: string;
}): Promise<{ job_id: string; provider: "github_actions"; status: "queued"; status_url: string; github_run_id: number }> {
  const cfg = getGithubActionsConfig();
  const dispatch = await dispatchWorkflow(cfg.enrichWorkflow, {
    enrich_limit: String(payload.limit),
    mode: payload.mode,
    source_kind: payload.sourceKind,
    heuristic_only: payload.heuristicOnly ? "true" : "false",
    model: payload.model
  });

  return {
    job_id: dispatch.job_id,
    provider: "github_actions",
    status: "queued",
    status_url: `/api/jobs/${dispatch.job_id}`,
    github_run_id: dispatch.github_run_id
  };
}

export async function getJobSummary(jobId: string): Promise<GithubRunSummary> {
  const cfg = getGithubActionsConfig();
  if (!cfg.enabled) {
    throw new Error("GitHub Actions integration is disabled or missing required environment variables.");
  }

  const runId = parseJobId(jobId);
  if (!runId) {
    throw new Error("Invalid job ID format. Expected gha_<run_id>.");
  }

  const run = await githubRequest<GithubWorkflowRun>(`/repos/${cfg.owner}/${cfg.repo}/actions/runs/${runId}`);
  const artifactsPayload = await githubRequest<{ artifacts?: GithubArtifact[] }>(
    `/repos/${cfg.owner}/${cfg.repo}/actions/runs/${runId}/artifacts`
  );

  const artifacts = (artifactsPayload.artifacts || []).filter((a) => !a.expired).map((a) => a.name);

  const status = mapRunStatus(run.status, run.conclusion);
  const finishedAt = status === "success" || status === "failed" ? String(run.updated_at || "") : "";

  return {
    job_id: `gha_${run.id}`,
    provider: "github_actions",
    workflow: String(run.name || ""),
    status,
    github_run_id: run.id,
    html_url: String(run.html_url || ""),
    created_at: String(run.created_at || ""),
    started_at: String(run.run_started_at || ""),
    updated_at: String(run.updated_at || ""),
    finished_at: finishedAt,
    conclusion: String(run.conclusion || ""),
    artifacts
  };
}

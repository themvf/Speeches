import { NextResponse } from "next/server";
import { getGithubActionsConfig } from "@/lib/server/env";

export async function POST(req: Request): Promise<NextResponse> {
  const token = process.env.GITHUB_ACTIONS_TOKEN;
  if (!token) {
    return NextResponse.json({ ok: false, error: "GITHUB_ACTIONS_TOKEN not configured" }, { status: 500 });
  }

  let body: { workflow?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ ok: false, error: "Invalid JSON" }, { status: 400 });
  }

  const { workflow } = body;
  if (!workflow) {
    return NextResponse.json({ ok: false, error: "workflow required" }, { status: 400 });
  }

  const { owner, repo } = getGithubActionsConfig();
  const headers = {
    Authorization: `Bearer ${token}`,
    Accept: "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
  };

  const runsController = new AbortController();
  const runsTimer = setTimeout(() => runsController.abort(), 10_000);
  let runsRes: Response;
  try {
    runsRes = await fetch(
      `https://api.github.com/repos/${owner}/${repo}/actions/workflows/${encodeURIComponent(workflow)}/runs?per_page=1`,
      { headers, cache: "no-store", signal: runsController.signal }
    );
  } finally {
    clearTimeout(runsTimer);
  }

  if (!runsRes.ok) {
    const d = await runsRes.json().catch(() => ({}));
    return NextResponse.json(
      { ok: false, error: (d as { message?: string }).message ?? "GitHub error" },
      { status: 502 }
    );
  }

  const runsData = await runsRes.json() as {
    workflow_runs: { id: number; status: string; conclusion: string | null }[];
  };
  const run = runsData.workflow_runs[0] ?? null;

  if (!run) {
    return NextResponse.json({ ok: false, error: "No runs found" }, { status: 404 });
  }
  if (run.status === "completed") {
    return NextResponse.json({ ok: false, error: "Run is already completed" }, { status: 409 });
  }

  const cancelController = new AbortController();
  const cancelTimer = setTimeout(() => cancelController.abort(), 10_000);
  let cancelRes: Response;
  try {
    cancelRes = await fetch(
      `https://api.github.com/repos/${owner}/${repo}/actions/runs/${run.id}/cancel`,
      { method: "POST", headers, signal: cancelController.signal }
    );
  } finally {
    clearTimeout(cancelTimer);
  }

  if (!cancelRes.ok && cancelRes.status !== 202) {
    const d = await cancelRes.json().catch(() => ({}));
    return NextResponse.json(
      { ok: false, error: (d as { message?: string }).message ?? "Cancel failed" },
      { status: 502 }
    );
  }

  return NextResponse.json({ ok: true });
}

import { type NextRequest, NextResponse } from "next/server";
import { getGithubActionsConfig } from "@/lib/server/env";

const ALLOWED_WORKFLOWS = new Set([
  "financial-news-daily.yml",
  "financial-news-ingest.yml",
  "financial-news-enrich.yml",
  "financial-news-enrich-scheduled.yml",
  "sec-speech-sync.yml",
  "knowledge-index-sync.yml",
  "trends-daily.yml",
  "policy-extraction.yml",
  "policy-extraction-scheduled.yml",
  "connector-gap-6hour.yml",
  "securities-market-sources-daily.yml",
  "intelligence-evidence.yml",
  "aml-news-ingest.yml",
  "daily-health-check.yml",
  "python-tests.yml",
]);

export async function GET(req: NextRequest): Promise<NextResponse> {
  const token = process.env.GITHUB_ACTIONS_TOKEN;
  if (!token) {
    return NextResponse.json({ ok: false, error: "GITHUB_ACTIONS_TOKEN not configured" }, { status: 500 });
  }

  const workflow = req.nextUrl.searchParams.get("workflow");
  if (!workflow) {
    return NextResponse.json({ ok: false, error: "workflow param required" }, { status: 400 });
  }
  if (!ALLOWED_WORKFLOWS.has(workflow)) {
    return NextResponse.json({ ok: false, error: "Unknown workflow" }, { status: 400 });
  }

  const { owner, repo } = getGithubActionsConfig();
  const url = `https://api.github.com/repos/${owner}/${repo}/actions/workflows/${encodeURIComponent(workflow)}/runs?per_page=1`;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 10_000);
  let ghRes: Response;
  try {
    ghRes = await fetch(url, {
      headers: {
        Authorization: `Bearer ${token}`,
        Accept: "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
      },
      signal: controller.signal,
      next: { revalidate: 0 },
    });
  } finally {
    clearTimeout(timer);
  }

  if (!ghRes.ok) {
    const data = await ghRes.json().catch(() => ({}));
    const message = (data as { message?: string }).message ?? `GitHub API returned ${ghRes.status}`;
    return NextResponse.json({ ok: false, error: message }, { status: 502 });
  }

  const data = await ghRes.json() as {
    workflow_runs: {
      id: number;
      status: string;
      conclusion: string | null;
      created_at: string;
      html_url: string;
    }[];
  };

  const run = data.workflow_runs[0] ?? null;
  return NextResponse.json({ ok: true, run });
}

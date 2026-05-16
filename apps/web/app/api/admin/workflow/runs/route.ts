import { type NextRequest, NextResponse } from "next/server";

const REPO = "themvf/Speeches";

export async function GET(req: NextRequest): Promise<NextResponse> {
  const token = process.env.GITHUB_ACTIONS_TOKEN;
  if (!token) {
    return NextResponse.json({ ok: false, error: "GITHUB_ACTIONS_TOKEN not configured" }, { status: 500 });
  }

  const workflow = req.nextUrl.searchParams.get("workflow");
  if (!workflow) {
    return NextResponse.json({ ok: false, error: "workflow param required" }, { status: 400 });
  }

  const url = `https://api.github.com/repos/${REPO}/actions/workflows/${encodeURIComponent(workflow)}/runs?per_page=1`;

  const ghRes = await fetch(url, {
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
    },
    next: { revalidate: 0 },
  });

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

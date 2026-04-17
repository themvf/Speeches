import { NextResponse } from "next/server";

const REPO = "themvf/Speeches";

export async function POST(req: Request) {
  const token = process.env.GITHUB_ADMIN_TOKEN;
  if (!token) {
    return NextResponse.json(
      { ok: false, error: "GITHUB_ADMIN_TOKEN is not configured in environment variables" },
      { status: 500 }
    );
  }

  let body: { workflow?: string; inputs?: Record<string, string> };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ ok: false, error: "Invalid JSON body" }, { status: 400 });
  }

  const { workflow, inputs = {} } = body;
  if (!workflow) {
    return NextResponse.json({ ok: false, error: "workflow filename is required" }, { status: 400 });
  }

  // Strip empty strings so GitHub uses workflow defaults
  const filteredInputs = Object.fromEntries(
    Object.entries(inputs).filter(([, v]) => v !== "")
  );

  const url = `https://api.github.com/repos/${REPO}/actions/workflows/${encodeURIComponent(workflow)}/dispatches`;

  const ghRes = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ ref: "main", inputs: filteredInputs }),
  });

  // GitHub returns 204 No Content on success
  if (ghRes.status === 204) {
    return NextResponse.json({ ok: true });
  }

  const data = await ghRes.json().catch(() => ({}));
  const message = (data as { message?: string }).message ?? `GitHub API returned ${ghRes.status}`;
  return NextResponse.json({ ok: false, error: message }, { status: ghRes.status < 500 ? ghRes.status : 502 });
}

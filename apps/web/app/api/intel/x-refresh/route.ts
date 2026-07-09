import { NextRequest, NextResponse } from "next/server";
import { refreshXTimelines } from "@/lib/server/x-timeline-ingestion";
import { parseXTimelineAccounts } from "@/lib/server/x-syndication";
import { checkCronAuth } from "@/lib/server/api-utils";

export const dynamic = "force-dynamic";
export const maxDuration = 55;

function requestedAccounts(req: NextRequest): string[] {
  const params = req.nextUrl.searchParams;
  const raw = [
    ...params.getAll("account"),
    ...params.getAll("accounts"),
    process.env.X_TIMELINE_ACCOUNTS || "",
  ].join(",");
  return parseXTimelineAccounts(raw);
}

function numberParam(req: NextRequest, name: string, fallback: number, max: number): number {
  const value = Number.parseInt(req.nextUrl.searchParams.get(name) || "", 10);
  return Math.max(0, Math.min(max, Number.isFinite(value) ? value : fallback));
}

async function handleRefresh(req: NextRequest): Promise<NextResponse> {
  const auth = checkCronAuth(req);
  if (!auth.ok) {
    return NextResponse.json({ ok: false, error: auth.error }, { status: auth.status });
  }

  try {
    const accounts = requestedAccounts(req);
    const limit = numberParam(req, "limit", Number.parseInt(process.env.X_TIMELINE_LIMIT || "20", 10) || 20, 50);
    const analysisLimit = numberParam(req, "analysisLimit", Number.parseInt(process.env.X_TIMELINE_ANALYSIS_LIMIT || "0", 10) || 0, 50);
    const data = await refreshXTimelines({ accounts, limit, analysisLimit, dueOnly: accounts.length === 0 });
    if (data.feeds.length === 0) {
      return NextResponse.json(
        { ok: false, error: "No X accounts configured. Add accounts in Admin, set X_TIMELINE_ACCOUNTS, or pass ?accounts=CISAgov,SECGov." },
        { status: 400 }
      );
    }
    return NextResponse.json({ ok: true, data });
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "Unknown X refresh error" },
      { status: 500 }
    );
  }
}

export async function GET(req: NextRequest): Promise<NextResponse> {
  return handleRefresh(req);
}

export async function POST(req: NextRequest): Promise<NextResponse> {
  return handleRefresh(req);
}

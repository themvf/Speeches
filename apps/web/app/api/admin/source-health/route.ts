import fs from "node:fs";
import path from "node:path";
import { NextResponse } from "next/server";
import { downloadGcsJson } from "@/lib/server/gcs-loader";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const SOURCE_HEALTH_BLOB = "source_health_log.json";

type SourceHealthPayload = {
  updated_at?: string;
  runs?: unknown[];
  sources?: Record<string, unknown>;
  latest_report?: unknown;
};

function emptyPayload(): SourceHealthPayload {
  return { updated_at: "", runs: [], sources: {}, latest_report: null };
}

function loadLocalPayload(): SourceHealthPayload | null {
  const candidates = [
    path.resolve(process.cwd(), "data", SOURCE_HEALTH_BLOB),
    path.resolve(process.cwd(), "..", "..", "data", SOURCE_HEALTH_BLOB),
  ];
  for (const candidate of candidates) {
    try {
      if (!fs.existsSync(candidate)) continue;
      const parsed = JSON.parse(fs.readFileSync(candidate, "utf-8"));
      if (parsed && typeof parsed === "object") {
        return parsed as SourceHealthPayload;
      }
    } catch {
      // Continue to the next local candidate.
    }
  }
  return null;
}

export async function GET(): Promise<NextResponse> {
  const remote = await downloadGcsJson<SourceHealthPayload>(SOURCE_HEALTH_BLOB);
  const payload = remote ?? loadLocalPayload() ?? emptyPayload();
  const runs = Array.isArray(payload.runs) ? payload.runs : [];
  const sources = payload.sources && typeof payload.sources === "object" ? payload.sources : {};

  return NextResponse.json(
    {
      ok: true,
      data: {
        updated_at: String(payload.updated_at || ""),
        runs: runs.slice(-200).reverse(),
        sources: Object.values(sources),
        latest_report: payload.latest_report ?? null,
      },
    },
    { headers: { "Cache-Control": "no-store" } }
  );
}

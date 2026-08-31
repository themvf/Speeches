import { type NextRequest, NextResponse } from "next/server";
import {
  getScheduledEditorialSettings,
  listEditorialRuns,
  runScheduledEditorial,
  sanitizeScheduledEditorialSettings,
  saveScheduledEditorialSettings,
  scheduledEditorialRuntimeStatus,
} from "@/lib/server/scheduled-editorial";

export const dynamic = "force-dynamic";
export const maxDuration = 300;

export async function GET(): Promise<NextResponse> {
  try {
    const [settings, runs] = await Promise.all([
      getScheduledEditorialSettings(),
      listEditorialRuns(20),
    ]);
    return NextResponse.json({
      ok: true,
      data: { settings, runs, runtime: scheduledEditorialRuntimeStatus() },
    });
  } catch (error) {
    console.error("[admin/briefings/scheduled GET]", error);
    return NextResponse.json({ ok: false, error: error instanceof Error ? error.message : String(error) }, { status: 500 });
  }
}

export async function PUT(req: NextRequest): Promise<NextResponse> {
  try {
    const body = await req.json() as { settings?: unknown };
    const settings = sanitizeScheduledEditorialSettings(body.settings);
    if (!settings.openai_enabled && !settings.deepseek_enabled) {
      return NextResponse.json({ ok: false, error: "Enable at least one provider." }, { status: 400 });
    }
    const saved = await saveScheduledEditorialSettings(settings);
    return NextResponse.json({ ok: true, data: { settings: saved } });
  } catch (error) {
    console.error("[admin/briefings/scheduled PUT]", error);
    return NextResponse.json({ ok: false, error: error instanceof Error ? error.message : String(error) }, { status: 500 });
  }
}

export async function POST(): Promise<NextResponse> {
  try {
    const result = await runScheduledEditorial("manual");
    return NextResponse.json({ ok: true, data: result });
  } catch (error) {
    console.error("[admin/briefings/scheduled POST]", error);
    return NextResponse.json({ ok: false, error: error instanceof Error ? error.message : String(error) }, { status: 500 });
  }
}

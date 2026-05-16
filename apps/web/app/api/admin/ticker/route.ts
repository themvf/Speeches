import { NextResponse } from "next/server";
import { getTickerConfig, setTickerConfig } from "@/lib/ticker-config";
import type { TickerEntry } from "@/lib/ticker-config";

const MAX_TICKERS = 10;

export async function GET() {
  const tickers = await getTickerConfig();
  return NextResponse.json({ ok: true, data: tickers });
}

export async function POST(req: Request) {
  let body: unknown;
  try { body = await req.json(); } catch { return NextResponse.json({ error: "Invalid JSON" }, { status: 400 }); }
  if (!Array.isArray(body)) {
    return NextResponse.json({ error: "Expected an array" }, { status: 400 });
  }
  const validated: TickerEntry[] = [];
  for (const entry of (body as unknown[]).slice(0, MAX_TICKERS)) {
    if (!entry || typeof entry !== "object") continue;
    const { symbol, name } = entry as Record<string, unknown>;
    if (typeof symbol !== "string" || !symbol.trim()) continue;
    if (typeof name !== "string" || !name.trim()) continue;
    validated.push({ symbol: symbol.trim().toUpperCase(), name: name.trim() } as TickerEntry);
  }
  try {
    await setTickerConfig(validated);
  } catch (e) {
    const msg = e instanceof Error ? e.message : "Unknown error";
    return NextResponse.json({ error: msg }, { status: 500 });
  }
  return NextResponse.json({ ok: true });
}

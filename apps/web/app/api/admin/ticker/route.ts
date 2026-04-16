import { NextResponse } from "next/server";
import { getTickerConfig, setTickerConfig } from "@/lib/ticker-config";
import type { TickerEntry } from "@/lib/ticker-config";

const MAX_TICKERS = 10;

export async function GET() {
  const tickers = await getTickerConfig();
  return NextResponse.json(tickers);
}

export async function POST(req: Request) {
  const body = await req.json();
  if (!Array.isArray(body)) {
    return NextResponse.json({ error: "Expected an array" }, { status: 400 });
  }
  const tickers = (body as TickerEntry[]).slice(0, MAX_TICKERS);
  await setTickerConfig(tickers);
  return NextResponse.json({ ok: true });
}

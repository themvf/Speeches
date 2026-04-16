import { NextResponse } from "next/server";
import { Redis } from "@upstash/redis";

const kv = new Redis({
  url: process.env.UPSTASH_REDIS_REST_URL!,
  token: process.env.UPSTASH_REDIS_REST_TOKEN!,
});

const KV_KEY = "ticker-config";
const MAX_TICKERS = 10;

export type TickerEntry = {
  symbol: string;
  name: string;
};

export const DEFAULT_TICKERS: TickerEntry[] = [
  { symbol: "SPY", name: "S&P 500" },
  { symbol: "DIA", name: "Dow Jones" },
  { symbol: "QQQ", name: "Nasdaq 100" },
  { symbol: "EWJ", name: "Nikkei (EWJ)" },
  { symbol: "EWU", name: "FTSE (EWU)" },
];

export async function getTickerConfig(): Promise<TickerEntry[]> {
  try {
    const stored = await kv.get<TickerEntry[]>(KV_KEY);
    return stored ?? DEFAULT_TICKERS;
  } catch {
    return DEFAULT_TICKERS;
  }
}

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
  await kv.set(KV_KEY, tickers);
  return NextResponse.json({ ok: true });
}

import { Redis } from "@upstash/redis";

const KV_KEY = "ticker-config";

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

function getRedis() {
  return new Redis({
    url: process.env.UPSTASH_REDIS_REST_URL!,
    token: process.env.UPSTASH_REDIS_REST_TOKEN!,
  });
}

export async function getTickerConfig(): Promise<TickerEntry[]> {
  try {
    const kv = getRedis();
    const stored = await kv.get<TickerEntry[]>(KV_KEY);
    return stored ?? DEFAULT_TICKERS;
  } catch {
    return DEFAULT_TICKERS;
  }
}

export async function setTickerConfig(tickers: TickerEntry[]): Promise<void> {
  const kv = getRedis();
  await kv.set(KV_KEY, tickers);
}

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
  const url =
    process.env.UPSTASH_REDIS_REST_URL ??
    process.env.KV_REST_API_URL;
  const token =
    process.env.UPSTASH_REDIS_REST_TOKEN ??
    process.env.KV_REST_API_TOKEN;

  if (!url || !token) {
    throw new Error(
      "Redis env vars not set. Expected UPSTASH_REDIS_REST_URL + UPSTASH_REDIS_REST_TOKEN " +
      "(or KV_REST_API_URL + KV_REST_API_TOKEN)."
    );
  }

  return new Redis({ url, token });
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

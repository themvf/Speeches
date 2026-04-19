import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import type { ExchangeInfo, ExchangeRegionGroup, MarketExchangesData, MarketStatus } from "@/lib/server/types";

export const runtime = "nodejs";
export const revalidate = 60;

type FinnhubMarketStatus = {
  exchange: string;
  holiday: string | null;
  isOpen: boolean;
  session: string;
  t: number;
  timezone: string;
};

interface ExchangeMeta {
  name: string;
  timezone: string;
  region: "Americas" | "Europe" | "Asia Pacific";
}

const EXCHANGE_META: Record<string, ExchangeMeta> = {
  US:  { name: "NYSE / NASDAQ",           timezone: "America/New_York",    region: "Americas" },
  TO:  { name: "Toronto Stock Exchange",   timezone: "America/Toronto",     region: "Americas" },
  MX:  { name: "Bolsa Mexicana",           timezone: "America/Mexico_City", region: "Americas" },
  SA:  { name: "B3 (Brasil)",              timezone: "America/Sao_Paulo",   region: "Americas" },
  L:   { name: "London Stock Exchange",    timezone: "Europe/London",       region: "Europe" },
  F:   { name: "Deutsche Börse (Xetra)",   timezone: "Europe/Berlin",       region: "Europe" },
  PA:  { name: "Euronext Paris",           timezone: "Europe/Paris",        region: "Europe" },
  AS:  { name: "Euronext Amsterdam",       timezone: "Europe/Amsterdam",    region: "Europe" },
  MI:  { name: "Borsa Italiana",           timezone: "Europe/Rome",         region: "Europe" },
  SW:  { name: "SIX Swiss Exchange",       timezone: "Europe/Zurich",       region: "Europe" },
  T:   { name: "Tokyo Stock Exchange",     timezone: "Asia/Tokyo",          region: "Asia Pacific" },
  HK:  { name: "Hong Kong Exchange",       timezone: "Asia/Hong_Kong",      region: "Asia Pacific" },
  SS:  { name: "Shanghai Stock Exchange",  timezone: "Asia/Shanghai",       region: "Asia Pacific" },
  SG:  { name: "Singapore Exchange",       timezone: "Asia/Singapore",      region: "Asia Pacific" },
  AU:  { name: "ASX",                      timezone: "Australia/Sydney",    region: "Asia Pacific" },
  KS:  { name: "Korea Stock Exchange",     timezone: "Asia/Seoul",          region: "Asia Pacific" },
};

function deriveStatus(s: FinnhubMarketStatus): MarketStatus {
  if (s.isOpen) return "OPEN";
  const session = (s.session ?? "").toLowerCase();
  if (session.includes("pre")) return "PRE";
  if (session.includes("after") || session.includes("post")) return "AFTER";
  return "CLOSED";
}

async function fetchStatus(code: string, apiKey: string): Promise<FinnhubMarketStatus | null> {
  try {
    const res = await fetch(
      `https://finnhub.io/api/v1/stock/market-status?exchange=${code}&token=${apiKey}`,
      { next: { revalidate: 60 } }
    );
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export async function GET() {
  const requestId = createRequestId();
  const apiKey = process.env.FINNHUB_API_KEY;
  if (!apiKey) return fail("FINNHUB_API_KEY not set", "NO_API_KEY", 500, requestId);

  const codes = Object.keys(EXCHANGE_META);

  const settled = await Promise.allSettled(
    codes.map((code) => fetchStatus(code, apiKey))
  );

  const exchangesByRegion = new Map<"Americas" | "Europe" | "Asia Pacific", ExchangeInfo[]>([
    ["Americas", []],
    ["Europe", []],
    ["Asia Pacific", []],
  ]);

  settled.forEach((result, i) => {
    const code = codes[i];
    const meta = EXCHANGE_META[code];
    const status = result.status === "fulfilled" && result.value
      ? deriveStatus(result.value)
      : "CLOSED";

    exchangesByRegion.get(meta.region)!.push({
      code,
      name: meta.name,
      timezone: meta.timezone,
      status,
    });
  });

  const regions: ExchangeRegionGroup[] = [
    { region: "Americas", exchanges: exchangesByRegion.get("Americas")! },
    { region: "Europe",   exchanges: exchangesByRegion.get("Europe")! },
    { region: "Asia Pacific", exchanges: exchangesByRegion.get("Asia Pacific")! },
  ];

  const data: MarketExchangesData = { regions, generatedAt: new Date().toISOString() };
  return ok(data, requestId);
}

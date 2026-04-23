import { createRequestId, ok } from "@/lib/server/api-utils";
import type { ExchangeInfo, ExchangeRegionGroup, MarketExchangesData, MarketStatus } from "@/lib/server/types";
import { fetchYahooQuote } from "@/lib/server/yahoo";

export const runtime = "nodejs";
export const revalidate = 60;

interface ExchangeMeta {
  name: string;
  timezone: string;
  region: "Americas" | "Europe" | "Asia Pacific";
  // Representative Yahoo Finance symbol whose marketState reflects exchange hours
  probe: string;
}

const EXCHANGE_META: Record<string, ExchangeMeta> = {
  US:  { name: "NYSE / NASDAQ",           timezone: "America/New_York",    region: "Americas",     probe: "^GSPC"     },
  TO:  { name: "Toronto Stock Exchange",   timezone: "America/Toronto",     region: "Americas",     probe: "^GSPTSE"   },
  MX:  { name: "Bolsa Mexicana",           timezone: "America/Mexico_City", region: "Americas",     probe: "^MXX"      },
  SA:  { name: "B3 (Brasil)",              timezone: "America/Sao_Paulo",   region: "Americas",     probe: "^BVSP"     },
  L:   { name: "London Stock Exchange",    timezone: "Europe/London",       region: "Europe",       probe: "^FTSE"     },
  F:   { name: "Deutsche Börse (Xetra)",   timezone: "Europe/Berlin",       region: "Europe",       probe: "^GDAXI"    },
  PA:  { name: "Euronext Paris",           timezone: "Europe/Paris",        region: "Europe",       probe: "^FCHI"     },
  AS:  { name: "Euronext Amsterdam",       timezone: "Europe/Amsterdam",    region: "Europe",       probe: "^AEX"      },
  MI:  { name: "Borsa Italiana",           timezone: "Europe/Rome",         region: "Europe",       probe: "^FTMIB"    },
  SW:  { name: "SIX Swiss Exchange",       timezone: "Europe/Zurich",       region: "Europe",       probe: "^SSMI"     },
  T:   { name: "Tokyo Stock Exchange",     timezone: "Asia/Tokyo",          region: "Asia Pacific", probe: "^N225"     },
  HK:  { name: "Hong Kong Exchange",       timezone: "Asia/Hong_Kong",      region: "Asia Pacific", probe: "^HSI"      },
  SS:  { name: "Shanghai Stock Exchange",  timezone: "Asia/Shanghai",       region: "Asia Pacific", probe: "000001.SS" },
  SG:  { name: "Singapore Exchange",       timezone: "Asia/Singapore",      region: "Asia Pacific", probe: "^STI"      },
  AU:  { name: "ASX",                      timezone: "Australia/Sydney",    region: "Asia Pacific", probe: "^AXJO"     },
  KS:  { name: "Korea Stock Exchange",     timezone: "Asia/Seoul",          region: "Asia Pacific", probe: "^KS11"     },
};

function deriveStatus(yahooStatus: MarketStatus): MarketStatus {
  return yahooStatus;
}

export async function GET() {
  const requestId = createRequestId();
  const codes = Object.keys(EXCHANGE_META);

  const settled = await Promise.allSettled(
    codes.map((code) => fetchYahooQuote(EXCHANGE_META[code].probe, 60))
  );

  const exchangesByRegion = new Map<"Americas" | "Europe" | "Asia Pacific", ExchangeInfo[]>([
    ["Americas", []],
    ["Europe", []],
    ["Asia Pacific", []],
  ]);

  settled.forEach((result, i) => {
    const code = codes[i];
    const meta = EXCHANGE_META[code];
    const status: MarketStatus = result.status === "fulfilled" && result.value
      ? deriveStatus(result.value.status)
      : "CLOSED";

    exchangesByRegion.get(meta.region)!.push({
      code,
      name: meta.name,
      timezone: meta.timezone,
      status,
    });
  });

  const regions: ExchangeRegionGroup[] = [
    { region: "Americas",     exchanges: exchangesByRegion.get("Americas")! },
    { region: "Europe",       exchanges: exchangesByRegion.get("Europe")! },
    { region: "Asia Pacific", exchanges: exchangesByRegion.get("Asia Pacific")! },
  ];

  const data: MarketExchangesData = { regions, generatedAt: new Date().toISOString() };
  return ok(data, requestId);
}

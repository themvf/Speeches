import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import type { MarketBondsData, TreasuryYield } from "@/lib/server/types";

export const runtime = "nodejs";
export const revalidate = 3600;

type FinnhubQuote = { c: number | null; d: number | null; dp: number | null };

async function fetchTreasuryXml(year: number, month: number): Promise<string | null> {
  const mm = String(month).padStart(2, "0");
  const url = `https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value_month=${year}${mm}`;
  try {
    const res = await fetch(url, {
      headers: { "User-Agent": "Mozilla/5.0 (compatible; MarketDashboard/1.0)" },
      next: { revalidate: 3600 },
    });
    if (!res.ok) return null;
    return res.text();
  } catch { return null; }
}

function extractField(block: string, field: string): number | null {
  const m = block.match(new RegExp(`<d:${field}[^>]*>([\\d.]+)<\\/d:${field}>`));
  return m ? parseFloat(m[1]) : null;
}

function parseYields(xml: string): { latest: Record<string, number>; prev: Record<string, number> } | null {
  const blocks = [...xml.matchAll(/<m:properties>([\s\S]*?)<\/m:properties>/g)].map((m) => m[1]);
  if (blocks.length === 0) return null;

  const fields = ["BC_2YEAR", "BC_5YEAR", "BC_10YEAR", "BC_30YEAR"];

  const extract = (block: string) =>
    Object.fromEntries(fields.map((f) => [f, extractField(block, f) ?? 0]));

  const latest = extract(blocks[blocks.length - 1]);
  const prev   = blocks.length >= 2 ? extract(blocks[blocks.length - 2]) : latest;

  return { latest, prev };
}

async function fetchUupQuote(apiKey: string): Promise<FinnhubQuote | null> {
  try {
    const res = await fetch(
      `https://finnhub.io/api/v1/quote?symbol=UUP&token=${apiKey}`,
      { next: { revalidate: 3600 } }
    );
    if (!res.ok) return null;
    const data: FinnhubQuote = await res.json();
    return data.c != null ? data : null;
  } catch { return null; }
}

export async function GET() {
  const requestId = createRequestId();
  const apiKey = process.env.FINNHUB_API_KEY;
  if (!apiKey) return fail("FINNHUB_API_KEY not set", "NO_API_KEY", 500, requestId);

  const now = new Date();

  // Try current month; fall back to previous if empty
  let xml = await fetchTreasuryXml(now.getFullYear(), now.getMonth() + 1);
  if (!xml || !xml.includes("<m:properties>")) {
    const prev = new Date(now.getFullYear(), now.getMonth() - 1, 1);
    xml = await fetchTreasuryXml(prev.getFullYear(), prev.getMonth() + 1);
  }

  const [parsed, uup] = await Promise.all([
    Promise.resolve(xml ? parseYields(xml) : null),
    fetchUupQuote(apiKey),
  ]);

  const YIELD_DEFS: { field: string; label: string }[] = [
    { field: "BC_2YEAR",  label: "2Y Treasury"  },
    { field: "BC_5YEAR",  label: "5Y Treasury"  },
    { field: "BC_10YEAR", label: "10Y Treasury" },
    { field: "BC_30YEAR", label: "30Y Treasury" },
  ];

  const yields: TreasuryYield[] = parsed
    ? YIELD_DEFS.map(({ field, label }) => {
        const rate = parsed.latest[field] ?? 0;
        const prev = parsed.prev[field]   ?? rate;
        const change = rate - prev;
        const pct    = prev !== 0 ? (change / prev) * 100 : 0;
        return { label, rate, change, pct, up: change >= 0 };
      }).filter((y) => y.rate > 0)
    : [];

  const dxy = uup?.c
    ? { price: uup.c, change: uup.d ?? 0, pct: uup.dp ?? 0, up: (uup.d ?? 0) >= 0 }
    : null;

  const data: MarketBondsData = { yields, dxy, generatedAt: new Date().toISOString() };
  return ok(data, requestId);
}

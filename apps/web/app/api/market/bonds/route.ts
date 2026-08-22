import { createRequestId, ok } from "@/lib/server/api-utils";
import type { MarketBondsData, TreasuryYield } from "@/lib/server/types";
import { fetchYahooQuote } from "@/lib/server/yahoo";

export const runtime = "nodejs";
export const revalidate = 3600;

async function fetchTreasuryXml(year: number, month: number): Promise<string | null> {
  const mm = String(month).padStart(2, "0");
  const url = `https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value_month=${year}${mm}`;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8_000);
  try {
    const res = await fetch(url, {
      headers: { "User-Agent": "Mozilla/5.0 (compatible; MarketDashboard/1.0)" },
      next: { revalidate: 3600 },
      signal: controller.signal,
    });
    if (!res.ok) return null;
    return res.text();
  } catch { return null; } finally { clearTimeout(timer); }
}

/**
 * Every tenor the Treasury XML actually carries. The feed also contains
 * `BC_30YEARDISPLAY` (a render-only duplicate of the 30Y) and a bare `BC_1`,
 * neither of which is a tenor -- which is why fields are matched by explicit
 * name rather than by scanning for `BC_*`.
 *
 * `benchmark` marks the four the compact Bonds & Rates list shows; the full
 * set feeds the curve plot.
 */
const YIELD_DEFS: { field: string; tenor: string; label: string; months: number; benchmark: boolean }[] = [
  { field: "BC_1MONTH", tenor: "1M",  label: "1M Treasury",  months: 1,   benchmark: false },
  { field: "BC_2MONTH", tenor: "2M",  label: "2M Treasury",  months: 2,   benchmark: false },
  { field: "BC_3MONTH", tenor: "3M",  label: "3M Treasury",  months: 3,   benchmark: true  },
  { field: "BC_4MONTH", tenor: "4M",  label: "4M Treasury",  months: 4,   benchmark: false },
  { field: "BC_6MONTH", tenor: "6M",  label: "6M Treasury",  months: 6,   benchmark: false },
  { field: "BC_1YEAR",  tenor: "1Y",  label: "1Y Treasury",  months: 12,  benchmark: false },
  { field: "BC_2YEAR",  tenor: "2Y",  label: "2Y Treasury",  months: 24,  benchmark: true  },
  { field: "BC_3YEAR",  tenor: "3Y",  label: "3Y Treasury",  months: 36,  benchmark: false },
  { field: "BC_5YEAR",  tenor: "5Y",  label: "5Y Treasury",  months: 60,  benchmark: false },
  { field: "BC_7YEAR",  tenor: "7Y",  label: "7Y Treasury",  months: 84,  benchmark: false },
  { field: "BC_10YEAR", tenor: "10Y", label: "10Y Treasury", months: 120, benchmark: true  },
  { field: "BC_20YEAR", tenor: "20Y", label: "20Y Treasury", months: 240, benchmark: false },
  { field: "BC_30YEAR", tenor: "30Y", label: "30Y Treasury", months: 360, benchmark: true  },
];

function extractField(block: string, field: string): number | null {
  const m = block.match(new RegExp(`<d:${field}[^>]*>([\\d.]+)<\\/d:${field}>`));
  return m ? parseFloat(m[1]) : null;
}

function parseYields(xml: string): { latest: Record<string, number>; prev: Record<string, number> } | null {
  const blocks = [...xml.matchAll(/<m:properties>([\s\S]*?)<\/m:properties>/g)].map((m) => m[1]);
  if (blocks.length === 0) return null;

  const extract = (block: string) =>
    Object.fromEntries(YIELD_DEFS.map(({ field }) => [field, extractField(block, field) ?? 0]));

  const latest = extract(blocks[blocks.length - 1]);
  const prev   = blocks.length >= 2 ? extract(blocks[blocks.length - 2]) : latest;

  return { latest, prev };
}

export async function GET() {
  const requestId = createRequestId();
  const now = new Date();

  // Try current month; fall back to previous if empty
  let xml = await fetchTreasuryXml(now.getFullYear(), now.getMonth() + 1);
  if (!xml || !xml.includes("<m:properties>")) {
    const prev = new Date(now.getFullYear(), now.getMonth() - 1, 1);
    xml = await fetchTreasuryXml(prev.getFullYear(), prev.getMonth() + 1);
  }

  const [parsed, uup] = await Promise.all([
    Promise.resolve(xml ? parseYields(xml) : null),
    fetchYahooQuote("UUP", 3600),
  ]);

  // A tenor missing from an older month's feed (the 4M was only introduced in
  // 2022, the 20Y reintroduced in 2020) extracts as 0 and is filtered out
  // rather than plotted as a zero-yield point.
  const yields: TreasuryYield[] = parsed
    ? YIELD_DEFS.map(({ field, tenor, label, months, benchmark }) => {
        const rate = parsed.latest[field] ?? 0;
        const prev = parsed.prev[field]   ?? rate;
        const change = rate - prev;
        const pct    = prev !== 0 ? (change / prev) * 100 : 0;
        return { label, tenor, months, benchmark, rate, change, pct, up: change >= 0 };
      }).filter((y) => y.rate > 0)
    : [];

  const dxy = uup
    ? { price: uup.price, change: uup.change, pct: uup.pct, up: uup.change >= 0 }
    : null;

  const data: MarketBondsData = { yields, dxy, generatedAt: new Date().toISOString() };
  return ok(data, requestId);
}

import { type NextRequest } from "next/server";
import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import type { CompanyKpi, MarketFundamentalsData } from "@/lib/server/types";
import { extractQuarterlySeries, FUNDAMENTALS_KPIS, type CompanyFactsJson } from "@/lib/server/companyfacts";
import industryConfig from "@/lib/server/industry-config.json" with { type: "json" };

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// SEC-54: on-demand fundamentals for any ticker in the 522-company industry
// universe, straight from SEC companyfacts (one request, cached an hour).
// The ticker->CIK map is the committed industry config, so lookups never
// need a live mapping service.

const UA = { "User-Agent": "PolicyResearchHub fundamentals (joshbandes@gmail.com)" };

function lookupCompany(ticker: string): { cik: string; name: string } | null {
  for (const industry of industryConfig.industries) {
    for (const entry of industry.tickers) {
      if (entry.ticker === ticker) return { cik: entry.cik, name: entry.name };
    }
  }
  return null;
}

export async function GET(req: NextRequest) {
  const requestId = createRequestId();
  const ticker = (req.nextUrl.searchParams.get("ticker") ?? "").trim().toUpperCase();
  if (!ticker) return fail("ticker is required.", "BAD_TICKER", 400, requestId);

  const company = lookupCompany(ticker);
  if (!company) {
    return fail(
      `${ticker} isn't in the tracked ${industryConfig.tickerCount}-company universe. Add it to build_industry_config.py's UNIVERSE and rerun to cover it.`,
      "UNKNOWN_TICKER",
      404,
      requestId
    );
  }

  const url = `https://data.sec.gov/api/xbrl/companyfacts/CIK${company.cik.padStart(10, "0")}.json`;
  let facts: CompanyFactsJson;
  try {
    const resp = await fetch(url, { headers: UA, next: { revalidate: 3600 } });
    if (!resp.ok) return fail(`SEC companyfacts returned HTTP ${resp.status}.`, "FETCH_FAILED", 502, requestId);
    facts = (await resp.json()) as CompanyFactsJson;
  } catch {
    return fail("SEC companyfacts fetch failed.", "FETCH_FAILED", 502, requestId);
  }

  const kpis: CompanyKpi[] = FUNDAMENTALS_KPIS.map((spec) => ({
    kpiKey: spec.kpiKey,
    label: spec.label,
    unit: spec.unit,
    series: extractQuarterlySeries(facts, spec.concepts, spec.factUnit).map((p) => ({
      periodEnd: p.end,
      value: p.value,
      derived: false,
    })),
  })).filter((kpi) => kpi.series.length > 0);

  if (kpis.length === 0) {
    return fail(`No quarterly GAAP facts found for ${ticker}.`, "NO_FACTS", 404, requestId);
  }

  const data: MarketFundamentalsData = {
    company: { ticker, name: facts.entityName || company.name, kpis },
    source: "SEC companyfacts XBRL (on demand, cached 1h)",
    note: "Calendar-quarter frames as filed; fiscal Q4 values that only appear inside a 10-K may be absent.",
    generatedAt: new Date().toISOString(),
  };
  return ok(data, requestId);
}

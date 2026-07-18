// SEC-54: quarterly fundamentals from SEC's companyfacts API - one free
// JSON request covers a company's full XBRL history. Pure extraction here
// (unit-testable); the route does the fetch.

export type CompanyFactsJson = {
  entityName?: string;
  facts?: Record<string, Record<string, { units?: Record<string, FactEntry[]> }>>;
};

type FactEntry = {
  start?: string;
  end?: string;
  val?: number;
  frame?: string;
  form?: string;
  fp?: string;
};

// A quarterly flow fact carries a CY####Q# frame; instant facts add "I".
// Frames are SEC's own dedup of the cumulative-vs-quarterly trap (the same
// disambiguation the KPI pilot validated), so filtering on them yields one
// canonical value per calendar quarter.
const QUARTER_FRAME_RE = /^CY\d{4}Q\d$/;

export function extractQuarterlySeries(
  facts: CompanyFactsJson,
  concepts: string[],
  unit: string,
  quarters = 8
): { end: string; value: number }[] {
  const gaap = facts.facts?.["us-gaap"] ?? {};
  // Extract every concept in the chain and keep the FRESHEST series (latest
  // period end; ties break by chain order). "First non-empty" is a trap:
  // frames migrate between related concepts over time (CAT's quarterly
  // frames sat on NetIncomeLoss until ~2011, then moved to ProfitLoss), so
  // an early chain entry can win with decade-old data.
  let best: { end: string; value: number }[] = [];
  for (const concept of concepts) {
    const entries = gaap[concept]?.units?.[unit] ?? [];
    const byEnd = new Map<string, number>();
    for (const entry of entries) {
      if (!entry.end || entry.val == null) continue;
      if (!entry.frame || !QUARTER_FRAME_RE.test(entry.frame)) continue;
      byEnd.set(entry.end, entry.val);
    }
    if (byEnd.size === 0) continue;
    const series = [...byEnd.entries()]
      .sort(([a], [b]) => (a < b ? -1 : 1))
      .slice(-quarters)
      .map(([end, value]) => ({ end, value }));
    const lastEnd = series[series.length - 1]!.end;
    const bestEnd = best.length ? best[best.length - 1]!.end : "";
    if (lastEnd > bestEnd) best = series;
  }
  return best;
}

export const FUNDAMENTALS_KPIS: { kpiKey: string; label: string; unit: "usd" | "usd_per_share"; concepts: string[]; factUnit: string }[] = [
  { kpiKey: "eps_diluted", label: "Diluted EPS", unit: "usd_per_share", concepts: ["EarningsPerShareDiluted"], factUnit: "USD/shares" },
  {
    kpiKey: "revenue", label: "Revenue", unit: "usd",
    concepts: ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "RevenuesNetOfInterestExpense"],
    factUnit: "USD",
  },
  { kpiKey: "net_income", label: "Net income", unit: "usd", concepts: ["NetIncomeLoss", "ProfitLoss"], factUnit: "USD" },
];

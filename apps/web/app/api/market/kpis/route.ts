import { createRequestId, ok } from "@/lib/server/api-utils";
import type { CompanyKpi, CompanyKpis, MarketKpiData, TierCKpiValue } from "@/lib/server/types";
import kpiPilotData from "@/lib/server/kpi-pilot-data.json" with { type: "json" };
import kpiTierCData from "@/lib/server/kpi-tier-c-data.json" with { type: "json" };

export const dynamic = "force-static";

// SEC-13: approved-only Tier C values per ticker. pending_review/rejected
// entries deliberately never leave the server - review happens in the
// committed kpi-tier-c-data.json itself (git-as-store, like SEC-34).
type TierCEntry = {
  label: string; unit: string; value: number; period: string;
  evidence: string; status: string;
};
function approvedTierC(ticker: string): TierCKpiValue[] {
  const company = (kpiTierCData.companies as Record<string, { sourceUrl?: string; kpis?: Record<string, TierCEntry> }>)[ticker];
  if (!company?.kpis) return [];
  return Object.entries(company.kpis)
    .filter(([, kpi]) => kpi.status === "approved")
    .map(([kpiKey, kpi]) => ({
      kpiKey,
      label: kpi.label,
      unit: kpi.unit as CompanyKpi["unit"],
      value: kpi.value,
      period: kpi.period,
      evidence: kpi.evidence,
      sourceUrl: company.sourceUrl ?? "",
    }));
}

// Static snapshot (SEC-8: all 22 public companies, built by
// build_kpi_snapshot.py from SEC XBRL) - statically imported so this route
// has zero network/DB dependency until SEC-9/SEC-10's live daily pipeline
// lands. isLive: false is load-bearing for the UI's "static snapshot"
// labeling; don't flip it without also wiring a live data source.
export async function GET() {
  const requestId = createRequestId();
  const companies: CompanyKpis[] = Object.entries(kpiPilotData.companies).map(([ticker, company]) => {
    const operational = approvedTierC(ticker);
    return {
      ticker,
      name: company.name,
      kpis: Object.entries(company.kpis).map(([kpiKey, kpi]): CompanyKpi => ({
        kpiKey,
        label: kpi.label,
        unit: kpi.unit as CompanyKpi["unit"],
        series: kpi.series.map((point) => ({
          periodEnd: point.end,
          value: point.value,
          derived: point.derived,
        })),
      })),
      ...(operational.length > 0 ? { operational } : {}),
    };
  });

  const data: MarketKpiData = {
    isLive: false,
    snapshotDate: kpiPilotData.generatedAt,
    source: kpiPilotData.source,
    companies,
    warning:
      "Auto-refreshed from SEC EDGAR (SEC-34): new 10-Q/10-K filings are detected hourly on weekdays and redeploy this data within ~1 hour of acceptance.",
  };
  return ok(data, requestId);
}

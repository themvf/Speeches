import { createRequestId, ok } from "@/lib/server/api-utils";
import type { CompanyKpi, CompanyKpis, MarketKpiData } from "@/lib/server/types";
import kpiPilotData from "@/lib/server/kpi-pilot-data.json" with { type: "json" };

export const dynamic = "force-static";

// Static snapshot (SEC-8: all 22 public companies, built by
// build_kpi_snapshot.py from SEC XBRL) - statically imported so this route
// has zero network/DB dependency until SEC-9/SEC-10's live daily pipeline
// lands. isLive: false is load-bearing for the UI's "static snapshot"
// labeling; don't flip it without also wiring a live data source.
export async function GET() {
  const requestId = createRequestId();
  const companies: CompanyKpis[] = Object.entries(kpiPilotData.companies).map(([ticker, company]) => ({
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
  }));

  const data: MarketKpiData = {
    isLive: false,
    snapshotDate: kpiPilotData.generatedAt,
    source: kpiPilotData.source,
    companies,
    warning: "Static snapshot from SEC XBRL filings (SEC-8) - not a live feed. Live daily ingestion is scoped in JIRA SEC-9/SEC-10.",
  };
  return ok(data, requestId);
}

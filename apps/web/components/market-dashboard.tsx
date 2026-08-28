"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  MarketAttentionData,
  MarketBondsData,
  MarketCommoditiesData,
  MarketCryptoData,
  MarketEarningsWeekData,
  MarketExchangesData,
  MarketIndustriesData,
  MarketKpiData,
  MarketMacroCalendarData,
  MarketMacroData,
  MarketMacroPredictionsData,
  MarketRatesCreditData,
  MarketMoversData,
  MarketOverviewData,
  MarketPredictionsData,
  MarketSectorsData,
} from "@/lib/server/types";
import type { RateTransmissionData } from "@/lib/rate-transmission";
import { OverviewTab } from "./market/overview-tab";
import { SectorsTab } from "./market/sectors-tab";
import { MoversTab } from "./market/movers-tab";
import { AttentionTab } from "./market/attention-tab";
import { CryptoTab } from "./market/crypto-tab";
import { ExchangesTab } from "./market/exchanges-tab";
import { CboeTab } from "./market/cboe-tab";
import { PredictionMarketsTab } from "./market/prediction-markets-tab";
import { MacroTab } from "./market/macro-tab";
import { IndustriesTab } from "./market/industries-tab";
import { EarningsTab } from "./market/earnings-tab";
import { TickerSearch } from "./market/ticker-search";
import { MarketGroupsTab } from "./market/market-groups-tab";

type TabId = "overview" | "macro" | "groups" | "sectors" | "industries" | "movers" | "attention" | "cboe" | "earnings" | "predictions" | "crypto" | "exchanges";

const TABS: { id: TabId; label: string }[] = [
  { id: "overview",    label: "Overview" },
  { id: "macro",       label: "Macro" },
  { id: "groups",      label: "Market Groups" },
  { id: "sectors",     label: "Sectors" },
  { id: "industries",  label: "Industries" },
  { id: "movers",      label: "Movers" },
  { id: "attention",   label: "Reddit" },
  { id: "cboe",        label: "CBOE" },
  { id: "earnings",    label: "Earnings" },
  { id: "predictions", label: "Prediction Markets" },
  { id: "crypto",      label: "Crypto" },
  { id: "exchanges",   label: "Exchanges" },
];

interface TabState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

function useTabData<T>(
  thisTab: TabId | readonly TabId[],
  activeTab: TabId,
  endpoint: string,
  pollMs: number,
): TabState<T> {
  const active = typeof thisTab === "string" ? activeTab === thisTab : thisTab.includes(activeTab);
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const loadedRef = useRef(false);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    fetch(endpoint)
      .then((r) => r.json())
      .then((env) => {
        if (env.ok && env.data) setData(env.data);
        else setError(env.error ?? "Failed to load");
      })
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, [endpoint]);

  useEffect(() => {
    if (!active) return;
    if (!loadedRef.current) { loadedRef.current = true; load(); }
    const id = setInterval(load, pollMs);
    return () => clearInterval(id);
  }, [active, load, pollMs]);

  return { data, loading, error };
}

export function MarketDashboard() {
  const [tab, setTab] = useState<TabId>("overview");

  // Global ticker/company search (persistent above the tab bar) jumps into
  // the Industries or CBOE tabs for a selected company. Nonce bumps on
  // every click so re-selecting the same target re-triggers the effect
  // even if it's unchanged from last time.
  const [industryExpandRequest, setIndustryExpandRequest] = useState<{ label: string; nonce: number } | null>(null);
  const [fundamentalsLookupRequest, setFundamentalsLookupRequest] = useState<{ ticker: string; nonce: number } | null>(null);

  // Overview sub-feeds (all keyed to "overview" tab)
  const overview    = useTabData<MarketOverviewData>   ("overview", tab, "/api/market/overview",    60_000);
  const commodities = useTabData<MarketCommoditiesData>("overview", tab, "/api/market/commodities", 120_000);
  // Bonds feeds two surfaces: the compact benchmark box on Overview and the
  // full curve on Macro, beside credit and the real yield.
  const bonds       = useTabData<MarketBondsData>      (["overview", "macro"], tab, "/api/market/bonds", 3_600_000);
  const macro       = useTabData<MarketMacroData>      ("macro",    tab, "/api/market/macro",       900_000);
  const ratesCredit = useTabData<MarketRatesCreditData>("macro",    tab, "/api/market/rates-credit", 900_000);
  const rateTransmission = useTabData<RateTransmissionData>("macro", tab, "/api/market/rate-transmission", 3_600_000);
  const macroPredictions = useTabData<MarketMacroPredictionsData>(["macro", "predictions"], tab, "/api/market/macro-contracts", 300_000);
  // Release schedules change on the order of months; the long poll matches
  // the route's 6-hour revalidate rather than re-asking FRED every minute.
  const macroCalendar = useTabData<MarketMacroCalendarData>("macro", tab, "/api/market/macro-calendar", 21_600_000);

  const sectors   = useTabData<MarketSectorsData>  (["groups", "sectors"], tab, "/api/market/sectors", 300_000);
  // Industry list is config + one cheap DB join; peer quotes load lazily per
  // expanded industry inside the tab, so a long poll here is fine.
  const industries = useTabData<MarketIndustriesData>(["groups", "industries"], tab, "/api/market/industries", 600_000);
  const movers    = useTabData<MarketMoversData>   ("movers",    tab, "/api/market/movers",    120_000);
  const attention = useTabData<MarketAttentionData>("attention", tab, "/api/market/attention", 300_000);
  // Static snapshot (SEC-17/19) - long poll interval since the route never
  // changes until the SEC-9/SEC-10 live pipeline replaces it.
  const cboe      = useTabData<MarketKpiData>      ("cboe",      tab, "/api/market/kpis",      3_600_000);
  // Static snapshot (SEC-25/28) - long poll interval like CBOE; the route
  // never changes until the SEC-26/27 live pipeline replaces it.
  const predictions = useTabData<MarketPredictionsData>("predictions", tab, "/api/market/predictions", 3_600_000);
  const earnings  = useTabData<MarketEarningsWeekData>("earnings",  tab, "/api/market/earnings-week", 600_000);
  const crypto    = useTabData<MarketCryptoData>   ("crypto",    tab, "/api/market/crypto",    120_000);
  const exchanges = useTabData<MarketExchangesData>("exchanges", tab, "/api/market/exchanges",  60_000);

  return (
    <div className="space-y-6">
      <TickerSearch
        onViewIndustry={(label) => {
          setIndustryExpandRequest({ label, nonce: Date.now() });
          setTab("industries");
        }}
        onViewFundamentals={(ticker) => {
          setFundamentalsLookupRequest({ ticker, nonce: Date.now() });
          setTab("cboe");
        }}
      />

      {/* Tab bar */}
      <div className="flex items-center gap-1 overflow-x-auto pb-0.5">
        {TABS.map(({ id, label }) => (
          <button
            key={id}
            type="button"
            onClick={() => setTab(id)}
            className={`whitespace-nowrap rounded-xl border px-4 py-2 text-sm font-medium transition-colors ${
              tab === id
                ? "border-[color:var(--line-strong)] bg-[color:rgba(15,32,50,0.92)] text-[color:var(--ink)] shadow-[inset_0_1px_0_rgba(79,213,255,0.15)]"
                : "border-transparent text-[color:var(--ink-faint)] hover:border-[color:var(--line)] hover:bg-[color:rgba(79,213,255,0.1)] hover:text-[color:var(--ink)]"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {tab === "overview"  && <OverviewTab  {...overview} commodities={commodities} bonds={bonds} />}
      {tab === "macro"     && <MacroTab     {...macro} predictions={macroPredictions} calendar={macroCalendar} bonds={bonds} ratesCredit={ratesCredit} rateTransmission={rateTransmission} />}
      {tab === "groups"    && <MarketGroupsTab sectors={sectors} industries={industries} />}
      {tab === "sectors"   && <SectorsTab   {...sectors} />}
      {tab === "industries" && <IndustriesTab {...industries} expandRequest={industryExpandRequest} />}
      {tab === "movers"    && <MoversTab    {...movers} />}
      {tab === "attention" && <AttentionTab {...attention} />}
      {tab === "cboe"      && <CboeTab      {...cboe} lookupRequest={fundamentalsLookupRequest} />}
      {tab === "earnings"  && <EarningsTab  {...earnings} />}
      {tab === "predictions" && <PredictionMarketsTab {...predictions} macro={macroPredictions} />}
      {tab === "crypto"    && <CryptoTab    {...crypto} />}
      {tab === "exchanges" && <ExchangesTab {...exchanges} />}
    </div>
  );
}

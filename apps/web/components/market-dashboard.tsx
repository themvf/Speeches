"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  MarketBondsData,
  MarketCommoditiesData,
  MarketCryptoData,
  MarketExchangesData,
  MarketMoversData,
  MarketOverviewData,
  MarketSectorsData,
} from "@/lib/server/types";
import { OverviewTab } from "./market/overview-tab";
import { SectorsTab } from "./market/sectors-tab";
import { MoversTab } from "./market/movers-tab";
import { CryptoTab } from "./market/crypto-tab";
import { ExchangesTab } from "./market/exchanges-tab";

type TabId = "overview" | "sectors" | "movers" | "crypto" | "exchanges";

const TABS: { id: TabId; label: string }[] = [
  { id: "overview",  label: "Overview" },
  { id: "sectors",   label: "Sectors" },
  { id: "movers",    label: "Movers" },
  { id: "crypto",    label: "Crypto" },
  { id: "exchanges", label: "Exchanges" },
];

interface TabState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

function useTabData<T>(
  thisTab: TabId,
  activeTab: TabId,
  endpoint: string,
  pollMs: number,
): TabState<T> {
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
    if (activeTab !== thisTab) return;
    if (!loadedRef.current) { loadedRef.current = true; load(); }
    const id = setInterval(load, pollMs);
    return () => clearInterval(id);
  }, [activeTab, thisTab, load, pollMs]);

  return { data, loading, error };
}

export function MarketDashboard() {
  const [tab, setTab] = useState<TabId>("overview");

  // Overview sub-feeds (all keyed to "overview" tab)
  const overview    = useTabData<MarketOverviewData>   ("overview", tab, "/api/market/overview",    60_000);
  const commodities = useTabData<MarketCommoditiesData>("overview", tab, "/api/market/commodities", 120_000);
  const bonds       = useTabData<MarketBondsData>      ("overview", tab, "/api/market/bonds",       3_600_000);

  const sectors   = useTabData<MarketSectorsData>  ("sectors",   tab, "/api/market/sectors",   300_000);
  const movers    = useTabData<MarketMoversData>   ("movers",    tab, "/api/market/movers",    120_000);
  const crypto    = useTabData<MarketCryptoData>   ("crypto",    tab, "/api/market/crypto",    120_000);
  const exchanges = useTabData<MarketExchangesData>("exchanges", tab, "/api/market/exchanges",  60_000);

  return (
    <div className="space-y-6">
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
      {tab === "sectors"   && <SectorsTab   {...sectors} />}
      {tab === "movers"    && <MoversTab    {...movers} />}
      {tab === "crypto"    && <CryptoTab    {...crypto} />}
      {tab === "exchanges" && <ExchangesTab {...exchanges} />}
    </div>
  );
}

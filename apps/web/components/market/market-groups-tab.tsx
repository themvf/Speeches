"use client";

import { useEffect, useMemo, useState } from "react";
import type {
  IndustryPeerRow,
  IndustrySummary,
  MarketIndustriesData,
  MarketSectorsData,
  SectorData,
  SectorStock,
} from "@/lib/server/types";

type RangeId = "d1" | "w1" | "m1" | "m3" | "ytd";
type ViewId = "moving" | "industries";
type PeerPreset = "essentials" | "financials" | "signals" | "all";
type PeerSort = "marketCap" | "move" | "revenue" | "profit" | "mentions" | "company";

interface DataState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

interface Props {
  sectors: DataState<MarketSectorsData>;
  industries: DataState<MarketIndustriesData>;
}

const RANGES: { id: RangeId; label: string }[] = [
  { id: "d1", label: "1D" },
  { id: "w1", label: "1W" },
  { id: "m1", label: "1M" },
  { id: "m3", label: "3M" },
  { id: "ytd", label: "YTD" },
];

const SECTOR_ETF: Record<string, string> = {
  Technology: "XLK",
  "Communication Services": "XLC",
  "Consumer Discretionary": "XLY",
  "Consumer Staples": "XLP",
  Energy: "XLE",
  Financials: "XLF",
  Healthcare: "XLV",
  Industrials: "XLI",
  Materials: "XLB",
  "Real Estate": "XLRE",
  Utilities: "XLU",
};

function fmtPct(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
}

function money(value: number | null): string {
  if (value == null) return "—";
  const absolute = Math.abs(value);
  const sign = value < 0 ? "−" : "";
  if (absolute >= 1e12) return `${sign}$${(absolute / 1e12).toFixed(2)}T`;
  if (absolute >= 1e9) return `${sign}$${(absolute / 1e9).toFixed(1)}B`;
  if (absolute >= 1e6) return `${sign}$${(absolute / 1e6).toFixed(0)}M`;
  return `${sign}$${absolute.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

function dateTimeLabel(value: string): string {
  return new Date(value).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZoneName: "short",
  });
}

function reportDateLabel(value: string | null): string {
  if (!value) return "—";
  return new Date(`${value}T00:00:00Z`).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  });
}

function moveColor(value: number): string {
  return value >= 0 ? "#41d39d" : "#f87171";
}

function strongest(stocks: SectorStock[]): SectorStock | null {
  return stocks.length > 0 ? [...stocks].sort((a, b) => b.pct - a.pct)[0] : null;
}

function weakest(stocks: SectorStock[]): SectorStock | null {
  return stocks.length > 0 ? [...stocks].sort((a, b) => a.pct - b.pct)[0] : null;
}

function SectorDetail({ sector, range, highlightedTicker }: { sector: SectorData; range: RangeId; highlightedTicker: string | null }) {
  const companies = [...sector.stocks].sort((a, b) => b.pct - a.pct);
  const leader = strongest(companies);
  const laggard = weakest(companies);
  const etf = SECTOR_ETF[sector.name];

  return (
    <section aria-labelledby="selected-sector-title" className="rounded-2xl border border-[color:var(--line-strong)] bg-[color:rgba(9,21,34,0.72)] p-4 sm:p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.1em] text-[color:var(--accent)]">Selected sector</p>
          <h3 id="selected-sector-title" className="mt-1 text-lg font-semibold text-[color:var(--ink)]">{sector.name}</h3>
          <p className="mt-1 text-xs text-[color:var(--ink-faint)]">
            {etf ? `${etf} ETF return` : "Sector proxy return"} · representative companies show today&apos;s move
          </p>
        </div>
        <div className="text-right">
          <p className="text-2xl font-semibold tabular-nums" style={{ color: moveColor(sector.pcts[range]) }}>
            {sector.pcts[range] >= 0 ? "▲ " : "▼ "}{fmtPct(sector.pcts[range])}
          </p>
          <p className="text-[10px] text-[color:var(--ink-faint)]">{RANGES.find((item) => item.id === range)?.label}</p>
        </div>
      </div>

      <div className="mt-4 grid gap-2 sm:grid-cols-2">
        <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(65,211,157,0.05)] p-3">
          <p className="text-[10px] uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Leading representative</p>
          <p className="mt-1 text-sm font-semibold text-[color:var(--ink)]">{leader ? `${leader.symbol} · ${leader.name}` : "Unavailable"}</p>
          {leader && <p className="mt-1 text-xs font-semibold" style={{ color: moveColor(leader.pct) }}>▲ {fmtPct(leader.pct)} today</p>}
        </div>
        <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(248,113,113,0.05)] p-3">
          <p className="text-[10px] uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">Lagging representative</p>
          <p className="mt-1 text-sm font-semibold text-[color:var(--ink)]">{laggard ? `${laggard.symbol} · ${laggard.name}` : "Unavailable"}</p>
          {laggard && <p className="mt-1 text-xs font-semibold" style={{ color: moveColor(laggard.pct) }}>▼ {fmtPct(laggard.pct)} today</p>}
        </div>
      </div>

      <div className="mt-4 overflow-hidden rounded-xl border border-[color:var(--line)]">
        <div className="grid grid-cols-[74px_1fr_auto] gap-3 border-b border-[color:var(--line)] px-3 py-2 text-[10px] uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">
          <span>Ticker</span><span>Company</span><span>Today</span>
        </div>
        {companies.map((company) => (
          <div
            key={company.symbol}
            className={`grid grid-cols-[74px_1fr_auto] gap-3 border-b border-[color:var(--line)] px-3 py-2.5 text-xs last:border-0 ${
              highlightedTicker === company.symbol ? "bg-[color:rgba(79,213,255,0.1)]" : ""
            }`}
          >
            <span className="font-bold text-[color:var(--accent)]">{company.symbol}</span>
            <span className="truncate text-[color:var(--ink-soft)]">{company.name}</span>
            <span className="font-semibold tabular-nums" style={{ color: moveColor(company.pct) }}>
              {company.pct >= 0 ? "▲ " : "▼ "}{fmtPct(company.pct)}
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}

function useIndustryPeers(industry: IndustrySummary | null) {
  const [data, setData] = useState<MarketIndustriesData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!industry) {
      setData(null);
      setLoading(false);
      setError(null);
      return;
    }
    const controller = new AbortController();
    setData(null);
    setLoading(true);
    setError(null);
    fetch(`/api/market/industries?industry=${encodeURIComponent(industry.label)}`, { signal: controller.signal })
      .then((response) => response.json().then((body) => ({ response, body })))
      .then(({ response, body }) => {
        if (!response.ok || !body.ok || !body.data) throw new Error(body.error ?? "Peer data is unavailable.");
        setData(body.data as MarketIndustriesData);
      })
      .catch((reason) => {
        if (reason instanceof Error && reason.name !== "AbortError") setError(reason.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
    });
    return () => controller.abort();
  }, [industry]);

  return { rows: data?.peers?.rows ?? [], loading, error };
}

function IndustrySnapshot({
  industry,
  rows,
  loading,
  error,
}: {
  industry: IndustrySummary;
  rows: IndustryPeerRow[];
  loading: boolean;
  error: string | null;
}) {
  const moves = rows.flatMap((row) => row.pricePct == null ? [] : [row.pricePct]).sort((a, b) => a - b);
  const middle = Math.floor(moves.length / 2);
  const median = moves.length === 0 ? null : moves.length % 2 === 0 ? (moves[middle - 1] + moves[middle]) / 2 : moves[middle];
  const ranked = rows.filter((row) => row.pricePct != null).sort((a, b) => (b.pricePct ?? 0) - (a.pricePct ?? 0));
  const leader = ranked[0] ?? null;
  const laggard = ranked.at(-1) ?? null;
  const advancers = moves.filter((move) => move > 0).length;
  const decliners = moves.filter((move) => move < 0).length;

  return (
    <section aria-labelledby="selected-industry-title" className="self-start rounded-2xl border border-[color:var(--line-strong)] bg-[color:rgba(9,21,34,0.72)] p-4 sm:p-5">
      <p className="text-[11px] font-semibold uppercase tracking-[0.1em] text-[color:var(--accent)]">Selected industry</p>
      <div className="mt-1 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 id="selected-industry-title" className="text-lg font-semibold text-[color:var(--ink)]">{industry.label}</h3>
          <p className="mt-1 text-xs text-[color:var(--ink-faint)]">SEC SIC {industry.sic} · {industry.tickers.length} companies</p>
        </div>
        <div className="flex gap-2 text-[10px]">
          <span className="rounded-full border border-[color:var(--line)] px-2 py-1 text-[color:var(--ink-soft)]">{industry.attentionTotal || 0} latest mentions</span>
          {industry.reportingSoon.length > 0 && <span className="rounded-full border border-[color:rgba(79,213,255,0.3)] px-2 py-1 text-[color:var(--accent)]">{industry.reportingSoon.length} reporting</span>}
        </div>
      </div>

      {loading && <div className="mt-5 space-y-2" aria-label="Loading industry peers">{[0, 1, 2, 3].map((item) => <div key={item} className="h-10 animate-pulse rounded-lg bg-[color:rgba(79,213,255,0.06)]" />)}</div>}
      {error && <p className="mt-5 rounded-xl border border-red-500/20 bg-red-500/5 p-3 text-xs text-red-300">{error}</p>}
      {!loading && !error && rows.length === 0 && <p className="mt-5 text-xs text-[color:var(--ink-faint)]">No peer data is available.</p>}

      {rows.length > 0 && (
        <div className="mt-4 grid gap-2 sm:grid-cols-2">
          <SnapshotMetric label="Median company move" value={median == null ? "—" : `${median >= 0 ? "▲ " : "▼ "}${fmtPct(median)}`} color={median == null ? undefined : moveColor(median)} detail={`${moves.length} priced companies`} />
          <SnapshotMetric label="Market breadth" value={`${advancers} up · ${decliners} down`} detail={`${moves.length - advancers - decliners} unchanged`} />
          <SnapshotMetric label="Leading company" value={leader ? `${leader.ticker} ${fmtPct(leader.pricePct!)}` : "—"} color={leader?.pricePct == null ? undefined : moveColor(leader.pricePct)} detail={leader?.name ?? "No quote available"} />
          <SnapshotMetric label="Lagging company" value={laggard ? `${laggard.ticker} ${fmtPct(laggard.pricePct!)}` : "—"} color={laggard?.pricePct == null ? undefined : moveColor(laggard.pricePct)} detail={laggard?.name ?? "No quote available"} />
        </div>
      )}
    </section>
  );
}

function SnapshotMetric({ label, value, detail, color }: { label: string; value: string; detail: string; color?: string }) {
  return <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(5,15,25,0.34)] p-3"><p className="text-[9px] uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">{label}</p><p className="mt-1 truncate text-sm font-semibold text-[color:var(--ink)]" style={{ color }}>{value}</p><p className="mt-0.5 truncate text-[10px] text-[color:var(--ink-faint)]">{detail}</p></div>;
}

function PeerComparison({ industry, rows, loading, error }: { industry: IndustrySummary; rows: IndustryPeerRow[]; loading: boolean; error: string | null }) {
  const [preset, setPreset] = useState<PeerPreset>("all");
  const [sort, setSort] = useState<PeerSort>("marketCap");
  const [search, setSearch] = useState("");
  const [showAll, setShowAll] = useState(false);
  const normalizedSearch = search.trim().toLowerCase();
  const filteredRows = useMemo(() => {
    const matching = rows.filter((row) => !normalizedSearch || row.ticker.toLowerCase().includes(normalizedSearch) || row.name.toLowerCase().includes(normalizedSearch));
    return matching.sort((a, b) => {
      if (sort === "company") return a.name.localeCompare(b.name);
      if (sort === "move") return (b.pricePct ?? -Infinity) - (a.pricePct ?? -Infinity);
      if (sort === "revenue") return (b.revenue ?? -Infinity) - (a.revenue ?? -Infinity);
      if (sort === "profit") return (b.profit ?? -Infinity) - (a.profit ?? -Infinity);
      if (sort === "mentions") return b.mentions - a.mentions;
      return (b.marketCap ?? -Infinity) - (a.marketCap ?? -Infinity);
    });
  }, [normalizedSearch, rows, sort]);
  const visibleRows = showAll ? filteredRows : filteredRows.slice(0, 25);
  const showMarket = preset === "essentials" || preset === "all";
  const showFinancials = preset === "financials" || preset === "all";
  const showSignals = preset === "signals" || preset === "all";
  const presets: { id: PeerPreset; label: string }[] = [
    { id: "essentials", label: "Essentials" },
    { id: "financials", label: "Financials" },
    { id: "signals", label: "Signals" },
    { id: "all", label: "All columns" },
  ];

  return (
    <section aria-labelledby="peer-comparison-title" className="rounded-2xl border border-[color:var(--line-strong)] bg-[color:rgba(9,21,34,0.58)]">
      <div className="space-y-3 border-b border-[color:var(--line)] p-4 sm:p-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div><p className="text-[11px] font-semibold uppercase tracking-[0.1em] text-[color:var(--accent)]">Full peer comparison</p><h3 id="peer-comparison-title" className="mt-1 text-base font-semibold text-[color:var(--ink)]">{industry.label}</h3></div>
          <p className="text-[10px] text-[color:var(--ink-faint)]">{filteredRows.length} matching companies</p>
        </div>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex max-w-full overflow-x-auto rounded-xl border border-[color:var(--line)] bg-[color:rgba(5,15,25,0.45)] p-1" aria-label="Peer table column preset">
            {presets.map((item) => <button key={item.id} type="button" onClick={() => setPreset(item.id)} aria-pressed={preset === item.id} className={`whitespace-nowrap rounded-lg px-3 py-1.5 text-[10px] font-semibold ${preset === item.id ? "bg-[color:rgba(79,213,255,0.16)] text-[color:var(--ink)]" : "text-[color:var(--ink-faint)] hover:text-[color:var(--ink)]"}`}>{item.label}</button>)}
          </div>
          <div className="flex flex-1 flex-wrap justify-end gap-2">
            <input value={search} onChange={(event) => { setSearch(event.target.value); setShowAll(false); }} placeholder="Filter peers" aria-label="Filter industry peers" className="min-w-36 rounded-lg border border-[color:var(--line)] bg-[color:rgba(5,15,25,0.5)] px-2.5 py-1.5 text-xs text-[color:var(--ink)] outline-none focus:border-[color:var(--accent)]" />
            <select value={sort} onChange={(event) => setSort(event.target.value as PeerSort)} aria-label="Sort industry peers" className="rounded-lg border border-[color:var(--line)] bg-[color:rgba(5,15,25,0.5)] px-2.5 py-1.5 text-xs text-[color:var(--ink)]">
              <option value="marketCap">Market cap</option><option value="move">Today&apos;s move</option><option value="revenue">Revenue</option><option value="profit">Profit</option><option value="mentions">Mentions</option><option value="company">Company name</option>
            </select>
          </div>
        </div>
      </div>

      {loading && <div className="space-y-2 p-5" aria-label="Loading full peer comparison">{[0, 1, 2, 3, 4].map((item) => <div key={item} className="h-10 animate-pulse rounded-lg bg-[color:rgba(79,213,255,0.06)]" />)}</div>}
      {error && <p className="m-5 rounded-xl border border-red-500/20 bg-red-500/5 p-3 text-xs text-red-300">{error}</p>}
      {!loading && !error && filteredRows.length === 0 && <p className="p-8 text-center text-xs text-[color:var(--ink-faint)]">No companies match this view.</p>}

      {visibleRows.length > 0 && (
        <>
          <div className="divide-y divide-[color:var(--line)] sm:hidden">
            {visibleRows.map((row) => <PeerMobileCard key={row.ticker} row={row} preset={preset} />)}
          </div>
          <div className="hidden overflow-x-auto sm:block">
            <table className={`w-full ${preset === "all" ? "min-w-[1180px]" : "min-w-[760px]"}`}>
              <thead><tr className="border-b border-[color:var(--line)] text-[10px] uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">
                <th className="sticky left-0 z-20 w-[84px] bg-[#091522] px-4 py-2.5 text-left font-semibold">Ticker</th>
                <th className="sticky left-[84px] z-20 min-w-[210px] bg-[#091522] px-3 py-2.5 text-left font-semibold">Company</th>
                {showMarket && <><th className="px-3 py-2.5 text-right font-semibold">Price</th><th className="px-3 py-2.5 text-right font-semibold">Today</th><th className="px-3 py-2.5 text-right font-semibold">Market cap</th></>}
                {showFinancials && <><th className="px-3 py-2.5 text-right font-semibold">Revenue</th><th className="px-3 py-2.5 text-right font-semibold">Expenses</th><th className="px-3 py-2.5 text-right font-semibold">Profit</th></>}
                {showSignals && <><th className="px-3 py-2.5 text-right font-semibold">Mentions</th><th className="px-4 py-2.5 text-right font-semibold">Reports</th></>}
              </tr></thead>
              <tbody>{visibleRows.map((row) => <tr key={row.ticker} className="border-b border-[color:var(--line)] text-xs last:border-0 hover:bg-[color:rgba(79,213,255,0.035)]">
                <td className="sticky left-0 z-10 bg-[#091522] px-4 py-3 font-bold text-[color:var(--accent)]">{row.ticker}</td>
                <td className="sticky left-[84px] z-10 max-w-[240px] truncate bg-[#091522] px-3 py-3 text-[color:var(--ink-soft)]">{row.name}</td>
                {showMarket && <><td className="px-3 py-3 text-right tabular-nums text-[color:var(--ink-soft)]">{row.price == null ? "—" : `$${row.price.toLocaleString(undefined, { maximumFractionDigits: 2 })}`}</td><td className="px-3 py-3 text-right font-semibold tabular-nums" style={{ color: row.pricePct == null ? undefined : moveColor(row.pricePct) }}>{row.pricePct == null ? "—" : `${row.pricePct >= 0 ? "▲ " : "▼ "}${fmtPct(row.pricePct)}`}</td><td className="px-3 py-3 text-right tabular-nums text-[color:var(--ink-soft)]">{money(row.marketCap)}</td></>}
                {showFinancials && <><td className="px-3 py-3 text-right tabular-nums text-[color:var(--ink-soft)]">{money(row.revenue)}</td><td className="px-3 py-3 text-right tabular-nums text-[color:var(--ink-faint)]">{money(row.expenses)}</td><td className="px-3 py-3 text-right font-semibold tabular-nums" style={{ color: row.profit == null ? undefined : moveColor(row.profit) }}>{money(row.profit)}</td></>}
                {showSignals && <><td className="px-3 py-3 text-right tabular-nums text-[color:var(--ink-soft)]">{row.mentions || "—"}</td><td className="px-4 py-3 text-right text-[color:var(--ink-soft)]">{reportDateLabel(row.reportDate)}</td></>}
              </tr>)}</tbody>
            </table>
          </div>
        </>
      )}

      {filteredRows.length > 25 && <div className="flex items-center justify-between gap-3 border-t border-[color:var(--line)] px-4 py-3"><p className="text-[10px] text-[color:var(--ink-faint)]">Showing {showAll ? "all" : "the first 25"} of {filteredRows.length} companies.</p><button type="button" onClick={() => setShowAll((current) => !current)} className="rounded-lg border border-[color:var(--line)] px-2.5 py-1.5 text-[10px] font-semibold text-[color:var(--accent)] hover:bg-[color:rgba(79,213,255,0.08)]">{showAll ? "Show first 25" : `Show all ${filteredRows.length}`}</button></div>}
      <p className="border-t border-[color:var(--line)] px-4 py-3 text-[10px] leading-relaxed text-[color:var(--ink-faint)]">Today is the live quote change. Market cap uses shares outstanding × current price. Revenue, expenses, and profit use the latest SEC XBRL quarter; fiscal periods may differ across peers.</p>
    </section>
  );
}

function PeerMobileCard({ row, preset }: { row: IndustryPeerRow; preset: PeerPreset }) {
  const showMarket = preset === "essentials" || preset === "all";
  const showFinancials = preset === "financials" || preset === "all";
  const showSignals = preset === "signals" || preset === "all";
  return <div className="p-4"><div className="flex items-start justify-between gap-3"><div className="min-w-0"><p className="font-bold text-[color:var(--accent)]">{row.ticker}</p><p className="truncate text-xs text-[color:var(--ink-soft)]">{row.name}</p></div><p className="shrink-0 text-xs font-semibold" style={{ color: row.pricePct == null ? undefined : moveColor(row.pricePct) }}>{row.pricePct == null ? "—" : `${row.pricePct >= 0 ? "▲ " : "▼ "}${fmtPct(row.pricePct)}`}</p></div><div className="mt-3 grid grid-cols-2 gap-x-5 gap-y-2 text-[10px]">
    {showMarket && <><MobileMetric label="Price" value={row.price == null ? "—" : `$${row.price.toLocaleString(undefined, { maximumFractionDigits: 2 })}`} /><MobileMetric label="Market cap" value={money(row.marketCap)} /></>}
    {showFinancials && <><MobileMetric label="Revenue" value={money(row.revenue)} /><MobileMetric label="Expenses" value={money(row.expenses)} /><MobileMetric label="Profit" value={money(row.profit)} /></>}
    {showSignals && <><MobileMetric label="Mentions" value={row.mentions ? String(row.mentions) : "—"} /><MobileMetric label="Reports" value={reportDateLabel(row.reportDate)} /></>}
  </div></div>;
}

function MobileMetric({ label, value }: { label: string; value: string }) {
  return <div className="flex justify-between gap-2 border-b border-[color:var(--line)] pb-1"><span className="text-[color:var(--ink-faint)]">{label}</span><span className="tabular-nums text-[color:var(--ink-soft)]">{value}</span></div>;
}

export function MarketGroupsTab({ sectors, industries }: Props) {
  const [view, setView] = useState<ViewId>("moving");
  const [range, setRange] = useState<RangeId>("d1");
  const [query, setQuery] = useState("");
  const [selectedSectorName, setSelectedSectorName] = useState<string | null>(null);
  const [selectedIndustrySic, setSelectedIndustrySic] = useState<string | null>(null);
  const [highlightedTicker, setHighlightedTicker] = useState<string | null>(null);
  const [multiCompanyOnly, setMultiCompanyOnly] = useState(true);

  const sortedSectors = useMemo(
    () => [...(sectors.data?.sectors ?? [])].sort((a, b) => b.pcts[range] - a.pcts[range]),
    [range, sectors.data],
  );
  const selectedSector = sortedSectors.find((sector) => sector.name === selectedSectorName) ?? sortedSectors[0] ?? null;

  const sortedIndustries = useMemo(() => {
    const groups = [...(industries.data?.industries ?? [])];
    groups.sort((a, b) => b.attentionTotal - a.attentionTotal || b.tickers.length - a.tickers.length || a.label.localeCompare(b.label));
    return groups;
  }, [industries.data]);
  const selectedIndustry = sortedIndustries.find((industry) => industry.sic === selectedIndustrySic) ?? null;
  const industryPeers = useIndustryPeers(selectedIndustry);

  const normalizedQuery = query.trim().toLowerCase();
  const searchResults = useMemo(() => {
    if (!normalizedQuery) return null;
    const sectorMatches = sortedSectors.filter((sector) => sector.name.toLowerCase().includes(normalizedQuery)).slice(0, 5);
    const companyMatches = sortedSectors.flatMap((sector) => sector.stocks.map((company) => ({ company, sector })))
      .filter(({ company }) => company.symbol.toLowerCase().includes(normalizedQuery) || company.name.toLowerCase().includes(normalizedQuery))
      .slice(0, 8);
    const industryMatches = sortedIndustries.filter((industry) =>
      industry.label.toLowerCase().includes(normalizedQuery)
      || industry.sic.includes(normalizedQuery)
      || industry.tickers.some((ticker) => ticker.toLowerCase().includes(normalizedQuery))
    ).slice(0, 10);
    return { sectorMatches, companyMatches, industryMatches };
  }, [normalizedQuery, sortedIndustries, sortedSectors]);

  const visibleIndustries = sortedIndustries.filter((industry) => !multiCompanyOnly || industry.tickers.length > 1).slice(0, 100);
  const totalCompanies = sortedIndustries.reduce((sum, industry) => sum + industry.tickers.length, 0);

  const chooseSector = (sector: SectorData, ticker: string | null = null) => {
    setView("moving");
    setSelectedSectorName(sector.name);
    setHighlightedTicker(ticker);
    setQuery("");
  };
  const chooseIndustry = (industry: IndustrySummary) => {
    setView("industries");
    setSelectedIndustrySic(industry.sic);
    setHighlightedTicker(null);
    setQuery("");
  };

  const hasNoData = !sectors.data && !industries.data;
  if (hasNoData && (sectors.loading || industries.loading)) {
    return <div className="flex items-center justify-center py-16 text-sm text-[color:var(--ink-faint)]">Loading market groups…</div>;
  }
  if (hasNoData && sectors.error && industries.error) {
    return <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-4 text-sm text-red-400">Market groups are unavailable. {sectors.error}</div>;
  }

  return (
    <div className="space-y-5">
      <header className="rounded-2xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.48)] p-4 sm:p-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-base font-semibold text-[color:var(--ink)]">Market Groups</h2>
              <span className="rounded-full border border-[color:rgba(79,213,255,0.3)] bg-[color:rgba(79,213,255,0.08)] px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.08em] text-[color:var(--accent)]">Beta</span>
            </div>
            <p className="mt-1 max-w-2xl text-xs leading-5 text-[color:var(--ink-faint)]">
              Start with what is moving, then inspect a sector, SEC-classified industry, or company. Sector and industry remain separate classification lenses while their mapping is validated.
            </p>
          </div>
          <p className="text-[10px] text-[color:var(--ink-faint)]">
            {sectors.data ? `Prices updated ${dateTimeLabel(sectors.data.generatedAt)}` : "Sector prices unavailable"}
          </p>
        </div>

        <label className="mt-4 block">
          <span className="sr-only">Search sectors, industries, companies, or tickers</span>
          <div className="flex items-center gap-2 rounded-xl border border-[color:var(--line-strong)] bg-[color:rgba(5,15,25,0.7)] px-3 py-2.5 focus-within:border-[color:var(--accent)]">
            <span aria-hidden="true" className="text-[color:var(--ink-faint)]">⌕</span>
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search ticker, company, sector, or industry"
              className="min-w-0 flex-1 bg-transparent text-sm text-[color:var(--ink)] outline-none placeholder:text-[color:var(--ink-faint)]"
            />
            {query && <button type="button" onClick={() => setQuery("")} className="text-xs text-[color:var(--ink-faint)] hover:text-[color:var(--ink)]">Clear</button>}
          </div>
        </label>
      </header>

      {searchResults ? (
        <section aria-label="Search results" className="grid gap-4 lg:grid-cols-3">
          <SearchGroup title="Sectors" empty="No matching sectors">
            {searchResults.sectorMatches.map((sector) => <SearchButton key={sector.name} primary={sector.name} secondary={`${SECTOR_ETF[sector.name] ?? "Sector proxy"} · ${fmtPct(sector.pcts[range])}`} onClick={() => chooseSector(sector)} />)}
          </SearchGroup>
          <SearchGroup title="Companies" empty="No matching representative companies">
            {searchResults.companyMatches.map(({ company, sector }) => <SearchButton key={`${sector.name}:${company.symbol}`} primary={`${company.symbol} · ${company.name}`} secondary={`${sector.name} · ${fmtPct(company.pct)} today`} onClick={() => chooseSector(sector, company.symbol)} />)}
          </SearchGroup>
          <SearchGroup title="SEC industries" empty="No matching industries">
            {searchResults.industryMatches.map((industry) => <SearchButton key={industry.sic} primary={industry.label} secondary={`SIC ${industry.sic} · ${industry.tickers.length} companies`} onClick={() => chooseIndustry(industry)} />)}
          </SearchGroup>
        </section>
      ) : (
        <>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.6)] p-1" role="tablist" aria-label="Market group view">
              <button type="button" role="tab" aria-selected={view === "moving"} onClick={() => setView("moving")} className={`rounded-lg px-3 py-1.5 text-xs font-semibold ${view === "moving" ? "bg-[color:rgba(79,213,255,0.16)] text-[color:var(--ink)]" : "text-[color:var(--ink-faint)]"}`}>What&apos;s moving</button>
              <button type="button" role="tab" aria-selected={view === "industries"} onClick={() => setView("industries")} className={`rounded-lg px-3 py-1.5 text-xs font-semibold ${view === "industries" ? "bg-[color:rgba(79,213,255,0.16)] text-[color:var(--ink)]" : "text-[color:var(--ink-faint)]"}`}>Browse industries</button>
            </div>
            {view === "moving" && (
              <div className="flex rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.6)] p-1" aria-label="Performance period">
                {RANGES.map((item) => <button key={item.id} type="button" onClick={() => setRange(item.id)} aria-pressed={range === item.id} className={`rounded-lg px-2.5 py-1 text-xs ${range === item.id ? "bg-[color:rgba(79,213,255,0.16)] text-[color:var(--ink)]" : "text-[color:var(--ink-faint)]"}`}>{item.label}</button>)}
              </div>
            )}
          </div>

          {view === "moving" && (
            <>
              <div className="grid gap-3 sm:grid-cols-3">
                <SummaryCard label="Leading sector" value={sortedSectors[0]?.name ?? "Unavailable"} detail={sortedSectors[0] ? `${fmtPct(sortedSectors[0].pcts[range])} · ${RANGES.find((item) => item.id === range)?.label}` : ""} tone="positive" />
                <SummaryCard label="Lagging sector" value={sortedSectors.at(-1)?.name ?? "Unavailable"} detail={sortedSectors.at(-1) ? `${fmtPct(sortedSectors.at(-1)!.pcts[range])} · ${RANGES.find((item) => item.id === range)?.label}` : ""} tone="negative" />
                <SummaryCard label="Research coverage" value={`${sortedIndustries.length} industries`} detail={`${totalCompanies.toLocaleString()} company classifications · SEC SIC`} />
              </div>
              {sectors.error && <p className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-3 text-xs text-amber-300">Sector data warning: {sectors.error}</p>}
              <div className="grid gap-4 lg:grid-cols-[minmax(300px,0.78fr)_minmax(0,1.22fr)]">
                <section aria-label="Sector ranking" className="overflow-hidden rounded-2xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.42)]">
                  <div className="border-b border-[color:var(--line)] px-4 py-3">
                    <h3 className="text-sm font-semibold text-[color:var(--ink)]">Sector ranking</h3>
                    <p className="mt-0.5 text-[10px] text-[color:var(--ink-faint)]">ETF proxy returns · select a sector for representative companies</p>
                  </div>
                  {sortedSectors.map((sector, index) => {
                    const leader = strongest(sector.stocks);
                    const selected = selectedSector?.name === sector.name;
                    return (
                      <button key={sector.name} type="button" onClick={() => chooseSector(sector)} aria-pressed={selected} className={`grid w-full grid-cols-[28px_1fr_auto] items-center gap-2 border-b border-[color:var(--line)] px-4 py-3 text-left last:border-0 hover:bg-[color:rgba(79,213,255,0.05)] ${selected ? "bg-[color:rgba(79,213,255,0.08)]" : ""}`}>
                        <span className="text-[10px] tabular-nums text-[color:var(--ink-faint)]">{index + 1}</span>
                        <span className="min-w-0"><span className="block truncate text-xs font-semibold text-[color:var(--ink)]">{sector.name}</span><span className="mt-0.5 block truncate text-[10px] text-[color:var(--ink-faint)]">{leader ? `Leader today: ${leader.symbol} ${fmtPct(leader.pct)}` : "Representatives unavailable"}</span></span>
                        <span className="text-xs font-semibold tabular-nums" style={{ color: moveColor(sector.pcts[range]) }}>{sector.pcts[range] >= 0 ? "▲ " : "▼ "}{fmtPct(sector.pcts[range])}</span>
                      </button>
                    );
                  })}
                </section>
                {selectedSector && <SectorDetail sector={selectedSector} range={range} highlightedTicker={highlightedTicker} />}
              </div>
            </>
          )}

          {view === "industries" && (
            <>
              <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.42)] px-4 py-3">
                <div><p className="text-xs font-semibold text-[color:var(--ink)]">SEC industry directory</p><p className="mt-0.5 text-[10px] text-[color:var(--ink-faint)]">Ordered by latest attention, then peer-group size</p></div>
                <label className="flex items-center gap-2 text-xs text-[color:var(--ink-faint)]"><input type="checkbox" checked={multiCompanyOnly} onChange={(event) => setMultiCompanyOnly(event.target.checked)} /> Multi-company groups only</label>
              </div>
              {industries.error && <p className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-3 text-xs text-amber-300">Industry data warning: {industries.error}</p>}
              <div className="grid gap-4 lg:grid-cols-[minmax(300px,0.78fr)_minmax(0,1.22fr)]">
                <section aria-label="Industry directory" className="max-h-[360px] overflow-y-auto rounded-2xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.42)]">
                  {visibleIndustries.map((industry) => {
                    const selected = selectedIndustry?.sic === industry.sic;
                    return <button key={industry.sic} type="button" onClick={() => chooseIndustry(industry)} aria-pressed={selected} className={`grid w-full grid-cols-[1fr_auto] gap-3 border-b border-[color:var(--line)] px-4 py-3 text-left last:border-0 hover:bg-[color:rgba(79,213,255,0.05)] ${selected ? "bg-[color:rgba(79,213,255,0.08)]" : ""}`}><span className="min-w-0"><span className="block truncate text-xs font-semibold text-[color:var(--ink)]">{industry.label}</span><span className="mt-0.5 block text-[10px] text-[color:var(--ink-faint)]">SIC {industry.sic} · {industry.tickers.length} companies</span></span><span className="self-center text-[10px] tabular-nums text-[color:var(--ink-faint)]">{industry.attentionTotal > 0 ? `${industry.attentionTotal} mentions` : ""}</span></button>;
                  })}
                </section>
                {selectedIndustry ? (
                  <IndustrySnapshot industry={selectedIndustry} {...industryPeers} />
                ) : (
                  <section className="flex min-h-52 items-center justify-center rounded-2xl border border-dashed border-[color:var(--line)] bg-[color:rgba(9,21,34,0.3)] p-6 text-center">
                    <div><p className="text-sm font-semibold text-[color:var(--ink)]">Select an industry</p><p className="mt-1 max-w-sm text-xs leading-5 text-[color:var(--ink-faint)]">Choose a peer group to load its current prices, market caps, and latest-quarter profit.</p></div>
                  </section>
                )}
              </div>
              {visibleIndustries.length < sortedIndustries.filter((industry) => !multiCompanyOnly || industry.tickers.length > 1).length && <p className="text-right text-[10px] text-[color:var(--ink-faint)]">Showing the first 100 groups. Use search to find any industry or ticker.</p>}
              {selectedIndustry && <PeerComparison key={selectedIndustry.sic} industry={selectedIndustry} {...industryPeers} />}
            </>
          )}
        </>
      )}
    </div>
  );
}

function SummaryCard({ label, value, detail, tone }: { label: string; value: string; detail: string; tone?: "positive" | "negative" }) {
  const color = tone === "positive" ? "#41d39d" : tone === "negative" ? "#f87171" : undefined;
  return <div className="rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.42)] p-3"><p className="text-[10px] uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">{label}</p><p className="mt-1 truncate text-sm font-semibold text-[color:var(--ink)]" style={{ color }}>{value}</p><p className="mt-0.5 text-[10px] text-[color:var(--ink-faint)]">{detail}</p></div>;
}

function SearchGroup({ title, empty, children }: { title: string; empty: string; children: React.ReactNode }) {
  const hasChildren = Array.isArray(children) ? children.length > 0 : Boolean(children);
  return <div className="overflow-hidden rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.42)]"><h3 className="border-b border-[color:var(--line)] px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">{title}</h3>{hasChildren ? children : <p className="px-3 py-5 text-center text-xs text-[color:var(--ink-faint)]">{empty}</p>}</div>;
}

function SearchButton({ primary, secondary, onClick }: { primary: string; secondary: string; onClick: () => void }) {
  return <button type="button" onClick={onClick} className="block w-full border-b border-[color:var(--line)] px-3 py-3 text-left last:border-0 hover:bg-[color:rgba(79,213,255,0.06)]"><span className="block truncate text-xs font-semibold text-[color:var(--ink)]">{primary}</span><span className="mt-0.5 block truncate text-[10px] text-[color:var(--ink-faint)]">{secondary}</span></button>;
}

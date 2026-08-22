"use client";

import { useEffect, useMemo, useState } from "react";
import type {
  IndustrySummary,
  MarketIndustriesData,
  MarketSectorsData,
  SectorData,
  SectorStock,
} from "@/lib/server/types";

type RangeId = "d1" | "w1" | "m1" | "m3" | "ytd";
type ViewId = "moving" | "industries";

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

function IndustryDetail({ industry }: { industry: IndustrySummary }) {
  const [data, setData] = useState<MarketIndustriesData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showAllPeers, setShowAllPeers] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    setData(null);
    setLoading(true);
    setError(null);
    setShowAllPeers(false);
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
  }, [industry.label]);

  const rows = data?.peers?.rows ?? [];
  const visibleRows = showAllPeers ? rows : rows.slice(0, 25);

  return (
    <section aria-labelledby="selected-industry-title" className="rounded-2xl border border-[color:var(--line-strong)] bg-[color:rgba(9,21,34,0.72)] p-4 sm:p-5">
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
        <div className="mt-4 overflow-hidden rounded-xl border border-[color:var(--line)]">
          <div className="hidden grid-cols-[76px_1fr_repeat(3,minmax(76px,auto))] gap-3 border-b border-[color:var(--line)] px-3 py-2 text-[10px] uppercase tracking-[0.08em] text-[color:var(--ink-faint)] sm:grid">
            <span>Ticker</span><span>Company</span><span className="text-right">Today</span><span className="text-right">Market cap</span><span className="text-right">Profit</span>
          </div>
          {visibleRows.map((row) => (
            <div key={row.ticker} className="border-b border-[color:var(--line)] px-3 py-3 last:border-0 sm:grid sm:grid-cols-[76px_1fr_repeat(3,minmax(76px,auto))] sm:items-center sm:gap-3 sm:py-2.5">
              <div className="flex items-center justify-between sm:block">
                <span className="font-bold text-[color:var(--accent)]">{row.ticker}</span>
                <span className="font-semibold tabular-nums sm:hidden" style={{ color: moveColor(row.pricePct ?? 0) }}>
                  {row.pricePct == null ? "—" : `${row.pricePct >= 0 ? "▲ " : "▼ "}${fmtPct(row.pricePct)}`}
                </span>
              </div>
              <p className="mt-0.5 truncate text-xs text-[color:var(--ink-soft)] sm:mt-0">{row.name}</p>
              <span className="hidden text-right text-xs font-semibold tabular-nums sm:block" style={{ color: moveColor(row.pricePct ?? 0) }}>
                {row.pricePct == null ? "—" : `${row.pricePct >= 0 ? "▲ " : "▼ "}${fmtPct(row.pricePct)}`}
              </span>
              <div className="mt-2 flex justify-between text-[10px] text-[color:var(--ink-faint)] sm:mt-0 sm:block sm:text-right sm:text-xs">
                <span className="sm:hidden">Market cap</span><span className="tabular-nums text-[color:var(--ink-soft)]">{money(row.marketCap)}</span>
              </div>
              <div className="mt-1 flex justify-between text-[10px] text-[color:var(--ink-faint)] sm:mt-0 sm:block sm:text-right sm:text-xs">
                <span className="sm:hidden">Latest-quarter profit</span><span className="tabular-nums" style={{ color: row.profit == null ? undefined : moveColor(row.profit) }}>{money(row.profit)}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {rows.length > 25 && (
        <div className="mt-3 flex items-center justify-between gap-3">
          <p className="text-[10px] text-[color:var(--ink-faint)]">
            Showing {showAllPeers ? "all" : "the 25 largest"} of {rows.length} peers by available market cap.
          </p>
          <button type="button" onClick={() => setShowAllPeers((current) => !current)} className="rounded-lg border border-[color:var(--line)] px-2.5 py-1.5 text-[10px] font-semibold text-[color:var(--accent)] hover:bg-[color:rgba(79,213,255,0.08)]">
            {showAllPeers ? "Show top 25" : `Show all ${rows.length}`}
          </button>
        </div>
      )}

      <p className="mt-3 text-[10px] leading-relaxed text-[color:var(--ink-faint)]">
        Today is the live quote change. Market cap uses shares outstanding × current price. Profit is the latest SEC XBRL quarter and may cover different fiscal periods across peers.
      </p>
    </section>
  );
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
                <section aria-label="Industry directory" className="max-h-[720px] overflow-y-auto rounded-2xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.42)]">
                  {visibleIndustries.map((industry) => {
                    const selected = selectedIndustry?.sic === industry.sic;
                    return <button key={industry.sic} type="button" onClick={() => chooseIndustry(industry)} aria-pressed={selected} className={`grid w-full grid-cols-[1fr_auto] gap-3 border-b border-[color:var(--line)] px-4 py-3 text-left last:border-0 hover:bg-[color:rgba(79,213,255,0.05)] ${selected ? "bg-[color:rgba(79,213,255,0.08)]" : ""}`}><span className="min-w-0"><span className="block truncate text-xs font-semibold text-[color:var(--ink)]">{industry.label}</span><span className="mt-0.5 block text-[10px] text-[color:var(--ink-faint)]">SIC {industry.sic} · {industry.tickers.length} companies</span></span><span className="self-center text-[10px] tabular-nums text-[color:var(--ink-faint)]">{industry.attentionTotal > 0 ? `${industry.attentionTotal} mentions` : ""}</span></button>;
                  })}
                </section>
                {selectedIndustry ? (
                  <IndustryDetail industry={selectedIndustry} />
                ) : (
                  <section className="flex min-h-52 items-center justify-center rounded-2xl border border-dashed border-[color:var(--line)] bg-[color:rgba(9,21,34,0.3)] p-6 text-center">
                    <div><p className="text-sm font-semibold text-[color:var(--ink)]">Select an industry</p><p className="mt-1 max-w-sm text-xs leading-5 text-[color:var(--ink-faint)]">Choose a peer group to load its current prices, market caps, and latest-quarter profit.</p></div>
                  </section>
                )}
              </div>
              {visibleIndustries.length < sortedIndustries.filter((industry) => !multiCompanyOnly || industry.tickers.length > 1).length && <p className="text-right text-[10px] text-[color:var(--ink-faint)]">Showing the first 100 groups. Use search to find any industry or ticker.</p>}
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

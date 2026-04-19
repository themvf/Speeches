"use client";

import type { ExchangeInfo, ExchangeRegionGroup, MarketExchangesData, MarketStatus } from "@/lib/server/types";

interface Props {
  data: MarketExchangesData | null;
  loading: boolean;
  error: string | null;
}

function StatusPill({ status }: { status: MarketStatus }) {
  const cfg: Record<MarketStatus, { label: string; cls: string }> = {
    OPEN:   { label: "OPEN",   cls: "border-[color:rgba(65,211,157,0.35)] bg-[color:rgba(65,211,157,0.12)] text-[#41d39d]" },
    CLOSED: { label: "CLOSED", cls: "border-[color:rgba(248,113,113,0.3)] bg-[color:rgba(248,113,113,0.08)] text-[#f87171]" },
    PRE:    { label: "PRE",    cls: "border-[color:rgba(242,171,67,0.35)] bg-[color:rgba(242,171,67,0.12)] text-[#f2ab43]" },
    AFTER:  { label: "AFTER",  cls: "border-[color:rgba(242,171,67,0.35)] bg-[color:rgba(242,171,67,0.12)] text-[#f2ab43]" },
  };
  const { label, cls } = cfg[status] ?? cfg.CLOSED;
  return (
    <span className={`rounded border px-2 py-0.5 text-[10px] font-bold tracking-[0.06em] ${cls}`}>
      {label}
    </span>
  );
}

function ExchangeRow({ ex }: { ex: ExchangeInfo }) {
  const tzAbbr = ex.timezone.split("/").pop()?.replace(/_/g, " ") ?? ex.timezone;
  return (
    <tr className="border-b border-[color:var(--line)] last:border-0 hover:bg-[color:rgba(79,213,255,0.03)]">
      <td className="pl-4 pr-2 py-2.5 w-14">
        <span className="text-xs font-bold text-[color:var(--accent)]">{ex.code}</span>
      </td>
      <td className="px-2 py-2.5 text-xs text-[color:var(--ink)]">{ex.name}</td>
      <td className="hidden sm:table-cell px-2 py-2.5 text-xs text-[color:var(--ink-faint)] text-right">{tzAbbr}</td>
      <td className="px-4 py-2.5 text-right">
        <StatusPill status={ex.status} />
      </td>
    </tr>
  );
}

function RegionGroup({ group }: { group: ExchangeRegionGroup }) {
  const openCount = group.exchanges.filter((e) => e.status === "OPEN").length;

  return (
    <div>
      <p className="px-4 py-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-[color:var(--ink-faint)] border-b border-[color:var(--line)] bg-[color:rgba(9,21,34,0.6)]">
        {group.region}
        {openCount > 0 && (
          <span className="ml-2 text-[#41d39d]">{openCount} open</span>
        )}
      </p>
      <table className="w-full">
        <tbody>
          {group.exchanges.map((ex) => <ExchangeRow key={ex.code} ex={ex} />)}
        </tbody>
      </table>
    </div>
  );
}

export function ExchangesTab({ data, loading, error }: Props) {
  if (loading && !data) {
    return (
      <div className="flex items-center justify-center py-16 text-sm text-[color:var(--ink-faint)]">
        Loading exchanges…
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-4 text-sm text-red-400">
        {error}
      </div>
    );
  }

  if (!data) return null;

  const totalOpen = data.regions.flatMap((r) => r.exchanges).filter((e) => e.status === "OPEN").length;
  const totalExchanges = data.regions.flatMap((r) => r.exchanges).length;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
          Global Exchanges
          <span className="ml-2 font-normal">{totalOpen}/{totalExchanges} open</span>
        </p>
        <span className="text-xs text-[color:var(--ink-faint)]">
          {new Date(data.generatedAt).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" })}
        </span>
      </div>

      <div className="overflow-hidden rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.4)] divide-y divide-[color:var(--line)]">
        {data.regions.map((group) => (
          <RegionGroup key={group.region} group={group} />
        ))}
      </div>
    </div>
  );
}

"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { PortfolioCandle, PortfolioQuote } from "@/lib/portfolio-types";

const CLIENT_CACHE_MS = 5 * 60 * 1_000;
const chartCache = new Map<string, { candles: PortfolioCandle[]; expiresAt: number }>();
const chartRequests = new Map<string, Promise<PortfolioCandle[]>>();

function validCandle(value: unknown): value is PortfolioCandle {
  if (!value || typeof value !== "object") return false;
  const row = value as Partial<PortfolioCandle>;
  return [row.t, row.open, row.high, row.low, row.close].every((item) => typeof item === "number" && Number.isFinite(item));
}

async function loadCandles(symbol: string): Promise<PortfolioCandle[]> {
  const cached = chartCache.get(symbol);
  if (cached && cached.expiresAt > Date.now()) return cached.candles;
  const active = chartRequests.get(symbol);
  if (active) return active;
  const request = fetch(`/api/portfolio-chart?symbol=${encodeURIComponent(symbol)}`)
    .then(async (response) => {
      const body = await response.json() as { ok?: boolean; data?: unknown; error?: string };
      if (!response.ok || !body.ok) throw new Error(body.error || "Chart data could not be loaded.");
      const candles = Array.isArray(body.data) ? body.data.filter(validCandle) : [];
      if (!candles.length) throw new Error("No complete daily candles are available.");
      chartCache.set(symbol, { candles, expiresAt: Date.now() + CLIENT_CACHE_MS });
      return candles;
    })
    .finally(() => chartRequests.delete(symbol));
  chartRequests.set(symbol, request);
  return request;
}

function price(value: number): string {
  if (Math.abs(value) >= 1) return value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  return value.toLocaleString("en-US", { minimumFractionDigits: 4, maximumFractionDigits: 4 });
}

function compactVolume(value: number | null): string {
  if (value == null) return "Unavailable";
  return Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

function CandleChart({ candles, selected, onSelected }: { candles: PortfolioCandle[]; selected: number; onSelected: (index: number) => void }) {
  const width = 380;
  const height = 180;
  const left = 8;
  const right = 48;
  const priceTop = 8;
  const priceBottom = 120;
  const volumeTop = 132;
  const volumeBottom = 167;
  const plotWidth = width - left - right;
  const lows = candles.map((candle) => candle.low);
  const highs = candles.map((candle) => candle.high);
  const rawMin = Math.min(...lows);
  const rawMax = Math.max(...highs);
  const padding = Math.max((rawMax - rawMin) * 0.06, rawMax * 0.002, 0.01);
  const min = rawMin - padding;
  const max = rawMax + padding;
  const priceHeight = priceBottom - priceTop;
  const maxVolume = Math.max(...candles.map((candle) => candle.volume ?? 0), 1);
  const step = plotWidth / candles.length;
  const bodyWidth = Math.max(1, Math.min(4, step * 0.65));
  const y = (value: number) => priceTop + ((max - value) / (max - min)) * priceHeight;
  const selectedX = left + (selected + 0.5) * step;
  const dateIndexes = Array.from(new Set([0, Math.floor((candles.length - 1) / 3), Math.floor((candles.length - 1) * 2 / 3), candles.length - 1]));

  function selectAt(clientX: number, element: SVGSVGElement) {
    const rect = element.getBoundingClientRect();
    const svgX = ((clientX - rect.left) / rect.width) * width;
    const index = Math.max(0, Math.min(candles.length - 1, Math.floor((svgX - left) / step)));
    onSelected(index);
  }

  return <svg
    viewBox={`0 0 ${width} ${height}`}
    className="block w-full touch-pan-y"
    role="img"
    aria-label="Six-month daily candlestick price chart with trading volume"
    onPointerMove={(event) => selectAt(event.clientX, event.currentTarget)}
    onPointerLeave={() => onSelected(candles.length - 1)}
  >
    {[0, 1, 2, 3].map((line) => {
      const value = max - ((max - min) * line / 3);
      const lineY = priceTop + (priceHeight * line / 3);
      return <g key={line}>
        <line x1={left} x2={width - right} y1={lineY} y2={lineY} stroke="rgba(148,163,184,.14)" strokeDasharray="3 3" />
        <text x={width - right + 5} y={lineY + 3} fill="rgba(203,213,225,.58)" fontSize="9">{price(value)}</text>
      </g>;
    })}
    {candles.map((candle, index) => {
      const x = left + (index + 0.5) * step;
      const rising = candle.close >= candle.open;
      const color = rising ? "#22c55e" : "#f43f5e";
      const openY = y(candle.open);
      const closeY = y(candle.close);
      const bodyTop = Math.min(openY, closeY);
      const bodyHeight = Math.max(1, Math.abs(openY - closeY));
      const volumeHeight = candle.volume == null ? 0 : ((candle.volume / maxVolume) * (volumeBottom - volumeTop));
      return <g key={candle.t}>
        <line x1={x} x2={x} y1={y(candle.high)} y2={y(candle.low)} stroke={color} strokeWidth="1" />
        <rect x={x - bodyWidth / 2} y={bodyTop} width={bodyWidth} height={bodyHeight} fill={color} rx=".4" />
        {volumeHeight > 0 ? <rect x={x - bodyWidth / 2} y={volumeBottom - volumeHeight} width={bodyWidth} height={volumeHeight} fill={color} opacity=".42" /> : null}
      </g>;
    })}
    <line x1={left} x2={width - right} y1={volumeTop - 6} y2={volumeTop - 6} stroke="rgba(148,163,184,.16)" />
    <text x={left} y={volumeTop + 5} fill="rgba(203,213,225,.48)" fontSize="8">VOL</text>
    {dateIndexes.map((index) => <text key={index} x={left + (index + 0.5) * step} y={height - 2} fill="rgba(203,213,225,.52)" fontSize="9" textAnchor={index === 0 ? "start" : index === candles.length - 1 ? "end" : "middle"}>{new Date(candles[index].t * 1000).toLocaleDateString("en-US", { month: "short" })}</text>)}
    <line x1={selectedX} x2={selectedX} y1={priceTop} y2={volumeBottom} stroke="rgba(125,211,252,.55)" strokeDasharray="2 2" pointerEvents="none" />
  </svg>;
}

export function PortfolioTickerChart({ symbol, name, quote }: { symbol: string; name?: string; quote?: PortfolioQuote }) {
  const triggerRef = useRef<HTMLButtonElement>(null);
  const popoverRef = useRef<HTMLDivElement>(null);
  const openTimer = useRef<number | null>(null);
  const closeTimer = useRef<number | null>(null);
  const [open, setOpen] = useState(false);
  const [candles, setCandles] = useState<PortfolioCandle[] | null>(null);
  const [selected, setSelected] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [position, setPosition] = useState({ left: 8, top: 8, width: 400 });
  const popoverId = `portfolio-chart-${symbol.replace(/[^A-Za-z0-9]/g, "-")}`;

  const clearTimers = useCallback(() => {
    if (openTimer.current != null) window.clearTimeout(openTimer.current);
    if (closeTimer.current != null) window.clearTimeout(closeTimer.current);
  }, []);
  const scheduleOpen = useCallback(() => {
    if (closeTimer.current != null) window.clearTimeout(closeTimer.current);
    if (openTimer.current != null) window.clearTimeout(openTimer.current);
    openTimer.current = window.setTimeout(() => setOpen(true), 120);
  }, []);
  const scheduleClose = useCallback(() => {
    if (openTimer.current != null) window.clearTimeout(openTimer.current);
    if (closeTimer.current != null) window.clearTimeout(closeTimer.current);
    closeTimer.current = window.setTimeout(() => setOpen(false), 180);
  }, []);
  const updatePosition = useCallback(() => {
    const trigger = triggerRef.current;
    if (!trigger) return;
    const rect = trigger.getBoundingClientRect();
    const width = Math.min(400, window.innerWidth - 16);
    const expectedHeight = 315;
    const left = Math.max(8, Math.min(window.innerWidth - width - 8, rect.left));
    const top = rect.bottom + 8 + expectedHeight <= window.innerHeight
      ? rect.bottom + 8
      : Math.max(8, rect.top - expectedHeight - 8);
    setPosition({ left, top, width });
  }, []);

  useEffect(() => () => clearTimers(), [clearTimers]);
  useEffect(() => {
    if (!open) return;
    updatePosition();
    const closeOnEscape = (event: KeyboardEvent) => { if (event.key === "Escape") { setOpen(false); triggerRef.current?.focus(); } };
    const closeOutside = (event: PointerEvent) => {
      const target = event.target as Node;
      if (!triggerRef.current?.contains(target) && !popoverRef.current?.contains(target)) setOpen(false);
    };
    window.addEventListener("resize", updatePosition);
    window.addEventListener("scroll", updatePosition, true);
    document.addEventListener("keydown", closeOnEscape);
    document.addEventListener("pointerdown", closeOutside);
    return () => {
      window.removeEventListener("resize", updatePosition);
      window.removeEventListener("scroll", updatePosition, true);
      document.removeEventListener("keydown", closeOnEscape);
      document.removeEventListener("pointerdown", closeOutside);
    };
  }, [open, updatePosition]);
  useEffect(() => {
    if (!open || candles || loading || error) return;
    setLoading(true);
    setError("");
    void loadCandles(symbol).then((data) => {
      setCandles(data);
      setSelected(data.length - 1);
    }).catch((reason) => setError(reason instanceof Error ? reason.message : "Chart data could not be loaded."))
      .finally(() => setLoading(false));
  }, [candles, error, loading, open, symbol]);

  const selectedCandle = candles?.[selected] ?? null;
  const rangeChange = useMemo(() => {
    if (!candles?.length) return null;
    const first = candles[0].close;
    const last = candles[candles.length - 1].close;
    return first ? ((last - first) / first) * 100 : null;
  }, [candles]);

  const popover = open && typeof document !== "undefined" ? createPortal(
    <div
      ref={popoverRef}
      id={popoverId}
      role="dialog"
      aria-label={`${symbol} six-month price chart`}
      className="fixed z-[100] overflow-hidden rounded-xl border border-cyan-300/25 bg-[#091522] text-slate-100 shadow-2xl shadow-black/50"
      style={{ left: position.left, top: position.top, width: position.width }}
      onPointerEnter={() => { if (closeTimer.current != null) window.clearTimeout(closeTimer.current); }}
      onPointerLeave={scheduleClose}
    >
      <div className="flex items-start justify-between gap-3 border-b border-white/10 px-4 py-3">
        <div className="min-w-0"><div className="flex items-center gap-2"><span className="text-base font-bold">{symbol}</span>{rangeChange != null ? <span className={`text-xs font-semibold ${rangeChange >= 0 ? "text-emerald-400" : "text-rose-400"}`}>{rangeChange >= 0 ? "+" : ""}{rangeChange.toFixed(2)}% <span className="font-normal text-slate-500">6M</span></span> : null}</div><p className="truncate text-[11px] text-slate-400">{name || quote?.name || "Yahoo Finance"}</p></div>
        <span className="shrink-0 rounded border border-white/10 px-2 py-1 text-[10px] uppercase tracking-wider text-slate-400">Daily</span>
      </div>
      {loading ? <div className="flex h-[220px] items-center justify-center text-sm text-slate-400">Loading chart…</div> : error ? <div className="flex h-[220px] flex-col items-center justify-center gap-3 px-6 text-center text-sm text-amber-300"><span>{error}</span><button type="button" onClick={() => setError("")} className="rounded-lg border border-amber-300/25 px-3 py-1.5 text-xs text-amber-200">Retry</button></div> : candles && selectedCandle ? <>
        <div className="grid grid-cols-5 gap-2 border-b border-white/10 px-4 py-2 text-[10px]">
          <div><span className="block text-slate-500">{new Date(selectedCandle.t * 1000).toLocaleDateString("en-US", { month: "short", day: "numeric" })}</span><span className="font-semibold text-slate-300">OHLC</span></div>
          <div><span className="block text-slate-500">Open</span><span className="tabular-nums">{price(selectedCandle.open)}</span></div>
          <div><span className="block text-slate-500">High</span><span className="tabular-nums">{price(selectedCandle.high)}</span></div>
          <div><span className="block text-slate-500">Low</span><span className="tabular-nums">{price(selectedCandle.low)}</span></div>
          <div><span className="block text-slate-500">Close</span><span className="tabular-nums">{price(selectedCandle.close)}</span></div>
        </div>
        <div className="px-2 pt-1"><CandleChart candles={candles} selected={selected} onSelected={setSelected} /></div>
        <div className="flex flex-wrap items-center justify-between gap-2 border-t border-white/10 px-4 py-2 text-[10px] text-slate-400"><span>Volume {compactVolume(selectedCandle.volume)}</span><span>{quote?.fiftyTwoWeekLow != null ? `52W low ${price(quote.fiftyTwoWeekLow)}` : ""}{quote?.fiftyTwoWeekLow != null && quote?.fiftyTwoWeekHigh != null ? " · " : ""}{quote?.fiftyTwoWeekHigh != null ? `52W high ${price(quote.fiftyTwoWeekHigh)}` : ""}</span><span>Yahoo Finance</span></div>
      </> : null}
    </div>, document.body
  ) : null;

  return <>
    <button
      ref={triggerRef}
      type="button"
      aria-expanded={open}
      aria-controls={popoverId}
      aria-label={`Show ${symbol} six-month price chart`}
      className="rounded font-semibold text-cyan-300 underline decoration-cyan-300/30 decoration-dotted underline-offset-4 hover:text-cyan-200 focus:outline-none focus:ring-2 focus:ring-cyan-300/50"
      onMouseEnter={scheduleOpen}
      onMouseLeave={scheduleClose}
      onFocus={scheduleOpen}
      onBlur={scheduleClose}
      onClick={() => {
        if (openTimer.current != null) window.clearTimeout(openTimer.current);
        setOpen((value) => !value);
      }}
    >{symbol}</button>
    {popover}
  </>;
}

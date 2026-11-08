"use client";

import { useCallback, useEffect, useState } from "react";
import { normalizePortfoliosPayload } from "@/lib/portfolio-calculations";
import type { Portfolio, PortfolioHolding, PortfoliosPayload } from "@/lib/portfolio-types";

const STORAGE_KEY = "policy_hub_portfolios_v1";

function id(prefix: string): string {
  return `${prefix}:${typeof crypto !== "undefined" && crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`}`;
}

function starterPayload(): PortfoliosPayload {
  const now = new Date().toISOString();
  return { portfolios: [{ id: id("portfolio"), name: "My Portfolio", holdings: [], createdAt: now, updatedAt: now }], updatedAt: now };
}

function readLocal(): PortfoliosPayload | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? normalizePortfoliosPayload(JSON.parse(raw)) : null;
  } catch { return null; }
}

function writeLocal(payload: PortfoliosPayload): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
}

async function readRemote(): Promise<PortfoliosPayload | null> {
  const response = await fetch("/api/portfolios", { cache: "no-store" });
  if (response.status === 401 || response.status === 503) return null;
  if (!response.ok) throw new Error("Portfolio sync failed.");
  const body = await response.json() as { ok?: boolean; data?: unknown };
  return body.ok ? normalizePortfoliosPayload(body.data) : null;
}

async function writeRemote(payload: PortfoliosPayload): Promise<boolean> {
  const response = await fetch("/api/portfolios", {
    method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
  });
  return response.ok;
}

export function usePortfolios() {
  const [payload, setPayload] = useState<PortfoliosPayload>({ portfolios: [], updatedAt: new Date(0).toISOString() });
  const [loaded, setLoaded] = useState(false);
  const [remoteChecked, setRemoteChecked] = useState(false);
  const [syncEnabled, setSyncEnabled] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [pendingSync, setPendingSync] = useState(0);

  useEffect(() => {
    const local = readLocal();
    const initial = local ?? starterPayload();
    setPayload(initial);
    writeLocal(initial);
    setLoaded(true);
    readRemote().then((remote) => {
      if (!remote) return;
      setSyncEnabled(true);
      const localIsNewer = Date.parse(initial.updatedAt) > Date.parse(remote.updatedAt);
      const chosen = localIsNewer ? initial : remote;
      const ready = chosen.portfolios.length ? chosen : starterPayload();
      setPayload(ready);
      writeLocal(ready);
      if (localIsNewer) setPendingSync(1);
    }).catch(() => setSyncEnabled(false)).finally(() => setRemoteChecked(true));
  }, []);

  useEffect(() => {
    if (!loaded || !remoteChecked || !syncEnabled || pendingSync === 0) return;
    const syncToken = pendingSync;
    const timer = window.setTimeout(() => {
      setSyncing(true);
      void writeRemote(payload).then((ok) => {
        if (!ok) {
          setSyncEnabled(false);
          return;
        }
        setPendingSync((current) => current === syncToken ? 0 : current);
      }).catch(() => setSyncEnabled(false)).finally(() => setSyncing(false));
    }, 2_000);
    return () => window.clearTimeout(timer);
  }, [loaded, payload, pendingSync, remoteChecked, syncEnabled]);

  const commit = useCallback((update: (current: Portfolio[]) => Portfolio[]) => {
    setPendingSync((current) => current + 1);
    setPayload((current) => {
      const updatedAt = new Date().toISOString();
      const next = { portfolios: update(current.portfolios), updatedAt };
      writeLocal(next);
      return next;
    });
  }, []);

  const createPortfolio = useCallback((name: string): string | null => {
    const clean = name.replace(/\s+/g, " ").trim().slice(0, 80);
    if (!clean) return null;
    const now = new Date().toISOString();
    const portfolio: Portfolio = { id: id("portfolio"), name: clean, holdings: [], createdAt: now, updatedAt: now };
    commit((current) => [...current, portfolio]);
    return portfolio.id;
  }, [commit]);

  const updatePortfolio = useCallback((portfolioId: string, transform: (portfolio: Portfolio) => Portfolio) => {
    const now = new Date().toISOString();
    commit((current) => current.map((portfolio) => portfolio.id === portfolioId ? { ...transform(portfolio), updatedAt: now } : portfolio));
  }, [commit]);

  const renamePortfolio = useCallback((portfolioId: string, name: string) => {
    const clean = name.replace(/\s+/g, " ").trim().slice(0, 80);
    if (clean) updatePortfolio(portfolioId, (portfolio) => ({ ...portfolio, name: clean }));
  }, [updatePortfolio]);

  const deletePortfolio = useCallback((portfolioId: string) => {
    commit((current) => current.filter((portfolio) => portfolio.id !== portfolioId));
  }, [commit]);

  const restorePortfolio = useCallback((portfolio: Portfolio, index: number) => {
    commit((current) => {
      if (current.some((item) => item.id === portfolio.id)) return current;
      const next = [...current];
      next.splice(Math.max(0, Math.min(index, next.length)), 0, portfolio);
      return next;
    });
  }, [commit]);

  const addHolding = useCallback((portfolioId: string, input: Pick<PortfolioHolding, "symbol" | "shares" | "tradePrice" | "tradeDate">) => {
    const now = new Date().toISOString();
    updatePortfolio(portfolioId, (portfolio) => ({
      ...portfolio,
      holdings: [...portfolio.holdings, { ...input, id: id("holding"), createdAt: now }],
    }));
  }, [updatePortfolio]);

  const updateHolding = useCallback((portfolioId: string, holdingId: string, patch: Partial<Pick<PortfolioHolding, "shares" | "tradePrice" | "tradeDate">>) => {
    updatePortfolio(portfolioId, (portfolio) => ({
      ...portfolio,
      holdings: portfolio.holdings.map((holding) => holding.id === holdingId ? { ...holding, ...patch } : holding),
    }));
  }, [updatePortfolio]);

  const deleteHolding = useCallback((portfolioId: string, holdingId: string) => {
    updatePortfolio(portfolioId, (portfolio) => ({ ...portfolio, holdings: portfolio.holdings.filter((holding) => holding.id !== holdingId) }));
  }, [updatePortfolio]);

  const restoreHolding = useCallback((portfolioId: string, holding: PortfolioHolding, index: number) => {
    updatePortfolio(portfolioId, (portfolio) => {
      if (portfolio.holdings.some((item) => item.id === holding.id)) return portfolio;
      const holdings = [...portfolio.holdings];
      holdings.splice(Math.max(0, Math.min(index, holdings.length)), 0, holding);
      return { ...portfolio, holdings };
    });
  }, [updatePortfolio]);

  return {
    portfolios: payload.portfolios,
    loaded,
    syncEnabled,
    syncing,
    syncPending: syncEnabled && pendingSync > 0,
    createPortfolio,
    renamePortfolio,
    deletePortfolio,
    restorePortfolio,
    addHolding,
    updateHolding,
    deleteHolding,
    restoreHolding,
  };
}

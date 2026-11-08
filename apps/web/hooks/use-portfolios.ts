"use client";

import { useCallback, useEffect, useState } from "react";
import { normalizePortfoliosPayload } from "@/lib/portfolio-calculations";
import type { Portfolio, PortfolioHolding, PortfoliosPayload } from "@/lib/portfolio-types";

const LEGACY_STORAGE_KEY = "policy_hub_portfolios_v1";
const SYNC_TOKEN_KEY = "policy_hub_portfolio_sync_token_v1";
const STORAGE_PREFIX = "policy_hub_portfolios_v2";
const TOKEN_PATTERN = /^[A-Za-z0-9_-]{43}$/;

function id(prefix: string): string {
  return `${prefix}:${typeof crypto !== "undefined" && crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`}`;
}

function starterPayload(): PortfoliosPayload {
  const now = new Date().toISOString();
  return { portfolios: [{ id: id("portfolio"), name: "My Portfolio", holdings: [], createdAt: now, updatedAt: now }], updatedAt: now };
}

function generateSyncToken(): string {
  const bytes = new Uint8Array(32);
  crypto.getRandomValues(bytes);
  return btoa(String.fromCharCode(...bytes)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

function resolveSyncIdentity(): { token: string; allowLegacy: boolean } {
  const stored = localStorage.getItem(SYNC_TOKEN_KEY);
  const storedToken = stored && TOKEN_PATTERN.test(stored) ? stored : null;
  const fragment = new URLSearchParams(window.location.hash.replace(/^#/, ""));
  const candidate = fragment.get("portfolio-sync");
  const incomingToken = candidate && TOKEN_PATTERN.test(candidate) ? candidate : null;
  const token = incomingToken ?? storedToken ?? generateSyncToken();
  const changedWorkspace = Boolean(incomingToken && incomingToken !== storedToken);
  localStorage.setItem(SYNC_TOKEN_KEY, token);
  if (candidate) window.history.replaceState(null, "", `${window.location.pathname}${window.location.search}`);
  return { token, allowLegacy: !changedWorkspace };
}

function storageKey(token: string): string {
  return `${STORAGE_PREFIX}:${token}`;
}

function readLocal(token: string, allowLegacy: boolean): PortfoliosPayload | null {
  try {
    const raw = localStorage.getItem(storageKey(token)) ?? (allowLegacy ? localStorage.getItem(LEGACY_STORAGE_KEY) : null);
    return raw ? normalizePortfoliosPayload(JSON.parse(raw)) : null;
  } catch { return null; }
}

function writeLocal(token: string, payload: PortfoliosPayload): void {
  localStorage.setItem(storageKey(token), JSON.stringify(payload));
}

async function readRemote(token: string): Promise<PortfoliosPayload | null> {
  const response = await fetch("/api/portfolios", { cache: "no-store", headers: { "x-portfolio-sync-token": token } });
  if (response.status === 401 || response.status === 429 || response.status === 503) return null;
  if (!response.ok) throw new Error("Portfolio sync failed.");
  const body = await response.json() as { ok?: boolean; data?: unknown };
  return body.ok ? normalizePortfoliosPayload(body.data) : null;
}

async function writeRemote(token: string, payload: PortfoliosPayload): Promise<boolean> {
  const response = await fetch("/api/portfolios", {
    method: "PUT", headers: { "Content-Type": "application/json", "x-portfolio-sync-token": token }, body: JSON.stringify(payload),
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
  const [syncToken, setSyncToken] = useState<string | null>(null);

  useEffect(() => {
    const identity = resolveSyncIdentity();
    setSyncToken(identity.token);
    const local = readLocal(identity.token, identity.allowLegacy);
    const initial = local ?? starterPayload();
    setPayload(initial);
    writeLocal(identity.token, initial);
    setLoaded(true);
    readRemote(identity.token).then((remote) => {
      if (!remote) return;
      setSyncEnabled(true);
      const remoteHasData = remote.portfolios.length > 0;
      const localIsNewer = Boolean(local) && Date.parse(initial.updatedAt) > Date.parse(remote.updatedAt);
      const chosen = !local && remoteHasData ? remote : localIsNewer || !remoteHasData ? initial : remote;
      const ready = chosen.portfolios.length ? chosen : starterPayload();
      setPayload(ready);
      writeLocal(identity.token, ready);
      if (localIsNewer || !remoteHasData) setPendingSync(1);
    }).catch(() => setSyncEnabled(false)).finally(() => setRemoteChecked(true));
  }, []);

  useEffect(() => {
    if (!loaded || !remoteChecked || !syncEnabled || !syncToken || pendingSync === 0) return;
    const syncVersion = pendingSync;
    const timer = window.setTimeout(() => {
      setSyncing(true);
      void writeRemote(syncToken, payload).then((ok) => {
        if (!ok) {
          setSyncEnabled(false);
          return;
        }
        setPendingSync((current) => current === syncVersion ? 0 : current);
      }).catch(() => setSyncEnabled(false)).finally(() => setSyncing(false));
    }, 2_000);
    return () => window.clearTimeout(timer);
  }, [loaded, payload, pendingSync, remoteChecked, syncEnabled, syncToken]);

  const commit = useCallback((update: (current: Portfolio[]) => Portfolio[]) => {
    setPendingSync((current) => current + 1);
    setPayload((current) => {
      const updatedAt = new Date().toISOString();
      const next = { portfolios: update(current.portfolios), updatedAt };
      if (syncToken) writeLocal(syncToken, next);
      return next;
    });
  }, [syncToken]);

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
    syncLink: syncToken && typeof window !== "undefined" ? `${window.location.origin}${window.location.pathname}#portfolio-sync=${syncToken}` : null,
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

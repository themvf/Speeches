import { NextResponse } from "next/server";

import type { ApiErrorPayload, ApiSuccessPayload } from "@/lib/server/types";

export function createRequestId(): string {
  const t = Date.now().toString(36);
  const r = Math.random().toString(36).slice(2, 10);
  return `req_${t}${r}`;
}

export function ok<T>(data: T, requestId?: string) {
  const payload: ApiSuccessPayload<T> = requestId ? { ok: true, data, request_id: requestId } : { ok: true, data };
  return NextResponse.json(payload, { status: 200 });
}

export function fail(error: string, code: string, status = 400, requestId?: string) {
  const payload: ApiErrorPayload = requestId
    ? { ok: false, error, code, request_id: requestId }
    : { ok: false, error, code };
  return NextResponse.json(payload, { status });
}

export function toInt(value: string | null, fallback: number, minValue: number, maxValue: number): number {
  const raw = Number.parseInt(String(value ?? ""), 10);
  const parsed = Number.isFinite(raw) ? raw : fallback;
  return Math.max(minValue, Math.min(maxValue, parsed));
}

export function parseDate(value: string | null): Date | null {
  if (!value) return null;
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return null;
  const d = new Date(value + "T00:00:00Z");
  return Number.isNaN(d.getTime()) ? null : d;
}

export function normalizeText(value: unknown): string {
  return String(value ?? "")
    .replace(/\s+/g, " ")
    .trim();
}

export type CronAuthResult =
  | { ok: true }
  | { ok: false; status: 401 | 503; error: string };

/**
 * Shared auth check for cron/maintenance-triggered refresh endpoints
 * (CRON_SECRET or RSS_REENRICH_SECRET as a Bearer token). Fails closed: if
 * neither secret is configured, the endpoint is treated as unavailable
 * rather than open to unauthenticated callers.
 */
export function checkCronAuth(req: { headers: { get(name: string): string | null } }): CronAuthResult {
  const secret = process.env.CRON_SECRET ?? "";
  const maintenanceSecret = process.env.RSS_REENRICH_SECRET ?? "";
  const acceptedTokens = [secret, maintenanceSecret].filter(Boolean).map((token) => `Bearer ${token}`);
  if (acceptedTokens.length === 0) {
    return {
      ok: false,
      status: 503,
      error: "Refresh endpoint is not configured (CRON_SECRET/RSS_REENRICH_SECRET unset).",
    };
  }
  const authHeader = req.headers.get("authorization") ?? "";
  if (!acceptedTokens.includes(authHeader)) {
    return { ok: false, status: 401, error: "Unauthorized" };
  }
  return { ok: true };
}
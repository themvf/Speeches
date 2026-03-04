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
  if (!value) {
    return null;
  }
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? null : d;
}

export function normalizeText(value: unknown): string {
  return String(value ?? "")
    .replace(/\s+/g, " ")
    .trim();
}
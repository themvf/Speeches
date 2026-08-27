import { neon } from "@neondatabase/serverless";

import type {
  MarketRatesCreditData,
  MarketRatesCreditGroup,
  MarketRatesCreditPoint,
} from "./types.ts";

interface SnapshotRow {
  series_id: string;
  source_group: MarketRatesCreditGroup;
  observation_date: string;
  value: number;
}

function databaseUrl(): string {
  const url = String(process.env.DATABASE_URL ?? "").trim();
  if (!url) throw new Error("DATABASE_URL env var is not set");
  return url;
}

async function ensureRatesCreditSnapshotSchema() {
  const sql = neon(databaseUrl());
  await sql`
    CREATE TABLE IF NOT EXISTS rates_credit_daily_snapshots (
      series_id       TEXT NOT NULL,
      source_group    TEXT NOT NULL,
      observation_date DATE NOT NULL,
      value           DOUBLE PRECISION NOT NULL,
      source          TEXT NOT NULL DEFAULT 'FRED',
      captured_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
      PRIMARY KEY (series_id, observation_date)
    )
  `;
  await sql`CREATE INDEX IF NOT EXISTS rates_credit_snapshots_date ON rates_credit_daily_snapshots (observation_date DESC)`;
  return sql;
}

export function snapshotRowsFromData(data: MarketRatesCreditData): SnapshotRow[] {
  return [data.treasuryCurve, data.realYields, data.investmentGrade, data.highYield]
    .flat()
    .map((metric) => ({
      series_id: metric.fredSeriesId,
      source_group: metric.group,
      observation_date: metric.observationDate,
      value: metric.value,
    }));
}

export async function persistRatesCreditSnapshots(data: MarketRatesCreditData): Promise<number> {
  const rows = snapshotRowsFromData(data);
  if (!rows.length) return 0;
  const sql = await ensureRatesCreditSnapshotSchema();
  await sql`
    INSERT INTO rates_credit_daily_snapshots (
      series_id,
      source_group,
      observation_date,
      value,
      source,
      captured_at
    )
    SELECT
      row.series_id,
      row.source_group,
      row.observation_date::date,
      row.value,
      'FRED',
      now()
    FROM jsonb_to_recordset(${JSON.stringify(rows)}::jsonb) AS row(
      series_id TEXT,
      source_group TEXT,
      observation_date TEXT,
      value DOUBLE PRECISION
    )
    ON CONFLICT (series_id, observation_date) DO UPDATE SET
      source_group = EXCLUDED.source_group,
      value = EXCLUDED.value,
      captured_at = now()
  `;
  return rows.length;
}

export async function loadRatesCreditHistory(
  seriesIds: string[],
  years = 10,
): Promise<Record<string, MarketRatesCreditPoint[]>> {
  if (!seriesIds.length) return {};
  const sql = await ensureRatesCreditSnapshotSchema();
  const safeYears = Math.max(1, Math.min(20, Math.round(years)));
  const rows = await sql`
    SELECT series_id, observation_date::text AS observation_date, value
    FROM rates_credit_daily_snapshots
    WHERE series_id = ANY(${seriesIds})
      AND observation_date >= current_date - (${safeYears} * interval '1 year')
    ORDER BY series_id, observation_date
  ` as unknown as Array<{ series_id: string; observation_date: string; value: number | string }>;

  const history: Record<string, MarketRatesCreditPoint[]> = {};
  for (const row of rows) {
    const value = Number(row.value);
    if (!Number.isFinite(value)) continue;
    (history[row.series_id] ??= []).push({ date: row.observation_date, value });
  }
  return history;
}

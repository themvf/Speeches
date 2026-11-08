import { neon } from "@neondatabase/serverless";
import { normalizePortfoliosPayload } from "@/lib/portfolio-calculations";
import type { PortfoliosPayload } from "@/lib/portfolio-types";

function getSql() {
  const url = process.env.DATABASE_URL;
  if (!url) throw new Error("DATABASE_URL is not configured.");
  return neon(url);
}

let schemaPromise: Promise<void> | null = null;

export async function ensurePortfolioSchema(): Promise<void> {
  if (!schemaPromise) {
    const sql = getSql();
    schemaPromise = sql`
      CREATE TABLE IF NOT EXISTS user_portfolios (
        user_id TEXT PRIMARY KEY,
        portfolios JSONB NOT NULL DEFAULT '[]'::jsonb,
        client_updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
      )
    `.then(() => undefined).catch((error) => {
      schemaPromise = null;
      throw error;
    });
  }
  await schemaPromise;
}

export async function loadUserPortfolios(userId: string): Promise<PortfoliosPayload> {
  await ensurePortfolioSchema();
  const sql = getSql();
  const rows = await sql`
    SELECT portfolios, client_updated_at
    FROM user_portfolios
    WHERE user_id = ${userId}
    LIMIT 1
  ` as unknown as Array<{ portfolios: unknown; client_updated_at: string | Date }>;
  if (!rows[0]) return { portfolios: [], updatedAt: new Date(0).toISOString() };
  return normalizePortfoliosPayload({
    portfolios: rows[0].portfolios,
    updatedAt: new Date(rows[0].client_updated_at).toISOString(),
  });
}

export async function saveUserPortfolios(userId: string, value: unknown): Promise<PortfoliosPayload> {
  const payload = normalizePortfoliosPayload(value);
  const updatedAt = new Date(payload.updatedAt);
  if (!Number.isFinite(updatedAt.getTime())) throw new Error("Portfolio update timestamp is invalid.");
  await ensurePortfolioSchema();
  const sql = getSql();
  await sql`
    INSERT INTO user_portfolios (user_id, portfolios, client_updated_at, updated_at)
    VALUES (${userId}, ${JSON.stringify(payload.portfolios)}::jsonb, ${updatedAt.toISOString()}, now())
    ON CONFLICT (user_id) DO UPDATE SET
      portfolios = EXCLUDED.portfolios,
      client_updated_at = EXCLUDED.client_updated_at,
      updated_at = now()
    WHERE user_portfolios.client_updated_at <= EXCLUDED.client_updated_at
  `;
  return loadUserPortfolios(userId);
}

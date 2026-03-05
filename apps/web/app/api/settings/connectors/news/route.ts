import { loadNewsConnectorSettings, saveNewsConnectorSettings } from "@/lib/server/data-store";
import { createRequestId, fail, normalizeText, ok } from "@/lib/server/api-utils";

export const runtime = "nodejs";

function clampInt(value: unknown, fallback: number, minValue: number, maxValue: number): number {
  const parsed = Number.parseInt(String(value ?? ""), 10);
  const n = Number.isFinite(parsed) ? parsed : fallback;
  return Math.max(minValue, Math.min(maxValue, n));
}

function normalizeSortBy(value: unknown): string {
  const raw = normalizeText(value || "publishedAt");
  return ["publishedAt", "relevancy", "popularity"].includes(raw) ? raw : "publishedAt";
}

export async function GET() {
  const requestId = createRequestId();

  try {
    const settings = await loadNewsConnectorSettings();
    return ok(settings, requestId);
  } catch (error) {
    return fail(
      `Failed to load connector settings: ${error instanceof Error ? error.message : "Unknown error"}`,
      "NEWS_SETTINGS_LOAD_FAILED",
      500,
      requestId
    );
  }
}

export async function PUT(request: Request) {
  const requestId = createRequestId();

  try {
    let body: Record<string, unknown> = {};
    try {
      body = (await request.json()) as Record<string, unknown>;
    } catch {
      body = {};
    }

    const payload = {
      updated_at: "",
      query: normalizeText(body.query),
      lookback_days: clampInt(body.lookback_days, 7, 1, 30),
      max_pages: clampInt(body.max_pages, 4, 1, 10),
      page_size: clampInt(body.page_size, 50, 10, 100),
      target_count: clampInt(body.target_count, 100, 10, 500),
      sort_by: normalizeSortBy(body.sort_by),
      organization_label: normalizeText(body.organization_label || "Financial News") || "Financial News",
      domains: normalizeText(body.domains),
      exclude_domains: normalizeText(body.exclude_domains),
      tags_csv: normalizeText(body.tags_csv)
    };

    const saved = await saveNewsConnectorSettings(payload);
    return ok(saved, requestId);
  } catch (error) {
    return fail(
      `Failed to save connector settings: ${error instanceof Error ? error.message : "Unknown error"}`,
      "NEWS_SETTINGS_SAVE_FAILED",
      500,
      requestId
    );
  }
}

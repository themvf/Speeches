import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import { fetchFredSeriesPoints } from "@/lib/server/fred-macro";
import {
  alignAsOf,
  attributeWindow,
  decompose,
  latestDifference,
  RATE_TRANSMISSION_SERIES,
  type RateTransmissionData,
} from "@/lib/rate-transmission";
import type { MarketMacroPoint } from "@/lib/server/types";

export const runtime = "nodejs";
export const revalidate = 3600;

const WINDOW_DAYS = { "1M": 30, "3M": 91, "6M": 182, "12M": 365 } as const;

export async function GET() {
  const requestId = createRequestId();
  try {
    const settled = await Promise.allSettled(
      RATE_TRANSMISSION_SERIES.map((series) => fetchFredSeriesPoints(series.id, { limit: 1_500 })),
    );
    const series = new Map<string, MarketMacroPoint[]>();
    const warnings: string[] = [];
    settled.forEach((result, index) => {
      const definition = RATE_TRANSMISSION_SERIES[index];
      if (result.status === "fulfilled") series.set(definition.id, result.value);
      else warnings.push(`${definition.label} (${definition.id}) unavailable: ${result.reason instanceof Error ? result.reason.message : "upstream request failed"}`);
    });
    if (!series.size) {
      const firstFailure = settled.find((result) => result.status === "rejected");
      throw firstFailure?.status === "rejected"
        ? firstFailure.reason
        : new Error("FRED returned no rate-transmission series.");
    }

    const mortgage = alignAsOf(series.get("DGS10") ?? [], series.get("MORTGAGE30US") ?? []);
    const dates = [...series.values()].flatMap((points) => points.at(-1)?.date ?? []).sort();
    const data: RateTransmissionData = {
      asOf: decompose(mortgage)?.observationDate ?? dates.at(-1) ?? "",
      generatedAt: new Date().toISOString(),
      levels: {
        mortgage: decompose(mortgage),
        corporate: {
          available: false,
          reason: "Moody's-derived corporate yield series are omitted because their FRED notes prohibit redistribution without prior written consent.",
        },
      },
      curve: {
        shortTail: latestDifference(series.get("DGS3MO") ?? [], series.get("DGS2") ?? []),
        belly: latestDifference(series.get("DGS2") ?? [], series.get("DGS10") ?? []),
        longTail: latestDifference(series.get("DGS10") ?? [], series.get("DGS30") ?? []),
        policyGap: latestDifference(series.get("DFF") ?? [], series.get("DGS2") ?? []),
      },
      attribution: Object.entries(WINDOW_DAYS).map(([window, days]) => ({
        window: window as keyof typeof WINDOW_DAYS,
        mortgage: attributeWindow(mortgage, days),
      })),
      warnings,
      sources: RATE_TRANSMISSION_SERIES
        .filter((definition) => series.has(definition.id))
        .map((definition) => ({
          seriesId: definition.id,
          label: definition.label,
          url: `https://fred.stlouisfed.org/series/${definition.id}`,
        })),
    };
    return ok(data, requestId);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to load rate transmission data.";
    const missingKey = message.includes("FRED_API_KEY");
    return fail(message, missingKey ? "FRED_NOT_CONFIGURED" : "FRED_UPSTREAM_ERROR", missingKey ? 503 : 502, requestId);
  }
}

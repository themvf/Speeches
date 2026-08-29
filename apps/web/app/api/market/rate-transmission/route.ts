import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import { fetchFredSeriesPoints } from "@/lib/server/fred-macro";
import { fetchFedCreditResearch, FED_CREDIT_RESEARCH_SOURCE_URL } from "@/lib/server/fed-credit-research";
import {
  alignAsOf,
  attributeWindow,
  attributeYearToDate,
  buildCreditResearch,
  crossCorrelate,
  decompose,
  latestDifference,
  RATE_TRANSMISSION_SERIES,
  rollingOls,
  type RateTransmissionData,
} from "@/lib/rate-transmission";
import type { MarketMacroPoint } from "@/lib/server/types";

export const runtime = "nodejs";
export const revalidate = 3600;

const WINDOW_DAYS = { "1M": 30, "3M": 91, "6M": 182, "12M": 365 } as const;

export async function GET() {
  const requestId = createRequestId();
  try {
    const [settled, creditResearchResult] = await Promise.all([
      Promise.allSettled(RATE_TRANSMISSION_SERIES.map((series) => fetchFredSeriesPoints(series.id, { limit: 1_500 }))),
      Promise.allSettled([fetchFedCreditResearch()]),
    ]);
    const series = new Map<string, MarketMacroPoint[]>();
    const warnings: string[] = [];
    settled.forEach((result, index) => {
      const definition = RATE_TRANSMISSION_SERIES[index];
      if (result.status === "fulfilled") series.set(definition.id, result.value);
      else warnings.push(`${definition.label} (${definition.id}) unavailable: ${result.reason instanceof Error ? result.reason.message : "upstream request failed"}`);
    });
    const creditResearchPoints = creditResearchResult[0].status === "fulfilled" ? creditResearchResult[0].value : [];
    if (creditResearchResult[0].status === "rejected") {
      warnings.push(`Federal Reserve corporate-credit research unavailable: ${creditResearchResult[0].reason instanceof Error ? creditResearchResult[0].reason.message : "upstream request failed"}`);
    }
    if (!series.size && !creditResearchPoints.length) {
      const firstFailure = settled.find((result) => result.status === "rejected");
      throw firstFailure?.status === "rejected"
        ? firstFailure.reason
        : new Error("FRED returned no rate-transmission series.");
    }

    const mortgage = alignAsOf(series.get("DGS10") ?? [], series.get("MORTGAGE30US") ?? []);
    const dates = [...series.values()].flatMap((points) => points.at(-1)?.date ?? []).concat(creditResearchPoints.at(-1)?.date ?? []).sort();
    const sources: RateTransmissionData["sources"] = RATE_TRANSMISSION_SERIES
      .filter((definition) => series.has(definition.id))
      .map((definition) => ({
        seriesId: definition.id,
        label: definition.label,
        url: `https://fred.stlouisfed.org/series/${definition.id}`,
      }));
    if (creditResearchPoints.length) sources.push({
      seriesId: "FED_EBP",
      label: "Federal Reserve corporate-credit research",
      url: FED_CREDIT_RESEARCH_SOURCE_URL,
    });
    const data: RateTransmissionData = {
      asOf: decompose(mortgage)?.observationDate ?? dates.at(-1) ?? "",
      generatedAt: new Date().toISOString(),
      levels: {
        mortgage: decompose(mortgage),
        corporate: {
          available: false,
          reason: "Moody's Aaa/Baa series and ICE BofA rating-specific and High Yield OAS are omitted because this public deployment does not hold redistribution rights.",
        },
      },
      curve: {
        shortTail: latestDifference(series.get("DGS3MO") ?? [], series.get("DGS2") ?? []),
        belly: latestDifference(series.get("DGS2") ?? [], series.get("DGS10") ?? []),
        longTail: latestDifference(series.get("DGS10") ?? [], series.get("DGS30") ?? []),
        policyGap: latestDifference(series.get("DFF") ?? [], series.get("DGS2") ?? []),
      },
      attribution: [
        ...Object.entries(WINDOW_DAYS).map(([window, days]) => ({
          window: window as keyof typeof WINDOW_DAYS,
          mortgage: attributeWindow(mortgage, days),
        })),
        { window: "YTD" as const, mortgage: attributeYearToDate(mortgage) },
      ],
      passThrough: { mortgage: rollingOls(mortgage, 52, 30) },
      leadLag: { mortgageTreasury: crossCorrelate(mortgage, 4, 30) },
      creditResearch: buildCreditResearch(creditResearchPoints, series.get("DGS10") ?? []),
      warnings,
      sources,
    };
    return ok(data, requestId);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to load rate transmission data.";
    const missingKey = message.includes("FRED_API_KEY");
    return fail(message, missingKey ? "FRED_NOT_CONFIGURED" : "FRED_UPSTREAM_ERROR", missingKey ? 503 : 502, requestId);
  }
}

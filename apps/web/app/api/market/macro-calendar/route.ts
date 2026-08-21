import { createRequestId, fail, ok, toInt } from "@/lib/server/api-utils";
import {
  DEFAULT_CALENDAR_HORIZON_DAYS,
  fetchFredReleaseCalendar,
  MAX_CALENDAR_HORIZON_DAYS,
  MIN_CALENDAR_HORIZON_DAYS,
} from "@/lib/server/fred-calendar";

export const runtime = "nodejs";
export const revalidate = 21600;

export async function GET(request: Request) {
  const requestId = createRequestId();
  const days = toInt(
    new URL(request.url).searchParams.get("days"),
    DEFAULT_CALENDAR_HORIZON_DAYS,
    MIN_CALENDAR_HORIZON_DAYS,
    MAX_CALENDAR_HORIZON_DAYS,
  );
  try {
    return ok(await fetchFredReleaseCalendar(days), requestId);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to load the FRED release calendar.";
    const missingKey = message.includes("FRED_API_KEY");
    return fail(
      message,
      missingKey ? "FRED_NOT_CONFIGURED" : "FRED_UPSTREAM_ERROR",
      missingKey ? 503 : 502,
      requestId,
    );
  }
}

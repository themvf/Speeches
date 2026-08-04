/**
 * Shared client-side fetch utility with content-type validation.
 *
 * Validates that the server responds with JSON before attempting to parse,
 * producing a clear error message instead of a cryptic "Unexpected token '<'"
 * when the server returns HTML (e.g., Vercel 500 page, auth redirect).
 */

interface ApiEnvelope<T> {
  ok: boolean;
  data?: T;
  error?: string;
  code?: string;
  request_id?: string;
}

export class FetchJsonError extends Error {
  public readonly status: number;
  public readonly code: string;

  constructor(message: string, status: number, code = "FETCH_FAILED") {
    super(message);
    this.name = "FetchJsonError";
    this.status = status;
    this.code = code;
  }
}

/**
 * Fetch a JSON API endpoint and unwrap the standard `{ ok, data, error }` envelope.
 *
 * Checks Content-Type before calling `res.json()` to avoid cryptic parse errors
 * when the server returns HTML (timeout pages, auth redirects, 404s).
 */
export async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...init,
    cache: "no-store",
    headers: {
      "Content-Type": "application/json",
      ...init?.headers
    }
  });

  const contentType = res.headers.get("content-type") || "";
  if (!contentType.includes("application/json")) {
    throw new FetchJsonError(
      `Server returned ${contentType || "unknown content-type"} instead of JSON (HTTP ${res.status}). ` +
        "The API may be temporarily unavailable.",
      res.status,
      "NOT_JSON"
    );
  }

  let payload: ApiEnvelope<T>;
  try {
    payload = (await res.json()) as ApiEnvelope<T>;
  } catch {
    throw new FetchJsonError(
      `Failed to parse JSON response (HTTP ${res.status}).`,
      res.status,
      "PARSE_FAILED"
    );
  }

  if (!res.ok || !payload?.ok || !payload.data) {
    throw new FetchJsonError(
      payload?.error || `Request failed (${res.status})`,
      res.status,
      payload?.code || "API_ERROR"
    );
  }

  return payload.data;
}

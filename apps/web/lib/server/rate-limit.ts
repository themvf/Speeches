import { Ratelimit } from "@upstash/ratelimit";
import { Redis } from "@upstash/redis";

type Duration = Parameters<typeof Ratelimit.slidingWindow>[1];

let _redis: Redis | null | undefined;

function getRedis(): Redis | null {
  if (_redis !== undefined) return _redis;
  const url = process.env.UPSTASH_REDIS_REST_URL ?? process.env.KV_REST_API_URL;
  const token = process.env.UPSTASH_REDIS_REST_TOKEN ?? process.env.KV_REST_API_TOKEN;
  _redis = url && token ? new Redis({ url, token }) : null;
  return _redis;
}

function makeLimiter(prefix: string, requests: number, window: Duration): Ratelimit | null {
  const r = getRedis();
  if (!r) return null;
  return new Ratelimit({
    redis: r,
    limiter: Ratelimit.slidingWindow(requests, window),
    prefix,
    analytics: false,
  });
}

// Per-route limiters — created once per process (cold-start safe: state lives in Redis)
let searchLimiter: Ratelimit | null | undefined;
let feedLimiter: Ratelimit | null | undefined;
let generateIpLimiter: Ratelimit | null | undefined;
let generateGlobalLimiter: Ratelimit | null | undefined;

export function getSearchLimiter() {
  if (searchLimiter === undefined) searchLimiter = makeLimiter("rl:search", 20, "1 m");
  return searchLimiter;
}

export function getFeedLimiter() {
  if (feedLimiter === undefined) feedLimiter = makeLimiter("rl:feed", 30, "1 m");
  return feedLimiter;
}

export function getGenerateIpLimiter() {
  if (generateIpLimiter === undefined) generateIpLimiter = makeLimiter("rl:generate:ip", 3, "1 m");
  return generateIpLimiter;
}

export function getGenerateGlobalLimiter() {
  if (generateGlobalLimiter === undefined) generateGlobalLimiter = makeLimiter("rl:generate:global", 10, "1 m");
  return generateGlobalLimiter;
}

export function getClientIp(headers: Headers): string {
  return headers.get("x-forwarded-for")?.split(",")[0]?.trim() ?? "anonymous";
}

// Returns true if the request is rate-limited. Fails open on Redis error so a
// Redis outage never blocks legitimate traffic.
export async function isRateLimited(limiter: Ratelimit | null, identifier: string): Promise<boolean> {
  if (!limiter) return false;
  try {
    const { success } = await limiter.limit(identifier);
    return !success;
  } catch (err) {
    console.error("[rate-limit] Redis error, failing open:", err);
    return false;
  }
}

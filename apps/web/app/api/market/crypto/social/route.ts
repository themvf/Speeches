import { neon } from "@neondatabase/serverless";
import { ok, fail } from "@/lib/server/api-utils";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

// Read-only: opening the dashboard never calls TwitterAPI.io.
export async function GET(request: Request) {
  const coin = new URL(request.url).searchParams.get("coin") ?? "ZCAT";
  if (!["ZCAT", "ZEC"].includes(coin)) return fail("Unknown coin", "INVALID_COIN", 400);
  if (!process.env.DATABASE_URL) return ok({ status: "not_configured" });
  const sql = neon(process.env.DATABASE_URL);
  try {
    const exists = await sql`SELECT to_regclass('public.crypto_social_pilot') AS relation`;
    if (!exists[0]?.relation) return ok({ status: "not_started" });
    const [pilot, daily, accounts, edges, posts] = await Promise.all([
      sql`SELECT p.*, (SELECT sum(estimated_credits) FROM crypto_social_requests) AS estimated_credits,
        (SELECT count(*) FROM crypto_social_requests WHERE status IN ('reserved','uncertain')) AS outstanding
        FROM crypto_social_pilot p WHERE id='zcat-zec-v1'`,
      sql`SELECT (w.start_at AT TIME ZONE 'UTC')::date::text AS day,
        count(DISTINCT p.id)::int AS posts, count(DISTINCT p.author_id)::int AS authors,
        count(DISTINCT p.id) FILTER (WHERE p.kind='original')::int AS originals,
        count(DISTINCT p.id) FILTER (WHERE p.kind='reply')::int AS replies,
        count(DISTINCT p.id) FILTER (WHERE p.kind='quote')::int AS quotes,
        count(DISTINCT p.id) FILTER (WHERE p.kind='repost')::int AS reposts,
        count(DISTINCT w.id) FILTER (WHERE w.status='search_exhausted')::int AS exhausted,
        count(DISTINCT w.id) FILTER (WHERE w.pages>0)::int AS searched,
        count(DISTINCT w.id)::int AS windows
        FROM crypto_social_windows w LEFT JOIN crypto_social_matches m ON m.window_id=w.id
        LEFT JOIN crypto_social_posts p ON p.id=m.post_id WHERE w.coin=${coin}
        GROUP BY 1 ORDER BY 1`,
      sql`WITH matched AS (
        SELECT DISTINCT p.* FROM crypto_social_posts p JOIN crypto_social_matches m ON m.post_id=p.id
        JOIN crypto_social_windows w ON w.id=m.window_id WHERE w.coin=${coin}
      ), metrics AS (
        SELECT DISTINCT ON(s.post_id) s.* FROM crypto_social_snapshots s
        JOIN matched m ON m.id=s.post_id ORDER BY s.post_id,s.observed_at DESC,s.request_id DESC
      ), amplification AS (
        SELECT e.target_id,count(DISTINCT e.source_id)::int AS amplifiers FROM crypto_social_edges e
        JOIN matched m ON m.id=e.post_id WHERE e.kind IN ('quote','repost') AND e.source_id<>e.target_id GROUP BY e.target_id
      ) SELECT a.id,a.handle,a.followers,count(*)::int AS posts,
        count(DISTINCT (m.posted_at AT TIME ZONE 'UTC')::date)::int AS active_days,
        round(count(*)::numeric/7,2)::float AS posts_per_day,
        percentile_cont(0.5) WITHIN GROUP(ORDER BY s.likes) AS median_likes,
        coalesce(max(amp.amplifiers),0)::int AS amplifiers
        FROM matched m JOIN crypto_social_accounts a ON a.id=m.author_id
        LEFT JOIN metrics s ON s.post_id=m.id LEFT JOIN amplification amp ON amp.target_id=a.id
        GROUP BY a.id ORDER BY amplifiers DESC,posts DESC,a.id LIMIT 20`,
      sql`WITH matched AS (
        SELECT DISTINCT m.post_id FROM crypto_social_matches m JOIN crypto_social_windows w ON w.id=m.window_id WHERE w.coin=${coin}
      ) SELECT e.source_id,e.target_id,e.kind,count(*)::int AS weight,
        coalesce(a.handle,e.source_id) AS source,coalesce(b.handle,e.target_id) AS target,
        min(p.url) AS evidence
        FROM crypto_social_edges e JOIN matched m ON m.post_id=e.post_id JOIN crypto_social_posts p ON p.id=e.post_id
        LEFT JOIN crypto_social_accounts a ON a.id=e.source_id LEFT JOIN crypto_social_accounts b ON b.id=e.target_id
        WHERE e.source_id<>e.target_id GROUP BY e.source_id,e.target_id,e.kind,a.handle,b.handle
        ORDER BY weight DESC,e.source_id,e.target_id,e.kind LIMIT 60`,
      sql`SELECT DISTINCT p.id,p.text,p.url,p.posted_at,a.handle FROM crypto_social_posts p
        JOIN crypto_social_accounts a ON a.id=p.author_id JOIN crypto_social_matches m ON m.post_id=p.id
        JOIN crypto_social_windows w ON w.id=m.window_id WHERE w.coin=${coin}
        ORDER BY p.posted_at DESC,p.id LIMIT 20`,
    ]);
    return ok({ status: pilot.length ? "ready" : "not_started", coin, pilot: pilot[0], daily, accounts, edges, posts });
  } catch {
    return fail("Crypto research data is temporarily unavailable", "SOCIAL_READ_FAILED", 503);
  }
}

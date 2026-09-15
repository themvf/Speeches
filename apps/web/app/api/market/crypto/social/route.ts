import { neon } from "@neondatabase/serverless";
import { ok, fail } from "@/lib/server/api-utils";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

// Read-only: opening the dashboard never calls TwitterAPI.io.
export async function GET(request: Request) {
  const coin = new URL(request.url).searchParams.get("coin") ?? "ZCAT";
  if (!["ZCAT", "ZEC", "PONS", "DPONS", "STANDARD"].includes(coin)) return fail("Unknown coin", "INVALID_COIN", 400);
  if (!process.env.DATABASE_URL) return ok({ status: "not_configured" });
  const sql = neon(process.env.DATABASE_URL);
  try {
    const exists = await sql`SELECT to_regclass('public.crypto_social_pilot') AS relation`;
    if (!exists[0]?.relation) return ok({ status: "not_started" });
    const [pilot, daily, accounts, edges, posts] = await Promise.all([
      sql`SELECT p.*, (SELECT sum(estimated_credits) FROM crypto_social_requests r WHERE to_jsonb(r)->>'endpoint' IS DISTINCT FROM 'historical_search') AS estimated_credits,
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
        LEFT JOIN crypto_social_posts p ON p.id=m.post_id WHERE w.coin=${coin} AND w.query<>'timeline text match'
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
    const trackingSchema = await sql`SELECT to_regclass('public.crypto_social_account_metrics') AS relation`;
    let tracking = null;
    if (trackingSchema[0]?.relation) {
      const [campaign, candidates, history, bioChanges, coverage, ledger, bioMatches] = await Promise.all([
        sql`SELECT * FROM crypto_social_tracking WHERE id='zcat-zec-v1'`,
        sql`SELECT * FROM crypto_social_account_metrics WHERE coin=${coin} ORDER BY tracked DESC,participants DESC,id`,
        sql`SELECT h.account_id,h.observed_at,h.followers,h.available FROM crypto_social_profile_history h
          JOIN crypto_social_candidates c ON c.account_id=h.account_id AND c.coin=${coin}
          WHERE c.tracked AND h.source IN ('daily_profile','daily_profile_missing')
          AND h.observed_at>=now()-interval '31 days' ORDER BY h.observed_at,h.account_id`,
        sql`WITH versions AS (
          SELECT h.account_id,h.bio,h.observed_at,
            lag(h.bio) OVER(PARTITION BY h.account_id ORDER BY h.observed_at,h.request_id) AS previous_bio
          FROM crypto_social_profile_history h JOIN crypto_social_candidates c
          ON c.account_id=h.account_id AND c.coin=${coin} WHERE h.bio IS NOT NULL AND h.available
        ) SELECT v.*,a.handle FROM versions v JOIN crypto_social_accounts a ON a.id=v.account_id
          WHERE previous_bio IS NOT NULL AND bio IS DISTINCT FROM previous_bio
          ORDER BY observed_at DESC LIMIT 30`,
        sql`SELECT v.*,a.handle FROM crypto_social_account_coverage v JOIN crypto_social_accounts a ON a.id=v.account_id
          JOIN crypto_social_candidates c ON c.account_id=v.account_id AND c.coin=${coin}
          ORDER BY end_at DESC,request_id DESC LIMIT 40`,
        sql`SELECT coalesce(parameters->>'allocation','initial_search') AS allocation,
          sum(reserved_credits)::int AS reserved_credits,sum(estimated_credits)::int AS estimated_credits,
          count(*)::int AS requests FROM crypto_social_requests GROUP BY 1 ORDER BY 1`,
        sql`SELECT m.* FROM crypto_social_profile_matches m JOIN (
          SELECT DISTINCT ON(account_id) account_id,request_id FROM crypto_social_profile_history
          ORDER BY account_id,observed_at DESC,request_id DESC
        ) h USING(account_id,request_id) WHERE m.coin=${coin}`,
      ]);
      tracking = { campaign: campaign[0] ?? null, accounts: candidates, history, bioChanges, coverage, ledger, bioMatches,
        keywordSearchStatus: 'pending_provider_page_bound', lookbackDays: 28 };
    }
    let history = null;
    const historySchema = await sql`SELECT to_regclass('public.crypto_social_history_campaign') AS relation`;
    if (["ZCAT","PONS","DPONS"].includes(coin) && historySchema[0]?.relation) {
      const campaignId=coin==='DPONS'?'dpons-july-2026':coin==='PONS'?'pons-july-2026':'zcat-july-2026';
      const [campaign, earliest, coverage] = await Promise.all([
        sql`SELECT h.*,(SELECT sum(r.estimated_credits) FROM crypto_social_requests r
          WHERE r.endpoint='historical_search' AND r.parameters->>'campaign'=h.id) AS estimated_credits FROM crypto_social_history_campaign h WHERE h.id=${campaignId}`,
        sql`SELECT DISTINCT p.id,p.text,p.url,p.posted_at,a.handle FROM crypto_social_history_windows h
          JOIN crypto_social_matches m ON m.window_id=h.window_id JOIN crypto_social_posts p ON p.id=m.post_id
          JOIN crypto_social_accounts a ON a.id=p.author_id WHERE h.campaign_id=${campaignId}
          ORDER BY p.posted_at,p.id LIMIT 30`,
        sql`SELECT count(*)::int AS windows,count(*) FILTER(WHERE w.pages>0)::int AS searched,
          count(*) FILTER(WHERE w.status='search_exhausted')::int AS exhausted
          FROM crypto_social_history_windows h JOIN crypto_social_windows w ON w.id=h.window_id
          WHERE h.campaign_id=${campaignId}`,
      ]);
      history = campaign.length ? { campaign: campaign[0], earliest, coverage: coverage[0] } : null;
    }
    let rolling = null;
    const rollingSchema = await sql`SELECT to_regclass('public.crypto_rolling_coins') AS relation`;
    if (rollingSchema[0]?.relation) {
      const rows = await sql`SELECT c.used_credits,c.credit_limit,p.end_at,
        (SELECT max(r.requested_at) FROM crypto_rolling_calls k JOIN crypto_social_requests r ON r.id=k.request_id
         WHERE k.campaign_id=c.campaign_id AND k.coin=c.coin AND r.status='saved') AS last_saved,
        (SELECT count(*)::int FROM crypto_rolling_windows k JOIN crypto_social_windows w ON w.id=k.window_id
         WHERE k.campaign_id=c.campaign_id AND w.coin=c.coin AND w.status IN ('pending','partial')) AS unfinished
        FROM crypto_rolling_coins c JOIN crypto_rolling_campaign p ON p.id=c.campaign_id
        WHERE c.coin=${coin} ORDER BY p.started_at DESC LIMIT 1`;
      rolling = rows[0] ?? null;
    }
    return ok({ rolling, history, tracking, status: pilot.length ? "ready" : "not_started", coin, pilot: pilot[0], daily, accounts, edges, posts });
  } catch {
    return fail("Crypto research data is temporarily unavailable", "SOCIAL_READ_FAILED", 503);
  }
}

import {neon} from '@neondatabase/serverless';
import {ok,fail} from '@/lib/server/api-utils';
import {IMPACT_VERSION,type ImpactBaseline,type ImpactRow} from '@/lib/crypto-impact';
export const dynamic='force-dynamic';
export const runtime='nodejs';
const COINS=['ZCAT','ZEC','PONS','DPONS','STANDARD'];
// Read-only: aggregates the immutable event study written by crypto_event_study.py. Never calls a provider.
export async function GET(request:Request){
 const coin=new URL(request.url).searchParams.get('coin')??'ALL';
 if(coin!=='ALL'&&!COINS.includes(coin))return fail('Unknown coin','INVALID_COIN',400);
 const empty={status:'not_started',version:IMPACT_VERSION,coin,accounts:[] as ImpactRow[],baselines:[] as ImpactBaseline[],events:0,episodes:0};
 if(!process.env.DATABASE_URL)return ok({...empty,status:'not_configured'});
 const sql=neon(process.env.DATABASE_URL);
 try{
  const exists=await sql`SELECT to_regclass('public.crypto_price_events') AS relation`;
  if(!exists[0]?.relation)return ok(empty);
  const filter=coin==='ALL'?COINS:[coin];
  const [baselines,accounts,totals]=await Promise.all([
   // Baseline: every archived hour's 24h-forward move on the pinned pool, so an account's median can be read against the coin's own drift.
   sql`SELECT s.coin,s.id AS source_id,count(*)::int AS hours,min(h.hour) AS first_hour,max(n.hour) AS last_hour,
     percentile_cont(0.5) WITHIN GROUP(ORDER BY n.close/h.close-1)::float AS median_24h,avg((n.close>h.close)::int)::float AS share_up_24h
     FROM crypto_market_sources s JOIN crypto_market_hourly_latest h ON h.source_id=s.id
     JOIN crypto_market_hourly_latest n ON n.source_id=s.id AND n.hour=h.hour+interval '24 hours'
     WHERE s.is_default AND s.coin=ANY(${filter}) GROUP BY s.coin,s.id ORDER BY s.coin`,
   sql`WITH base AS (
     SELECT s.coin,percentile_cont(0.5) WITHIN GROUP(ORDER BY n.close/h.close-1) AS median_24h
     FROM crypto_market_sources s JOIN crypto_market_hourly_latest h ON h.source_id=s.id
     JOIN crypto_market_hourly_latest n ON n.source_id=s.id AND n.hour=h.hour+interval '24 hours' WHERE s.is_default GROUP BY s.coin
   ), ev AS (
     SELECT e.*,p.url,e.price_after_1h/e.price_0-1 AS r1,e.price_after_6h/e.price_0-1 AS r6,e.price_after_24h/e.price_0-1 AS r24,
      e.price_after_24h/e.price_0-1-coalesce(b.median_24h,0) AS x24,
      CASE WHEN e.volume_before_24h>0 AND e.hours_before_24h>=12 AND e.hours_after_24h>=12 THEN e.volume_after_24h/e.volume_before_24h END AS vr
     FROM crypto_price_events e JOIN crypto_social_posts p ON p.id=e.post_id LEFT JOIN base b ON b.coin=e.coin
     WHERE e.version=${IMPACT_VERSION} AND e.episode AND e.coin=ANY(${filter})
   ), best AS (SELECT DISTINCT ON(account_id) account_id,post_id AS best_id,url AS best_url FROM ev ORDER BY account_id,r24 DESC,post_id)
   SELECT ev.account_id,a.handle,a.followers::float AS followers,array_agg(DISTINCT ev.coin ORDER BY ev.coin) AS coins,
    count(*)::int AS episodes,
    (SELECT count(*)::int FROM crypto_price_events t WHERE t.version=${IMPACT_VERSION} AND t.account_id=ev.account_id AND t.coin=ANY(${filter})) AS posts,
    percentile_cont(0.5) WITHIN GROUP(ORDER BY r1)::float AS median_1h,percentile_cont(0.5) WITHIN GROUP(ORDER BY r6)::float AS median_6h,
    percentile_cont(0.5) WITHIN GROUP(ORDER BY r24)::float AS median_24h,percentile_cont(0.5) WITHIN GROUP(ORDER BY x24)::float AS median_excess_24h,
    avg((r24>0)::int)::float AS share_up_24h,avg((x24>0)::int)::float AS share_beat_24h,
    percentile_cont(0.5) WITHIN GROUP(ORDER BY vr)::float AS median_volume_ratio,
    min(ev.posted_at) AS first_at,max(ev.posted_at) AS last_at,max(best.best_id) AS best_post_id,max(best.best_url) AS best_post_url,
    max(r24)::float AS best_return_24h,min(r24)::float AS worst_return_24h
   FROM ev JOIN crypto_social_accounts a ON a.id=ev.account_id LEFT JOIN best ON best.account_id=ev.account_id
   GROUP BY ev.account_id,a.handle,a.followers ORDER BY episodes DESC,ev.account_id LIMIT 500`,
   sql`SELECT count(*)::int AS events,count(*) FILTER(WHERE episode)::int AS episodes FROM crypto_price_events WHERE version=${IMPACT_VERSION} AND coin=ANY(${filter})`,
  ]);
  return ok({status:accounts.length?'ready':'no_events',version:IMPACT_VERSION,coin,accounts:accounts as ImpactRow[],baselines:baselines as ImpactBaseline[],events:totals[0]?.events??0,episodes:totals[0]?.episodes??0,asOf:new Date().toISOString()});
 }catch{return fail('Price impact evidence could not be loaded','IMPACT_READ_FAILED',503);}
}

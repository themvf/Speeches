import {neon} from '@neondatabase/serverless';
import {ok,fail} from '@/lib/server/api-utils';
import {COINS} from '@/lib/crypto-coins';
import {withCdnCache} from '@/lib/server/crypto-ranking-cache';
export const dynamic='force-dynamic';
export const runtime='nodejs';
export type CoinOverview={symbol:string;name:string;label:string;networkLabel:string;archiveStart:string;posts_24h:number;authors_24h:number;posts_7d:number;authors_7d:number;new_authors_7d:number;last_post_at:string|null;price_now:number|null;price_24h_ago:number|null;price_observed_at:string|null;linked_posts:number};
// Read-only landing summary: saved-post activity and the pinned pool's last 24h, per tracked coin.
export async function GET(){
 const base=COINS.map(c=>({symbol:c.symbol,name:c.name,label:c.label,networkLabel:c.networkLabel,archiveStart:c.archiveStart,posts_24h:0,authors_24h:0,posts_7d:0,authors_7d:0,new_authors_7d:0,last_post_at:null,price_now:null,price_24h_ago:null,price_observed_at:null,linked_posts:0} as CoinOverview));
 if(!process.env.DATABASE_URL)return ok({status:'not_configured',coins:base});
 const sql=neon(process.env.DATABASE_URL);
 try{
  const exists=await sql`SELECT to_regclass('public.crypto_social_posts') AS posts,to_regclass('public.crypto_market_hourly') AS hourly,to_regclass('public.crypto_price_events') AS events`;
  if(!exists[0]?.posts)return ok({status:'not_started',coins:base});
  const [activity,prices,linked]=await Promise.all([
   sql`WITH matched AS (SELECT DISTINCT w.coin,p.id,p.author_id,p.posted_at FROM crypto_social_posts p JOIN crypto_social_matches m ON m.post_id=p.id JOIN crypto_social_windows w ON w.id=m.window_id WHERE p.kind<>'repost'),
    firsts AS (SELECT coin,author_id,min(posted_at) AS first_at FROM matched GROUP BY coin,author_id)
    SELECT m.coin,count(*) FILTER(WHERE m.posted_at>=now()-interval '24 hours')::int AS posts_24h,count(DISTINCT m.author_id) FILTER(WHERE m.posted_at>=now()-interval '24 hours')::int AS authors_24h,
     count(*) FILTER(WHERE m.posted_at>=now()-interval '7 days')::int AS posts_7d,count(DISTINCT m.author_id) FILTER(WHERE m.posted_at>=now()-interval '7 days')::int AS authors_7d,
     (SELECT count(*)::int FROM firsts f WHERE f.coin=m.coin AND f.first_at>=now()-interval '7 days') AS new_authors_7d,max(m.posted_at) AS last_post_at
    FROM matched m GROUP BY m.coin`,
   exists[0].hourly?sql`SELECT s.coin,
     (SELECT close FROM crypto_market_hourly_latest h WHERE h.source_id=s.id ORDER BY hour DESC LIMIT 1) AS price_now,
     (SELECT hour FROM crypto_market_hourly_latest h WHERE h.source_id=s.id ORDER BY hour DESC LIMIT 1) AS price_observed_at,
     (SELECT close FROM crypto_market_hourly_latest h WHERE h.source_id=s.id AND h.hour<=(SELECT max(hour) FROM crypto_market_hourly_latest x WHERE x.source_id=s.id)-interval '24 hours' ORDER BY hour DESC LIMIT 1) AS price_24h_ago
     FROM crypto_market_sources s WHERE s.is_default`:Promise.resolve([]),
   exists[0].events?sql`SELECT coin,count(*)::int AS linked FROM crypto_price_events WHERE episode GROUP BY coin`:Promise.resolve([]),
  ]);
  const coins=base.map(c=>{const a=activity.find(r=>r.coin===c.symbol),p=prices.find(r=>r.coin===c.symbol),l=linked.find(r=>r.coin===c.symbol);
   return {...c,posts_24h:a?.posts_24h??0,authors_24h:a?.authors_24h??0,posts_7d:a?.posts_7d??0,authors_7d:a?.authors_7d??0,new_authors_7d:a?.new_authors_7d??0,last_post_at:a?.last_post_at??null,
    price_now:p?.price_now==null?null:Number(p.price_now),price_24h_ago:p?.price_24h_ago==null?null:Number(p.price_24h_ago),price_observed_at:p?.price_observed_at??null,linked_posts:l?.linked??0};});
  return withCdnCache(ok({status:'ready',coins,asOf:new Date().toISOString()}));
 }catch{return fail('Overview could not be loaded','OVERVIEW_READ_FAILED',503);}
}

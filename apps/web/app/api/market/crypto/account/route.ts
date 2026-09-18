import {neon} from '@neondatabase/serverless';
import {ok,fail} from '@/lib/server/api-utils';
import {loadLeaders} from '@/lib/server/crypto-account-query';
import {withCdnCache} from '@/lib/server/crypto-ranking-cache';
export const dynamic='force-dynamic';
export const runtime='nodejs';
// One account across every tracked coin: profile, per-coin activity, roles, price-linked episodes and latest posts.
export async function GET(request:Request){
 const id=new URL(request.url).searchParams.get('id')??'';
 if(!/^[0-9]{1,25}$/.test(id))return fail('Invalid account id','INVALID_ACCOUNT',400);
 if(!process.env.DATABASE_URL)return ok({status:'not_configured',id});
 try{const sql=neon(process.env.DATABASE_URL);
  const [account,profile,activity,posts,summary,weekly]=await Promise.all([
   sql`SELECT id,handle,name,followers::float AS followers,observed_at FROM crypto_social_accounts WHERE id=${id}`,
   sql`SELECT bio,followers::float AS followers,following::float AS following,available,observed_at FROM crypto_social_profile_history WHERE account_id=${id} ORDER BY observed_at DESC,request_id DESC LIMIT 60`.catch(()=>[]),
   sql`SELECT w.coin,count(DISTINCT p.id)::int AS posts,count(DISTINCT p.id) FILTER(WHERE p.kind='original')::int AS originals,min(p.posted_at) AS first_at,max(p.posted_at) AS last_at
     FROM crypto_social_posts p JOIN crypto_social_matches m ON m.post_id=p.id JOIN crypto_social_windows w ON w.id=m.window_id WHERE p.author_id=${id} AND p.kind<>'repost' GROUP BY w.coin ORDER BY w.coin`,
   sql`SELECT DISTINCT ON(p.id) p.id,p.text,p.url,p.posted_at,p.kind,(SELECT array_agg(DISTINCT w.coin ORDER BY w.coin) FROM crypto_social_matches m JOIN crypto_social_windows w ON w.id=m.window_id WHERE m.post_id=p.id) AS coins,
     (SELECT e.price_after_24h/e.price_0-1 FROM crypto_price_events e WHERE e.post_id=p.id ORDER BY e.coin LIMIT 1)::float AS return_24h
     FROM crypto_social_posts p WHERE p.author_id=${id} ORDER BY p.id,p.posted_at DESC LIMIT 200`.then(rows=>rows.sort((a,b)=>String(b.posted_at).localeCompare(String(a.posted_at))).slice(0,30)).catch(()=>[]),
   loadLeaders(sql,id).then(rows=>rows[0]??null),
   // Posts per ISO week per coin, for the interest-over-time strip; reposts excluded like the activity table.
   sql`SELECT w.coin,to_char(date_trunc('week',p.posted_at AT TIME ZONE 'UTC'),'YYYY-MM-DD') AS week,count(DISTINCT p.id)::int AS posts
     FROM crypto_social_posts p JOIN crypto_social_matches m ON m.post_id=p.id JOIN crypto_social_windows w ON w.id=m.window_id WHERE p.author_id=${id} AND p.kind<>'repost' GROUP BY w.coin,week ORDER BY w.coin,week`.catch(()=>[]),
  ]);
  if(!account[0])return fail('Unknown account','UNKNOWN_ACCOUNT',404);
  return withCdnCache(ok({status:'ready',id,account:account[0],latest_profile:profile[0]??null,profile_history:profile,activity,posts,summary,weekly}));
 }catch{return fail('Account evidence could not be loaded','ACCOUNT_READ_FAILED',503);}
}

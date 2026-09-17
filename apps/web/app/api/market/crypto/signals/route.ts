import {neon} from '@neondatabase/serverless';
import {ok,fail} from '@/lib/server/api-utils';
import {COINS,COIN_SYMBOLS} from '@/lib/crypto-coins';
import {boardRows,evaluateSignals,RULES,type BoardCoin,type SignalPost,type WatchedAccount} from '@/lib/crypto-signals';
import {loadLeaders} from '@/lib/server/crypto-account-query';
import {rankLeaders,whyLeader} from '@/lib/crypto-leaders';
import {withCdnCache} from '@/lib/server/crypto-ranking-cache';
import watcherAccounts from '@/lib/crypto-watcher-accounts.json';
export const dynamic='force-dynamic';
export const runtime='nodejs';
// Home page payload: the coin board and the named-rule signals for the last 24 hours. Read-only.
export async function GET(request:Request){
 const hours=Math.min(72,Math.max(6,Number(new URL(request.url).searchParams.get('hours')??24)||24));
 const empty={status:'not_configured',hours,board:[] as ReturnType<typeof boardRows>,signals:[],rules:RULES,watched:0};
 if(!process.env.DATABASE_URL)return ok(empty);
 const sql=neon(process.env.DATABASE_URL);
 try{
  const exists=await sql`SELECT to_regclass('public.crypto_social_posts') AS posts,to_regclass('public.crypto_market_hourly') AS hourly,to_regclass('public.crypto_price_events') AS events`;
  if(!exists[0]?.posts)return ok({...empty,status:'not_started'});
  const windowStart=new Date(Date.now()-hours*3600000).toISOString(),prevStart=new Date(Date.now()-2*hours*3600000).toISOString();
  const leaders=rankLeaders(await loadLeaders(sql)).filter(l=>l.early_coins||l.episodes>=3);
  const watched:WatchedAccount[]=[...watcherAccounts.map(w=>({id:w.id,handle:w.handle,reason:'reviewed whale & volume watcher'})),...leaders.filter(l=>!watcherAccounts.some(w=>w.id===l.account_id)).map(l=>({id:l.account_id,handle:l.handle,reason:whyLeader(l)}))];
  const watchedIds=watched.map(w=>w.id);
  const [activity,prices,linked,posts,firsts,shares]=await Promise.all([
   sql`WITH matched AS (SELECT DISTINCT w.coin,p.id,p.author_id,p.posted_at FROM crypto_social_posts p JOIN crypto_social_matches m ON m.post_id=p.id JOIN crypto_social_windows w ON w.id=m.window_id WHERE p.kind<>'repost' AND p.posted_at>=${prevStart}::timestamptz)
    SELECT coin,count(*) FILTER(WHERE posted_at>=${windowStart}::timestamptz)::int AS posts_24h,count(*) FILTER(WHERE posted_at<${windowStart}::timestamptz)::int AS posts_prev_24h,
     count(DISTINCT author_id) FILTER(WHERE posted_at>=${windowStart}::timestamptz)::int AS authors_24h,
     count(DISTINCT author_id) FILTER(WHERE posted_at>=${windowStart}::timestamptz AND author_id=ANY(${watchedIds}))::int AS watched_posting_24h FROM matched GROUP BY coin`,
   exists[0].hourly?sql`SELECT s.coin,count(*)::int AS hourly_hours,
     (SELECT close FROM crypto_market_hourly_latest h WHERE h.source_id=s.id ORDER BY hour DESC LIMIT 1) AS price_now,
     (SELECT close FROM crypto_market_hourly_latest h WHERE h.source_id=s.id AND h.hour<=(SELECT max(hour) FROM crypto_market_hourly_latest x WHERE x.source_id=s.id)-make_interval(hours=>${hours}) ORDER BY hour DESC LIMIT 1) AS price_24h_ago,
     (SELECT sum(volume) FROM crypto_market_hourly_latest h WHERE h.source_id=s.id AND h.hour>(SELECT max(hour) FROM crypto_market_hourly_latest x WHERE x.source_id=s.id)-make_interval(hours=>${hours})) AS volume_24h,
     (SELECT sum(volume) FROM crypto_market_hourly_latest h WHERE h.source_id=s.id AND h.hour>(SELECT max(hour) FROM crypto_market_hourly_latest x WHERE x.source_id=s.id)-make_interval(hours=>${2*hours}) AND h.hour<=(SELECT max(hour) FROM crypto_market_hourly_latest x WHERE x.source_id=s.id)-make_interval(hours=>${hours})) AS volume_prev_24h
     FROM crypto_market_sources s JOIN crypto_market_hourly_latest hh ON hh.source_id=s.id WHERE s.is_default GROUP BY s.coin,s.id`:Promise.resolve([]),
   exists[0].events?sql`SELECT coin,count(*)::int AS linked FROM crypto_price_events WHERE episode GROUP BY coin`:Promise.resolve([]),
   sql`SELECT p.id,p.author_id,a.handle,p.text,p.posted_at,p.kind,p.url,a.followers::float AS followers,(SELECT array_agg(DISTINCT w.coin ORDER BY w.coin) FROM crypto_social_matches m JOIN crypto_social_windows w ON w.id=m.window_id WHERE m.post_id=p.id) AS coins
     FROM crypto_social_posts p JOIN crypto_social_accounts a ON a.id=p.author_id WHERE p.author_id=ANY(${watchedIds}) AND p.posted_at>=${windowStart}::timestamptz AND p.kind<>'repost' ORDER BY p.posted_at DESC LIMIT 300`,
   sql`SELECT DISTINCT ON (w.coin) w.coin,p.id AS post_id,a.handle,p.author_id,p.posted_at,p.url,c.address FROM crypto_social_posts p JOIN crypto_social_accounts a ON a.id=p.author_id JOIN crypto_social_matches m ON m.post_id=p.id JOIN crypto_social_windows w ON w.id=m.window_id JOIN crypto_social_coins c ON c.symbol=w.coin
     WHERE c.address IS NOT NULL AND position(c.address IN p.text)>0 AND p.kind<>'repost' ORDER BY w.coin,p.posted_at,p.id`,
   sql`WITH matched AS (SELECT DISTINCT w.coin,p.id,p.author_id FROM crypto_social_posts p JOIN crypto_social_matches m ON m.post_id=p.id JOIN crypto_social_windows w ON w.id=m.window_id WHERE p.kind<>'repost' AND p.posted_at>=${windowStart}::timestamptz),
     per AS (SELECT coin,author_id,count(*) AS n FROM matched GROUP BY coin,author_id), ranked AS (SELECT coin,n,row_number() OVER(PARTITION BY coin ORDER BY n DESC) AS rk FROM per)
     SELECT coin,(sum(n) FILTER(WHERE rk<=3)::float/nullif(sum(n),0))::float AS share FROM ranked GROUP BY coin`,
  ]);
  const coins:BoardCoin[]=COINS.map(c=>{const a=activity.find(r=>r.coin===c.symbol),p=prices.find(r=>r.coin===c.symbol),l=linked.find(r=>r.coin===c.symbol);
   return {symbol:c.symbol,name:c.name,price_now:p?.price_now==null?null:Number(p.price_now),price_24h_ago:p?.price_24h_ago==null?null:Number(p.price_24h_ago),volume_24h:p?.volume_24h==null?null:Number(p.volume_24h),volume_prev_24h:p?.volume_prev_24h==null?null:Number(p.volume_prev_24h),posts_24h:a?.posts_24h??0,posts_prev_24h:a?.posts_prev_24h??0,authors_24h:a?.authors_24h??0,watched_posting_24h:a?.watched_posting_24h??0,linked_posts:l?.linked??0,hourly_hours:p?.hourly_hours??0};});
  const contracts=new Map(COINS.filter(c=>c.address).map(c=>[c.symbol,c.address!]));
  const signalPosts:SignalPost[]=posts.map(p=>{const cs=(p.coins as string[]|null)??[];const text=String(p.text);return {id:String(p.id),author_id:String(p.author_id),handle:String(p.handle),text,posted_at:new Date(p.posted_at as string).toISOString(),kind:String(p.kind),url:String(p.url),coins:cs,contract:cs.some(c=>{const a=contracts.get(c);return !!a&&(a.startsWith('0x')?text.toLowerCase().includes(a):text.includes(a));}),followers:p.followers==null?null:Number(p.followers)};});
  const signals=evaluateSignals({coins,posts:signalPosts,watched,firstContract:firsts.map(f=>({coin:String(f.coin),post_id:String(f.post_id),handle:String(f.handle),account_id:String(f.author_id),posted_at:new Date(f.posted_at as string).toISOString(),url:String(f.url)})),windowStart,topAuthorShare:shares.map(s=>({coin:String(s.coin),share:Number(s.share??0)}))});
  return withCdnCache(ok({status:'ready',hours,board:boardRows(coins),signals:signals.slice(0,60),rules:RULES,watched:watched.length,coins:COIN_SYMBOLS,asOf:new Date().toISOString()}),120);
 }catch{return fail('Signals could not be computed','SIGNALS_READ_FAILED',503);}
}

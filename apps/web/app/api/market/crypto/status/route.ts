import {neon} from '@neondatabase/serverless';
import {ok,fail} from '@/lib/server/api-utils';
import {COINS} from '@/lib/crypto-coins';
import {withCdnCache} from '@/lib/server/crypto-ranking-cache';
import watcherAccounts from '@/lib/crypto-watcher-accounts.json';
export const dynamic='force-dynamic';
export const runtime='nodejs';
// Data page payload: coverage per coin, credit ledgers and the registry, in one read-only call.
export async function GET(){
 const registry=COINS.map(c=>({symbol:c.symbol,name:c.name,network:c.networkLabel,address:c.address,archiveStart:c.archiveStart,originFrom:c.originFrom??null,official:c.official,matches:[c.address?'contract':null,...c.words.map(()=>'name or cashtag pattern'),...c.contextWords.map(()=>'word with token context'),...c.exclude.map(()=>'exclusion')].filter(Boolean)}));
 const empty={status:'not_configured',coverage:[],ledgers:[],registry,watchers:watcherAccounts,lastCollection:null};
 if(!process.env.DATABASE_URL)return ok(empty);
 const sql=neon(process.env.DATABASE_URL);
 try{
  const exists=await sql`SELECT to_regclass('public.crypto_social_windows') AS windows,to_regclass('public.crypto_rolling_coins') AS rolling,to_regclass('public.crypto_watcher_campaign') AS watchers,to_regclass('public.crypto_origin_windows') AS origin,to_regclass('public.crypto_market_hourly') AS hourly,to_regclass('public.crypto_social_history_campaign') AS history`;
  if(!exists[0]?.windows)return ok({...empty,status:'not_started'});
  const [days,origin,hourly,rolling,watchers,pilot,history,last]=await Promise.all([
   sql`SELECT coin,count(DISTINCT (start_at AT TIME ZONE 'UTC')::date)::int AS days_total,count(DISTINCT (start_at AT TIME ZONE 'UTC')::date) FILTER(WHERE pages>0)::int AS days_searched,count(*) FILTER(WHERE status IN ('pending','partial'))::int AS unfinished,min(start_at) AS first_window,max(end_at) FILTER(WHERE pages>0) AS searched_through FROM crypto_social_windows WHERE query<>'timeline text match' GROUP BY coin`,
   exists[0].origin?sql`SELECT o.coin,w.pages,w.status,w.start_at,w.end_at FROM crypto_origin_windows o JOIN crypto_social_windows w ON w.id=o.window_id`:Promise.resolve([]),
   exists[0].hourly?sql`SELECT s.coin,s.id AS source_id,count(*)::int AS hours,max(h.hour) AS latest FROM crypto_market_sources s JOIN crypto_market_hourly_latest h ON h.source_id=s.id WHERE s.is_default GROUP BY s.coin,s.id`:Promise.resolve([]),
   exists[0].rolling?sql`SELECT c.coin,c.used_credits,c.credit_limit,p.end_at FROM crypto_rolling_coins c JOIN crypto_rolling_campaign p ON p.id=c.campaign_id ORDER BY c.coin`:Promise.resolve([]),
   exists[0].watchers?sql`SELECT used_credits,credit_limit,end_at FROM crypto_watcher_campaign WHERE id='watchers-ten-v1'`:Promise.resolve([]),
   sql`SELECT reserved_credits,credit_limit FROM crypto_social_pilot WHERE id='zcat-zec-v1'`,
   exists[0].history?sql`SELECT id,reserved_credits,credit_limit,end_at FROM crypto_social_history_campaign ORDER BY id`:Promise.resolve([]),
   sql`SELECT max(requested_at) AS last_saved,count(*) FILTER(WHERE status IN ('reserved','uncertain'))::int AS outstanding FROM crypto_social_requests`,
  ]);
  const coverage=COINS.map(c=>{const d=days.find(r=>r.coin===c.symbol),o=origin.find(r=>r.coin===c.symbol),h=hourly.find(r=>r.coin===c.symbol);
   return {symbol:c.symbol,days_total:d?.days_total??0,days_searched:d?.days_searched??0,unfinished:d?.unfinished??0,first_window:d?.first_window??null,searched_through:d?.searched_through??null,origin:o?{pages:Number(o.pages),status:String(o.status),start_at:o.start_at,end_at:o.end_at}:c.originFrom?{pages:0,status:'pending',start_at:c.originFrom,end_at:null}:null,hourly_hours:h?.hours??0,hourly_latest:h?.latest??null,source_id:h?.source_id??null};});
  const ledgers=[...rolling.map(r=>({name:`Rolling · ${r.coin}`,used:Number(r.used_credits),ceiling:Number(r.credit_limit),ends:r.end_at})),...watchers.map(w=>({name:'Watchers · 10 accounts',used:Number(w.used_credits),ceiling:Number(w.credit_limit),ends:w.end_at})),...pilot.map(p=>({name:'Profiles & bios',used:Number(p.reserved_credits),ceiling:Number(p.credit_limit),ends:null})),...history.map(h=>({name:`History · ${String(h.id).replace('-july-2026','').toUpperCase()}`,used:Number(h.reserved_credits),ceiling:Number(h.credit_limit),ends:h.end_at}))];
  return withCdnCache(ok({status:'ready',coverage,ledgers,registry,watchers:watcherAccounts,lastCollection:last[0]?.last_saved??null,outstanding:last[0]?.outstanding??0,asOf:new Date().toISOString()}));
 }catch{return fail('Status could not be loaded','STATUS_READ_FAILED',503);}
}

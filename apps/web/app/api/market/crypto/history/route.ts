import {neon} from '@neondatabase/serverless';
import {ok,fail} from '@/lib/server/api-utils';
import type {RunMarket,Pool} from '@/lib/crypto-run';
export const dynamic='force-dynamic';
export const runtime='nodejs';
// Read-only: a page visit never fetches providers or mutates the archive.
export async function GET(request:Request){
 const params=new URL(request.url).searchParams,coin=params.get('coin')??'ZCAT';
 if(!['ZCAT','ZEC','PONS'].includes(coin))return fail('Unknown coin','INVALID_COIN',400);
 const result:RunMarket={status:'unavailable',points:[],pools:[],selected:null,source:coin!=='ZEC'?'GeckoTerminal':'CoinGecko',
  sourceUrl:coin!=='ZEC'?'https://www.geckoterminal.com/':'https://www.coingecko.com/en/coins/zcash',
  note:'No archived market history yet. Collection saves public observations to Postgres; missing prices remain blank.',observedAt:'',storage:'postgres'};
 const start=coin==='PONS'?'2026-07-01':'2026-07-25',campaignId=coin==='PONS'?'pons-july-2026':'zcat-july-2026';
 if(!process.env.DATABASE_URL)return ok(result);
 const sql=neon(process.env.DATABASE_URL);
 try{
  const exists=await sql`SELECT to_regclass('public.crypto_market_latest') AS relation`;
  if(!exists[0]?.relation)return ok(result);
  const sources=await sql`SELECT id,provider,source_url,metadata,is_default FROM crypto_market_sources WHERE coin=${coin} ORDER BY is_default DESC,id`;
  result.pools=coin!=='ZEC'?sources.map(s=>s.metadata as Pool):[];
  const requested=params.get('pool');
  const selected=requested?sources.find(s=>s.metadata.id===requested):sources.find(s=>s.is_default);
  if(!selected){if(requested)return fail('Pool has no saved history for this coin','INVALID_POOL',400);return ok(result);}
  const rows=await sql`SELECT day::text,close,volume,open,high,low,complete,kind,sample_at,retrieved_at,fetch_id::text
   FROM crypto_market_latest WHERE source_id=${selected.id} AND day>=${start}::date AND day<=(now() AT TIME ZONE 'UTC')::date ORDER BY day`;
  const fetched=await sql`SELECT max(retrieved_at) AS last_saved,count(*)::int AS revisions FROM crypto_market_fetches WHERE source_id=${selected.id}`;
  result.points=rows.map(p=>({day:p.day,close:Number(p.close),volume:Number(p.volume),complete:p.complete,observedAt:p.retrieved_at,fetchId:p.fetch_id}));
  result.status=rows.length?'ready':'unavailable';result.source=selected.provider;result.sourceUrl=selected.source_url;
  result.selected=coin!=='ZEC'?selected.metadata as Pool:null;
  result.observedAt=fetched[0]?.last_saved??'';
  result.note=coin!=='ZEC'?'Archived USD daily close and volume for this pool only. Incomplete candles are labeled; earlier gaps remain blank. Default pool is pinned.':'Archived CoinGecko daily price observations and rolling 24-hour volume, not exchange closing prices.';
  result.archiveRevisions=fetched[0]?.revisions??0;
  if(coin!=='ZEC'){
   const batches=await sql`SELECT to_regclass('public.crypto_social_history_batches') AS relation`;
   if(batches[0]?.relation){
    const latest=await sql`SELECT focus_start,focus_end,focus_reason,started_at,status,max_requests,requests_saved FROM crypto_social_history_batches WHERE campaign_id=${campaignId} ORDER BY id DESC LIMIT 1`;
    result.investigation=latest[0] as RunMarket['investigation'];
   }
  }
  return ok(result);
 }catch{return fail('Archived market history is temporarily unavailable','MARKET_ARCHIVE_READ_FAILED',503);}
}

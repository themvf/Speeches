import {ok,fail} from '@/lib/server/api-utils';
import {verifiedPools,candles} from '@/lib/server/crypto-run-market';
import type {RunMarket} from '@/lib/crypto-run';
export const revalidate=3600;
export const runtime='nodejs';
const base='https://api.geckoterminal.com/api/v2/networks/solana';
async function get(url:string){const r=await fetch(url,{next:{revalidate:3600},signal:AbortSignal.timeout(10000),headers:{Accept:'application/json;version=20230302'}});if(!r.ok)throw new Error('Market provider unavailable');return r.json();}
export async function GET(request:Request){
 const params=new URL(request.url).searchParams;const coin=params.get('coin')??'ZCAT';
 if(!['ZCAT','ZEC'].includes(coin))return fail('Unknown coin','INVALID_COIN',400);
 const result:RunMarket={status:'unavailable',points:[],pools:[],selected:null,source:'GeckoTerminal',sourceUrl:'https://www.geckoterminal.com/',note:'Historical market data is unavailable. Gaps are not zero prices.',observedAt:new Date().toISOString()};
 try{
  if(coin==='ZEC'){
   result.source='CoinGecko';result.sourceUrl='https://www.coingecko.com/en/coins/zcash';
   const data=await get('https://api.coingecko.com/api/v3/coins/zcash/market_chart?vs_currency=usd&days=90&interval=daily');
   if(!Array.isArray(data.prices)||!Array.isArray(data.total_volumes))throw new Error('Invalid history');
   const volumes=new Map<number,number>(data.total_volumes);result.points=data.prices.flatMap((p:number[])=>{
    if(!Array.isArray(p)||!Number.isFinite(p[0])||!Number.isFinite(new Date(p[0]).getTime()))return [];const day=new Date(p[0]).toISOString().slice(0,10),v=volumes.get(p[0]);return day>='2026-07-25'&&day<=result.observedAt.slice(0,10)&&Number.isFinite(p[1])&&p[1]>0&&v!=null&&Number.isFinite(v)&&v>=0?[{day,close:p[1],volume:v}]:[];
   });result.points=Array.from(new Map(result.points.map(p=>[p.day,p])).values()).sort((a,b)=>a.day.localeCompare(b.day));result.status=result.points.length?'ready':'unavailable';result.note='CoinGecko daily price observations and rolling 24-hour volume; not exchange closing prices.';return ok(result);
  }
  const data=await get(base+'/tokens/HcRLc9VDgjLeK154xDawfb1dmVJ98DoSqcwTHGqiDeJR/pools');
  result.pools=verifiedPools(data.data??[]);
  const requested=params.get('pool');
  // Prefer the oldest of the five currently most liquid returned pools. Never splice pools.
  const candidates=[...result.pools].sort((a,b)=>b.liquidity-a.liquidity).slice(0,5).sort((a,b)=>a.created.localeCompare(b.created));
  const selected=requested?result.pools.find(p=>p.id===requested):candidates[0];
  if(!selected){if(requested)return fail('Pool does not match this token','INVALID_POOL',400);return ok(result);}
  result.selected=selected;result.sourceUrl='https://www.geckoterminal.com/solana/pools/'+selected.id;
  const ohlcv=await get(base+'/pools/'+selected.id+'/ohlcv/day?aggregate=1&limit=100&currency=usd&include_empty_intervals=false&token='+selected.side);
  result.points=candles(ohlcv.data?.attributes?.ohlcv_list);result.status=result.points.length?'ready':'unavailable';
  result.note='USD daily close and trading volume for the selected pool only. Current-day candle is incomplete. No prices are filled before available history.';
  return ok(result);
 }catch{return ok(result);}
}

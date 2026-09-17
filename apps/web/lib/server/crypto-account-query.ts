import type {NeonQueryFunction} from '@neondatabase/serverless';
import {summarize,type CoinImpact,type CoinRole,type Leader} from '../crypto-leaders.ts';
import {VOICE_VERSION} from '../crypto-voices.ts';
import {WATCHER_VERSION} from '../crypto-watchers.ts';
import {IMPACT_VERSION} from '../crypto-impact.ts';
type Sql=NeonQueryFunction<false,false>;
// Roles come from the collection-time voice snapshots; impact from the immutable event study; watcher category from the watcher snapshot.
export async function loadRoles(sql:Sql,accountId?:string):Promise<Map<string,{handle:string;followers:number|null;roles:CoinRole[]}>>{
 const exists=await sql`SELECT to_regclass('public.crypto_ranking_cache') AS relation`;const out=new Map<string,{handle:string;followers:number|null;roles:CoinRole[]}>();
 if(!exists[0]?.relation)return out;
 const rows=accountId
  ?await sql`SELECT substr(c.key,8) AS coin,v FROM crypto_ranking_cache c,jsonb_array_elements(c.payload->'voices') v WHERE c.key LIKE 'voices:%' AND c.version=${VOICE_VERSION} AND v->>'id'=${accountId}`
  :await sql`SELECT substr(c.key,8) AS coin,v FROM crypto_ranking_cache c,jsonb_array_elements(c.payload->'voices') v WHERE c.key LIKE 'voices:%' AND c.version=${VOICE_VERSION} AND ((v->'scores'->>'Early discoverers')::float>0 OR (v->'scores'->>'Original analysis')::float>0 OR (v->>'subsequent')::int>=3)`;
 for(const r of rows){const v=r.v as {id:string;handle:string;followers:number|null;role:string;posts:number;first:string;scores:Record<string,number>};
  const entry=out.get(v.id)??{handle:v.handle,followers:v.followers,roles:[]};
  entry.roles.push({coin:String(r.coin),role:v.role,early:(v.scores['Early discoverers']??0)>0,analysis:(v.scores['Original analysis']??0)>0,amplifier:v.role==='Amplifiers',posts:v.posts,first:v.first});out.set(v.id,entry);}
 return out;
}
export async function loadImpact(sql:Sql,accountId?:string):Promise<Map<string,{handle:string;followers:number|null;impact:CoinImpact[]}>>{
 const exists=await sql`SELECT to_regclass('public.crypto_price_events') AS relation`;const out=new Map<string,{handle:string;followers:number|null;impact:CoinImpact[]}>();
 if(!exists[0]?.relation)return out;
 const rows=await sql`WITH base AS (SELECT s.coin,percentile_cont(0.5) WITHIN GROUP(ORDER BY n.close/h.close-1) AS median_24h FROM crypto_market_sources s JOIN crypto_market_hourly_latest h ON h.source_id=s.id JOIN crypto_market_hourly_latest n ON n.source_id=s.id AND n.hour=h.hour+interval '24 hours' WHERE s.is_default GROUP BY s.coin)
  SELECT e.account_id,a.handle,a.followers::float AS followers,e.coin,count(*)::int AS episodes,percentile_cont(0.5) WITHIN GROUP(ORDER BY e.price_after_24h/e.price_0-1)::float AS median_24h,
   percentile_cont(0.5) WITHIN GROUP(ORDER BY e.price_after_24h/e.price_0-1-coalesce(b.median_24h,0))::float AS median_excess_24h,avg((e.price_after_24h>e.price_0)::int)::float AS share_up_24h
  FROM crypto_price_events e JOIN crypto_social_accounts a ON a.id=e.account_id LEFT JOIN base b ON b.coin=e.coin
  WHERE e.version=${IMPACT_VERSION} AND e.episode AND (${accountId??null}::text IS NULL OR e.account_id=${accountId??null}) GROUP BY e.account_id,a.handle,a.followers,e.coin`;
 for(const r of rows){const entry=out.get(String(r.account_id))??{handle:String(r.handle),followers:r.followers==null?null:Number(r.followers),impact:[]};entry.impact.push({coin:String(r.coin),episodes:Number(r.episodes),median_24h:r.median_24h,median_excess_24h:r.median_excess_24h,share_up_24h:r.share_up_24h});out.set(String(r.account_id),entry);}
 return out;
}
export async function loadWatchers(sql:Sql):Promise<Map<string,string>>{
 const exists=await sql`SELECT to_regclass('public.crypto_ranking_cache') AS relation`;const out=new Map<string,string>();
 if(!exists[0]?.relation)return out;
 const rows=await sql`SELECT v->>'id' AS id,v->>'category' AS category FROM crypto_ranking_cache c,jsonb_array_elements(c.payload->'accounts') v WHERE c.key='watchers:ALL' AND c.version=${WATCHER_VERSION}`;
 for(const r of rows)out.set(String(r.id),String(r.category));return out;
}
export async function loadLeaders(sql:Sql,accountId?:string):Promise<Leader[]>{
 const [roles,impact,watchers]=await Promise.all([loadRoles(sql,accountId),loadImpact(sql,accountId),loadWatchers(sql)]);
 const ids=new Set([...roles.keys(),...impact.keys()]);if(accountId)ids.add(accountId);
 return [...ids].map(id=>{const r=roles.get(id),i=impact.get(id);return summarize(id,r?.handle??i?.handle??id,r?.followers??i?.followers??null,r?.roles??[],i?.impact??[],watchers.get(id)??null);});
}

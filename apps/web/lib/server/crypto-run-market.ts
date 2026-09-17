import type {Pool,MarketPoint} from '../crypto-run';
import {coinConfig} from '../crypto-coins.ts';
export const ZCAT_ADDRESS=coinConfig('ZCAT').address!;
type RawPool={attributes?:{address?:string;name?:string;pool_created_at?:string;reserve_in_usd?:string};relationships?:{base_token?:{data?:{id?:string}};quote_token?:{data?:{id?:string}}}};
export function verifiedPools(raw:RawPool[]):Pool[]{return raw.flatMap(p=>{
 const a=p.attributes;const base=p.relationships?.base_token?.data?.id,quote=p.relationships?.quote_token?.data?.id;
 const side=base===`solana_${ZCAT_ADDRESS}`?'base':quote===`solana_${ZCAT_ADDRESS}`?'quote':null;
 if(!side||!a?.address||! /^[1-9A-HJ-NP-Za-km-z]{32,44}$/.test(a.address)||!a.pool_created_at||!Number.isFinite(Date.parse(a.pool_created_at)))return [];
 const liquidity=Number(a.reserve_in_usd);return [{id:a.address,name:a.name??a.address,created:a.pool_created_at,liquidity:Number.isFinite(liquidity)?liquidity:0,side}];
});}
export function candles(raw:unknown):MarketPoint[]{if(!Array.isArray(raw))return [];const map=new Map<string,MarketPoint>();for(const row of raw){if(!Array.isArray(row)||row.length<6||!row.slice(0,6).every(v=>typeof v==='number'&&Number.isFinite(v))||row[4]<=0||row[5]<0)continue;const date=new Date(row[0]*1000);if(!Number.isFinite(date.getTime()))continue;const day=date.toISOString().slice(0,10);if(day<'2026-07-25'||day>new Date().toISOString().slice(0,10))continue;map.set(day,{day,close:row[4],volume:row[5]});}return [...map.values()].sort((a,b)=>a.day.localeCompare(b.day));}

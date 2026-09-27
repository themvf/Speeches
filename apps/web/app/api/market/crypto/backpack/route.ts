import {BACKPACK_CACHE_CONTROL} from '@/lib/backpack-cache';
import {unstable_cache} from 'next/cache';
import {NextResponse} from 'next/server';
import {readBackpack} from '@/lib/server/backpack-store';
import {readBpIntel} from '@/lib/server/bp-intel-store';
import {CSV_COLUMNS,parseIntelQuery,toCsv,withAmounts,type IntelQuery} from '@/lib/bp-intel';
const cachedRead=unstable_cache(readBackpack,['backpack-observations-v6'],{revalidate:3600,tags:['backpack-observations']});
// Hourly worker data: a shorter cache than the daily overview, invalidated by the same revalidation hook.
const cachedIntel=unstable_cache(readBpIntel,['bp-intel-v1'],{revalidate:300,tags:['backpack-observations']});
const INTEL_CACHE_CONTROL='public, s-maxage=300, stale-while-revalidate=900';
export const runtime='nodejs';
export const dynamic='force-dynamic';
async function intel(query:IntelQuery){
 const data=await cachedIntel(query);
 if(data.status==='not_found')return NextResponse.json({error:'That cohort version or portfolio read does not exist.'},{status:404});
 console.info(JSON.stringify({metric:'bp_intel_response',section:query.section,format:query.format,rows:data.rows.length}));
 if(query.format==='json')return NextResponse.json(data,{headers:{'Cache-Control':INTEL_CACHE_CONTROL}});
 if(query.section==='overview')return NextResponse.json({error:'CSV export needs a specific section'},{status:400});
 const version=data.meta?.version?.version_id??'none',raw=query.section==='roster'&&query.ranking==='raw';
 const body=raw?toCsv(data.extra.raw??[],CSV_COLUMNS.raw_ranking):toCsv(withAmounts(query.section,data.rows),CSV_COLUMNS[query.section]);
 return new NextResponse(body,{headers:{'Content-Type':'text/csv; charset=utf-8','Cache-Control':INTEL_CACHE_CONTROL,
  'Content-Disposition':`attachment; filename="bp-${raw?'raw-ranking':query.section}-cohort-${version}${query.mint?`-${query.mint.slice(0,8)}`:''}${query.wallet?`-${query.wallet.slice(0,8)}`:''}.csv"`}});
}
export async function GET(req:Request){
 const params=new URL(req.url).searchParams;
 if(params.has('section')){
  const {query,error}=parseIntelQuery(params);
  if(!query)return NextResponse.json({error},{status:400});
  try{return await intel(query);}
  catch{return NextResponse.json({error:'BP holder intelligence could not be loaded.'},{status:503});}
 }
 const id=params.get('asset');
 if(id!==null&&!/^[1-9]\d{0,14}$/.test(id))return NextResponse.json({error:'Invalid asset ID'},{status:400});
 try{const data=await cachedRead(id?Number(id):undefined);
 const bytes=Buffer.byteLength(JSON.stringify(data));
 console.info(JSON.stringify({metric:'backpack_api_response',bytes,scope:id?'asset':'overview'}));
 return NextResponse.json(data,{headers:{'Cache-Control':BACKPACK_CACHE_CONTROL}});}
 catch{return NextResponse.json({error:'Backpack observations could not be loaded.'},{status:503});}
}

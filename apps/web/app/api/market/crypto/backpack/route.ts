import {BACKPACK_CACHE_CONTROL} from '@/lib/backpack-cache';
import {unstable_cache} from 'next/cache';
import {NextResponse} from 'next/server';
import {readBackpack} from '@/lib/server/backpack-store';
const cachedRead=unstable_cache(readBackpack,['backpack-observations-v2'],{revalidate:3600,tags:['backpack-observations']});
export const runtime='nodejs';
export const dynamic='force-dynamic';
export async function GET(req:Request){
 const id=new URL(req.url).searchParams.get('asset');
 if(id!==null&&!/^[1-9]\d{0,14}$/.test(id))return NextResponse.json({error:'Invalid asset ID'},{status:400});
 try{const data=await cachedRead(id?Number(id):undefined);
 const bytes=Buffer.byteLength(JSON.stringify(data));
 console.info(JSON.stringify({metric:'backpack_api_response',bytes,scope:id?'asset':'overview'}));
 return NextResponse.json(data,{headers:{'Cache-Control':BACKPACK_CACHE_CONTROL}});}
 catch{return NextResponse.json({error:'Backpack observations could not be loaded.'},{status:503});}
}

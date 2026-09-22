import {NextResponse} from 'next/server';
import {readBackpack} from '@/lib/server/backpack-store';
export const runtime='nodejs';
export const dynamic='force-dynamic';
export async function GET(req:Request){
 const id=new URL(req.url).searchParams.get('asset');
 if(id!==null&&!/^[1-9]\d{0,14}$/.test(id))return NextResponse.json({error:'Invalid asset ID'},{status:400});
 try{return NextResponse.json(await readBackpack(id?Number(id):undefined),{headers:{'Cache-Control':'public, s-maxage=300, stale-while-revalidate=300'}});}
 catch{return NextResponse.json({error:'Backpack observations could not be loaded.'},{status:503});}
}

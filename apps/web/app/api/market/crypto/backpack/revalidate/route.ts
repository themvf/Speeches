import {validRevalidationToken} from '@/lib/backpack-cache';
import {revalidatePath,revalidateTag} from 'next/cache';
import {NextResponse} from 'next/server';
export const runtime='nodejs';
export async function POST(req:Request){
 const secret=process.env.BACKPACK_REVALIDATE_SECRET;
 const provided=req.headers.get('authorization')??'';
 if(!validRevalidationToken(provided,secret)){
  return NextResponse.json({error:'Unauthorized'},{status:401,headers:{'Cache-Control':'no-store'}});
 }
 revalidateTag('backpack-observations');
 revalidatePath('/market/crypto/backpack','layout');
 revalidatePath('/api/market/crypto/backpack');
 return NextResponse.json({revalidated:true,limitation:'Existing CDN responses retain their Cache-Control lifetime.'},{headers:{'Cache-Control':'no-store'}});
}

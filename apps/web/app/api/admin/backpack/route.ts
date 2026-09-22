import {createHash,timingSafeEqual} from 'node:crypto';
import {cookies} from 'next/headers';
import {NextResponse} from 'next/server';
import {neon} from '@neondatabase/serverless';
import {getGithubActionsConfig} from '@/lib/server/env';
import {labelError,solanaAddress,sameOrigin} from '@/lib/backpack-admin';
import {BP_MINT} from '@/lib/backpack';
export const runtime='nodejs';
function reply(error:string,status:number){return NextResponse.json({error},{status});}
async function authorized(){
 const secret=process.env.ADMIN_SECRET,actual=(await cookies()).get('admin_auth')?.value;
 if(!secret||!actual||!/^[a-f0-9]{64}$/.test(actual))return false;
 const expected=createHash('sha256').update(secret).digest('hex');
 return actual.length===expected.length&&timingSafeEqual(Buffer.from(actual),Buffer.from(expected));
}
export async function GET(req:Request){
 if(!await authorized())return reply('Administrator access required',401);
 if(new URL(req.url).searchParams.get('labels')==='1'){
  if(!process.env.DATABASE_URL)return reply('Database not configured',503);
  try{const sql=neon(process.env.DATABASE_URL);const labels=await sql`SELECT * FROM backpack_wallet_labels ORDER BY verified_at DESC LIMIT 1000`;
   return NextResponse.json({authorized:true,labels},{headers:{'Cache-Control':'no-store'}});
  }catch{return reply('Label registry unavailable. Apply the Backpack migration first.',503);}
 }
 return NextResponse.json({authorized:true});
}
export async function POST(req:Request){
 if(!await authorized())return reply('Administrator access required',401);
 if(!sameOrigin(req))return reply('Same-origin request required',403);
 let body:Record<string,unknown>;
 try{body=await req.json();}catch{return reply('Invalid JSON',400);}
 if(!body||typeof body!=='object'||Array.isArray(body))return reply('Expected a JSON object',400);
 if(body.action==='refresh'){
  const c=getGithubActionsConfig();
  if(!c.token)return reply('GitHub Actions dispatch is not configured',503);
  try{
   const res=await fetch(`https://api.github.com/repos/${c.owner}/${c.repo}/actions/workflows/backpack-monitor.yml/dispatches`,{
    method:'POST',headers:{Authorization:`Bearer ${c.token}`,Accept:'application/vnd.github+json','Content-Type':'application/json'},
    body:JSON.stringify({ref:c.ref}),signal:AbortSignal.timeout(10000)});
   return res.status===204?NextResponse.json({ok:true,message:'Daily capture queued. Existing snapshots are preserved; failed assets can retry.'}):reply('Could not queue collector',502);
  }catch{return reply('Collector dispatch unavailable',503);}
 }
 if(body.action==='label_wallet'){
  const error=labelError(body);if(error)return reply(error,400);
  if(!process.env.DATABASE_URL)return reply('Database not configured',503);
  try{
   const sql=neon(process.env.DATABASE_URL);
   // One statement: current label and its immutable evidence revision commit together.
   await sql`WITH saved AS (
    INSERT INTO backpack_wallet_labels(wallet_address,label,entity,confidence,source,verified_at,notes)
    VALUES(${String(body.wallet_address)},${String(body.label)},${String(body.entity).trim()},${String(body.confidence)},${String(body.source)},now(),${String(body.notes).trim()})
    ON CONFLICT(wallet_address) DO UPDATE SET label=excluded.label,entity=excluded.entity,confidence=excluded.confidence,
      source=excluded.source,verified_at=excluded.verified_at,notes=excluded.notes RETURNING *
   ) INSERT INTO backpack_wallet_label_revisions(wallet_address,label,entity,confidence,source,verified_at,notes)
      SELECT wallet_address,label,entity,confidence,source,verified_at,notes FROM saved`;
   return NextResponse.json({ok:true,message:'Wallet evidence saved. Applies to future captures; historical ownership remains unchanged.'});
  }catch{return reply('Label could not be saved. Apply the Backpack migration first.',503);}
 }
 if(body.action!=='add_asset')return reply('Unsupported action',400);
 const required=['token_symbol','token_name','solana_mint','underlying_symbol','underlying_exchange','underlying_name','asset_type','issuer','official_source','approval_notes'];
 if(required.some(k=>typeof body[k]!=='string'||!String(body[k]).trim()||String(body[k]).length>2000))return reply('All registry and approval fields are required',400);
 const mint=String(body.solana_mint);
 if(!solanaAddress(mint)||mint===BP_MINT)return reply('Supply a valid security mint; BP is managed separately',400);
 if(!['common_stock','etf','other_security'].includes(String(body.asset_type)))return reply('Invalid security type',400);
 try{if(new URL(String(body.official_source)).protocol!=='https:')throw new Error();}catch{return reply('Evidence must be an HTTPS source URL',400);}
 if(body.approved!==true)return reply('Explicit manual registry approval is required',400);
 const launch=body.launch_date?String(body.launch_date):null;
 if(launch&&(!/^\d{4}-\d{2}-\d{2}$/.test(launch)||Number.isNaN(Date.parse(launch))))return reply('Invalid launch date',400);
 if(!process.env.DATABASE_URL)return reply('Database not configured',503);
 try{
  const sql=neon(process.env.DATABASE_URL);
  const rows=await sql`INSERT INTO backpack_assets(token_symbol,token_name,solana_mint,underlying_symbol,underlying_exchange,
    underlying_name,asset_type,issuer,launch_date,official_source,source_verified_at,verification_status,approval_notes)
    VALUES(${String(body.token_symbol).trim()},${String(body.token_name).trim()},${mint},${String(body.underlying_symbol).trim()},
     ${String(body.underlying_exchange).trim()},${String(body.underlying_name).trim()},${String(body.asset_type)},${String(body.issuer).trim()},
     ${launch}::date,${String(body.official_source)},now(),'manual_approved',${String(body.approval_notes)})
    ON CONFLICT(solana_mint) DO NOTHING RETURNING id`;
  return rows.length?NextResponse.json({ok:true,id:rows[0].id}):reply('Mint already registered',409);
 }catch{return reply('Registry unavailable. Apply the Backpack migration first.',503);}
}

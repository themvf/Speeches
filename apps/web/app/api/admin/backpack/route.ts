import {createHash,timingSafeEqual} from 'node:crypto';
import {cookies} from 'next/headers';
import {NextResponse} from 'next/server';
import {neon} from '@neondatabase/serverless';
import {getGithubActionsConfig} from '@/lib/server/env';
import {BP_MINT} from '@/lib/backpack';
export const runtime='nodejs';
function reply(error:string,status:number){return NextResponse.json({error},{status});}
async function authorized(){
 const secret=process.env.ADMIN_SECRET,actual=(await cookies()).get('admin_auth')?.value;
 if(!secret||!actual||!/^[a-f0-9]{64}$/.test(actual))return false;
 const expected=createHash('sha256').update(secret).digest('hex');
 return actual.length===expected.length&&timingSafeEqual(Buffer.from(actual),Buffer.from(expected));
}
export async function GET(){
 if(!await authorized())return reply('Administrator access required',401);
 return NextResponse.json({authorized:true});
}
export async function POST(req:Request){
 if(!await authorized())return reply('Administrator access required',401);
 if(req.headers.get('origin')!==new URL(req.url).origin)return reply('Same-origin request required',403);
 let body:Record<string,unknown>;
 try{body=await req.json();}catch{return reply('Invalid JSON',400);}
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
 if(body.action!=='add_asset')return reply('Unsupported action',400);
 const required=['token_symbol','token_name','solana_mint','underlying_symbol','underlying_exchange','underlying_name','asset_type','issuer','official_source','approval_notes'];
 if(required.some(k=>typeof body[k]!=='string'||!String(body[k]).trim()||String(body[k]).length>2000))return reply('All registry and approval fields are required',400);
 const mint=String(body.solana_mint);
 if(!/^[1-9A-HJ-NP-Za-km-z]{32,44}$/.test(mint)||mint===BP_MINT)return reply('Supply a security mint; BP is managed separately',400);
 // Decode base58 to ensure this is a 32-byte public key rather than just a ticker-shaped string.
 let value=0n;const alphabet='123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz';
 for(const char of mint)value=value*58n+BigInt(alphabet.indexOf(char));
 let bytes=0;for(let v=value;v>0n;v>>=8n)bytes++;
 bytes+=(mint.match(/^1*/)?.[0].length??0);
 if(bytes!==32)return reply('Mint must decode to a 32-byte Solana address',400);
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

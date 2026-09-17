import {neon} from '@neondatabase/serverless';
import {ok,fail} from '@/lib/server/api-utils';
import {COINS} from '@/lib/crypto-coins';
export const dynamic='force-dynamic';
export const runtime='nodejs';
// Header search: a coin symbol or name, a contract address, or an account handle.
export async function GET(request:Request){
 const q=(new URL(request.url).searchParams.get('q')??'').trim().replace(/^[@$#]/,'');
 if(q.length<2||q.length>80)return fail('Type at least two characters','INVALID_QUERY',400);
 const lower=q.toLowerCase();
 const coins=COINS.filter(c=>c.symbol.toLowerCase().includes(lower)||c.name.toLowerCase().includes(lower)||(c.address&&c.address.toLowerCase()===lower)).map(c=>({symbol:c.symbol,name:c.name}));
 const untracked=!coins.length&&/^(0x[a-f0-9]{40}|[1-9A-HJ-NP-Za-km-z]{32,44})$/i.test(q)?q:null;
 if(!process.env.DATABASE_URL)return ok({coins,accounts:[],untrackedContract:untracked});
 try{const sql=neon(process.env.DATABASE_URL);
  const accounts=await sql`SELECT id,handle,followers::float AS followers FROM crypto_social_accounts WHERE handle ILIKE ${'%'+q+'%'} AND handle<>id ORDER BY (lower(handle)=${lower}) DESC,followers DESC NULLS LAST LIMIT 8`;
  return ok({coins,accounts,untrackedContract:untracked});
 }catch{return fail('Search unavailable','SEARCH_FAILED',503);}
}

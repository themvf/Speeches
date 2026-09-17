import {neon} from '@neondatabase/serverless';
import {ok,fail} from '@/lib/server/api-utils';
import {rankLeaders} from '@/lib/crypto-leaders';
import {loadLeaders} from '@/lib/server/crypto-account-query';
import {withCdnCache} from '@/lib/server/crypto-ranking-cache';
export const dynamic='force-dynamic';
export const runtime='nodejs';
// Accounts that were early, analytical or price-linked on more than one coin. Read-only; built from snapshots and the event study.
export async function GET(){
 if(!process.env.DATABASE_URL)return ok({status:'not_configured',leaders:[]});
 try{const sql=neon(process.env.DATABASE_URL);
  const leaders=rankLeaders(await loadLeaders(sql)).filter(l=>l.early_coins||l.episodes>=3||l.roles.some(r=>r.analysis)).slice(0,50);
  return withCdnCache(ok({status:leaders.length?'ready':'no_evidence',leaders,asOf:new Date().toISOString()}));
 }catch{return fail('Leaders could not be loaded','LEADERS_READ_FAILED',503);}
}

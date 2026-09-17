import {neon} from '@neondatabase/serverless';
import {ok,fail} from '@/lib/server/api-utils';
import {rankLeaders} from '@/lib/crypto-leaders';
import {loadLeaders} from '@/lib/server/crypto-account-query';
import {withCdnCache} from '@/lib/server/crypto-ranking-cache';
export const dynamic='force-dynamic';
export const runtime='nodejs';
// People payload: every account with saved role, watcher or price-linked evidence, plus last activity. Read-only.
export async function GET(request:Request){
 const all=new URL(request.url).searchParams.get('all')==='1';
 if(!process.env.DATABASE_URL)return ok({status:'not_configured',leaders:[]});
 try{const sql=neon(process.env.DATABASE_URL);
  const ranked=rankLeaders(await loadLeaders(sql));
  const leaders=(all?ranked:ranked.filter(l=>l.early_coins||l.episodes>=3||l.roles.some(r=>r.analysis))).slice(0,all?600:50);
  const ids=leaders.map(l=>l.account_id);
  const activity=ids.length?await sql`SELECT author_id,max(posted_at) AS last_at,count(*)::int AS posts,count(DISTINCT posted_at::date)::int AS days FROM crypto_social_posts WHERE author_id=ANY(${ids}) AND kind<>'repost' GROUP BY author_id`:[];
  const byId=new Map(activity.map(a=>[String(a.author_id),a]));
  return withCdnCache(ok({status:leaders.length?'ready':'no_evidence',leaders:leaders.map(l=>({...l,last_at:byId.get(l.account_id)?.last_at??null,posts:byId.get(l.account_id)?.posts??0,days:byId.get(l.account_id)?.days??0})),asOf:new Date().toISOString()}));
 }catch{return fail('Leaders could not be loaded','LEADERS_READ_FAILED',503);}
}

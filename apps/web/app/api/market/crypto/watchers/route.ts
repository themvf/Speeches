import {neon} from '@neondatabase/serverless';
import {ok,fail} from '@/lib/server/api-utils';
import {rankWatchers,type WatcherPost} from '@/lib/crypto-watchers';
import {WATCHER_QUERY} from '@/lib/server/crypto-watchers-query';
export const dynamic='force-dynamic';
export const runtime='nodejs';
export async function GET(request:Request){
 const coin=new URL(request.url).searchParams.get('coin')??'ALL';
 if(!['ALL','ZCAT','ZEC','PONS','DPONS','STANDARD'].includes(coin))return fail('Unknown coin','INVALID_COIN',400);
 if(!process.env.DATABASE_URL)return ok({accounts:[],loaded:0,total:0,candidates:0});
 try{
  const sql=neon(process.env.DATABASE_URL);
  const exists=await sql`SELECT to_regclass('public.crypto_social_profile_history') AS relation`;
  if(!exists[0]?.relation)return ok({accounts:[],loaded:0,total:0,candidates:0});
  const posts=await sql.query(WATCHER_QUERY,[]) as WatcherPost[];
  const ranked=rankWatchers(posts,coin);
  return ok({accounts:ranked.slice(0,10),loaded:posts.length,total:posts[0]?.corpus_total??0,candidates:ranked.length,coin,asOf:new Date().toISOString()});
 }catch{return fail('Saved watcher evidence could not be loaded','WATCHER_READ_FAILED',503);}
}

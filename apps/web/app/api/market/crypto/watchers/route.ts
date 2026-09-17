import {neon} from '@neondatabase/serverless';
import {ok,fail} from '@/lib/server/api-utils';
import {rankWatchers,WATCHER_VERSION,type WatcherPost} from '@/lib/crypto-watchers';
import {WATCHER_QUERY} from '@/lib/server/crypto-watchers-query';
import {isCoin} from '@/lib/crypto-coins';
import {readRankingSnapshot,withCdnCache} from '@/lib/server/crypto-ranking-cache';
export const dynamic='force-dynamic';
export const runtime='nodejs';
export async function GET(request:Request){
 const coin=new URL(request.url).searchParams.get('coin')??'ALL';
 if(coin!=='ALL'&&!isCoin(coin))return fail('Unknown coin','INVALID_COIN',400);
 if(!process.env.DATABASE_URL)return ok({accounts:[],loaded:0,total:0,candidates:0});
 try{
  const sql=neon(process.env.DATABASE_URL);
  const exists=await sql`SELECT to_regclass('public.crypto_social_profile_history') AS relation`;
  if(!exists[0]?.relation)return ok({accounts:[],loaded:0,total:0,candidates:0});
  const cached=await readRankingSnapshot<{accounts:unknown[];loaded:number;total:number;candidates:number;coin:string}>(sql,'watchers:'+coin,WATCHER_VERSION);
  if(cached)return withCdnCache(ok({...cached.payload,source:'snapshot',asOf:cached.computed_at}));
  const posts=await sql.query(WATCHER_QUERY,[]) as WatcherPost[];
  const ranked=rankWatchers(posts,coin);
  return withCdnCache(ok({accounts:ranked.slice(0,10),loaded:posts.length,total:posts[0]?.corpus_total??0,candidates:ranked.length,coin,source:'live',asOf:new Date().toISOString()}));
 }catch{return fail('Saved watcher evidence could not be loaded','WATCHER_READ_FAILED',503);}
}

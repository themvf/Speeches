import {neon} from '@neondatabase/serverless';
import {ok,fail} from '@/lib/server/api-utils';
import {withCdnCache} from '@/lib/server/crypto-ranking-cache';
import {discoverCoins,EXTRACTION_VERSION,type DiscoveryPost} from '@/lib/crypto-coin-discovery';
export const dynamic='force-dynamic';
const validDate=(s:string)=>/^\d{4}-\d{2}-\d{2}$/.test(s)&&!Number.isNaN(Date.parse(s))&&new Date(s).toISOString().slice(0,10)===s;
export async function GET(request:Request){
 const params=new URL(request.url).searchParams,from=params.get('from')??'2026-09-01',to=params.get('to')??new Date().toISOString().slice(0,10);
 if(!validDate(from)||!validDate(to)||from>to)return fail('Choose a valid date range','INVALID_DATES',400);
 const empty={coins:[],loaded:0,total:0,unfinished:0,version:EXTRACTION_VERSION,from,to};
 if(!process.env.DATABASE_URL)return ok(empty);
 try{const sql=neon(process.env.DATABASE_URL);const exists=await sql`SELECT to_regclass('public.crypto_watcher_posts') AS relation`;if(!exists[0]?.relation)return ok(empty);
  const end=new Date(Date.parse(to)+86400000).toISOString();
  const posts=await sql`SELECT p.id,p.author_id,a.handle,p.text,p.posted_at,p.kind,p.url,count(*) OVER()::int AS total
   FROM crypto_social_posts p JOIN crypto_social_accounts a ON a.id=p.author_id
   WHERE p.posted_at>=${from}::timestamptz AND p.posted_at<${end}::timestamptz AND p.kind!='repost'
   AND EXISTS(SELECT 1 FROM crypto_watcher_posts wp JOIN crypto_watcher_windows w ON w.id=wp.window_id WHERE wp.post_id=p.id AND w.campaign_id='watchers-ten-v1')
   ORDER BY p.posted_at DESC,p.id DESC LIMIT 10000`;
  const coverage=await sql`SELECT count(*)::int AS unfinished FROM crypto_watcher_windows WHERE campaign_id='watchers-ten-v1' AND status!='search_exhausted' AND start_at<${end}::timestamptz AND end_at>${from}::timestamptz`;
  return withCdnCache(ok({...empty,coins:discoverCoins(posts as DiscoveryPost[]),loaded:posts.length,total:posts[0]?.total??0,unfinished:coverage[0]?.unfinished??0}));
 }catch{return fail('Saved coin discoveries could not be loaded','DISCOVERY_READ_FAILED',503);}
}

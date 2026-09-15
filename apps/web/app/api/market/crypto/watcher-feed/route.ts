import {neon} from '@neondatabase/serverless';
import {ok,fail} from '@/lib/server/api-utils';
import accounts from '@/lib/crypto-watcher-accounts.json';
export const dynamic='force-dynamic';
export async function GET(request:Request){
 const params=new URL(request.url).searchParams,account=params.get('account')??'';
 const offset=Number(params.get('offset')??0);
 if((account&&!accounts.some(a=>a.id===account))||!Number.isSafeInteger(offset)||offset<0||offset>100000)return fail('Invalid filter','INVALID_FILTER',400);
 const empty={posts:[],hasMore:false,accounts,campaign:null,coverage:[]};
 if(!process.env.DATABASE_URL)return ok(empty);
 try{
  const sql=neon(process.env.DATABASE_URL);
  const exists=await sql`SELECT to_regclass('public.crypto_watcher_campaign') AS relation`;
  if(!exists[0]?.relation)return ok(empty);
  const campaign=await sql`SELECT *, (SELECT max(r.requested_at) FROM crypto_watcher_calls c JOIN crypto_social_requests r ON r.id=c.request_id WHERE r.status='saved') AS last_saved,
   EXISTS(SELECT 1 FROM crypto_social_requests WHERE status IN ('reserved','uncertain')) AS outstanding_request
   FROM crypto_watcher_campaign WHERE id='watchers-ten-v1'`;
  const coverage=await sql`SELECT w.account_id,count(*) FILTER(WHERE w.status!='search_exhausted')::int AS unfinished,
   max(w.end_at) FILTER(WHERE w.status='search_exhausted') AS searched_through
   FROM crypto_watcher_windows w WHERE campaign_id='watchers-ten-v1' GROUP BY w.account_id`;
  const posts=await sql`SELECT p.id,p.author_id,a.handle,p.text,p.posted_at,p.kind,p.url FROM crypto_social_posts p
   JOIN crypto_social_accounts a ON a.id=p.author_id
   WHERE (${account}='' OR p.author_id=${account}) AND EXISTS(SELECT 1 FROM crypto_watcher_posts wp JOIN crypto_watcher_windows w ON w.id=wp.window_id WHERE wp.post_id=p.id AND w.campaign_id='watchers-ten-v1')
   ORDER BY p.posted_at DESC,p.id DESC LIMIT 26 OFFSET ${offset}`;
  return ok({posts:posts.slice(0,25),hasMore:posts.length>25,accounts,campaign:campaign[0]??null,coverage});
 }catch{return fail('Watcher feed could not be loaded','WATCHER_FEED_FAILED',503);}
}

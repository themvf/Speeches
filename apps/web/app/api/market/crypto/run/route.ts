import {neon} from '@neondatabase/serverless';
import {ok,fail} from '@/lib/server/api-utils';
export const dynamic='force-dynamic';
export const runtime='nodejs';
export async function GET(request:Request){
 const coin=new URL(request.url).searchParams.get('coin')??'ZCAT';
 if(!['ZCAT','ZEC','PONS'].includes(coin))return fail('Unknown coin','INVALID_COIN',400);
 const start=coin==='PONS'?'2026-07-01':'2026-07-25';
 if(!process.env.DATABASE_URL)return ok({status:'not_configured',posts:[],days:[],total:0,limit:10000});
 const sql=neon(process.env.DATABASE_URL);
 try{
  const exists=await sql`SELECT to_regclass('public.crypto_social_posts') AS relation`;
  if(!exists[0]?.relation)return ok({status:'not_started',posts:[],days:[],total:0,limit:10000});
  const [posts,days,total]=await Promise.all([
   sql`WITH matched AS (SELECT DISTINCT p.* FROM crypto_social_posts p JOIN crypto_social_matches m ON m.post_id=p.id
    JOIN crypto_social_windows w ON w.id=m.window_id WHERE w.coin=${coin} AND p.posted_at>=${start}::timestamptz)
    SELECT p.*,a.handle,coalesce((SELECT jsonb_agg(jsonb_build_object('target_id',e.target_id,'target',coalesce(b.handle,e.target_id),'kind',e.kind) ORDER BY e.target_id,e.kind)
     FROM crypto_social_edges e LEFT JOIN crypto_social_accounts b ON b.id=e.target_id WHERE e.post_id=p.id),'[]'::jsonb) AS edges
    FROM matched p JOIN crypto_social_accounts a ON a.id=p.author_id ORDER BY p.posted_at,p.id LIMIT 10000`,
   sql`SELECT (start_at AT TIME ZONE 'UTC')::date::text AS day,count(*)::int AS windows,
    count(*) FILTER(WHERE pages>0)::int AS searched,count(*) FILTER(WHERE status='search_exhausted')::int AS exhausted
    FROM crypto_social_windows WHERE coin=${coin} AND start_at>=${start}::timestamptz AND query<>'timeline text match' GROUP BY 1 ORDER BY 1`,
   sql`SELECT count(DISTINCT p.id)::int AS total FROM crypto_social_posts p JOIN crypto_social_matches m ON m.post_id=p.id
    JOIN crypto_social_windows w ON w.id=m.window_id WHERE w.coin=${coin} AND p.posted_at>=${start}::timestamptz`,
  ]);
  return ok({status:'ready',posts,days,total:total[0].total,limit:10000,start,end:new Date().toISOString().slice(0,10)});
 }catch{return fail('Saved timeline is temporarily unavailable','RUN_READ_FAILED',503);}
}

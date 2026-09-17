import {neon} from '@neondatabase/serverless';
import {ok,fail} from '@/lib/server/api-utils';
import {rankVoices,type VoicePost} from '@/lib/crypto-voices';
export const dynamic='force-dynamic';
export async function GET(request:Request){
 const coin=new URL(request.url).searchParams.get('coin')??'ZCAT';if(!['ZCAT','PONS','DPONS','ZEC','STANDARD'].includes(coin))return fail('Invalid coin','INVALID_COIN',400);
 if(!process.env.DATABASE_URL)return ok({...rankVoices([],coin),total:0,unfinished:0,snapshots:[]});
 try{const sql=neon(process.env.DATABASE_URL);const posts=await sql`SELECT p.*,a.handle,a.followers::float,a.observed_at AS followers_observed_at,
 coalesce((SELECT jsonb_agg(jsonb_build_object('target_id',e.target_id,'target',e.target_id,'kind',e.kind)) FROM crypto_social_edges e WHERE e.post_id=p.id),'[]'::jsonb) AS edges,count(*) OVER()::int AS total
 FROM crypto_social_posts p JOIN crypto_social_accounts a ON a.id=p.author_id WHERE EXISTS(SELECT 1 FROM crypto_social_matches m JOIN crypto_social_windows w ON w.id=m.window_id WHERE m.post_id=p.id AND w.coin=${coin}) ORDER BY p.posted_at,p.id LIMIT 50000`;
 const coverage=await sql`SELECT count(*) FILTER(WHERE status!='search_exhausted')::int AS unfinished FROM crypto_social_windows WHERE coin=${coin} AND query!='timeline text match'`;
 const exists=await sql`SELECT to_regclass('public.crypto_voice_snapshots') AS relation`;
 const snapshots=exists[0]?.relation?await sql`SELECT week,created_at,model,selection,baseline,evaluation FROM crypto_voice_snapshots WHERE coin=${coin} ORDER BY week DESC LIMIT 4`:[];
 return ok({...rankVoices(posts as VoicePost[],coin),total:posts[0]?.total??0,unfinished:coverage[0]?.unfinished??0,snapshots});
 }catch{return fail('Voice evidence unavailable','VOICES_FAILED',503);}
}

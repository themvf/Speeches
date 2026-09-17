import {neon} from '@neondatabase/serverless';
import {ok,fail} from '@/lib/server/api-utils';
import {rankVoices,VOICE_VERSION,type VoicePost} from '@/lib/crypto-voices';
import {isCoin} from '@/lib/crypto-coins';
import {VOICES_QUERY,VOICES_UNFINISHED_QUERY} from '@/lib/server/crypto-watchers-query';
import {readRankingSnapshot,withCdnCache} from '@/lib/server/crypto-ranking-cache';
export const dynamic='force-dynamic';
const LIMIT=50000;
export async function GET(request:Request){
 const coin=new URL(request.url).searchParams.get('coin')??'ZCAT';if(!isCoin(coin))return fail('Invalid coin','INVALID_COIN',400);
 if(!process.env.DATABASE_URL)return ok({...rankVoices([],coin),total:0,unfinished:0,snapshots:[]});
 try{const sql=neon(process.env.DATABASE_URL);
 const exists=await sql`SELECT to_regclass('public.crypto_voice_snapshots') AS relation`;
 const snapshots=exists[0]?.relation?await sql`SELECT week,created_at,model,selection,baseline,evaluation FROM crypto_voice_snapshots WHERE coin=${coin} ORDER BY week DESC LIMIT 4`:[];
 // Collection-time snapshot (every saved row, ranked by the same code) when present; otherwise rank live over a bounded load.
 const cached=await readRankingSnapshot<ReturnType<typeof rankVoices>&{total:number;unfinished:number}>(sql,'voices:'+coin,VOICE_VERSION);
 if(cached)return withCdnCache(ok({...cached.payload,snapshots,source:'snapshot',computedAt:cached.computed_at}));
 const posts=await sql.query(VOICES_QUERY(LIMIT),[coin]) as (VoicePost&{total:number})[];
 const coverage=await sql.query(VOICES_UNFINISHED_QUERY,[coin]) as {unfinished:number}[];
 return withCdnCache(ok({...rankVoices(posts,coin),total:posts[0]?.total??0,unfinished:coverage[0]?.unfinished??0,snapshots,source:'live'}));
 }catch{return fail('Voice evidence unavailable','VOICES_FAILED',503);}
}

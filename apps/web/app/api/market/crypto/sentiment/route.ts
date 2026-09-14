import {neon} from '@neondatabase/serverless';
import {ok,fail} from '@/lib/server/api-utils';
export const dynamic='force-dynamic';
export async function GET(){
 if(!process.env.DATABASE_URL)return ok({status:'not_configured',annotations:[]});
 try{const sql=neon(process.env.DATABASE_URL);const exists=await sql`SELECT to_regclass('public.crypto_post_sentiment') AS relation`;
 if(!exists[0]?.relation)return ok({status:'awaiting_pilot',annotations:[]});
 const annotations=await sql`SELECT post_id,coin,label,confidence::float,explanation,model,version,observed_at FROM crypto_post_sentiment WHERE version='coin-stance-v1' ORDER BY observed_at DESC LIMIT 10000`;
 return ok({status:'pilot',annotations});}catch{return fail('Sentiment could not be loaded','SENTIMENT_READ_FAILED',503);}
}

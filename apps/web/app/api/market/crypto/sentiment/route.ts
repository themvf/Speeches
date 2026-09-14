import {neon} from '@neondatabase/serverless';
import {ok,fail} from '@/lib/server/api-utils';
export const dynamic='force-dynamic';
export async function GET(){
 if(!process.env.DATABASE_URL)return ok({status:'not_configured',annotations:[]});
 try{const sql=neon(process.env.DATABASE_URL);const [exists]=await sql`SELECT to_regclass('public.crypto_post_sentiment') AS pilot,to_regclass('public.crypto_assistant_sentiment_reviews') AS reviews`;
 const pilot=exists?.pilot?await sql`SELECT post_id,coin,label,confidence::float,explanation,model,version,observed_at FROM crypto_post_sentiment WHERE version='coin-stance-v1' ORDER BY observed_at DESC LIMIT 10000`:[];
 const reviews=exists?.reviews?await sql`SELECT post_id,coin,label,NULL::float AS confidence,confidence_label,explanation,excluded,model,version,observed_at FROM crypto_assistant_sentiment_reviews WHERE version='assistant-zcat-50-v1' ORDER BY observed_at DESC`:[];
 // The explicit evaluation sample takes precedence, preserving the automated records separately.
 const annotations=new Map(pilot.map(a=>[`${a.coin}:${a.post_id}`,a]));for(const a of reviews)annotations.set(`${a.coin}:${a.post_id}`,a);
 return ok({status:reviews.length?'assistant_review':exists?.pilot?'pilot':'awaiting_pilot',annotations:[...annotations.values()]});}catch{return fail('Sentiment could not be loaded','SENTIMENT_READ_FAILED',503);}
}

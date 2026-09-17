import type {NeonQueryFunction} from '@neondatabase/serverless';
// Collection-time snapshots written by crypto_rankings_cache.py. A route serves the snapshot when present and
// falls back to live computation when the table or row is missing, so a deploy never waits on a collector.
export type RankingSnapshot<T>={payload:T;computed_at:string;version:string;source:'snapshot'};
export async function readRankingSnapshot<T>(sql:NeonQueryFunction<false,false>,key:string,version:string):Promise<RankingSnapshot<T>|null>{
 const exists=await sql`SELECT to_regclass('public.crypto_ranking_cache') AS relation`;
 if(!exists[0]?.relation)return null;
 const rows=await sql`SELECT payload,computed_at,version FROM crypto_ranking_cache WHERE key=${key} AND version=${version}`;
 if(!rows[0])return null;
 return {payload:rows[0].payload as T,computed_at:String(rows[0].computed_at),version:String(rows[0].version),source:'snapshot'};
}
// Collection runs every two to six hours; a short shared cache keeps repeat page loads off Neon.
export function withCdnCache<T extends Response>(response:T,seconds=300):T{response.headers.set('Cache-Control',`public, s-maxage=${seconds}, stale-while-revalidate=${seconds*6}`);return response;}

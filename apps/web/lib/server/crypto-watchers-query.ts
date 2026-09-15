// Shared by the read-only route and the saved-corpus review job.
export const WATCHER_QUERY = `
WITH matched AS (
 SELECT p.id,array_agg(DISTINCT w.coin ORDER BY w.coin) AS coins
 FROM crypto_social_posts p JOIN crypto_social_matches m ON m.post_id=p.id
 JOIN crypto_social_windows w ON w.id=m.window_id
 WHERE w.coin IN ('ZCAT','ZEC','PONS','DPONS','STANDARD') GROUP BY p.id
), profiles AS MATERIALIZED (
 SELECT DISTINCT ON(account_id) account_id,bio FROM crypto_social_profile_history
 WHERE bio IS NOT NULL AND available ORDER BY account_id,observed_at DESC,request_id DESC
)
SELECT p.id,p.author_id,p.text,p.url,p.posted_at,p.kind,a.handle,a.followers::float AS followers,
 a.observed_at AS followers_observed_at,b.bio,m.coins,'[]'::json AS edges,count(*) OVER()::int AS corpus_total
 FROM matched m JOIN crypto_social_posts p ON p.id=m.id JOIN crypto_social_accounts a ON a.id=p.author_id
 LEFT JOIN profiles b ON b.account_id=a.id ORDER BY p.posted_at DESC,p.id LIMIT 50000
`;

-- All graph metrics use the full stored coin graph, before display limits.
CREATE OR REPLACE VIEW crypto_social_account_metrics AS
WITH matched AS (
 SELECT DISTINCT w.coin,p.* FROM crypto_social_posts p
 JOIN crypto_social_matches m ON m.post_id=p.id JOIN crypto_social_windows w ON w.id=m.window_id
 WHERE p.posted_at>=now()-interval '28 days'
), latest_metrics AS (
 SELECT DISTINCT ON(s.post_id) s.* FROM crypto_social_snapshots s
 ORDER BY s.post_id,s.observed_at DESC,s.request_id DESC
), age_metrics AS (
 SELECT DISTINCT ON(s.post_id) s.* FROM crypto_social_snapshots s JOIN crypto_social_posts p ON p.id=s.post_id
 WHERE s.observed_at-p.posted_at BETWEEN interval '24 hours' AND interval '30 hours'
 ORDER BY s.post_id,s.observed_at,s.request_id
), activity AS (
 SELECT m.coin,m.author_id,count(*)::int AS posts,
 count(DISTINCT (m.posted_at AT TIME ZONE 'UTC')::date)::int AS active_days,
 count(*) FILTER(WHERE m.kind='original')::int AS originals,
 count(*) FILTER(WHERE m.kind='reply')::int AS replies,
 count(*) FILTER(WHERE m.kind='quote')::int AS quotes,
 count(*) FILTER(WHERE m.kind='repost')::int AS reposts,
 percentile_cont(0.5) WITHIN GROUP(ORDER BY s.likes) AS median_likes,
 percentile_cont(0.5) WITHIN GROUP(ORDER BY s.likes+s.replies+s.quotes+s.reposts) AS median_engagement,
 count(s.likes+s.replies+s.quotes+s.reposts)::int AS engagement_samples,
 percentile_cont(0.5) WITHIN GROUP(ORDER BY a.likes+a.replies+a.quotes+a.reposts) AS median_engagement_24_30h,
 count(a.likes+a.replies+a.quotes+a.reposts)::int AS age_matched_samples,
 min(m.posted_at) AS first_post_at,max(m.posted_at) AS last_post_at
 FROM matched m LEFT JOIN latest_metrics s ON s.post_id=m.id LEFT JOIN age_metrics a ON a.post_id=m.id
 GROUP BY m.coin,m.author_id
), connections AS (
 SELECT DISTINCT m.coin,e.source_id,e.target_id,e.kind,m.posted_at,m.id,m.url
 FROM crypto_social_edges e JOIN matched m ON m.id=e.post_id WHERE e.source_id<>e.target_id
), pair_days AS (
 SELECT coin,target_id,source_id,count(DISTINCT (posted_at AT TIME ZONE 'UTC')::date)::int AS days
 FROM connections GROUP BY coin,target_id,source_id
), concentration AS (
 SELECT coin,target_id,source_id,days,row_number() OVER(PARTITION BY coin,target_id ORDER BY days DESC,source_id) AS rank
 FROM pair_days
), incoming AS (
 SELECT coin,target_id,count(DISTINCT source_id)::int AS participants,
 count(DISTINCT source_id) FILTER(WHERE kind IN ('quote','repost'))::int AS amplifiers,
 count(DISTINCT source_id) FILTER(WHERE posted_at>=now()-interval '7 days')::int AS participants_7d,
 count(DISTINCT source_id) FILTER(WHERE posted_at>=now()-interval '14 days' AND posted_at<now()-interval '7 days')::int AS participants_previous_7d,
 min(url) AS evidence
 FROM connections GROUP BY coin,target_id
), breadth AS (
 SELECT coin,account_id,count(DISTINCT counterpart)::int AS connections FROM (
 SELECT coin,target_id AS account_id,source_id AS counterpart FROM connections
 UNION SELECT coin,source_id,target_id FROM connections
 ) x GROUP BY coin,account_id
), repeat_attention AS (
 SELECT coin,target_id,count(*) FILTER(WHERE days>=2)::int AS repeat_participants,
 round(sum(days) FILTER(WHERE rank<=5)::numeric/nullif(sum(days),0)*100,1)::float AS top5_attention_percent
 FROM concentration GROUP BY coin,target_id
), reciprocal AS (
 SELECT a.coin,a.target_id,count(DISTINCT a.source_id)::int AS reciprocal_participants
 FROM connections a JOIN connections b ON a.coin=b.coin AND a.source_id=b.target_id AND a.target_id=b.source_id
 GROUP BY a.coin,a.target_id
)
SELECT c.coin,a.id,a.handle,a.name,c.category,c.tracked,c.reason,c.first_seen_at,
 h.followers,h.following,h.available,h.observed_at AS profile_observed_at,
 h.bio,h.source AS profile_source,
 baseline.followers AS followers_7d_ago,baseline.observed_at AS baseline_at,
 older.followers AS followers_14d_ago,older.observed_at AS older_baseline_at,
 CASE WHEN h.available AND h.followers IS NOT NULL AND baseline.available THEN h.followers-baseline.followers END AS growth_7d,
 CASE WHEN h.available AND baseline.available AND baseline.followers>0
 THEN round((h.followers-baseline.followers)::numeric/baseline.followers*100,2)::float END AS growth_percent_7d,
 CASE WHEN h.available AND baseline.available AND older.available AND baseline.followers IS NOT NULL AND older.followers IS NOT NULL
 THEN ((h.followers-baseline.followers)::numeric/nullif(extract(epoch FROM h.observed_at-baseline.observed_at)/86400,0)
 -(baseline.followers-older.followers)::numeric/nullif(extract(epoch FROM baseline.observed_at-older.observed_at)/86400,0))::float END AS growth_acceleration,
 coalesce(v.posts,0)::int AS posts,coalesce(v.active_days,0)::int AS active_days,
 v.originals,v.replies,v.quotes,v.reposts,v.median_likes,v.median_engagement,v.engagement_samples,
 v.median_engagement_24_30h,v.age_matched_samples,v.first_post_at,v.last_post_at,
 coalesce(i.participants,0)::int AS participants,coalesce(i.amplifiers,0)::int AS amplifiers,
 coalesce(i.participants_7d,0)::int AS participants_7d,
 coalesce(i.participants_previous_7d,0)::int AS participants_previous_7d,
 coalesce(b.connections,0)::int AS connections,coalesce(r.repeat_participants,0)::int AS repeat_participants,
 r.top5_attention_percent,coalesce(re.reciprocal_participants,0)::int AS reciprocal_participants,i.evidence,
 (SELECT count(*)::int FROM crypto_social_profile_history ph WHERE ph.account_id=a.id AND ph.source='daily_profile') AS daily_observations
FROM crypto_social_candidates c JOIN crypto_social_accounts a ON a.id=c.account_id
LEFT JOIN LATERAL (
 SELECT * FROM crypto_social_profile_history ph WHERE ph.account_id=a.id
 ORDER BY ph.observed_at DESC,ph.request_id DESC LIMIT 1
) h ON true
LEFT JOIN LATERAL (
 SELECT * FROM crypto_social_profile_history ph WHERE ph.account_id=a.id AND ph.source IN ('daily_profile','daily_profile_missing')
 AND (ph.observed_at AT TIME ZONE 'UTC')::date=(h.observed_at AT TIME ZONE 'UTC')::date-7
 ORDER BY ph.observed_at DESC LIMIT 1
) baseline ON true
LEFT JOIN LATERAL (
 SELECT * FROM crypto_social_profile_history ph WHERE ph.account_id=a.id AND ph.source IN ('daily_profile','daily_profile_missing')
 AND (ph.observed_at AT TIME ZONE 'UTC')::date=(h.observed_at AT TIME ZONE 'UTC')::date-14
 ORDER BY ph.observed_at DESC LIMIT 1
) older ON true
LEFT JOIN activity v ON v.coin=c.coin AND v.author_id=a.id
LEFT JOIN incoming i ON i.coin=c.coin AND i.target_id=a.id
LEFT JOIN breadth b ON b.coin=c.coin AND b.account_id=a.id
LEFT JOIN repeat_attention r ON r.coin=c.coin AND r.target_id=a.id
LEFT JOIN reciprocal re ON re.coin=c.coin AND re.target_id=a.id;

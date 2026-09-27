import {neon} from '@neondatabase/serverless';
import {LIVE_STATUS,withAmounts,type IntelPayload,type IntelQuery,type IntelRow} from '@/lib/bp-intel';

/** Read-only BP holder intelligence. Never runs DDL; returns schema_pending until the collector migrates.
 *  Every template below is also executed by tests/test_bp_intel_integration.py against PostgreSQL. */
export async function readBpIntel(q:IntelQuery):Promise<IntelPayload>{
 const empty:IntelPayload={status:'not_configured',section:q.section,meta:null,rows:[],extra:{}};
 if(!process.env.DATABASE_URL)return empty;
 const sql=neon(process.env.DATABASE_URL);
 const schema=await sql`SELECT to_regclass('public.bp_cohorts') AS cohorts, to_regclass('public.bp_reconciliation') AS reconciliation`;
 if(!schema[0]?.cohorts||!schema[0]?.reconciliation)return {...empty,status:'schema_pending'};
 const meaningful=Number(process.env.BP_MEANINGFUL_USD??'100');
 const threshold=Number.isFinite(meaningful)&&meaningful>0?meaningful:100;
 const [versions,runs,latestRuns]=await Promise.all([
  sql`SELECT version_id,kind,source_date::text AS source_date,effective_at,size,entered,left_count,rank_changed,queued,
      entrant_cap_bound,bootstrap,status,exclusion_fingerprint,excluded_count,eligible_count,target_size,exit_rank,exit_runs,
      entrant_cap,derived_from_version,approved_at,approved_by,methodology
      FROM bp_cohorts ORDER BY kind='original' DESC,source_date DESC LIMIT 90`,
  sql`SELECT r.run_id,r.started_at,r.completed_at,r.status,r.assets_attempted AS wallets_attempted,r.assets_succeeded AS wallets_complete,
      r.assets_failed AS wallets_incomplete,r.assets_skipped AS wallets_skipped,r.source_errors
      FROM backpack_ingestion_runs r WHERE COALESCE(to_jsonb(r)->>'job','daily')='bp_intel'
      AND EXISTS(SELECT 1 FROM bp_portfolio_wallets w WHERE w.run_id=r.run_id) ORDER BY r.started_at DESC LIMIT 48`,
  sql`SELECT r.run_id,r.started_at,r.completed_at,r.status,r.source_errors FROM backpack_ingestion_runs r
      WHERE COALESCE(to_jsonb(r)->>'job','daily')='bp_intel' ORDER BY r.started_at DESC LIMIT 1`,
 ]);
 // An explicitly requested version or read outside the listed window is looked up directly; if it does not
 // exist the answer is not_found, never a silent switch to another scope or to empty (zero-looking) results.
 const cohortId=q.cohort,runId=q.run;
 let version=(cohortId?versions.find(v=>Number(v.version_id)===cohortId):versions.find(v=>v.kind==='current')??versions[0])??null;
 if(cohortId&&!version)version=(await sql`SELECT version_id,kind,source_date::text AS source_date,effective_at,size,entered,left_count,rank_changed,
   queued,entrant_cap_bound,bootstrap,status,exclusion_fingerprint,excluded_count,eligible_count,target_size,exit_rank,exit_runs,
   entrant_cap,derived_from_version,approved_at,approved_by,methodology FROM bp_cohorts WHERE version_id=${cohortId}`)[0]??null;
 if(cohortId&&!version)return {...empty,status:'not_found'};
 if(!version)return {...empty,status:'no_cohort',meta:{versions,version:null,runs,run:null,latestRun:latestRuns[0]??null,coverage:null,meaningfulUsd:threshold,live:LIVE_STATUS}};
 let run=(runId?runs.find(r=>r.run_id===runId):runs[0])??null;
 if(runId&&!run)run=(await sql`SELECT r.run_id,r.started_at,r.completed_at,r.status,r.assets_attempted AS wallets_attempted,
   r.assets_succeeded AS wallets_complete,r.assets_failed AS wallets_incomplete,r.assets_skipped AS wallets_skipped,r.source_errors
   FROM backpack_ingestion_runs r WHERE r.run_id=${runId}::uuid AND COALESCE(to_jsonb(r)->>'job','daily')='bp_intel'
   AND EXISTS(SELECT 1 FROM bp_portfolio_wallets w WHERE w.run_id=r.run_id)`)[0]??null;
 if(runId&&!run)return {...empty,status:'not_found'};
 const v=Number(version.version_id),r=run?String(run.run_id):null,mint=q.mint,wallet=q.wallet,days=q.days;
 const coverage=await sql`SELECT (SELECT count(*) FROM bp_cohort_members WHERE version_id=${v} AND member)::int AS members,
   count(w.wallet_address) FILTER(WHERE w.status='complete')::int AS read_complete,
   count(w.wallet_address) FILTER(WHERE w.status='partial')::int AS read_partial,
   count(w.wallet_address) FILTER(WHERE w.status='unavailable')::int AS read_unavailable,
   count(w.wallet_address) FILTER(WHERE w.status='oversized')::int AS read_oversized,
   count(*) FILTER(WHERE w.wallet_address IS NULL)::int AS read_missing,
   count(c.wallet_address) FILTER(WHERE c.backfill_status='complete')::int AS history_complete,
   count(c.wallet_address) FILTER(WHERE c.backfill_status='capped' OR c.poll_status='capped' OR jsonb_array_length(c.gaps)>0)::int AS history_gaps,
   count(m.wallet_address) FILTER(WHERE c.wallet_address IS NULL OR c.backfill_status IN ('pending','incomplete'))::int AS history_pending,
   min(c.last_poll_at) AS oldest_poll,max(c.last_poll_at) AS newest_poll,
   (SELECT count(*) FROM bp_portfolio_balances b JOIN bp_cohort_members x ON x.wallet_address=b.wallet_address AND x.version_id=${v} AND x.member
     WHERE b.run_id=${r}::uuid AND b.pricing_status='stale')::int AS stale_prices,
   (SELECT count(*) FROM bp_reconciliation WHERE status='discrepancy' AND date>=current_date-7)::int AS reconciliation_discrepancies
   FROM bp_cohort_members m LEFT JOIN bp_portfolio_wallets w ON w.wallet_address=m.wallet_address AND w.run_id=${r}::uuid
   LEFT JOIN bp_history_coverage c ON c.wallet_address=m.wallet_address WHERE m.version_id=${v} AND m.member`;
 const meta={versions,version,runs,run,latestRun:latestRuns[0]??null,coverage:coverage[0]??null,meaningfulUsd:threshold,live:LIVE_STATUS};
 let rows:IntelRow[]=[];const extra:Record<string,IntelRow[]>={};
 if(q.section==='roster'){
  const [members,raw]=await Promise.all([
   sql`SELECT m.wallet_address,m.member,m.rank,m.raw_balance::text AS raw_balance,m.previous_rank,m.event,m.below_exit_runs,m.queued_since,
       m.exit_reason,r.decimals,r.label,r.label_confidence,r.label_entity,r.label_source,l.label AS current_label,l.confidence AS current_confidence
       FROM bp_cohort_members m JOIN bp_cohorts c USING(version_id)
       LEFT JOIN bp_holder_rankings r ON r.source_date=c.source_date AND r.ranking='filtered' AND r.wallet_address=m.wallet_address
       LEFT JOIN backpack_wallet_labels l ON l.wallet_address=m.wallet_address
       WHERE m.version_id=${v} ORDER BY m.member DESC,m.rank NULLS LAST,m.wallet_address`,
   sql`SELECT rank,wallet_address,raw_balance::text AS raw_balance,decimals,label,label_confidence,label_entity,label_source,excluded
       FROM bp_holder_rankings WHERE ranking='raw' AND rank<=200
       AND source_date=(SELECT source_date FROM bp_cohorts WHERE version_id=${v}) ORDER BY rank`,
  ]);
  rows=members;extra.raw=withAmounts('roster',raw);
 }
 if(q.section==='overlap'||q.section==='overview'){
  rows=await sql`WITH members AS (SELECT wallet_address FROM bp_cohort_members WHERE version_id=${v} AND member),
   wallets AS (SELECT w.wallet_address,w.priced_value_usd FROM bp_portfolio_wallets w JOIN members USING(wallet_address)
     WHERE w.run_id=${r}::uuid AND w.status IN ('complete','partial')),
   bal AS (SELECT b.*,w.priced_value_usd FROM bp_portfolio_balances b JOIN wallets w USING(wallet_address) WHERE b.run_id=${r}::uuid),
   swaps AS (SELECT e.* FROM bp_economic_events e JOIN members USING(wallet_address) WHERE e.kind='swap'
     AND e.tier IN ('parsed_swap','inferred_swap') AND NOT e.pre_membership AND e.block_time>=now()-interval '7 days'),
   buys AS (SELECT output_mint AS mint,
     count(DISTINCT wallet_address) FILTER(WHERE new_position AND block_time>=now()-interval '1 hour')::int AS new_buyers_1h,
     count(DISTINCT wallet_address) FILTER(WHERE new_position AND block_time>=now()-interval '24 hours')::int AS new_buyers_24h,
     count(DISTINCT wallet_address) FILTER(WHERE new_position)::int AS new_buyers_7d,
     count(DISTINCT wallet_address)::int AS buyers_7d,sum(output_raw) AS purchased_raw_7d,sum(usd_value) AS purchased_usd_7d,
     count(*) FILTER(WHERE usd_value IS NULL)::int AS unpriced_purchases_7d,count(*) FILTER(WHERE tier='inferred_swap')::int AS inferred_purchases_7d
     FROM swaps GROUP BY output_mint),
   sells AS (SELECT input_mint AS mint,count(DISTINCT wallet_address)::int AS sellers_7d,sum(input_raw) AS sold_raw_7d,
     sum(usd_value) AS sold_usd_7d,count(*) FILTER(WHERE usd_value IS NULL)::int AS unpriced_sales_7d FROM swaps GROUP BY input_mint),
   agg AS (SELECT mint,count(*) FILTER(WHERE raw_amount>0)::int AS holders,
     count(*) FILTER(WHERE value_usd>=${threshold})::int AS meaningful_holders,
     count(*) FILTER(WHERE balance_visibility='partial')::int AS partial_visibility,
     count(*) FILTER(WHERE value_usd IS NULL AND raw_amount>0)::int AS unpriced_holders,
     sum(value_usd) AS combined_value_usd,
     percentile_cont(0.5) WITHIN GROUP(ORDER BY value_usd/priced_value_usd) FILTER(WHERE value_usd IS NOT NULL AND priced_value_usd>0) AS median_weight,
     max(raw_amount)/NULLIF(sum(raw_amount),0) AS largest_owner_share,sum(raw_amount)::text AS total_raw,max(decimals) AS decimals,
     max(price) AS price,max(price_at) AS price_at,bool_or(pricing_status='stale') AS stale_price,max(holding_class) AS holding_class
     FROM bal GROUP BY mint),
   mints AS (SELECT mint FROM agg UNION SELECT mint FROM buys UNION SELECT mint FROM sells)
   SELECT x.mint,COALESCE(a.holders,0) AS holders,COALESCE(a.meaningful_holders,0) AS meaningful_holders,a.partial_visibility,
     a.unpriced_holders,a.combined_value_usd,a.median_weight,a.largest_owner_share,a.total_raw,COALESCE(a.decimals,t.decimals) AS decimals,
     a.price,a.price_at,a.stale_price,(SELECT count(*) FROM members)::int AS cohort_size,(SELECT count(*) FROM wallets)::int AS wallets_read,
     t.symbol,t.name,COALESCE(t.asset_class,a.holding_class) AS asset_class,t.is_sol,t.is_stable,t.is_bp,t.spam_class,t.spam_reason,t.class_reason,
     t.extension_flags,t.liquidity_usd,t.market_created_at,b.new_buyers_1h,b.new_buyers_24h,b.new_buyers_7d,b.buyers_7d,
     b.purchased_raw_7d::text AS purchased_raw_7d,b.purchased_usd_7d,b.unpriced_purchases_7d,b.inferred_purchases_7d,s.sellers_7d,
     s.sold_raw_7d::text AS sold_raw_7d,s.sold_usd_7d,s.unpriced_sales_7d,(COALESCE(b.purchased_raw_7d,0)-COALESCE(s.sold_raw_7d,0))::text AS net_raw_7d
   FROM mints x LEFT JOIN agg a ON a.mint=x.mint LEFT JOIN bp_tracked_assets t ON t.network='solana-mainnet' AND t.mint=x.mint
   LEFT JOIN buys b ON b.mint=x.mint LEFT JOIN sells s ON s.mint=x.mint
   ORDER BY COALESCE(a.meaningful_holders,0) DESC,COALESCE(a.holders,0) DESC,a.combined_value_usd DESC NULLS LAST,x.mint LIMIT 1500`;
 }
 if(q.section==='activity'||q.section==='overview'||q.section==='token'||q.section==='wallet'){
  const events=await sql`SELECT e.signature,e.wallet_address,e.event_index,e.slot,e.block_time,e.kind,e.tier,e.input_mint,
   e.input_raw::text AS input_raw,e.input_decimals,e.output_mint,e.output_raw::text AS output_raw,e.output_decimals,e.usd_value,
   e.valuation_source,e.valuation_status,e.pre_input_raw::text AS pre_input_raw,e.post_input_raw::text AS post_input_raw,
   e.pre_output_raw::text AS pre_output_raw,e.post_output_raw::text AS post_output_raw,e.pre_balance_source,e.new_position,e.ata_created,
   e.first_observed_purchase,e.re_entry,e.first_group_purchase,e.recently_launched,e.pre_membership,e.owner_signed,e.venue,e.programs,
   e.detail,e.parser_version,e.finality,ti.symbol AS input_symbol,ta.symbol AS output_symbol
   FROM bp_economic_events e JOIN bp_cohort_members m ON m.wallet_address=e.wallet_address AND m.version_id=${v} AND m.member
   LEFT JOIN bp_tracked_assets ti ON ti.network=e.network AND ti.mint=e.input_mint
   LEFT JOIN bp_tracked_assets ta ON ta.network=e.network AND ta.mint=e.output_mint
   WHERE e.block_time>=now()-make_interval(days=>${days}::int)
   AND (${mint}::text IS NULL OR e.input_mint=${mint}::text OR e.output_mint=${mint}::text)
   AND (${wallet}::text IS NULL OR e.wallet_address=${wallet}::text)
   ORDER BY e.block_time DESC,e.signature,e.event_index LIMIT 1000`;
  if(q.section==='activity')rows=events;else extra.activity=events;
 }
 if(q.section==='alerts'||q.section==='overview'||q.section==='token'){
  const alerts=await sql`SELECT a.alert_key,a.rule,a.cohort_version_id,a.mint,COALESCE(a.display_name,t.symbol) AS symbol,a.window_start,a.window_end,
   a.wallet_count,a.wallets,a.inferred_wallets,a.lowest_tier,a.quantity_raw::text AS quantity_raw,a.decimals,a.usd_value,a.valuation_status,
   a.signatures,a.finality,a.data_through,a.detail,a.first_raised_at,a.updated_at
   FROM bp_alerts a LEFT JOIN bp_tracked_assets t ON t.network='solana-mainnet' AND t.mint=a.mint
   WHERE a.cohort_version_id=${v} AND (${mint}::text IS NULL OR a.mint=${mint}::text) ORDER BY a.window_start DESC LIMIT 300`;
  if(q.section==='alerts')rows=alerts;else extra.alerts=alerts;
 }
 if(q.section==='wallet'){
  const [holdings,read,history]=await Promise.all([
   sql`SELECT b.mint,b.raw_amount::text AS raw_amount,b.decimals,b.ui_amount,b.token_accounts,b.slot,b.observed_at,b.price,b.price_at,
       b.price_source,b.pricing_status,b.value_usd,b.frozen,b.balance_visibility,b.holding_class,b.spam_class,b.dust,b.classification_reason,
       t.symbol,t.name,t.is_sol,t.is_stable,t.is_bp FROM bp_portfolio_balances b
       LEFT JOIN bp_tracked_assets t ON t.network=b.network AND t.mint=b.mint
       WHERE b.run_id=${r}::uuid AND b.wallet_address=${wallet}::text ORDER BY b.value_usd DESC NULLS LAST,b.mint`,
   sql`SELECT * FROM bp_portfolio_wallets WHERE run_id=${r}::uuid AND wallet_address=${wallet}::text`,
   sql`SELECT * FROM bp_history_coverage WHERE wallet_address=${wallet}::text`,
  ]);
  rows=withAmounts('wallet',holdings);extra.read=read;extra.history=history;
 }
 if(q.section==='token'){
  const [holders,asset]=await Promise.all([
   sql`SELECT m.rank,b.wallet_address,b.raw_amount::text AS raw_amount,b.decimals,b.value_usd,b.pricing_status,b.balance_visibility,b.observed_at
       FROM bp_portfolio_balances b JOIN bp_cohort_members m ON m.wallet_address=b.wallet_address AND m.version_id=${v} AND m.member
       WHERE b.run_id=${r}::uuid AND b.mint=${mint}::text ORDER BY b.raw_amount DESC,b.wallet_address`,
   sql`SELECT * FROM bp_tracked_assets WHERE network='solana-mainnet' AND mint=${mint}::text`,
  ]);
  rows=withAmounts('token',holders);extra.asset=asset;
 }
 return {status:'ready',section:q.section,meta,rows,extra};
}

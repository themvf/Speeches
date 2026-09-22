import {neon} from '@neondatabase/serverless';
export async function readBackpackOperations(){
 if(!process.env.DATABASE_URL)return {status:'not_configured'};
 const sql=neon(process.env.DATABASE_URL);
 const schema=await sql`SELECT to_regclass('public.backpack_readiness_checks') AS ready,
   to_regclass('public.backpack_storage_observations') AS storage`;
 if(!schema[0]?.ready||!schema[0]?.storage)return {status:'schema_pending'};
 const [checks,runs,assets,reconciliation,storage,alerts]=await Promise.all([
  sql`SELECT check_name,status,detail,checked_at FROM backpack_readiness_checks
      WHERE run_id=(SELECT run_id FROM backpack_readiness_checks ORDER BY checked_at DESC LIMIT 1) ORDER BY check_name`,
  sql`SELECT snapshot_date,status,started_at,completed_at,assets_attempted,assets_succeeded,assets_failed,assets_skipped
      FROM backpack_ingestion_runs ORDER BY started_at DESC LIMIT 5`,
  sql`SELECT count(*) FILTER(WHERE asset_type<>'bp' AND active AND verification_status<>'pending') AS approved,
      (SELECT max(date)::text FROM backpack_asset_daily_snapshots) AS latest_capture FROM backpack_assets`,
  sql`SELECT a.token_symbol,s.date::text,s.holders_complete,s.token_supply,c.balance_tokens,
      c.aggregates_validated,s.reference_aum_usd,s.token_supply*s.underlying_price AS reproduced_aum,
      s.underlying_price_timestamp,s.quality_status
      FROM backpack_assets a JOIN backpack_asset_daily_snapshots s ON s.asset_id=a.id
      LEFT JOIN backpack_holder_checkpoints c ON c.asset_id=s.asset_id AND c.date=s.date
      WHERE s.date=(SELECT max(date) FROM backpack_asset_daily_snapshots) ORDER BY a.token_symbol`,
  sql`SELECT relation_name,total_bytes,index_bytes,estimated_rows,measured_at FROM backpack_storage_observations
      WHERE date=(SELECT max(date) FROM backpack_storage_observations) ORDER BY total_bytes DESC`,
  sql`SELECT date::text,metric,detail FROM backpack_operational_alerts ORDER BY date DESC LIMIT 30`,
 ]);
 return {status:'ready',checks,runs,assets:assets[0],reconciliation,storage,alerts};
}

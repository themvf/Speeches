import {neon} from '@neondatabase/serverless';
import type {MonitorData, Quality, Row} from '@/lib/backpack';
export async function readBackpack(assetId?:number):Promise<MonitorData>{
 const empty:MonitorData={status:'not_configured',asOf:null,assets:[],history:[],ecosystem:[],bp:[],quality:[],quotes:[],holders:[],dex:[],runs:[],usage:[],whales:[]};
 if(!process.env.DATABASE_URL)return empty;
 const sql=neon(process.env.DATABASE_URL);
 const exists=await sql`SELECT to_regclass('public.backpack_assets') AS registry`;
 if(!exists[0]?.registry)return {...empty,status:'schema_pending'};
 const latest=await sql`SELECT max(date)::text AS date FROM backpack_asset_daily_snapshots`;
 const day=latest[0]?.date??null;
 const [assets,history,ecosystem,bp,quality,quotes,holders,dex,runs,usage,whales]=await Promise.all([
  sql`SELECT a.*, s.*, a.id, a.solana_mint, s.date::text AS date FROM backpack_assets a
       LEFT JOIN backpack_asset_daily_snapshots s ON s.asset_id=a.id AND s.date=${day}::date
       WHERE a.active AND (${assetId??null}::bigint IS NULL OR a.id=${assetId??null}) ORDER BY a.asset_type='bp',a.token_symbol`,
  sql`SELECT *,date::text AS date FROM backpack_asset_daily_snapshots s
       WHERE (${assetId??null}::bigint IS NULL OR asset_id=${assetId??null}) ORDER BY s.date DESC LIMIT 20000`,
  sql`SELECT *,date::text AS date FROM backpack_ecosystem_daily_snapshots e ORDER BY e.date DESC LIMIT 5000`,
  sql`SELECT b.*, s.* ,s.date::text AS date FROM backpack_bp_daily_snapshots b
       JOIN backpack_asset_daily_snapshots s USING(asset_id,date) ORDER BY s.date DESC LIMIT 5000`,
  sql`SELECT * FROM backpack_data_quality_events WHERE date=${day}::date
       AND (${assetId??null}::bigint IS NULL OR asset_id=${assetId??null}) ORDER BY id DESC LIMIT 5000`,
  sql`SELECT * FROM backpack_asset_liquidity_daily_snapshots WHERE date=${day}::date
       AND (${assetId??null}::bigint IS NULL OR asset_id=${assetId??null}) ORDER BY notional_usd,direction`,
  assetId?sql`SELECT h.* FROM backpack_asset_holder_daily_snapshots h WHERE h.date=${day}::date AND h.asset_id=${assetId}
       ORDER BY h.balance_tokens DESC LIMIT 100`:Promise.resolve([]),
  sql`SELECT * FROM backpack_asset_dex_daily_snapshots WHERE date=${day}::date
       AND (${assetId??null}::bigint IS NULL OR asset_id=${assetId??null})`,
  sql`SELECT * FROM backpack_ingestion_runs ORDER BY started_at DESC LIMIT 10`,
  sql`SELECT u.* FROM backpack_provider_usage u JOIN backpack_ingestion_runs r USING(run_id) ORDER BY r.started_at DESC LIMIT 30`,
  sql`SELECT *,date::text AS date FROM backpack_bp_whale_daily_snapshots w
       WHERE (${assetId??null}::bigint IS NULL OR asset_id=${assetId??null}) ORDER BY w.date DESC,threshold_usd LIMIT 15000`,
 ]);
 return {status:day?'ready':'awaiting_capture',asOf:day,assets:assets as Row[],history:(history as Row[]).reverse(),
  ecosystem:(ecosystem as Row[]).reverse(),bp:(bp as Row[]).reverse(),quality:quality as Quality[],quotes:quotes as Row[],holders:holders as Row[],dex:dex as Row[],runs:runs as Row[],usage:usage as Row[],whales:(whales as Row[]).reverse()};
}

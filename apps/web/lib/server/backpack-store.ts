import {neon} from '@neondatabase/serverless';
import type {MonitorData, Quality, Row} from '@/lib/backpack';
export async function readBackpack(assetId?:number):Promise<MonitorData>{
 const empty:MonitorData={status:'not_configured',asOf:null,assets:[],history:[],ecosystem:[],bp:[],quality:[],quotes:[],holders:[],dex:[],runs:[],usage:[],whales:[],registryCandidates:[],externalObservations:[],economyEvents:[]};
 if(!process.env.DATABASE_URL)return empty;
 const sql=neon(process.env.DATABASE_URL);
 const exists=await sql`SELECT to_regclass('public.backpack_assets') AS registry, to_regclass('public.backpack_environment_daily') AS environment, to_regclass('public.tokenized_security_daily_snapshots') AS competitors, to_regclass('public.backpack_growth_daily') AS growth, to_regclass('public.backpack_adoption_daily') AS adoption, to_regclass('public.backpack_adoption_assessments') AS adoption_growth, to_regclass('public.backpack_registry_candidates_daily') AS registry_candidates, to_regclass('public.backpack_external_observations') AS external_observations, to_regclass('public.backpack_economy_events') AS economy_events`;
 if(!exists[0]?.registry)return {...empty,status:'schema_pending'};
 const latest=await sql`SELECT max(date)::text AS date FROM backpack_asset_daily_snapshots`;
 const day=latest[0]?.date??null;
 const [assets,history,ecosystem,bp,quality,quotes,holders,dex,runs,usage,whales,analytics,environment,competitors,growth,adoption,adoptionGrowth,registryCandidates,externalObservations,economyEvents]=await Promise.all([
  sql`SELECT a.*, s.*, a.id, a.solana_mint, s.date::text AS date FROM backpack_assets a
       LEFT JOIN backpack_asset_daily_snapshots s ON s.asset_id=a.id AND s.date=${day}::date
       WHERE a.active AND (${assetId??null}::bigint IS NULL OR a.id=${assetId??null}) ORDER BY a.asset_type='bp',a.token_symbol`,
  assetId?sql`SELECT *,date::text AS date FROM backpack_asset_daily_snapshots s
       WHERE (${assetId??null}::bigint IS NULL OR asset_id=${assetId??null}) ORDER BY s.date DESC LIMIT 20000`:Promise.resolve([]),
  sql`SELECT *,date::text AS date FROM backpack_ecosystem_daily_snapshots e ORDER BY e.date DESC LIMIT 5000`,
  sql`SELECT b.*,s.token_supply,s.holders_over_100,s.onchain_price,s.top_20_holder_pct,s.economic_top_20_holder_pct,s.date::text AS date FROM backpack_bp_daily_snapshots b
       JOIN backpack_asset_daily_snapshots s USING(asset_id,date) ORDER BY s.date DESC LIMIT 5000`,
  sql`SELECT * FROM backpack_data_quality_events WHERE date=${day}::date
       AND (${assetId??null}::bigint IS NULL OR asset_id=${assetId??null}) ORDER BY id DESC LIMIT 5000`,
  sql`SELECT * FROM backpack_asset_liquidity_daily_snapshots WHERE date=${day}::date
       AND (${assetId??null}::bigint IS NULL OR asset_id=${assetId??null}) ORDER BY notional_usd,direction`,
  assetId?sql`SELECT h.*,h.last_seen_at::text AS date FROM backpack_current_holders h WHERE h.last_seen_at=${day}::date AND h.asset_id=${assetId}
       ORDER BY h.balance_tokens DESC LIMIT 100`:Promise.resolve([]),
  sql`SELECT * FROM backpack_asset_dex_daily_snapshots WHERE date=${day}::date
       AND (${assetId??null}::bigint IS NULL OR asset_id=${assetId??null})`,
  sql`SELECT * FROM backpack_ingestion_runs ORDER BY started_at DESC LIMIT 10`,
  sql`SELECT u.* FROM backpack_provider_usage u JOIN backpack_ingestion_runs r USING(run_id) ORDER BY r.started_at DESC LIMIT 30`,
  sql`SELECT *,date::text AS date FROM backpack_bp_whale_daily_snapshots w
       WHERE w.date=${day}::date AND (${assetId??null}::bigint IS NULL OR asset_id=${assetId??null}) ORDER BY w.date DESC,threshold_usd`,
  sql`SELECT * FROM backpack_analytical_daily_metrics WHERE date=${day}::date`,
  exists[0]?.environment?sql`SELECT * FROM backpack_environment_daily WHERE date=${day}::date ORDER BY period_days`:Promise.resolve([]),
  exists[0]?.competitors?sql`SELECT i.id,i.name,count(DISTINCT a.id)::int AS assets,max(s.date)::text AS latest_capture
      FROM tokenized_security_issuers i LEFT JOIN tokenized_security_assets a ON a.issuer_id=i.id AND a.active
      LEFT JOIN tokenized_security_daily_snapshots s ON s.asset_id=a.id
      GROUP BY i.id,i.name ORDER BY i.name`:Promise.resolve([]),
  exists[0]?.growth?sql`SELECT *,date::text AS date FROM backpack_growth_daily WHERE date=${day}::date ORDER BY period_days`:Promise.resolve([]),
  exists[0]?.adoption?sql`SELECT data FROM backpack_adoption_daily ORDER BY date DESC LIMIT 5000`:Promise.resolve([]),
  exists[0]?.adoption_growth?sql`SELECT data FROM backpack_adoption_assessments WHERE date=${day}::date ORDER BY period_days`:Promise.resolve([]),
  exists[0]?.registry_candidates?sql`SELECT *,date::text AS date FROM backpack_registry_candidates_daily
      WHERE date=(SELECT max(date) FROM backpack_registry_candidates_daily) ORDER BY match_status,token_symbol`:Promise.resolve([]),
  exists[0]?.external_observations?sql`SELECT *,published_at::text AS published_at FROM backpack_external_observations
      WHERE review_status='approved' ORDER BY published_at DESC LIMIT 100`:Promise.resolve([]),
  exists[0]?.economy_events?sql`SELECT *,event_at::text AS event_at,campaign_end_at::text AS campaign_end_at
      FROM backpack_economy_events ORDER BY event_at DESC LIMIT 100`:Promise.resolve([]),
 ]);
 return {status:day?'ready':'awaiting_capture',asOf:day,assets:assets as Row[],history:(history as Row[]).reverse(),
  ecosystem:(ecosystem as Row[]).reverse(),bp:(bp as Row[]).reverse(),quality:quality as Quality[],quotes:quotes as Row[],holders:holders as Row[],dex:dex as Row[],runs:runs as Row[],usage:usage as Row[],whales:(whales as Row[]).reverse(),analytics:analytics as Row[],environment:environment as Row[],competitors:competitors as Row[],growth:growth as Row[],adoption:adoption.map(r=>r.data as Row).reverse(),adoptionGrowth:adoptionGrowth.map(r=>r.data as Row),registryCandidates:registryCandidates as Row[],externalObservations:externalObservations as Row[],economyEvents:economyEvents as Row[]};
}

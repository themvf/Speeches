import {neon} from '@neondatabase/serverless';

/** Freezes one existing current-cohort version as the original cohort, in a single statement. Mirrors
 *  backpack/cohorts.py::approve_original; tests/test_bp_intel_integration.py executes this exact SQL.
 *  Returns no row when an original already exists or the version is not a current version. */
export async function approveOriginalCohort(version:number,notes:string){
 const sql=neon(process.env.DATABASE_URL!),actor='authenticated_admin';
 return sql`WITH original AS (
   INSERT INTO bp_cohorts(kind,effective_at,source_date,source_run_id,source_slot_start,source_slot_end,derived_from_version,
    exclusion_fingerprint,excluded_count,eligible_count,target_size,exit_rank,exit_runs,entrant_cap,size,entered,left_count,rank_changed,
    queued,entrant_cap_bound,bootstrap,status,methodology,approved_at,approved_by,approval_notes)
   SELECT 'original',effective_at,source_date,source_run_id,source_slot_start,source_slot_end,version_id,exclusion_fingerprint,
    excluded_count,eligible_count,target_size,exit_rank,exit_runs,entrant_cap,size,size,0,0,0,false,true,status,
    'Original cohort: frozen copy of an approved current version; members stay tracked after selling BP. '||methodology,
    now(),${actor},${notes} FROM bp_cohorts
   WHERE version_id=${version} AND kind='current' AND NOT EXISTS(SELECT 1 FROM bp_cohorts WHERE kind='original')
   RETURNING version_id,derived_from_version),
  members AS (INSERT INTO bp_cohort_members(version_id,wallet_address,member,rank,raw_balance,previous_rank,event,below_exit_runs)
   SELECT o.version_id,m.wallet_address,true,m.rank,m.raw_balance,NULL,'bootstrap',0 FROM original o
   JOIN bp_cohort_members m ON m.version_id=o.derived_from_version AND m.member RETURNING wallet_address),
  tracked AS (UPDATE bp_tracked_wallets t SET active=true,updated_at=now()
   WHERE t.wallet_address IN (SELECT wallet_address FROM members) AND NOT t.active RETURNING t.wallet_address)
  SELECT version_id,(SELECT count(*) FROM members)::int AS members FROM original`;
}

export async function readCohortAdmin(){
 const sql=neon(process.env.DATABASE_URL!);
 const schema=await sql`SELECT to_regclass('public.bp_cohorts') AS cohorts`;
 if(!schema[0]?.cohorts)return {status:'schema_pending',versions:[]};
 const versions=await sql`SELECT version_id,kind,source_date::text AS source_date,effective_at,size,entered,left_count,queued,
   entrant_cap_bound,bootstrap,status,exclusion_fingerprint,approved_at,approved_by,approval_notes,derived_from_version
   FROM bp_cohorts ORDER BY kind='original' DESC,source_date DESC LIMIT 15`;
 return {status:'ready',versions};
}

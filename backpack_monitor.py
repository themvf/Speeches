"""Explicit production operations. Failures never print provider/connection secrets."""
import argparse
import json
import os
from backpack.collector import run, setup
from backpack.readiness import preflight, audit
from backpack.registry import seed_starter


def main():
    parser=argparse.ArgumentParser()
    for action in ('migrate','execute','preflight','audit','seed-starter','maintenance','cost-report','research'):
        parser.add_argument('--'+action,action='store_true')
    parser.add_argument('--import-billing',metavar='CSV')
    # BP holder intelligence (docs/bp-holder-intelligence-spec.md).
    for action in ('bp-cohort','bp-intel','bp-feasibility'):
        parser.add_argument('--'+action,action='store_true')
    parser.add_argument('--bp-steps',default='portfolio,history,alerts,reconcile,maintenance',
        help='Comma-separated --bp-intel steps')
    parser.add_argument('--bp-approve-original',metavar='VERSION',type=int)
    parser.add_argument('--bp-notes',default='')
    args=parser.parse_args()
    if not any(v for k,v in vars(args).items() if k not in ('bp_steps','bp_notes')):
        print('Use --migrate, --preflight, --execute or --audit. Live RPC cannot backfill history.')
        return 0
    if not os.environ.get('DATABASE_URL'):
        print(json.dumps({'status':'Unavailable','check':'database','detail':'DATABASE_URL is not configured'}))
        return 1
    conn=None
    try:
        import psycopg2
        conn=psycopg2.connect(os.environ['DATABASE_URL'],connect_timeout=15)
        if args.migrate:
            setup(conn)
            print(json.dumps({'schema':'initialized'}))
        failed=False
        if args.seed_starter:
            result=seed_starter(conn)
            print(json.dumps(result,default=str))
            failed=result['failed']>0
        if args.preflight:
            result=preflight(conn)
            print(json.dumps(result,default=str))
            failed=failed or not result['ready_for_capture']
        if args.execute:
            result=run(conn)
            print(json.dumps(result,default=str))
            failed=failed or result['status'] in ('failed','partial')
        if args.research:
            from backpack.research import capture_research
            from datetime import datetime, timezone
            capture_research(conn,datetime.now(timezone.utc).date(),os.environ)
            print(json.dumps({'stored_research':'processed'}))
        if args.maintenance:
            from backpack.storage import maintain
            from datetime import datetime, timezone
            maintain(conn, datetime.now(timezone.utc).date(), os.environ)
            print(json.dumps({'maintenance':'completed'}))
        if args.import_billing:
            from backpack.cost_review import import_billing
            print(json.dumps(import_billing(conn,args.import_billing)))
        if args.cost_report:
            from backpack.storage import cost_report
            print(json.dumps(cost_report(conn),default=str))
        if args.audit:print(json.dumps(audit(conn),default=str))
        if args.bp_cohort:
            from backpack.cohorts import refresh_cohort
            print(json.dumps({'bp_cohort':refresh_cohort(conn,os.environ)},default=str))
        if args.bp_approve_original:
            from backpack.cohorts import approve_original
            print(json.dumps({'bp_original_cohort':approve_original(conn,args.bp_approve_original,'cli_operator',args.bp_notes)}))
        if args.bp_intel:
            from backpack.intel import run as run_intel
            result=run_intel(conn,env=os.environ,steps=tuple(s.strip() for s in args.bp_steps.split(',') if s.strip()))
            print(json.dumps({'bp_intel':result},default=str))
            failed=failed or result['status'] in ('failed','schema_pending')
        if args.bp_feasibility:
            from backpack.feasibility import run as feasibility
            report=feasibility(conn,os.environ)
            print(json.dumps({'bp_feasibility':report},default=str))
            failed=failed or report.get('status')!='completed'
        return int(failed)
    except Exception as error:
        print(json.dumps({'status':'Unavailable','detail':'Operation failed: '+type(error).__name__}))
        return 1
    finally:
        if conn is not None:conn.close()


if __name__=='__main__':raise SystemExit(main())

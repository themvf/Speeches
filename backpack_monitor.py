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
    args=parser.parse_args()
    if not any(vars(args).values()):
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
            failed=failed or not result['ready_for_security_capture']
        if args.execute:
            result=run(conn)
            print(json.dumps(result,default=str))
            failed=failed or result['status'] in ('failed','partial')
        if args.maintenance:
            from backpack.storage import maintain
            from datetime import datetime, timezone
            maintain(conn, datetime.now(timezone.utc).date(), os.environ)
            print(json.dumps({'maintenance':'completed'}))
        if args.research:
            from backpack.research import capture_research
            from datetime import datetime, timezone
            capture_research(conn,datetime.now(timezone.utc).date(),os.environ)
            print(json.dumps({'stored_research':'processed'}))
        if args.import_billing:
            from backpack.cost_review import import_billing
            print(json.dumps(import_billing(conn,args.import_billing)))
        if args.cost_report:
            from backpack.storage import cost_report
            print(json.dumps(cost_report(conn),default=str))
        if args.audit:print(json.dumps(audit(conn),default=str))
        return int(failed)
    except Exception as error:
        print(json.dumps({'status':'Unavailable','detail':'Operation failed: '+type(error).__name__}))
        return 1
    finally:
        if conn is not None:conn.close()


if __name__=='__main__':raise SystemExit(main())

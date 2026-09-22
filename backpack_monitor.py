"""Backpack collector. Explicit --migrate and --execute; no actions on import."""
import argparse
import json
import os
from backpack.collector import run, setup

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--migrate',action='store_true')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.migrate and not args.execute:
        print('Use --migrate to initialize schema; --execute to capture today. Live RPC is never historical backfill.')
    else:
        import psycopg2
        conn=psycopg2.connect(os.environ['DATABASE_URL'])
        try:
            if args.migrate: setup(conn)
            if args.execute:
                result=run(conn)
                print(json.dumps(result))
                if result['status'] in ('failed','partial'): raise SystemExit(1)
        finally: conn.close()

"""ZCAT September 14 catch-up, using the existing live pilot search allowance."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from crypto_social_pilot import initialize, build_windows, collect, PILOT

START=datetime(2026,9,14,tzinfo=timezone.utc)
END=START+timedelta(days=1)

def cutoff(now):
    return min(END,now.astimezone(timezone.utc).replace(microsecond=0))

def setup(conn,end):
    if not START<end<=END:raise ValueError('No September 14 interval available')
    initialize(conn,end)
    with conn,conn.cursor() as cur:
        cur.execute('SELECT id FROM crypto_social_pilot WHERE id=%s FOR UPDATE',(PILOT,))
        # Append only beyond previously captured cutoff; preserve saved cursors.
        cur.execute("SELECT max(end_at) FROM crypto_social_windows WHERE coin='ZCAT' AND start_at>=%s AND end_at<=%s AND query<>'timeline text match'",(START,END))
        start=cur.fetchone()[0] or START
        for row in build_windows(start,end):
            if row[0]=='ZCAT':
                cur.execute('INSERT INTO crypto_social_windows(coin,start_at,end_at,query) VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING',row)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    end=cutoff(datetime.now(timezone.utc))
    plan={'coin':'ZCAT','start':str(START),'end_exclusive':str(end),'day_complete':end==END,
          'max_requests':40,'max_reservation':12000,'budget':'existing live pilot; existing 16,800 search sublimit'}
    if not args.execute:
        print(json.dumps({'mode':'plan_only','paid_calls':0,**plan}));return
    import psycopg2
    conn=psycopg2.connect(os.environ['DATABASE_URL'],connect_timeout=15)
    try:
        setup(conn,end)
        calls=collect(conn,os.environ['TWITTERAPI_IO_API_KEY'],40,max_pages=8,scope=(START,end,'ZCAT'))
        with conn,conn.cursor() as cur:
            cur.execute('SELECT reserved_credits,credit_limit FROM crypto_social_pilot WHERE id=%s',(PILOT,))
            spent,limit=cur.fetchone()
            cur.execute("SELECT count(*) FILTER(WHERE pages>0),count(*),count(*) FILTER(WHERE status='search_exhausted') FROM crypto_social_windows WHERE coin='ZCAT' AND start_at>=%s AND end_at<=%s",(START,end))
            searched,windows,exhausted=cur.fetchone()
        print(json.dumps({**plan,'requests_saved':calls,'live_reserved_credits':spent,'live_credit_limit':limit,
                          'searched_windows':searched,'windows':windows,'exhausted_windows':exhausted}))
    finally:conn.close()

if __name__=='__main__':main()

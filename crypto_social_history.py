"""ZCAT historical investigation: July 25–September 13, 2026 (UTC).

A separate, non-resetting 75,000-credit allowance. The live tracking pilot keeps
its original 50,000-credit limit. Default execution is a no-network plan.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from crypto_social_pilot import PILOT, COINS, PAGE_RESERVE, ENDPOINT, initialize, save_page

START=datetime(2026,7,25,tzinfo=timezone.utc)
END=datetime(2026,9,14,tzinfo=timezone.utc)
HISTORY='zcat-july-2026'
LIMIT=75000
SCHEMA='''
CREATE TABLE IF NOT EXISTS crypto_social_history_campaign (
 id text PRIMARY KEY CHECK(id='zcat-july-2026'), start_at timestamptz NOT NULL,
 end_at timestamptz NOT NULL, credit_limit integer NOT NULL DEFAULT 75000 CHECK(credit_limit=75000),
 reserved_credits integer NOT NULL DEFAULT 0 CHECK(reserved_credits BETWEEN 0 AND 75000),
 created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS crypto_social_history_windows (
 campaign_id text NOT NULL REFERENCES crypto_social_history_campaign(id),
 window_id bigint NOT NULL REFERENCES crypto_social_windows(id),
 PRIMARY KEY(campaign_id,window_id)
);
'''

def setup(conn):
    initialize(conn,END)
    with conn,conn.cursor() as cur:
        cur.execute(SCHEMA)
        cur.execute('INSERT INTO crypto_social_history_campaign(id,start_at,end_at) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING',(HISTORY,START,END))
        current=START
        while current<END:
            until=min(current+timedelta(hours=6),END)
            query=f'{COINS["ZCAT"][1]} since_time:{int(current.timestamp())} until_time:{int(until.timestamp())}'
            cur.execute('''INSERT INTO crypto_social_windows(coin,start_at,end_at,query)
                VALUES ('ZCAT',%s,%s,%s) ON CONFLICT DO NOTHING''',(current,until,query))
            cur.execute("SELECT id FROM crypto_social_windows WHERE coin='ZCAT' AND start_at=%s AND end_at=%s AND query=%s",(current,until,query))
            row=cur.fetchone()
            if not row: raise ValueError('Existing window has a different query; review before reusing coverage')
            cur.execute('INSERT INTO crypto_social_history_windows VALUES (%s,%s) ON CONFLICT DO NOTHING',(HISTORY,row[0]))
            current=until

def reserve(conn):
    with conn,conn.cursor() as cur:
        # Same lock order as the other collectors; unknown outcomes block all calls.
        cur.execute('SELECT id FROM crypto_social_pilot WHERE id=%s FOR UPDATE',(PILOT,))
        cur.execute("SELECT count(*) FROM crypto_social_requests WHERE status IN ('reserved','uncertain')")
        if cur.fetchone()[0]: raise RuntimeError('Outstanding or uncertain request; review ledger')
        cur.execute('SELECT reserved_credits,credit_limit FROM crypto_social_history_campaign WHERE id=%s FOR UPDATE',(HISTORY,))
        spent,limit=cur.fetchone()
        if spent+PAGE_RESERVE>min(limit,LIMIT):return None
        # Scan every period once, then deepen the earliest unfinished periods first.
        cur.execute('''SELECT w.id,w.start_at,w.end_at,w.query,w.cursor FROM crypto_social_windows w
            JOIN crypto_social_history_windows h ON h.window_id=w.id
            WHERE h.campaign_id=%s AND w.status IN ('pending','partial')
            ORDER BY (w.pages=0) DESC,w.start_at,w.id LIMIT 1 FOR UPDATE OF w''',(HISTORY,))
        window=cur.fetchone()
        if not window:return None
        cur.execute('UPDATE crypto_social_history_campaign SET reserved_credits=reserved_credits+%s WHERE id=%s',(PAGE_RESERVE,HISTORY))
        cur.execute('''INSERT INTO crypto_social_requests(window_id,endpoint,parameters,reserved_credits)
            VALUES (%s,'historical_search',%s,%s) RETURNING id''',
            (window[0],json.dumps({'allocation':'history','campaign':HISTORY}),PAGE_RESERVE))
        return cur.fetchone()[0],window

def collect_history(conn,key,max_requests=40,fetch=None):
    import requests
    calls=0
    while calls<max_requests:
        item=reserve(conn)
        if item is None:break
        rid,window=item
        try:
            response=(fetch or requests.get)(ENDPOINT,params={'query':window[3],'queryType':'Latest','cursor':window[4]},
                headers={'X-API-Key':key},timeout=30,allow_redirects=False)
            if response.status_code!=200:raise ValueError(f'HTTP {response.status_code}')
            data=response.json()
            # Catch repeated pages even when the provider changes the cursor.
            ids=[str(t.get('id','')) for t in data.get('tweets',[])] if isinstance(data,dict) else []
            if ids:
                with conn,conn.cursor() as cur:
                    cur.execute('SELECT count(*) FROM crypto_social_matches WHERE window_id=%s AND post_id=ANY(%s)',(window[0],ids))
                    if cur.fetchone()[0]==len(ids):raise ValueError('Repeated historical page; stop to avoid paying for a loop')
            save_page(conn,rid,window,data)
        except Exception as exc:
            with conn,conn.cursor() as cur:
                cur.execute("UPDATE crypto_social_requests SET status='uncertain',error=%s WHERE id=%s",(type(exc).__name__,rid))
            raise RuntimeError(f'History request {rid} stopped; reservation retained. Review provider and ledger.') from None
        calls+=1
    return calls

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--max-requests',type=int,default=40)
    args=parser.parse_args()
    if not 1<=args.max_requests<=250:parser.error('Request bound must be 1–250')
    if not args.execute:
        print(json.dumps({'mode':'plan_only','coin':'ZCAT','start':str(START),'end_exclusive':str(END),
            'history_credit_ceiling':LIMIT,'first_pass_before_reuse':int((END-START).days)*4*PAGE_RESERVE,
            'max_run_reservation':args.max_requests*PAGE_RESERVE,'paid_calls':0},indent=2));return
    import psycopg2
    conn=psycopg2.connect(os.environ['DATABASE_URL'],connect_timeout=15)
    try:
        setup(conn)
        calls=collect_history(conn,os.environ['TWITTERAPI_IO_API_KEY'],args.max_requests)
        with conn,conn.cursor() as cur:
            cur.execute('SELECT reserved_credits FROM crypto_social_history_campaign WHERE id=%s',(HISTORY,))
            spent=cur.fetchone()[0]
        print(json.dumps({'requests_saved':calls,'history_reserved_credits':spent,'history_ceiling':LIMIT}))
    finally:conn.close()

if __name__=='__main__':main()

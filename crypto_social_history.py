"""ZCAT historical investigation: July 25–September 13, 2026 (UTC).

A separate, non-resetting 150,000-credit allowance. The live tracking pilot keeps
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
PONS_ADDRESS='0x39dbed3a2bd333467115de45665cc57f813c4571'
PONS_QUERY=f'("$PONS" OR "{PONS_ADDRESS}" OR (PONS Robinhood) OR from:ponsdotfamily)'

def settings(coin):
    if coin=='ZCAT':return HISTORY,START,END,COINS['ZCAT'][1]
    if coin=='PONS':return 'pons-july-2026',datetime(2026,7,1,tzinfo=timezone.utc),datetime(2026,9,15,tzinfo=timezone.utc),PONS_QUERY
    raise ValueError('Unsupported historical coin')

LIMIT=150000
BATCH_REQUESTS=80
MAX_WINDOW_PAGES=8
SCHEMA='''
CREATE TABLE IF NOT EXISTS crypto_social_history_campaign (
 id text PRIMARY KEY CHECK(id IN ('zcat-july-2026','pons-july-2026')), start_at timestamptz NOT NULL,
 end_at timestamptz NOT NULL, credit_limit integer NOT NULL DEFAULT 150000 CHECK(credit_limit=150000),
 reserved_credits integer NOT NULL DEFAULT 0 CHECK(reserved_credits BETWEEN 0 AND 150000),
 created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS crypto_social_history_windows (
 campaign_id text NOT NULL REFERENCES crypto_social_history_campaign(id),
 window_id bigint NOT NULL REFERENCES crypto_social_windows(id),
 PRIMARY KEY(campaign_id,window_id)
);
CREATE TABLE IF NOT EXISTS crypto_social_history_batches (
 id bigserial PRIMARY KEY, campaign_id text NOT NULL REFERENCES crypto_social_history_campaign(id),
 started_at timestamptz NOT NULL DEFAULT now(), completed_at timestamptz,
 focus_start timestamptz, focus_end timestamptz, focus_reason text NOT NULL,
 market_source text, market_fetch_id bigint, max_requests integer NOT NULL,
 status text NOT NULL DEFAULT 'running', requests_saved integer NOT NULL DEFAULT 0
);
'''

def setup(conn,coin='ZCAT',now=None):
    campaign,start,end,query_base=settings(coin)
    initialize(conn,end)
    with conn,conn.cursor() as cur:
        cur.execute(SCHEMA)
        cur.execute("ALTER TABLE crypto_social_history_campaign DROP CONSTRAINT IF EXISTS crypto_social_history_campaign_id_check; ALTER TABLE crypto_social_history_campaign ADD CONSTRAINT crypto_social_history_campaign_id_check CHECK(id IN ('zcat-july-2026','pons-july-2026'))")
        cur.execute('INSERT INTO crypto_social_coins VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING',
                    ('PONS','Pons',PONS_QUERY,PONS_ADDRESS,'Robinhood Chain; exact contract matched to explorer and pool relationships'))
        # Upgrade the existing allowance transactionally; never reset reservations.
        cur.execute('''ALTER TABLE crypto_social_history_campaign
            DROP CONSTRAINT IF EXISTS crypto_social_history_campaign_credit_limit_check,
            DROP CONSTRAINT IF EXISTS crypto_social_history_campaign_reserved_credits_check;
            ALTER TABLE crypto_social_history_campaign ALTER COLUMN credit_limit SET DEFAULT 150000;
            UPDATE crypto_social_history_campaign SET credit_limit=150000;
            ALTER TABLE crypto_social_history_campaign
            ADD CONSTRAINT crypto_social_history_campaign_credit_limit_check CHECK(credit_limit=150000),
            ADD CONSTRAINT crypto_social_history_campaign_reserved_credits_check
                CHECK(reserved_credits BETWEEN 0 AND credit_limit);''')
        cur.execute('INSERT INTO crypto_social_history_campaign(id,start_at,end_at) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING',(campaign,start,end))
        current=start
        now=now or datetime.now(timezone.utc)
        closed=now.replace(hour=(now.hour//6)*6,minute=0,second=0,microsecond=0)
        available=min(end,closed)
        while current<available:
            until=min(current+timedelta(hours=6),available)
            query=f'{query_base} since_time:{int(current.timestamp())} until_time:{int(until.timestamp())}'
            cur.execute('''INSERT INTO crypto_social_windows(coin,start_at,end_at,query)
                VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING''',(coin,current,until,query))
            cur.execute("SELECT id FROM crypto_social_windows WHERE coin=%s AND start_at=%s AND end_at=%s AND query=%s",(coin,current,until,query))
            row=cur.fetchone()
            if not row: raise ValueError('Existing window has a different query; review before reusing coverage')
            cur.execute('INSERT INTO crypto_social_history_windows VALUES (%s,%s) ON CONFLICT DO NOTHING',(campaign,row[0]))
            current=until

def choose_focus(conn,coin='ZCAT'):
    """Earliest >=50% complete, consecutive daily gain; timing heuristic, not causation."""
    campaign,start,end,_=settings(coin)
    with conn,conn.cursor() as cur:
        cur.execute("SELECT to_regclass('crypto_market_latest')")
        if not cur.fetchone()[0]:return None
        cur.execute("""SELECT m.day,m.close,m.source_id,m.fetch_id FROM crypto_market_latest m
            JOIN crypto_market_sources s ON s.id=m.source_id WHERE s.coin=%s AND s.is_default
            AND m.complete AND m.day>=%s AND m.day<%s ORDER BY m.day""",(coin,start.date(),end.date()))
        rows=cur.fetchall()
    for previous,current in zip(rows,rows[1:]):
        if current[0]-previous[0]==timedelta(days=1) and previous[1]>0 and current[1]/previous[1]>=1.5:
            anchor=datetime.combine(current[0],datetime.min.time(),tzinfo=timezone.utc)
            return {'start':max(start,anchor-timedelta(days=5)),'end':min(end,anchor+timedelta(days=3)),
                'reason':f'First saved complete daily gain >=50% on {current[0]}; five preceding days and two following days. Not a causal claim.',
                'source':current[2],'fetch_id':current[3]}
    return None


def start_batch(conn,focus,max_requests,coin='ZCAT'):
    campaign,_,_,_=settings(coin)
    with conn,conn.cursor() as cur:
        cur.execute("""INSERT INTO crypto_social_history_batches(campaign_id,focus_start,focus_end,focus_reason,
            market_source,market_fetch_id,max_requests) VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
            (campaign,focus['start'] if focus else None,focus['end'] if focus else None,
             focus['reason'] if focus else 'No qualifying saved market move; continue chronological gap filling.',
             focus['source'] if focus else None,focus['fetch_id'] if focus else None,max_requests))
        return cur.fetchone()[0]


def reserve(conn,focus=None,baseline=False,batch_id=None,coin='ZCAT'):
    campaign,start,end,_=settings(coin)
    with conn,conn.cursor() as cur:
        # Same lock order as the other collectors; unknown outcomes block all calls.
        cur.execute('SELECT id FROM crypto_social_pilot WHERE id=%s FOR UPDATE',(PILOT,))
        cur.execute("SELECT count(*) FROM crypto_social_requests WHERE status IN ('reserved','uncertain')")
        if cur.fetchone()[0]: raise RuntimeError('Outstanding or uncertain request; review ledger')
        cur.execute('SELECT reserved_credits,credit_limit FROM crypto_social_history_campaign WHERE id=%s FOR UPDATE',(campaign,))
        spent,limit=cur.fetchone()
        if spent+PAGE_RESERVE>min(limit,LIMIT):return None
        # Cover unsearched windows first, prioritizing the buildup; then deepen in rounds.
        cur.execute('''SELECT w.id,w.start_at,w.end_at,w.query,w.cursor FROM crypto_social_windows w
            JOIN crypto_social_history_windows h ON h.window_id=w.id
            WHERE h.campaign_id=%s AND w.status IN ('pending','partial') AND w.pages<%s
            ORDER BY w.pages,CASE WHEN %s AND w.start_at>=%s AND w.start_at<%s THEN 0 ELSE 1 END,
            w.start_at,w.id LIMIT 1 FOR UPDATE OF w''',
            (campaign,MAX_WINDOW_PAGES,bool(focus) and not baseline,focus['start'] if focus else start,focus['end'] if focus else end))
        window=cur.fetchone()
        if not window:return None
        cur.execute('UPDATE crypto_social_history_campaign SET reserved_credits=reserved_credits+%s WHERE id=%s',(PAGE_RESERVE,campaign))
        cur.execute('''INSERT INTO crypto_social_requests(window_id,endpoint,parameters,reserved_credits)
            VALUES (%s,'historical_search',%s,%s) RETURNING id''',
            (window[0],json.dumps({'allocation':'history','campaign':campaign,'batch_id':batch_id,'strategy':'targeted' if focus and not baseline and focus['start']<=window[1]<focus['end'] else 'baseline'}),PAGE_RESERVE))
        return cur.fetchone()[0],window

def collect_history(conn,key,max_requests=BATCH_REQUESTS,fetch=None,focus=None,batch_id=None,coin='ZCAT'):
    import requests
    calls=0
    while calls<max_requests:
        item=reserve(conn,focus,False,batch_id,coin)
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
        if batch_id is not None:
            with conn,conn.cursor() as cur:
                cur.execute('UPDATE crypto_social_history_batches SET requests_saved=%s WHERE id=%s',(calls,batch_id))
        print(json.dumps({'progress_requests_saved':calls,'max_requests':max_requests,'window_start':str(window[1])}),flush=True)
    return calls

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coin',choices=['ZCAT','PONS'],default='ZCAT')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--max-requests',type=int,default=BATCH_REQUESTS)
    args=parser.parse_args()
    campaign,start,end,_=settings(args.coin)
    if not 1<=args.max_requests<=BATCH_REQUESTS:parser.error('Request bound must be 1–80')
    if not args.execute:
        print(json.dumps({'mode':'plan_only','coin':args.coin,'start':str(start),'end_exclusive':str(end),
            'history_credit_ceiling':LIMIT,'first_pass_before_reuse':int((end-start).days)*4*PAGE_RESERVE,
            'max_run_reservation':args.max_requests*PAGE_RESERVE,'paid_calls':0},indent=2));return
    import psycopg2
    conn=psycopg2.connect(os.environ['DATABASE_URL'],connect_timeout=15)
    try:
        setup(conn,args.coin)
        focus=choose_focus(conn,args.coin)
        batch_id=start_batch(conn,focus,args.max_requests,args.coin)
        try:
            calls=collect_history(conn,os.environ['TWITTERAPI_IO_API_KEY'],args.max_requests,focus=focus,batch_id=batch_id,coin=args.coin)
        except Exception:
            with conn,conn.cursor() as cur:
                cur.execute("UPDATE crypto_social_history_batches SET status='stopped',completed_at=now(),requests_saved=(SELECT count(*) FROM crypto_social_requests WHERE status='saved' AND parameters->>'batch_id'=%s) WHERE id=%s",(str(batch_id),batch_id))
            raise
        with conn,conn.cursor() as cur:
            cur.execute("UPDATE crypto_social_history_batches SET status='completed',completed_at=now(),requests_saved=%s WHERE id=%s",(calls,batch_id))
        with conn,conn.cursor() as cur:
            cur.execute('SELECT reserved_credits FROM crypto_social_history_campaign WHERE id=%s',(campaign,))
            spent=cur.fetchone()[0]
        print(json.dumps({'requests_saved':calls,'history_reserved_credits':spent,'history_ceiling':LIMIT,'batch_id':batch_id,'focus':focus},default=str))
    finally:conn.close()

if __name__=='__main__':main()

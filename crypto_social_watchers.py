"""Two-hour searches for the ten reviewed watcher accounts; separate non-resetting budget."""
import argparse,json,os
from datetime import datetime,timedelta,timezone
from pathlib import Path
from crypto_social_pilot import PILOT,ENDPOINT,initialize,validate_page
from crypto_social_tracking import save_posts
ACCOUNTS=json.loads(Path(__file__).parent.joinpath('apps/web/lib/crypto-watcher-accounts.json').read_text())
CAMPAIGN='watchers-ten-v1'
LIMIT=150000
SCHEMA='''
CREATE TABLE IF NOT EXISTS crypto_watcher_campaign (
 id text PRIMARY KEY,started_at timestamptz NOT NULL,end_at timestamptz NOT NULL,
 credit_limit integer NOT NULL CHECK(credit_limit=150000),used_credits integer NOT NULL DEFAULT 0 CHECK(used_credits BETWEEN 0 AND 150000)
);
CREATE TABLE IF NOT EXISTS crypto_watcher_windows (
 id bigserial PRIMARY KEY,campaign_id text REFERENCES crypto_watcher_campaign(id),account_id text NOT NULL,
 start_at timestamptz NOT NULL,end_at timestamptz NOT NULL,query text NOT NULL,cursor text NOT NULL DEFAULT '',
 pages integer NOT NULL DEFAULT 0,status text NOT NULL DEFAULT 'pending',UNIQUE(campaign_id,account_id,end_at)
);
CREATE TABLE IF NOT EXISTS crypto_watcher_calls (
 request_id bigint PRIMARY KEY REFERENCES crypto_social_requests(id),window_id bigint REFERENCES crypto_watcher_windows(id),
 run_slot timestamptz NOT NULL
);
CREATE TABLE IF NOT EXISTS crypto_watcher_posts (
 post_id text REFERENCES crypto_social_posts(id),window_id bigint REFERENCES crypto_watcher_windows(id),PRIMARY KEY(post_id,window_id)
);
'''

def slot(now):return now.replace(hour=now.hour//2*2,minute=0,second=0,microsecond=0)

def setup(conn,now):
    initialize(conn,slot(now))
    with conn,conn.cursor() as cur:
        cur.execute(SCHEMA)
        cur.execute('INSERT INTO crypto_watcher_campaign(id,started_at,end_at,credit_limit) VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING',(CAMPAIGN,now,now+timedelta(days=30),LIMIT))
        cur.execute('SELECT end_at FROM crypto_watcher_campaign WHERE id=%s',(CAMPAIGN,))
        if now>=cur.fetchone()[0]:return False
        for a in ACCOUNTS:
            cur.execute('SELECT handle FROM crypto_social_accounts WHERE id=%s',(a['id'],))
            found=cur.fetchone();handle=found[0] if found else a['handle']
            if not handle.replace('_','').isalnum():raise ValueError('Invalid saved handle')
            cur.execute('SELECT max(end_at) FROM crypto_watcher_windows WHERE campaign_id=%s AND account_id=%s',(CAMPAIGN,a['id']))
            last=cur.fetchone()[0]
            end=last+timedelta(hours=2) if last else slot(now)
            while end<=slot(now):
                start=end-timedelta(hours=3 if last else 24)
                query=f'from:{handle} since_time:{int(start.timestamp())} until_time:{int(end.timestamp())}'
                cur.execute('INSERT INTO crypto_watcher_windows(campaign_id,account_id,start_at,end_at,query) VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING',(CAMPAIGN,a['id'],start,end,query))
                last=end;end+=timedelta(hours=2)
    return True


def reserve(conn,account,now):
    with conn,conn.cursor() as cur:
        cur.execute('SELECT id FROM crypto_social_pilot WHERE id=%s FOR UPDATE',(PILOT,))
        cur.execute("SELECT count(*) FROM crypto_social_requests WHERE status IN ('reserved','uncertain')")
        if cur.fetchone()[0]:raise RuntimeError('Outstanding or uncertain request; review ledger')
        cur.execute('SELECT used_credits,credit_limit,end_at FROM crypto_watcher_campaign WHERE id=%s FOR UPDATE',(CAMPAIGN,))
        used,limit,end=cur.fetchone()
        if now>=end or used+300>limit:return None
        cur.execute('''SELECT count(*) FROM crypto_watcher_calls c JOIN crypto_watcher_windows w ON w.id=c.window_id
            WHERE w.campaign_id=%s AND w.account_id=%s AND c.run_slot=%s''',(CAMPAIGN,account,slot(now)))
        if cur.fetchone()[0]>=2:return None
        cur.execute('''SELECT id,start_at,end_at,query,cursor FROM crypto_watcher_windows
            WHERE campaign_id=%s AND account_id=%s AND status IN ('pending','partial') AND end_at<=%s
            ORDER BY pages,end_at DESC,id LIMIT 1 FOR UPDATE''',(CAMPAIGN,account,slot(now)))
        window=cur.fetchone()
        if not window:return None
        cur.execute('UPDATE crypto_watcher_campaign SET used_credits=used_credits+300 WHERE id=%s',(CAMPAIGN,))
        cur.execute('''INSERT INTO crypto_social_requests(endpoint,parameters,reserved_credits)
            VALUES ('watcher_search',%s,300) RETURNING id''',(json.dumps({'campaign':CAMPAIGN,'account':account,'query':window[3],'cursor':window[4]}),))
        rid=cur.fetchone()[0]
        cur.execute('INSERT INTO crypto_watcher_calls VALUES (%s,%s,%s)',(rid,window[0],slot(now)))
        return rid,window


def collect_one(conn,key,account,now,fetch=None):
    import requests
    item=reserve(conn,account,now)
    if not item:return False
    rid,w=item
    try:
        response=(fetch or requests.get)(ENDPOINT,params={'query':w[3],'queryType':'Latest','cursor':w[4]},headers={'X-API-Key':key},timeout=30,allow_redirects=False)
        if response.status_code!=200:raise ValueError('Provider HTTP error')
        data=response.json()
        if isinstance(data,dict) and data.get('status') not in (None,'success'):raise ValueError('Provider error')
        posts,cursor,more=validate_page(data,w[1],w[2],w[4])
        if any(p['author_id']!=account for p in posts):raise ValueError('Unexpected author; check saved handle')
        with conn,conn.cursor() as cur:
            if posts:
                cur.execute('SELECT count(*) FROM crypto_watcher_posts WHERE window_id=%s AND post_id=ANY(%s)',(w[0],[p['id'] for p in posts]))
                if cur.fetchone()[0]==len(posts):raise ValueError('Repeated watcher page')
            returned,accepted,estimated=save_posts(cur,rid,data,w[1],w[2],account=account if posts else None)
            if estimated>300:raise ValueError('Charge exceeds reservation')
            for p in posts:cur.execute('INSERT INTO crypto_watcher_posts VALUES (%s,%s) ON CONFLICT DO NOTHING',(p['id'],w[0]))
            cur.execute('UPDATE crypto_watcher_windows SET pages=pages+1,cursor=%s,status=%s WHERE id=%s',(cursor,'partial' if more else 'search_exhausted',w[0]))
            cur.execute("UPDATE crypto_social_requests SET status='saved',estimated_credits=%s,returned_count=%s,accepted_count=%s WHERE id=%s",(estimated,returned,accepted,rid))
            cur.execute('UPDATE crypto_watcher_campaign SET used_credits=used_credits-%s WHERE id=%s',(300-estimated,CAMPAIGN))
        print(json.dumps({'account':account,'saved':accepted,'estimated_credits':estimated}),flush=True)
    except Exception:
        with conn,conn.cursor() as cur:cur.execute("UPDATE crypto_social_requests SET status='uncertain',error='watcher_search_failed' WHERE id=%s",(rid,))
        raise RuntimeError(f'Watcher request {rid} uncertain; reservation retained; inspect ledger before continuing.') from None
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');args=p.parse_args()
    if not args.execute:print(json.dumps({'accounts':ACCOUNTS,'every_hours':2,'ceiling':LIMIT,'days':30,'max_pages_per_account_per_run':2,'paid_calls':0}));return
    import psycopg2
    conn=psycopg2.connect(os.environ['DATABASE_URL'],connect_timeout=15)
    try:
        now=datetime.now(timezone.utc)
        if setup(conn,now):
            for a in ACCOUNTS:
                for _ in range(2):
                    if not collect_one(conn,os.environ['TWITTERAPI_IO_API_KEY'],a['id'],now):break
        with conn,conn.cursor() as cur:
            cur.execute('SELECT used_credits,credit_limit,end_at FROM crypto_watcher_campaign WHERE id=%s',(CAMPAIGN,));budget=cur.fetchone()
            cur.execute('''SELECT count(DISTINCT p.post_id),count(DISTINCT w.account_id) FILTER(WHERE p.post_id IS NOT NULL),count(DISTINCT w.id) FILTER(WHERE w.status!='search_exhausted')
                FROM crypto_watcher_windows w LEFT JOIN crypto_watcher_posts p ON p.window_id=w.id WHERE w.campaign_id=%s''',(CAMPAIGN,));counts=cur.fetchone()
        print(json.dumps({'campaign':CAMPAIGN,'used_credits':budget[0],'ceiling':budget[1],'ends':budget[2],'posts':counts[0],'accounts_with_posts':counts[1],'unfinished_windows':counts[2]},default=str),flush=True)
    finally:conn.close()

if __name__=='__main__':main()

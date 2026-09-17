"""Six-hour post collection and daily profile snapshots; independent bounded 30-day pilot."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from crypto_voice_research import snapshot,evaluate,initialize as initialize_research
from crypto_social_pilot import PILOT, COINS, initialize, save_page
from crypto_social_history import PONS_ADDRESS, DPONS_ADDRESS
from crypto_social_tracking import save_profiles,save_posts

STANDARD_ADDRESS='0x88ad8ddf1e3898412146a534538d418c6f8a9062'
TRACKED={**COINS,
 'PONS':('Pons',f'(\"{PONS_ADDRESS}\" OR (PONS Robinhood) OR from:ponsdotfamily)',PONS_ADDRESS,'Robinhood Chain'),
 'DPONS':('Diamond Pons',f'(DPONS OR \"DiamondPons\" OR \"Diamond Pons\" OR \"{DPONS_ADDRESS}\")',DPONS_ADDRESS,'Robinhood Chain'),
 'STANDARD':('The Standard Reserve',f'(\"The Standard Reserve\" OR from:standard_rsv OR to:standard_rsv OR \"{STANDARD_ADDRESS}\")',STANDARD_ADDRESS,'Robinhood Chain')}
CAMPAIGN='rolling-five-coins-v1'
COIN_LIMIT=30000
DAILY_LIMIT=5400
PAGES_PER_RUN=4
BASE='https://api.twitterapi.io'
SCHEMA='''
CREATE TABLE IF NOT EXISTS crypto_voice_focus (window_id bigint PRIMARY KEY,coin text NOT NULL);

CREATE TABLE IF NOT EXISTS crypto_rolling_campaign (
 id text PRIMARY KEY, started_at timestamptz NOT NULL, end_at timestamptz NOT NULL
);
CREATE TABLE IF NOT EXISTS crypto_rolling_coins (
 campaign_id text REFERENCES crypto_rolling_campaign(id),coin text REFERENCES crypto_social_coins(symbol),
 credit_limit integer NOT NULL CHECK(credit_limit=30000), used_credits integer NOT NULL DEFAULT 0 CHECK(used_credits BETWEEN 0 AND 30000),
 PRIMARY KEY(campaign_id,coin)
);
CREATE TABLE IF NOT EXISTS crypto_rolling_windows (
 campaign_id text REFERENCES crypto_rolling_campaign(id),window_id bigint REFERENCES crypto_social_windows(id),
 PRIMARY KEY(campaign_id,window_id)
);
CREATE TABLE IF NOT EXISTS crypto_rolling_calls (
 request_id bigint PRIMARY KEY REFERENCES crypto_social_requests(id),campaign_id text NOT NULL,
 coin text NOT NULL,run_slot timestamptz NOT NULL,day date NOT NULL,kind text NOT NULL,
 charged integer NOT NULL CHECK(charged>=0)
);
'''

def slot(now):
    now=now.astimezone(timezone.utc)
    return now.replace(hour=now.hour//6*6,minute=0,second=0,microsecond=0)


def setup(conn,now):
    anchor=slot(now)
    initialize(conn,anchor)
    with conn,conn.cursor() as cur:
        cur.execute(SCHEMA)
        initialize_research(cur)
        cur.execute('INSERT INTO crypto_rolling_campaign VALUES (%s,%s,%s) ON CONFLICT DO NOTHING',(CAMPAIGN,now,now+timedelta(days=30)))
        cur.execute('SELECT end_at FROM crypto_rolling_campaign WHERE id=%s',(CAMPAIGN,))
        if now>=cur.fetchone()[0]:return False
        for coin,(name,query,address,note) in TRACKED.items():
            cur.execute('INSERT INTO crypto_social_coins VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING',(coin,name,query,address,note))
            cur.execute('INSERT INTO crypto_rolling_coins(campaign_id,coin,credit_limit) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING',(CAMPAIGN,coin,COIN_LIMIT))
            cur.execute('''SELECT max(w.end_at) FROM crypto_social_windows w JOIN crypto_rolling_windows r ON r.window_id=w.id
                WHERE r.campaign_id=%s AND w.coin=%s AND NOT EXISTS(SELECT 1 FROM crypto_voice_focus f WHERE f.window_id=w.id)''',(CAMPAIGN,coin))
            last=cur.fetchone()[0]
            end=last+timedelta(hours=6) if last else anchor-timedelta(hours=42)
            while end<=anchor:
                start=end-timedelta(hours=7)  # one-hour overlap catches delayed indexing
                q=f'{query} since_time:{int(start.timestamp())} until_time:{int(end.timestamp())}'
                cur.execute('INSERT INTO crypto_social_windows(coin,start_at,end_at,query) VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING',(coin,start,end,q))
                cur.execute('SELECT id FROM crypto_social_windows WHERE coin=%s AND start_at=%s AND end_at=%s AND query=%s',(coin,start,end,q))
                row=cur.fetchone()
                if not row:raise ValueError('Conflicting rolling window; preserve existing evidence')
                cur.execute('INSERT INTO crypto_rolling_windows VALUES (%s,%s) ON CONFLICT DO NOTHING',(CAMPAIGN,row[0]))
                end+=timedelta(hours=6)
        # Preserve broad-search evidence, but retire its unfinished queue after tightening identity queries.
        cur.execute("""UPDATE crypto_social_windows w SET status='retired_query' WHERE w.status IN ('pending','partial')
          AND w.coin IN ('PONS','STANDARD') AND position('$' in w.query)>0 AND EXISTS(
          SELECT 1 FROM crypto_rolling_windows r WHERE r.window_id=w.id AND r.campaign_id=%s)""",(CAMPAIGN,))
    return True


def setup_focus(conn,coin,now):
    address=TRACKED[coin][2]
    if not address:return
    with conn,conn.cursor() as cur:
        cur.execute('SELECT 1 FROM crypto_voice_focus WHERE coin=%s LIMIT 1',(coin,))
        if cur.fetchone():return
        cur.execute('SELECT min(posted_at) FROM crypto_social_posts WHERE position(%s in '+('lower(text)' if address.startswith('0x') else 'text')+')>0',(address.lower() if address.startswith('0x') else address,))
        anchor=cur.fetchone()[0]
        if not anchor:return
        start=anchor.replace(minute=0,second=0,microsecond=0)-timedelta(hours=6)
        for i in range(30):
            a=start+timedelta(hours=i);b=a+timedelta(hours=1)
            if b>now:continue
            q=f'"{address}" since_time:{int(a.timestamp())} until_time:{int(b.timestamp())}'
            cur.execute('INSERT INTO crypto_social_windows(coin,start_at,end_at,query) VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING',(coin,a,b,q))
            cur.execute('SELECT id FROM crypto_social_windows WHERE coin=%s AND start_at=%s AND end_at=%s AND query=%s',(coin,a,b,q))
            row=cur.fetchone()
            if row:
                cur.execute('INSERT INTO crypto_rolling_windows VALUES (%s,%s) ON CONFLICT DO NOTHING',(CAMPAIGN,row[0]))
                cur.execute('INSERT INTO crypto_voice_focus VALUES (%s,%s) ON CONFLICT DO NOTHING',(row[0],coin))


def reserve(conn,coin,now,kind='posts'):
    with conn,conn.cursor() as cur:
        cur.execute('SELECT id FROM crypto_social_pilot WHERE id=%s FOR UPDATE',(PILOT,))
        cur.execute("SELECT count(*) FROM crypto_social_requests WHERE status IN ('reserved','uncertain')")
        if cur.fetchone()[0]:raise RuntimeError('Outstanding or uncertain request; review ledger before collecting')
        cur.execute('SELECT end_at FROM crypto_rolling_campaign WHERE id=%s',(CAMPAIGN,))
        if now>=cur.fetchone()[0]:return None
        cur.execute('SELECT used_credits,credit_limit FROM crypto_rolling_coins WHERE campaign_id=%s AND coin=%s FOR UPDATE',(CAMPAIGN,coin))
        used,limit=cur.fetchone()
        cur.execute('SELECT coalesce(sum(charged),0) FROM crypto_rolling_calls WHERE campaign_id=%s AND coin=%s AND day=%s',(CAMPAIGN,coin,now.date()))
        daily=cur.fetchone()[0]
        cur.execute('SELECT count(*) FROM crypto_rolling_calls WHERE campaign_id=%s AND coin=%s AND kind=%s AND '+('day=%s' if kind!='posts' else 'run_slot=%s'),(CAMPAIGN,coin,kind,now.date() if kind!='posts' else slot(now)))
        cur_slot_count=cur.fetchone()[0]
        if cur_slot_count>=(1 if kind!='posts' else PAGES_PER_RUN):return None
        ids=[];window=None
        if kind=='engagement':
            cur.execute("""SELECT p.id FROM crypto_social_posts p WHERE p.kind='original' AND p.posted_at BETWEEN %s AND %s
              AND EXISTS(SELECT 1 FROM crypto_social_matches m JOIN crypto_social_windows w ON w.id=m.window_id WHERE m.post_id=p.id AND w.coin=%s)
              AND NOT EXISTS(SELECT 1 FROM crypto_social_snapshots s WHERE s.post_id=p.id AND s.observed_at-p.posted_at BETWEEN interval '24 hours' AND interval '30 hours')
              ORDER BY p.posted_at,p.id LIMIT 10""",(now-timedelta(hours=30),now-timedelta(hours=24),coin))
            ids=[r[0] for r in cur.fetchall()]
            if not ids:return None
            credits=300 # Conservative page reservation, release only after a validated response.
        elif kind=='profiles':
            cur.execute('SELECT selection FROM crypto_voice_snapshots WHERE coin=%s ORDER BY week DESC LIMIT 1',(coin,))
            cohort=cur.fetchone()
            ids=[a['id'] for a in cohort[0]][:20] if cohort else []
            if not ids:return None
            credits=len(ids)*18
        else:
            cur.execute('''SELECT w.id,w.start_at,w.end_at,w.query,w.cursor FROM crypto_social_windows w
                JOIN crypto_rolling_windows r ON r.window_id=w.id WHERE r.campaign_id=%s AND w.coin=%s
                AND w.status IN ('pending','partial') AND w.end_at<=%s
                ORDER BY CASE WHEN EXISTS(SELECT 1 FROM crypto_voice_focus f WHERE f.window_id=w.id)=%s THEN 0 ELSE 1 END,CASE WHEN %s THEN CASE WHEN w.pages=0 THEN 0 ELSE 1 END ELSE CASE WHEN w.pages>0 THEN 0 ELSE 1 END END,
                CASE WHEN w.pages>0 THEN w.start_at END ASC,w.end_at DESC,w.id LIMIT 1 FOR UPDATE OF w''',(CAMPAIGN,coin,slot(now),cur_slot_count==3,cur_slot_count==0))
            window=cur.fetchone()
            if not window:return None
            credits=300
        if used+credits>limit or daily+credits>DAILY_LIMIT:return None
        parameters={'campaign':CAMPAIGN,'coin':coin,'kind':kind,'run_slot':slot(now).isoformat(),'user_ids':ids}
        cur.execute('UPDATE crypto_rolling_coins SET used_credits=used_credits+%s WHERE campaign_id=%s AND coin=%s',(credits,CAMPAIGN,coin))
        cur.execute('''INSERT INTO crypto_social_requests(window_id,endpoint,parameters,reserved_credits)
            VALUES (%s,%s,%s,%s) RETURNING id''',(window[0] if window else None,'rolling_'+kind,json.dumps(parameters),credits))
        rid=cur.fetchone()[0]
        cur.execute('INSERT INTO crypto_rolling_calls VALUES (%s,%s,%s,%s,%s,%s,%s)',(rid,CAMPAIGN,coin,slot(now),now.date(),kind,credits))
        return rid,window,ids,credits


def collect_one(conn,key,coin,now,kind='posts',fetch=None):
    import requests
    item=reserve(conn,coin,now,kind)
    if item is None:return False
    rid,window,ids,credits=item
    try:
        endpoint='/twitter/user/batch_info_by_ids' if kind=='profiles' else '/twitter/tweets' if kind=='engagement' else '/twitter/tweet/advanced_search'
        params={('tweet_ids' if kind=='engagement' else 'userIds'):','.join(ids)} if ids else {'query':window[3],'queryType':'Latest','cursor':window[4]}
        response=(fetch or requests.get)(BASE+endpoint,params=params,headers={'X-API-Key':key},timeout=30,allow_redirects=False)
        if response.status_code!=200:raise ValueError(f'HTTP {response.status_code}')
        data=response.json()
        if not isinstance(data,dict) or data.get('status') not in (None,'success'):raise ValueError('Provider error')
        if kind=='posts':
            post_ids=[str(t.get('id','')) for t in data.get('tweets',[])]
            if post_ids:
                with conn,conn.cursor() as cur:
                    cur.execute('SELECT count(*) FROM crypto_social_matches WHERE window_id=%s AND post_id=ANY(%s)',(window[0],post_ids))
                    if cur.fetchone()[0]==len(post_ids):raise ValueError('Repeated page; stop before paying for a loop')
            save_page(conn,rid,window,data)
        with conn,conn.cursor() as cur:
            if kind=='engagement':
                returned,accepted,estimated=save_posts(cur,rid,data,now-timedelta(hours=30),now,refresh_ids=ids)
                cur.execute("UPDATE crypto_social_requests SET status='saved',returned_count=%s,accepted_count=%s,estimated_credits=%s WHERE id=%s",(returned,accepted,estimated,rid))
            elif kind=='profiles':
                returned,accepted,estimated=save_profiles(cur,rid,data,ids)
                cur.execute("UPDATE crypto_social_requests SET status='saved',returned_count=%s,accepted_count=%s,estimated_credits=%s WHERE id=%s",(returned,accepted,estimated,rid))
            else:
                cur.execute('SELECT estimated_credits FROM crypto_social_requests WHERE id=%s',(rid,))
                estimated=cur.fetchone()[0]
            if not 0<=estimated<=credits:raise ValueError('Charge exceeds reservation')
            # Release unused headroom only after a validated saved response.
            cur.execute('UPDATE crypto_rolling_coins SET used_credits=used_credits-%s WHERE campaign_id=%s AND coin=%s',(credits-estimated,CAMPAIGN,coin))
            cur.execute('UPDATE crypto_rolling_calls SET charged=%s WHERE request_id=%s',(estimated,rid))
    except Exception:
        with conn,conn.cursor() as cur:
            cur.execute("UPDATE crypto_social_requests SET status='uncertain',error='rolling_request_failed' WHERE id=%s",(rid,))
        raise RuntimeError(f'Rolling request {rid} uncertain; retained reservation and stopped. Review ledger.') from None
    print(json.dumps({'coin':coin,'kind':kind,'request_id':rid,'estimated_credits':estimated}),flush=True)
    return True


def report(conn):
    with conn,conn.cursor() as cur:
        cur.execute('''SELECT c.coin,c.used_credits,c.credit_limit,count(DISTINCT p.id),count(DISTINCT p.author_id),max(p.posted_at),
            count(DISTINCT w.id) FILTER(WHERE w.status IN ('pending','partial'))
            FROM crypto_rolling_coins c LEFT JOIN crypto_rolling_windows r ON r.campaign_id=c.campaign_id
            LEFT JOIN crypto_social_windows w ON w.id=r.window_id AND w.coin=c.coin
            LEFT JOIN crypto_social_matches m ON m.window_id=w.id LEFT JOIN crypto_social_posts p ON p.id=m.post_id
            WHERE c.campaign_id=%s GROUP BY c.coin,c.used_credits,c.credit_limit ORDER BY c.coin''',(CAMPAIGN,))
        return [dict(zip(['coin','charged_or_reserved','ceiling','posts','authors','latest_post','unfinished_windows'],row)) for row in cur.fetchall()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps({'mode':'plan_only','coins':list(TRACKED),'days':30,'total_ceiling':COIN_LIMIT*len(TRACKED),'per_coin_ceiling':COIN_LIMIT,'pages_per_coin_per_run':PAGES_PER_RUN,'profiles_per_coin_daily':20,'paid_calls':0}));return
    import psycopg2
    conn=psycopg2.connect(os.environ['DATABASE_URL'],connect_timeout=15)
    try:
        now=datetime.now(timezone.utc)
        if setup(conn,now):
            key=os.environ['TWITTERAPI_IO_API_KEY']
            evaluate(conn,now)
            for coin in TRACKED:
                setup_focus(conn,coin,now)
                snapshot(conn,coin,now)
                collect_one(conn,key,coin,now,'engagement')
                for _ in range(PAGES_PER_RUN):
                    if not collect_one(conn,key,coin,now):break
                collect_one(conn,key,coin,now,'profiles')
        print(json.dumps({'campaign':CAMPAIGN,'results':report(conn)},default=str),flush=True)
    finally:conn.close()


if __name__=='__main__':main()

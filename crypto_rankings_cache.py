"""Collection-time ranking snapshots for the crypto research routes. No provider calls, no X credits.

Loads every saved row (no 50,000-row truncation), ranks it with the dashboard's own TypeScript
via node, and stores one JSON payload per (key, version) in crypto_ranking_cache. Routes serve
the snapshot when present and compute live otherwise, so nothing depends on this having run.
"""
import argparse,json,os,subprocess
from datetime import datetime,timezone
from pathlib import Path
from crypto_coins import SYMBOLS

ROOT=Path(__file__).parent
QUERIES=json.loads(ROOT.joinpath('apps','web','lib','server','crypto-ranking-queries.json').read_text())
SCHEMA='''CREATE TABLE IF NOT EXISTS crypto_ranking_cache (
 key text NOT NULL,version text NOT NULL,payload jsonb NOT NULL,rows_loaded integer NOT NULL,
 computed_at timestamptz NOT NULL DEFAULT now(),PRIMARY KEY(key,version));'''


def rank(kind,coin,posts,unfinished=0,run=None):
    run=run or (lambda payload:subprocess.run(['node','--experimental-strip-types','scripts/crypto-rankings.mts'],cwd=ROOT,input=payload,text=True,capture_output=True,check=True).stdout)
    return json.loads(run(json.dumps({'kind':kind,'coin':coin,'posts':posts,'unfinished':unfinished},default=str)))


def _rows(cur):
    names=[d[0] for d in cur.description];return [dict(zip(names,r)) for r in cur.fetchall()]


def refresh(conn,kinds=('voices','watchers'),run=None,now=None):
    now=now or datetime.now(timezone.utc);written=[]
    with conn,conn.cursor() as cur:cur.execute(SCHEMA)
    quoted=','.join("'"+s+"'" for s in SYMBOLS)
    with conn,conn.cursor() as cur:
        if 'watchers' in kinds:
            cur.execute("SELECT to_regclass('crypto_social_profile_history')")
            if cur.fetchone()[0]:
                cur.execute(QUERIES['watchers'].replace('__COINS__',quoted).replace('LIMIT 50000','LIMIT 500000'))
                posts=_rows(cur)
                for coin in ['ALL']+SYMBOLS:
                    result=rank('watchers',coin,posts,run=run)
                    cur.execute('INSERT INTO crypto_ranking_cache(key,version,payload,rows_loaded,computed_at) VALUES (%s,%s,%s,%s,%s) ON CONFLICT(key,version) DO UPDATE SET payload=EXCLUDED.payload,rows_loaded=EXCLUDED.rows_loaded,computed_at=EXCLUDED.computed_at',
                        ('watchers:'+coin,result['version'],json.dumps(result['payload'],default=str),len(posts),now))
                    written.append({'key':'watchers:'+coin,'rows':len(posts),'candidates':result['payload']['candidates']})
        if 'voices' in kinds:
            for coin in SYMBOLS:
                cur.execute(QUERIES['voices'].replace('__COIN__','%s').replace('__LIMIT__','500000'),(coin,))
                posts=_rows(cur)
                cur.execute(QUERIES['voices_unfinished'].replace('__COIN__','%s'),(coin,))
                unfinished=cur.fetchone()[0]
                result=rank('voices',coin,posts,unfinished,run=run)
                cur.execute('INSERT INTO crypto_ranking_cache(key,version,payload,rows_loaded,computed_at) VALUES (%s,%s,%s,%s,%s) ON CONFLICT(key,version) DO UPDATE SET payload=EXCLUDED.payload,rows_loaded=EXCLUDED.rows_loaded,computed_at=EXCLUDED.computed_at',
                    ('voices:'+coin,result['version'],json.dumps(result['payload'],default=str),len(posts),now))
                written.append({'key':'voices:'+coin,'rows':len(posts),'voices':len(result['payload']['voices'])})
    return written


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--kinds',default='voices,watchers');args=parser.parse_args()
    kinds=tuple(k for k in args.kinds.split(',') if k)
    if not args.execute:print(json.dumps({'mode':'plan_only','kinds':kinds,'network_requests':0,'twitter_credits':0}));return
    import psycopg2
    conn=psycopg2.connect(os.environ['DATABASE_URL'],connect_timeout=15)
    try:print(json.dumps({'written':refresh(conn,kinds)},default=str),flush=True)
    finally:conn.close()

if __name__=='__main__':main()

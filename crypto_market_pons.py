"""Archive a pinned PONS pool with exact Robinhood Chain contract matching."""
import argparse
from datetime import datetime,timezone,date
import json
import os
import time
import requests
from crypto_market_history import setup,pools,normalize,save
from crypto_social_history import PONS_ADDRESS

BASE='https://api.geckoterminal.com/api/v2/networks/robinhood'

def refresh(conn):
    setup(conn)
    def get(url):
        time.sleep(3)
        r=requests.get(url,timeout=25,allow_redirects=False)
        if r.status_code!=200:raise ValueError(f'Market HTTP {r.status_code}')
        return r.json()
    catalog=get(BASE+'/tokens/'+PONS_ADDRESS+'/pools')
    candidates=pools(catalog,PONS_ADDRESS,'robinhood')
    with conn,conn.cursor() as cur:
        cur.execute("SELECT metadata FROM crypto_market_sources WHERE coin='PONS' AND is_default")
        pinned=cur.fetchone()
    pool=pinned[0] if pinned else min(candidates,key=lambda p:(p['created'],p['id'])) if candidates else None
    if not pool:raise ValueError('No verified PONS pool')
    url=BASE+'/pools/'+pool['id']+'/ohlcv/day?aggregate=1&limit=100&currency=usd&include_empty_intervals=false&token='+pool['side']
    raw=get(url);now=datetime.now(timezone.utc)
    points=normalize(raw,'ohlcv',now,start=date(2026,7,1))
    source=dict(id='geckoterminal:robinhood:'+pool['id'],coin='PONS',provider='GeckoTerminal',
                url='https://www.geckoterminal.com/robinhood/pools/'+pool['id'],metadata=pool)
    fid=save(conn,source,{'catalog':catalog,'ohlcv':raw},points,url,now)
    with conn,conn.cursor() as cur:
        cur.execute("UPDATE crypto_market_sources SET is_default=true WHERE id=%s AND NOT EXISTS(SELECT 1 FROM crypto_market_sources WHERE coin='PONS' AND is_default)",(source['id'],))
    return {'coin':'PONS','fetch_id':fid,'points':len(points),'pool':pool['id'],'twitter_credits':0}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:print(json.dumps({'mode':'plan_only','public_requests':2,'twitter_credits':0}))
    else:
        import psycopg2
        conn=psycopg2.connect(os.environ['DATABASE_URL'],connect_timeout=15)
        try:print(json.dumps(refresh(conn)))
        finally:conn.close()

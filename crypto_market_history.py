"""Archive public ZCAT/ZEC market history. No X calls; default is a no-network plan."""
import argparse
from datetime import datetime, timezone, date
import json
import math
import os
from pathlib import Path
import re

ADDRESS='HcRLc9VDgjLeK154xDawfb1dmVJ98DoSqcwTHGqiDeJR'
START=date(2026,7,25)
BASE='https://api.geckoterminal.com/api/v2/networks/solana'
ZEC_URL='https://api.coingecko.com/api/v3/coins/zcash/market_chart?vs_currency=usd&days=90&interval=daily'


def setup(conn):
    with conn,conn.cursor() as cur:
        cur.execute(Path(__file__).with_name('sql').joinpath('crypto_market_history.sql').read_text())


def finite(value):
    return isinstance(value,(float,int)) and not isinstance(value,bool) and math.isfinite(value)


def pools(data):
    result=[]
    for p in data.get('data',[]):
        a=p.get('attributes') or {};r=p.get('relationships') or {}
        base=(r.get('base_token') or {}).get('data',{}).get('id')
        quote=(r.get('quote_token') or {}).get('data',{}).get('id')
        side='base' if base=='solana_'+ADDRESS else 'quote' if quote=='solana_'+ADDRESS else None
        try:
            created=datetime.fromisoformat(a['pool_created_at'].replace('Z','+00:00'))
            liquidity=float(a.get('reserve_in_usd') or 0)
        except (KeyError,TypeError,ValueError):continue
        if not side or not re.fullmatch(r'[1-9A-HJ-NP-Za-km-z]{32,44}',a.get('address','')) or not created.tzinfo:continue
        result.append(dict(id=a['address'],name=a.get('name',a['address']),created=created.isoformat(),
            liquidity=liquidity if math.isfinite(liquidity) else 0,side=side,contract=ADDRESS))
    return sorted(result,key=lambda p:(-p['liquidity'],p['id']))[:5]


def normalize(data,kind,now):
    points={}
    if kind=='ohlcv':
        rows=data.get('data',{}).get('attributes',{}).get('ohlcv_list')
        if not isinstance(rows,list):raise ValueError('Missing candles')
    else:
        prices=data.get('prices');volumes=data.get('total_volumes')
        if not isinstance(prices,list) or not isinstance(volumes,list):raise ValueError('Missing observations')
        vol={v[0]:v[1] for v in volumes if isinstance(v,list) and len(v)>=2 and finite(v[0]) and finite(v[1])}
        rows=[[p[0]/1000,None,None,None,p[1],vol.get(p[0])] for p in prices
              if isinstance(p,list) and len(p)>=2 and finite(p[0])]
    for row in rows:
        if not isinstance(row,list) or len(row)<6 or not all(finite(row[i]) for i in [0,4,5]):continue
        if row[4]<=0 or row[5]<0:continue
        if kind=='ohlcv' and (not all(finite(row[i]) and row[i]>0 for i in [1,2,3]) or row[2]<max(row[1],row[3],row[4]) or row[3]>min(row[1],row[2],row[4])):continue
        try:stamp=datetime.fromtimestamp(row[0],timezone.utc)
        except (ValueError,OverflowError,OSError):continue
        if stamp>now or stamp.date()<START:continue
        day=stamp.date()
        point=dict(day=day,sample_at=stamp,close=row[4],volume=row[5],open=row[1],high=row[2],low=row[3],
                   complete=day<now.date(),kind=kind)
        # CoinGecko may return both midnight and latest observations for the same day.
        if day not in points or stamp>points[day]['sample_at']:points[day]=point
    return [points[d] for d in sorted(points)]


def save(conn,source,data,points,url,now):
    if not points:raise ValueError('No valid market observations')
    from psycopg2.extras import Json,execute_values
    with conn,conn.cursor() as cur:
        cur.execute('''INSERT INTO crypto_market_sources(id,coin,provider,source_url,metadata)
            VALUES (%s,%s,%s,%s,%s) ON CONFLICT(id) DO UPDATE SET metadata=EXCLUDED.metadata''',
            (source['id'],source['coin'],source['provider'],source['url'],Json(source['metadata'])))
        cur.execute('''INSERT INTO crypto_market_fetches(source_id,retrieved_at,request_url,metadata,raw_response)
            VALUES (%s,%s,%s,%s,%s) RETURNING id''',(source['id'],now,url,Json(source['metadata']),Json(data)))
        fid=cur.fetchone()[0]
        execute_values(cur,'''INSERT INTO crypto_market_observations(fetch_id,day,sample_at,close,volume,open,high,low,complete,kind) VALUES %s''',
            [(fid,p['day'],p['sample_at'],p['close'],p['volume'],p['open'],p['high'],p['low'],p['complete'],p['kind']) for p in points])
    return fid


def refresh(conn,fetch=None,now=None):
    import requests
    fetch=fetch or requests.get;now=now or datetime.now(timezone.utc);setup(conn)
    saved=[];errors=[]
    # Database lock also serializes invocations outside GitHub Actions.
    with conn,conn.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(hashtext('crypto-market-history'))")
        if not cur.fetchone()[0]:return {'saved':[],'errors':['refresh_already_running']}
    def get(url):
        response=fetch(url,timeout=25,allow_redirects=False,headers={'Accept':'application/json'})
        if response.status_code!=200:raise ValueError('Market HTTP '+str(response.status_code))
        return response.json()
    try:
        try:
            catalog=get(BASE+'/tokens/'+ADDRESS+'/pools');selected=pools(catalog)
            with conn,conn.cursor() as cur:
                cur.execute("SELECT metadata FROM crypto_market_sources WHERE coin='ZCAT' AND is_default")
                row=cur.fetchone()
            # Keep refreshing the pinned source even after it leaves the top five.
            if row and not any(p['id']==row[0]['id'] for p in selected):selected.append(row[0])
            for pool in sorted(selected,key=lambda p:(p['created'],p['id'])):
                try:
                    url=BASE+'/pools/'+pool['id']+'/ohlcv/day?aggregate=1&limit=100&currency=usd&include_empty_intervals=false&token='+pool['side']
                    raw=get(url);points=normalize(raw,'ohlcv',now)
                    source=dict(id='geckoterminal:'+pool['id'],coin='ZCAT',provider='GeckoTerminal',url='https://www.geckoterminal.com/solana/pools/'+pool['id'],metadata=pool)
                    fid=save(conn,source,{'catalog':catalog,'ohlcv':raw},points,url,now)
                    saved.append({'source':source['id'],'fetch_id':fid,'points':len(points)})
                except (ValueError,requests.RequestException) as exc:errors.append('ZCAT pool: '+type(exc).__name__)
        except (ValueError,requests.RequestException) as exc:errors.append('ZCAT catalog: '+type(exc).__name__)
        try:
            raw=get(ZEC_URL);points=normalize(raw,'price_observation',now)
            source=dict(id='coingecko:zcash',coin='ZEC',provider='CoinGecko',url='https://www.coingecko.com/en/coins/zcash',metadata={'id':'zcash'})
            fid=save(conn,source,raw,points,ZEC_URL,now);saved.append({'source':source['id'],'fetch_id':fid,'points':len(points)})
        except (ValueError,requests.RequestException) as exc:errors.append('ZEC: '+type(exc).__name__)
        # Pin only after observations have been saved. Never silently switch sources later.
        with conn,conn.cursor() as cur:
            for coin in ['ZCAT','ZEC']:
                cur.execute('''UPDATE crypto_market_sources SET is_default=true WHERE id=(
                    SELECT id FROM crypto_market_sources WHERE coin=%s
                    ORDER BY metadata->>'created',id LIMIT 1)
                    AND NOT EXISTS(SELECT 1 FROM crypto_market_sources WHERE coin=%s AND is_default)''',(coin,coin))
    finally:
        with conn,conn.cursor() as cur:cur.execute("SELECT pg_advisory_unlock(hashtext('crypto-market-history'))")
    return {'saved':saved,'errors':errors,'twitter_credits':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:
        print(json.dumps({'mode':'plan_only','max_public_requests':8,'twitter_credits':0,'database_writes':0}));return
    import psycopg2
    conn=psycopg2.connect(os.environ['DATABASE_URL'],connect_timeout=15)
    try:
        result=refresh(conn);print(json.dumps(result));
        if result['errors']:raise SystemExit(1)
    finally:conn.close()

if __name__=='__main__':main()

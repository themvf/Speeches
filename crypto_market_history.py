"""Archive public market history for every tracked coin. No X calls; default is a no-network plan.

Daily candles are archived for up to five pools per contract coin; hourly candles for the
pinned default pool only, because the post-to-price event study reads one source per coin.
"""
import argparse
from datetime import datetime, timedelta, timezone, date
import json
import math
import os
from pathlib import Path
import re
from crypto_coins import markets

ADDRESS='HcRLc9VDgjLeK154xDawfb1dmVJ98DoSqcwTHGqiDeJR'
START=date(2026,7,25)
GECKO='https://api.geckoterminal.com/api/v2/networks/'
BASE=GECKO+'solana'
ZEC_URL='https://api.coingecko.com/api/v3/coins/zcash/market_chart?vs_currency=usd&days=90&interval=daily'
ZEC_HOURLY_URL='https://api.coingecko.com/api/v3/coins/zcash/market_chart?vs_currency=usd&days=30'
# Contract coins: network, address, first archived day (from the shared registry).
MARKETS=markets()


def setup(conn):
    with conn,conn.cursor() as cur:
        cur.execute(Path(__file__).with_name('sql').joinpath('crypto_market_history.sql').read_text())


def finite(value):
    return isinstance(value,(float,int)) and not isinstance(value,bool) and math.isfinite(value)


def pools(data,address=ADDRESS,network='solana'):
    result=[]
    for p in data.get('data',[]):
        a=p.get('attributes') or {};r=p.get('relationships') or {}
        base=(r.get('base_token') or {}).get('data',{}).get('id')
        quote=(r.get('quote_token') or {}).get('data',{}).get('id')
        expected=network+'_'+address
        if network!='solana':base=str(base).lower();quote=str(quote).lower();expected=expected.lower()
        side='base' if base==expected else 'quote' if quote==expected else None
        try:
            created=datetime.fromisoformat(a['pool_created_at'].replace('Z','+00:00'))
            liquidity=float(a.get('reserve_in_usd') or 0)
        except (KeyError,TypeError,ValueError):continue
        if not side or not re.fullmatch(r'[1-9A-HJ-NP-Za-km-z]{32,44}' if network=='solana' else r'0x(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})',a.get('address','')) or not created.tzinfo:continue
        result.append(dict(id=a['address'],name=a.get('name',a['address']),created=created.isoformat(),
            liquidity=liquidity if math.isfinite(liquidity) else 0,side=side,contract=address))
    return sorted(result,key=lambda p:(-p['liquidity'],p['id']))[:5]


def _rows(data,kind):
    if kind=='ohlcv':
        rows=data.get('data',{}).get('attributes',{}).get('ohlcv_list')
        if not isinstance(rows,list):raise ValueError('Missing candles')
    else:
        prices=data.get('prices');volumes=data.get('total_volumes')
        if not isinstance(prices,list) or not isinstance(volumes,list):raise ValueError('Missing observations')
        # total_volumes is a ROLLING 24-HOUR total, not the volume traded in this interval: the
        # series drifts a couple of percent an hour where a real hourly series swings by tens.
        # Storing it would put a smoothed window into a column every reader treats as per-interval
        # trading, so no volume is recorded. Its presence is still required, because a payload
        # missing it is not the shape the provider documents.
        rows=[[p[0]/1000,None,None,None,p[1],None] for p in prices
              if isinstance(p,list) and len(p)>=2 and finite(p[0])]
    for row in rows:
        if not isinstance(row,list) or len(row)<6 or not all(finite(row[i]) for i in [0,4]):continue
        if row[4]<=0:continue
        # Volume is absent only where the provider does not measure the interval; a source that
        # does report it must report it validly.
        if kind=='ohlcv' and (not finite(row[5]) or row[5]<0):continue
        if kind!='ohlcv' and row[5] is not None:continue
        if kind=='ohlcv' and (not all(finite(row[i]) and row[i]>0 for i in [1,2,3]) or row[2]<max(row[1],row[3],row[4]) or row[3]>min(row[1],row[2],row[4])):continue
        try:stamp=datetime.fromtimestamp(row[0],timezone.utc)
        except (ValueError,OverflowError,OSError):continue
        yield stamp,row


def normalize(data,kind,now,start=START):
    points={}
    for stamp,row in _rows(data,kind):
        if stamp>now or stamp.date()<start:continue
        day=stamp.date()
        point=dict(day=day,sample_at=stamp,close=row[4],volume=row[5],open=row[1],high=row[2],low=row[3],
                   complete=day<now.date(),kind=kind)
        # CoinGecko may return both midnight and latest observations for the same day.
        if day not in points or stamp>points[day]['sample_at']:points[day]=point
    return [points[d] for d in sorted(points)]


def normalize_hourly(data,kind,now,start=START):
    """Hour buckets keyed by their UTC start; a bucket is complete once the following hour has begun."""
    points={}
    for stamp,row in _rows(data,kind):
        if stamp>now or stamp.date()<start:continue
        hour=stamp.replace(minute=0,second=0,microsecond=0)
        point=dict(hour=hour,sample_at=stamp,close=row[4],volume=row[5],open=row[1],high=row[2],low=row[3],
                   complete=hour+timedelta(hours=1)<=now,kind=kind)
        if hour not in points or stamp>points[hour]['sample_at']:points[hour]=point
    return [points[h] for h in sorted(points)]


def save(conn,source,data,points,url,now,hourly=False):
    if not points:raise ValueError('No valid market observations')
    from psycopg2.extras import Json,execute_values
    table,key=('crypto_market_hourly','hour') if hourly else ('crypto_market_observations','day')
    with conn,conn.cursor() as cur:
        cur.execute('''INSERT INTO crypto_market_sources(id,coin,provider,source_url,metadata)
            VALUES (%s,%s,%s,%s,%s) ON CONFLICT(id) DO UPDATE SET metadata=EXCLUDED.metadata''',
            (source['id'],source['coin'],source['provider'],source['url'],Json(source['metadata'])))
        cur.execute('''INSERT INTO crypto_market_fetches(source_id,retrieved_at,request_url,metadata,raw_response)
            VALUES (%s,%s,%s,%s,%s) RETURNING id''',(source['id'],now,url,Json(source['metadata']),Json(data)))
        fid=cur.fetchone()[0]
        execute_values(cur,f'''INSERT INTO {table}(fetch_id,{key},sample_at,close,volume,open,high,low,complete,kind) VALUES %s''',
            [(fid,p[key],p['sample_at'],p['close'],p['volume'],p['open'],p['high'],p['low'],p['complete'],p['kind']) for p in points])
    return fid


def source_id(network,pool):
    # Existing archives used 'geckoterminal:<pool>' for Solana and 'geckoterminal:robinhood:<pool>' elsewhere.
    return 'geckoterminal:'+pool if network=='solana' else 'geckoterminal:'+network+':'+pool


def refresh(conn,fetch=None,now=None,wait=None):
    import requests
    import time
    wait=wait or time.sleep
    fetch=fetch or requests.get;now=now or datetime.now(timezone.utc);setup(conn)
    saved=[];errors=[];skipped=[]
    # Database lock also serializes invocations outside GitHub Actions.
    with conn,conn.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(hashtext('crypto-market-history'))")
        if not cur.fetchone()[0]:return {'saved':[],'errors':['refresh_already_running']}
    def get(url):
        for attempt in range(4):
            wait(4)  # Public market APIs are shared and rate-limited; never burst requests.
            response=fetch(url,timeout=25,allow_redirects=False,headers={'Accept':'application/json'})
            if response.status_code==429 and attempt<3:
                try:delay=float(getattr(response,'headers',{}).get('Retry-After','15'))
                except (ValueError,TypeError):delay=15
                if not math.isfinite(delay):delay=15
                if delay>30:raise ValueError('Market cooldown exceeds retry bound; retain saved history')
                wait(max(3,delay))
                continue
            if response.status_code!=200:raise ValueError('Market HTTP '+str(response.status_code))
            return response.json()
        raise ValueError('Market rate limit')
    try:
        for coin,(network,address,start) in MARKETS.items():
            try:
                catalog=get(GECKO+network+'/tokens/'+address+'/pools');selected=pools(catalog,address,network)
                with conn,conn.cursor() as cur:
                    cur.execute("SELECT metadata FROM crypto_market_sources WHERE coin=%s AND is_default",(coin,))
                    row=cur.fetchone()
                # Keep refreshing the pinned source even after it leaves the top five.
                if row and not any(p['id']==row[0]['id'] for p in selected):selected.append(row[0])
                # A contract with no indexed pool is a coverage gap, not a failed run.
                if not selected:skipped.append(coin+': no indexed pool for this contract');continue
                # The pin rule is the oldest pool among those archived; the hourly series follows the same choice.
                primary=row[0]['id'] if row else min(selected,key=lambda p:(p['created'],p['id']))['id']
                # Daily candles: the pinned pool plus the two most liquid others; every extra pool is another rate-limited call.
                daily=[p for p in sorted(selected,key=lambda p:(-p['liquidity'],p['id'])) if p['id']!=primary][:2]
                for pool in sorted(selected,key=lambda p:(p['created'],p['id'])):
                    if pool['id']!=primary and pool not in daily:continue
                    source=dict(id=source_id(network,pool['id']),coin=coin,provider='GeckoTerminal',url='https://www.geckoterminal.com/'+network+'/pools/'+pool['id'],metadata=pool)
                    for timeframe,hourly in [('day',False),('hour',True)]:
                        if hourly and pool['id']!=primary:continue
                        try:
                            url=GECKO+network+'/pools/'+pool['id']+'/ohlcv/'+timeframe+'?aggregate=1&limit='+('1000' if hourly else '100')+'&currency=usd&include_empty_intervals=false&token='+pool['side']
                            raw=get(url);points=(normalize_hourly if hourly else normalize)(raw,'ohlcv',now,start=start)
                            fid=save(conn,source,{'catalog':catalog,'ohlcv':raw},points,url,now,hourly=hourly)
                            saved.append({'source':source['id'],'fetch_id':fid,'points':len(points),'timeframe':timeframe})
                        except (ValueError,requests.RequestException) as exc:errors.append(coin+' pool '+pool['id']+' '+timeframe+': '+type(exc).__name__+' '+str(exc)[:160])
            except (ValueError,requests.RequestException) as exc:errors.append(coin+' catalog: '+type(exc).__name__+' '+str(exc)[:160])
        source=dict(id='coingecko:zcash',coin='ZEC',provider='CoinGecko',url='https://www.coingecko.com/en/coins/zcash',metadata={'id':'zcash'})
        for url,hourly in [(ZEC_URL,False),(ZEC_HOURLY_URL,True)]:
            try:
                raw=get(url);points=(normalize_hourly if hourly else normalize)(raw,'price_observation',now)
                fid=save(conn,source,raw,points,url,now,hourly=hourly);saved.append({'source':source['id'],'fetch_id':fid,'points':len(points),'timeframe':'hour' if hourly else 'day'})
            except (ValueError,requests.RequestException) as exc:errors.append('ZEC'+(' hourly' if hourly else '')+': '+type(exc).__name__)
        # Pin only after observations have been saved. Never silently switch sources later.
        with conn,conn.cursor() as cur:
            for coin in list(MARKETS)+['ZEC']:
                cur.execute('''UPDATE crypto_market_sources SET is_default=true WHERE id=(
                    SELECT id FROM crypto_market_sources WHERE coin=%s
                    ORDER BY metadata->>'created',id LIMIT 1)
                    AND NOT EXISTS(SELECT 1 FROM crypto_market_sources WHERE coin=%s AND is_default)''',(coin,coin))
    finally:
        with conn,conn.cursor() as cur:cur.execute("SELECT pg_advisory_unlock(hashtext('crypto-market-history'))")
    return {'saved':saved,'errors':errors,'skipped':skipped,'twitter_credits':0}


def repair_observation_volume(conn):
    """Clear volumes written before price_observation sources stopped recording a rolling total.

    One-shot and idempotent; deliberately not on the sweep path, because a repeated scan of the
    observation tables on every run is the pattern that caused a production deadlock elsewhere.
    Rows are corrected rather than deleted: the price observations themselves were always valid.
    """
    cleared={}
    with conn,conn.cursor() as cur:
        for table in ('crypto_market_observations','crypto_market_hourly'):
            cur.execute('UPDATE '+table+" SET volume=NULL WHERE kind='price_observation' AND volume IS NOT NULL")
            cleared[table]=cur.rowcount
    return cleared


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--repair-volume',action='store_true',
                        help='clear rolling-window volumes recorded for price_observation sources, then exit')
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps({'mode':'plan_only','max_public_requests':30,'twitter_credits':0,'database_writes':0}));return
    import psycopg2
    conn=psycopg2.connect(os.environ['DATABASE_URL'],connect_timeout=15)
    if args.repair_volume:
        try:
            setup(conn);print(json.dumps({'repaired':repair_observation_volume(conn)}))
        finally:conn.close()
        return
    try:
        result=refresh(conn);print(json.dumps(result));
        # Provider rate limits on secondary pools are noise; a run with nothing archived is the failure.
        if not result['saved']:raise SystemExit(1)
    finally:conn.close()

if __name__=='__main__':main()

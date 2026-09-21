from datetime import datetime,timezone,timedelta
import pytest
from test_crypto_social_pilot import db
from crypto_market_history import (ADDRESS,MARKETS,normalize,normalize_hourly,pools,setup,save,refresh,
                                   repair_observation_volume)
CONTRACT_COINS=len(MARKETS)

NOW=datetime(2026,9,14,12,tzinfo=timezone.utc)
POOL='BTccxxTFi7a9xJTE1exKn38Jgie35s6gNeRxd8DM61Rc'

def catalog(address=ADDRESS):
    return {'data':[{'attributes':{'address':POOL,'name':'ZEC / ZCAT','pool_created_at':'2026-08-30T23:29:55Z','reserve_in_usd':'1000'},
       'relationships':{'quote_token':{'data':{'id':'solana_'+address}}}}]}

def candles(day='2026-09-13',close=2):
    return {'data':{'attributes':{'ohlcv_list':[[datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp(),1,max(3,close),.5,close,100]]}}}

def source():
    return dict(id='geckoterminal:'+POOL,coin='ZCAT',provider='GeckoTerminal',url='https://www.geckoterminal.com/solana/pools/'+POOL,metadata=pools(catalog())[0])

def test_exact_contract_and_quote_orientation():
    assert pools(catalog())[0]['side']=='quote'
    assert pools(catalog('different-token'))==[]

def test_partial_day_requires_a_later_retrieval_to_become_complete():
    assert not normalize(candles('2026-09-14'),'ohlcv',NOW)[0]['complete']
    assert normalize(candles('2026-09-14'),'ohlcv',NOW+timedelta(days=1))[0]['complete']

def test_observation_gaps_bad_values_and_future_days():
    data=candles();data['data']['attributes']['ohlcv_list'] += [[1e100,1,2,1,2,10],[1,1,2,1,float('nan'),10]]
    assert len(normalize(data,'ohlcv',NOW))==1
    assert normalize(candles('2026-09-15'),'ohlcv',NOW)==[]

def test_coingecko_daily_observation_preserves_time_and_is_not_ohlcv():
    t=int(NOW.timestamp()*1000);data={'prices':[[t-3600000,1],[t,2]],'total_volumes':[[t-3600000,5],[t,10]]}
    points=normalize(data,'price_observation',NOW)
    assert len(points)==1 and points[0]['close']==2 and points[0]['sample_at']==NOW
    assert points[0]['kind']=='price_observation' and points[0]['open'] is None

def test_db_archive_preserves_revisions_and_days_omitted_by_later_fetch(db):
    setup(db);s=source();a=candles('2026-09-12');b=candles('2026-09-13')
    a['data']['attributes']['ohlcv_list']+=b['data']['attributes']['ohlcv_list']
    first=save(db,s,a,normalize(a,'ohlcv',NOW),'https://example.test/a',NOW)
    second_raw=candles('2026-09-13',4)
    second=save(db,s,second_raw,normalize(second_raw,'ohlcv',NOW),'https://example.test/b',NOW+timedelta(hours=1))
    with db,db.cursor() as c:
        c.execute('SELECT count(*) FROM crypto_market_observations');assert c.fetchone()[0]==3
        c.execute('SELECT day,close,fetch_id FROM crypto_market_latest ORDER BY day');rows=c.fetchall()
        assert len(rows)==2 and rows[0][2]==first and rows[1][1:]==(4,second)
        c.execute('SELECT raw_response FROM crypto_market_fetches WHERE id=%s',(first,));assert c.fetchone()[0]==a

def test_db_refresh_partial_failure_preserves_saved_sources_and_pins_default(db):
    class Response:
        status_code=200
        def __init__(self,data):self.data=data
        def json(self):return self.data
    calls=[]
    def fetch(url,**kwargs):
        calls.append(url)
        if 'coingecko.com' in url:raise ValueError('provider unavailable')
        return Response(catalog() if url.endswith('/pools') else candles())
    result=refresh(db,fetch=fetch,now=NOW,wait=lambda _:None)
    # ZCAT daily + hourly saved; both ZEC requests fail; three other contract catalogs return no pool.
    assert len(result['saved'])==2 and len(result['errors'])==2 and len(calls)==CONTRACT_COINS+4
    with db,db.cursor() as c:
        c.execute('SELECT id FROM crypto_market_sources WHERE is_default');assert c.fetchone()[0]==source()['id']
        c.execute('SELECT count(*) FROM crypto_market_latest');assert c.fetchone()[0]==1


def test_db_market_rate_limit_has_one_bounded_retry_without_x_calls(db):
    calls=[];delays=[]
    class Response:
        headers={'Retry-After':'20'}
        def __init__(self,status,data=None):self.status_code=status;self.data=data
        def json(self):return self.data
    def fetch(url,**kwargs):
        calls.append(url)
        if calls.count(url)==1:return Response(429)
        if 'coingecko.com' in url:
            t=int(NOW.timestamp()*1000)
            return Response(200,{'prices':[[t,2]],'total_volumes':[[t,100]]})
        return Response(200,catalog() if url.endswith('/pools') else candles())
    result=refresh(db,fetch=fetch,now=NOW,wait=delays.append)
    assert not result['errors'] and len(result['saved'])==4
    assert len(calls)==2*(CONTRACT_COINS+4) and max(delays)==20 and min(delays)==4
    assert all('twitter' not in url for url in calls)


def test_db_long_provider_cooldown_is_not_shortened(db):
    calls=[]
    class Response:
        status_code=429
        headers={'Retry-After':'999'}
    def fetch(url,**kwargs):calls.append(url);return Response()
    result=refresh(db,fetch=fetch,now=NOW,wait=lambda _:None)
    assert len(calls)==CONTRACT_COINS+2 and len(set(calls))==CONTRACT_COINS+2  # one long cooldown per request is not retried
    assert len(result['errors'])==CONTRACT_COINS+2 and not result['saved']


def test_pons_pool_requires_network_contract_and_accepts_v4_ids():
    from crypto_market_history import pools
    from crypto_social_history import PONS_ADDRESS
    item={'attributes':{'address':'0x'+'a'*64,'pool_created_at':'2026-07-17T00:00:00Z','reserve_in_usd':'20'},
          'relationships':{'base_token':{'data':{'id':'robinhood_'+PONS_ADDRESS.upper()}}}}
    assert len(pools({'data':[item]},PONS_ADDRESS,'robinhood'))==1
    assert pools({'data':[item]})==[]
    item['relationships']['base_token']['data']['id']='ethereum_'+PONS_ADDRESS
    assert pools({'data':[item]},PONS_ADDRESS,'robinhood')==[]


def observations(t=None,volumes=True):
    """A CoinGecko market_chart payload. total_volumes is a rolling 24h total, not hourly volume."""
    t=t or int(NOW.timestamp()*1000)
    data={'prices':[[t-3600000,1.0],[t,2.0]]}
    if volumes:data['total_volumes']=[[t-3600000,1.2e9],[t,1.19e9]]
    return data

def test_a_rolling_24h_total_is_never_recorded_as_this_interval_volume():
    """CoinGecko reports volume over a trailing day, so the hour's own trading is unknown.

    Recording it would put a smoothed window into the column every reader treats as per-interval
    trading: summing 24 of them implied ~$29bn of daily Zcash volume against the ~$1.2bn the
    series itself reports.
    """
    for points in (normalize(observations(),'price_observation',NOW),
                   normalize_hourly(observations(),'price_observation',NOW)):
        assert points and all(p['volume'] is None for p in points)
        assert all(p['close']>0 for p in points)  # the price observations were always valid

def test_a_payload_missing_total_volumes_is_still_rejected():
    # The field is unused but its absence means the provider changed shape, which is worth failing on.
    with pytest.raises(ValueError):normalize(observations(volumes=False),'price_observation',NOW)

def test_an_ohlcv_source_must_still_carry_a_real_volume():
    good=candles();assert len(normalize(good,'ohlcv',NOW))==1
    for bad in (None,-1,float('nan')):
        d=candles();d['data']['attributes']['ohlcv_list'][0][5]=bad
        assert normalize(d,'ohlcv',NOW)==[],f'volume {bad!r} must not be archived'

def test_db_null_volume_round_trips_and_legacy_rows_repair(db):
    setup(db)
    src=dict(id='coingecko:zcash',coin='ZEC',provider='CoinGecko',url='https://example.test/z',metadata={'id':'zcash'})
    data=observations()
    fid=save(db,src,data,normalize_hourly(data,'price_observation',NOW),'https://example.test/z',NOW,hourly=True)
    with db,db.cursor() as cur:
        cur.execute('SELECT count(*) FROM crypto_market_hourly WHERE fetch_id=%s AND volume IS NULL',(fid,))
        assert cur.fetchone()[0]>0   # CHECK(volume>=0) must not reject a NULL
        # A row written before the correction still carries the rolling total.
        cur.execute("""INSERT INTO crypto_market_hourly(fetch_id,hour,sample_at,close,volume,complete,kind)
                       VALUES (%s,%s,%s,1450.0,1.2e9,true,'price_observation')""",(fid,NOW-timedelta(days=9),NOW))
        cur.execute("SELECT count(*) FROM crypto_market_hourly WHERE kind='price_observation' AND volume IS NOT NULL")
        assert cur.fetchone()[0]==1
    assert repair_observation_volume(db)['crypto_market_hourly']==1
    assert repair_observation_volume(db)['crypto_market_hourly']==0   # idempotent
    with db,db.cursor() as cur:
        cur.execute("SELECT count(*) FROM crypto_market_hourly WHERE kind='price_observation' AND volume IS NOT NULL")
        assert cur.fetchone()[0]==0
        cur.execute('SELECT count(*) FROM crypto_market_hourly WHERE close IS NULL')
        assert cur.fetchone()[0]==0   # prices are corrected, never discarded

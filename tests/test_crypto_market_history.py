from datetime import datetime,timezone,timedelta
import pytest
from test_crypto_social_pilot import db
from crypto_market_history import ADDRESS,normalize,pools,setup,save,refresh

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
    assert len(result['saved'])==1 and len(result['errors'])==1 and len(calls)==3
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
    assert not result['errors'] and len(result['saved'])==2
    assert len(calls)==6 and max(delays)==20
    assert all('twitter' not in url for url in calls)


def test_db_long_provider_cooldown_is_not_shortened(db):
    calls=[]
    class Response:
        status_code=429
        headers={'Retry-After':'999'}
    def fetch(url,**kwargs):calls.append(url);return Response()
    result=refresh(db,fetch=fetch,now=NOW,wait=lambda _:None)
    assert len(calls)==2 and len(set(calls))==2
    assert len(result['errors'])==2 and not result['saved']
